"""
Universal dtype inference for backend datasets.

Tradeoffs and limits:
- The inference engine intentionally depends only on pandas, numpy, and re so it can be tested and reused without
  backend framework imports.
- The code iterates over columns only. It does not iterate over rows, and all value-level checks use vectorised pandas
  or numpy operations.
- Sampling is deterministic and computed once per column. Full-column uniqueness is skipped for large columns and
  replaced with deterministic sample cardinality to protect latency.
- Normalisation is cached during each top-level inference call. The cache is cleared at function start to prevent stale
  values from crossing requests.
- Rejections are conservative. A rejected phase stops later phases, leaving the source column unchanged.
- Ambiguous day/month dates are blocked unless ISO-like formats dominate, because silent calendar swaps are costly.
- Percent strings are converted to fractional numeric values only when the percent pattern clearly dominates.
- Categorical conversion uses cardinality thresholds; identifier-like or high-cardinality text remains string-like.
- Integer casts use pandas nullable integer dtypes when nulls are present, preserving missing values.
"""

import pandas as pd
import numpy as np
import re


LARGE_COL_CUTOFF = 100000
SAMPLE_SIZE = 25000
RANDOM_STATE = 42

BOOL_ACCEPT_THRESHOLD = 0.95
NUMERIC_ACCEPT_THRESHOLD = 0.92
DATETIME_ACCEPT_THRESHOLD = 0.90
PERCENT_ACCEPT_THRESHOLD = 0.90
CATEGORY_MAX_RATIO = 0.50
CATEGORY_MAX_UNIQUE = 500
CATEGORY_SAMPLE_UNIQUE_MAX = 500
LOW_CONFIDENCE_MARGIN = 0.04
SKEW_TOP_SHARE_THRESHOLD = 0.95
PHASE_0B_NULL_RATE_THRESHOLD = 0.98

LOG_FIELDS = [
    "column",
    "original_dtype",
    "new_dtype",
    "cast_type",
    "accepted",
    "null_rate_before",
    "null_rate_after",
    "null_delta",
    "low_confidence",
    "skew_warning",
    "memory_before_bytes",
    "memory_after_bytes",
    "memory_delta_bytes",
    "inference_mode",
    "coverage_note",
    "value_sample",
    "datetime_format",
    "ambiguity_detected",
    "competing_formats",
    "percent_scaled",
    "tradeoff_note",
    "phase_0b_triggered",
    "rejection_reason",
    "error_message",
]

_INT_TYPES = [
    ("Int8", np.iinfo(np.int8).min, np.iinfo(np.int8).max),
    ("Int16", np.iinfo(np.int16).min, np.iinfo(np.int16).max),
    ("Int32", np.iinfo(np.int32).min, np.iinfo(np.int32).max),
    ("Int64", np.iinfo(np.int64).min, np.iinfo(np.int64).max),
]

_UINT_TYPES = [
    ("UInt8", np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
    ("UInt16", np.iinfo(np.uint16).min, np.iinfo(np.uint16).max),
    ("UInt32", np.iinfo(np.uint32).min, np.iinfo(np.uint32).max),
    ("UInt64", np.iinfo(np.uint64).min, np.iinfo(np.uint64).max),
]

_norm_cache = {}


def _serializable_sample(series):
    return series.dropna().head(5).tolist()


def _empty_log(column, col_raw, cast_type, inference_mode, value_sample):
    memory_before = int(col_raw.memory_usage(deep=True))
    null_before = float(col_raw.isna().mean()) if len(col_raw) else 0.0
    return {
        "column": str(column),
        "original_dtype": str(col_raw.dtype),
        "new_dtype": str(col_raw.dtype),
        "cast_type": str(cast_type),
        "accepted": False,
        "null_rate_before": null_before,
        "null_rate_after": null_before,
        "null_delta": 0.0,
        "low_confidence": False,
        "skew_warning": False,
        "memory_before_bytes": memory_before,
        "memory_after_bytes": memory_before,
        "memory_delta_bytes": 0,
        "inference_mode": str(inference_mode),
        "coverage_note": "",
        "value_sample": value_sample,
        "datetime_format": None,
        "ambiguity_detected": False,
        "competing_formats": None,
        "percent_scaled": False,
        "tradeoff_note": None,
        "phase_0b_triggered": False,
        "rejection_reason": None,
        "error_message": None,
    }


def _finalize_log(log, col_raw, col_out):
    before = int(col_raw.memory_usage(deep=True))
    after = int(col_out.memory_usage(deep=True))
    null_before = float(col_raw.isna().mean()) if len(col_raw) else 0.0
    null_after = float(col_out.isna().mean()) if len(col_out) else 0.0
    log["new_dtype"] = str(col_out.dtype)
    log["null_rate_before"] = null_before
    log["null_rate_after"] = null_after
    log["null_delta"] = float(null_after - null_before)
    log["memory_before_bytes"] = before
    log["memory_after_bytes"] = after
    log["memory_delta_bytes"] = int(before - after)
    return _coerce_log_types(log)


def _coerce_log_types(log):
    normalized = {
        "column": str(log.get("column", "")),
        "original_dtype": str(log.get("original_dtype", "")),
        "new_dtype": str(log.get("new_dtype", "")),
        "cast_type": str(log.get("cast_type", "")),
        "accepted": bool(log.get("accepted", False)),
        "null_rate_before": float(log.get("null_rate_before", 0.0)),
        "null_rate_after": float(log.get("null_rate_after", 0.0)),
        "null_delta": float(log.get("null_delta", 0.0)),
        "low_confidence": bool(log.get("low_confidence", False)),
        "skew_warning": bool(log.get("skew_warning", False)),
        "memory_before_bytes": int(log.get("memory_before_bytes", 0)),
        "memory_after_bytes": int(log.get("memory_after_bytes", 0)),
        "memory_delta_bytes": int(log.get("memory_delta_bytes", 0)),
        "inference_mode": str(log.get("inference_mode", "")),
        "coverage_note": str(log.get("coverage_note", "")),
        "value_sample": list(log.get("value_sample", [])),
        "datetime_format": log.get("datetime_format") if log.get("datetime_format") is None else str(log.get("datetime_format")),
        "ambiguity_detected": bool(log.get("ambiguity_detected", False)),
        "competing_formats": None if log.get("competing_formats") is None else list(log.get("competing_formats")),
        "percent_scaled": bool(log.get("percent_scaled", False)),
        "tradeoff_note": log.get("tradeoff_note") if log.get("tradeoff_note") is None else str(log.get("tradeoff_note")),
        "phase_0b_triggered": bool(log.get("phase_0b_triggered", False)),
        "rejection_reason": log.get("rejection_reason") if log.get("rejection_reason") is None else str(log.get("rejection_reason")),
        "error_message": log.get("error_message") if log.get("error_message") is None else str(log.get("error_message")),
    }
    return {field: normalized[field] for field in LOG_FIELDS}


def _normalise(col_name, series):
    key = (str(col_name), id(series))
    if key not in _norm_cache:
        text = series.astype(str).str.strip()
        _norm_cache[key] = text.mask(series.isna(), np.nan)
    return _norm_cache[key]


def _sample_once(series):
    non_null = series.dropna()
    if len(non_null) > SAMPLE_SIZE:
        return non_null.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    return non_null


def _safe_nunique(series, sample, row_count):
    if row_count > LARGE_COL_CUTOFF:
        return int(sample.nunique(dropna=True))
    return int(series.nunique(dropna=True))


def _top_share(sample):
    if len(sample) == 0:
        return 0.0
    counts = sample.value_counts(dropna=True, sort=True)
    if len(counts) == 0:
        return 0.0
    return float(counts.iloc[0] / len(sample))


def _bool_candidate(norm):
    lowered = norm.astype(str).str.lower().str.strip()
    true_mask = lowered.isin(["true", "t", "yes", "y", "1"])
    false_mask = lowered.isin(["false", "f", "no", "n", "0"])
    parsed = pd.Series(pd.NA, index=norm.index, dtype="boolean")
    parsed = parsed.mask(true_mask, True)
    parsed = parsed.mask(false_mask, False)
    valid = true_mask | false_mask
    return parsed, valid


def _numeric_candidate(norm):
    cleaned = norm.astype(str).str.strip()
    percent_mask = cleaned.str.contains(r"%\s*$", regex=True, na=False)
    no_percent = cleaned.str.replace(r"%\s*$", "", regex=True)
    no_currency = no_percent.str.replace(r"^[\s$€£₹¥]+", "", regex=True)
    no_commas = no_currency.str.replace(",", "", regex=False)
    paren_neg = no_commas.str.match(r"^\([+-]?\d+(\.\d+)?\)$", na=False)
    signed = no_commas.str.replace(r"^\(([+-]?\d+(\.\d+)?)\)$", r"-\1", regex=True)
    valid_pattern = signed.str.match(r"^[+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]?\d+)?$", na=False)
    parsed = pd.to_numeric(signed.where(valid_pattern), errors="coerce")
    return parsed, valid_pattern, percent_mask, paren_neg


def _smallest_int_dtype(values, has_nulls):
    finite = values.dropna()
    if len(finite) == 0:
        return "Int8"
    min_v = float(finite.min())
    max_v = float(finite.max())
    candidates = _UINT_TYPES if min_v >= 0 else _INT_TYPES
    for dtype_name, low, high in candidates:
        if min_v >= low and max_v <= high:
            return dtype_name
    if min_v >= 0:
        return "UInt64"
    return "Int64" if has_nulls else "int64"


def _float_dtype(values):
    finite = values.dropna()
    if len(finite) == 0:
        return "float32"
    as64 = finite.astype("float64")
    as32 = as64.astype("float32").astype("float64")
    close = np.isclose(as64.to_numpy(), as32.to_numpy(), rtol=1e-06, atol=1e-08, equal_nan=True)
    return "float32" if bool(np.all(close)) else "float64"


def _datetime_patterns(norm):
    text = norm.astype(str).str.strip()
    iso = text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?$", na=False)
    slash_mdy_dmy = text.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", na=False)
    dash_dmy = text.str.match(r"^\d{1,2}-\d{1,2}-\d{2,4}$", na=False)
    month_name = text.str.match(r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}$|^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}$", na=False)
    first_num = pd.to_numeric(text.str.extract(r"^(\d{1,2})", expand=False), errors="coerce")
    second_num = pd.to_numeric(text.str.extract(r"^\d{1,2}[/-](\d{1,2})", expand=False), errors="coerce")
    ambiguous = ((slash_mdy_dmy | dash_dmy) & first_num.between(1, 12) & second_num.between(1, 12))
    return iso, slash_mdy_dmy, dash_dmy, month_name, ambiguous


def _cast_datetime(norm, iso, slash_mdy_dmy, dash_dmy, month_name):
    parsed_iso = pd.to_datetime(norm.where(iso), errors="coerce", format="%Y-%m-%d")
    parsed_iso_time = pd.to_datetime(norm.where(iso & parsed_iso.isna()), errors="coerce")
    parsed_slash = pd.to_datetime(norm.where(slash_mdy_dmy), errors="coerce", dayfirst=False)
    parsed_dash = pd.to_datetime(norm.where(dash_dmy), errors="coerce", dayfirst=True)
    parsed_month = pd.to_datetime(norm.where(month_name), errors="coerce")
    return parsed_iso.combine_first(parsed_iso_time).combine_first(parsed_slash).combine_first(parsed_dash).combine_first(parsed_month)


def infer_universal_dtypes(df):
    _norm_cache.clear()
    df_out = df.copy()
    cast_log = []
    row_count_before = len(df)

    for column in df.columns:
        col_raw = df[column]
        value_sample = _serializable_sample(col_raw)
        log = _empty_log(column, col_raw, "unchanged", "unstarted", value_sample)

        try:
            sample_raw = _sample_once(col_raw)
            sample_norm = _normalise(column, sample_raw)
            norm = _normalise(column, col_raw)
            non_null_count = int(col_raw.notna().sum())
            null_rate = float(col_raw.isna().mean()) if len(col_raw) else 0.0
            top_share = _top_share(sample_raw)
            skew_warning = bool(top_share >= SKEW_TOP_SHARE_THRESHOLD and non_null_count > 0)

            if non_null_count == 0:
                log["cast_type"] = "all_null"
                log["accepted"] = True
                log["inference_mode"] = "phase_0a_all_null"
                log["coverage_note"] = "all values are null"
                log["tradeoff_note"] = "all-null columns are preserved because no target dtype can be inferred deterministically"
                log = _finalize_log(log, col_raw, df_out[column])
                cast_log.append(log)
                continue

            if null_rate >= PHASE_0B_NULL_RATE_THRESHOLD or skew_warning:
                log["cast_type"] = "rejected"
                log["accepted"] = False
                log["inference_mode"] = "phase_0b_quality_gate"
                log["coverage_note"] = "quality gate triggered before semantic inference"
                log["skew_warning"] = skew_warning
                log["phase_0b_triggered"] = True
                log["rejection_reason"] = "extreme_null_rate_or_sample_skew"
                log["tradeoff_note"] = "high-null or highly skewed columns are left unchanged to avoid overfitting sparse evidence"
                log = _finalize_log(log, col_raw, df_out[column])
                cast_log.append(log)
                continue

            if pd.api.types.is_bool_dtype(col_raw.dtype):
                df_out[column] = col_raw.astype("boolean")
                log["cast_type"] = "boolean"
                log["accepted"] = True
                log["inference_mode"] = "phase_1_existing_boolean"
                log["coverage_note"] = "existing boolean dtype accepted"
                log["skew_warning"] = skew_warning
                log = _finalize_log(log, col_raw, df_out[column])
                cast_log.append(log)
                continue

            bool_parsed, bool_valid = _bool_candidate(norm)
            bool_coverage = float(bool_valid.sum() / non_null_count)
            if bool_coverage >= BOOL_ACCEPT_THRESHOLD:
                df_out[column] = bool_parsed
                log["cast_type"] = "boolean"
                log["accepted"] = True
                log["low_confidence"] = bool(bool_coverage < BOOL_ACCEPT_THRESHOLD + LOW_CONFIDENCE_MARGIN)
                log["skew_warning"] = skew_warning
                log["inference_mode"] = "phase_1_boolean"
                log["coverage_note"] = "boolean coverage %.6f" % bool_coverage
                log = _finalize_log(log, col_raw, df_out[column])
                cast_log.append(log)
                continue

            parsed_num, valid_num, percent_mask, paren_neg = _numeric_candidate(norm)
            num_coverage = float(valid_num.sum() / non_null_count)
            percent_coverage = float((valid_num & percent_mask).sum() / non_null_count)
            if num_coverage >= NUMERIC_ACCEPT_THRESHOLD:
                out_num = parsed_num / 100.0 if percent_coverage >= PERCENT_ACCEPT_THRESHOLD else parsed_num
                finite = out_num.dropna()
                as_float = finite.to_numpy(dtype="float64")
                is_integer_like = bool(len(finite) > 0 and np.all(np.equal(np.mod(as_float, 1.0), 0.0)))
                if is_integer_like:
                    dtype_name = _smallest_int_dtype(out_num, bool(out_num.isna().any()))
                    df_out[column] = out_num.round().astype(dtype_name)
                    cast_name = "integer"
                else:
                    dtype_name = _float_dtype(out_num)
                    df_out[column] = out_num.astype(dtype_name)
                    cast_name = "float"
                log["cast_type"] = cast_name
                log["accepted"] = True
                log["low_confidence"] = bool(num_coverage < NUMERIC_ACCEPT_THRESHOLD + LOW_CONFIDENCE_MARGIN)
                log["skew_warning"] = skew_warning
                log["inference_mode"] = "phase_2_numeric"
                log["coverage_note"] = "numeric coverage %.6f" % num_coverage
                log["percent_scaled"] = bool(percent_coverage >= PERCENT_ACCEPT_THRESHOLD)
                log["tradeoff_note"] = "percent strings are converted to fractional numeric values" if log["percent_scaled"] else None
                log = _finalize_log(log, col_raw, df_out[column])
                cast_log.append(log)
                continue

            iso, slash_mdy_dmy, dash_dmy, month_name, ambiguous = _datetime_patterns(norm)
            dt_candidates = iso | slash_mdy_dmy | dash_dmy | month_name
            dt_coverage = float(dt_candidates.sum() / non_null_count)
            ambiguity_detected = bool(ambiguous.any())
            if dt_coverage >= DATETIME_ACCEPT_THRESHOLD:
                if ambiguity_detected and float(iso.sum() / non_null_count) < DATETIME_ACCEPT_THRESHOLD:
                    log["cast_type"] = "rejected"
                    log["accepted"] = False
                    log["low_confidence"] = True
                    log["skew_warning"] = skew_warning
                    log["inference_mode"] = "phase_3_datetime_rejected"
                    log["coverage_note"] = "datetime coverage %.6f blocked by ambiguous day/month ordering" % dt_coverage
                    log["ambiguity_detected"] = True
                    log["competing_formats"] = ["MM/DD/YYYY", "DD/MM/YYYY"]
                    log["rejection_reason"] = "ambiguous_datetime_format"
                    log["tradeoff_note"] = "ambiguous dates are preserved to avoid irreversible calendar swaps"
                    log = _finalize_log(log, col_raw, df_out[column])
                    cast_log.append(log)
                    continue

                parsed_dt = _cast_datetime(norm, iso, slash_mdy_dmy, dash_dmy, month_name)
                parsed_coverage = float(parsed_dt.notna().sum() / non_null_count)
                if parsed_coverage >= DATETIME_ACCEPT_THRESHOLD:
                    df_out[column] = parsed_dt
                    fmt = "%Y-%m-%d" if bool(iso.sum() >= slash_mdy_dmy.sum() and iso.sum() >= dash_dmy.sum() and iso.sum() >= month_name.sum()) else None
                    log["cast_type"] = "datetime"
                    log["accepted"] = True
                    log["low_confidence"] = bool(parsed_coverage < DATETIME_ACCEPT_THRESHOLD + LOW_CONFIDENCE_MARGIN)
                    log["skew_warning"] = skew_warning
                    log["inference_mode"] = "phase_3_datetime"
                    log["coverage_note"] = "datetime coverage %.6f" % parsed_coverage
                    log["datetime_format"] = fmt
                    log["ambiguity_detected"] = ambiguity_detected
                    log["competing_formats"] = ["ISO", "slash", "dash", "month_name"]
                    log = _finalize_log(log, col_raw, df_out[column])
                    cast_log.append(log)
                    continue

                log["cast_type"] = "rejected"
                log["accepted"] = False
                log["low_confidence"] = True
                log["skew_warning"] = skew_warning
                log["inference_mode"] = "phase_3_datetime_rejected"
                log["coverage_note"] = "datetime parse coverage %.6f" % parsed_coverage
                log["ambiguity_detected"] = ambiguity_detected
                log["competing_formats"] = ["ISO", "slash", "dash", "month_name"]
                log["rejection_reason"] = "datetime_parse_coverage_below_threshold"
                log = _finalize_log(log, col_raw, df_out[column])
                cast_log.append(log)
                continue

            unique_count = _safe_nunique(col_raw, sample_raw, len(col_raw))
            denom = max(non_null_count, 1)
            unique_ratio = float(unique_count / denom)
            if unique_count <= CATEGORY_MAX_UNIQUE and unique_ratio <= CATEGORY_MAX_RATIO:
                df_out[column] = col_raw.astype("category")
                log["cast_type"] = "category"
                log["accepted"] = True
                log["low_confidence"] = bool(unique_count > CATEGORY_SAMPLE_UNIQUE_MAX or unique_ratio > CATEGORY_MAX_RATIO - LOW_CONFIDENCE_MARGIN)
                log["skew_warning"] = skew_warning
                log["inference_mode"] = "phase_4_category"
                log["coverage_note"] = "unique_count %d unique_ratio %.6f" % (unique_count, unique_ratio)
                log["tradeoff_note"] = "large columns use sampled cardinality to avoid full-column nunique"
                log = _finalize_log(log, col_raw, df_out[column])
                cast_log.append(log)
                continue

            if pd.api.types.is_object_dtype(col_raw.dtype) or pd.api.types.is_string_dtype(col_raw.dtype):
                df_out[column] = col_raw.astype("string")
                log["cast_type"] = "string"
                log["accepted"] = True
                log["skew_warning"] = skew_warning
                log["inference_mode"] = "phase_5_string_fallback"
                log["coverage_note"] = "semantic casts did not meet thresholds"
                log["tradeoff_note"] = "fallback preserves values as pandas string dtype"
                log = _finalize_log(log, col_raw, df_out[column])
                cast_log.append(log)
                continue

            log["cast_type"] = "unchanged"
            log["accepted"] = True
            log["skew_warning"] = skew_warning
            log["inference_mode"] = "phase_5_dtype_preserved"
            log["coverage_note"] = "non-object dtype preserved"
            log = _finalize_log(log, col_raw, df_out[column])
            cast_log.append(log)
            continue

        except Exception as exc:
            log["cast_type"] = "error"
            log["accepted"] = False
            log["inference_mode"] = "column_exception"
            log["coverage_note"] = "column failed during inference"
            log["rejection_reason"] = "exception"
            log["error_message"] = str(exc)
            log = _finalize_log(log, col_raw, df_out[column])
            cast_log.append(log)
            continue

    assert len(df_out) == row_count_before
    assert len(cast_log) == len(df.columns)
    return df_out, cast_log


def dtype_summary_report(cast_log):
    log_df = pd.DataFrame(cast_log)
    for field in LOG_FIELDS:
        if field not in log_df.columns:
            log_df[field] = np.nan

    if len(log_df) == 0:
        return pd.DataFrame({
            "cast_type": pd.Series(dtype="string"),
            "columns_affected": pd.Series(dtype="int64"),
            "total_memory_saved_kb": pd.Series(dtype="float64"),
            "avg_null_delta": pd.Series(dtype="float64"),
            "acceptance_rate": pd.Series(dtype="float64"),
            "low_confidence_count": pd.Series(dtype="int64"),
            "skew_warning_count": pd.Series(dtype="int64"),
            "ambiguous_dates_blocked": pd.Series(dtype="int64"),
            "inference_modes": pd.Series(dtype="object"),
        })

    accepted_num = log_df["accepted"].astype(bool).astype(int)
    low_num = log_df["low_confidence"].astype(bool).astype(int)
    skew_num = log_df["skew_warning"].astype(bool).astype(int)
    ambiguous_blocked = (log_df["ambiguity_detected"].astype(bool) & (~log_df["accepted"].astype(bool))).astype(int)

    work = log_df.copy()
    work["_accepted_num"] = accepted_num
    work["_low_num"] = low_num
    work["_skew_num"] = skew_num
    work["_ambiguous_blocked"] = ambiguous_blocked

    modes = work.groupby("cast_type", dropna=False, sort=True)["inference_mode"].unique().reset_index(name="inference_modes")
    modes["inference_modes"] = modes["inference_modes"].map(lambda values: sorted(pd.Series(values).dropna().astype(str).unique().tolist()))

    report = work.groupby("cast_type", dropna=False, sort=True).agg(
        columns_affected=("column", "count"),
        total_memory_saved_kb=("memory_delta_bytes", "sum"),
        avg_null_delta=("null_delta", "mean"),
        acceptance_rate=("_accepted_num", "mean"),
        low_confidence_count=("_low_num", "sum"),
        skew_warning_count=("_skew_num", "sum"),
        ambiguous_dates_blocked=("_ambiguous_blocked", "sum"),
    ).reset_index()
    report["total_memory_saved_kb"] = report["total_memory_saved_kb"].astype("float64") / 1024.0
    report = report.merge(modes, on="cast_type", how="left")
    report["columns_affected"] = report["columns_affected"].astype("int64")
    report["low_confidence_count"] = report["low_confidence_count"].astype("int64")
    report["skew_warning_count"] = report["skew_warning_count"].astype("int64")
    report["ambiguous_dates_blocked"] = report["ambiguous_dates_blocked"].astype("int64")
    return report


def dtype_review_flags(cast_log):
    log_df = pd.DataFrame(cast_log)
    for field in LOG_FIELDS:
        if field not in log_df.columns:
            log_df[field] = np.nan
    if len(log_df) == 0:
        return pd.DataFrame(columns=[
            "column",
            "cast_type",
            "accepted",
            "low_confidence",
            "skew_warning",
            "ambiguity_detected",
            "rejection_reason",
            "coverage_note",
            "error_message",
        ])
    mask = (
        (~log_df["accepted"].astype(bool))
        | log_df["low_confidence"].astype(bool)
        | log_df["skew_warning"].astype(bool)
        | log_df["ambiguity_detected"].astype(bool)
    )
    return log_df.loc[mask, [
        "column",
        "cast_type",
        "accepted",
        "low_confidence",
        "skew_warning",
        "ambiguity_detected",
        "rejection_reason",
        "coverage_note",
        "error_message",
    ]]
