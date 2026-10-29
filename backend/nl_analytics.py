"""Universal, dataset-aware NL analytics engine.

Replaces the previous behaviour where every assistant question was routed to a
regression/classification training run (backend ``execute_ml_assistant`` always
trained ridge/random-forest on an auto-picked revenue column, and the frontend
``handleAskAssistant`` only rendered local template text without touching data).

Pipeline implemented here (pandas/numpy/re only, no backend imports so it is
unit-testable in isolation)::

    question + frame -> profile -> intent -> schema mapping -> plan -> execute -> validate -> response

Security: no ``exec``/``eval`` of model output, no free-form SQL.  Only
validated column names from the live frame are used, and only a closed set of
pandas operations (filter / groupby-agg / sort / describe / correlation /
trend).  All numbers are computed from the real frame; the LLM is never asked
to produce numbers (this module is fully deterministic and LLM-free).
"""

from __future__ import annotations

import re
import warnings
import hashlib
from difflib import SequenceMatcher
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from purchase_predict import (
    build_purchase_histories,
    customers_at_risk,
    customers_for_product,
    discover_purchase_schema,
    predict_next_for_customer,
    predict_timing_for_customer,
    validate_methods_temporal,
)

__all__ = [
    "ANALYTICAL_INTENTS",
    "ML_INTENTS",
    "PURCHASE_INTENTS",
    "profile_dataset",
    "dataset_signature",
    "classify_intent",
    "answer_question",
    "resolve_metric_column",
    "resolve_dimension_column",
    "resolve_filter",
    "extract_entities",
]

ANALYTICAL_INTENTS = frozenset({
    "descriptive", "filter", "ranking", "comparison", "trend", "statistics",
})
PURCHASE_INTENTS = frozenset({
    "next_purchase", "recommend", "buyers", "churn", "confidence", "explain",
})
ML_INTENTS = frozenset({
    "forecast", "regression", "classification", "anomaly", "segmentation",
    *PURCHASE_INTENTS,
})

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _norm_nospace(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def dataset_signature(frame: pd.DataFrame, dataset_id: str = "") -> str:
    """Content signature used to invalidate cached profiles on replacement."""
    cols = "|".join(str(c) for c in frame.columns)
    try:
        row_hashes = pd.util.hash_pandas_object(frame, index=True, categorize=True).to_numpy(dtype=np.uint64)
        content_hash = hashlib.sha256(row_hashes.tobytes()).hexdigest()[:20]
    except (TypeError, ValueError):
        sample = frame.iloc[np.unique(np.linspace(0, max(len(frame) - 1, 0), min(len(frame), 16), dtype=int))]
        sample_hashes = pd.util.hash_pandas_object(sample.astype(str), index=True, categorize=True).to_numpy(dtype=np.uint64)
        content_hash = hashlib.sha256(sample_hashes.tobytes()).hexdigest()[:20]
    return f"{dataset_id}::{cols}::{len(frame)}::{content_hash}"


# ---------------------------------------------------------------------------
# Synonym dictionaries (semantic hints only -- every candidate must exist in
# the live schema and pass a type/role check before use).
# ---------------------------------------------------------------------------

METRIC_SYNONYMS: dict[str, list[str]] = {
    # canonical metric kind -> phrase hints found in user questions
    "revenue": ["revenue", "revenues", "sales", "net sales", "total sales", "sale value",
                "turnover", "gmv", "income", "takings"],
    "units": ["units shipped", "units sold", "units", "quantity shipped", "qty shipped",
              "quantity sold", "qty sold", "quantity", "volume shipped", "shipments units"],
    "orders": ["order counts", "order count", "orders", "number of orders", "num orders",
               "order volume", "order numbers", "transactions", "order quantity rows"],
    "price": ["price", "selling price", "average price", "avg price", "asp", "unit price", "rate"],
    "profit": ["profit", "gross profit", "net profit", "margin"],
    "cost": ["cost", "cogs", "expense", "freight", "spend"],
}

# Column-name tokens preferred for each metric kind (validated, scored).
METRIC_COLUMN_TOKENS: dict[str, list[str]] = {
    "revenue": ["net_sales", "sale_value", "total_value_sale", "gmv", "turnover",
                "revenue", "sales", "total_value", "amount"],
    "units": ["units_shipped", "units_sold", "units", "quantity_shipped", "qty"],
    "orders": ["order_count", "order_counts", "orders", "n_orders"],
    "price": ["avg_selling_price", "average_price", "asp", "unit_price", "price"],
    "profit": ["gross_profit", "net_profit", "profit"],
    "cost": ["cogs", "freight_cost", "cost", "expense"],
}

DIMENSION_SYNONYMS: dict[str, list[str]] = {
    "product_line": ["product lines", "product line", "product", "products", "product family",
                     "product category", "item line", "sku group"],
    "customer_segment": ["customer segment", "customer segments", "segment", "customer type",
                         "customer group", "client segment", "market segment"],
    "region": ["region", "plant region", "plant", "territory", "area", "location"],
}

FILTER_VALUE_HINTS = ["for the", "for", "where", "with", "in the", "of the", "segment", "named", "="]

SEGMENT_TOKEN_SCORES: list[tuple[str, int]] = [
    ("customer_segment", 100), ("segment", 80), ("customer_type", 80),
    ("customer_group", 75), ("client_segment", 75), ("market_segment", 70),
]
PRODUCT_TOKEN_SCORES: list[tuple[str, int]] = [
    ("product_line", 100), ("product_category", 85), ("product_family", 80),
    ("product", 50), ("category", 40), ("line", 20),
]

# ---------------------------------------------------------------------------
# Intent classification (structured signals, not single-keyword matching)
# ---------------------------------------------------------------------------

_PREDICT_VERBS = re.compile(r"\b(predict|prediction|estimate the future|will .* be|what will|forecast|project)\b", re.I)
_FORECAST_VERBS = re.compile(r"\b(forecast|project)\b", re.I)
_FORECAST_HORIZON = re.compile(
    r"\b(next|coming|following)\s+(?:(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|\w+)\s+)?"
    r"(day|week|month|quarter|year|period)s?\b|\b(next|coming)\s+(day|week|month|quarter|year)\b", re.I)
_CLASSIFY_HINTS = re.compile(r"\b(classify|classification|which class|what category will|churn|fraud|yes\/no|approve|default)\b", re.I)
_ANOMALY_HINTS = re.compile(r"\b(anomal\w*|outlier|unusual|abnormal|fraud|weird)\b", re.I)
_SEGMENT_HINTS = re.compile(r"\b(cluster|clustering|k-?means|group similar|find natural groups|segment customers|discover groups|segmentation|segment (rows|data|records|customers?|into))\b", re.I)
_RANK_HINTS = re.compile(r"\b(highest|lowest|top\s*\d*|bottom\s*\d*|best|worst|largest|smallest|rank|leading|winner)\b", re.I)
_TREND_HINTS = re.compile(r"\b(trend|over time|time series|month over month|year over year|mom|yoy|growth|change over|by month|by year|by quarter|by week|by day|timeline)\b", re.I)
_COMPARE_HINTS = re.compile(r"\b(compar|versus|\bvs\b|against|difference between|relative to|share of)\b", re.I)
_FILTER_HINTS = re.compile(r"\b(show|list|find|filter|display|get|give me|retrieve|records where|rows where)\b", re.I)
_STAT_HINTS = re.compile(r"\b(average|mean|median|std|standard deviation|variance|correlation|correlat|distribution|histogram|percentile|quartile|skew|min\b|max\b|sum\b|total\b|count\b)\b", re.I)
_FUTURE_HINTS = re.compile(r"\b(next month|next quarter|next year|next week|future|upcoming|tomorrow|going forward)\b", re.I)
# Purchase / recommendation signals (checked before generic predict/forecast branches).
_NEXT_BUY_HINTS = re.compile(
    r"\b(next purchases?|buy next|purchase again|buy again|order again|reorder|repeat purchases?|"
    r"next order|next item|next product|what .* (buy|purchase|order) next|"
    r"will .* (buy|purchase|order)|going to (buy|purchase|order))\b", re.I)
_RECOMMEND_HINTS = re.compile(
    r"\b(recommend|suggest|should .* (buy|purchase|order)|most likely to (buy|purchase|order)|"
    r"likely to (buy|purchase|order)|best .* for (this |the |a )?customer|top \d* ?products? .* (buy|purchase))\b", re.I)
_CHURN_HINTS = re.compile(
    r"\b(churn|at risk|at-risk|not purchas\w*|stop buying|stopped buying|inactive customers?|"
    r"lapse|lapsing|retain|retention|defect|lost customers?|win back)\b", re.I)
_BUYERS_HINTS = re.compile(
    r"\bwhich customers.*(buy|purchase|order)|customers .* likely .*(buy|purchase|order)|"
    r"who (buys|buy|will buy|bought|purchased)|find customers|customers for\b", re.I)
_TIMING_HINTS = re.compile(
    r"\b(when .* (buy|purchase|order|again)|time.*next (purchase|order)|how soon|"
    r"purchase again|buy again|next purchase date|expected.*(purchase|order))\b", re.I)
_CONFIDENCE_HINTS = re.compile(
    r"\b(confiden\w*|probability|how sure|how certain|likelihood|uncertain|sure about|"
    r"trust this|reliable|accuracy of this)\b", re.I)
_EXPLAIN_HINTS = re.compile(
    r"\b(why .* (recommend|predict|suggest|chose|choose)|explain .* (recommend|predict|suggest|prediction)|"
    r"reason for .* (recommend|predict)|how did you|basis for)\b", re.I)
_THIS_CUSTOMER_HINTS = re.compile(r"\b(this customer|that customer|current customer|this client|this user)\b", re.I)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _supervised_hint(question: str, profile: dict[str, Any] | None) -> dict[str, str] | None:
    """Detect an explicitly framed supervised task: a real column named as the
    object of predict/forecast/classify, or as the subject of a confidence /
    accuracy / model / explanation ask. Returns {column} or None."""
    q = question or ""
    raw_columns = profile.get("columns", []) if profile else []
    columns = [str(c.get("name", c) if isinstance(c, dict) else c) for c in raw_columns]
    if not columns:
        return None
    by_norm = {_norm(c): c for c in columns}
    # Verb object: "predict X from ...", "classify Y ...", "forecast Z ...".
    m = re.search(
        r"\b(predict|forecast|classify|estimate|project|assign|label|identify|determine)\s+(?:(?:the|a|an)\s+)?([a-z_][a-z0-9_ ]{0,40}?)"
        r"\s+(from|using|based on|with|given|where|for\b|$)", q, re.I)
    if m and _norm(m.group(2)) in by_norm:
        return {"column": by_norm[_norm(m.group(2))]}
    # Confidence/model/explanation subject: "confidence of the X model",
    # "accuracy of X", "explain the X prediction".
    m2 = re.search(
        r"\b(confiden\w*|accuracy|performance|reliability|model|explain|explanation|"
        r"prediction|evaluation|metrics)\w*\b.{0,12}?\b(of|for)\b\s+(?:the\s+)?([a-z_][a-z0-9_ ]{0,40}?)"
        r"(\s+(model|prediction|forecast))?\b", q, re.I)
    if m2 and _norm(m2.group(3)) in by_norm:
        return {"column": by_norm[_norm(m2.group(3))]}
    return None


def _row_classification_target(question: str, profile: dict[str, Any] | None) -> str | None:
    """Resolve a categorical target in an open-ended row-classification question.

    The target comes from the active schema. This catches questions framed as
    "which <field> is likely for a record" without binding the parser to a
    particular industry or column name.
    """
    q = question or ""
    if not (re.search(r"\b(?:which|what)\b", q, re.I)
            and re.search(r"\b(?:likely|probable|prediction|predicted)\b", q, re.I)
            and re.search(r"\b(?:record|row|case|observation|entry)\b", q, re.I)):
        return None
    targets = _column_mentions(q, profile or {}, roles={"categorical", "boolean"})
    return targets[0] if len(targets) == 1 else None


def classify_intent(question: str, profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return structured intent: {intent, confidence, signals, needs_ml, operations}."""
    q = question or ""
    ql = q.lower()
    signals: dict[str, bool] = {
        "future": bool(_FUTURE_HINTS.search(q) or _FORECAST_HORIZON.search(q)),
        "predict_verb": bool(_PREDICT_VERBS.search(q)),
        "classify": bool(_CLASSIFY_HINTS.search(q)),
        "anomaly": bool(_ANOMALY_HINTS.search(q)),
        "segmentation": bool(_SEGMENT_HINTS.search(q)),
        "ranking": bool(_RANK_HINTS.search(q)),
        "trend": bool(_TREND_HINTS.search(q)),
        "compare": bool(_COMPARE_HINTS.search(q)),
        "filter": bool(_FILTER_HINTS.search(q)),
        "stat": bool(_STAT_HINTS.search(q)),
        "next_buy": bool(_NEXT_BUY_HINTS.search(q)),
        "recommend": bool(_RECOMMEND_HINTS.search(q)),
        "churn": bool(_CHURN_HINTS.search(q)),
        "buyers": bool(_BUYERS_HINTS.search(q)),
        "timing": bool(_TIMING_HINTS.search(q)),
        "confidence_q": bool(_CONFIDENCE_HINTS.search(q)),
        "explain_q": bool(_EXPLAIN_HINTS.search(q)),
    }
    has_date = bool((profile or {}).get("datetime_columns") or (profile or {}).get("date_columns"))
    purchase_vocab = signals["next_buy"] or signals["recommend"] or signals["churn"] or signals["buyers"]
    has_classify_verb = bool(re.search(r"\bclassify\b|\bclassification\b", ql))
    row_classification_target = _row_classification_target(q, profile)
    explicit_forecast = bool(_FORECAST_VERBS.search(q) or _FORECAST_HORIZON.search(q))
    supervised_framing = bool(re.search(r"\bfrom\b|\bwhere\b|\bwith\b|\bgiven\b|\binto\b|\bas\b", ql))
    # Bare "churn / at-risk" wording with no supervised framing belongs to purchase
    # churn analysis, not classifier training.
    churn_word_only = (signals["churn"] and not has_classify_verb
                       and not re.search(r"\bfrom\b", ql) and not signals["predict_verb"])

    # Order matters: genuine ML only when prediction of unknown values is asked.
    if signals["anomaly"]:
        intent, conf, ops = "anomaly", 80, ["anomaly_scan"]
    elif signals["segmentation"]:
        intent, conf, ops = "segmentation", 75, ["cluster"]
    elif row_classification_target and not churn_word_only:
        intent, conf, ops = "classification", 80, ["classify"]
    elif signals["classify"] and not churn_word_only and (
            has_classify_verb or signals["predict_verb"] or supervised_framing):
        # "Classify X ..." / "Classify X from A, B" / "... where ..." requests a
        # classifier build. Pure churn wording defers to purchase churn analysis.
        intent, conf, ops = "classification", 82, ["classify"]
    elif signals["classify"] and not churn_word_only:
        # "which category ..." without prediction verbs is retrieval, not ML.
        intent, conf, ops = "filter", 70, ["filter"]
    elif signals["churn"]:
        intent, conf, ops = "churn", 84, ["purchase_schema", "histories", "churn_scan"]
    elif signals["buyers"]:
        intent, conf, ops = "buyers", 84, ["purchase_schema", "histories", "buyers_rank"]
    elif explicit_forecast and not signals["next_buy"]:
        # Explicit horizon/forecast language takes precedence over a target
        # mention; otherwise "Forecast temperature next month" is mistaken for
        # row-level regression merely because temperature is a numeric column.
        intent, conf, ops = "forecast", 85 if has_date else 60, ["forecast"]
    elif _supervised_hint(q, profile) and (
            signals["explain_q"] or signals["confidence_q"] or signals["predict_verb"]
            or has_classify_verb or bool(re.search(r"\b(?:assign|label|identify|determine)\b", ql))):
        # Multi-intent with an explicit supervised task ("predict X ... and explain",
        # "confidence of the X model"): the supervised execution comes first; its
        # summary already carries confidence, intervals, and feature importance.
        hint_col = _supervised_hint(q, profile)["column"]  # type: ignore[index]
        meta = next((c for c in (profile.get("columns", []) if profile else [])
                     if c["name"] == hint_col), {})
        # Problem type follows the target role (validated again at execution);
        # an explicit classify verb forces classification.
        if has_classify_verb or meta.get("role") in ("categorical", "boolean"):
            intent, conf, ops = "classification", 83, ["classify"]
        else:
            intent, conf, ops = "regression", 83, ["regress"]
    elif signals["explain_q"] and purchase_vocab:
        intent, conf, ops = "explain", 82, ["purchase_schema", "histories", "validate", "explain"]
    elif signals["confidence_q"]:
        # Confidence about a prediction (purchase or otherwise). Executed with
        # evidence when purchase data supports it; otherwise an honest limit.
        intent, conf, ops = "confidence", 80, ["purchase_schema", "histories", "validate", "confidence"]
    elif signals["next_buy"] or (signals["recommend"] and purchase_vocab):
        intent, conf, ops = "next_purchase", 86, ["purchase_schema", "histories", "validate", "predict_next"]
    elif signals["recommend"]:
        intent, conf, ops = "recommend", 82, ["purchase_schema", "histories", "validate", "recommend"]
    elif signals["timing"] and purchase_vocab:
        intent, conf, ops = "next_purchase", 80, ["purchase_schema", "histories", "predict_timing"]
    elif signals["future"] or (signals["predict_verb"] and (signals["trend"] or has_date)):
        # A date column alone does not turn a row-level prediction into a
        # forecast. Explicit future language is required; otherwise prefer the
        # supervised intent determined from the question's target/features.
        intent, conf, ops = ("forecast", 85, ["forecast"]) if has_date else ("regression", 60, ["regress"])
    elif signals["predict_verb"] and re.search(r"\b(regress|continuous|numeric|amount|value)\b", ql):
        intent, conf, ops = "regression", 78, ["regress"]
    elif signals["predict_verb"] and not (signals["ranking"] or signals["filter"] or signals["stat"]):
        # Bare "predict X from Y" with no descriptive signals -> regression/classification
        # decided later by target type; default to regression path with target validation.
        intent, conf, ops = "regression", 65, ["regress"]
    elif signals["ranking"]:
        ops = ["filter", "groupby", "rank"]
        # Multi-metric ("sales, units shipped, and order counts") -> multi-step plan.
        n_metrics = sum(1 for kind in ("revenue", "units", "orders", "profit", "cost", "price")
                        if any(p in ql for p in METRIC_SYNONYMS[kind]))
        intent, conf = "ranking", 88 if n_metrics >= 2 else 85
        if n_metrics >= 2:
            ops = ["filter", "groupby_multi", "rank_each"]
    elif signals["trend"]:
        intent, conf, ops = "trend", 82, ["filter", "groupby_time", "trend"]
    elif signals["compare"]:
        intent, conf, ops = "comparison", 80, ["filter", "groupby", "compare"]
    elif signals["stat"]:
        intent, conf, ops = "statistics", 78, ["describe"]
    elif signals["filter"]:
        intent, conf, ops = "filter", 75, ["filter"]
    else:
        intent, conf, ops = "descriptive", 60, ["describe"]
        # Bare "which product lines have ..." without superlative still descriptive/ranking-ish.
        if re.search(r"\bwhich\b.*\b(have|has|show)\b", ql):
            intent, conf, ops = "ranking", 65, ["filter", "groupby", "rank"]

    return {
        "intent": intent,
        "confidence": conf,
        "signals": signals,
        "needs_ml": intent in ML_INTENTS,
        "operations": ops,
        "sub_intents": [k for k in ("timing", "confidence_q", "explain_q", "buyers", "churn")
                        if signals.get(k)],
    }


_HORIZON_NUMBER_WORDS = {**_NUMBER_WORDS, "eleven": 11, "twelve": 12}


def extract_forecast_horizon(question: str, default: int = 3, maximum: int = 36) -> int:
    """Read a bounded future-step count without assuming a particular frequency."""
    match = re.search(
        r"\b(?:next|coming|following)\s+(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
        r"(?:days?|weeks?|months?|quarters?|years?|periods?)\b", question or "", re.I)
    if match:
        raw = match.group("count").lower()
        count = int(raw) if raw.isdigit() else _HORIZON_NUMBER_WORDS.get(raw, default)
    elif re.search(r"\b(?:next|coming|following)\s+(?:day|week|month|quarter|year)\b", question or "", re.I):
        count = 1
    else:
        count = default
    return max(1, min(int(count), maximum))


# ---------------------------------------------------------------------------
# Dataset profiling
# ---------------------------------------------------------------------------

def _infer_role(series: pd.Series, name: str) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        # Numeric identifiers (id-like names + unique per row) are not measures.
        if re.search(r"(^|_)(id|uuid|guid|code|number|no|seq|key)$", str(name).lower()) and series.nunique(dropna=True) == len(series):
            return "identifier"
        return "numeric"
    values = series.dropna()
    if len(values) and any(isinstance(value, Decimal) for value in values.head(200)) and all(
        isinstance(value, (Decimal, int, float, np.number)) for value in values.head(200)
    ):
        return 'numeric'
    # Try datetime parse for object columns.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(series.dropna().head(200), errors="coerce")
        if len(parsed) and parsed.notna().mean() >= 0.8:
            return "datetime"
    except Exception:
        pass
    non_null = series.dropna()
    if len(series) and non_null.nunique() == len(series) and len(series) > 10:
        return "identifier"
    nunique = int(non_null.nunique()) if len(non_null) else 0
    if nunique <= 500 and (nunique / max(len(non_null), 1)) <= 0.5:
        return "categorical"
    if re.search(r"(desc|note|comment|text|name|address|email)", str(name).lower()) and nunique > 50:
        return "text"
    return "categorical" if nunique <= 2000 else "text"


def profile_dataset(frame: pd.DataFrame, dataset_id: str = "") -> dict[str, Any]:
    """Build reusable metadata for the live frame. No hardcoded column names."""
    columns: list[dict[str, Any]] = []
    for col in frame.columns:
        name = str(col)
        series = frame[col]
        total = len(series)
        non_null = int(series.notna().sum())
        null_count = int(total - non_null)
        try:
            nunique = int(series.nunique(dropna=True))
        except Exception:
            nunique = 0
        role = _infer_role(series, name)
        entry: dict[str, Any] = {
            "name": name, "dtype": str(series.dtype), "role": role,
            "non_null": non_null, "null_count": null_count,
            "null_rate": round(null_count / total, 4) if total else 0.0,
            "unique_count": nunique,
            "samples": [str(v) for v in series.dropna().head(5).tolist()],
            "is_constant": nunique <= 1,
            "is_empty": non_null == 0,
        }
        if role == "numeric":
            numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(numeric):
                entry["stats"] = {
                    "min": float(numeric.min()), "max": float(numeric.max()),
                    "mean": float(numeric.mean()), "median": float(numeric.median()),
                    "std": float(numeric.std() or 0.0), "sum": float(numeric.sum()),
                }
                entry["is_additive_candidate"] = not re.search(
                    r"(price|pct|percent|ratio|rate|margin|score|year|month|day|lat|lon|zip)",
                    name.lower())
            else:
                entry["stats"] = {}
                entry["is_additive_candidate"] = False
        if role == "categorical":
            try:
                top = series.value_counts(dropna=True).head(10)
                entry["top_values"] = [{"value": str(k), "count": int(v)} for k, v in top.items()]
            except Exception:
                entry["top_values"] = []
        columns.append(entry)

    roles = {}
    for c in columns:
        roles.setdefault(c["role"], []).append(c["name"])
    date_cols = roles.get("datetime", [])
    # Detect datetime-like object columns missed above (full-column check, capped).
    return {
        "dataset_id": dataset_id,
        "signature": dataset_signature(frame, dataset_id),
        "rows": int(len(frame)),
        "cols": int(len(frame.columns)),
        "columns": columns,
        "roles": roles,
        "numeric_columns": roles.get("numeric", []),
        "categorical_columns": roles.get("categorical", []),
        "datetime_columns": date_cols,
        "identifier_columns": roles.get("identifier", []),
        "duplicate_rows": int(frame.duplicated().sum()),
        "time_coverage": _time_coverage(frame, date_cols),
    }


def _time_coverage(frame: pd.DataFrame, date_cols: list[str]) -> dict[str, Any] | None:
    if not date_cols:
        return None
    try:
        parsed = pd.to_datetime(frame[date_cols[0]], errors="coerce").dropna()
        if not len(parsed):
            return None
        return {"column": date_cols[0], "min": str(parsed.min()),
                "max": str(parsed.max()), "periods": int(parsed.nunique())}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Schema-aware mapping
# ---------------------------------------------------------------------------

def _score_tokens(name: str, scores: list[tuple[str, int]]) -> int:
    lowered = _norm(name).replace(" ", "_")
    best = 0
    for token, score in scores:
        if token in lowered:
            best = max(best, score)
    return best


def resolve_dimension_column(frame: pd.DataFrame, profile: dict[str, Any],
                             kind: str, question: str) -> str | None:
    """Resolve grouping dimension (e.g. product line) from live schema."""
    candidates = [c for c in profile["columns"]
                  if c["role"] in ("categorical",) and not c["is_empty"] and not c["is_constant"]]
    if not candidates:
        return None
    ql = (question or "").lower()
    if kind == "product_line":
        scores_map = PRODUCT_TOKEN_SCORES
    elif kind == "customer_segment":
        scores_map = SEGMENT_TOKEN_SCORES
    else:
        scores_map = [(kind, 90)]
    scored = []
    for c in candidates:
        s = _score_tokens(c["name"], scores_map)
        # Prefer dimensions actually mentioned/constrained sensibly: low-cardinality grouping.
        if c["unique_count"] > 500:
            s -= 30
        if any(t in ql for t in [c["name"].lower().replace("_", " ")]):
            s += 25
        scored.append((s, -c["unique_count"], c["name"]))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return scored[0][2] if scored and scored[0][0] > 0 else None


def resolve_metric_column(frame: pd.DataFrame, profile: dict[str, Any], kind: str,
                          *, allow_fallback: bool = True) -> str | None:
    """Resolve a metric column for a canonical kind; validates numeric role + variance."""
    tokens = METRIC_COLUMN_TOKENS.get(kind, [kind])
    numeric = {c["name"]: c for c in profile["columns"] if c["role"] == "numeric" and not c["is_empty"]}
    if not numeric:
        return None
    scored: list[tuple[int, float, str]] = []
    for name, meta in numeric.items():
        lowered = _norm(name).replace(" ", "_")
        token_score = 0
        for tok in tokens:
            t = _norm(tok).replace(" ", "_")
            if lowered == t:
                token_score = max(token_score, 100)
            elif lowered.endswith(t) or lowered.startswith(t):
                token_score = max(token_score, 85)
            elif t in lowered:
                token_score = max(token_score, 60)
        if token_score <= 0:
            continue
        try:
            numeric_values = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
            var = float(numeric_values.var(skipna=True))
        except Exception:
            var = 0.0
        if not np.isfinite(var) or var <= 0:
            continue
        scored.append((token_score, var, name))
    if scored:
        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        return scored[0][2]
    # Fallback for revenue-like kinds only: highest-variance numeric that is not an identifier/date part.
    if allow_fallback and kind in ("revenue", "profit", "cost"):
        fallback = []
        for name, meta in numeric.items():
            if re.search(r"(^id$|_id$|year|month|day|quarter)", name.lower()):
                continue
            try:
                var = float(pd.to_numeric(frame[name], errors="coerce").var(skipna=True) or 0.0)
            except Exception:
                continue
            if var > 0:
                fallback.append((var, name))
        if fallback:
            fallback.sort(key=lambda t: (-t[0], t[1]))
            return fallback[0][1]
    return None


def resolve_filter(frame: pd.DataFrame, profile: dict[str, Any], question: str) -> dict[str, Any] | None:
    """Find a (column, value) filter by matching quoted/known category values in live data.

    Generic: scans categorical columns' unique values for a case-insensitive
    occurrence in the question. Returns None when nothing matches.
    """
    q = question or ""
    ql = q.lower()
    # Explicit "col = value" / "col: value" first.
    m = re.search(r"(\w[\w\s]*?)\s*(=|:|is)\s*['\"]?([A-Za-z0-9 _\-/&]+)['\"]?", q)
    if m:
        col_hint, val_hint = _norm(m.group(1)), m.group(3).strip()
        for c in profile["columns"]:
            if c["role"] not in ("categorical", "boolean", "text"):
                continue
            if _norm(c["name"]) == col_hint or col_hint in _norm(c["name"]) or _norm(c["name"]) in col_hint:
                actual = _match_value(frame[c["name"]], val_hint)
                if actual is not None:
                    return {"column": c["name"], "value": actual, "matched_hint": val_hint}
    # Quoted literals: "OEM", 'OEM'.
    for quoted in re.findall(r"['\"]([^'\"]{1,60})['\"]", q):
        actual_col, actual_val = _find_value_everywhere(frame, profile, quoted)
        if actual_val is not None:
            return {"column": actual_col, "value": actual_val, "matched_hint": quoted}
    # "<column-token> <value>" pattern (e.g. "cohort B", "segment OEM"): when a
    # categorical column's name token appears, treat the following token as a
    # candidate value -- this also supports single-character values.
    q_words = re.findall(r"[a-z0-9]+", ql)
    for c in profile["columns"]:
        if c["role"] not in ("categorical", "boolean"):
            continue
        name_tokens = set(_norm(c["name"]).split())
        for idx, word in enumerate(q_words):
            if word in name_tokens and idx + 1 < len(q_words):
                if q_words[idx + 1] in {'by', 'for', 'where', 'with', 'and', 'or', 'of', 'in'}:
                    continue
                actual = _match_value(frame[c["name"]], q_words[idx + 1])
                if actual is not None:
                    return {"column": c["name"], "value": actual,
                            "matched_hint": q_words[idx + 1]}
    # Unquoted: match any categorical unique value appearing as a word/phrase.
    best: tuple[int, str, str, str] | None = None  # (len, col, val, hint)
    for c in profile["columns"]:
        if c["role"] not in ("categorical", "boolean"):
            continue
        try:
            uniques = [str(v) for v in frame[c["name"]].dropna().unique().tolist()][:2000]
        except Exception:
            continue
        for v in uniques:
            vl = v.lower()
            if len(vl) < 2:
                continue
            if re.search(r"(^|[^a-z0-9])" + re.escape(vl) + r"([^a-z0-9]|$)", ql):
                cand = (len(vl), c["name"], v, v)
                if best is None or cand[0] > best[0]:
                    best = cand
    if best:
        return {"column": best[1], "value": best[2], "matched_hint": best[3]}
    return None


def _match_value(series: pd.Series, hint: str) -> str | None:
    try:
        uniques = [str(v) for v in series.dropna().unique().tolist()]
    except Exception:
        return None
    hn = _norm_nospace(hint)
    for v in uniques:
        if _norm_nospace(v) == hn:
            return v
    for v in uniques:
        if len(hn) >= 3 and len(_norm_nospace(v)) >= 3 and (hn in _norm_nospace(v) or _norm_nospace(v) in hn):
            return v
    return None


def _find_value_everywhere(frame: pd.DataFrame, profile: dict[str, Any], hint: str) -> tuple[str | None, str | None]:
    for c in profile["columns"]:
        if c["role"] not in ("categorical", "boolean", "text"):
            continue
        actual = _match_value(frame[c["name"]], hint)
        if actual is not None:
            return c["name"], actual
    return None, None


def detect_requested_metrics(question: str) -> list[str]:
    ql = (question or "").lower()
    kinds = []
    for kind, phrases in METRIC_SYNONYMS.items():
        if any(p in ql for p in phrases):
            kinds.append(kind)
    # "units shipped, and order counts" etc. imply plural metrics; keep order stable.
    order = ["revenue", "units", "orders", "profit", "cost", "price"]
    return sorted(kinds, key=order.index)


def extract_top_k(question: str, default: int = 1) -> int:
    """Top-k from 'top 3' / 'top three' phrasing; defaults to 1 (3 for bare plural asks)."""
    ql = (question or "").lower()
    m = re.search(r"\btop\s+(\d{1,2})\b", ql)
    if m:
        return max(1, min(int(m.group(1)), 20))
    for word, num in _NUMBER_WORDS.items():
        if re.search(r"\btop\s+" + word + r"\b", ql):
            return num
    if re.search(r"\btop\b.*\bproducts?\b", ql):
        return 3
    return default


def extract_entities(question: str, frame: pd.DataFrame, profile: dict[str, Any],
                     schema: dict[str, Any] | None = None,
                     context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve customer / product references against live values (never invented).

    Returns {customer_id, customer_status, product, product_status, top_k} where
    status is 'resolved' | 'clarify' | 'not_found' | 'not_needed'.
    """
    q = question or ""
    context = context or {}
    schema = schema or {}
    mapping = schema.get("mapping", {}) if isinstance(schema, dict) else {}
    entities: dict[str, Any] = {"top_k": extract_top_k(q)}

    # --- customer ---
    ccol = mapping.get("customer_col")
    customer_status, customer_id = "not_needed", None
    needs_customer = True
    if ccol and ccol in frame.columns:
        try:
            uniques = [str(v) for v in frame[ccol].dropna().unique().tolist()]
        except Exception:
            uniques = []
        norm_map = {_norm_nospace(v): v for v in uniques}
        found = None
        for quoted in re.findall(r"['\"]([^'\"]{1,80})['\"]", q):
            if _norm_nospace(quoted) in norm_map:
                found = norm_map[_norm_nospace(quoted)]
                break
        if found is None:
            m = re.search(r"\bcustomer(?:\s+id|\s+no|\s+number)?\s*[:=]?\s*([A-Za-z0-9_\-./]+)", q, re.I)
            if m and _norm_nospace(m.group(1)) in norm_map:
                found = norm_map[_norm_nospace(m.group(1))]
        if found is None:
            # Bare token scan for id-like values (validated against live uniques).
            q_tokens = set(re.findall(r"[a-z0-9]+", q.lower()))
            for norm_val, actual in norm_map.items():
                if len(norm_val) >= 3 and norm_val in q_tokens:
                    found = actual
                    break
        ctx_id = context.get("customer_id")
        if found is not None:
            customer_id, customer_status = found, "resolved"
        elif ctx_id is not None and str(ctx_id) in set(uniques):
            customer_id, customer_status = str(ctx_id), "resolved"
        elif ctx_id is not None:
            customer_id, customer_status = None, "not_found"
        elif _THIS_CUSTOMER_HINTS.search(q):
            customer_id, customer_status = None, "clarify"
        elif needs_customer and re.search(r"customer|client|user|buyer|shopper|who will|this\b", q, re.I):
            # Question is customer-specific but names nobody.
            customer_id, customer_status = None, "clarify"
    entities["customer_id"] = customer_id
    entities["customer_status"] = customer_status
    entities["customer_col"] = ccol

    # --- product ---
    pcol = mapping.get("product_col")
    product_status, product_val = "not_needed", None
    if pcol and pcol in frame.columns and re.search(r"product|item|sku|buy|purchase|order", q, re.I):
        try:
            puniques = [str(v) for v in frame[pcol].dropna().unique().tolist()]
        except Exception:
            puniques = []
        pnorm = {_norm_nospace(v): v for v in puniques}
        pfound = None
        for quoted in re.findall(r"['\"]([^'\"]{1,80})['\"]", q):
            if _norm_nospace(quoted) in pnorm:
                pfound = pnorm[_norm_nospace(quoted)]
                break
        if pfound is None:
            ql = q.lower()
            best = None
            for norm_val, actual in pnorm.items():
                # Short SKUs (e.g. "P1") are common product codes: allow len>=2 here.
                if len(norm_val) >= 2 and re.search(r"(^|[^a-z0-9])" + re.escape(norm_val) + r"([^a-z0-9]|$)", ql):
                    if best is None or len(norm_val) > len(best[0]):
                        best = (norm_val, actual)
            if best:
                pfound = best[1]
        ctx_prod = context.get("product_id") or context.get("product")
        if pfound is not None:
            product_val, product_status = pfound, "resolved"
        elif ctx_prod is not None and str(ctx_prod) in set(puniques):
            product_val, product_status = str(ctx_prod), "resolved"
    entities["product"] = product_val
    entities["product_status"] = product_status
    entities["product_col"] = pcol
    return entities


# ---------------------------------------------------------------------------
# Deterministic execution engine (validated pandas ops only)
# ---------------------------------------------------------------------------

_METRIC_AGG: dict[str, str] = {
    "revenue": "sum", "units": "sum", "orders": "sum_or_count",
    "profit": "sum", "cost": "sum", "price": "weighted_or_mean",
}


def _aggregation_for(frame: pd.DataFrame, profile: dict[str, Any], column: str, kind: str) -> dict[str, Any]:
    meta = next((c for c in profile["columns"] if c["name"] == column), {})
    if kind == "orders" and column in frame.columns:
        # order_count-like columns hold per-group counts -> sum when non-overlapping;
        # otherwise count rows / distinct order ids.
        if re.search(r"count|orders|transactions", column.lower()):
            return {"func": "sum", "assumption": f"summed '{column}' (per-group counts)"}
        id_col = next((c["name"] for c in profile["columns"]
                       if c["role"] == "identifier" and re.search(r"order", c["name"], re.I)), None)
        if id_col:
            return {"func": "nunique", "id_column": id_col,
                    "assumption": f"distinct count of '{id_col}'"}
        return {"func": "count", "assumption": "row count (no order-count measure found)"}
    if kind == "price":
        units_col = resolve_metric_column(frame, profile, "units")
        if units_col and column != units_col:
            return {"func": "weighted_mean", "weight": units_col,
                    "assumption": f"weighted mean of '{column}' by '{units_col}'"}
        return {"func": "mean", "assumption": f"mean of '{column}'"}
    if kind in ("revenue", "units", "profit", "cost"):
        additive = bool(meta.get("is_additive_candidate", True))
        if additive:
            return {"func": "sum", "assumption": f"sum of '{column}' (additive)"}
        return {"func": "mean", "assumption": f"mean of '{column}' (non-additive guard)"}
    return {"func": "sum", "assumption": f"sum of '{column}'"}


_AGGREGATION_WORDS: dict[str, set[str]] = {
    "sum": {"sum", "total", "aggregate", "combined", "altogether"},
    "mean": {"mean", "average", "avg"},
    "median": {"median", "midpoint"},
    "min": {"minimum", "smallest", "least"},
    "max": {"maximum", "largest", "greatest"},
    "count": {"count", "number"},
}


def _column_mentions(question: str, profile: dict[str, Any], roles: set[str] | None = None) -> list[str]:
    """Resolve literal schema mentions after linguistic normalization.

    This is intentionally schema-first: it does not require a fixed list of
    question templates and it never invents a column. Similarity is used only
    for high-confidence single-token spelling variants; ambiguous matches are
    returned to the caller for clarification.
    """
    normalized = f" {_norm(question)} "
    candidates: list[tuple[int, str]] = []
    for column in profile.get("columns", []):
        if roles and column.get("role") not in roles:
            continue
        name = str(column["name"])
        normalized_name = _norm(name)
        if normalized_name and f" {normalized_name} " in normalized:
            candidates.append((len(normalized_name.split()), name))
    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [name for _, name in candidates]

    # Conservative typo tolerance for a schema term: require a strong match
    # and a clear margin over the next candidate. No fuzzy value matching.
    tokens = _norm(question).split()
    scored: list[tuple[float, str]] = []
    for column in profile.get("columns", []):
        if roles and column.get("role") not in roles:
            continue
        name_tokens = _norm(column["name"]).split()
        if len(name_tokens) != 1:
            continue
        score = max((SequenceMatcher(None, token, name_tokens[0]).ratio() for token in tokens), default=0.0)
        if score >= 0.90:
            scored.append((score, str(column["name"])))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if scored and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.08):
        return [scored[0][1]]
    return []


def parse_grouped_aggregation(frame: pd.DataFrame, profile: dict[str, Any], question: str) -> dict[str, Any] | None:
    """Parse a simple grouped aggregation from a free-text question.

    The parser composes an operation from a requested aggregation, literal
    measure, grouping cue, and schema columns. It does not infer a measure from
    a business domain or execute unsupported expressions.
    """
    words = set(_norm(question).split())
    aggregations = [name for name, terms in _AGGREGATION_WORDS.items() if words & terms]
    if len(aggregations) != 1:
        return None
    measure_names = _column_mentions(question, profile, {"numeric"})
    if len(measure_names) != 1:
        return None
    normalized = f" {_norm(question)} "
    cue_match = re.search(r"\b(?:by|per|each|every)\b", normalized)
    if not cue_match:
        return None
    grouping_tail = normalized[cue_match.end():]
    grouping_names = [name for name in _column_mentions(grouping_tail, profile, {"categorical", "boolean"})]
    if len(grouping_names) != 1:
        return None
    return {"type": "grouped_aggregation", "aggregation": aggregations[0],
            "measure": measure_names[0], "group_by": grouping_names[0],
            "filters": [], "sort_direction": None, "limit": None}


def execute_grouped_aggregation(frame: pd.DataFrame, operation: dict[str, Any]) -> dict[str, Any]:
    measure, group_by, aggregation = operation["measure"], operation["group_by"], operation["aggregation"]
    valid = frame.loc[frame[group_by].notna()]
    values = pd.to_numeric(valid[measure], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = valid.assign(__measure=values).dropna(subset=["__measure"])
    grouped = valid.groupby(group_by, dropna=True)["__measure"]
    if aggregation == "sum":
        result = grouped.sum(min_count=1)
    elif aggregation == "mean":
        result = grouped.mean()
    elif aggregation == "median":
        result = grouped.median()
    elif aggregation == "min":
        result = grouped.min()
    elif aggregation == "max":
        result = grouped.max()
    elif aggregation == "count":
        result = grouped.count()
    else:
        return {"status": "unsupported", "intent": "aggregation",
                "message": f"Aggregation '{aggregation}' is not supported."}
    operation = {**operation, "null_handling": "Rows missing the measure or group value are excluded."}
    return {"status": "success", "intent": "aggregation", "operation": operation,
            "grouped_result": {str(key): float(value) for key, value in result.items()},
            "coverage": {"filtered_rows": int(len(valid)), "total_rows": int(len(frame))}}


def _unmatched_filter_message(profile: dict[str, Any], question: str) -> str | None:
    """Recognize a categorical field/value constraint that failed resolution."""
    normalized = f" {_norm(question)} "
    for column in profile.get("columns", []):
        if column.get("role") not in ("categorical", "boolean"):
            continue
        name = _norm(column.get("name", ""))
        if not name or f" {name} " not in normalized:
            continue
        if re.search(r"\b(?:each|every|per)\s+" + re.escape(name) + r"\b", normalized):
            continue
        # Common compositional forms: "in the X region", "region X", and
        # "region is X". The value is still matched against live data below.
        patterns = [
            r"\b(?:in|within|for)\s+(?:the\s+)?([a-z0-9_-]+(?:\s+[a-z0-9_-]+){0,2})\s+" + re.escape(name) + r"\b",
            r"\b" + re.escape(name) + r"\s+(?:is|equals|=|:)\s+([a-z0-9_-]+(?:\s+[a-z0-9_-]+){0,2})",
            r"\b" + re.escape(name) + r"\s+([a-z0-9_-]+)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if not match:
                continue
            value_hint = match.group(1).strip()
            if value_hint in {"each", "every", "the", "by", "for", "from", "where", "using",
                              "highest", "lowest", "with", "and", "or", "as", "to", "of"}:
                continue
            return f"The requested value '{value_hint}' could not be matched to column '{column['name']}'. Check the value or choose one present in the dataset."
    return None


def _apply_filter(frame: pd.DataFrame, filt: dict[str, Any] | None) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if not filt:
        return frame, None
    col, val = filt["column"], filt["value"]
    if col not in frame.columns:
        return frame, {"error": f"Filter column '{col}' not in dataset."}
    mask = frame[col].astype(str).str.lower() == str(val).lower()
    return frame[mask].copy(), {"column": col, "value": str(val), "matched_rows": int(mask.sum())}


def execute_ranking_plan(frame: pd.DataFrame, profile: dict[str, Any], question: str,
                         top_n: int = 10) -> dict[str, Any]:
    """Multi-metric ranking: filter -> groupby each metric -> independent ranks."""
    filt = resolve_filter(frame, profile, question)
    explicit_filter = re.search(r'\bwhere\s+([\w ]+?)\s*(?:=|:|\bis\b)\s*[\'"]?([^\'"]+)', question, re.I)
    if explicit_filter:
        column_hint = _norm(explicit_filter.group(1))
        value_hint = explicit_filter.group(2).strip().rstrip('?.')
        column = next((str(name) for name in frame.columns if _norm(name) == column_hint), None)
        value = _match_value(frame[column], value_hint) if column else None
        if column is None or value is None:
            return {'status': 'clarify', 'intent': 'ranking',
                    'message': 'The requested filter could not be matched. Check the column and value.'}
        filt = {'column': column, 'value': value, 'matched_hint': value_hint}
    filtered, filter_info = _apply_filter(frame, filt)
    if filtered.empty:
        return {"status": "empty",
                "message": ("No records matched the requested filter. "
                            + (f"Filter was {filt['column']} = '{filt['value']}'." if filt else "No filter resolved.")),
                "filter": filt, "intent": "ranking"}

    normalized_question = _norm(question)
    subject = re.split(r'\b(?:by|for|where|with|using)\b', normalized_question, maxsplit=1)[0]
    explicit_dimensions = [c['name'] for c in profile['columns']
                           if c['role'] == 'categorical' and re.search(
                               r'(?<!\w)' + re.escape(_norm(c['name'])) + r's?(?!\w)', subject)]
    group_col = explicit_dimensions[0] if len(explicit_dimensions) == 1 else resolve_dimension_column(frame, profile, "product_line", question)
    if group_col is None:
        # Generic fallback: best low-cardinality categorical, preferring a
        # dimension actually named in the question.
        ql = (question or "").lower()
        cands = [c for c in profile["columns"] if c["role"] == "categorical"
                 and not c["is_empty"] and c["unique_count"] <= 50 and c["unique_count"] >= 2]
        if not cands:
            return {"status": "unsupported",
                    "message": "No suitable grouping dimension found in this dataset.",
                    "intent": "ranking"}
        def _cand_key(c: dict[str, Any]) -> tuple[int, int, str]:
            mentioned = 0 if c["name"].lower().replace("_", " ") in ql else 1
            return (mentioned, c["unique_count"], c["name"])
        cands.sort(key=_cand_key)
        group_col = cands[0]["name"]

    metric_kinds = detect_requested_metrics(question)
    explicit_numeric = [name for name in profile['numeric_columns']
                        if re.search(r'(?<!\w)' + re.escape(_norm(name)) + r'(?!\w)', normalized_question)]
    if not metric_kinds and explicit_numeric:
        if not re.search(r'\b(?:sum|total|average|mean|count|minimum|maximum)\b', normalized_question):
            return {'status': 'unsupported', 'intent': 'ranking',
                    'message': 'Specify how to aggregate the named measure, such as total or average.'}
        metric_kinds = explicit_numeric
    if not metric_kinds:
        # Default: first numeric measure.
        nums = profile["numeric_columns"]
        if not nums:
            return {"status": "unsupported", "message": "No numeric measures found.", "intent": "ranking"}
        metric_kinds = ["revenue"]
    resolved: list[dict[str, Any]] = []
    ambiguous: list[str] = []
    for kind in metric_kinds:
        col = kind if kind in explicit_numeric else resolve_metric_column(frame, profile, kind, allow_fallback=False)
        if col is None:
            if kind == "orders":
                # No explicit order-count measure: fall back to row counts per group
                # (e.g. daily shipment ledgers without an order_count column).
                resolved.append({"kind": kind, "column": None,
                                 "agg": {"func": "count",
                                         "assumption": "row count per group (no order-count measure in schema)"}})
                continue
            ambiguous.append(kind)
            continue
        aggregation = _aggregation_for(frame, profile, col, kind)
        if re.search(r'\b(?:average|mean)\b', normalized_question):
            aggregation = {'func': 'mean', 'assumption': f"mean of '{col}' (requested)"}
        elif kind in explicit_numeric and re.search(r'\b(?:sum|total)\b', normalized_question):
            aggregation = {'func': 'sum', 'assumption': f"sum of '{col}' (requested)"}
        resolved.append({"kind": kind, "column": col, "agg": aggregation})

    if not resolved:
        return {"status": "unsupported",
                "message": ("None of the requested metrics (" + ", ".join(ambiguous)
                            + ") map to numeric columns in this dataset. Available numeric: "
                            + ", ".join(profile["numeric_columns"]) + "."),
                "intent": "ranking", "filter": filt}

    tables: dict[str, Any] = {}
    winners: dict[str, Any] = {}
    for r in resolved:
        col, kind, agg = r["column"], r["kind"], r["agg"]
        func = agg["func"]
        try:
            if func == "sum":
                g = filtered.groupby(group_col, dropna=True)[col].sum(min_count=1)
            elif func == "mean":
                g = filtered.groupby(group_col, dropna=True)[col].mean()
            elif func == "nunique":
                g = filtered.groupby(group_col, dropna=True)[agg["id_column"]].nunique()
            elif func == "count":
                g = filtered.groupby(group_col, dropna=True).size()
            elif func == "weighted_mean":
                valid = filtered[col].notna() & filtered[agg['weight']].notna()
                pair = filtered.loc[valid]
                if (pair[agg['weight']] < 0).any():
                    raise ValueError('Weighted averages require nonnegative weights.')
                w = pair.groupby(group_col, dropna=True)[agg["weight"]].sum(min_count=1)
                s = (pair[col] * pair[agg["weight"]]).groupby(pair[group_col], dropna=True).sum(min_count=1)
                g = (s / w.replace(0, np.nan)).dropna()
            else:
                g = filtered.groupby(group_col, dropna=True)[col].sum(min_count=1)
        except Exception as exc:
            tables[kind] = {"error": str(exc)}
            continue
        ascending = bool(re.search(r'\b(?:lowest|bottom|smallest|least|ascending)\b', normalized_question))
        requested = re.search(r'\b(?:top|bottom|highest|lowest|largest|smallest)\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b', normalized_question)
        count = requested.group(1) if requested else None
        limit = min(50, max(1, int(count) if count and count.isdigit() else _NUMBER_WORDS.get(count, top_n)))
        g = g.dropna()
        order = sorted(g.index, key=lambda idx: (g.loc[idx] if ascending else -g.loc[idx], str(idx)))
        ranked = g.loc[order]
        # Tie handling: dense rank, report ties explicitly.
        rows = []
        dense_rank = 0
        prior_value = None
        for idx, val in ranked.head(limit).items():
            if prior_value is None or val != prior_value:
                dense_rank += 1
            rows.append({"group": str(idx), "value": float(val), "rank": dense_rank,
                         "tied": bool((ranked == val).sum() > 1),
                         **({'exact_value': str(val)} if isinstance(val, Decimal) else {})})
            prior_value = val
        tables[kind] = {"column": col, "agg": func, "assumption": agg.get("assumption"),
                        "rows": rows, "groups_ranked": int(len(ranked)),
                        "direction": "ascending" if ascending else "descending"}
        if len(ranked):
            top_val = float(ranked.iloc[0])
            tied = ranked[ranked == ranked.iloc[0]].index.tolist()
            winners[kind] = {"group": str(ranked.index[0]), "value": top_val,
                             "tied_with": [str(t) for t in tied[1:]]}

    return {
        "status": "success", "intent": "ranking",
        "filter": filt, "filter_info": filter_info,
        "group_by": group_col, "metrics": resolved,
        "operation": {"type": "ranking",
                      "measures": [{"kind": item["kind"], "column": item["column"],
                                    "aggregation": item["agg"]["func"]} for item in resolved],
                      "group_by": group_col, "filters": [filt] if filt else [],
                      "sort_direction": "ascending" if re.search(r'\b(?:lowest|bottom|smallest|least|ascending)\b', normalized_question) else "descending",
                      "limit": limit},
        "ambiguous_metrics": ambiguous, "tables": tables, "winners": winners,
        "coverage": {"filtered_rows": int(len(filtered)), "total_rows": int(len(frame)),
                     "time_coverage": profile.get("time_coverage")},
    }


def execute_simple(frame: pd.DataFrame, profile: dict[str, Any], question: str,
                   intent: str) -> dict[str, Any]:
    """Descriptive / filter / comparison / statistics / trend execution."""
    filt = resolve_filter(frame, profile, question)
    filtered, filter_info = _apply_filter(frame, filt)
    if intent == "filter":
        cols = list(frame.columns)[:20]
        return {"status": "success", "intent": intent, "filter": filt, "filter_info": filter_info,
                "rows": filtered.head(50).astype(str).to_dict(orient="records"),
                "matched_rows": int(len(filtered)), "total_rows": int(len(frame)),
                "columns": cols}
    if intent == "trend":
        date_cols = profile["datetime_columns"]
        if not date_cols:
            return {"status": "unsupported", "intent": intent,
                    "message": "Trend analysis needs a date column; none detected.",
                    "filter": filt}
        metric_kinds = detect_requested_metrics(question) or ["revenue"]
        col = resolve_metric_column(frame, profile, metric_kinds[0])
        if col is None:
            col = (profile["numeric_columns"] or [None])[0]
            if col is None:
                return {"status": "unsupported", "intent": intent, "message": "No numeric measure for trend."}
        try:
            tmp = filtered.copy()
            tmp["_period"] = pd.to_datetime(tmp[date_cols[0]], errors="coerce")
            tmp = tmp.dropna(subset=["_period"])
            if tmp.empty:
                return {"status": "empty", "intent": intent, "message": "No valid dates after filtering."}
            grain = "ME" if (tmp["_period"].max() - tmp["_period"].min()).days > 120 else "W-MON" if (tmp["_period"].max() - tmp["_period"].min()).days > 21 else "D"
            tmp["_bucket"] = tmp["_period"].dt.to_period("M" if grain == "ME" else "W" if grain.startswith("W") else "D").astype(str)
            g = tmp.groupby("_bucket")[col].sum(min_count=1)
            return {"status": "success", "intent": intent, "filter": filt,
                    "date_column": date_cols[0], "metric": col, "grain": grain,
                    "points": [{"period": str(k), "value": round(float(v), 2)} for k, v in g.items()],
                    "coverage": {"filtered_rows": int(len(filtered)), "total_rows": int(len(frame))}}
        except Exception as exc:
            return {"status": "error", "intent": intent, "message": str(exc)}
    if intent in ("comparison", "descriptive", "statistics"):
        metric_kinds = detect_requested_metrics(question) or []
        numeric_cols = profile["numeric_columns"][:6]
        summary = {}
        for col in numeric_cols:
            s = pd.to_numeric(filtered[col], errors="coerce").dropna()
            if len(s):
                summary[col] = {"count": int(len(s)), "mean": round(float(s.mean()), 2),
                                "median": round(float(s.median()), 2), "std": round(float(s.std() or 0), 2),
                                "min": round(float(s.min()), 2), "max": round(float(s.max()), 2),
                                "sum": round(float(s.sum()), 2)}
        group_col = resolve_dimension_column(frame, profile, "product_line", question)
        grouped = None
        if group_col and metric_kinds:
            grouped = {}
            for kind in metric_kinds:
                col = resolve_metric_column(frame, profile, kind)
                if col:
                    try:
                        grouped[kind] = (filtered.groupby(group_col)[col].sum(min_count=1)
                                         .sort_values(ascending=False).head(10).round(2)
                                         .to_dict())
                    except Exception:
                        pass
        corr = None
        if intent == "statistics" and len(numeric_cols) >= 2:
            try:
                corr = filtered[numeric_cols].corr(numeric_only=True).round(3).fillna(0).to_dict()
            except Exception:
                corr = None
        return {"status": "success", "intent": intent, "filter": filt, "filter_info": filter_info,
                "summary": summary, "grouped": grouped, "correlations": corr,
                "operation": {"type": intent, "columns": list(summary),
                              "filters": [filt] if filt else [],
                              "assumptions": []},
                "coverage": {"filtered_rows": int(len(filtered)), "total_rows": int(len(frame))}}
    return {"status": "unsupported", "intent": intent,
            "message": f"Intent '{intent}' is not executable deterministically on this dataset."}


# ---------------------------------------------------------------------------
# Purchase prediction execution (next-purchase / recommend / buyers / churn /
# confidence / explain). All values computed from the live frame.
# ---------------------------------------------------------------------------

def _purchase_plan(intent: str, routing: dict[str, Any], entities: dict[str, Any],
                   schema: dict[str, Any], status: str) -> dict[str, Any]:
    m = (schema.get("mapping", {}) or {})
    return {
        "intent": intent,
        "sub_intents": routing.get("sub_intents", []),
        "operations": routing.get("operations", []),
        "entities": {k: v for k, v in entities.items()},
        "columns": {k: v for k, v in m.items() if v},
        "grain": schema.get("grain"),
        "data_checks": {"sufficient": schema.get("sufficient"),
                        "issues": schema.get("issues", [])},
        "status": status,
    }


def _closest_alternative(schema: dict[str, Any]) -> str | None:
    grain = schema.get("grain")
    grouped = any("grouping" in issue for issue in schema.get("issues", []))
    if grain == "aggregate" or grouped:
        return ("This dataset is aggregated (no individual customer histories). "
                "Supported alternatives: aggregate sales ranking, trend analysis, "
                "and time-series forecasting of totals.")
    return None


def execute_purchase(frame: pd.DataFrame, profile: dict[str, Any], question: str,
                     intent: str, routing: dict[str, Any],
                     context: dict[str, Any] | None = None,
                     top_n: int = 10) -> dict[str, Any]:
    """End-to-end purchase flow: schema -> histories -> validate -> execute."""
    context = context or {}
    schema = discover_purchase_schema(frame, profile)

    if not schema["sufficient"]:
        alt = _closest_alternative(schema)
        msg = ("; ".join(schema["issues"]) or
               "This dataset cannot support customer purchase prediction.")
        if alt:
            msg += " " + alt
        else:
            msg += (" Required: a customer identifier, a product/item field, and "
                    "purchase timestamps at transaction grain.")
        return {"status": "unsupported", "intent": intent, "routing": routing,
                "profile": _slim(profile),
                "plan": _purchase_plan(intent, routing,
                                       extract_entities(question, frame, profile, schema, context),
                                       schema, "insufficient_data"),
                "message": msg, "answer": msg}

    built = build_purchase_histories(frame, schema)
    histories = built["customers"]
    report = built["report"]
    clean = built["clean"]
    if not histories:
        msg = "No usable customer purchase histories could be built from this dataset."
        return {"status": "empty", "intent": intent, "routing": routing,
                "profile": _slim(profile),
                "plan": _purchase_plan(intent, routing,
                                       extract_entities(question, frame, profile, schema, context),
                                       schema, "no_histories"),
                "cleaning_report": report, "message": msg, "answer": msg}
    validation = validate_methods_temporal(histories, top_k=3)
    method = validation.get("chosen_method", "repeat_preference")
    # Deterministic reference point: dataset's own max date (never wall-clock).
    try:
        as_of = pd.to_datetime(clean["date"]).max()
    except Exception:
        as_of = None

    entities = extract_entities(question, frame, profile, schema, context)
    top_k = max(1, min(entities.get("top_k", 1) or 1, top_n, 20))
    provenance = {"method": method, "validation": validation,
                  "cleaning_report": {k: v for k, v in report.items()},
                  "schema_assumptions": schema.get("assumptions", []),
                  "customers_with_history": len(histories),
                  "reference_date": str(as_of.date()) if as_of is not None and pd.notna(as_of) else None}

    if intent == "churn":
        risk = customers_at_risk(histories, as_of=as_of, limit=top_n)
        answer = _render_churn(risk, as_of)
        return {"status": "success", "intent": intent, "routing": routing,
                "profile": _slim(profile), "at_risk": risk["at_risk"],
                "evaluated": risk["evaluated"], "rule": risk["rule"],
                "plan": _purchase_plan(intent, routing, entities, schema, "completed"),
                "provenance": provenance, "answer": answer}

    if intent == "buyers":
        if not entities["product"]:
            msg = ("Which product should I find buyers for? Please name a product "
                   "from the dataset (e.g. in quotes).")
            return {"status": "clarify", "intent": intent, "routing": routing,
                    "profile": _slim(profile), "needs": ["product"],
                    "plan": _purchase_plan(intent, routing, entities, schema, "needs_product"),
                    "provenance": provenance, "message": msg, "answer": msg}
        res = customers_for_product(clean, histories, entities["product"], limit=top_n)
        answer = _render_buyers(res)
        return {"status": "success", "intent": intent, "routing": routing,
                "profile": _slim(profile), "buyers": res["buyers"],
                "buyer_count": res["buyer_count"], "product": res["product"],
                "plan": _purchase_plan(intent, routing, entities, schema, "completed"),
                "provenance": provenance, "answer": answer}

    # next_purchase / recommend / confidence / explain all need one customer.
    if entities["customer_status"] == "clarify" or entities["customer_id"] is None:
        if entities["customer_status"] == "not_found":
            msg = ("The supplied customer reference was not found in the active dataset. "
                   "Please provide a valid customer ID present in "
                   f"'{entities.get('customer_col')}'.")
            needs = ["customer_id"]
        else:
            msg = ("Which customer should I predict for? Please provide a customer ID "
                   "or select a customer.")
            needs = ["customer_id"]
        return {"status": "clarify", "intent": intent, "routing": routing,
                "profile": _slim(profile), "needs": needs,
                "plan": _purchase_plan(intent, routing, entities, schema, "needs_customer"),
                "provenance": provenance, "message": msg, "answer": msg}

    customer = entities["customer_id"]
    pred = predict_next_for_customer(clean, histories, customer, method, top_k=top_k)
    # Label honestly: only temporally validated, non-cold-start output is a prediction.
    if pred.get("cold_start") or not validation.get("validated"):
        pred["label"] = "recommendation"
    else:
        pred["label"] = "prediction"
    timing = predict_timing_for_customer(histories, customer)
    confidence = _confidence_block(pred, validation, timing, histories.get(customer))
    explanation = _explanation_block(pred, timing, validation, schema)
    label = pred.get("label", "prediction")
    answer = _render_prediction(pred, timing, confidence, intent, top_k)

    result: dict[str, Any] = {
        "status": "success", "intent": intent, "routing": routing,
        "profile": _slim(profile), "customer": customer,
        "prediction": pred, "timing": timing, "confidence": confidence,
        "explanation": explanation, "output_label": label,
        "plan": _purchase_plan(intent, routing, entities, schema, "completed"),
        "provenance": provenance, "answer": answer,
    }
    if intent in ("confidence", "explain") and not routing.get("sub_intents"):
        pass
    return result


def _confidence_block(pred: dict[str, Any], validation: dict[str, Any],
                      timing: dict[str, Any], history: dict[str, Any] | None) -> dict[str, Any]:
    """Honest uncertainty: scores are scores; reliability is measured hit-rate.

    Never emits a calibrated probability (these scorers are uncalibrated), so
    ``probability`` type is only ever absent; callers must render
    ``Confidence unavailable`` for probability asks when that is the case.
    """
    top_score = (pred.get("ranked") or [{}])[0].get("score")
    block: dict[str, Any] = {
        "ranking_score": {"type": "ranking_score", "value": top_score,
                          "meaning": "Relative ranking score (0-1) among candidates, NOT a probability."},
        "probability": {"type": "unavailable",
                        "value": None,
                        "reason": ("Confidence unavailable as a probability: the selected "
                                   "counting-based method is not probability-calibrated.")},
    }
    if validation.get("validated"):
        m = validation["metrics"][validation["chosen_method"]]
        block["method_reliability"] = {
            "type": "hit_rate", "hit_rate_at_3": m["hit_rate_at_3"],
            "hit_rate_at_1": m["hit_rate_at_1"], "mrr": m["mrr"],
            "evaluable_customers": validation["evaluable_customers"],
            "meaning": ("Measured on held-out last purchases: share of customers whose "
                        "actual next purchase appeared in the top-k ranked products."),
        }
    else:
        block["method_reliability"] = {
            "type": "unavailable", "value": None,
            "reason": "Confidence unavailable: " + validation.get("reason", "not validated.")}
    if timing.get("status") == "estimate":
        block["timing"] = {"type": "interval_estimate", **{k: v for k, v in timing.items() if k != "status"}}
    else:
        block["timing"] = {"type": "unavailable", "value": None,
                           "reason": timing.get("reason", "Confidence unavailable.")}
    if pred.get("cold_start"):
        block["overall"] = ("Recommendation (not a validated prediction): this customer has no "
                            "history, so global best-sellers are shown.")
    elif validation.get("validated"):
        block["overall"] = ("Validated prediction: method chosen by temporal holdout "
                            f"({validation['chosen_method']}).")
    else:
        block["overall"] = ("Recommendation (not a validated prediction): " +
                            validation.get("reason", "insufficient history for validation."))
    return block


def _explanation_block(pred: dict[str, Any], timing: dict[str, Any],
                       validation: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    items = []
    for r in pred.get("ranked", []):
        ev = r.get("evidence", {})
        items.append({"product": r["product"], "score": r["score"],
                      "times_bought": ev.get("times_bought_by_customer"),
                      "last_bought": ev.get("last_bought"),
                      "also_bought_with": ev.get("also_bought_with", []),
                      "reason": ev.get("reason") or
                      (f"Bought {ev.get('times_bought_by_customer', 0)} time(s), "
                       f"last on {ev.get('last_bought')}.")})
    return {"method": pred.get("method"),
            "why": items,
            "validation": ({k: validation.get(k) for k in ("validated", "evaluable_customers", "chosen_method")}
                           if isinstance(validation, dict) else None),
            "timing_note": timing.get("note") or timing.get("reason"),
            "assumptions": schema.get("assumptions", [])}


def _render_prediction(pred: dict[str, Any], timing: dict[str, Any],
                       confidence: dict[str, Any], intent: str, top_k: int) -> str:
    cust = pred.get("customer")
    ranked = pred.get("ranked", [])
    label = pred.get("label", "prediction")
    if not ranked:
        return f"No candidate products could be scored for customer {cust}."
    if intent == "confidence":
        rel = confidence.get("method_reliability", {})
        if rel.get("type") == "hit_rate":
            return ("Confidence: no calibrated probability is available (scores are ranking "
                    f"scores, top score {confidence['ranking_score']['value']}). Method reliability on "
                    f"held-out last purchases: hit-rate@3 {rel['hit_rate_at_3']}, MRR {rel['mrr']} "
                    f"over {rel['evaluable_customers']} customers using {pred.get('method')}. " +
                    (f"Timing: next purchase estimated around {timing.get('predicted_next_date')} "
                     f"(range {timing.get('range', {}).get('lower')} to {timing.get('range', {}).get('upper')})."
                     if timing.get("status") == "estimate" else timing.get("reason", "")))
        return ("Confidence unavailable as a probability. " +
                (rel.get("reason", "")) + " " + (timing.get("reason", "")
                if timing.get("status") != "estimate" else ""))
    if intent == "explain":
        top = ranked[0]
        ev = top.get("evidence", {})
        return (f"Recommended '{top['product']}' for customer {cust} because: bought "
                f"{ev.get('times_bought_by_customer', 0)} time(s)"
                f"{', last on ' + str(ev.get('last_bought')) if ev.get('last_bought') else ''}"
                f"{'; often bought with ' + ', '.join(ev.get('also_bought_with', [])[:3]) if ev.get('also_bought_with') else ''}. "
                f"Method: {pred.get('method')} ({confidence.get('overall', '')})")
    names = "; ".join(f"{i + 1}. {r['product']} (score {r['score']}, bought "
                      f"{r.get('evidence', {}).get('times_bought_by_customer', 0)}x before)"
                      for i, r in enumerate(ranked))
    verb = "Predicted next purchase" if label == "prediction" else "Recommended"
    text = f"{verb} for customer {cust}: {names}."
    if timing.get("status") == "estimate":
        text += (f" Next purchase estimated around {timing['predicted_next_date']} "
                 f"(range {timing['range']['lower']} to {timing['range']['upper']}).")
    return text


def _render_churn(risk: dict[str, Any], as_of: Any) -> str:
    rows = risk.get("at_risk", [])
    if not rows:
        return ("No customers currently exceed their typical repurchase interval "
                f"({risk.get('evaluated', 0)} customers with repeat history evaluated).")
    top = "; ".join(f"{r['customer']} (silent {r['days_since_last']}d vs typical "
                    f"{r['typical_interval_days']}d, last {r['last_purchase']})" for r in rows[:5])
    return (f"{len(rows)} customer(s) at risk of not purchasing again "
            f"(reference {str(as_of.date()) if as_of is not None and pd.notna(as_of) else 'dataset max date'}): {top}.")


def _render_buyers(res: dict[str, Any]) -> str:
    buyers = res.get("buyers", [])
    if not buyers:
        return f"No recorded buyers of '{res.get('product')}' in this dataset."
    top = "; ".join(f"{b['customer']} ({b['times_bought']}x, last {b['last_bought']})" for b in buyers[:5])
    return (f"{res.get('buyer_count')} customer(s) bought '{res.get('product')}'; "
            f"most likely next buyers: {top}.")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def answer_question(frame: pd.DataFrame, question: str, dataset_id: str = "",
                    top_n: int = 10, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Route + execute + validate. Never fabricates numbers; all values from frame."""
    if frame is None or frame.empty:
        return {"status": "error", "intent": "unknown",
                "message": "No data is loaded. Upload a dataset first.", "answer": None}
    if not question or not question.strip():
        return {"status": "error", "intent": "unknown",
                "message": "Enter a question about the active dataset.", "answer": None}
    profile = profile_dataset(frame, dataset_id)
    routing = classify_intent(question, profile)
    intent = routing["intent"]

    if intent in PURCHASE_INTENTS:
        # Executed end-to-end on the live frame (never a bare ml_plan).
        return execute_purchase(frame, profile, question, intent, routing,
                                context=context, top_n=top_n)

    row_classification_target = _row_classification_target(question, profile)
    explicit_supervised_target = _supervised_hint(question, profile)
    resolved_filter = resolve_filter(frame, profile, question)
    supervised_classification = routing["intent"] == "classification" and explicit_supervised_target is not None
    unresolved_filter = (None if resolved_filter or row_classification_target or supervised_classification
                         else _unmatched_filter_message(profile, question))
    if unresolved_filter:
        return {"status": "clarify", "intent": intent, "routing": routing,
                "message": unresolved_filter, "answer": unresolved_filter,
                "profile": _slim(profile)}

    if intent in ("anomaly", "segmentation"):
        # Executed in-engine via ml_execute (seeded sklearn, no training service needed).
        from ml_execute import run_anomaly_detection, run_segmentation
        executed = (run_anomaly_detection(frame, profile, top_n=top_n)
                    if intent == "anomaly"
                    else run_segmentation(frame, profile))
        executed["routing"] = routing
        executed["profile"] = _slim(profile)
        if "answer" not in executed:
            executed["answer"] = executed.get("message", "Analysis complete.")
        return executed

    if routing["needs_ml"]:
        # Do not run ML here; validate feasibility and hand a grounded ML plan
        # to the existing training/forecasting APIs (preserve genuine ML workflows).
        date_ok = bool(profile["datetime_columns"]) if intent == "forecast" else True
        numeric_ok = bool(profile["numeric_columns"])
        if intent == "forecast" and not date_ok:
            return {"status": "unsupported", "intent": intent, "routing": routing, "profile": _slim(profile),
                    "message": ("Forecasting needs a date column and a numeric target; "
                                "this dataset has neither usable combination."),
                    "answer": ("Forecasting needs a date column and a numeric target; "
                               "this dataset has neither usable combination.")}
        if not numeric_ok:
            return {"status": "unsupported", "intent": intent, "routing": routing, "profile": _slim(profile),
                    "message": "Predictive modelling needs at least one numeric target; none detected.",
                    "answer": "Predictive modelling needs at least one numeric target; none detected."}
        explicit_target = explicit_supervised_target
        if explicit_target is None and row_classification_target:
            explicit_target = {"column": row_classification_target}
        target = explicit_target.get("column") if explicit_target else None
        target_reason = "explicit target named in the question" if target else None
        date_column = None
        if intent == "forecast":
            requested_dates = _column_mentions(question, profile, {"datetime"})
            if len(requested_dates) == 1:
                date_column = requested_dates[0]
            elif len(profile["datetime_columns"]) == 1:
                date_column = profile["datetime_columns"][0]
            elif len(profile["datetime_columns"]) > 1:
                message = ("Which date column should define the forecast timeline? Available date columns: "
                           + ", ".join(profile["datetime_columns"]) + ".")
                return {"status": "clarify", "intent": intent, "routing": routing,
                        "profile": _slim(profile), "needs": ["date_column"],
                        "message": message, "answer": message}
        if intent == "forecast" and target:
            target_meta = next((column for column in profile["columns"] if column["name"] == target), {})
            if target_meta.get("role") != "numeric":
                message = f"Forecast target '{target}' is not a numeric measure. Choose a numeric column to forecast."
                return {"status": "unsupported", "intent": intent, "routing": routing,
                        "profile": _slim(profile), "message": message, "answer": message}
        forecast_group_columns: list[str] = []
        if intent == "forecast":
            normalized_forecast_question = f" {_norm(question)} "
            for column in profile["columns"]:
                if column.get("role") not in {"categorical", "boolean"}:
                    continue
                name = _norm(column["name"])
                if not name:
                    continue
                if re.search(r"\b(?:by|per|for each|within each)\s+" + re.escape(name) + r"\b",
                             normalized_forecast_question):
                    forecast_group_columns.append(str(column["name"]))
            forecast_group_columns = list(dict.fromkeys(forecast_group_columns))
            if len(forecast_group_columns) > 1:
                message = ("This forecast names more than one grouping dimension. "
                           "Choose one dimension or request a single overall forecast.")
                return {"status": "clarify", "intent": intent, "routing": routing,
                        "profile": _slim(profile), "needs": ["group_by"],
                        "message": message, "answer": message}
        if intent == "forecast" and not target:
            requested_metrics = detect_requested_metrics(question)
            metric_kind = requested_metrics[0] if len(requested_metrics) == 1 else None
            if metric_kind:
                target = resolve_metric_column(frame, profile, metric_kind, allow_fallback=False)
                if target:
                    target_reason = f"mapped the requested {metric_kind} measure to an available column"
            if not target and len(profile["numeric_columns"]) == 1:
                target = profile["numeric_columns"][0]
                target_reason = "the dataset has a single numeric column"
            if not target:
                available = ", ".join(profile["numeric_columns"]) or "none"
                message = f"Which numeric column should I forecast? Available numeric columns: {available}."
                return {"status": "clarify", "intent": intent, "routing": routing,
                        "profile": _slim(profile), "needs": ["target"],
                        "message": message, "answer": message}
        if intent != "forecast" and not target:
            # The assistant runner performs its own target suitability checks;
            # this plan value is a review suggestion, not an execution target.
            target = resolve_metric_column(frame, profile, "revenue")
            target_reason = "schema-based target suggestion; requires user review" if target else None
        features = [c for c in list(frame.columns)[:30] if c != target
                    and not re.search(r"(^id$|_id$|uuid)", str(c), re.I)]
        return {"status": "ml_required", "intent": intent, "routing": routing,
                "profile": _slim(profile),
                "ml_plan": {"target": target, "target_reason": target_reason,
                            **({"date_column": date_column} if intent == "forecast" and date_column else {}),
                            **({"group_by": forecast_group_columns[0]} if forecast_group_columns else {}),
                            **({"horizon": extract_forecast_horizon(question)} if intent == "forecast" else {}),
                            "features": features[:12],
                            "task": ("forecast" if intent == "forecast"
                                     else "classification" if _target_looks_categorical(frame, target)
                                     else "regression"),
                            "note": ("Use the existing Time Series / ML Forecast or ML training APIs; "
                                     "no prediction was fabricated here.")},
                "message": ("This is a predictive question. A grounded ML plan is attached; "
                            "run it through the existing training/forecast workflow."),
                "answer": ("This is a predictive question. A grounded ML plan is attached; "
                           "run it through the existing training/forecast workflow.")}


    operation = parse_grouped_aggregation(frame, profile, question) if intent != "ranking" else None
    if operation:
        result = execute_grouped_aggregation(frame, operation)
        result["routing"] = routing
        result["profile"] = _slim(profile)
        result["answer"] = (
            f"Computed {operation['aggregation']} of '{operation['measure']}' grouped by "
            f"'{operation['group_by']}' using {result['coverage']['filtered_rows']} valid rows."
        )
        return result
    if intent == "ranking":
        result = execute_ranking_plan(frame, profile, question, top_n=top_n)
    else:
        result = execute_simple(frame, profile, question, intent)
    result["routing"] = routing
    result["profile"] = _slim(profile)
    result["answer"] = _render_answer(result, question)
    return result


def _target_looks_categorical(frame: pd.DataFrame, target: str | None) -> bool:
    if not target or target not in frame.columns:
        return False
    s = frame[target]
    return (not pd.api.types.is_numeric_dtype(s)) or s.nunique(dropna=True) <= 12


def _slim(profile: dict[str, Any]) -> dict[str, Any]:
    return {"dataset_id": profile.get("dataset_id"), "signature": profile.get("signature"),
            "rows": profile.get("rows"), "cols": profile.get("cols"),
            "numeric_columns": profile.get("numeric_columns"),
            "categorical_columns": profile.get("categorical_columns"),
            "datetime_columns": profile.get("datetime_columns"),
            "duplicate_rows": profile.get("duplicate_rows"),
            "time_coverage": profile.get("time_coverage")}


def _render_answer(result: dict[str, Any], question: str) -> str:
    status = result.get("status")
    if status == "empty":
        return result.get("message", "No matching records.")
    if status in ("unsupported", "error", "clarify"):
        return result.get("message", "This question cannot be answered from the dataset.")
    if result.get("intent") == "ranking":
        parts = []
        filt = result.get("filter")
        if filt:
            parts.append(f"Filtered to {filt['column']} = '{filt['value']}' "
                         f"({(result.get('filter_info') or {}).get('matched_rows', '?')} rows).")
        parts.append(f"Grouped by {result.get('group_by')}.")
        for kind, table in (result.get("tables") or {}).items():
            if "rows" not in table or not table["rows"]:
                continue
            label = 'Lowest' if table.get('direction') == 'ascending' else 'Highest'
            entries = ', '.join(f"{row['group']} = {row.get('exact_value', row['value'])}" + (' (tie)' if row.get('tied') else '')
                                for row in table['rows'])
            parts.append(f"{label} {kind} ({table.get('column')}, {table.get('agg')}): {entries}.")
        winners = result.get("winners") or {}
        if len(winners) >= 2:
            AVC = ", ".join(f"{k}: {v['group']}" for k, v in winners.items())
            parts.append(f"Winners by metric — {AVC}.")
        cov = result.get("coverage") or {}
        if cov:
            parts.append(f"Coverage: {cov.get('filtered_rows')} filtered / {cov.get('total_rows')} total rows.")
        amb = result.get("ambiguous_metrics")
        if amb:
            parts.append(f"Could not map metrics: {', '.join(amb)}.")
        return " ".join(parts) or "Ranking computed."
    if result.get("intent") == "filter":
        return (f"Found {result.get('matched_rows')} matching rows "
                f"out of {result.get('total_rows')} total.")
    return f"{result.get('intent', 'analysis').title()} computed over {(result.get('coverage') or {}).get('filtered_rows', '?')} rows."
