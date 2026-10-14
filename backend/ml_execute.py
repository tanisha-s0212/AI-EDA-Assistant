"""Shared ML execution helpers for the dataset-aware assistant.

Companion to ``nl_analytics.py`` (intent/routing) and ``purchase_predict.py``
(recommendations). This module holds:

- supervised target + leakage-safe feature selection,
- validated ``col = value`` feature extraction for single predictions,
- deterministic anomaly-detection and segmentation executors,
- confidence/uncertainty semantics shared by every model result,
- ``CAPABILITY_REGISTRY``: one record per supported intent (required inputs,
  dataset fields, validation, output schema, confidence strategy, limits).

Only pandas/numpy/sklearn/re are used (all existing dependencies). No
``exec``/``eval``/SQL; every column is validated against the live frame.
Training itself reuses the application's ``train_universal_model`` service in
``main.py``; the helpers here prepare its inputs and interpret its outputs.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CAPABILITY_REGISTRY",
    "CONFIDENCE_SEMANTICS",
    "select_supervised_target",
    "select_features",
    "extract_feature_values",
    "run_anomaly_detection",
    "run_segmentation",
    "summarize_supervised_result",
    "expected_calibration_error",
    "run_fallback_forecast",
]

RANDOM_SEED = 42
MIN_SUPERVISED_ROWS = 10
MIN_CLASSIFICATION_PER_CLASS = 3

CONFIDENCE_SEMANTICS: dict[str, str] = {
    "ranking_score": "Relative 0-1 ranking score among candidates. NOT a probability.",
    "hit_rate": "Measured share of held-out cases whose true outcome ranked in top-k. A reliability rate, not a per-case probability.",
    "predicted_probability": "Class probability estimate returned by the fitted classifier's predict_proba method. Calibration is a separate metric and may be unavailable.",
    "approximate_error_band": "Approximate numeric error band based on validation RMSE. It is not a calibrated prediction interval or a confidence percentage.",
    "forecast_interval": "Model 95% interval (or holdout-residual fallback) per horizon step.",
    "anomaly_score": "IsolationForest anomaly score (lower = more anomalous). A ranking signal, not a probability.",
    "silhouette": "Mean silhouette of the clustering (validation quality, not confidence).",
    "unavailable": "Metric returned as unavailable with a reason instead of an invented value.",
}


def safe_numeric_imputation(series: pd.Series | None, fitted_value: Any) -> tuple[float, str]:
    """Resolve a finite numeric default, preferring the model bundle's imputer.

    A zero fallback is used only when neither the fitted imputer nor the live
    source column contains a finite value; the returned source explains it.
    """
    try:
        fitted = float(fitted_value)
        if np.isfinite(fitted):
            return fitted, "training median"
    except (TypeError, ValueError, OverflowError):
        pass

    if series is not None:
        numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if not numeric.empty:
            median = float(numeric.median())
            if np.isfinite(median):
                return median, "current dataset median (saved imputer unavailable)"
    return 0.0, "0.0 fallback because no finite numeric values are available"


def safe_categorical_imputation(series: pd.Series | None, fitted_value: Any) -> tuple[str, str]:
    """Resolve a categorical default from the fitted mode or live column."""
    try:
        if fitted_value is not None and not pd.isna(fitted_value) and str(fitted_value).strip():
            return str(fitted_value), "training mode"
    except (TypeError, ValueError):
        pass
    if series is not None:
        values = series.dropna().astype(str)
        values = values[values.str.strip().ne("")]
        mode = values.mode()
        if not mode.empty:
            return str(mode.iloc[0]), "current dataset mode (saved imputer unavailable)"
    return "missing", "'missing' fallback because no known category is available"


def decode_class_label(value: Any, label_encoder: Any = None) -> str:
    """Decode an estimator's class value with its persisted label encoder."""
    if label_encoder is not None:
        try:
            number = float(value)
            if not np.isfinite(number) or not number.is_integer():
                raise ValueError("encoded class is not a finite integer")
            return str(label_encoder.inverse_transform([int(number)])[0])
        except (TypeError, ValueError, IndexError):
            pass
    return str(value)


def classification_probability_summary(estimator: Any, X_row: Any,
                                       label_encoder: Any = None) -> dict[str, Any]:
    """Return validated probabilities in estimator class order, or an honest reason.

    This does not infer or normalize malformed probability output.
    """
    if not callable(getattr(estimator, "predict_proba", None)):
        return {"probabilities": None, "confidence": None,
                "probability_class": None,
                "probability_note": "This estimator does not provide class probabilities."}
    try:
        raw = np.asarray(estimator.predict_proba(X_row), dtype=float)
        if raw.ndim != 2 or raw.shape[0] != 1:
            raise ValueError("predict_proba returned an unexpected shape")
        values = raw[0]
        classes = getattr(estimator, "classes_", None)
        if classes is None or len(classes) != len(values):
            raise ValueError("estimator class labels do not match probability columns")
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("probabilities contain non-finite or out-of-range values")
        if not np.isclose(float(values.sum()), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("probabilities do not sum to 1")
        labels = [decode_class_label(class_value, label_encoder) for class_value in classes]
        if len(set(labels)) != len(labels):
            raise ValueError("decoded estimator classes are not unique")
        top_idx = int(np.argmax(values))
        return {"probabilities": {labels[i]: float(values[i]) for i in range(len(values))},
                "confidence": float(values[top_idx]),
                "probability_class": labels[top_idx],
                "probability_note": None}
    except Exception as error:
        return {"probabilities": None, "confidence": None,
                "probability_class": None,
                "probability_note": f"Class probabilities unavailable: {error}."}

CAPABILITY_REGISTRY: dict[str, dict[str, Any]] = {
    "descriptive": {"executor": "deterministic", "requires": [],
                    "validation": "exact recomputation from live frame",
                    "output": "summary statistics", "confidence": "n/a (facts)",
                    "limits": "none"},
    "filter": {"executor": "deterministic", "requires": [],
               "validation": "row-count reconciliation",
               "output": "matching rows", "confidence": "n/a (facts)",
               "limits": "50-row preview cap"},
    "ranking": {"executor": "deterministic", "requires": ["numeric measure", "grouping dimension"],
                "validation": "independent groupby recomputation + tie flags",
                "output": "ranked tables + winners", "confidence": "n/a (facts)",
                "limits": "ambiguous metrics reported, not guessed"},
    "comparison": {"executor": "deterministic", "requires": ["grouping dimension"],
                   "validation": "per-group aggregation", "output": "grouped comparison",
                   "confidence": "n/a (facts)", "limits": "none"},
    "trend": {"executor": "deterministic", "requires": ["datetime column", "numeric measure"],
              "validation": "bucketed sums", "output": "time series points",
              "confidence": "n/a (facts)", "limits": "needs a parseable date column"},
    "statistics": {"executor": "deterministic", "requires": ["numeric columns"],
                   "validation": "pandas describe + correlation",
                   "output": "summary + correlations", "confidence": "n/a (facts)",
                   "limits": "none"},
    "regression": {"executor": "train_universal_model (existing service)",
                   "requires": ["numeric target (>=10 rows)", ">=1 usable feature"],
                   "validation": "CV or holdout RMSE/R2 + approximate error band",
                   "output": "metrics + model_id + optional single prediction",
                   "confidence": "prediction_interval; no universal percentage",
                   "limits": "datetime/id/leakage features excluded; small data degrades to holdout"},
    "classification": {"executor": "train_universal_model (existing service)",
                       "requires": ["categorical target (>=2 classes, >=3 rows/class)", ">=1 usable feature"],
                       "validation": "CV or holdout accuracy/F1 + ECE calibration check",
                       "output": "metrics + class probabilities (when supported) + model_id",
                       "confidence": "predicted_probability with ECE; accuracy reported separately",
                       "limits": "probabilities only from predict_proba estimators"},
    "forecast": {"executor": "TS pipeline (existing service: 3-model walk-forward)",
                 "requires": ["datetime column", "numeric target", ">=24 periods"],
                 "validation": "walk-forward SMAPE/MAE + seasonal-naive baseline",
                 "output": "future points + 95% intervals + model comparison",
                 "confidence": "forecast_interval per step; SMAPE-graded reliability label",
                 "limits": "short series refused; intervals widen honestly with horizon"},
    "anomaly": {"executor": "IsolationForest (seeded) + IQR fallback",
                "requires": [">=1 numeric column", ">=10 rows"],
                "validation": "contamination cap + univariate IQR cross-check",
                "output": "ranked anomalous rows + scores",
                "confidence": "anomaly_score ranking signal; no probabilities",
                "limits": "scores are relative to this dataset snapshot"},
    "segmentation": {"executor": "KMeans over scaled numerics (seeded), k via silhouette",
                     "requires": [">=2 numeric columns", ">=10 rows"],
                     "validation": "silhouette over k=2..5",
                     "output": "cluster labels + sizes + centers + silhouette",
                     "confidence": "silhouette quality; no per-row confidence",
                     "limits": "k capped at 5; high-cardinality categoricals excluded"},
    "next_purchase": {"executor": "purchase_predict (temporal holdout)",
                      "requires": ["customer id", "product field", "timestamps", "transaction grain"],
                      "validation": "last-purchase holdout HitRate@K/MRR",
                      "output": "ranked products + timing + evidence",
                      "confidence": "ranking_score + hit_rate reliability; probabilities unavailable",
                      "limits": "unvalidated output labeled recommendation; segments refused"},
    "recommend": {"executor": "purchase_predict (temporal holdout)",
                  "requires": ["customer id or cold-start fallback data"],
                  "validation": "same as next_purchase when history exists",
                  "output": "ranked products + fallback provenance",
                  "confidence": "same as next_purchase",
                  "limits": "cold start uses labeled popularity/category fallback"},
    "buyers": {"executor": "purchase_predict buyer ranking",
               "requires": ["product reference", "customer + product fields"],
               "validation": "recency x frequency recomputation",
               "output": "ranked buyer list", "confidence": "ranking_score",
               "limits": "needs a resolvable product"},
    "churn": {"executor": "purchase_predict interval rule",
              "requires": ["repeat histories (>=2 orders)"],
              "validation": "median + 2*std silence rule vs dataset max date",
              "output": "at-risk list with evidence", "confidence": "rule-based flag, not a probability",
              "limits": "single-purchase customers unevaluable"},
    "confidence": {"executor": "re-execution of underlying prediction + honesty block",
                   "requires": ["resolvable underlying question"],
                   "validation": "same as underlying capability",
                   "output": "typed confidence breakdown",
                   "confidence": "typed; unavailable typed as unavailable",
                   "limits": "never a universal percentage"},
    "explain": {"executor": "evidence renderer over executed prediction",
                "requires": ["executed or executable underlying prediction"],
                "validation": "evidence traced to live rows",
                "output": "per-item reasons + method + validation",
                "confidence": "same as underlying capability",
                "limits": "no post-hoc stories beyond measured evidence"},
}


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _mentions(question: str, column: str) -> bool:
    ql = (question or "").lower()
    name = str(column).lower().replace("_", " ")
    return bool(name) and name in ql


def select_supervised_target(frame: pd.DataFrame, profile: dict[str, Any],
                             question: str, intent: str) -> dict[str, Any]:
    """Pick a validated (target, problem_type); never invent one.

    Priority: explicitly named column in the question > intent-appropriate auto
    pick. Classification intent forces a categorical target; regression forces
    numeric. Returns {"target", "problem_type", "reason"} or {"issue"}.
    """
    cols = {c["name"]: c for c in profile.get("columns", [])}
    # 1. Explicit mention. Prefer the verb's object ("predict X", "classify Y into",
    # "forecast Z") over other columns merely named as features.
    verb_obj = re.search(
        r"\b(predict|forecast|classify|estimate|project)\s+(?:the\s+)?([a-z_][a-z0-9_ ]{0,40}?)"
        r"\s+(from|using|based on|with|given|where|for\b|$)",
        question or "", re.I)
    ordered: list[str] = []
    verb_target: str | None = None
    if verb_obj:
        hint = _norm(verb_obj.group(2))
        for name in frame.columns:
            if _norm(str(name)) == hint and str(name) not in ordered:
                ordered.append(str(name))
                verb_target = str(name)
    for name in frame.columns:
        if str(name) not in ordered and _mentions(question, name):
            ordered.append(str(name))
    for name in ordered:
        meta = cols.get(str(name), {})
        role = meta.get("role", "")
        if intent == "classification":
            if role in ("categorical", "boolean"):
                if meta.get("unique_count", 0) >= 2:
                    return {"target": str(name), "problem_type": "classification",
                            "reason": f"Column '{name}' named in the question; categorical target."}
                if verb_target == str(name):
                    return {"issue": (f"Column '{name}' was requested as the target but has only "
                                      f"{meta.get('unique_count', 0)} class(es); classification needs >= 2.")}
                continue
            if role == "numeric" and meta.get("unique_count", 99) <= 12:
                return {"target": str(name), "problem_type": "classification",
                        "reason": f"Column '{name}' named in the question; low-cardinality target."}
            if verb_target == str(name):
                return {"issue": (f"Column '{name}' was requested as the classification target "
                                  f"but is not categorical; no valid target found.")}
            continue  # named but unsuitable as a classification target
        # regression / numeric intent: the named column must be usable numeric
        # (enough valid rows + positive variance), else keep looking.
        try:
            numeric = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        except Exception:
            if verb_target == str(name):
                return {"issue": f"Column '{name}' was requested as the target but is not numeric."}
            continue
        n_valid = int(numeric.notna().sum())
        var = float(numeric.var(skipna=True)) if n_valid else 0.0
        if n_valid >= MIN_SUPERVISED_ROWS and np.isfinite(var) and var > 0:
            return {"target": str(name), "problem_type": "regression",
                    "reason": f"Column '{name}' named in the question; numeric target."}
        if verb_target == str(name):
            return {"issue": (f"Column '{name}' was requested as the target but has {n_valid} "
                              f"valid rows and variance {round(var, 4)}; regression needs "
                              f"{MIN_SUPERVISED_ROWS}+ rows with positive variance.")}
    # 2. Intent-appropriate auto pick.
    if intent == "classification":
        cands = [c for c in profile.get("columns", [])
                 if c["role"] in ("categorical", "boolean")
                 and not c.get("is_empty") and 2 <= c.get("unique_count", 0) <= 20
                 and "id" not in str(c["name"]).lower()]
        if cands:
            cands.sort(key=lambda c: (c["unique_count"], c["name"]))
            return {"target": cands[0]["name"], "problem_type": "classification",
                    "reason": f"Auto target '{cands[0]['name']}': low-cardinality categorical "
                              f"({cands[0]['unique_count']} classes)."}
        return {"issue": ("Classification needs a categorical target with 2-20 classes; "
                          "none found. Available categorical: "
                          + ", ".join(profile.get("categorical_columns", [])) + ".")}
    # regression / forecast-style numeric prediction
    numeric = [c for c in profile.get("columns", [])
               if c["role"] == "numeric" and not c.get("is_empty")]
    if not numeric:
        return {"issue": "Regression needs a numeric target; no usable numeric column found."}
    # Prefer revenue-like, then highest variance (mirrors nl_analytics scoring).
    try:
        from nl_analytics import resolve_metric_column  # local import: avoids hard cycle at module load
    except Exception:
        resolve_metric_column = None  # type: ignore
    target = None
    if resolve_metric_column is not None:
        target = resolve_metric_column(frame, profile, "revenue")
    if target is None:
        scored = []
        for c in numeric:
            try:
                numeric_values = pd.to_numeric(frame[c["name"]], errors="coerce").replace([np.inf, -np.inf], np.nan)
                var = float(numeric_values.var(skipna=True))
            except Exception:
                var = 0.0
            if np.isfinite(var) and var > 0 and not re.search(r"(^id$|_id$|year|month|day|quarter)", c["name"].lower()):
                scored.append((var, c["name"]))
        scored.sort(key=lambda t: (-t[0], t[1]))
        target = scored[0][1] if scored else None
    if target is None:
        return {"issue": "No numeric target with positive variance found."}
    n_valid = int(pd.to_numeric(frame[target], errors="coerce").notna().sum())
    if n_valid < MIN_SUPERVISED_ROWS:
        return {"issue": (f"Target '{target}' has only {n_valid} valid rows "
                          f"(need {MIN_SUPERVISED_ROWS} for supervised training).")}
    return {"target": target, "problem_type": "regression",
            "reason": f"Target '{target}' ({n_valid} valid rows)."}


def select_features(frame: pd.DataFrame, profile: dict[str, Any], target: str,
                    max_features: int = 12) -> dict[str, Any]:
    """Leakage-safe features: drop target, identifiers, datetimes, constants,
    empties, and near-duplicate names of the target. Ordered stable."""
    excluded: dict[str, str] = {}
    kept: list[str] = []
    tnorm = _norm(target).replace(" ", "")
    for c in profile.get("columns", []):
        name = c["name"]
        if name == target:
            continue
        if c.get("is_empty") or c.get("is_constant"):
            excluded[name] = "empty/constant"
            continue
        if c.get("role") in ("identifier", "datetime"):
            excluded[name] = f"excluded role {c.get('role')} (leakage/engineering guard)"
            continue
        if re.search(r"(^id$|_id$|uuid|guid)", str(name), re.I):
            excluded[name] = "identifier-like name"
            continue
        if re.search(r"(^|_)(post|after|future)_(outcome|result|target)(_|$)|(^|_)(outcome|result|target)_(post|after|future)(_|$)", str(name), re.I):
            excluded[name] = "post-outcome/future information (leakage guard)"
            continue
        cnorm = _norm(name).replace(" ", "")
        if cnorm and (cnorm == tnorm or tnorm in cnorm or cnorm in tnorm):
            excluded[name] = "name-overlaps-target (leakage guard)"
            continue
        kept.append(name)
    # Prefer categoricals + numerics actually present; cap for speed, stable order.
    kept = kept[:max_features]
    if not kept:
        return {"features": [], "excluded": excluded,
                "issue": "No usable features remain after leakage guards."}
    return {"features": kept, "excluded": excluded}


def extract_feature_values(question: str, frame: pd.DataFrame,
                           features: list[str]) -> dict[str, Any]:
    """Parse `col = value` / `col: value` / `col is value` pairs from the question.

    The column pattern is built from the validated feature names themselves
    (longest first), so multi-word columns match exactly and unknown columns
    can never be injected. Numeric values are coerced; others kept as strings.
    Returns {values, missing}."""
    values: dict[str, Any] = {}
    if not features:
        return {"values": values, "missing": []}
    ordered = sorted(features, key=len, reverse=True)
    col_pat = "|".join(re.escape(f) for f in ordered)
    pair_re = re.compile(
        r"(?P<col>" + col_pat + r")\s*(=|:|\bis\b)\s*['\"]?(?P<val>[A-Za-z0-9 _\-/&.,]+?)['\"]?"
        r"(?=\s*(,|and\b|with\b|where\b|;|$))", re.I)
    for match in pair_re.finditer(question or ""):
        text = match.group("col")
        feat = next((f for f in features if f.lower() == text.lower()
                     or _norm(f) == _norm(text)), None)
        if feat is None or feat in values:
            continue
        raw = match.group("val").strip().strip("'\"")
        series = frame[feat]
        if pd.api.types.is_numeric_dtype(series):
            try:
                values[feat] = float(raw.replace(",", ""))
            except ValueError:
                continue
        else:
            values[feat] = raw
    missing = [f for f in features if f not in values]
    return {"values": values, "missing": missing}


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray,
                               n_bins: int = 10) -> float | None:
    """Binary ECE on holdout probabilities; None when not computable."""
    try:
        y = np.asarray(y_true).astype(int).ravel()
        p = np.asarray(proba, dtype=float).ravel()
        if (len(y) != len(p) or len(np.unique(y)) != 2 or len(y) < 10
                or set(np.unique(y).tolist()) != {0, 1} or n_bins < 1
                or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)):
            return None
        edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lower_ok = p >= edges[i] if i == 0 else p > edges[i]
            mask = lower_ok & (p <= edges[i + 1])
            if mask.sum() == 0:
                continue
            ece += (mask.sum() / len(y)) * abs(y[mask].mean() - p[mask].mean())
        return round(float(ece), 4)
    except Exception:
        return None


def run_anomaly_detection(frame: pd.DataFrame, profile: dict[str, Any],
                          top_n: int = 10) -> dict[str, Any]:
    """IsolationForest over scaled numerics (seeded) with IQR cross-check."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    numeric = [c for c in profile.get("numeric_columns", []) if c in frame.columns]
    if not numeric:
        return {"status": "unsupported", "intent": "anomaly",
                "message": "Anomaly detection needs at least one numeric column."}
    finite = frame[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    varying = [c for c in numeric if finite[c].nunique(dropna=True) > 1]
    if not varying:
        return {"status": "unsupported", "intent": "anomaly",
                "message": "Anomaly detection needs a numeric column with more than one finite value."}
    use_cols = varying[:12]
    clean = finite[use_cols].dropna()
    if len(clean) < MIN_SUPERVISED_ROWS:
        return {"status": "unsupported", "intent": "anomaly",
                "message": (f"Only {len(clean)} complete numeric rows "
                            f"(need {MIN_SUPERVISED_ROWS} for anomaly detection).")}
    X = StandardScaler().fit_transform(clean[use_cols].to_numpy(dtype=float))
    contamination = min(0.1, max(1.0 / len(clean), 10.0 / len(clean)))
    forest = IsolationForest(contamination=contamination, random_state=RANDOM_SEED)
    labels = forest.fit_predict(X)
    scores = (-forest.score_samples(X)).tolist()  # higher = more anomalous
    order = np.argsort([-s for s in scores])  # ascending original score
    # IQR cross-check on the highest-variance column.
    variances = {c: float(clean[c].var() or 0.0) for c in use_cols}
    top_col = max(variances, key=variances.get)
    q1, q3 = float(clean[top_col].quantile(0.25)), float(clean[top_col].quantile(0.75))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    rows = []
    for rank, idx in enumerate(order[:top_n], start=1):
        row_idx = clean.index[idx]
        record = {c: (None if pd.isna(clean.at[row_idx, c]) else round(float(clean.at[row_idx, c]), 4))
                  for c in use_cols}
        record["_row"] = int(row_idx)
        out = bool(clean.at[row_idx, top_col] < lo or clean.at[row_idx, top_col] > hi)
        rows.append({"rank": rank, "score": round(float(scores[idx]), 4),
                     "iqr_confirmed": out, "record": record})
    return {"status": "success", "intent": "anomaly",
            "method": "IsolationForest", "columns": use_cols,
            "rows_flagged": int((labels == -1).sum()), "rows_scored": int(len(clean)),
            "contamination": round(float(contamination), 4),
            "anomalies": rows,
            "answer": (f"Flagged {int((labels == -1).sum())} of {len(clean)} rows as anomalous "
                       f"(IsolationForest over {', '.join(use_cols)}). Top anomaly score "
                       f"{rows[0]['score'] if rows else 'n/a'}; scores rank anomalousness, not probabilities.")}


def run_segmentation(frame: pd.DataFrame, profile: dict[str, Any],
                     max_k: int = 5) -> dict[str, Any]:
    """KMeans (seeded) over scaled numerics; k chosen by silhouette in 2..max_k."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    numeric = [c for c in profile.get("numeric_columns", []) if c in frame.columns]
    if len(numeric) < 2:
        return {"status": "unsupported", "intent": "segmentation",
                "message": "Segmentation needs at least 2 numeric columns."}
    finite = frame[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    varying = [c for c in numeric if finite[c].nunique(dropna=True) > 1]
    if len(varying) < 2:
        return {"status": "unsupported", "intent": "segmentation",
                "message": "Segmentation needs at least 2 numeric columns with varying finite values."}
    use_cols = varying[:12]
    clean = finite[use_cols].dropna()
    if len(clean) < MIN_SUPERVISED_ROWS:
        return {"status": "unsupported", "intent": "segmentation",
                "message": (f"Only {len(clean)} complete numeric rows "
                            f"(need {MIN_SUPERVISED_ROWS} for segmentation).")}
    X = StandardScaler().fit_transform(clean[use_cols].to_numpy(dtype=float))
    hi_k = max(2, min(max_k, len(clean) // 5))
    best: dict[str, Any] = {"k": 2, "score": -1.0, "labels": None}
    for k in range(2, hi_k + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(X)
            if len(set(labels.tolist())) < 2:
                continue
            score = float(silhouette_score(X, labels))
            if score > best["score"]:
                best = {"k": k, "score": score, "labels": labels}
        except Exception:
            continue
    if best["labels"] is None:
        return {"status": "error", "intent": "segmentation",
                "message": "Clustering failed to separate the data."}
    labels = best["labels"]
    clusters = []
    for c in range(best["k"]):
        mask = labels == c
        center = {col: round(float(clean[col][mask].mean()), 4) for col in use_cols}
        clusters.append({"cluster": int(c), "size": int(mask.sum()),
                         "share": round(float(mask.mean()), 4), "center": center})
    clusters.sort(key=lambda d: -d["size"])
    return {"status": "success", "intent": "segmentation",
            "method": "KMeans", "k": best["k"],
            "silhouette": round(float(best["score"]), 4),
            "columns": use_cols, "rows_clustered": int(len(clean)),
            "clusters": clusters,
            "answer": (f"Split {len(clean)} rows into {best['k']} segments "
                       f"(silhouette {round(float(best['score']), 3)}). Largest segment: "
                       f"{clusters[0]['size']} rows ({clusters[0]['share'] * 100:.1f}%).")}


def summarize_supervised_result(train_result: dict[str, Any], target: str,
                                features: list[str], problem_type: str,
                                single: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize a train_universal_model dict into an assistant result with
    explicit confidence semantics (no universal percentage)."""
    validation_scores = train_result.get("scores", {}) or {}
    fitted_metrics = train_result.get("metrics", {}) or {}
    if problem_type == "regression":
        metrics = {"RMSE": (validation_scores.get("rmse") or {}).get("mean"),
                   "R2": (validation_scores.get("r2") or {}).get("mean")}
    else:
        metrics = {"Accuracy": (validation_scores.get("accuracy") or {}).get("mean"),
                   "F1 Score": (validation_scores.get("f1_weighted") or {}).get("mean")}
    metrics = {key: value for key, value in metrics.items() if value is not None}
    validation_missing = not metrics
    if validation_missing:
        metrics = fitted_metrics
    opt = train_result.get("optimization", {}) or {}
    eval_rows = int(opt.get("cv_rows_evaluated") or 0)
    folds = int(opt.get("cv_folds_used") or 0)
    method_note = ("cross-validation" if folds and folds > 1
                   else "holdout split" if eval_rows else "training fit")
    confidence: dict[str, Any] = {}
    if problem_type == "regression":
        rmse = metrics.get("RMSE")
        confidence["validation_metrics"] = {
            "type": "validation_metrics", "R2": metrics.get("R2"),
            "RMSE": rmse, "MAE": metrics.get("MAE"),
            "evaluated_rows": eval_rows, "method": method_note,
            "meaning": ("Validation error magnitudes; lower is better. Not a confidence percentage."
                        if not validation_missing else
                        "Validation metrics are unavailable; displayed metrics are training-fit values.")}
        if single is not None and rmse:
            margin = round(1.96 * float(rmse), 4)
            single["error_band"] = [round(single["prediction"] - margin, 4),
                                     round(single["prediction"] + margin, 4)]
            single["error_band_note"] = CONFIDENCE_SEMANTICS["approximate_error_band"]
        confidence["approximate_error_band"] = single.get("error_band") if single else None
    else:
        confidence["validation_metrics"] = {
            "type": "validation_metrics", "Accuracy": metrics.get("Accuracy"),
            "F1": metrics.get("F1 Score"), "Precision": metrics.get("Precision"),
            "Recall": metrics.get("Recall"),
            "evaluated_rows": eval_rows, "method": method_note,
            "meaning": (("Classification quality on held-out data. Accuracy is overall "
                         "correctness, distinct from any single prediction's probability.")
                        if not validation_missing else
                        "Validation metrics are unavailable; displayed metrics are training-fit values.")}
        if single is not None and single.get("status") != "unavailable":
            confidence["predicted_probability"] = {
                "type": "predicted_probability" if single.get("probabilities") else "unavailable",
                "value": single.get("confidence"),
                "class": single.get("probability_class") or single.get("top_class"),
                "ece": single.get("ece"),
                "meaning": CONFIDENCE_SEMANTICS["predicted_probability"]}
    parts = [f"Trained {train_result.get('model', 'model')} to predict '{target}' "
             f"from {len(features)} feature(s) ({method_note}, {eval_rows} rows evaluated)."]
    show = ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:4])
    if show:
        metric_label = "Validation" if not validation_missing else "Training fit (validation unavailable)"
        parts.append(f"{metric_label}: {show}.")
    if single is not None and single.get("status") == "unavailable":
        parts.append(single.get("message", "Single-row prediction is unavailable."))
    elif single is not None:
        if problem_type == "regression":
            parts.append(f"Prediction for the supplied inputs: {single['prediction']}"
                         + (f" (approximate error band {single.get('error_band')}; not a calibrated 95% prediction interval)."
                            if single.get("error_band") else "."))
        else:
            parts.append(f"Predicted class for the supplied inputs: {single.get('top_class')}"
                         + (f" (probability {single.get('confidence')})."
                            if single.get("confidence") is not None else "."))
    else:
        parts.append("No complete input row was supplied in the question, so no single-row "
                     "prediction was made; use the Prediction tab with model "
                     f"{train_result.get('model_id')} to score new rows.")
    warn = train_result.get("warnings", []) or []
    if validation_missing:
        warn = [*warn, "Validation scores were unavailable; displayed metrics are training-fit values, not validation results."]
    if warn:
        parts.append("Notes: " + " ".join(w for w in warn[:2]))
    return {"status": "success", "intent": problem_type,
            "model": train_result.get("model"), "model_id": train_result.get("model_id"),
            "target": target, "features": features,
            "metrics": metrics, "training_metrics": fitted_metrics if not validation_missing else None,
            "evaluation": {"rows": eval_rows, "method": method_note,
                                               "cv_folds": folds},
            "feature_importance": (train_result.get("feature_importance", []) or [])[:10],
            "single_prediction": single, "confidence": confidence,
            "warnings": warn, "answer": " ".join(parts)}


def run_fallback_forecast(frame: pd.DataFrame, date_col: str, target_col: str,
                          frequency: str = "monthly", horizon: int = 3) -> dict[str, Any]:
    """sklearn lag-regression fallback when the statsmodels/prophet TS stack is
    unavailable. Chronological holdout, seasonal-naive baseline, recursive
    multi-step forecast with residual intervals. Deterministic (seeded)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge

    work = pd.DataFrame({"date": pd.to_datetime(frame[date_col], errors="coerce"),
                         "y": pd.to_numeric(frame[target_col], errors="coerce")}).dropna()
    if work.empty:
        return {"status": "unsupported", "intent": "forecast",
                "message": "No valid date/target rows for forecasting."}
    work = work.sort_values("date")
    freq_code = {"daily": "D", "weekly": "W-MON"}.get(frequency, "MS")
    period = work["date"].dt.to_period("D" if freq_code == "D" else "W" if freq_code.startswith("W") else "M")
    grouped = work.assign(_p=period).groupby("_p")["y"].sum()
    try:
        idx = grouped.index.to_timestamp()
    except Exception:
        idx = pd.DatetimeIndex(sorted(work["date"].unique()))
        grouped = work.groupby(work["date"])["y"].sum()
    y = grouped.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(idx)
    season = {"daily": 7, "weekly": 52}.get(frequency, 12)
    if len(y) < 12:
        return {"status": "unsupported", "intent": "forecast",
                "message": (f"Only {len(y)} {frequency} periods observed (need 12 for the "
                            f"fallback forecaster; 24+ for the full 3-model pipeline).")}
    n_lags = max(2, min(6, len(y) // 3))

    def features_for(history: list[float], pos: int, stamp: pd.Timestamp) -> list[float]:
        lags = [(history[pos - l] if pos - l >= 0 else history[0]) for l in range(1, n_lags + 1)]
        window = history[max(0, pos - season):pos] or [history[0]]
        return lags + [float(np.mean(window)), float(pos),
                       float(np.sin(2 * np.pi * stamp.month / 12)),
                       float(np.cos(2 * np.pi * stamp.month / 12))]

    test_n = max(2, min(6, len(y) // 4))
    split = len(y) - test_n
    hist_all = y.tolist()
    Xtr = [features_for(hist_all, i, dates[i]) for i in range(n_lags, split)]
    ytr = y[n_lags:split]
    Xte = [features_for(hist_all, i, dates[i]) for i in range(split, len(y))]
    yte = y[split:]
    cands = {"Ridge": Ridge(alpha=1.0),
             "HistGradientBoosting": HistGradientBoostingRegressor(
                 max_iter=100, learning_rate=0.05, max_depth=3, random_state=RANDOM_SEED)}
    best_name, best_est, best_mae = "", None, float("inf")
    for name, est in cands.items():
        try:
            est.fit(np.asarray(Xtr, dtype=float), np.asarray(ytr, dtype=float))
            mae = float(np.mean(np.abs(np.asarray(yte, dtype=float)
                                                  - est.predict(np.asarray(Xte, dtype=float)))))
            if mae < best_mae:
                best_name, best_est, best_mae = name, est, mae
        except Exception:
            continue
    if best_est is None:
        return {"status": "failed", "intent": "forecast",
                "message": "Fallback forecast training failed on this series."}
    preds = np.asarray(best_est.predict(np.asarray(Xte, dtype=float)))
    mae = float(np.mean(np.abs(yte - preds)))
    rmse = float(np.sqrt(np.mean((yte - preds) ** 2)))
    denom = np.abs(yte) + np.abs(preds)
    smape = round(float(np.mean(2 * np.abs(yte - preds) / np.where(denom == 0, 1e-8, denom)) * 100), 2)
    naive = np.array([y[i - season] if i - season >= 0 else y[0] for i in range(split, len(y))])
    nden = np.abs(yte) + np.abs(naive)
    baseline_smape = round(float(np.mean(2 * np.abs(yte - naive) / np.where(nden == 0, 1e-8, nden)) * 100), 2)
    resid_std = float(np.std(yte - preds)) if len(yte) > 1 else abs(float(yte[0])) * 0.1 or 1.0
    # Recursive future forecast.
    history = y.tolist()
    future_dates = pd.date_range(dates[-1], periods=horizon + 1, freq=freq_code)[1:]
    points = []
    for step, stamp in enumerate(future_dates, start=1):
        feat = features_for(history, len(history), stamp)
        point = float(best_est.predict(np.asarray([feat], dtype=float))[0])
        margin = round(1.96 * resid_std * float(np.sqrt(step)), 2)
        label = stamp.strftime("%Y-%m-%d" if frequency != "monthly" else "%Y-%m")
        points.append({"period": label, "forecast": round(point, 2), "predicted": round(point, 2),
                       "lower": round(point - margin, 2), "upper": round(point + margin, 2)})
        history.append(point)
    reliability = "high" if smape < 15 else "medium" if smape < 30 else "low"
    return {"status": "success", "intent": "forecast",
            "model": f"LagRegression({best_name})", "target": target_col,
            "date_column": date_col, "frequency": frequency,
            "metrics": {"SMAPE": smape, "MAE": round(mae, 2), "RMSE": round(rmse, 2)},
            "model_comparison": [{"model": f"LagRegression({best_name})", "status": "completed",
                                  "smape": smape, "mae": round(mae, 2)},
                                 {"model": "SeasonalNaive", "status": "completed",
                                  "smape": baseline_smape, "mae": None}],
            "baseline_smape": baseline_smape, "future_forecast": points,
            "stationarity": {"status": "not_assessed",
                             "note": "Fallback path: stationarity test unavailable without statsmodels."},
            "selection_reason": (f"LagRegression({best_name}) chosen on chronological holdout "
                                 f"MAE {round(best_mae, 2)} over {len(yte)} held-out periods."),
            "reliability": reliability,
            "fallback": True,
            "fallback_note": ("statsmodels/prophet stack unavailable: used sklearn lag-regression "
                              "fallback with chronological validation."),
            "confidence": {"forecast_interval": "95% residual intervals widening with horizon steps.",
                           "reliability": reliability, "evaluated_periods": len(yte)}}
