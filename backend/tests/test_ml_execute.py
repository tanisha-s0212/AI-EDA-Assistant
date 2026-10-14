"""Tests for executable ML intents: regression, classification, forecasting,
anomaly detection, segmentation, plus registry, leakage, and cold-start rules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml_execute import (  # noqa: E402
    CAPABILITY_REGISTRY,
    classification_probability_summary,
    expected_calibration_error,
    extract_feature_values,
    run_anomaly_detection,
    run_segmentation,
    safe_categorical_imputation,
    safe_numeric_imputation,
    select_features,
    select_supervised_target,
)
from nl_analytics import answer_question, classify_intent, profile_dataset  # noqa: E402
from purchase_predict import build_purchase_histories, discover_purchase_schema, predict_next_for_customer  # noqa: E402

SAMPLES = BACKEND_DIR.parent / "samples" / "sales"


@pytest.fixture()
def synth() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 150
    df = pd.DataFrame({
        "size": rng.uniform(10, 100, n).round(1),
        "rooms": rng.randint(1, 6, n),
        "age": rng.randint(0, 30, n),
    })
    df["price"] = (3 * df["size"] + 5000 * df["rooms"] - 200 * df["age"]
                   + rng.normal(0, 1500, n)).round(0)
    df["band"] = pd.cut(df["price"], 3, labels=["low", "mid", "high"]).astype(str)
    return df


@pytest.fixture()
def client():
    import main
    return TestClient(main.app)


def _rows(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


# --- Registry -------------------------------------------------------------------

def test_registry_covers_every_classified_intent(synth):
    profile = profile_dataset(synth, "t")
    intents = {classify_intent(q, profile)["intent"] for q in [
        "What is total price?", "Show rows for rooms 3", "Highest price by rooms",
        "Compare price across bands", "Average price?", "Predict price from size",
        "Classify band from size", "Find anomalies in price", "Segment rows by size and price",
        "What will price be next month?", "Predict the next purchase of customer C1",
    ]}
    missing = {i for i in intents if i not in CAPABILITY_REGISTRY}
    assert not missing, f"intents without registry entries: {missing}"
    for intent, record in CAPABILITY_REGISTRY.items():
        for key in ("executor", "requires", "validation", "output", "confidence", "limits"):
            assert key in record, f"{intent} missing registry key {key}"


def test_supported_ml_intents_execute_not_plan(client, synth):
    for question in ["Predict price from size, rooms and age",
                     "Classify band from size, rooms and age"]:
        body = client.post("/api/assistant/ask",
                           json={"question": question, "data": _rows(synth)}).json()
        assert body["status"] == "success", body.get("message")
        assert body["status"] != "ml_required"
        assert body["ml_plan"]["status"] == "executed"


# --- Regression -------------------------------------------------------------------

def test_regression_end_to_end_metrics(client, synth):
    body = client.post("/api/assistant/ask", json={
        "question": "Predict price from size, rooms and age", "data": _rows(synth)}).json()
    assert body["status"] == "success" and body["intent"] == "regression"
    assert body["target"] == "price"
    assert body["metrics"]["R2"] > 0.9
    assert body["evaluation"]["rows"] >= 20
    assert body["evaluation"]["method"] in ("holdout split", "cross-validation")
    assert body["model_id"]
    assert body["answer"]


def test_regression_single_prediction_with_approximate_error_band(client, synth):
    body = client.post("/api/assistant/ask", json={
        "question": "Predict price where size = 50, rooms = 3, age = 5",
        "data": _rows(synth)}).json()
    assert body["status"] == "success"
    single = body["single_prediction"]
    assert single is not None
    # True value 3*50 + 5000*3 - 200*5 = 14150 (noise sigma 1500).
    assert abs(single["prediction"] - 14150) < 6000
    lo, hi = single["error_band"]
    assert lo <= single["prediction"] <= hi
    assert "not a calibrated" in single["error_band_note"]
    assert "probabilities" not in single or single.get("probabilities") is None


def test_regression_summary_uses_validation_scores_not_training_fit_metrics():
    from ml_execute import summarize_supervised_result

    result = summarize_supervised_result({
        "model": "Ridge", "model_id": "m1",
        "scores": {"rmse": {"mean": 12.5, "std": 1.0}, "r2": {"mean": 0.8, "std": 0.1}},
        "metrics": {"RMSE": 0.01, "R2": 0.999},
        "optimization": {"cv_rows_evaluated": 20, "cv_folds_used": 1},
    }, "price", ["size"], "regression",
       {"inputs": {"size": 5}, "prediction": 20.0})
    assert result["metrics"]["RMSE"] == 12.5
    assert result["training_metrics"]["RMSE"] == 0.01
    assert result["single_prediction"]["error_band"] == pytest.approx([-4.5, 44.5])
    assert "not a calibrated prediction interval" in result["single_prediction"]["error_band_note"]
    assert "Validation: RMSE=12.5" in result["answer"]


def test_temporal_preprocessing_fits_only_on_earlier_rows():
    import main

    frame = pd.DataFrame({
        "event_date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "signal": [0, 1, 2, 3, 4, 5, 6, 7, 100, 200],
        "target": [0, 2, 4, 6, 8, 10, 12, 14, 200, 400],
    })
    request = main.TrainRequest(
        target_column="target", feature_columns=["signal"],
        problem_type="regression", model_type="auto",
        time_order_column="event_date", test_size=0.2,
    )
    _, _, preprocessing, _, _, _ = main.prepare_universal_training_data(frame, request)
    # The final two holdout rows (100, 200) must not affect the scaler/imputer.
    assert preprocessing["scaler"].mean_.tolist() == pytest.approx([3.5])
    assert preprocessing["imputers"]["numeric"]["signal"] == 3.5


def test_prediction_api_uses_validation_error_band_not_training_rmse(client, monkeypatch):
    import main

    class FixedRegressor:
        def predict(self, _features):
            return np.asarray([100.0])

    bundle = {
        "model": FixedRegressor(), "problem_type": "regression",
        "feature_columns": ["signal"], "rmse": 0.01, "validation_rmse": 10.0,
        "preprocessing": {
            "raw_feature_columns": ["signal"], "numeric_features": ["signal"],
            "categorical_features": [], "imputers": {"numeric": {"signal": 0.0}},
            "encoded_feature_names": ["signal"], "scaler": None,
        },
    }
    monkeypatch.setattr(main, "load_model_bundle", lambda _model_id: bundle)
    response = client.post("/api/predict", json={"model_id": "m", "features": {"signal": 1}})
    assert response.status_code == 200
    body = response.json()
    assert body["error_band"]["margin"] == pytest.approx(19.6)
    assert body["confidence_interval"] == body["error_band"]  # legacy response key
    assert "not a calibrated 95%" in body["error_band_note"]


def test_unhandled_api_errors_do_not_return_exception_text():
    import asyncio
    import main
    from starlette.requests import Request

    request = Request({"type": "http", "method": "GET", "path": "/test",
                      "headers": [], "query_string": b""})
    response = asyncio.run(main.structured_unhandled_exception_handler(
        request, RuntimeError("secret path C:/private/model.joblib")))
    assert response.status_code == 500
    assert b"secret path" not in response.body
    assert b"unexpected server error" in response.body


def test_regression_wording_does_not_trigger_on_ranking(client, synth):
    body = client.post("/api/assistant/ask", json={
        "question": "Which rooms have the highest average price?", "data": _rows(synth)}).json()
    assert body["status"] == "success"
    assert body["routing"]["needs_ml"] is False
    assert "model_id" not in body or not body.get("model_id")


# --- Classification -------------------------------------------------------------------

def test_classification_probabilities_and_calibration(client, synth):
    body = client.post("/api/assistant/ask", json={
        "question": "Classify band where size = 60, rooms = 4, age = 10",
        "data": _rows(synth)}).json()
    assert body["status"] == "success" and body["intent"] == "classification"
    single = body["single_prediction"]
    assert single is not None
    proba = single["probabilities"]
    assert proba is not None
    assert abs(sum(proba.values()) - 1.0) < 1e-6
    assert single["top_class"] in {"low", "mid", "high"}
    # Calibration assessed or honestly missing — never silently skipped.
    assert single.get("ece") is not None or single.get("ece_note")


def test_classification_without_inputs_trains_only(client, synth):
    body = client.post("/api/assistant/ask", json={
        "question": "Classify band from size, rooms and age", "data": _rows(synth)}).json()
    assert body["status"] == "success"
    assert body["single_prediction"] is None
    assert body["metrics"]["Accuracy"] > 0.5
    assert "Prediction tab" in body["answer"]


class _ProbabilityEstimator:
    classes_ = np.asarray([1, 0])  # deliberately differs from encoder order

    def __init__(self):
        self.proba_calls = 0

    def predict(self, X):
        return np.asarray([1])

    def predict_proba(self, X):
        self.proba_calls += 1
        return np.asarray([[0.8, 0.2]])


@pytest.mark.parametrize("question,should_default", [
    ("Classify label where amount = 7, zone = north", False),
    ("Classify label where amount = 7", True),
])
def test_classification_probabilities_are_available_with_explicit_or_defaulted_inputs(
        monkeypatch, question, should_default):
    import main
    from sklearn.preprocessing import LabelEncoder

    df = pd.DataFrame({"amount": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                       "zone": ["north", "south"] * 6,
                       "label": ["alpha", "beta"] * 6})
    estimator = _ProbabilityEstimator()
    encoder = LabelEncoder().fit(["alpha", "beta"])
    bundle = {
        "model": estimator,
        "label_encoder": encoder,
        "feature_columns": ["amount", "zone"],
        "preprocessing": {
            "raw_feature_columns": ["amount", "zone"],
            "numeric_features": ["amount"],
            "categorical_features": ["zone"],
            "imputers": {"numeric": {"amount": 6.5}, "categorical": {"zone": "north"}},
            "encoded_feature_names": ["amount", "zone_north", "zone_south"],
            "scaler": None,
        },
    }
    train_result = {
        "status": "success", "model": "FakeClassifier", "model_id": "fake",
        "scores": {"accuracy": {"mean": 0.75, "std": 0.0},
                   "f1_weighted": {"mean": 0.75, "std": 0.0}},
        "metrics": {"Accuracy": 1.0},
        "optimization": {"cv_rows_evaluated": 3, "cv_folds_used": 1},
        "warnings": [], "feature_importance": [],
    }
    monkeypatch.setattr(main, "train_universal_model",
                        lambda request, http_request, **kwargs: train_result)
    monkeypatch.setattr(main, "load_model_bundle", lambda _model_id: bundle)
    monkeypatch.setattr(main, "transform_features_for_bundle", lambda _row, _bundle: np.asarray([[0.0]]))
    result = main._assistant_run_supervised(
        {"intent": "classification", "ml_plan": {}}, df, question, None, object())

    single = result["single_prediction"]
    assert result["status"] == "success"
    assert single is not None
    assert single["prediction"] == "beta"
    assert single["probabilities"] == {"beta": pytest.approx(0.8), "alpha": pytest.approx(0.2)}
    assert sum(single["probabilities"].values()) == pytest.approx(1.0)
    assert single["confidence"] == pytest.approx(0.8)
    assert single["probability_class"] == "beta"
    assert estimator.proba_calls == 1
    assert bool(single.get("defaulted_inputs")) is should_default
    assert single.get("ece") is None and single.get("ece_note")


def test_probability_validation_rejects_malformed_estimator_output():
    class BadEstimator:
        classes_ = np.asarray(["a", "b"])

        def predict_proba(self, _X):
            return np.asarray([[0.2, 0.2]])

    summary = classification_probability_summary(BadEstimator(), np.zeros((1, 1)))
    assert summary["probabilities"] is None
    assert summary["confidence"] is None
    assert "sum to 1" in summary["probability_note"]


def test_safe_imputation_uses_finite_values_and_preserves_zero():
    assert safe_numeric_imputation(pd.Series([0.0, 0.0, 2.0]), 0.0) == (0.0, "training median")
    value, source = safe_numeric_imputation(pd.Series([np.inf, -np.inf, 4.0]), np.inf)
    assert value == 4.0 and "current dataset median" in source
    value, source = safe_numeric_imputation(pd.Series([np.inf, -np.inf, np.nan]), np.nan)
    assert value == 0.0 and "no finite" in source
    assert safe_categorical_imputation(pd.Series([None, "west", "west"]), None) == (
        "west", "current dataset mode (saved imputer unavailable)")
    assert safe_categorical_imputation(pd.Series([None, " "]), None) == (
        "missing", "'missing' fallback because no known category is available")


def test_ece_rejects_invalid_probabilities_without_rescaling():
    y = np.asarray([0, 1] * 5)
    assert expected_calibration_error(y, np.asarray([0.1, 0.9] * 5)) is not None
    assert expected_calibration_error(y, np.asarray([0.2, 1.4] * 5)) is None
    assert expected_calibration_error(y, np.asarray([0.2] * 9)) is None


# --- Forecast -------------------------------------------------------------------

def test_forecast_end_to_end_with_intervals(client):
    df = pd.read_csv(SAMPLES / "sales_monthly.csv")
    body = client.post("/api/assistant/ask", json={
        "question": "What will net_sales be next month?", "data": _rows(df)}).json()
    assert body["status"] == "success" and body["intent"] == "forecast", body.get("message")
    future = body["future_forecast"]
    assert len(future) == 3
    for point in future:
        assert point["lower"] <= point["forecast"] <= point["upper"]
    assert body["baseline_smape"] is not None
    assert body["reliability"] in {"high", "medium", "low"}
    assert body["metrics"]["SMAPE"] >= 0


def test_forecast_short_series_refuses_honestly(client):
    df = pd.DataFrame({"d": pd.date_range("2023-01-01", periods=6, freq="MS").strftime("%Y-%m-%d"),
                       "v": [1, 2, 3, 4, 5, 6]})
    body = client.post("/api/assistant/ask", json={
        "question": "Forecast v for next month", "data": _rows(df)}).json()
    assert body["status"] in ("unsupported", "failed")
    assert not body.get("future_forecast")


# --- Anomaly / segmentation -------------------------------------------------------------------

def test_anomaly_finds_injected_outlier(synth):
    df = synth.copy()
    df.loc[len(df)] = {"size": 10000.0, "rooms": 2, "age": 5, "price": 9_999_999.0, "band": "high"}
    result = run_anomaly_detection(df, profile_dataset(df, "t"), top_n=5)
    assert result["status"] == "success"
    top = result["anomalies"][0]
    assert top["record"]["size"] == 10000.0
    scores = [a["score"] for a in result["anomalies"]]
    assert scores == sorted(scores, reverse=True)
    assert "probability" not in str(result).lower() or "not probabilities" in result["answer"]


def test_anomaly_endpoint(synth, client):
    body = client.post("/api/assistant/ask", json={
        "question": "Find anomalies in price and size", "data": _rows(synth)}).json()
    assert body["status"] == "success" and body["intent"] == "anomaly"
    assert body["anomalies"] and body["rows_flagged"] >= 1


def test_segmentation_two_blobs():
    rng = np.random.RandomState(3)
    df = pd.DataFrame({
        "x": np.concatenate([rng.normal(0, 1, 40), rng.normal(10, 1, 40)]),
        "y": np.concatenate([rng.normal(0, 1, 40), rng.normal(10, 1, 40)]),
    })
    result = run_segmentation(df, profile_dataset(df, "t"))
    assert result["status"] == "success"
    assert result["k"] == 2
    assert result["silhouette"] > 0.5
    assert sum(c["size"] for c in result["clusters"]) == 80


def test_segmentation_insufficient_data():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    result = run_segmentation(df, profile_dataset(df, "t"))
    assert result["status"] == "unsupported"


# --- Leakage / validation -------------------------------------------------------------------

def test_target_leak_column_excluded(synth):
    df = synth.copy()
    df["price_copy"] = df["price"]
    feats = select_features(df, profile_dataset(df, "t"), "price")
    assert "price_copy" not in feats["features"]
    assert "price_copy" in feats["excluded"]


def test_constant_target_refused():
    df = pd.DataFrame({"a": range(20), "b": [5] * 20})
    picked = select_supervised_target(df, profile_dataset(df, "t"), "Predict b from a", "regression")
    assert "issue" in picked


def test_feature_values_require_full_coverage(synth):
    parsed = extract_feature_values("Predict price where size = 50", synth, ["size", "rooms", "age"])
    assert parsed["values"] == {"size": 50.0}
    assert parsed["missing"] == ["rooms", "age"]


# --- Multi-intent / model-confidence ------------------------------------------------------

def test_multi_intent_runs_supervised_task_first(client, synth):
    body = client.post("/api/assistant/ask", json={
        "question": "Predict price from size and explain the prediction",
        "data": _rows(synth)}).json()
    assert body["status"] == "success" and body["intent"] == "regression"
    assert body["target"] == "price"
    # Explanation-relevant evidence ships with the executed summary.
    assert body["feature_importance"]


def test_confidence_of_named_model_executes(client, synth):
    body = client.post("/api/assistant/ask", json={
        "question": "What is the confidence of the price model?",
        "data": _rows(synth)}).json()
    assert body["status"] == "success" and body["intent"] == "regression"
    assert body["confidence"]["validation_metrics"]["evaluated_rows"] >= 20


# --- Cold start -----------------------------------------------------------------------------------

def test_cold_start_fallback_is_labeled_with_evidence():
    rows = []
    for c in range(6):
        for w in range(3):
            rows.append({"cid": f"C{c}", "prod": f"P{(c + w) % 3}", "ord": f"O{c}{w}",
                         "ts": pd.Timestamp("2023-01-01") + pd.Timedelta(days=w * 30),
                         "category": "CatA" if (c + w) % 2 == 0 else "CatB"})
    df = pd.DataFrame(rows)
    schema = discover_purchase_schema(df, profile_dataset(df, "t"))
    built = build_purchase_histories(df, schema)
    pred = predict_next_for_customer(built["clean"], built["customers"], "GHOST", "popularity", 2)
    assert pred["label"] == "recommendation" and pred["cold_start"] is True
    assert pred["fallback"]["is_personalized"] is False
    assert pred["fallback"]["missing"]
    assert pred["category_popularity"]  # category evidence used alongside popularity
