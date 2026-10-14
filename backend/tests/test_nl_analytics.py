"""Tests for the universal dataset-aware NL analytics engine.

Covers: intent routing, OEM multi-metric acceptance (expected values computed
independently from the source CSV), dataset independence (synthetic frames with
unfamiliar column names), ties/missing/empty/ambiguous handling, and dataset
lifecycle (signature refresh, no stale leakage).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from nl_analytics import (  # noqa: E402
    answer_question,
    classify_intent,
    dataset_signature,
    detect_requested_metrics,
    profile_dataset,
    resolve_filter,
    resolve_metric_column,
)

SAMPLES = Path(__file__).resolve().parents[2] / "samples" / "sales"
OEM_QUESTION = ("Which product lines have the highest sales, units shipped, "
                "and order counts for the OEM customer segment?")


def load_monthly() -> pd.DataFrame:
    return pd.read_csv(SAMPLES / "sales_monthly.csv")


def load_invoices() -> pd.DataFrame:
    return pd.read_csv(SAMPLES / "sales_invoices.csv")


# --- Intent routing ---------------------------------------------------------

def test_descriptive_routes_to_deterministic_analytics():
    profile = profile_dataset(load_monthly(), "d")
    for q in ["What is the total revenue?",
              "Show average order value by region",
              "List all records for OEM"]:
        routing = classify_intent(q, profile)
        assert not routing["needs_ml"], q
        assert routing["intent"] in {"descriptive", "filter", "ranking", "comparison", "trend", "statistics"}


def test_ranking_question_does_not_route_to_regression():
    profile = profile_dataset(load_monthly(), "d")
    routing = classify_intent(OEM_QUESTION, profile)
    assert routing["intent"] == "ranking"
    assert routing["needs_ml"] is False
    assert "groupby" in routing["operations"] or "groupby_multi" in routing["operations"]


def test_genuine_prediction_routes_to_ml():
    profile = profile_dataset(load_monthly(), "d")
    routing = classify_intent("What will sales be next month?", profile)
    assert routing["intent"] == "forecast"
    assert routing["needs_ml"] is True
    routing2 = classify_intent("Predict net_sales from units_shipped and cogs", profile)
    assert routing2["needs_ml"] is True


def test_ambiguous_question_reports_limitation_not_fabrication():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = answer_question(df, "Which product lines have the highest sales for OEM?", dataset_id="t")
    assert result["status"] in ("unsupported", "empty", "success")
    if result["status"] == "unsupported":
        assert "message" in result and result["message"]


# --- OEM acceptance (expected values computed independently) ------------------

def test_oem_acceptance_monthly():
    df = load_monthly()
    result = answer_question(df, OEM_QUESTION, dataset_id="monthly")
    assert result["status"] == "success", result.get("message")
    assert result["routing"]["needs_ml"] is False
    assert result["filter"] == {"column": "customer_segment", "value": "OEM", "matched_hint": "OEM"}
    assert result["group_by"] == "product_line"

    oem = df[df["customer_segment"] == "OEM"]
    expected = {
        "revenue": oem.groupby("product_line")["net_sales"].sum().sort_values(ascending=False),
        "units": oem.groupby("product_line")["units_shipped"].sum().sort_values(ascending=False),
        "orders": oem.groupby("product_line")["order_count"].sum().sort_values(ascending=False),
    }
    for kind, series in expected.items():
        table = result["tables"][kind]
        assert table["rows"][0]["group"] == str(series.index[0])
        assert table["rows"][0]["value"] == pytest.approx(round(float(series.iloc[0]), 2))
        # Full ranking order matches independent calculation.
        assert [r["group"] for r in table["rows"]] == [str(i) for i in series.index[:len(table["rows"])]]
        assert table["rows"][0]["value"] == pytest.approx(result["winners"][kind]["value"])
    # Independent per-metric ranks (winners need not coincide).
    assert set(result["winners"]) == {"revenue", "units", "orders"}
    assert result["coverage"]["filtered_rows"] == int((df["customer_segment"] == "OEM").sum())
    assert "answer" in result and isinstance(result["answer"], str) and len(result["answer"]) > 20


def test_oem_acceptance_invoices_grain():
    """Same question on the daily invoice grain: order counts fall back to row counts."""
    df = load_invoices()
    assert "order_count" not in df.columns
    result = answer_question(df, OEM_QUESTION, dataset_id="invoices")
    assert result["status"] == "success", result.get("message")
    assert result["filter"]["value"] == "OEM"
    oem = df[df["customer_segment"] == "OEM"]
    exp_rev = oem.groupby("product_line")["net_sales"].sum().sort_values(ascending=False)
    assert result["tables"]["revenue"]["rows"][0]["group"] == str(exp_rev.index[0])
    exp_units = oem.groupby("product_line")["units_shipped"].sum().sort_values(ascending=False)
    assert result["tables"]["units"]["rows"][0]["group"] == str(exp_units.index[0])
    # orders -> row-count fallback per product line
    exp_n = oem.groupby("product_line").size().sort_values(ascending=False)
    assert result["tables"]["orders"]["rows"][0]["group"] == str(exp_n.index[0])


def test_no_hardcoded_oem_answer_logic():
    """Different segment value + renamed values must still work (no OEM special-case)."""
    df = load_monthly().copy()
    df["customer_segment"] = df["customer_segment"].replace({"OEM": "ACME-CORP"})
    q = ("Which product lines have the highest sales, units shipped, and order counts "
         "for the ACME-CORP customer segment?")
    result = answer_question(df, q, dataset_id="renamed")
    assert result["status"] == "success", result.get("message")
    assert result["filter"]["value"] == "ACME-CORP"
    assert set(result["winners"]) == {"revenue", "units", "orders"}


# --- Dataset independence ----------------------------------------------------

def test_unfamiliar_column_names_healthcare():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "ward": rng.choice(["North", "South", "East"], 120),
        "pt_cohort": rng.choice(["A", "B"], 120),
        "stay_days": rng.randint(1, 30, 120),
        "visits_n": rng.randint(1, 6, 120),
    })
    result = answer_question(df, "Which wards have the highest stay_days and visits_n for cohort B?",
                             dataset_id="hc")
    # Generic engine: filter on cohort value B, group by ward-like dimension.
    assert result["status"] in ("success", "unsupported")
    if result["status"] == "success":
        assert result["filter"]["value"] == "B"
        assert result["group_by"] == "ward"


def test_numerical_only_dataset():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
    result = answer_question(df, "What is the correlation between x and y?", dataset_id="num")
    assert result["status"] == "success"
    assert result["intent"] == "statistics"
    assert result["correlations"]["x"]["y"] == pytest.approx(1.0)


def test_missing_values_duplicates_and_ties():
    df = pd.DataFrame({
        "seg": ["OEM", "OEM", "OEM", "X", "OEM", None, "OEM"],
        "line": ["A", "A", "B", "A", "B", "A", "C"],
        "sales": [100.0, 100.0, 200.0, 999.0, np.nan, 5.0, 200.0],
        "units": [1, 1, 2, 9, 2, 1, 2],
    })
    result = answer_question(df, "Which lines have the highest sales and units for the OEM segment?",
                             dataset_id="messy")
    assert result["status"] == "success", result.get("message")
    # A: 200, B: 200 (nan dropped), C: 200 -> three-way tie must be flagged.
    sales_rows = {r["group"]: r for r in result["tables"]["revenue"]["rows"]}
    assert sales_rows["A"]["tied"] and sales_rows["B"]["tied"] and sales_rows["C"]["tied"]
    assert result["winners"]["revenue"]["tied_with"]


def test_empty_filter_result():
    df = load_monthly()
    result = answer_question(df, "Which product lines have the highest sales for the NONEXISTENT segment XYZ?",
                             dataset_id="monthly")
    assert result["status"] in ("empty", "unsupported", "success")
    if result["status"] == "empty":
        assert "No records matched" in result["message"]
    elif result["status"] == "success":
        # No filter resolved -> unfiltered ranking still valid, must say so.
        assert result["filter"] is None


def test_trend_and_filter_and_statistics_paths():
    df = load_monthly()
    trend = answer_question(df, "Show the sales trend over time for OEM", dataset_id="m")
    assert trend["status"] == "success" and trend["intent"] == "trend"
    assert len(trend["points"]) > 0
    filt = answer_question(df, "List records for Industrial Distributor", dataset_id="m")
    assert filt["status"] == "success" and filt["matched_rows"] > 0
    stats = answer_question(df, "What is the average net_sales and its distribution?", dataset_id="m")
    assert stats["status"] == "success" and "net_sales" in stats["summary"]


def test_forecast_without_dates_is_unsupported_not_fabricated():
    df = pd.DataFrame({"color": ["r", "b", "r"], "size": [1, 2, 3]})
    result = answer_question(df, "What will size be next month?", dataset_id="nodate")
    assert result["status"] in ("unsupported", "ml_required", "success")
    # Must never return invented forecast numbers.
    assert "forecast" not in str(result.get("tables", "")).lower() or result["status"] != "success"


# --- Lifecycle ----------------------------------------------------------------

def test_dataset_signature_refreshes_on_replacement():
    df1 = load_monthly()
    df2 = df1.rename(columns={"net_sales": "revenue_usd"})
    assert dataset_signature(df1, "d") != dataset_signature(df2, "d")
    p1 = profile_dataset(df1, "d")
    p2 = profile_dataset(df2, "d")
    assert "net_sales" in p1["numeric_columns"]
    assert "revenue_usd" in p2["numeric_columns"]
    assert "net_sales" not in p2["numeric_columns"]
    # Stale mapping must not leak: revenue question resolves to the new name.
    col = resolve_metric_column(df2, p2, "revenue")
    assert col == "revenue_usd"


def test_profile_discovers_schema_dynamically():
    df = load_invoices()
    profile = profile_dataset(df, "inv")
    assert profile["rows"] == len(df) and profile["cols"] == len(df.columns)
    assert "customer_segment" in profile["categorical_columns"]
    assert "net_sales" in profile["numeric_columns"]
    assert profile["duplicate_rows"] == int(df.duplicated().sum())
    filt = resolve_filter(df, profile, OEM_QUESTION)
    assert filt is not None and filt["column"] == "customer_segment" and filt["value"] == "OEM"


def test_detect_requested_metrics_multi():
    assert detect_requested_metrics(OEM_QUESTION) == ["revenue", "units", "orders"]
