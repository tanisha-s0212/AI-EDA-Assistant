"""Regression contracts for schema-grounded, non-LLM question understanding."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from nl_analytics import answer_question, classify_intent, profile_dataset  # noqa: E402


def test_explicit_row_prediction_is_not_routed_to_forecast_just_because_dates_exist():
    frame = pd.DataFrame({
        "period_start": pd.date_range("2023-01-01", periods=30, freq="MS"),
        "load_kw": np.arange(30, dtype=float) + 10,
        "station": [f"S{index % 3}" for index in range(30)],
    })
    profile = profile_dataset(frame)

    supervised = classify_intent("Predict load_kw using station", profile)
    future = classify_intent("Forecast load_kw for next month", profile)

    assert supervised["intent"] == "regression"
    assert future["intent"] == "forecast"


def test_question_understanding_handles_unseen_wording_for_grouped_mean():
    frame = pd.DataFrame({
        "facility": ["East", "East", "West", "West"],
        "vibration": [2.0, 4.0, 10.0, 14.0],
    })

    result = answer_question(frame, "Give me the mean vibration for each facility")

    assert result["status"] == "success"
    assert result["intent"] in {"comparison", "aggregation"}
    assert result["operation"]["measure"] == "vibration"
    assert result["operation"]["aggregation"] == "mean"
    assert result["operation"]["group_by"] == "facility"
    assert result["grouped_result"] == {"East": 3.0, "West": 12.0}


def test_unresolved_filter_is_not_silently_dropped_from_ranking():
    frame = pd.DataFrame({
        "region": ["North", "North", "South", "South"],
        "department": ["A", "B", "A", "B"],
        "net_sales": [10.0, 30.0, 100.0, 300.0],
    })

    result = answer_question(
        frame,
        "Which departments have the highest sales in the Atlantis region?",
    )

    assert result["status"] == "clarify"
    assert "atlantis" in result["message"].lower()


def test_forecast_plan_keeps_the_explicit_target_in_non_sales_data():
    frame = pd.DataFrame({
        "observed_month": pd.date_range("2022-01-01", periods=30, freq="MS"),
        "vibration": np.linspace(3.0, 4.0, 30),
        "background_noise": np.linspace(100.0, 900.0, 30),
    })

    result = answer_question(frame, "Forecast vibration for the next three months")

    assert result["status"] == "ml_required"
    assert result["intent"] == "forecast"
    assert result["ml_plan"]["target"] == "vibration"
    assert result["ml_plan"]["date_column"] == "observed_month"
    assert result["ml_plan"]["horizon"] == 3


def test_grouped_forecast_is_preserved_for_safe_refusal_by_single_series_runner():
    frame = pd.DataFrame({
        "month_start": pd.date_range("2022-01-01", periods=30, freq="MS"),
        "energy_reading": np.linspace(20.0, 50.0, 30),
        "site": ["A", "B"] * 15,
    })

    result = answer_question(frame, "Forecast energy_reading per site for the next three months")

    assert result["status"] == "ml_required"
    assert result["intent"] == "forecast"
    assert result["ml_plan"]["target"] == "energy_reading"
    assert result["ml_plan"]["group_by"] == "site"
