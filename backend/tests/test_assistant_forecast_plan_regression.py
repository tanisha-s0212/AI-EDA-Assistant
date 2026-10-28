"""The assistant forecast plan must reach the existing TS runner unchanged."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402


def test_assistant_forecast_uses_planned_target_date_and_horizon(monkeypatch):
    frame = pd.DataFrame({
        "observed_month": pd.date_range("2023-01-01", periods=30, freq="MS"),
        "vibration": range(30),
        "background": range(100, 130),
    })
    assert_series_call = {}
    assert_horizon = {}

    def prepare(source, date_column, target_column):
        assert_series_call.update(date=date_column, target=target_column)
        return pd.DataFrame({"period": pd.date_range("2023-01-01", periods=30, freq="MS"),
                             "sales": range(30)}), "MS", "month"

    def train(*args, **kwargs):
        horizon = args[6]
        assert_horizon["value"] = horizon
        series = pd.DataFrame({"period": pd.date_range("2023-01-01", periods=30, freq="MS"),
                               "sales": range(30)})
        return {}, series.sales.iloc[:24], series.sales.iloc[24:], series.iloc[:24], series.iloc[24:], series, None

    monkeypatch.setattr(main, "build_column_info_from_frame", lambda _frame: [])
    monkeypatch.setattr(main, "prepare_sales_series", prepare)
    monkeypatch.setattr(main, "normalize_ts_frequency", lambda *_args: ("monthly", 12))
    monkeypatch.setattr(main, "infer_dataset_grain", lambda _frame: {"grain": "unknown"})
    monkeypatch.setattr(main, "check_stationarity", lambda *_args: {})
    monkeypatch.setattr(main, "train_all_ts_models", train)
    monkeypatch.setattr(main, "auto_select_ts_model", lambda _results: ("baseline", {"smape": 1.0, "mae": 1.0, "rmse": 1.0}, "selected"))
    monkeypatch.setattr(main, "generate_ts_future_forecast", lambda *_args: [{"period": "next", "forecast": 1.0}])

    result = main._assistant_run_forecast({
        "ml_plan": {"target": "vibration", "date_column": "observed_month", "horizon": 5}
    }, frame, None, None)

    assert result["status"] == "success"
    assert assert_series_call == {"date": "observed_month", "target": "vibration"}
    assert assert_horizon["value"] == 5
    assert result["ml_plan"]["horizon"] == 5
    assert result["target"] == "vibration"
    assert result["date_column"] == "observed_month"


def test_single_series_runner_refuses_requested_dimension_breakdown():
    result = main._assistant_run_forecast({
        "ml_plan": {"target": "vibration", "date_column": "observed_month",
                    "horizon": 3, "group_by": "station"}
    }, pd.DataFrame(), None, None)

    assert result["status"] == "unsupported"
    assert "per-station" in result["message"].lower()
