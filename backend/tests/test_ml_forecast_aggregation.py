"""ML forecast aggregation: high-cardinality timestamps must not become one chart point per row."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from ml_forecast_pipeline import (  # noqa: E402
    _build_future_row,
    _naive_holdout_predictions,
    _recursive_holdout_predictions,
    build_models_dict,
    clean_data,
    engineer_features,
    forecast_future,
    get_config,
    prepare_frame_for_ml_pipeline,
    run_full_pipeline,
)


def _high_cardinality_frame(rows: int = 240) -> pd.DataFrame:
    start = pd.Timestamp('2020-01-01 08:15:00')
    return pd.DataFrame({
        'created': [start + pd.Timedelta(days=index) + pd.Timedelta(hours=index % 11) for index in range(rows)],
        'sales': [100.0 + (index % 13) for index in range(rows)],
    })


def test_prepare_frame_for_ml_coarsens_unique_timestamps():
    raw = _high_cardinality_frame(240)
    assert len(raw) == 240
    assert raw['created'].nunique() == 240

    prepared, date_col, target_col, frequency, period_label = prepare_frame_for_ml_pipeline(
        raw, 'created', 'sales'
    )

    assert date_col == 'created'
    assert target_col == 'sales'
    assert period_label in {'day', 'week', 'month', 'quarter', 'year'}
    assert frequency in {'weekly', 'monthly'}
    # Must be far fewer than unique raw timestamps (not one chart/train point per row).
    assert len(prepared) < len(raw)
    assert len(prepared) <= 60
    assert prepared[date_col].nunique() == len(prepared)
    assert prepared.attrs.get('period_label') == period_label


def test_run_full_pipeline_history_matches_usable_periods(tmp_path: Path):
    frame = _high_cardinality_frame(240)
    # Prefer gradient boosting only path via auto — still trains ensemble but fixture is small enough.
    result = run_full_pipeline(
        tmp_path,
        target_col='sales',
        date_col='created',
        horizon=3,
        frequency='auto',
        frame=frame,
        model_type='gradient_boosting',
    )

    actual_rows = [row for row in result['forecast_line'] if row['type'] == 'actual']
    history_len = len(actual_rows)
    assert history_len < len(frame)
    assert history_len == int(result['metadata']['usable_periods'])
    assert result['metadata']['period_label'] in {'day', 'week', 'month', 'quarter', 'year'}
    assert history_len == len({row['period'] for row in actual_rows})


def test_lightgbm_is_not_an_ml_forecast_candidate(tmp_path: Path):
    assert set(build_models_dict()) == {'Gradient Boosting', 'XGBoost', 'Prophet'}
    with pytest.raises(ValueError, match='Unsupported ML forecast model'):
        run_full_pipeline(tmp_path, frame=pd.DataFrame(), model_type='lightgbm')


def test_engineered_features_exclude_source_columns_that_are_unavailable_at_forecast_time():
    frame = pd.DataFrame({
        'period': pd.date_range('2020-01-01', periods=48, freq='MS'),
        'sales': np.linspace(100.0, 300.0, 48),
        # This would be target leakage if it were accepted as a feature.
        'gross_profit': np.linspace(40.0, 120.0, 48),
        'promotion_cost': np.linspace(5.0, 15.0, 48),
    })
    engineered, features = engineer_features(
        frame, 'sales', 'period', 'monthly', get_config('monthly'), ['trend', 'lags']
    )
    assert len(engineered) == len(frame)
    assert 'gross_profit' not in features
    assert 'promotion_cost' not in features
    assert set(features) <= {'trend_index', 'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12'}


def test_recursive_holdout_does_not_read_holdout_actuals_as_lag_features():
    frame = pd.DataFrame({
        'period': pd.date_range('2020-01-01', periods=48, freq='MS'),
        'sales': 100.0 + np.arange(48) * 2.0 + 8.0 * np.sin(np.arange(48)),
    })
    engineered, features = engineer_features(frame, 'sales', 'period', 'monthly', get_config('monthly'))
    split = 36
    model = GradientBoostingRegressor(random_state=7, n_estimators=20, max_depth=2)
    predictions = _recursive_holdout_predictions(
        model, engineered.iloc[:split], engineered.iloc[split:]['period'], features, 'sales', 'period', 'monthly'
    )
    altered = engineered.copy()
    altered.loc[split:, 'sales'] = 999999.0
    altered_predictions = _recursive_holdout_predictions(
        model, altered.iloc[:split], altered.iloc[split:]['period'], features, 'sales', 'period', 'monthly'
    )
    assert predictions == altered_predictions
    baseline, label = _naive_holdout_predictions(engineered.iloc[:split]['sales'].tolist(), len(predictions), 'monthly')
    assert len(baseline) == len(predictions)
    assert label == 'seasonal-naive (12-period)'


@pytest.mark.parametrize("frequency, date_frequency, periods", [
    ("weekly", "W-MON", 100),
    ("monthly", "MS", 48),
])
def test_optimized_recursive_forecast_matches_original_features(frequency, date_frequency, periods):
    date_col, target = "period", "sales"
    frame = pd.DataFrame({
        date_col: pd.date_range("2020-01-01", periods=periods, freq=date_frequency),
        target: 300.0 + np.arange(periods) * 0.7 + 18.0 * np.sin(np.arange(periods) * 2 * np.pi / 12),
    })
    config = get_config(frequency)
    clean, _ = clean_data(frame, target, date_col, frequency)
    feature_frame, features = engineer_features(clean, target, date_col, frequency, config)
    model = GradientBoostingRegressor(random_state=42, n_estimators=25, max_depth=2).fit(
        feature_frame[features], feature_frame[target]
    )

    history = feature_frame[[date_col, target]].copy().sort_values(date_col).reset_index(drop=True)
    next_date = pd.Timestamp(history.iloc[-1][date_col]) + config["offset"]
    legacy = []
    for _ in range(24):
        row = _build_future_row(history, date_col, target, frequency, config, next_date)
        row = row.reindex(columns=features).fillna(float(history[target].mean()))
        prediction = max(0.0, float(model.predict(row)[0]))
        legacy.append({
            "period": next_date.strftime("%Y-%m-%d"),
            "period_label": f"Week of {next_date.strftime('%Y-%m-%d')}" if frequency == "weekly" else f"Month of {next_date.strftime('%Y-%m')}",
            "forecast_value": prediction,
        })
        history = pd.concat([history, pd.DataFrame([{date_col: next_date, target: prediction}])], ignore_index=True)
        next_date += config["offset"]

    optimized = forecast_future(model, feature_frame, features, target, date_col, 24, frequency)
    assert optimized == legacy
