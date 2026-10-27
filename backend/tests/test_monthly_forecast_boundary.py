"""Regression coverage for the inclusive monthly forecast boundary."""
from __future__ import annotations

import sys
import json
import builtins
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402


def monthly_frame(periods: int) -> pd.DataFrame:
    return pd.DataFrame({
        'month': pd.date_range('2023-01-01', periods=periods, freq='MS'),
        'sales': np.arange(periods, dtype=float),
    })


def test_24_months_keep_zero_period_and_use_chronological_nonseasonal_folds(monkeypatch):
    calls = []
    stationarity_lengths = []
    def stationarity(values, _frequency):
        stationarity_lengths.append(len(values))
        return {'differencing_required': False}
    monkeypatch.setattr(main, 'check_stationarity', stationarity)

    def fit(name, frame, next_date, target, date, frequency, freq_period):
        calls.append((name, len(frame), frame[date].max(), next_date, freq_period))
        return 1.0, None, None, {}

    monkeypatch.setattr(main, '_fit_predict_ts_candidate', fit)
    results, _, _, train, test, clean, _ = main.train_all_ts_models(
        monthly_frame(24), 'sales', 'month', 'monthly', 12, monthly_aggregate=True,
    )
    assert len(clean) == 24
    assert len(train) == 19 and len(test) == 5
    assert stationarity_lengths == [19]
    assert all(last < next_date for _, _, last, next_date, _ in calls)
    assert all(period == 1 for name, _, _, _, period in calls if name in {'SARIMA', 'HoltWinters'})
    assert all(period == 12 for name, _, _, _, period in calls if name == 'Prophet')
    assert results['SARIMA']['seasonal_period_used'] == 1
    assert 'nonseasonal' in results['SARIMA']['note'].lower()


def test_23_months_are_below_the_forecast_minimum(monkeypatch):
    monkeypatch.setattr(main, 'check_stationarity', lambda *_: {'differencing_required': False})
    with pytest.raises(ValueError, match='at least 24'):
        main.train_all_ts_models(monthly_frame(23), 'sales', 'month', 'monthly', 12, monthly_aggregate=True)


def test_25_months_are_also_accepted(monkeypatch):
    monkeypatch.setattr(main, 'check_stationarity', lambda *_: {'differencing_required': False})
    monkeypatch.setattr(main, '_fit_predict_ts_candidate', lambda *_: (1.0, None, None, {}))
    result = main.train_all_ts_models(monthly_frame(25), 'sales', 'month', 'monthly', 12, monthly_aggregate=True)
    assert len(result[5]) == 25


def test_nonmonthly_source_keeps_existing_zero_trimming(monkeypatch):
    monkeypatch.setattr(main, 'check_stationarity', lambda *_: {'differencing_required': False})
    monkeypatch.setattr(main, '_fit_predict_ts_candidate', lambda *_: (1.0, None, None, {}))
    result = main.train_all_ts_models(monthly_frame(25), 'sales', 'month', 'monthly', 12)
    assert len(result[5]) == 24
    assert result[5]['sales'].iloc[0] == 1.0


def test_long_monthly_training_window_keeps_twelve_month_season(monkeypatch):
    calls = []
    monkeypatch.setattr(main, 'check_stationarity', lambda *_: {'differencing_required': False})
    def fit(name, _frame, _next, _target, _date, _frequency, period):
        calls.append((name, period))
        return 1.0, None, None, {}
    monkeypatch.setattr(main, '_fit_predict_ts_candidate', fit)
    main.train_all_ts_models(monthly_frame(48), 'sales', 'month', 'monthly', 12, monthly_aggregate=True)
    assert calls and all(period == 12 for _, period in calls)


def test_failed_final_fit_tries_next_validated_candidate(monkeypatch):
    models = {
        'SARIMA': {'status': 'completed', 'smape': 1.0, 'mae': 1.0, 'seasonal_period_used': 1, 'differencing_applied': False},
        'HoltWinters': {'status': 'completed', 'smape': 2.0, 'mae': 2.0, 'seasonal_period_used': 1, 'differencing_applied': False},
    }
    calls = []
    def forecast(name, _frame, _target, _date, _frequency, period, _horizon, **kwargs):
        calls.append((name, period, kwargs['differencing_required']))
        if name == 'SARIMA':
            raise ValueError('final fit failed')
        return [{'period': 'Month of 2025-01', 'predicted': 1.0}]
    monkeypatch.setattr(main, 'generate_ts_future_forecast', forecast)
    best, _, _, future = main.select_and_forecast_ts(models, monthly_frame(24), 'sales', 'month', 'monthly', 12, 1)
    assert best == 'HoltWinters'
    assert calls == [('SARIMA', 1, False), ('HoltWinters', 1, False)]
    assert models['SARIMA']['status'] == 'failed'
    assert len(future) == 1


def test_nonseasonal_sarima_fit_does_not_request_seasonal_arima(monkeypatch):
    options = {}
    class Model:
        order = (1, 0, 0)
        seasonal_order = (0, 0, 0, 0)
        def aic(self): return 1.0
        def predict(self, **_kwargs): return np.array([1.0]), np.array([[0.0, 2.0]])

    def auto_arima(_values, **kwargs):
        options.update(kwargs)
        return Model()

    monkeypatch.setitem(sys.modules, 'pmdarima', SimpleNamespace(auto_arima=auto_arima))
    frame = monthly_frame(19)
    main._fit_predict_ts_candidate('SARIMA', frame, pd.Timestamp('2024-08-01'), 'sales', 'month', 'monthly', 1)
    assert options['seasonal'] is False
    assert options['m'] == 1


def test_reforecast_keeps_monthly_zero_and_selected_model_settings(monkeypatch):
    frame = monthly_frame(24)
    frame.attrs['monthly_aggregate'] = True
    monkeypatch.setattr(main, 'ACTIVITY_DB_AVAILABLE', False)
    monkeypatch.setattr(main, 'get_workspace_context', lambda *_: {
        'best_model': 'SARIMA', 'seasonal_period_used': 1, 'differencing_applied': False,
    })
    monkeypatch.setattr(main, 'load_ts_dataset', lambda *_args, **_kwargs: (frame, 'month', 'sales'))
    calls = []
    def forecast(name, clean, _target, _date, frequency, period, horizon, **kwargs):
        calls.append((name, len(clean), frequency, period, horizon, kwargs.get('differencing_required')))
        return [{'period': 'Month of 2025-01', 'predicted': 1.0}]
    monkeypatch.setattr(main, 'generate_ts_future_forecast', forecast)
    response = main.forecast_ts_best_model(main.TsForecastRunRequest(dataset_id='test', horizon=1))
    assert response.status_code == 200
    assert json.loads(response.body)['status'] == 'completed'
    assert calls == [('SARIMA', 24, 'monthly', 1, 1, False)]


def test_missing_statsmodels_explains_how_to_fix_runtime(monkeypatch):
    original_import = builtins.__import__
    def missing_statsmodels(name, *args, **kwargs):
        if name == 'statsmodels.tsa.stattools':
            raise ModuleNotFoundError("No module named 'statsmodels'", name='statsmodels')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing_statsmodels)
    with pytest.raises(ModuleNotFoundError, match='project virtual environment'):
        main.check_stationarity(pd.Series([1., 2., 3., 4.]), 'monthly')


def test_output_files_preserve_model_note_and_missing_intervals(tmp_path):
    frame = monthly_frame(24)
    metrics = {
        'status': 'completed', 'mae': 1.0, 'rmse': 1.0, 'mape': 1.0, 'smape': 1.0,
        'predictions': [1.0], 'conf_int_lower': [None], 'conf_int_upper': [None],
        'note': 'Nonseasonal fit due to the short training window.',
    }
    future = [{'period': 'Month of 2025-01', 'forecast': 1.0, 'lower': 0.0, 'upper': 2.0}]
    main.write_ts_output_files(str(tmp_path), 'monthly', 12, 'sales', 'month', frame, frame,
                               frame.iloc[:-1], frame.iloc[-1:], {}, {'SARIMA': metrics},
                               'SARIMA', metrics, 'test', future, {})
    comparison = json.loads((tmp_path / 'model_comparison.json').read_text())
    assert comparison[0]['note'] == metrics['note']
    lines = json.loads((tmp_path / 'forecast_line.json').read_text())
    backtest = next(row for row in lines if row['type'] == 'backtest')
    assert backtest['lower'] is None and backtest['upper'] is None


def test_monthly_primary_arima_has_valid_nonseasonal_search_parameters(monkeypatch):
    options = {}
    def auto_arima(_series, **kwargs):
        options.update(kwargs)
        return SimpleNamespace(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    class FakeSarimax:
        def __init__(self, *_args, **_kwargs): pass
        def fit(self, **_kwargs): return SimpleNamespace(forecast=lambda periods: [1.0] * periods)
    monkeypatch.setattr(main, 'auto_arima', auto_arima)
    monkeypatch.setattr(main, 'SARIMAX', FakeSarimax)
    forecast, meta = main.fit_arima_family_forecast(list(range(24)), 1, 'month', seasonal=False, monthly_aggregate=True)
    assert options['start_P'] == options['start_Q'] == 0
    assert options['max_P'] == options['max_Q'] == 0
    assert meta['selection'] == 'auto_arima' and forecast == [1.0]
