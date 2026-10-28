"""Independent model-selection windows and explicit financial mappings."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException
from sklearn.dummy import DummyRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402
import ml_forecast_pipeline as pipeline  # noqa: E402


def test_ml_model_selection_never_sees_final_holdout(monkeypatch, tmp_path):
    frame = pd.DataFrame({'period': pd.date_range('2021-01-01', periods=36, freq='MS'),
                          'measure': [100 + i * 2 for i in range(36)]})
    seen = {}
    monkeypatch.setattr(pipeline, 'build_models_dict', lambda: {'Gradient Boosting': DummyRegressor()})
    def validation(data, *args, **kwargs):
        seen['selection_last'] = data['period'].max()
        return {'Gradient Boosting': {'status': 'completed', 'smape': 1., 'mae': 1., 'rmse': 1., 'mape': 1., 'note': 'completed'}}
    def outputs(*args, **kwargs):
        features = args[5]
        seen['holdout_first'] = features.iloc[int(len(features) * 0.8)]['period']
        return {}
    monkeypatch.setattr(pipeline, 'walk_forward_validate', validation)
    monkeypatch.setattr(pipeline, 'write_all_outputs', outputs)
    monkeypatch.setattr(pipeline, 'compute_shap_importance', lambda *args: [])
    monkeypatch.setattr(pipeline, 'print_summary', lambda *args: None)
    pipeline.run_full_pipeline(tmp_path, frame=frame, date_col='period', target_col='measure', test_percentage=20)
    assert seen['selection_last'] < seen['holdout_first']


def test_financial_role_mapping_rejects_same_measure_for_tax_and_cost():
    frame = pd.DataFrame({'a': [12., 13.], 'b': [1., 2.]})
    with pytest.raises(HTTPException):
        main.validate_financial_column_mapping(frame, {'revenue': 'a', 'tax': 'b', 'cogs': 'b'})


def test_negative_return_measure_is_not_silently_discarded():
    result, explanation = main.normalize_loss_driver(pd.Series([-10., -3., 0.]), 'negative')
    assert result.tolist() == [10., 3., 0.]
    assert explanation


def test_forecast_quality_reports_absent_periods_before_imputation():
    frame = pd.DataFrame({'period': pd.to_datetime(['2024-01-01', '2024-03-01', '2024-04-01']),
                          'measure': [10., 30., 40.]})
    series = main.aggregate_sales_series_at_frequency(frame, 'period', 'measure', 'MS')
    quality = main.build_forecast_data_quality(series, 'month')
    assert quality['observed_periods'] == 3
    assert quality['missing_periods'] == 1
    assert any('missing' in item.lower() for item in quality['issues'])


def test_forecast_preparation_preserves_observed_leading_zero():
    frame = pd.DataFrame({'period': pd.date_range('2023-01-01', periods=14, freq='MS'),
                          'measure': [0.] + [float(i) for i in range(1, 14)]})
    cleaned, report = pipeline.clean_data(frame, 'measure', 'period', 'monthly')
    assert len(cleaned) == len(frame)
    assert cleaned.iloc[0]['measure'] == 0.
    assert report['rows_trimmed'] == 0


def test_mixed_return_signs_need_confirmation():
    with pytest.raises(HTTPException):
        main.normalize_loss_driver(pd.Series([-2., 4.]), 'auto')


def test_constant_series_can_have_valid_zero_backtest_error(monkeypatch):
    series = pd.DataFrame({'period': pd.date_range('2022-01-01', periods=24, freq='MS'), 'sales': [4.] * 24})
    result = {'model_type': 'seasonal_naive', 'model_name': 'Seasonal naive', 'status': 'completed',
              'metrics': {'mae': 0., 'rmse': 0., 'mape': 0.}, 'test_forecast': [], 'residual_std': 0.}
    monkeypatch.setattr(main, 'evaluate_seasonal_naive_candidate', lambda *args: result)
    monkeypatch.setattr(main, 'build_future_for_selected_model', lambda *args: [{'period': '2024-01', 'predicted': 4.}])
    selected = main.auto_select_forecast_model(series, 1, 4, 1, [], 'MS', 'month', ['seasonal_naive'])
    assert selected['selected']['metrics']['mae'] == 0.


def test_signed_nonfinancial_forecast_baseline_preserves_negative_values():
    assert main.naive_forecast_step([-7.5]) == -7.5
    lower, upper = main.build_confidence_bounds(-7.5, 1.)
    assert lower < -7.5 < upper


def test_small_measurement_forecast_is_not_rounded_to_zero():
    history = pd.DataFrame({'period': pd.date_range('2023-01-01', periods=12, freq='MS'),
                            'measure': [0.00123] * 12})
    estimator = DummyRegressor(strategy='constant', constant=0.00123).fit([[0.]], [0.00123])
    result = pipeline.forecast_future(estimator, history, ['lag_1'], 'measure', 'period', 1, 'monthly')
    assert result[0]['forecast_value'] == pytest.approx(0.00123)


def test_small_forecast_error_is_not_rounded_to_zero():
    metrics = pipeline.compute_all_metrics([0.0012, 0.0013], [0.0013, 0.0014])
    assert metrics['mae'] == pytest.approx(0.0001)
    assert main.calculate_forecast_metrics([0.0012, 0.0013], [0.0013, 0.0014])['mae'] == pytest.approx(0.0001)


def test_signed_forecasts_do_not_clip_negative_measurements():
    history = pd.DataFrame({'period': pd.date_range('2023-01-01', periods=12, freq='MS'),
                            'measure': [-2.] * 12})
    estimator = DummyRegressor(strategy='constant', constant=-2.).fit([[0.]], [-2.])
    forecast = pipeline.forecast_future(estimator, history, ['lag_1'], 'measure', 'period', 1, 'monthly')
    assert forecast[0]['forecast_value'] == -2.
    assert main.statistical_forecast_step([-5., -6., -7.], 1, 'arima') < 0
