"""Synthetic checks for explicit features and supervised assistant review."""
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402
from ml_execute import select_features  # noqa: E402
from nl_analytics import profile_dataset  # noqa: E402


def test_explicit_features_are_honored_and_direct_duplicate_is_rejected():
    frame = pd.DataFrame({'sensor_a': [1., 3., 4., 7.], 'sensor_b': [2., 4., 6., 8.],
                          'outcome': [2., 5., 7., 9.], 'copy': [2., 5., 7., 9.]})
    profile = profile_dataset(frame)
    selected = select_features(frame, profile, 'outcome', requested_features=['sensor_a'])
    assert selected['features'] == ['sensor_a']
    rejected = select_features(frame, profile, 'outcome', requested_features=['copy'])
    assert not rejected['features'] and 'copy' in rejected['excluded']


def test_assistant_requires_review_before_training(monkeypatch):
    monkeypatch.setattr(main, 'record_activity', lambda **kwargs: None)
    monkeypatch.setattr(main, 'ACTIVITY_DB_AVAILABLE', False)
    monkeypatch.setattr(main, 'train_universal_model', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('trained before review')))
    rows = [{'sensor_a': i, 'sensor_b': i % 3, 'outcome': i * 2 + i % 2} for i in range(30)]
    response = TestClient(main.app, raise_server_exceptions=False).post('/api/assistant/ask', json={
        'data': rows, 'question': 'Predict outcome from sensor_a', 'feature_columns': ['sensor_a']})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['status'] == 'needs_confirmation'
    assert result['ml_plan']['target'] == 'outcome'
    assert result['ml_plan']['features'] == ['sensor_a']


@pytest.mark.parametrize(('question', 'expected_features'), [
    ('Which segment is most likely for a record with this taxable and tax amount?', {'taxable_amt', 'tax_amt'}),
    ('Classify the segment using quantity and gross amount.', {'qty', 'gross_amt'}),
    ('Predict the segment using discount and free quantity.', {'disc_amt', 'free_qty'}),
    ('Assign the segment based on gross, taxable, and tax amounts.', {'gross_amt', 'taxable_amt', 'tax_amt'}),
])
def test_assistant_analyzes_categorical_target_and_natural_feature_terms(monkeypatch, question, expected_features):
    monkeypatch.setattr(main, 'record_activity', lambda **kwargs: None)
    monkeypatch.setattr(main, 'ACTIVITY_DB_AVAILABLE', False)
    rows = [
        {'segment': ['ALPHA', 'BETA', 'GAMMA', 'DELTA'][i % 4],
         'taxable_amt': 100 + i * 10, 'tax_amt': 10 + i * 2,
         'gross_amt': 200 + i * 5, 'qty': 5 + i, 'disc_amt': i % 5,
         'free_qty': i % 3, 'unmentioned_metric': 500 - i}
        for i in range(40)
    ]
    response = TestClient(main.app, raise_server_exceptions=False).post('/api/assistant/ask', json={
        'data': rows,
        'question': question,
    })

    assert response.status_code == 200, response.text
    result = response.json()
    assert result['status'] == 'needs_confirmation', result.get('answer')
    assert result['ml_plan']['target'] == 'segment'
    assert result['ml_plan']['task'] == 'classification'
    assert set(result['ml_plan']['features']) == expected_features


@pytest.mark.parametrize('question', ['Predict outcome from unknown_sensor', 'Predict outcome from sensor and unknown_sensor'])
def test_unknown_named_features_do_not_trigger_automatic_selection(monkeypatch, question):
    monkeypatch.setattr(main, 'record_activity', lambda **kwargs: None)
    monkeypatch.setattr(main, 'ACTIVITY_DB_AVAILABLE', False)
    rows = [{'sensor': i, 'outcome': i * 2 + i % 3} for i in range(20)]
    result = TestClient(main.app, raise_server_exceptions=False).post('/api/assistant/ask', json={
        'data': rows, 'question': question}).json()
    assert result['status'] in ('unsupported', 'clarify')


def test_high_correlation_is_a_warning_not_automatic_exclusion():
    frame = pd.DataFrame({'sensor': [1., 2., 3., 4., 5.], 'outcome': [2., 4., 6., 8., 10.]})
    result = select_features(frame, profile_dataset(frame), 'outcome', requested_features=['sensor'])
    assert result['features'] == ['sensor']
    assert result['warnings']


def test_missing_values_do_not_make_a_feature_an_exact_target_copy():
    frame = pd.DataFrame({'sensor': pd.Series([1., None, 3., 4., 5.], dtype='Float64'), 'outcome': [1., 2., 3., 4., 5.]})
    result = select_features(frame, profile_dataset(frame), 'outcome', requested_features=['sensor'])
    assert result['features'] == ['sensor']


def test_target_components_are_reported_as_possible_leakage():
    frame = pd.DataFrame({'a': [1., 2., 6., 3., 8.], 'b': [4., 8., 3., 1., 6.]})
    frame['outcome'] = frame['a'] + frame['b']
    result = select_features(frame, profile_dataset(frame), 'outcome', requested_features=['a', 'b'])
    assert result['features'] == ['a', 'b']
    assert any('derived' in warning.lower() for warning in result['warnings'])


def test_direct_training_rejects_target_as_input(monkeypatch):
    monkeypatch.setattr(main, 'record_activity', lambda **kwargs: None)
    monkeypatch.setattr(main, 'ACTIVITY_DB_AVAILABLE', False)
    rows = [{'x': i, 'outcome': i * 2} for i in range(12)]
    response = TestClient(main.app, raise_server_exceptions=False).post('/api/train', json={
        'data': rows, 'target_column': 'outcome', 'feature_columns': ['x', 'outcome'],
        'problem_type': 'regression', 'model_type': 'ridge_regression'})
    assert response.status_code == 422


def test_direct_training_rejects_an_exact_copy_of_target(monkeypatch):
    monkeypatch.setattr(main, 'record_activity', lambda **kwargs: None)
    monkeypatch.setattr(main, 'ACTIVITY_DB_AVAILABLE', False)
    rows = [{'x': i * 3, 'outcome': i * 3} for i in range(12)]
    response = TestClient(main.app, raise_server_exceptions=False).post('/api/train', json={
        'data': rows, 'target_column': 'outcome', 'feature_columns': ['x'],
        'problem_type': 'regression', 'model_type': 'ridge_regression'})
    assert response.status_code == 422


def test_partial_prediction_inputs_are_not_silently_defaulted(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'record_activity', lambda **kwargs: None)
    monkeypatch.setattr(main, 'ACTIVITY_DB_AVAILABLE', False)
    monkeypatch.setattr(main, 'MODEL_DIR', tmp_path)
    rows = [{'x': i, 'z': i % 3, 'outcome': i * 2 + i % 2} for i in range(30)]
    payload = {'data': rows, 'question': 'Predict outcome where x = 3', 'feature_columns': ['x', 'z']}
    client = TestClient(main.app, raise_server_exceptions=False)
    review = client.post('/api/assistant/ask', json=payload).json()
    assert review['status'] == 'needs_confirmation'
    result = client.post('/api/assistant/ask', json={**payload, 'confirm_training': True,
                         'review_token': review['ml_plan']['review_token']}).json()
    assert result['status'] == 'success'
    assert result.get('single_prediction') is None
    assert 'z' in result.get('single_prediction_issue', '')


def test_review_token_cannot_be_reused_after_data_changes(monkeypatch):
    monkeypatch.setattr(main, 'record_activity', lambda **kwargs: None)
    monkeypatch.setattr(main, 'ACTIVITY_DB_AVAILABLE', False)
    monkeypatch.setattr(main, 'train_universal_model', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('stale review trained')))
    client = TestClient(main.app, raise_server_exceptions=False)
    rows = [{'x': i, 'outcome': i * 2 + i % 3} for i in range(30)]
    payload = {'data': rows, 'question': 'Predict outcome from x', 'feature_columns': ['x']}
    review = client.post('/api/assistant/ask', json=payload).json()
    changed = [*rows, {'x': 50, 'outcome': 101}]
    result = client.post('/api/assistant/ask', json={**payload, 'data': changed, 'confirm_training': True,
                         'review_token': review['ml_plan']['review_token']}).json()
    assert result['status'] == 'needs_confirmation'


def test_chronological_holdout_keeps_equal_dates_together():
    dates = pd.to_datetime(['2024-01-01'] * 3 + ['2024-02-01'] * 3 + ['2024-03-01'] * 3)
    split = main.chronological_split_position(pd.Series(dates), 0.2)
    assert split == 6
    assert dates[split - 1] < dates[split]


def test_training_holds_out_complete_periods(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'record_activity', lambda **kwargs: None)
    monkeypatch.setattr(main, 'ACTIVITY_DB_AVAILABLE', False)
    monkeypatch.setattr(main, 'MODEL_DIR', tmp_path)
    rows = [{'period': f'2024-0{month}-01', 'x': month * 10 + item,
             'outcome': month * 20 + item} for month in range(1, 4) for item in range(4)]
    response = TestClient(main.app, raise_server_exceptions=False).post('/api/train', json={
        'data': rows, 'target_column': 'outcome', 'feature_columns': ['x'], 'time_order_column': 'period',
        'problem_type': 'regression', 'model_type': 'ridge_regression', 'test_size': 0.2})
    assert response.status_code == 200, response.text
    assert response.json()['status'] == 'success'
    assert response.json()['optimization']['cv_rows_evaluated'] == 4


def test_manual_training_feature_review_warns_about_derived_target(monkeypatch):
    monkeypatch.setattr(main, 'ACTIVITY_DB_AVAILABLE', False)
    rows = [{'a': a, 'b': b, 'outcome': a + b} for a, b in [(1, 4), (2, 8), (6, 3), (3, 1), (8, 6)]]
    result = TestClient(main.app, raise_server_exceptions=False).post('/api/ml/feature-review', json={
        'data': rows, 'target_column': 'outcome', 'feature_columns': ['a', 'b']})
    assert result.status_code == 200
    assert result.json()['features'] == ['a', 'b']
    assert any('derived' in warning.lower() for warning in result.json()['warnings'])
