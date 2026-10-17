"""Prediction contracts: decimal outputs, strict inputs, class order, and file parity."""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from prediction_validation import read_prediction_table


class DecimalModel:
    def predict(self, values):
        return np.asarray([12.3456] * len(values))


@pytest.fixture
def prediction_client(monkeypatch):
    bundle = {
        'model': DecimalModel(), 'problem_type': 'regression', 'owner_session_id': 'prediction-contract',
        'feature_columns': ['amount', 'zone'], 'target_column': 'result', 'validation_rmse': 2.0,
        'preprocessing': {
            'raw_feature_columns': ['amount', 'zone'], 'numeric_features': ['amount'],
            'categorical_features': ['zone'], 'encoded_feature_names': ['amount', 'zone_north'],
            'categories': {'zone': ['north']}, 'imputers': {'numeric': {'amount': 0}},
            'numeric_ranges': {'amount': {'min': 0, 'max': 10}}, 'scaler': None,
        },
    }
    monkeypatch.setattr(main, 'load_model_bundle', lambda _id: bundle)
    return TestClient(main.app), bundle


HEADERS = {'X-Client-Session-Id': 'prediction-contract'}


def test_decimal_output_and_distribution_warnings(prediction_client):
    client, _ = prediction_client
    response = client.post('/api/predict', headers=HEADERS, json={
        'model_id': 'decimal', 'features': {'amount': 15, 'zone': 'new-zone'},
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['prediction'] == pytest.approx(12.3456)
    assert len(result['warnings']) == 2
    assert result['inputs'] == {'amount': 15.0, 'zone': 'new-zone'}
    assert result['readiness']['status'] == 'caution'
    assert 'calibrated' in result['error_band_note']


@pytest.mark.parametrize('value', ['12oops', 'Infinity', 'NaN', '', None, [1], {'a': 1}])
def test_invalid_numeric_values_are_rejected(prediction_client, value):
    client, _ = prediction_client
    response = client.post('/api/predict', headers=HEADERS, json={
        'model_id': 'decimal', 'features': {'amount': value, 'zone': 'north'},
    })
    assert response.status_code == 422, response.text
    assert response.json()['status'] == 'failed'


def test_schema_and_batch_require_authorization(prediction_client):
    client, _ = prediction_client
    assert client.get('/api/prediction/schema/decimal').status_code == 403
    assert client.post('/api/predict/batch', data={'model_id': 'decimal'}, files={
        'file': ('inputs.csv', b'amount,zone\n1,north\n', 'text/csv'),
    }).status_code == 403
    schema = client.get('/api/prediction/schema/decimal', headers=HEADERS).json()
    assert [field['name'] for field in schema['fields']] == ['amount', 'zone']
    assert schema['fields'][0]['range'] == {'min': 0, 'max': 10}


@pytest.mark.parametrize('extension', ['csv', 'tsv', 'xlsx', 'parquet'])
def test_batch_file_formats_keep_rejected_rows_and_quoted_categories(prediction_client, extension):
    client, _ = prediction_client
    frame = pd.DataFrame({'amount': ['1.5', 'bad', '2'], 'zone': ['north', 'north', 'south, east']})
    stream = io.BytesIO()
    if extension in ('csv', 'tsv'):
        content = frame.to_csv(index=False, sep='\t' if extension == 'tsv' else ',').encode()
    else:
        if extension == 'xlsx':
            frame.to_excel(stream, index=False)
        else:
            frame.to_parquet(stream, index=False)
        content = stream.getvalue()
    response = client.post('/api/predict/batch', headers=HEADERS, data={'model_id': 'decimal'}, files={
        'file': (f'inputs.{extension}', content, 'application/octet-stream'),
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['summary'] == {'total': 3, 'succeeded': 2, 'failed': 1, 'with_warnings': 1}
    assert [row['row'] for row in result['results']] == [1, 2, 3]
    assert result['results'][1]['status'] == 'failed'
    assert result['results'][2]['inputs']['zone'] == 'south, east'


def test_batch_excel_requires_explicit_sheet_and_preserves_leading_zero_categories():
    stream = io.BytesIO()
    with pd.ExcelWriter(stream) as writer:
        pd.DataFrame({'code': ['0012']}).to_excel(writer, sheet_name='First', index=False)
        pd.DataFrame({'code': ['0013']}).to_excel(writer, sheet_name='Second', index=False)
    with pytest.raises(ValueError, match='Choose a worksheet'):
        read_prediction_table(stream.getvalue(), 'inputs.xlsx')
    assert read_prediction_table(stream.getvalue(), 'inputs.xlsx', 'Second').iloc[0]['code'] == '0013'
    assert read_prediction_table(b'code\n0012\n', 'inputs.csv').iloc[0]['code'] == '0012'


def test_delimited_parser_rejects_shifted_columns_and_keeps_blank_data_rows():
    with pytest.raises(ValueError, match='Data row 1 has more values'):
        read_prediction_table(b'amount,zone\n1,north,extra\n', 'inputs.csv')
    parsed = read_prediction_table(b'amount,zone\n1,north\n\n2\n', 'inputs.csv')
    assert len(parsed) == 3
    assert parsed.iloc[1].tolist() == ['', '']
    assert parsed.iloc[2].tolist() == ['2', '']
    with pytest.raises(ValueError, match='at most 1'):
        read_prediction_table(b'amount\n1\n2\n', 'inputs.csv', limit=1)


def test_classifier_uses_actual_estimator_class_order(prediction_client):
    client, bundle = prediction_client

    class PartialClassifier:
        classes_ = np.asarray([0, 2])

        def predict(self, _values):
            return np.asarray([2])

        def predict_proba(self, _values):
            return np.asarray([[0.1, 0.9]])

    bundle.update(model=PartialClassifier(), problem_type='classification',
                  label_encoder=LabelEncoder().fit(['alpha', 'beta', 'gamma']))
    response = client.post('/api/predict', headers=HEADERS, json={
        'model_id': 'classifier', 'features': {'amount': 3, 'zone': 'north'},
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['prediction_label'] == 'gamma'
    assert result['probabilities'] == {'alpha': 0.1, 'gamma': 0.9}
    assert 'calibration' in result['probability_note'].lower()


def test_newly_trained_model_schema_uses_raw_features_and_training_bounds(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'MODEL_DIR', tmp_path)
    client = TestClient(main.app)
    training = client.post('/api/train', headers=HEADERS, json={
        'data': [{'amount': index * 0.1, 'zone': 'north' if index % 2 else 'south', 'target': index * 0.25} for index in range(40)],
        'target_column': 'target', 'feature_columns': ['amount', 'zone'],
        'model_type': 'ridge_regression', 'problem_type': 'regression', 'training_mode': 'fast',
    })
    assert training.status_code == 200 and training.json()['status'] == 'success', training.text
    model_id = training.json()['model_id']
    schema = client.get(f'/api/prediction/schema/{model_id}', headers=HEADERS).json()
    assert [field['name'] for field in schema['fields']] == ['amount', 'zone']
    assert schema['fields'][1]['categories'] == ['north', 'south']
    assert schema['readiness']['rows'] > 0
    assert schema['fields'][0]['range']['max'] == pytest.approx(3.9)
    main.MODEL_CACHE.pop(model_id, None)


def test_models_on_same_dataset_keep_distinct_predictions_and_batch_rechecks_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'DATASET_DIR', tmp_path / 'datasets')
    monkeypatch.setattr(main, 'DATASET_CACHE_INDEX_PATH', tmp_path / 'datasets' / 'index.json')
    monkeypatch.setattr(main, 'MODEL_DIR', tmp_path / 'models')
    main.DATASET_DIR.mkdir()
    main.MODEL_DIR.mkdir()
    client = TestClient(main.app)
    cached = client.post('/api/cache-dataset', headers=HEADERS, json={
        'file_name': 'training.csv',
        'data': [{'signal': index, 'target': 2 * index + .5, 'other': -3 * index + 200} for index in range(40)],
    })
    assert cached.status_code == 200, cached.text
    dataset_id = cached.json()['datasetId']
    identifiers = []
    try:
        for target in ['target', 'other']:
            trained = client.post('/api/train', headers=HEADERS, json={
                'dataset_id': dataset_id, 'target_column': target, 'feature_columns': ['signal'],
                'model_type': 'ridge_regression', 'problem_type': 'regression', 'training_mode': 'fast',
            })
            assert trained.status_code == 200 and trained.json()['status'] == 'success', trained.text
            identifiers.append(trained.json()['model_id'])
        assert identifiers[0] != identifiers[1]
        scores = [client.post('/api/predict', headers=HEADERS, json={
            'model_id': model_id, 'dataset_id': dataset_id, 'features': {'signal': 10},
        }).json()['prediction'] for model_id in identifiers]
        assert scores[0] == pytest.approx(20.5, abs=1)
        assert scores[1] == pytest.approx(170, abs=1)
        original_reader = main.read_prediction_table

        def replace_during_parse(*args, **kwargs):
            frame = original_reader(*args, **kwargs)
            main.set_dataset_cache_entry(dataset_id, dict(main.DATASET_CACHE[dataset_id]))
            return frame

        monkeypatch.setattr(main, 'read_prediction_table', replace_during_parse)
        response = client.post('/api/predict/batch', headers=HEADERS,
                               data={'model_id': identifiers[0], 'dataset_id': dataset_id},
                               files={'file': ('inputs.csv', b'signal\n10\n', 'text/csv')})
        assert response.status_code == 409, response.text
    finally:
        main.DATASET_CACHE.pop(dataset_id, None)
        for model_id in identifiers:
            main.MODEL_CACHE.pop(model_id, None)
