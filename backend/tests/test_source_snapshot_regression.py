"""Cleaning must preserve the uploaded snapshot and its ownership metadata."""
import io
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402


@pytest.fixture
def isolated_client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'DATASET_DIR', tmp_path)
    monkeypatch.setattr(main, 'DATASET_CACHE_INDEX_PATH', tmp_path / 'index.json')
    monkeypatch.setattr(main, 'ACTIVITY_DB_AVAILABLE', False)
    monkeypatch.setattr(main, 'record_activity', lambda **kwargs: None)
    monkeypatch.setattr(main, 'DATASET_CACHE', {})
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.mark.parametrize('infer_dtypes', [True, False])
def test_cleaning_keeps_original_upload_and_versions_transformed_copy(isolated_client, infer_dtypes):
    source = pd.DataFrame({'group': ['A', 'B', 'C'], 'measure': [128765.43, -2.19, 0.125]})
    buffer = io.BytesIO()
    source.to_parquet(buffer, index=False)
    headers = {'X-Client-Session-Id': 'synthetic-owner'}
    upload = isolated_client.post('/api/parse-dataset', headers=headers,
                                  files={'file': ('synthetic.parquet', buffer.getvalue(), 'application/octet-stream')},
                                  data={'description': 'Synthetic source'})
    assert upload.status_code == 200, upload.text
    dataset_id = upload.json()['datasetId']
    original = Path(main.DATASET_CACHE[dataset_id]['parquet_path'])
    original_bytes = original.read_bytes()
    version = main.DATASET_CACHE[dataset_id]['dataset_version']
    cleaned = isolated_client.post('/api/clean-dataset', headers=headers, json={
        'dataset_id': dataset_id, 'remove_duplicates': False, 'handle_missing': False,
        'convert_dates': False, 'standardize_names': False, 'infer_dtypes': infer_dtypes,
    })
    assert cleaned.status_code == 200, cleaned.text
    entry = main.DATASET_CACHE[dataset_id]
    assert original.read_bytes() == original_bytes
    assert Path(entry['parquet_path']) != original
    assert entry['dataset_version'] == version + 1
    assert entry['owner_session_id'] == headers['X-Client-Session-Id']
    assert entry['description'] == 'Synthetic source'
    pd.testing.assert_frame_equal(pd.read_parquet(entry['parquet_path']), source)


def test_serialization_preserves_decimal_when_float_would_change_value():
    original = Decimal('123456789.123456789')
    serialized = main.safe_serialize(original)
    assert Decimal(str(serialized)) == original
