"""Regression tests for Data Cleaning date corruption and median imputation identity."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from dtype_inference import infer_universal_dtypes  # noqa: E402
from main import (  # noqa: E402
    DATASET_CACHE,
    app,
    repair_zero_padded_century_timestamp,
    set_dataset_cache_entry,
    try_parse_datetime_series,
)


@pytest.fixture()
def client():
    return TestClient(app)


def _cache_frame(tmp_path: Path, dataset_id: str, frame: pd.DataFrame, filename: str = 'fixture.csv') -> str:
    frame_path = tmp_path / f'{dataset_id}.joblib'
    import joblib

    joblib.dump(frame, frame_path)
    set_dataset_cache_entry(dataset_id, {
        'frame_path': str(frame_path),
        'filename': filename,
        'row_count': int(len(frame)),
        'column_count': int(len(frame.columns)),
        'columns': list(frame.columns),
        'duplicate_count': 0,
    })
    return dataset_id


@pytest.fixture()
def truncated_year_dataset(tmp_path: Path):
    frame = pd.DataFrame({
        'created': [
            '0014-11-18 15:40:26',
            '0015-10-04 12:44:59',
            '0014-11-18 16:45:00',
            '0015-10-04 08:00:00',
        ],
        'ended': [
            '0014-11-18 18:40:26',
            '0015-10-04 14:44:59',
            '0014-11-18 19:45:00',
            '0015-10-04 11:00:00',
        ],
        'metric': [10.0, 20.0, None, 20.0],
        'code': ['0014', '0015', '0014', '99'],
    })
    dataset_id = 'cleaning_dates_truncated_year'
    _cache_frame(tmp_path, dataset_id, frame)
    yield dataset_id
    DATASET_CACHE.pop(dataset_id, None)


def test_century_repair_helper_maps_00xx_to_20xx():
    assert repair_zero_padded_century_timestamp('0014-11-18 15:40:26') == '2014-11-18 15:40:26'
    assert repair_zero_padded_century_timestamp('0015-10-04 12:44:59') == '2015-10-04 12:44:59'
    assert repair_zero_padded_century_timestamp('2014-11-18 15:40:26') == '2014-11-18 15:40:26'
    assert repair_zero_padded_century_timestamp('0014') == '0014'


def test_try_parse_repairs_truncated_years():
    series = pd.Series(['0014-11-18 15:40:26', '0015-10-04 12:44:59'])
    parsed = try_parse_datetime_series(series)
    assert parsed is not None
    assert list(parsed.dt.year) == [2014, 2015]
    assert list(parsed.dt.hour) == [15, 12]


def test_clean_dataset_century_repair_and_time_preservation(client: TestClient, truncated_year_dataset: str):
    response = client.post('/api/clean-dataset', json={
        'dataset_id': truncated_year_dataset,
        'remove_duplicates': False,
        'handle_missing': True,
        'convert_dates': True,
        'standardize_names': False,
        'infer_dtypes': True,
        'sales_preset': False,
    })
    assert response.status_code == 200, response.text
    body = response.json()
    rows = body['data']
    assert len(rows) == 4

    created_years = {str(row['created'])[:4] for row in rows}
    ended_years = {str(row['ended'])[:4] for row in rows}
    assert created_years == {'2014', '2015'}
    assert ended_years == {'2014', '2015'}
    assert not any(year in created_years for year in ('2001', '2004', '2018', '2031'))

    for row in rows:
        created_raw = str(row['created'])
        ended_raw = str(row['ended'])
        # Excel-safe export: no ISO "T" separator (Excel often reopens those as midnight).
        assert 'T' not in created_raw
        assert 'T' not in ended_raw
        assert '00:00:00' not in created_raw
        assert '00:00:00' not in ended_raw
        created = pd.Timestamp(created_raw)
        ended = pd.Timestamp(ended_raw)
        assert created != ended
        delta_hours = (ended - created).total_seconds() / 3600.0
        assert 1.0 <= delta_hours <= 5.0
        assert created.hour != 0 or created.minute != 0 or created.second != 0


def test_clean_dataset_api_preserves_wall_clock_without_iso_t(client: TestClient, truncated_year_dataset: str):
    """Downloaded/preview payload must keep session times as space-separated strings."""
    response = client.post('/api/clean-dataset', json={
        'dataset_id': truncated_year_dataset,
        'remove_duplicates': False,
        'handle_missing': False,
        'convert_dates': True,
        'standardize_names': False,
        'infer_dtypes': True,
        'sales_preset': False,
    })
    assert response.status_code == 200, response.text
    first = response.json()['data'][0]
    assert first['created'] == '2014-11-18 15:40:26'
    assert first['ended'] == '2014-11-18 18:40:26'

    audit = (response.json().get('dtypeInference') or {}).get('audit') or []
    created_log = next(item for item in audit if item.get('column') == 'created')
    assert created_log.get('datetime_format') == '%Y-%m-%d %H:%M:%S'


def test_dtype_inference_time_first_does_not_truncate_to_midnight():
    frame = pd.DataFrame({
        'created': [
            '2014-11-18 15:40:26',
            '2014-11-19 17:40:26',
            '2014-11-21 12:05:46',
        ]
    })
    inferred, cast_log = infer_universal_dtypes(frame)
    created_log = next(item for item in cast_log if item['column'] == 'created')
    assert created_log.get('accepted') is True
    assert created_log.get('datetime_format') == '%Y-%m-%d %H:%M:%S'
    hours = list(pd.to_datetime(inferred['created']).dt.hour)
    assert hours == [15, 17, 12]


def test_non_date_column_with_0014_token_not_century_rewritten(client: TestClient, truncated_year_dataset: str):
    response = client.post('/api/clean-dataset', json={
        'dataset_id': truncated_year_dataset,
        'remove_duplicates': False,
        'handle_missing': False,
        'convert_dates': True,
        'standardize_names': False,
        # Keep dtype inference off so we isolate convert_dates (no integer zero-stripping).
        'infer_dtypes': False,
        'sales_preset': False,
    })
    assert response.status_code == 200, response.text
    codes = [str(row['code']) for row in response.json()['data']]
    assert codes == ['0014', '0015', '0014', '99']
    # Must not have been rewritten into 20xx calendar dates.
    assert all(not re.match(r'^\d{4}-\d{2}-\d{2}', code) for code in codes)


def test_dtype_inference_does_not_scramble_short_ambiguous_dates():
    frame = pd.DataFrame({'stamp': ['14-11-18', '15-10-04', '14-01-01', '15-12-31']})
    inferred, cast_log = infer_universal_dtypes(frame)
    stamp_log = next(item for item in cast_log if item['column'] == 'stamp')
    assert stamp_log.get('cast_type') != 'datetime'
    # Values must remain unchanged strings — not dayfirst-scrambled timestamps.
    assert list(inferred['stamp'].astype(str)) == ['14-11-18', '15-10-04', '14-01-01', '15-12-31']


def test_median_imputation_does_not_overwrite_non_nulls(client: TestClient, tmp_path: Path):
    frame = pd.DataFrame({
        'metric': [10.0, 20.0, None, 20.0, 30.0, None, 20.0, 40.0, 50.0, None],
        'label': ['a'] * 10,
    })
    original_nan_count = int(frame['metric'].isna().sum())
    median = float(frame['metric'].median())
    pre_existing_median_count = int(((frame['metric'] == median) & frame['metric'].notna()).sum())

    dataset_id = 'cleaning_median_identity'
    _cache_frame(tmp_path, dataset_id, frame)
    try:
        response = client.post('/api/clean-dataset', json={
            'dataset_id': dataset_id,
            'remove_duplicates': False,
            'handle_missing': True,
            'convert_dates': False,
            'standardize_names': False,
            'infer_dtypes': False,
            'sales_preset': False,
        })
        assert response.status_code == 200, response.text
        values = [float(row['metric']) for row in response.json()['data']]
        median_count = sum(1 for value in values if value == median)
        assert median_count == original_nan_count + pre_existing_median_count

        detail = next(
            log['detail'] for log in response.json()['logs'] if log['action'] == 'Handled Missing Values'
        )
        assert 'strategy=median' in detail
        assert f'filled={original_nan_count}' in detail
        assert f'pre_existing_equal={pre_existing_median_count}' in detail
    finally:
        DATASET_CACHE.pop(dataset_id, None)
