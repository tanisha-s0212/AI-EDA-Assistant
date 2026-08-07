"""Usable periods from build_dataset_profile match aggregated series length (Understanding period_count)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from main import (  # noqa: E402
    DATASET_CACHE,
    app,
    build_dataset_profile,
    prepare_sales_series,
    set_dataset_cache_entry,
)


@pytest.fixture()
def client():
    return TestClient(app)


def test_build_dataset_profile_uses_target_column_after_rename():
    frame = pd.DataFrame({
        'order_date': pd.date_range('2014-01-03', periods=100, freq='D'),
        'Sales': [10.0 + i for i in range(100)],
    })
    # Mimic load_ts_dataset rename: period/sales → date/target names
    series = pd.DataFrame({
        'order_date': frame['order_date'],
        'Sales': frame['Sales'],
    })
    profile = build_dataset_profile(series, 'day', value_column='Sales')
    assert profile['usable_periods'] == 100
    assert profile['detected_frequency'] == 'day'
    assert 'volatility' in profile


def test_prepare_sales_series_period_count_matches_profile():
    # Superstore-like daily span long enough to stay on day grain (2 * 365 = 730).
    frame = pd.DataFrame({
        'order_date': pd.date_range('2014-01-03', periods=1458, freq='D'),
        'sales': [100.0 + (i % 50) for i in range(1458)],
    })
    series_frame, _freq, period_label = prepare_sales_series(frame, 'order_date', 'sales')
    profile = build_dataset_profile(series_frame, period_label, value_column='sales')
    assert profile['usable_periods'] == len(series_frame)
    assert profile['usable_periods'] == 1458


def test_stationarity_usable_periods_matches_sales_readiness(client: TestClient, tmp_path: Path):
    frame = pd.DataFrame({
        'order_date': pd.date_range('2014-01-03', periods=1458, freq='D'),
        'sales': [100.0 + (i % 40) for i in range(1458)],
    })
    dataset_id = 'usable_periods_parity'
    frame_path = tmp_path / f'{dataset_id}.parquet'
    frame.to_parquet(frame_path, index=False)
    set_dataset_cache_entry(dataset_id, {
        'frame_path': str(frame_path),
        'parquet_path': str(frame_path),
        'filename': 'superstore_like.csv',
        'row_count': int(len(frame)),
        'column_count': int(len(frame.columns)),
        'columns': list(frame.columns),
        'column_info': [
            {'name': 'order_date', 'role': 'datetime', 'uniqueCount': 1458},
            {'name': 'sales', 'role': 'numeric', 'uniqueCount': 50},
        ],
    })
    fake_stationarity = {
        'status': 'stationary',
        'adf_pvalue': 0.01,
        'kpss_pvalue': 0.1,
        'note': 'mocked',
        'recommended_model': 'SARIMA',
        'differencing_required': False,
    }
    try:
        readiness = client.post('/api/sales/readiness', json={
            'dataset_id': dataset_id,
            'date_column': 'order_date',
            'target_column': 'sales',
        })
        assert readiness.status_code == 200, readiness.text
        period_count = readiness.json()['period_count']

        with patch('main.check_stationarity', return_value=fake_stationarity):
            stationarity = client.post('/api/ts-forecast/stationarity', json={
                'dataset_id': dataset_id,
                'date_column': 'order_date',
                'target_column': 'sales',
            })
        assert stationarity.status_code == 200, stationarity.text
        body = stationarity.json()
        assert body['usable_periods'] == period_count
        assert body['dataset_profile']['usable_periods'] == period_count
        assert period_count == 1458
    finally:
        DATASET_CACHE.pop(dataset_id, None)
