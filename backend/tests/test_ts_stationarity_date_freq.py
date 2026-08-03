"""Regression tests for TS date pick, frequency coarsening, and 422 status preservation."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from main import (  # noqa: E402
    DATASET_CACHE,
    app,
    load_ts_dataset,
    prepare_sales_series,
    set_dataset_cache_entry,
)


@pytest.fixture()
def client():
    return TestClient(app)


def test_prepare_sales_series_auto_coarsens_short_daily_span():
    # ~600 daily points fails 2y day/week/month/quarter mins after reindex, but YS yields 2 years.
    frame = pd.DataFrame({
        'created': pd.date_range('2023-01-01', periods=600, freq='D'),
        'dollars': [10.0 + i for i in range(600)],
    })
    series_frame, freq, period_label = prepare_sales_series(frame, 'created', 'dollars')
    assert period_label == 'year'
    assert freq == 'YS'
    assert series_frame.attrs.get('frequency_auto_adjusted') is True
    assert series_frame.attrs.get('inferred_period_label') == 'day'
    assert len(series_frame) >= 2


def test_prepare_sales_series_raises_422_when_even_yearly_is_too_short():
    frame = pd.DataFrame({
        'created': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'dollars': [1.0, 2.0, 3.0],
    })
    with pytest.raises(HTTPException) as exc_info:
        prepare_sales_series(frame, 'created', 'dollars')
    assert exc_info.value.status_code == 422


def test_stationarity_endpoint_preserves_422(client: TestClient, tmp_path: Path):
    frame = pd.DataFrame({
        'created': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'month': [1, 1, 1],
        'dollars': [1.0, 2.0, 3.0],
    })
    dataset_id = 'ts_short_span_422'
    frame_path = tmp_path / f'{dataset_id}.parquet'
    frame.to_parquet(frame_path, index=False)
    set_dataset_cache_entry(dataset_id, {
        'frame_path': str(frame_path),
        'parquet_path': str(frame_path),
        'filename': 'short.csv',
        'row_count': int(len(frame)),
        'column_count': int(len(frame.columns)),
        'columns': list(frame.columns),
        'column_info': [
            {'name': 'created', 'role': 'datetime', 'uniqueCount': 3},
            {'name': 'month', 'role': 'numeric', 'uniqueCount': 1},
            {'name': 'dollars', 'role': 'numeric', 'uniqueCount': 3},
        ],
    })
    try:
        response = client.post('/api/ts-forecast/stationarity', json={
            'dataset_id': dataset_id,
            'date_column': 'created',
            'target_column': 'dollars',
        })
        assert response.status_code == 422, response.text
        body = response.json()
        detail = body.get('error') or body.get('detail') or ''
        assert 'Forecasting needs at least' in str(detail)
    finally:
        DATASET_CACHE.pop(dataset_id, None)


def test_load_ts_auto_picks_created_not_month(tmp_path: Path):
    frame = pd.DataFrame({
        'created': pd.date_range('2020-01-01', periods=800, freq='D'),
        'year': [2020 + (i // 365) for i in range(800)],
        'month': [((i % 365) // 30) + 1 for i in range(800)],
        'day': [(i % 28) + 1 for i in range(800)],
        'dayofweek': [i % 7 for i in range(800)],
        'date': pd.date_range('2020-01-01', periods=800, freq='D').date,
        'dollars': [5.0 + (i % 17) for i in range(800)],
    })
    dataset_id = 'ts_created_vs_month'
    frame_path = tmp_path / f'{dataset_id}.parquet'
    frame.to_parquet(frame_path, index=False)
    set_dataset_cache_entry(dataset_id, {
        'frame_path': str(frame_path),
        'parquet_path': str(frame_path),
        'filename': 'station.csv',
        'row_count': int(len(frame)),
        'column_count': int(len(frame.columns)),
        'columns': list(frame.columns),
        'column_info': [
            {'name': 'created', 'role': 'datetime', 'uniqueCount': 800},
            {'name': 'month', 'role': 'numeric', 'uniqueCount': 12},
            {'name': 'dayofweek', 'role': 'numeric', 'uniqueCount': 7},
            {'name': 'date', 'role': 'date', 'uniqueCount': 800},
            {'name': 'dollars', 'role': 'numeric', 'uniqueCount': 17},
        ],
    })
    try:
        _df, date_col, target_col = load_ts_dataset(dataset_id)
        assert date_col == 'created'
        assert target_col == 'dollars'

        # Explicit/stale date-part override must be ignored by resolve_sales_columns.
        _df2, date_col2, target_col2 = load_ts_dataset(dataset_id, date_column='month', target_column='dollars')
        assert date_col2 == 'created'
        assert target_col2 == 'dollars'
    finally:
        DATASET_CACHE.pop(dataset_id, None)


def test_stationarity_endpoint_surfaces_frequency_meta(client: TestClient, tmp_path: Path):
    frame = pd.DataFrame({
        'created': pd.date_range('2023-01-01', periods=600, freq='D'),
        'dollars': [10.0 + i for i in range(600)],
    })
    dataset_id = 'ts_freq_meta'
    frame_path = tmp_path / f'{dataset_id}.parquet'
    frame.to_parquet(frame_path, index=False)
    set_dataset_cache_entry(dataset_id, {
        'frame_path': str(frame_path),
        'parquet_path': str(frame_path),
        'filename': 'mid.csv',
        'row_count': int(len(frame)),
        'column_count': int(len(frame.columns)),
        'columns': list(frame.columns),
        'column_info': [
            {'name': 'created', 'role': 'datetime', 'uniqueCount': 600},
            {'name': 'dollars', 'role': 'numeric', 'uniqueCount': 600},
        ],
    })
    try:
        with patch('main.check_stationarity', return_value={
            'status': 'stationary',
            'adf_pvalue': 0.01,
            'kpss_pvalue': 0.1,
            'note': 'ok',
            'recommended_model': 'SARIMA',
            'differencing_required': False,
        }):
            response = client.post('/api/ts-forecast/stationarity', json={
                'dataset_id': dataset_id,
                'date_column': 'created',
                'target_column': 'dollars',
            })
        assert response.status_code == 200, response.text
        body = response.json()
        assert body.get('date_column') == 'created'
        assert body.get('frequency_auto_adjusted') is True
        assert body.get('period_label') == 'year'
        assert body.get('inferred_period_label') == 'day'
    finally:
        DATASET_CACHE.pop(dataset_id, None)
