"""Sales-domain regression tests for core forecast column contract and readiness."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from main import DATASET_CACHE, app, set_dataset_cache_entry  # noqa: E402
from sales_domain import (  # noqa: E402
    pick_best_date_column,
    pick_best_revenue_column,
    resolve_sales_columns,
    should_enable_sales_preset,
)


SAMPLES = REPO_ROOT / 'samples' / 'sales'


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def monthly_dataset_id(tmp_path: Path):
    source = SAMPLES / 'sales_monthly.csv'
    assert source.exists(), 'Golden sample missing; generate samples/sales first.'
    frame = pd.read_csv(source)
    dataset_id = 'sales_monthly_test'
    frame_path = tmp_path / f'{dataset_id}.parquet'
    frame.to_parquet(frame_path, index=False)
    set_dataset_cache_entry(dataset_id, {
        'frame_path': str(frame_path),
        'parquet_path': str(frame_path),
        'filename': 'sales_monthly.csv',
        'row_count': int(len(frame)),
        'column_count': int(len(frame.columns)),
        'columns': list(frame.columns),
        'column_info': [
            {'name': 'year_month', 'role': 'datetime'},
            {'name': 'total_total_value_sale_free', 'role': 'numeric'},
            {'name': 'cogs', 'role': 'numeric'},
            {'name': 'region', 'role': 'categorical'},
            {'name': 'category', 'role': 'categorical'},
        ],
    })
    yield dataset_id
    DATASET_CACHE.pop(dataset_id, None)


def test_sales_domain_prefers_revenue_over_cogs():
    columns = ['year_month', 'cogs', 'total_total_value_sale_free', 'region']
    frame = pd.DataFrame({
        'year_month': pd.date_range('2022-01-01', periods=24, freq='MS'),
        'cogs': range(24),
        'total_total_value_sale_free': [1000 + i * 10 for i in range(24)],
        'region': ['North'] * 24,
    })
    date_col = pick_best_date_column(columns)
    target_col = pick_best_revenue_column(columns, frame=frame)
    assert date_col == 'year_month'
    assert target_col == 'total_total_value_sale_free'


def test_should_enable_sales_preset_detection():
    assert should_enable_sales_preset(['sessionId', 'kwhTotal', 'dollars', 'created']) is False
    assert should_enable_sales_preset(['amount', 'total']) is False
    assert should_enable_sales_preset(['year_month', 'total_total_value_sale_free', 'region']) is True
    assert should_enable_sales_preset(['revenue', 'invoice_date']) is True


def test_resolve_sales_columns_respects_explicit_overrides():
    date_col, target_col = resolve_sales_columns(
        ['invoice_date', 'net_sales', 'cogs'],
        date_column='invoice_date',
        target_column='net_sales',
    )
    assert date_col == 'invoice_date'
    assert target_col == 'net_sales'


def test_stationarity_accepts_explicit_columns(client: TestClient, monthly_dataset_id: str):
    response = client.post('/api/ts-forecast/stationarity', json={
        'dataset_id': monthly_dataset_id,
        'date_column': 'year_month',
        'target_column': 'total_total_value_sale_free',
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get('date_column') == 'year_month'
    assert body.get('target_column') == 'total_total_value_sale_free'
    assert 'status' in body


def test_sales_readiness(client: TestClient, monthly_dataset_id: str):
    response = client.post('/api/sales/readiness', json={'dataset_id': monthly_dataset_id})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['period_count'] >= 24
    assert body['target_column'] == 'total_total_value_sale_free'
    assert body['status'] in {'ready', 'needs_attention'}


def test_upload_formats_exist():
    for name in ('sales_monthly.csv', 'sales_monthly.tsv', 'sales_monthly.xlsx', 'sales_monthly.parquet', 'sales_invoices.csv'):
        assert (SAMPLES / name).exists(), name


def test_sales_cleaning_preset(client: TestClient, monthly_dataset_id: str):
    response = client.post('/api/clean-parquet', json={
        'dataset_id': monthly_dataset_id,
        'sales_preset': True,
        'protect_forecast_target': True,
    })
    assert response.status_code == 200, response.text


def test_forecast_periods_accepts_forecast_key():
    from main import forecast_periods_to_frame

    frame = forecast_periods_to_frame(
        [{'period': '2025-01-01', 'forecast': 100.5, 'lower': 90, 'upper': 110}],
        'forecasted_revenue',
    )
    assert len(frame) == 1
    assert float(frame.iloc[0]['forecasted_revenue']) == 100.5


def test_demo_model_artifact_exists():
    demo = BACKEND_DIR / 'models' / 'demo' / 'sales_monthly_demo.joblib'
    meta = BACKEND_DIR / 'models' / 'demo' / 'sales_monthly_demo.json'
    assert demo.exists(), 'Run backend/scripts/train_sales_demo_model.py'
    assert meta.exists()
