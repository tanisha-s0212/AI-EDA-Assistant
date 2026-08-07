"""Profit illustrative_assumptions comes from resolver is_fallback bools, not audit substrings."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from main import (  # noqa: E402
    DATASET_CACHE,
    app,
    ensure_session_state,
    resolve_cogs_series,
    resolve_operating_expense_series,
    set_dataset_cache_entry,
)


@pytest.fixture()
def client():
    return TestClient(app)


def _seed_session(session_id: str, frame: pd.DataFrame, future_revenue: float = 2800.0) -> None:
    tmpdir = Path(tempfile.mkdtemp())
    pq = tmpdir / f'{session_id}.parquet'
    frame.to_parquet(pq, index=False)
    set_dataset_cache_entry(session_id, {
        'frame_path': str(pq),
        'parquet_path': str(pq),
        'filename': f'{session_id}.csv',
        'row_count': int(len(frame)),
        'column_count': int(len(frame.columns)),
        'columns': list(frame.columns),
        'column_info': [
            {
                'name': column,
                'role': 'datetime' if 'date' in column else (
                    'numeric' if pd.api.types.is_numeric_dtype(frame[column]) else 'categorical'
                ),
            }
            for column in frame.columns
        ],
    })
    state = ensure_session_state(session_id)
    periods = pd.date_range('2019-01-01', periods=5, freq='D')
    futures = [
        {'period': str(period.date()), 'predicted': future_revenue * (1 + 0.01 * index)}
        for index, period in enumerate(periods)
    ]
    state['time_series_result'] = {
        'date_column': 'order_date',
        'target_column': 'sales',
        'period_label': 'day',
        'future_forecast': futures,
    }
    state['ml_forecast_result'] = {
        'date_column': 'order_date',
        'target_column': 'sales',
        'period_label': 'day',
        'future_forecast': futures,
    }
    # Prior confirmed loss rows — mirrors Superstore UI where Loss runs before Profit.
    state['loss_forecast_result'] = [
        {
            'period': str(period.date()),
            'total_loss': round(future_revenue * 0.08, 2),
            'revenue_loss': 1.0,
            'operational_loss': 1.0,
            'inventory_loss': 1.0,
            'discount_loss': 1.0,
        }
        for period in periods
    ]


def test_resolve_cogs_is_fallback_true_for_sales_only():
    frame = pd.DataFrame({
        'order_date': pd.date_range('2018-01-01', periods=20, freq='D'),
        'sales': [100.0 + i for i in range(20)],
    })
    _series, _source, ratio, is_fallback = resolve_cogs_series(frame, 'sales')
    assert ratio == 0.60
    assert is_fallback is True


def test_resolve_cogs_is_fallback_false_when_cogs_mapped():
    frame = pd.DataFrame({
        'order_date': pd.date_range('2018-01-01', periods=20, freq='D'),
        'sales': [100.0] * 20,
        'cogs': [55.0] * 20,
    })
    _series, source, _ratio, is_fallback = resolve_cogs_series(frame, 'sales')
    assert is_fallback is False
    assert 'mapped cost' in source


def test_resolve_opex_is_fallback_false_when_mapped():
    frame = pd.DataFrame({
        'order_date': pd.date_range('2018-01-01', periods=20, freq='D'),
        'sales': [100.0] * 20,
        'operating_expense': [10.0] * 20,
    })
    _series, source, _ratio, is_fallback = resolve_operating_expense_series(frame, 'sales')
    assert is_fallback is False
    assert 'mapped operating' in source


def test_profit_illustrative_true_for_sales_only_with_prior_loss(client: TestClient):
    """Original Superstore false-negative: reused loss rows + thin loss_audit must still be illustrative."""
    session_id = 'profit_illustrative_sales_only'
    frame = pd.DataFrame({
        'order_date': pd.date_range('2018-01-01', periods=40, freq='D'),
        'sales': [120.0 + (i % 10) for i in range(40)],
    })
    try:
        _seed_session(session_id, frame)
        response = client.post('/api/profit-forecast/run', json={
            'session_id': session_id,
            'forecast_periods': 5,
            'confirmed_assumptions': True,
        })
        assert response.status_code == 200, response.text
        body = response.json()
        audit_text = ' '.join(str(note) for note in (body.get('assumptions_audit') or []))
        assert 'Existing confirmed loss forecast rows reused' in audit_text
        assert body['illustrative_assumptions'] is True
        assert body['assumption_mode'] == 'illustrative'
        assert body['scenarios']['baseline']
    finally:
        DATASET_CACHE.pop(session_id, None)


def test_profit_illustrative_false_when_cost_and_opex_mapped(client: TestClient):
    """Negative regression: mapped cost drivers must not force illustrative=True."""
    session_id = 'profit_illustrative_mapped_costs'
    frame = pd.DataFrame({
        'order_date': pd.date_range('2018-01-01', periods=40, freq='D'),
        'sales': [200.0] * 40,
        'cogs': [110.0] * 40,
        'operating_expense': [20.0] * 40,
        'discount': [5.0] * 40,
        'inventory_value': [40.0] * 40,
    })
    try:
        _seed_session(session_id, frame, future_revenue=3000.0)
        _cogs, _cs, _cr, cogs_fb = resolve_cogs_series(frame, 'sales')
        _opex, _os, _or, opex_fb = resolve_operating_expense_series(frame, 'sales')
        assert cogs_fb is False
        assert opex_fb is False

        response = client.post('/api/profit-forecast/run', json={
            'session_id': session_id,
            'forecast_periods': 5,
            'confirmed_assumptions': True,
        })
        assert response.status_code == 200, response.text
        body = response.json()
        assert body['illustrative_assumptions'] is False
        assert body['assumption_mode'] == 'mapped'
    finally:
        DATASET_CACHE.pop(session_id, None)


def test_profit_illustrative_true_when_only_opex_fallback(client: TestClient):
    """Partial fallback: mapped COGS + hard OpEx fallback → illustrative True; audit shows both."""
    session_id = 'profit_illustrative_opex_only_fallback'
    frame = pd.DataFrame({
        'order_date': pd.date_range('2018-01-01', periods=40, freq='D'),
        'sales': [200.0] * 40,
        'cogs': [110.0] * 40,
    })
    try:
        _seed_session(session_id, frame)
        _cogs, _cs, _cr, cogs_fb = resolve_cogs_series(frame, 'sales')
        _opex, _os, _or, opex_fb = resolve_operating_expense_series(frame, 'sales')
        assert cogs_fb is False
        assert opex_fb is True

        response = client.post('/api/profit-forecast/run', json={
            'session_id': session_id,
            'forecast_periods': 5,
            'confirmed_assumptions': True,
        })
        assert response.status_code == 200, response.text
        body = response.json()
        audit_text = ' '.join(str(note) for note in (body.get('assumptions_audit') or [])).lower()
        assert body['illustrative_assumptions'] is True
        assert 'mapped cost' in audit_text and 'standard 12%' in audit_text
    finally:
        DATASET_CACHE.pop(session_id, None)


def test_profit_illustrative_true_when_only_cogs_fallback(client: TestClient):
    """Partial fallback: mapped OpEx + hard COGS fallback → illustrative True; audit shows both.

    Use column name 'overhead' so OpEx maps without also feeding COGS via universal_cost scan
    (names like operating_expense / opex match universal_cost's expense|opex tokens).
    """
    session_id = 'profit_illustrative_cogs_only_fallback'
    frame = pd.DataFrame({
        'order_date': pd.date_range('2018-01-01', periods=40, freq='D'),
        'sales': [200.0] * 40,
        'overhead': [20.0] * 40,
    })
    try:
        _seed_session(session_id, frame)
        _cogs, _cs, _cr, cogs_fb = resolve_cogs_series(frame, 'sales')
        _opex, _os, _or, opex_fb = resolve_operating_expense_series(frame, 'sales')
        assert cogs_fb is True
        assert opex_fb is False

        response = client.post('/api/profit-forecast/run', json={
            'session_id': session_id,
            'forecast_periods': 5,
            'confirmed_assumptions': True,
        })
        assert response.status_code == 200, response.text
        body = response.json()
        audit_text = ' '.join(str(note) for note in (body.get('assumptions_audit') or [])).lower()
        assert body['illustrative_assumptions'] is True
        assert 'fallback assumption' in audit_text and 'mapped operating' in audit_text
    finally:
        DATASET_CACHE.pop(session_id, None)
