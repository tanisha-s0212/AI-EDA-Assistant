"""Revenue_loss floors: without mapped revenue_loss, do not let inferred hist dips dominate."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from main import DATASET_CACHE, app, ensure_session_state, set_dataset_cache_entry  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(app)


def _seed(session_id: str, frame: pd.DataFrame, future_revenues: list[float]) -> None:
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
    periods = pd.date_range('2019-01-01', periods=len(future_revenues), freq='D')
    futures = [
        {'period': str(period.date()), 'predicted': future_revenues[index]}
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


def test_sales_only_revenue_loss_bounded_by_1p5pct_floor_when_no_shortfall(client: TestClient):
    """Finding A soften: no mapped revenue_loss + no shortfall → revenue_loss ≈ rev×1.5%×pressure."""
    session_id = 'loss_rev_floor_sales_only'
    # Volatile daily sales so inferred hist dips would be large if still used.
    sales = [100.0, 400.0, 80.0, 450.0, 90.0, 500.0, 70.0, 480.0] * 10
    frame = pd.DataFrame({
        'order_date': pd.date_range('2018-01-01', periods=len(sales), freq='D'),
        'sales': sales,
    })
    avg_sales = float(sum(sales) / len(sales))
    # Futures at/above average → forecast_shortfall_loss = 0
    future_revenues = [avg_sales * 1.05] * 3
    try:
        _seed(session_id, frame, future_revenues)
        response = client.post('/api/loss-forecast/run', json={
            'session_id': session_id,
            'forecast_periods': 3,
            'confirmed_assumptions': True,
        })
        assert response.status_code == 200, response.text
        row0 = response.json()['loss_forecast'][0]
        rev0 = future_revenues[0]
        pressure = 1.0  # index 0
        max_allowed = rev0 * 0.015 * pressure * 1.02  # eps for rounding
        assert row0['revenue_loss'] <= max_allowed + 1e-6
        assert row0['revenue_loss'] >= rev0 * 0.015 * pressure * 0.98
        # Floor sum ≈ 7.8% when rates are defaults (tol 5% of that sum)
        assert row0['total_loss'] / rev0 <= 0.078 * 1.05
        # Side effect accepted: was High (~0.5) under old hist floor; expect not High now
        assert row0['risk_label'] in {'Low', 'Medium'}
        assert row0['loss_risk_score'] <= 0.15
    finally:
        DATASET_CACHE.pop(session_id, None)


def test_mapped_revenue_loss_still_uses_historical_mean(client: TestClient):
    """With an explicit revenue_loss column, historical mean may exceed the 1.5% floor."""
    session_id = 'loss_rev_floor_mapped'
    n = 40
    frame = pd.DataFrame({
        'order_date': pd.date_range('2018-01-01', periods=n, freq='D'),
        'sales': [200.0] * n,
        # Large mapped driver so hist mean >> 1.5% of future rev
        'lost_revenue': [50.0] * n,
    })
    future_revenues = [200.0] * 3
    try:
        _seed(session_id, frame, future_revenues)
        response = client.post('/api/loss-forecast/run', json={
            'session_id': session_id,
            'forecast_periods': 3,
            'confirmed_assumptions': True,
        })
        assert response.status_code == 200, response.text
        row0 = response.json()['loss_forecast'][0]
        # Mapped path: hist mean (~50) should beat 1.5% of 200 (=3)
        assert row0['revenue_loss'] >= 40.0
    finally:
        DATASET_CACHE.pop(session_id, None)
