"""ML forecast aggregation: high-cardinality timestamps must not become one chart point per row."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from ml_forecast_pipeline import (  # noqa: E402
    prepare_frame_for_ml_pipeline,
    run_full_pipeline,
)


def _high_cardinality_frame(rows: int = 240) -> pd.DataFrame:
    start = pd.Timestamp('2020-01-01 08:15:00')
    return pd.DataFrame({
        'created': [start + pd.Timedelta(days=index) + pd.Timedelta(hours=index % 11) for index in range(rows)],
        'sales': [100.0 + (index % 13) for index in range(rows)],
    })


def test_prepare_frame_for_ml_coarsens_unique_timestamps():
    raw = _high_cardinality_frame(240)
    assert len(raw) == 240
    assert raw['created'].nunique() == 240

    prepared, date_col, target_col, frequency, period_label = prepare_frame_for_ml_pipeline(
        raw, 'created', 'sales'
    )

    assert date_col == 'created'
    assert target_col == 'sales'
    assert period_label in {'day', 'week', 'month', 'quarter', 'year'}
    assert frequency in {'weekly', 'monthly'}
    # Must be far fewer than unique raw timestamps (not one chart/train point per row).
    assert len(prepared) < len(raw)
    assert len(prepared) <= 60
    assert prepared[date_col].nunique() == len(prepared)
    assert prepared.attrs.get('period_label') == period_label


def test_run_full_pipeline_history_matches_usable_periods(tmp_path: Path):
    frame = _high_cardinality_frame(240)
    # Prefer gradient boosting only path via auto — still trains ensemble but fixture is small enough.
    result = run_full_pipeline(
        tmp_path,
        target_col='sales',
        date_col='created',
        horizon=3,
        frequency='auto',
        frame=frame,
        model_type='gradient_boosting',
    )

    actual_rows = [row for row in result['forecast_line'] if row['type'] == 'actual']
    history_len = len(actual_rows)
    assert history_len < len(frame)
    assert history_len == int(result['metadata']['usable_periods'])
    assert result['metadata']['period_label'] in {'day', 'week', 'month', 'quarter', 'year'}
    assert history_len == len({row['period'] for row in actual_rows})
