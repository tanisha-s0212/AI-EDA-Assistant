"""Agentic date/target selection must use shared sales_domain helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from agentic.agentic_adapter import preferred_date_column, preferred_target_column  # noqa: E402


def test_preferred_date_column_prefers_created_over_month():
    frame = pd.DataFrame({
        'month': [1, 2, 3, 4, 5, 6],
        'dayofweek': [0, 1, 2, 3, 4, 5],
        'created': pd.date_range('2024-01-01', periods=6, freq='D'),
        'dollars': [10, 20, 30, 40, 50, 60],
    })
    assert preferred_date_column(frame) == 'created'


def test_preferred_target_column_prefers_revenueish_numeric():
    frame = pd.DataFrame({
        'created': pd.date_range('2024-01-01', periods=6, freq='D'),
        'month': [1, 2, 3, 4, 5, 6],
        'kwhTotal': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'total_total_value_sale_free': [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
    })
    assert preferred_target_column(frame) == 'total_total_value_sale_free'
