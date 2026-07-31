"""Shared sales-domain column heuristics for core forecasting workflows.

Used by time-series loaders, ML forecast detection, and loss/profit column matching.
Does not depend on the agentic layer.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

import pandas as pd

# Prefer explicit business date names over generic "start"/"time" tokens.
DATE_TOKEN_SCORES: list[tuple[str, int]] = [
    ('invoice_date', 100),
    ('order_date', 95),
    ('bill_date', 90),
    ('doc_date', 88),
    ('transaction_date', 85),
    ('sale_date', 85),
    ('year_month', 80),
    ('period', 70),
    ('month', 60),
    ('week', 55),
    ('date', 50),
    ('time', 20),
    ('start', 15),
]

# Prefer named revenue/sales columns over generic totals/amounts.
REVENUE_TOKEN_SCORES: list[tuple[str, int]] = [
    ('sale_free', 100),
    ('net_sales', 98),
    ('sale_value', 95),
    ('total_value_sale', 92),
    ('gmv', 90),
    ('turnover', 88),
    ('revenue', 85),
    ('sales', 80),
    ('total_value', 70),
    ('amount', 40),
    ('total', 25),
]

REVENUE_EXCLUDE = re.compile(r'loss|lost|missed|return|refund|cost|cogs|expense|tax|qty|quantity|unit_price|price', re.IGNORECASE)
TARGET_ID_EXCLUDE = re.compile(r'(^id$|_id$|uuid|index|code|sku$)', re.IGNORECASE)

DATE_PATTERN = re.compile(
    r'invoice_date|order_date|bill_date|doc_date|transaction_date|sale_date|year_month|period|month|week|date|time|start',
    re.IGNORECASE,
)
REVENUE_PATTERN = re.compile(
    r'^(?!.*(?:loss|lost|missed|return|refund|cost|cogs|expense)).*(?:sale_free|net_sales|sale_value|total_value_sale|gmv|turnover|revenue|sales|total_value|amount)',
    re.IGNORECASE,
)


def _token_score(name: str, token_scores: Sequence[tuple[str, int]]) -> int:
    lowered = str(name).lower()
    best = 0
    for token, score in token_scores:
        if token in lowered:
            best = max(best, score)
    return best


def score_date_column(name: str) -> int:
    return _token_score(name, DATE_TOKEN_SCORES)


def score_revenue_column(name: str) -> int:
    lowered = str(name).lower()
    if REVENUE_EXCLUDE.search(lowered) and not any(t in lowered for t in ('sale_free', 'sale_value', 'net_sales', 'revenue', 'sales')):
        return 0
    if TARGET_ID_EXCLUDE.search(lowered):
        return 0
    return _token_score(name, REVENUE_TOKEN_SCORES)


def pick_best_date_column(columns: Iterable[str], column_info: list[dict[str, Any]] | None = None) -> str | None:
    available = [str(c) for c in columns]
    scored: list[tuple[int, str]] = []
    for name in available:
        score = score_date_column(name)
        if score <= 0 and column_info:
            meta = next((item for item in column_info if isinstance(item, dict) and item.get('name') == name), None)
            if meta and meta.get('role') in ('datetime', 'date'):
                score = 45
        if score > 0:
            scored.append((score, name))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def pick_best_revenue_column(
    columns: Iterable[str],
    frame: pd.DataFrame | None = None,
    column_info: list[dict[str, Any]] | None = None,
    exclude: set[str] | None = None,
) -> str | None:
    excluded = exclude or set()
    available = [str(c) for c in columns if str(c) not in excluded]
    scored: list[tuple[int, float, str]] = []
    for name in available:
        name_score = score_revenue_column(name)
        variance = 0.0
        if frame is not None and name in frame.columns:
            numeric = pd.to_numeric(frame[name], errors='coerce')
            if numeric.notna().sum() == 0:
                continue
            variance = float(numeric.var(skipna=True) or 0.0)
        elif column_info:
            meta = next((item for item in column_info if isinstance(item, dict) and item.get('name') == name), None)
            if meta and meta.get('role') != 'numeric':
                if name_score <= 0:
                    continue
        if name_score <= 0 and variance <= 0:
            continue
        # Name score dominates; variance breaks ties among similarly named columns.
        scored.append((name_score, variance, name))
    if scored:
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return scored[0][2]

    # Fallback: highest-variance numeric excluding ids/dates.
    if frame is not None:
        numeric_fallback: list[tuple[float, str]] = []
        for name in available:
            if TARGET_ID_EXCLUDE.search(name) or score_date_column(name) > 0:
                continue
            numeric = pd.to_numeric(frame[name], errors='coerce')
            if numeric.notna().sum() == 0:
                continue
            variance = float(numeric.var(skipna=True) or 0.0)
            if variance > 0:
                numeric_fallback.append((variance, name))
        if numeric_fallback:
            numeric_fallback.sort(key=lambda item: (-item[0], item[1]))
            return numeric_fallback[0][1]
    return None


def resolve_sales_columns(
    columns: Iterable[str],
    *,
    date_column: str | None = None,
    target_column: str | None = None,
    frame: pd.DataFrame | None = None,
    column_info: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    available = [str(c) for c in columns]
    date_col = date_column if date_column and date_column in available else pick_best_date_column(available, column_info)
    if not date_col:
        raise ValueError(f'No date column found. Available columns: {available}')
    target_col = target_column if target_column and target_column in available else pick_best_revenue_column(
        available, frame=frame, column_info=column_info, exclude={date_col},
    )
    if not target_col:
        raise ValueError(f'No sales/revenue target column found. Available columns: {available}')
    return date_col, target_col


def sales_mapping_payload(date_column: str, target_column: str, source: str = 'auto') -> dict[str, Any]:
    return {
        'date_column': date_column,
        'target_column': target_column,
        'source': source,
    }
