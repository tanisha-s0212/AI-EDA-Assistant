"""Shared sales-domain column heuristics for core forecasting workflows.

Used by time-series loaders, ML forecast detection, and loss/profit column matching.
Does not depend on the agentic layer.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

import pandas as pd

# Prefer explicit business date names over generic "start"/"time" tokens.
# Keep in sync with frontend sales-domain.ts DATE_TOKEN_SCORES.
# Do not include bare `month`/`week` — those match engineered date-part columns
# (and `dayofweek` via substring) and collapse aggregation.
DATE_TOKEN_SCORES: list[tuple[str, int]] = [
    ('invoice_date', 100),
    ('order_date', 95),
    ('bill_date', 90),
    ('doc_date', 88),
    ('transaction_date', 85),
    ('sale_date', 85),
    ('timestamp', 82),
    ('datetime', 80),
    ('year_month', 80),
    ('created', 78),
    ('ended', 76),
    ('period', 70),
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

# Engineered date-part columns — never pick as the series index.
DATE_PART_EXCLUDE = re.compile(
    r'^(year|month|day|hour|minute|second|dayofweek|weekday|week|quarter|weekofyear|doy)$',
    re.IGNORECASE,
)

DATE_PATTERN = re.compile(
    r'invoice_date|order_date|bill_date|doc_date|transaction_date|sale_date|year_month|timestamp|datetime|created|ended|period|date|time|start',
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


def is_date_part_column(name: str) -> bool:
    return bool(DATE_PART_EXCLUDE.match(str(name).strip()))


def score_date_column(name: str) -> int:
    if is_date_part_column(name):
        return 0
    return _token_score(name, DATE_TOKEN_SCORES)


def score_revenue_column(name: str) -> int:
    lowered = str(name).lower()
    if REVENUE_EXCLUDE.search(lowered) and not any(t in lowered for t in ('sale_free', 'sale_value', 'net_sales', 'revenue', 'sales')):
        return 0
    if TARGET_ID_EXCLUDE.search(lowered):
        return 0
    return _token_score(name, REVENUE_TOKEN_SCORES)


def _column_unique_count(
    name: str,
    *,
    frame: pd.DataFrame | None = None,
    column_info: list[dict[str, Any]] | None = None,
) -> int:
    if frame is not None and name in frame.columns:
        try:
            return int(pd.Series(frame[name]).nunique(dropna=True))
        except Exception:
            return 0
    if column_info:
        meta = next((item for item in column_info if isinstance(item, dict) and item.get('name') == name), None)
        if meta:
            for key in ('uniqueCount', 'unique_count', 'nunique'):
                value = meta.get(key)
                if value is not None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        pass
    return 0


def pick_best_date_column(
    columns: Iterable[str],
    column_info: list[dict[str, Any]] | None = None,
    frame: pd.DataFrame | None = None,
) -> str | None:
    available = [str(c) for c in columns]
    scored: list[tuple[int, int, str]] = []
    for name in available:
        if is_date_part_column(name):
            continue
        score = score_date_column(name)
        if score <= 0 and column_info:
            meta = next((item for item in column_info if isinstance(item, dict) and item.get('name') == name), None)
            if meta and meta.get('role') in ('datetime', 'date'):
                score = 45
        if score > 0:
            unique_count = _column_unique_count(name, frame=frame, column_info=column_info)
            scored.append((score, unique_count, name))
    if not scored:
        return None
    # Higher name score first; break ties by higher cardinality (real timestamps beat sparse labels).
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return scored[0][2]


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
        if is_date_part_column(name):
            continue
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
            if is_date_part_column(name) or TARGET_ID_EXCLUDE.search(name) or score_date_column(name) > 0:
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
    date_col = date_column if date_column and date_column in available else pick_best_date_column(
        available, column_info=column_info, frame=frame,
    )
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


# Keep in sync with frontend sales-domain.ts SALES_PRESET_REVENUE_SCORE_THRESHOLD
SALES_PRESET_REVENUE_SCORE_THRESHOLD = 70


def should_enable_sales_preset(columns: Iterable[str]) -> bool:
    """Enable sales cleaning preset when a strong sales/revenue column name is present."""
    return any(score_revenue_column(str(name)) >= SALES_PRESET_REVENUE_SCORE_THRESHOLD for name in columns)
