"""Read-only dataset coverage calculations for Data Understanding."""
from __future__ import annotations

import numpy as np
import pandas as pd


def monthly_coverage(frame: pd.DataFrame, date_column: str, target_column: str, dimensions: list[str]) -> dict:
    dates = pd.to_datetime(frame[date_column], errors='coerce', utc=True).dt.tz_localize(None)
    months = dates.dt.to_period('M')
    values = pd.to_numeric(frame[target_column], errors='coerce')
    usable = months.notna() & values.notna() & np.isfinite(values)
    observed = months.dropna().drop_duplicates().sort_values()
    usable_months = months[usable].drop_duplicates().sort_values()
    missing: list[str] = []
    missing_count = 0
    if len(observed):
        first, last = observed.iloc[0], observed.iloc[-1]
        span = last.ordinal - first.ordinal + 1
        # Return only a bounded list, while retaining an accurate count.
        missing_count = max(0, span - len(observed))
        observed_set = set(observed.astype(str))
        if missing_count:
            for period in pd.period_range(first, last, freq='M'):
                if str(period) not in observed_set:
                    missing.append(str(period))
                    if len(missing) == 100:
                        break
    repeated = None
    if dimensions:
        keys = frame.loc[months.notna(), dimensions].copy()
        month_key = '__observed_month__'
        while month_key in keys.columns:
            month_key += '_'
        keys.insert(0, month_key, months[months.notna()].astype(str).values)
        repeated = int(keys.duplicated().sum())
    return {
        'first_month': str(observed.iloc[0]) if len(observed) else None,
        'last_month': str(observed.iloc[-1]) if len(observed) else None,
        'observed_months': int(len(observed)), 'usable_months': int(len(usable_months)),
        'invalid_date_rows': int(dates.isna().sum()),
        'invalid_target_rows': int((~(values.notna() & np.isfinite(values))).sum()),
        'missing_months': missing, 'missing_month_count': missing_count,
        'duplicate_key_rows': repeated, 'dimensions': dimensions,
        'minimum_months': 24, 'meets_minimum': len(usable_months) >= 24,
    }
