#!/usr/bin/env python3
"""Train a demo sklearn regression model from samples/sales/sales_monthly.csv.

Customer production path remains train-on-upload in the ML Assistant tab.
This script only builds an optional demo artifact under backend/models/demo/.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / 'samples' / 'sales' / 'sales_monthly.csv'
OUT_DIR = Path(__file__).resolve().parents[1] / 'models' / 'demo'


def main() -> None:
    if not SAMPLE.exists():
        raise SystemExit(f'Missing sample dataset: {SAMPLE}')
    frame = pd.read_csv(SAMPLE)
    frame['month_number'] = pd.to_datetime(frame['year_month']).dt.month
    frame['trend_index'] = range(len(frame))
    feature_cols = ['month_number', 'trend_index', 'region', 'category']
    target = 'total_total_value_sale_free'
    X = frame[feature_cols]
    y = frame[target]
    model = Pipeline([
        ('prep', ColumnTransformer([
            ('num', 'passthrough', ['month_number', 'trend_index']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['region', 'category']),
        ])),
        ('model', GradientBoostingRegressor(random_state=42)),
    ])
    model.fit(X, y)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / 'sales_monthly_demo.joblib'
    meta_path = OUT_DIR / 'sales_monthly_demo.json'
    joblib.dump({
        'model': model,
        'feature_columns': feature_cols,
        'target_column': target,
        'problem_type': 'regression',
        'source': str(SAMPLE.relative_to(ROOT)),
    }, model_path)
    meta_path.write_text(json.dumps({
        'model_path': str(model_path.name),
        'feature_columns': feature_cols,
        'target_column': target,
        'problem_type': 'regression',
        'rows_trained': int(len(frame)),
    }, indent=2), encoding='utf-8')
    print(f'Wrote {model_path}')
    print(f'Wrote {meta_path}')


if __name__ == '__main__':
    main()
