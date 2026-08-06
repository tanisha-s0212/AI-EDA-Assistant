"""Regression tests for large-dataset Parquet persist, ML column-pruned IO, and EDA sampling."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from main import (  # noqa: E402
    DATASET_CACHE,
    EDA_ANALYSIS_ROW_CAP,
    EDA_CHART_POINT_CAP,
    AdvancedEdaRequest,
    app,
    build_advanced_eda_payload,
    estimate_kde,
    load_cached_analysis_frame,
    persist_inferred_dataset_frame,
    set_dataset_cache_entry,
)
from ml_forecast_pipeline import load_and_detect, prepare_frame_for_ml_pipeline  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(app)


def test_persist_inferred_dataset_frame_writes_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('main.DATASET_DIR', tmp_path)
    dataset_id = 'persist_parquet_demo'
    frame = pd.DataFrame({
        'doc_date': pd.date_range('2023-01-01', periods=120, freq='D'),
        'amount': np.arange(120, dtype=float),
        'extra': [f'c{i % 5}' for i in range(120)],
    })
    path = persist_inferred_dataset_frame(dataset_id, {'filename': 'demo.parquet'}, frame)
    entry = DATASET_CACHE[dataset_id]
    assert path.suffix == '.parquet'
    assert path.exists()
    assert entry.get('parquet_path') == str(path)
    assert 'frame_path' not in entry
    assert 'csv_path' not in entry
    DATASET_CACHE.pop(dataset_id, None)


def test_clean_parquet_infer_dtypes_keeps_parquet_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient):
    monkeypatch.setattr('main.DATASET_DIR', tmp_path)
    dataset_id = 'clean_keep_parquet'
    frame = pd.DataFrame({
        'created': pd.date_range('2024-01-01', periods=150, freq='D'),
        'sales': np.linspace(10, 160, 150),
        'zone': ['N', 'S'] * 75,
    })
    parquet_path = tmp_path / f'{dataset_id}.parquet'
    frame.to_parquet(parquet_path, index=False)
    set_dataset_cache_entry(dataset_id, {
        'parquet_path': str(parquet_path),
        'filename': 'wide.parquet',
        'row_count': int(len(frame)),
        'column_count': int(len(frame.columns)),
        'columns': list(frame.columns),
        'duplicate_count': 0,
    })

    response = client.post('/api/clean-dataset', json={
        'dataset_id': dataset_id,
        'remove_duplicates': True,
        'handle_missing': True,
        'convert_dates': True,
        'standardize_names': True,
        'infer_dtypes': True,
    })
    assert response.status_code == 200, response.text
    entry = DATASET_CACHE[dataset_id]
    assert entry.get('parquet_path')
    assert Path(entry['parquet_path']).suffix == '.parquet'
    assert Path(entry['parquet_path']).exists()
    assert not entry.get('frame_path')

    analysis_frame, total_rows = load_cached_analysis_frame(entry)
    assert total_rows == len(frame) or total_rows == entry['row_count']
    assert len(analysis_frame) <= EDA_ANALYSIS_ROW_CAP
    DATASET_CACHE.pop(dataset_id, None)


def test_ml_load_and_detect_parquet_matches_two_col_prepare(tmp_path: Path):
    rows = 240
    start = pd.Timestamp('2020-01-01 08:15:00')
    base = pd.DataFrame({
        'created': [start + pd.Timedelta(days=i) + pd.Timedelta(hours=i % 11) for i in range(rows)],
        'sales': [100.0 + (i % 13) for i in range(rows)],
    })
    wide = base.copy()
    for index in range(20):
        wide[f'extra_{index}'] = np.random.default_rng(index).normal(size=rows)

    narrow_path = tmp_path / 'narrow.parquet'
    wide_path = tmp_path / 'wide.parquet'
    base.to_parquet(narrow_path, index=False)
    wide.to_parquet(wide_path, index=False)

    narrow_frame, d1, t1 = load_and_detect(narrow_path, 'created', 'sales')
    wide_frame, d2, t2 = load_and_detect(wide_path, 'created', 'sales')
    assert d1 == d2 == 'created'
    assert t1 == t2 == 'sales'
    assert list(wide_frame.columns) == ['created', 'sales']
    assert len(wide_frame) == len(narrow_frame)

    prep_n, *_ = prepare_frame_for_ml_pipeline(narrow_frame, 'created', 'sales')
    prep_w, *_ = prepare_frame_for_ml_pipeline(wide_frame, 'created', 'sales')
    assert abs(len(prep_n) - len(prep_w)) <= 1
    assert prep_n['sales'].sum() == pytest.approx(prep_w['sales'].sum(), rel=1e-9)


def test_advanced_eda_samples_large_frame(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('main.DATASET_DIR', tmp_path)
    monkeypatch.setattr('main.EDA_ANALYSIS_ROW_CAP', 500)
    dataset_id = 'eda_sample_large'
    rows = 2_000
    frame = pd.DataFrame({
        'a': np.random.default_rng(0).normal(size=rows),
        'b': np.random.default_rng(1).normal(size=rows),
        'c': np.random.default_rng(2).integers(0, 5, size=rows),
    })
    path = tmp_path / f'{dataset_id}.parquet'
    frame.to_parquet(path, index=False)
    set_dataset_cache_entry(dataset_id, {
        'parquet_path': str(path),
        'filename': 'big.parquet',
        'row_count': rows,
        'column_count': 3,
        'columns': list(frame.columns),
        'duplicate_count': 0,
    })

    payload = build_advanced_eda_payload(AdvancedEdaRequest(dataset_id=dataset_id, data=[]))
    assert payload['row_count'] == rows
    assert payload['sampled_row_count'] <= 500
    assert payload['sampled_row_count'] < payload['row_count']
    assert payload['analysis_sampled'] is True
    DATASET_CACHE.pop(dataset_id, None)


def test_estimate_kde_caps_input_size():
    values = np.linspace(0, 1, EDA_CHART_POINT_CAP * 3)
    result = estimate_kde(values)
    assert result is not None
    x_points, density = result
    assert len(x_points) == 160
    assert len(density) == 160


def test_advanced_eda_small_frame_not_sampled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('main.DATASET_DIR', tmp_path)
    dataset_id = 'eda_small'
    frame = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0], 'y': [4.0, 3.0, 2.0, 1.0]})
    path = tmp_path / f'{dataset_id}.parquet'
    frame.to_parquet(path, index=False)
    set_dataset_cache_entry(dataset_id, {
        'parquet_path': str(path),
        'filename': 'small.parquet',
        'row_count': 4,
        'column_count': 2,
        'columns': list(frame.columns),
        'duplicate_count': 0,
    })
    payload = build_advanced_eda_payload(AdvancedEdaRequest(dataset_id=dataset_id, data=[]))
    assert payload['row_count'] == 4
    assert payload['sampled_row_count'] == 4
    assert payload['analysis_sampled'] is False
    DATASET_CACHE.pop(dataset_id, None)
