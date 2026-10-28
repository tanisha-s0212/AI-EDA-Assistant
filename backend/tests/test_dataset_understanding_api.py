from __future__ import annotations

import sys
import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'DATASET_DIR', tmp_path)
    monkeypatch.setattr(main, 'DATASET_CACHE_INDEX_PATH', tmp_path / 'index.json')
    monkeypatch.setattr(main, 'ACTIVITY_DB_AVAILABLE', False)
    monkeypatch.setattr(main, 'DATASET_CACHE', {})
    return TestClient(main.app, raise_server_exceptions=False)


def test_review_uses_full_uploaded_months_and_checks_owner(client):
    months = pd.date_range('2023-01-01', periods=24, freq='MS')
    frame = pd.DataFrame({'month': months.strftime('%Y-%m-%d'), 'sales': range(24), 'item': ['A'] * 24})
    owner = {'X-Client-Session-Id': 'owner'}
    uploaded = client.post('/api/parse-dataset', headers=owner, files={'file': ('sales.csv', frame.to_csv(index=False), 'text/csv')}, data={'description': 'Monthly sales'})
    assert uploaded.status_code == 200, uploaded.text
    dataset_id = uploaded.json()['datasetId']
    review = client.post('/api/dataset-understanding', headers=owner, json={'dataset_id': dataset_id, 'date_column': 'month', 'target_column': 'sales', 'dimensions': ['item'], 'grain': 'monthly'})
    assert review.status_code == 200, review.text
    assert review.json()['coverage']['observed_months'] == 24
    assert review.json()['coverage']['usable_months'] == 24
    assert review.json()['coverage']['meets_minimum'] is True
    assert review.json()['description'] == 'Monthly sales'
    denied = client.get('/api/dataset-preview', headers={'X-Client-Session-Id': 'other'}, params={'dataset_id': dataset_id})
    assert denied.status_code == 403
    denied_review = client.post('/api/dataset-understanding', headers={'X-Client-Session-Id': 'other'}, json={'dataset_id': dataset_id, 'date_column': 'month', 'target_column': 'sales'})
    assert denied_review.status_code == 403


def test_delimited_override_is_saved_and_used(client):
    result = client.post('/api/parse-dataset', files={'file': ('sales.csv', 'skip this line\nmonth;sales\n2024-01-01;12\n2024-02-01;20\n', 'text/csv')}, data={'separator': ';', 'encoding': 'utf-8', 'header_row': '1'})
    assert result.status_code == 200, result.text
    body = result.json()
    assert body['columns'] == ['month', 'sales']
    assert body['rowCount'] == 2
    assert body['parseOptions'] == {'separator': ';', 'encoding': 'utf-8', 'header_row': 1}


def test_review_rejects_numeric_month_parts_as_full_dates(client):
    result = client.post('/api/parse-dataset', files={'file': ('sales.csv', 'month,sales\n1,12\n2,20\n', 'text/csv')})
    assert result.status_code == 200, result.text
    reviewed = client.post('/api/dataset-understanding', json={'dataset_id': result.json()['datasetId'], 'date_column': 'month', 'target_column': 'sales'})
    assert reviewed.status_code == 422
    assert 'date' in reviewed.json()['error'].lower()


def test_workbook_preview_checks_ownership_and_stacking_schema(client):
    content = io.BytesIO()
    with pd.ExcelWriter(content, engine='openpyxl') as writer:
        pd.DataFrame({'date': ['2024-01-01', '2024-02-01'], 'sales': [1, 2]}).to_excel(writer, sheet_name='Sales', index=False)
        pd.DataFrame({'date': ['2024-01-01', '2024-02-01'], 'units': [1, 2]}).to_excel(writer, sheet_name='Units', index=False)
    owner = {'X-Client-Session-Id': 'owner'}
    result = client.post('/api/parse-dataset', headers=owner, files={'file': ('book.xlsx', content.getvalue(), 'application/octet-stream')})
    assert result.status_code == 200, result.text
    dataset_id = result.json()['datasetId']
    preview = client.get('/api/dataset-sheet-preview', headers=owner, params={'dataset_id': dataset_id, 'sheet': 'Sales'})
    assert preview.status_code == 200 and preview.json()['columns'] == ['date', 'sales']
    denied = client.get('/api/dataset-sheet-preview', headers={'X-Client-Session-Id': 'other'}, params={'dataset_id': dataset_id, 'sheet': 'Sales'})
    assert denied.status_code == 403
    stacked = client.post('/api/parse-dataset-sheet-selection', headers=owner, json={'dataset_id': dataset_id, 'selected_sheets': ['Sales', 'Units'], 'merge_mode': 'stack'})
    assert stacked.status_code == 400


def test_optional_full_quality_reports_missing_and_duplicates(client):
    result = client.post('/api/parse-dataset', files={'file': ('quality.csv', 'a,b\n1,x\n1,x\n2,\n', 'text/csv')})
    assert result.status_code == 200, result.text
    summary = client.post('/api/dataset-quality-summary', json={'dataset_id': result.json()['datasetId']})
    assert summary.status_code == 200, summary.text
    body = summary.json()
    assert body['row_count'] == 3
    assert body['duplicate_rows'] == 1
    assert body['missing_cells'] == 1
    assert next(column for column in body['columns'] if column['name'] == 'b')['null_count'] == 1


def test_explicit_grain_confirmation_updates_version_only_when_it_changes(client):
    months = pd.date_range('2023-01-01', periods=24, freq='MS')
    frame = pd.DataFrame({'month': months.strftime('%Y-%m-%d'), 'sales': range(24)})
    upload = client.post('/api/parse-dataset', files={'file': ('sales.csv', frame.to_csv(index=False), 'text/csv')})
    dataset_id = upload.json()['datasetId']
    old_version = main.DATASET_CACHE[dataset_id]['dataset_version']
    payload = {'dataset_id': dataset_id, 'date_column': 'month', 'target_column': 'sales', 'grain': 'records'}
    review = client.post('/api/dataset-understanding', json=payload)
    assert review.status_code == 200, review.text
    assert review.json()['grain_changed'] is True
    assert main.DATASET_CACHE[dataset_id]['grain_capability']['grain'] == 'unknown'
    assert main.DATASET_CACHE[dataset_id]['dataset_version'] == old_version + 1
    again = client.post('/api/dataset-understanding', json=payload)
    assert again.status_code == 200
    assert again.json()['grain_changed'] is False
    assert main.DATASET_CACHE[dataset_id]['dataset_version'] == old_version + 1
    restored = client.post('/api/dataset-understanding', json={**payload, 'grain': 'auto'})
    assert restored.status_code == 200
    assert main.DATASET_CACHE[dataset_id]['grain_capability']['grain'] == 'monthly_aggregate'
    assert main.DATASET_CACHE[dataset_id]['dataset_version'] == old_version + 2
