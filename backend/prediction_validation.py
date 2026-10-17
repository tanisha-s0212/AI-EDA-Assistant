"""Input contracts and readiness information shared by prediction routes."""
from __future__ import annotations

import math
import csv
import io
from pathlib import Path
from typing import Any

import pandas as pd


def read_prediction_table(content: bytes, filename: str, sheet_name: str | None = None, limit: int = 5000) -> pd.DataFrame:
    """Parse tabular prediction inputs without guessing category or identifier types."""
    suffix = Path(filename).suffix.lower()
    stream = io.BytesIO(content)
    if suffix in ('.csv', '.tsv'):
        try:
            text = content.decode('utf-8-sig')
        except UnicodeDecodeError as error:
            raise ValueError('Save CSV/TSV input as UTF-8 before uploading.') from error
        separator = '\t' if suffix == '.tsv' else ','
        reader = csv.reader(io.StringIO(text), delimiter=separator, strict=True)
        headers = next(reader, [])
        if not headers or any(not value.strip() for value in headers) or len(headers) != len(set(headers)):
            raise ValueError('Use unique, non-empty column headers matching the model feature names.')
        records = []
        for row_number, row in enumerate(reader, start=1):
            if row_number > limit:
                raise ValueError(f'Batch prediction accepts at most {limit:,} rows per file. Split the file and retry.')
            if len(row) > len(headers):
                raise ValueError(f'Data row {row_number} has more values than column headers. Check its delimiter and quoting.')
            # Retain empty and incomplete data rows so validation can report
            # their original positions instead of silently dropping them.
            records.append(row + [''] * (len(headers) - len(row)))
        frame = pd.DataFrame(records, columns=headers, dtype=object)
    elif suffix in ('.xlsx', '.xls'):
        workbook = pd.ExcelFile(stream)
        if sheet_name and sheet_name not in workbook.sheet_names:
            raise ValueError(f'Sheet "{sheet_name}" was not found. Available sheets: {", ".join(workbook.sheet_names)}.')
        if not sheet_name and len(workbook.sheet_names) > 1:
            raise ValueError(f'Choose a worksheet by name. Available sheets: {", ".join(workbook.sheet_names)}.')
        selected = sheet_name or workbook.sheet_names[0]
        raw = pd.read_excel(workbook, sheet_name=selected, header=None, nrows=limit + 2, dtype=object, keep_default_na=False)
        if raw.empty:
            raise ValueError('The selected worksheet is empty.')
        headers = [str(value) for value in raw.iloc[0]]
        if any(not value.strip() for value in headers) or len(headers) != len(set(headers)):
            raise ValueError('Use unique, non-empty column headers in the first worksheet row.')
        frame = raw.iloc[1:].copy()
        frame.columns = headers
    elif suffix == '.parquet':
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(stream)
        if parquet.metadata.num_rows > limit:
            raise ValueError(f'Batch prediction accepts at most {limit:,} rows per file. Split the file and retry.')
        frame = parquet.read().to_pandas()
    else:
        raise ValueError('Use a CSV, TSV, Excel (.xlsx/.xls), or Parquet file.')
    if frame.empty:
        raise ValueError('The file must contain at least one data row.')
    if len(frame) > limit:
        raise ValueError(f'Batch prediction accepts at most {limit:,} rows per file. Split the file and retry.')
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column headers are not supported.')
    return frame


def prediction_schema(bundle: dict[str, Any]) -> dict[str, Any]:
    prep = bundle.get('preprocessing') or {}
    names = prep.get('raw_feature_columns') or bundle.get('feature_columns') or []
    fields = []
    for name in names:
        numeric = name in prep.get('numeric_features', [])
        fields.append({
            'name': name,
            'type': prep.get('feature_types', {}).get(name, 'numeric' if numeric else 'categorical' if name in prep.get('categorical_features', []) else 'unknown'),
            'range': prep.get('numeric_ranges', {}).get(name),
            'categories': prep.get('categories', {}).get(name, []),
        })
    evidence = bundle.get('prediction_validation') or {}
    reasons = list(evidence.get('reasons') or [])
    if not evidence:
        reasons.append('Validation evidence is unavailable for this artifact. Review its origin and performance before using results.')
    if not prep:
        reasons.append('Training input types and distributions are unavailable; the uploaded pipeline must validate its own inputs.')
    return {
        'fields': fields,
        'target': bundle.get('target_column'),
        'problem_type': bundle.get('problem_type'),
        'trained_at': bundle.get('trained_at'),
        'dataset_id': bundle.get('dataset_id'),
        'dataset_version': bundle.get('dataset_version'),
        'dataset_fingerprint': bundle.get('dataset_fingerprint'),
        'readiness': {**evidence, 'status': 'caution' if reasons else 'available', 'reasons': reasons},
    }


def validate_prediction_inputs(features: dict[str, Any], bundle: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    schema = prediction_schema(bundle)
    if not schema['fields']:
        raise ValueError('The model has no named input schema. Upload a pipeline with feature names or train a model in ML Assistant.')
    validated: dict[str, Any] = {}
    warnings: list[str] = []
    for field in schema['fields']:
        name = field['name']
        value = features.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f'Enter a value for "{name}". Missing inputs are not filled automatically.')
        if isinstance(value, (list, dict, tuple)):
            raise ValueError(f'"{name}" must contain a single value.')
        if field['type'] in ('numeric', 'boolean'):
            if field['type'] == 'boolean' and isinstance(value, str) and value.lower() in ('true', 'false'):
                value = 1 if value.lower() == 'true' else 0
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f'"{name}" must be a finite number.') from error
            if not math.isfinite(value):
                raise ValueError(f'"{name}" must be a finite number.')
            if field['type'] == 'boolean' and value not in (0, 1):
                raise ValueError(f'"{name}" must be true/false or 0/1.')
            bounds = field.get('range')
            if bounds and not bounds['min'] <= value <= bounds['max']:
                warnings.append(f'"{name}" is outside the training range [{bounds["min"]}, {bounds["max"]}]. This prediction extrapolates beyond observed inputs.')
        else:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f'"{name}" must contain a finite, non-missing value.')
            categories = field.get('categories') or []
            if categories and str(value) not in categories:
                warnings.append(f'"{name}" contains a category unseen during training. Saved encoding ignores that category; review the result.')
        validated[name] = value
    extra = sorted(set(features) - set(validated))
    if extra:
        warnings.append(f'Unused columns were ignored: {", ".join(extra)}.')
    return validated, warnings
