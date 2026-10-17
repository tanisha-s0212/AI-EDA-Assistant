from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


from sales_domain import is_date_part_column, pick_best_date_column, pick_best_revenue_column, score_date_column, score_revenue_column

DATE_PATTERNS = ['invoice_date', 'order_date', 'bill_date', 'year_month', 'date', 'period', 'start', 'time', 'created', 'ended', 'timestamp']
TARGET_PATTERNS = ['sale_free', 'sale_value', 'net_sales', 'total_value_sale', 'gmv', 'turnover', 'revenue', 'sales', 'total_value']
MODEL_PRIORITY = {'XGBoost': 0, 'Gradient Boosting': 1, 'Prophet': 2}
SUPPORTED_MODEL_TYPES = {'auto', 'gradient_boosting', 'xgboost', 'prophet'}


def get_config(frequency: str) -> dict[str, Any]:
    """Return the frequency-specific feature and validation configuration."""
    if frequency == 'weekly':
        return {
            'lags': [1, 2, 3],
            'rolling_windows': [3, 6],
            'rolling_std_windows': [3],
            'calendar': ['trend_index', 'month_number', 'quarter_number', 'weekday_number', 'is_month_end'],
            'min_rows': 20,
            'cv_splits': 5,
            'prophet_mode': 'multiplicative',
            'yearly_seasonality': False,
            'period_label': 'Week of YYYY-MM-DD',
            'period_unit': 'Week',
            'offset': pd.offsets.Week(weekday=0),
            'horizon_label': 'week horizon',
        }
    return {
        'lags': [1, 2, 3, 6, 12],
        'rolling_windows': [3, 6, 12],
        'rolling_std_windows': [3, 6],
        'calendar': ['trend_index', 'month_number', 'quarter_number', 'is_quarter_end', 'is_year_end'],
        'min_rows': 12,
        'cv_splits': 3,
        'prophet_mode': 'multiplicative',
        'yearly_seasonality': True,
        'period_label': 'Month of YYYY-MM',
        'period_unit': 'Month',
        'offset': pd.offsets.MonthBegin(1),
        'horizon_label': 'month horizon',
    }


def _parse_object_dates(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns to datetimes when enough values parse cleanly."""
    working = frame.copy()
    for column in working.select_dtypes(include=['object', 'string']).columns:
        parsed = pd.to_datetime(working[column], errors='coerce')
        if parsed.notna().sum() >= max(2, int(0.6 * len(working))):
            working[column] = parsed
    return working


def _detect_date_column(frame: pd.DataFrame) -> str | None:
    """Detect the best sales-domain date column."""
    return pick_best_date_column(frame.columns)


def _detect_target_column(frame: pd.DataFrame) -> str | None:
    """Detect the best sales/revenue target column using name score then variance."""
    return pick_best_revenue_column(frame.columns, frame=frame)


def load_and_detect(path: str | Path, date_col: str | None = None, target_col: str | None = None) -> tuple[pd.DataFrame, str, str]:
    """Load a parquet, csv, tsv, or excel file/directory and detect/override date and target columns.

    Parquet loads only the resolved date+target columns via Polars scan (TS-style IO),
    not the full wide frame.
    """
    source = Path(path)
    if source.is_dir():
        parquet_files = sorted(source.glob('*.parquet'))
        if not parquet_files:
            raise ValueError(f'No parquet files found in directory: {source}')
        # Directory of parquets: still prefer column-pruned Polars reads when possible.
        frames = []
        for file in parquet_files:
            frames.append(_load_parquet_date_target(file, date_col, target_col))
        frame = pd.concat([item[0] for item in frames], ignore_index=True)
        # Resolve columns once from first successful file metadata.
        _, resolved_date, resolved_target = frames[0]
        frame = _finalize_date_target_frame(frame, resolved_date, resolved_target)
        return frame, resolved_date, resolved_target

    suffix = source.suffix.lower()
    if suffix == '.parquet':
        frame, resolved_date, resolved_target = _load_parquet_date_target(source, date_col, target_col)
        frame = _finalize_date_target_frame(frame, resolved_date, resolved_target)
        return frame, resolved_date, resolved_target

    if suffix in ('.xlsx', '.xls'):
        frame = pd.read_excel(source)
    elif suffix == '.tsv':
        frame = pd.read_csv(source, sep='\t')
    else:
        frame = pd.read_csv(source)

    frame = _parse_object_dates(frame)
    resolved_date = (
        date_col
        if date_col and date_col in frame.columns and not is_date_part_column(date_col)
        else _detect_date_column(frame)
    )
    resolved_target = target_col if target_col in frame.columns else _detect_target_column(frame)
    if not resolved_date or not resolved_target:
        columns = ', '.join(map(str, frame.columns.tolist()))
        missing = []
        if not resolved_date:
            missing.append('date column')
        if not resolved_target:
            missing.append('target column')
        raise ValueError(f'Missing {" and ".join(missing)}. Available columns: {columns}')

    return _finalize_date_target_frame(frame[[resolved_date, resolved_target]].copy(), resolved_date, resolved_target), str(resolved_date), str(resolved_target)


def _resolve_parquet_date_target(
    path: Path,
    date_col: str | None,
    target_col: str | None,
) -> tuple[str, str]:
    import polars as pl

    schema_names = list(pl.scan_parquet(str(path)).collect_schema().names())
    resolved_date = (
        date_col
        if date_col and date_col in schema_names and not is_date_part_column(date_col)
        else pick_best_date_column(schema_names)
    )
    # Variance-based revenue pick needs a thin frame sample when target not provided.
    if target_col and target_col in schema_names:
        resolved_target = target_col
    else:
        sample = (
            pl.scan_parquet(str(path))
            .select(schema_names)
            .head(5_000)
            .collect()
            .to_pandas(use_pyarrow_extension_array=False)
        )
        resolved_target = pick_best_revenue_column(schema_names, frame=sample, exclude={resolved_date} if resolved_date else None)
    if not resolved_date or not resolved_target:
        missing = []
        if not resolved_date:
            missing.append('date column')
        if not resolved_target:
            missing.append('target column')
        raise ValueError(f'Missing {" and ".join(missing)}. Available columns: {", ".join(schema_names)}')
    return str(resolved_date), str(resolved_target)


def _load_parquet_date_target(
    path: Path,
    date_col: str | None,
    target_col: str | None,
) -> tuple[pd.DataFrame, str, str]:
    import polars as pl

    resolved_date, resolved_target = _resolve_parquet_date_target(path, date_col, target_col)
    frame = (
        pl.scan_parquet(str(path))
        .select([resolved_date, resolved_target])
        .collect()
        .to_pandas(use_pyarrow_extension_array=False)
    )
    return frame, resolved_date, resolved_target


def _finalize_date_target_frame(frame: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    working = frame[[date_col, target_col]].copy()
    working[date_col] = pd.to_datetime(working[date_col], errors='coerce')
    working[target_col] = pd.to_numeric(working[target_col], errors='coerce')
    working = working.dropna(subset=[date_col])
    # Aggregate only date+target (other columns are not used after prepare_frame_for_ml_pipeline).
    deduped = working.groupby(date_col, as_index=False).agg({target_col: 'sum'}).sort_values(date_col)
    return deduped


def load_and_detect_frame(frame: pd.DataFrame, date_col: str | None = None, target_col: str | None = None) -> tuple[pd.DataFrame, str, str]:
    """Detect and aggregate a supplied dataframe with optional column overrides."""
    working = _parse_object_dates(frame)
    resolved_date = (
        date_col
        # An explicit user selection is authoritative, including a column named
        # "month" that contains full dates.  Heuristics apply only to auto-detect.
        if date_col and date_col in working.columns
        else _detect_date_column(working)
    )
    resolved_target = target_col if target_col in working.columns else _detect_target_column(working)
    if not resolved_date or not resolved_target:
        raise ValueError(f'Missing date or target column. Available columns: {", ".join(map(str, working.columns.tolist()))}')
    return _finalize_date_target_frame(working, resolved_date, resolved_target), str(resolved_date), str(resolved_target)


def detect_frequency(df: pd.DataFrame, date_col: str) -> str:
    """Detect weekly or monthly frequency from the median day gap."""
    ordered = pd.to_datetime(df[date_col], errors='coerce').dropna().sort_values().drop_duplicates()
    if len(ordered) < 2:
        logger.warning('Frequency detection needs at least two dates; falling back to monthly.')
        return 'monthly'
    median_gap = float(ordered.diff().dropna().dt.days.median())
    if median_gap <= 10:
        return 'weekly'
    if median_gap <= 40:
        return 'monthly'
    logger.warning('Unsupported forecast frequency median gap %.2f days; falling back to monthly.', median_gap)
    return 'monthly'


def ml_minimum_periods_for_label(period_label: str) -> int:
    """ML feature configs only support weekly/monthly grains; keep mins aligned with get_config."""
    if period_label in {'day', 'week'}:
        return 20
    if period_label == 'month':
        return 12
    if period_label == 'quarter':
        return 8
    return 3


def period_label_to_ml_frequency(period_label: str) -> str:
    return 'weekly' if period_label in {'day', 'week'} else 'monthly'


def prepare_frame_for_ml_pipeline(
    frame: pd.DataFrame,
    date_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, str, str, str, str]:
    """Coarsen raw rows with shared prepare_sales_series helpers for train + chart.

    Returns (ml_frame, date_col, target_col, ml_frequency, period_label).
    """
    # Lazy import avoids circular import: main.py imports this module at startup.
    from main import (  # noqa: WPS433
        FREQ_FOR_PERIOD_LABEL,
        aggregate_sales_series_at_frequency,
        attach_sales_series_frequency_attrs,
        prepare_sales_series,
    )
    from fastapi import HTTPException

    try:
        series_frame, _freq, period_label = prepare_sales_series(
            frame,
            date_col,
            target_col,
            minimum_periods_for_label=ml_minimum_periods_for_label,
        )
    except HTTPException as error:
        detail = error.detail if isinstance(error.detail, str) else str(error.detail)
        raise ValueError(detail) from error

    inferred_period_label = str(series_frame.attrs.get('inferred_period_label') or period_label)

    # ML feature calendars are weekly|monthly only. Prefer week when daily series is dense enough.
    if period_label == 'day':
        working = frame[[date_col, target_col]].copy()
        working[date_col] = pd.to_datetime(working[date_col], errors='coerce')
        working[target_col] = pd.to_numeric(working[target_col], errors='coerce')
        working = working.dropna(subset=[date_col])
        week_frame = aggregate_sales_series_at_frequency(working, date_col, target_col, FREQ_FOR_PERIOD_LABEL['week'])
        month_frame = aggregate_sales_series_at_frequency(working, date_col, target_col, FREQ_FOR_PERIOD_LABEL['month'])
        if len(week_frame) >= ml_minimum_periods_for_label('week'):
            series_frame = week_frame
            period_label = 'week'
        elif len(month_frame) >= ml_minimum_periods_for_label('month'):
            series_frame = month_frame
            period_label = 'month'
        attach_sales_series_frequency_attrs(
            series_frame,
            period_label=period_label,
            inferred_period_label=inferred_period_label,
            frequency_auto_adjusted=period_label != inferred_period_label,
        )

    ml_frequency = period_label_to_ml_frequency(period_label)
    ml_frame = series_frame.rename(columns={'period': date_col, 'sales': target_col}).copy()
    ml_frame.attrs.update(dict(series_frame.attrs))
    return ml_frame, date_col, target_col, ml_frequency, period_label


def clean_data(df: pd.DataFrame, target_col: str, date_col: str, frequency: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sort, trim leading zero target rows, drop null targets, and enforce minimum rows."""
    config = get_config(frequency)
    rows_before = int(len(df))
    working = df.copy().sort_values(date_col).reset_index(drop=True)
    working[target_col] = pd.to_numeric(working[target_col], errors='coerce')
    trim_limit = int(rows_before * (0.6 if frequency == 'weekly' else 0.4))
    trim_index = 0
    while trim_index < min(trim_limit, len(working)) and float(working.loc[trim_index, target_col] or 0.0) == 0.0:
        trim_index += 1
    clean = working.iloc[trim_index:].dropna(subset=[target_col]).reset_index(drop=True)
    rows_remaining = int(len(clean))
    report = {'rows_before': rows_before, 'rows_trimmed': int(trim_index), 'rows_remaining': rows_remaining}
    logger.info('ML forecast clean_data report: %s', report)
    if rows_remaining < config['min_rows']:
        raise ValueError(f'Insufficient data after trimming: {rows_remaining} rows')
    return clean, report


def _add_calendar_features(frame: pd.DataFrame, date_col: str, frequency: str, config: dict[str, Any]) -> pd.DataFrame:
    """Add only the calendar features allowed for the detected frequency."""
    working = frame.copy()
    dates = pd.to_datetime(working[date_col])
    if 'trend_index' in config['calendar']:
        working['trend_index'] = np.arange(1, len(working) + 1, dtype=float)
    if 'month_number' in config['calendar']:
        working['month_number'] = dates.dt.month.astype(float)
    if 'quarter_number' in config['calendar']:
        working['quarter_number'] = dates.dt.quarter.astype(float)
    if frequency == 'weekly':
        working['weekday_number'] = dates.dt.dayofweek.astype(float)
        working['is_month_end'] = dates.dt.is_month_end.astype(int).astype(float)
    else:
        working['is_quarter_end'] = dates.dt.is_quarter_end.astype(int).astype(float)
        working['is_year_end'] = dates.dt.is_year_end.astype(int).astype(float)
    return working


def engineer_features(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    frequency: str,
    config: dict[str, Any],
    feature_groups: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build only forecast-time-safe, target-derived features.

    Source columns are deliberately excluded.  Values such as discounts, costs, or
    margins are often unavailable for future periods and can leak the target into
    validation.  This keeps training and future inference on the same inputs.
    """
    allowed_groups = {'trend', 'calendar', 'lags', 'rolling'}
    selected_groups = set(feature_groups or allowed_groups) & allowed_groups
    if not selected_groups:
        raise ValueError('Choose at least one ML feature group.')
    working = df[[date_col, target_col]].copy().sort_values(date_col).reset_index(drop=True)
    
    # 3. Add calendar features
    working = _add_calendar_features(working, date_col, frequency, config)
    target = pd.to_numeric(working[target_col], errors='coerce').astype(float)
    
    # Add lags from config
    for lag in config.get('lags', []):
        column = f'lag_{lag}'
        if column not in working.columns:
            working[column] = target.shift(lag)
            if frequency == 'monthly' and lag == 12 and len(working) <= 12:
                fallback = target.shift(1).expanding(min_periods=1).mean()
                working[column] = working[column].fillna(fallback).fillna(target.expanding(min_periods=1).mean())
                
    # Add rolling windows from config
    for window in config.get('rolling_windows', []):
        column = f'rolling_mean_{window}'
        if column not in working.columns:
            working[column] = target.shift(1).rolling(window=window, min_periods=window).mean()
            
    for window in config.get('rolling_std_windows', []):
        column = f'rolling_std_{window}'
        if column not in working.columns:
            working[column] = target.shift(1).rolling(window=window, min_periods=window).std().fillna(0.0)

    # 4. Generate the shared feature vocabulary. Selection happens below so future
    # rows and historical rows use exactly the same set of inputs.
    dates = pd.to_datetime(working[date_col])
    
    if 'trend_index' not in working.columns:
        working['trend_index'] = np.arange(1, len(working) + 1, dtype=float)
    if 'month_number' not in working.columns:
        working['month_number'] = dates.dt.month.astype(float)
    if 'quarter_number' not in working.columns:
        working['quarter_number'] = dates.dt.quarter.astype(float)
    if 'weekday_number' not in working.columns:
        working['weekday_number'] = dates.dt.dayofweek.astype(float)
    if 'is_month_end' not in working.columns:
        working['is_month_end'] = dates.dt.is_month_end.astype(int).astype(float)
    if 'lag_1' not in working.columns:
        working['lag_1'] = target.shift(1)
    if 'lag_2' not in working.columns:
        working['lag_2'] = target.shift(2)
    if 'lag_3' not in working.columns:
        working['lag_3'] = target.shift(3)
    if 'rolling_mean_3' not in working.columns:
        working['rolling_mean_3'] = target.shift(1).rolling(window=3, min_periods=1).mean()
    if 'rolling_mean_6' not in working.columns:
        working['rolling_mean_6'] = target.shift(1).rolling(window=6, min_periods=1).mean()
    if 'rolling_std_3' not in working.columns:
        working['rolling_std_3'] = target.shift(1).rolling(window=3, min_periods=1).std().fillna(0.0)

    # 5. Only include the feature groups selected in the UI.  Do not append raw
    # dataset columns: they are not reliably known at forecast time.
    feature_set: set[str] = set()
    if 'trend' in selected_groups:
        feature_set.add('trend_index')
    if 'calendar' in selected_groups:
        feature_set.update(config.get('calendar', []))
    if 'lags' in selected_groups:
        feature_set.update([f'lag_{lag}' for lag in config.get('lags', [])])
        feature_set.update({'lag_1', 'lag_2', 'lag_3'})
    if 'rolling' in selected_groups:
        feature_set.update([f'rolling_mean_{window}' for window in config.get('rolling_windows', [])])
        feature_set.update([f'rolling_std_{window}' for window in config.get('rolling_std_windows', [])])
        feature_set.update({'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3'})
    
    feature_list = [col for col in working.columns if col in feature_set and col not in (date_col, target_col)]
    
    feature_df = working.dropna(subset=[target_col]).copy()
    for column in feature_list:
        if feature_df[column].isna().any():
            fill_value = float(target.expanding(min_periods=1).mean().iloc[-1]) if len(target) else 0.0
            if pd.isna(fill_value):
                fill_value = 0.0
            feature_df[column] = feature_df[column].fillna(fill_value)
            
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_list).reset_index(drop=True)
    if feature_df.empty:
        raise ValueError('Insufficient data after feature engineering: 0 rows')
    constant_features = [column for column in feature_list if feature_df[column].nunique(dropna=False) <= 1]
    if constant_features:
        logger.info('Dropping constant forecast features: %s', constant_features)
        feature_list = [column for column in feature_list if column not in constant_features]
    if not feature_list:
        raise ValueError('Selected feature groups produced no varying forecast-time-safe features.')
        
    for column in feature_list:
        zero_share = float((feature_df[column] == 0).mean())
        if zero_share > 0.10:
            logger.warning('Feature column %s has %.1f%% zeros', column, zero_share * 100)
            
    return feature_df, feature_list


def compute_smape(actual: list[float] | np.ndarray, predicted: list[float] | np.ndarray) -> float:
    """Compute symmetric mean absolute percentage error with epsilon zero protection."""
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    denominator = np.maximum(np.abs(actual_arr) + np.abs(predicted_arr), 1e-8)
    return float(100.0 * np.mean((2.0 * np.abs(actual_arr - predicted_arr)) / denominator))


def compute_all_metrics(actual: list[float] | np.ndarray, predicted: list[float] | np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE, and SMAPE."""
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    mae = float(mean_absolute_error(actual_arr, predicted_arr))
    rmse = float(np.sqrt(mean_squared_error(actual_arr, predicted_arr)))
    non_zero = np.abs(actual_arr) > 1e-8
    mape = float(np.mean(np.abs((actual_arr[non_zero] - predicted_arr[non_zero]) / actual_arr[non_zero])) * 100.0) if np.any(non_zero) else 0.0
    return {'mae': round(mae, 2), 'rmse': round(rmse, 2), 'mape': round(mape, 2), 'smape': round(compute_smape(actual_arr, predicted_arr), 2)}


def build_models_dict() -> dict[str, Any]:
    """Build the production candidate dictionary."""
    models: dict[str, Any] = {'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.05, max_depth=3)}
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, objective='reg:squarederror', random_state=42, n_jobs=1)
    except Exception as error:
        models['XGBoost'] = {'status': 'failed', 'note': f'failed: {error}'}
    models['Prophet'] = {'type': 'prophet'}
    return models


def _fit_predict_prophet(train: pd.DataFrame, validation_dates: pd.Series, date_col: str, target_col: str, frequency: str) -> list[float]:
    """Fit Prophet inside a fold and predict validation dates."""
    try:
        from prophet import Prophet
    except Exception as error:
        raise RuntimeError(f'Prophet unavailable: {error}') from error
    config = get_config(frequency)
    model = Prophet(seasonality_mode=config['prophet_mode'], yearly_seasonality=config['yearly_seasonality'], weekly_seasonality=False, daily_seasonality=False)
    model.fit(train.rename(columns={date_col: 'ds', target_col: 'y'})[['ds', 'y']])
    forecast = model.predict(pd.DataFrame({'ds': pd.to_datetime(validation_dates)}))
    return [max(0.0, float(value)) for value in forecast['yhat'].tolist()]


def _recursive_holdout_predictions(
    model: Any,
    train: pd.DataFrame,
    test_dates: pd.Series,
    features: list[str],
    target: str,
    date_col: str,
    frequency: str,
) -> list[float]:
    """Forecast an unseen block without reading its actual target values."""
    if isinstance(model, dict) and model.get('type') == 'prophet':
        return _fit_predict_prophet(train, test_dates, date_col, target, frequency)
    estimator = clone(model)
    estimator.fit(train[features], train[target])
    config = get_config(frequency)
    history = pd.to_numeric(train[target], errors='coerce').astype(float).tolist()
    predictions: list[float] = []
    for test_date in pd.to_datetime(test_dates):
        row = _build_future_feature_row(history, pd.Timestamp(test_date), frequency, config, features)
        prediction = max(0.0, float(estimator.predict(row)[0]))
        predictions.append(prediction)
        history.append(prediction)
    return predictions


def walk_forward_validate(df: pd.DataFrame, features: list[str], target: str, models_dict: dict[str, Any], n_splits: int, frequency: str, date_col: str = 'period') -> dict[str, dict[str, Any]]:
    """Validate candidates in chronological folds without using fold targets as lags."""
    results: dict[str, dict[str, Any]] = {}
    splits = min(n_splits, max(2, len(df) - 2))
    splitter = TimeSeriesSplit(n_splits=splits)
    for model_name, model in models_dict.items():
        if isinstance(model, dict) and model.get('status') == 'failed':
            results[model_name] = {'mae': None, 'rmse': None, 'mape': None, 'smape': None, 'status': 'failed', 'note': model.get('note', 'unavailable')}
            continue
        actuals: list[float] = []
        predictions: list[float] = []
        try:
            for train_index, test_index in splitter.split(df):
                train = df.iloc[train_index].copy()
                test = df.iloc[test_index].copy()
                fold_predictions = _recursive_holdout_predictions(
                    model, train, test[date_col], features, target, date_col, frequency
                )
                actuals.extend(pd.to_numeric(test[target], errors='coerce').astype(float).tolist())
                predictions.extend(fold_predictions)
            metrics = compute_all_metrics(actuals, predictions)
            results[model_name] = {**metrics, 'status': 'completed', 'note': 'completed'}
        except Exception as error:
            results[model_name] = {'mae': None, 'rmse': None, 'mape': None, 'smape': None, 'status': 'failed', 'note': str(error)}
    return results


def auto_select_model(results_dict: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Select the completed model with lowest SMAPE and deterministic near-tie priority."""
    completed = [(name, metrics) for name, metrics in results_dict.items() if metrics.get('status') == 'completed']
    if not completed:
        raise ValueError('All models failed')
    completed.sort(key=lambda item: (float(item[1]['smape']), MODEL_PRIORITY.get(item[0], 99)))
    best_name, best_metrics = completed[0]
    if len(completed) > 1 and abs(float(completed[0][1]['smape']) - float(completed[1][1]['smape'])) < 2.0:
        near_ties = [item for item in completed if abs(float(item[1]['smape']) - float(best_metrics['smape'])) < 2.0]
        best_name, best_metrics = sorted(near_ties, key=lambda item: MODEL_PRIORITY.get(item[0], 99))[0]
    return best_name, best_metrics


def retrain_on_full(model: Any, feature_df: pd.DataFrame, feature_list: list[str], target_col: str, date_col: str = 'period', frequency: str = 'monthly') -> Any:
    """Retrain the selected model on all available feature rows."""
    if isinstance(model, dict) and model.get('type') == 'prophet':
        try:
            from prophet import Prophet
        except Exception as error:
            raise RuntimeError(f'Prophet unavailable: {error}') from error
        config = get_config(frequency)
        prophet = Prophet(seasonality_mode=config['prophet_mode'], yearly_seasonality=config['yearly_seasonality'], weekly_seasonality=False, daily_seasonality=False)
        prophet.fit(feature_df.rename(columns={date_col: 'ds', target_col: 'y'})[['ds', 'y']])
        return prophet
    estimator = clone(model)
    estimator.fit(feature_df[feature_list], feature_df[target_col])
    return estimator


def _build_future_row(history_frame: pd.DataFrame, date_col: str, target_col: str, frequency: str, config: dict[str, Any], next_date: pd.Timestamp) -> pd.DataFrame:
    """Create one future feature row by recomputing features on appended history."""
    placeholder = history_frame.iloc[[-1]].copy()
    placeholder[date_col] = next_date
    placeholder[target_col] = float(pd.to_numeric(history_frame[target_col], errors='coerce').mean())
    combined = pd.concat([history_frame, placeholder], ignore_index=True)
    engineered, features = engineer_features(combined, target_col, date_col, frequency, config)
    return engineered.iloc[[-1]][features]


def format_period_label(value: pd.Timestamp, frequency: str) -> str:
    """Format a period according to the detected frequency."""
    timestamp = pd.Timestamp(value)
    if frequency == 'weekly':
        return f"Week of {timestamp.strftime('%Y-%m-%d')}"
    return f"Month of {timestamp.strftime('%Y-%m')}"


def forecast_future(model: Any, df: pd.DataFrame, features: list[str], target: str, date_col: str, horizon: int, frequency: str) -> list[dict[str, Any]]:
    """Iteratively predict future periods without rebuilding the full history each step."""
    config = get_config(frequency)
    history = df[[date_col, target]].copy().sort_values(date_col).reset_index(drop=True)
    history_values = pd.to_numeric(history[target], errors='coerce').astype(float).tolist()
    forecasts: list[dict[str, Any]] = []
    next_date = pd.Timestamp(history.iloc[-1][date_col]) + config['offset']
    for _ in range(horizon):
        if model.__class__.__name__ == 'Prophet':
            prediction = float(model.predict(pd.DataFrame({'ds': [next_date]}))['yhat'].iloc[0])
        else:
            row = _build_future_feature_row(history_values, next_date, frequency, config, features)
            prediction = float(model.predict(row)[0])
        prediction = max(0.0, prediction)
        forecasts.append({'period': next_date.strftime('%Y-%m-%d'), 'period_label': format_period_label(next_date, frequency), 'forecast_value': round(prediction, 2)})
        history_values.append(prediction)
        next_date = next_date + config['offset']
    return forecasts


def _build_future_feature_row(
    history: list[float], next_date: pd.Timestamp, frequency: str,
    config: dict[str, Any], features: list[str],
) -> pd.DataFrame:
    """Calculate the engineered features for one future period in O(feature count)."""
    if not history:
        raise ValueError('At least one historical target value is required for forecasting.')
    values = np.asarray(history, dtype=float)
    fallback = float(np.mean(values))
    date = pd.Timestamp(next_date)
    row: dict[str, float] = {
        'trend_index': float(len(values) + 1),
        'month_number': float(date.month),
        'quarter_number': float(date.quarter),
        'weekday_number': float(date.dayofweek),
        'is_month_end': float(date.is_month_end),
        'is_quarter_end': float(date.is_quarter_end),
        'is_year_end': float(date.is_year_end),
    }
    requested_lags = {int(name.split('_', 1)[1]) for name in features if name.startswith('lag_') and name.split('_', 1)[1].isdigit()}
    for lag in set(config.get('lags', [])) | requested_lags:
        row[f'lag_{lag}'] = float(values[-lag]) if len(values) >= lag else fallback
    requested_means = {int(name.rsplit('_', 1)[1]) for name in features if name.startswith('rolling_mean_') and name.rsplit('_', 1)[1].isdigit()}
    for window in set(config.get('rolling_windows', [])) | requested_means:
        row[f'rolling_mean_{window}'] = float(np.mean(values[-window:])) if len(values) >= window else fallback
    requested_stds = {int(name.rsplit('_', 1)[1]) for name in features if name.startswith('rolling_std_') and name.rsplit('_', 1)[1].isdigit()}
    for window in set(config.get('rolling_std_windows', [])) | requested_stds:
        row[f'rolling_std_{window}'] = float(np.std(values[-window:], ddof=1)) if len(values) >= window else 0.0
    return pd.DataFrame([{feature: row.get(feature, fallback) for feature in features}], columns=features)


def compute_shap_importance(model: Any, X: pd.DataFrame, feature_names: list[str]) -> list[dict[str, Any]]:
    """Return normalized feature importance for tree models, or a Prophet proxy."""
    if model.__class__.__name__ == 'Prophet':
        names = ['trend', 'yearly', 'seasonality']
        values = np.asarray([0.5, 0.3, 0.2], dtype=float)
        return [{'feature': name, 'importance': round(float(value), 4)} for name, value in zip(names, values)]
    if hasattr(model, 'feature_importances_'):
        values = np.asarray(model.feature_importances_, dtype=float)[:len(feature_names)]
    else:
        values = np.ones(len(feature_names), dtype=float)
    total = float(values.sum())
    if total <= 0:
        values = np.ones(len(feature_names), dtype=float)
        total = float(values.sum())
    rows = [{'feature': feature_names[index], 'importance': round(float(values[index] / total), 4)} for index in range(len(feature_names))]
    return sorted(rows, key=lambda item: item['importance'], reverse=True)[:10]


def _naive_holdout_predictions(train_values: list[float], test_size: int, frequency: str) -> tuple[list[float], str]:
    """Return a transparent baseline for the same horizon used by the ML holdout."""
    if not train_values or test_size <= 0:
        return [], 'last-observation'
    season = 52 if frequency == 'weekly' else 12
    if len(train_values) >= season:
        history = list(train_values)
        predictions = []
        for _ in range(test_size):
            predictions.append(float(history[-season]))
            history.append(predictions[-1])
        return predictions, f'seasonal-naive ({season}-period)'
    return [float(train_values[-1])] * test_size, 'last-observation'


def _data_quality(clean_df: pd.DataFrame, target_col: str, date_col: str, frequency: str) -> dict[str, Any]:
    """Score the prepared series and make the reasons visible to business users."""
    config = get_config(frequency)
    values = pd.to_numeric(clean_df[target_col], errors='coerce')
    completeness = 1.0 - float(values.isna().mean())
    zero_penalty = min(0.5, float((values.fillna(0) == 0).mean()))
    period_score = min(1.0, len(clean_df) / max(1, config['min_rows'] * 2))
    nonpositive_share = float((values.fillna(0) <= 0).mean())
    expected_frequency = 'W-MON' if frequency == 'weekly' else 'MS'
    dates = pd.to_datetime(clean_df[date_col], errors='coerce').dropna().sort_values()
    expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq=expected_frequency) if len(dates) else pd.DatetimeIndex([])
    missing_periods = max(0, len(expected) - len(dates.unique()))
    gap_penalty = min(0.25, missing_periods / max(1, len(expected)))
    score = round(100.0 * ((0.40 * completeness) + (0.35 * period_score) + (0.15 * (1.0 - zero_penalty)) + (0.10 * (1.0 - gap_penalty))), 1)
    issues: list[str] = []
    if missing_periods:
        issues.append(f'{missing_periods} expected {frequency} period(s) are absent after aggregation.')
    if nonpositive_share > 0.10:
        issues.append(f'{nonpositive_share:.0%} of periods are zero or negative.')
    if len(clean_df) < config['min_rows'] * 2:
        issues.append('History is short for a stable seasonal forecast.')
    status = 'pass' if score >= 80 and not missing_periods else ('warning' if score >= 60 else 'fail')
    return {'score': score, 'status': status, 'missing_periods': missing_periods, 'issues': issues, 'nonpositive_share': nonpositive_share}


def _comparison_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert validation results into the artifact comparison contract."""
    rows = []
    for model_name in ['Gradient Boosting', 'Prophet', 'XGBoost']:
        metrics = results.get(model_name, {'status': 'failed', 'note': 'not evaluated'})
        rows.append({
            'model': model_name,
            'status': metrics.get('status', 'failed'),
            'mae': metrics.get('mae'),
            'rmse': metrics.get('rmse'),
            'mape': metrics.get('mape'),
            'smape': metrics.get('smape'),
            'note': metrics.get('note', ''),
        })
    return rows


def _holdout_backtest(model: Any, feature_df: pd.DataFrame, features: list[str], target_col: str) -> list[float | None]:
    """Build holdout predictions for the last 20 percent of engineered rows."""
    split = max(1, int(len(feature_df) * 0.8))
    predictions: list[float | None] = [None] * len(feature_df)
    if split >= len(feature_df):
        return predictions
    if model.__class__.__name__ == 'Prophet':
        forecast = model.predict(pd.DataFrame({'ds': pd.to_datetime(feature_df.iloc[split:]['period'])}))
        values = forecast['yhat'].tolist()
    else:
        estimator = clone(model)
        estimator.fit(feature_df.iloc[:split][features], feature_df.iloc[:split][target_col])
        values = estimator.predict(feature_df.iloc[split:][features]).tolist()
    for offset, value in enumerate(values):
        predictions[split + offset] = round(max(0.0, float(value)), 2)
    return predictions


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def indian_format(value: float) -> str:
    """Format numbers with Indian digit grouping for display text."""
    sign = '-' if value < 0 else ''
    integer, _, decimal = f'{abs(float(value)):.2f}'.partition('.')
    if len(integer) <= 3:
        grouped = integer
    else:
        grouped = integer[-3:]
        prefix = integer[:-3]
        while prefix:
            grouped = prefix[-2:] + ',' + grouped
            prefix = prefix[:-2]
    return f'{sign}{grouped}.{decimal}'


def write_all_outputs(
    input_path: str | Path,
    trim_report: dict[str, Any],
    results: dict[str, dict[str, Any]],
    best_model_name: str,
    best_metrics: dict[str, Any],
    feature_df: pd.DataFrame,
    forecast: list[dict[str, Any]],
    shap: list[dict[str, Any]],
    clean_df: pd.DataFrame,
    target_col: str,
    date_col: str,
    frequency: str,
    horizon: int,
    best_model: Any,
    feature_list: list[str],
    holdout_predictions: list[float],
    holdout_actuals: list[float],
    baseline_predictions: list[float],
    baseline_name: str,
    test_percentage: int,
    model_was_requested: bool,
    *,
    period_label: str | None = None,
) -> dict[str, Any]:
    """Write all ML forecast artifacts and return the API-compatible output bundle."""
    source = Path(input_path)
    output_dir = (source if source.is_dir() else source.parent) / 'ml_forecast_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    config = get_config(frequency)
    values = pd.to_numeric(clean_df[target_col], errors='coerce').astype(float)
    volatility = float(values.std() / values.mean()) if float(values.mean() or 0.0) != 0.0 else 0.0
    zero_value_share = float((values.fillna(0) <= 0).mean()) if len(values) else 0.0
    missing_share = float(values.isna().mean()) if len(values) else 0.0
    quality = _data_quality(clean_df, target_col, date_col, frequency)
    quality_score, quality_status = quality['score'], quality['status']
    baseline_metrics = compute_all_metrics(holdout_actuals, baseline_predictions)
    naive_mae = baseline_metrics['mae']
    top_driver = shap[0]['feature'] if shap else ''
    top_importance = float(shap[0]['importance']) if shap else 0.0
    improvement = round(((naive_mae - float(best_metrics['mae'])) / naive_mae) * 100.0, 2) if naive_mae else 0.0
    comparison = _comparison_rows(results)
    selection_note = (
        f'{best_model_name} was selected by the user; its validation metrics are shown alongside the baseline.'
        if model_was_requested else
        f'{best_model_name} selected by lowest SMAPE. Note: SMAPE used as primary metric to avoid near-zero denominator inflation.'
    )
    resolved_period_label = period_label or ('week' if frequency == 'weekly' else 'month')
    metadata = {
        'frequency': frequency,
        'detected_frequency_label': config['period_unit'],
        'period_label': resolved_period_label,
        'target_col': target_col,
        'date_col': date_col,
        'usable_periods': int(len(clean_df)),
        'volatility': round(volatility, 4),
        'zero_value_share': round(zero_value_share, 4),
        'missing_share': round(missing_share, 4),
        'training_split_pct': 100 - int(test_percentage),
        'test_split_pct': int(test_percentage),
        'horizon_periods': int(horizon),
        'generated_feature_count': int(len(feature_list)),
        'feature_policy': 'Only time and prior target values are used; source business columns are excluded to prevent future-data leakage.',
        'missing_periods': quality['missing_periods'],
        'pipeline_status': 'completed',
        'next_tab': 'loss_forecast',
        'retrain_available': True,
    }
    selected_model = {
        'model_name': best_model_name,
        'mae': best_metrics['mae'],
        'rmse': best_metrics['rmse'],
        'mape': best_metrics['mape'],
        'smape': best_metrics['smape'],
        'top_driver': top_driver,
        'top_driver_importance': top_importance,
        'naive_baseline_mae': naive_mae,
        'mae_improvement_pct': improvement,
        'selection_reason': (
            f'{best_model_name} was selected by the user.' if model_was_requested
            else f'{best_model_name} had the best SMAPE among completed candidates.'
        ),
        'selection_metric': 'SMAPE',
        'selection_note': selection_note,
        'data_quality_score': quality_score,
        'data_quality_status': quality_status,
        'baseline_name': baseline_name,
        'baseline_metrics': baseline_metrics,
        'evaluation_method': 'Chronological recursive holdout. Each holdout forecast is generated without its actual value or later actual values.',
    }
    residual_quantile = float(np.quantile(np.abs(np.asarray(holdout_actuals) - np.asarray(holdout_predictions)), 0.9)) if holdout_predictions else 0.0
    for step, row in enumerate(forecast, start=1):
        spread = residual_quantile * float(np.sqrt(step))
        row['lower'] = round(max(0.0, float(row['forecast_value']) - spread), 2)
        row['upper'] = round(float(row['forecast_value']) + spread, 2)
    forecast_line = []
    for _, row in clean_df.iterrows():
        forecast_line.append({'period': pd.Timestamp(row[date_col]).strftime('%Y-%m-%d'), 'actual': round(float(row[target_col]), 2), 'backtest': None, 'forecast': None, 'type': 'actual'})
    feature_dates = [pd.Timestamp(value).strftime('%Y-%m-%d') for value in feature_df[date_col].tolist()]
    holdout_start = len(feature_dates) - len(holdout_predictions)
    for offset, prediction in enumerate(holdout_predictions):
        forecast_line.append({'period': feature_dates[holdout_start + offset], 'actual': None, 'backtest': round(float(prediction), 2), 'forecast': None, 'type': 'backtest'})
    for row in forecast:
        forecast_line.append({
            'period': row['period'], 'actual': None, 'backtest': None,
            'forecast': row['forecast_value'], 'lower': row.get('lower'), 'upper': row.get('upper'),
            'type': 'forecast',
        })
    future_values = [float(row['forecast_value']) for row in forecast]
    peak_index = int(np.argmax(future_values)) if future_values else 0
    future_table = {
        'horizon_avg': round(float(np.mean(future_values)), 2) if future_values else 0.0,
        'peak_value': round(future_values[peak_index], 2) if future_values else 0.0,
        'peak_period': forecast[peak_index]['period'] if forecast else '',
        'first_period': forecast[0]['period'] if forecast else '',
        'last_period': forecast[-1]['period'] if forecast else '',
        'latest_actual': round(float(values.iloc[-1]), 2) if len(values) else 0.0,
        'horizon_label': config['horizon_label'],
        'rows': [{'period': row['period'], 'forecast': row['forecast_value'], 'lower': row.get('lower'), 'upper': row.get('upper')} for row in forecast],
    }
    sample_start = int(len(feature_df) * 0.4)
    sample_end = max(sample_start + 1, int(len(feature_df) * 0.5))
    sample_rows = feature_df.iloc[sample_start:sample_end].head(10)
    if sample_rows.empty:
        sample_rows = feature_df.tail(min(10, len(feature_df)))
    feature_sample = {'columns': feature_list, 'rows': safe_records(sample_rows[feature_list].round(4))}
    top3 = ', '.join(item['feature'] for item in shap[:3]) or 'engineered time features'
    insight = {
        'insight_text': f'ML forecasting selected {best_model_name} after chronological validation. Strongest drivers: {top3}. Recursive holdout MAE {best_metrics["mae"]}, RMSE {best_metrics["rmse"]}, SMAPE {best_metrics["smape"]}%.',
        'audit_trail': 'Candidates are compared with chronological walk-forward validation. The final holdout and baseline use the same recursive forecast window without future actual values.',
        'data_quality_score': quality_score,
        'data_quality_status': quality_status,
    }
    _write_json(output_dir / 'forecast_metadata.json', metadata)
    _write_json(output_dir / 'model_comparison.json', comparison)
    _write_json(output_dir / 'selected_model.json', selected_model)
    _write_json(output_dir / 'forecast_line.json', forecast_line)
    _write_json(output_dir / 'future_forecast_table.json', future_table)
    _write_json(output_dir / 'shap_importance.json', shap)
    _write_json(output_dir / 'feature_table_sample.json', feature_sample)
    _write_json(output_dir / 'forecast_insight.json', insight)
    pd.DataFrame(forecast_line).to_csv(output_dir / 'forecast_output.csv', index=False)
    feature_df.to_csv(output_dir / 'features_clean.csv', index=False)
    return {
        'output_dir': str(output_dir),
        'metadata': metadata,
        'model_comparison': comparison,
        'selected_model': selected_model,
        'forecast_line': forecast_line,
        'future_forecast_table': future_table,
        'shap_importance': shap,
        'feature_table_sample': feature_sample,
        'forecast_insight': insight,
        'trim_report': trim_report,
        'data_quality': quality,
    }


def safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize dataframe records with JSON-safe values."""
    return json.loads(frame.replace([np.inf, -np.inf], np.nan).fillna(0).to_json(orient='records'))


def print_summary(trim_report: dict[str, Any], results: dict[str, dict[str, Any]], best_model_name: str, best_metrics: dict[str, Any], forecast: list[dict[str, Any]]) -> None:
    """Print a concise Indian-formatted pipeline summary."""
    logger.info('Trim report: %s', trim_report)
    logger.info('Model results: %s', results)
    logger.info('Selected %s with MAE %s and SMAPE %.2f%%', best_model_name, indian_format(float(best_metrics['mae'])), float(best_metrics['smape']))
    logger.info('Future forecast: %s', [{'period': row['period'], 'forecast': indian_format(row['forecast_value'])} for row in forecast])


def run_full_pipeline(
    path: str | Path,
    target_col: str | None = None,
    date_col: str | None = None,
    horizon: int = 3,
    frequency: str = 'auto',
    frame: pd.DataFrame | None = None,
    model_type: str = 'auto',
    feature_groups: list[str] | None = None,
    lag_periods: int | None = None,
    test_percentage: int = 20,
) -> dict[str, Any]:
    """Run the complete Aroha IDA ML sales forecast pipeline with an 80/20 train-test split on the selected model."""
    normalized_model_type = str(model_type or 'auto').strip().lower().replace(' ', '_').replace('-', '_')
    if normalized_model_type not in SUPPORTED_MODEL_TYPES:
        supported = ', '.join(sorted(SUPPORTED_MODEL_TYPES))
        raise ValueError(f'Unsupported ML forecast model "{model_type}". Choose one of: {supported}.')
    if frame is None:
        raw_df, detected_date_col, detected_target_col = load_and_detect(path, date_col, target_col)
    else:
        raw_df, detected_date_col, detected_target_col = load_and_detect_frame(frame, date_col, target_col)
    date_col = date_col if date_col in raw_df.columns else detected_date_col
    target_col = target_col if target_col in raw_df.columns else detected_target_col

    # Coarsen with shared sales-series helpers so history/chart/train share one period grain.
    df, date_col, target_col, prepared_frequency, period_label = prepare_frame_for_ml_pipeline(
        raw_df, date_col, target_col
    )
    if frequency == 'auto':
        frequency = prepared_frequency
    elif frequency not in {'weekly', 'monthly'}:
        logger.warning('Unsupported frequency %s; using prepared frequency %s.', frequency, prepared_frequency)
        frequency = prepared_frequency
    if frequency not in {'weekly', 'monthly'}:
        frequency = 'monthly'
    config = dict(get_config(frequency))
    if lag_periods is not None:
        validated_lags = max(1, min(12, int(lag_periods)))
        config['lags'] = list(range(1, validated_lags + 1))
    clean_df, trim_report = clean_data(df, target_col, date_col, frequency)
    feature_df, feature_list = engineer_features(clean_df, target_col, date_col, frequency, config, feature_groups)
    if len(feature_df) < max(3, config['cv_splits'] + 2):
        raise ValueError(f'Insufficient data after feature engineering: {len(feature_df)} rows')
    models = build_models_dict()
    results = walk_forward_validate(feature_df, feature_list, target_col, models, config['cv_splits'], frequency, date_col)
    
    # Select the requested model or the best completed candidate.
    selected_model_name = 'Gradient Boosting'
    if normalized_model_type != 'auto':
        normalized_requested = normalized_model_type
        for key in models.keys():
            if key.lower().replace(' ', '_').replace('-', '_') == normalized_requested:
                selected_model_name = key
                break
    else:
        best_model_name, _ = auto_select_model(results)
        selected_model_name = best_model_name

    # Chronological holdout.  The requested split is used by both the model and
    # its baseline so the UI never reports a configuration it ignored.
    validated_test_percentage = max(10, min(50, int(test_percentage)))
    split_idx = int(len(feature_df) * (100 - validated_test_percentage) / 100)
    if split_idx < 1:
        split_idx = 1
    if split_idx >= len(feature_df):
        split_idx = len(feature_df) - 1

    train_df = feature_df.iloc[:split_idx]
    test_df = feature_df.iloc[split_idx:]

    chosen_model = models[selected_model_name]
    if isinstance(chosen_model, dict) and chosen_model.get('status') == 'failed':
        raise ValueError(f'{selected_model_name} is unavailable: {chosen_model.get("note", "unknown error")}')
    
    # Train and evaluate on split
    if isinstance(chosen_model, dict) and chosen_model.get('type') == 'prophet':
        try:
            from prophet import Prophet
        except Exception as error:
            raise RuntimeError(f'Prophet unavailable: {error}') from error
        
        prophet_model = Prophet(seasonality_mode=config['prophet_mode'], yearly_seasonality=config['yearly_seasonality'], weekly_seasonality=False, daily_seasonality=False)
        prophet_model.fit(train_df.rename(columns={date_col: 'ds', target_col: 'y'})[['ds', 'y']])
        
        test_preds = _recursive_holdout_predictions(
            chosen_model, train_df, test_df[date_col], feature_list, target_col, date_col, frequency
        )
        
        # Retrain on full
        full_model = Prophet(seasonality_mode=config['prophet_mode'], yearly_seasonality=config['yearly_seasonality'], weekly_seasonality=False, daily_seasonality=False)
        full_model.fit(feature_df.rename(columns={date_col: 'ds', target_col: 'y'})[['ds', 'y']])
        best_model = full_model
    else:
        test_preds = _recursive_holdout_predictions(
            chosen_model, train_df, test_df[date_col], feature_list, target_col, date_col, frequency
        )
        
        # Retrain on full
        full_model = clone(chosen_model)
        full_model.fit(feature_df[feature_list], feature_df[target_col])
        best_model = full_model

    # Compute metrics on test split
    holdout_actuals = pd.to_numeric(test_df[target_col], errors='coerce').astype(float).tolist()
    best_metrics = compute_all_metrics(holdout_actuals, test_preds)
    baseline_predictions, baseline_name = _naive_holdout_predictions(
        pd.to_numeric(train_df[target_col], errors='coerce').astype(float).tolist(), len(holdout_actuals), frequency
    )
    
    forecast = forecast_future(best_model, feature_df, feature_list, target_col, date_col, horizon, frequency)
    shap = compute_shap_importance(best_model, feature_df[feature_list], feature_list)
    outputs = write_all_outputs(
        path,
        trim_report,
        results,
        selected_model_name,
        best_metrics,
        feature_df,
        forecast,
        shap,
        clean_df,
        target_col,
        date_col,
        frequency,
        horizon,
        best_model,
        feature_list,
        test_preds,
        holdout_actuals,
        baseline_predictions,
        baseline_name,
        validated_test_percentage,
        normalized_model_type != 'auto',
        period_label=period_label,
    )
    print_summary(trim_report, results, selected_model_name, best_metrics, forecast)
    return outputs
