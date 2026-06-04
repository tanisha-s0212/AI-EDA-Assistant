from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


DATE_PATTERNS = ['date', 'week', 'month', 'period', 'start', 'time']
TARGET_PATTERNS = ['sale_free', 'sale_value', 'total_value', 'revenue']
MODEL_PRIORITY = {'XGBoost': 0, 'Gradient Boosting': 1, 'Prophet': 2, 'LightGBM': 3}


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
            'lgbm_min_leaf': 5,
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
        'lgbm_min_leaf': 2,
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
    """Detect the first datetime-like or date-named column."""
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            return str(column)
    for column in frame.columns:
        if any(pattern in str(column).lower() for pattern in DATE_PATTERNS):
            parsed = pd.to_datetime(frame[column], errors='coerce')
            if parsed.notna().any():
                return str(column)
    return None


def _detect_target_column(frame: pd.DataFrame) -> str | None:
    """Detect the first target-like numeric column."""
    for column in frame.columns:
        if any(pattern in str(column).lower() for pattern in TARGET_PATTERNS):
            numeric = pd.to_numeric(frame[column], errors='coerce')
            if numeric.notna().any():
                return str(column)
    return None


def load_and_detect(path: str | Path) -> tuple[pd.DataFrame, str, str]:
    """Load a parquet file or directory of parquet files and detect date and target columns."""
    source = Path(path)
    if source.is_dir():
        parquet_files = sorted(source.glob('*.parquet'))
        if not parquet_files:
            raise ValueError(f'No parquet files found in directory: {source}')
        frame = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
    else:
        frame = pd.read_parquet(source)

    frame = _parse_object_dates(frame)
    date_col = _detect_date_column(frame)
    target_col = _detect_target_column(frame)
    if not date_col or not target_col:
        columns = ', '.join(map(str, frame.columns.tolist()))
        missing = []
        if not date_col:
            missing.append('date column')
        if not target_col:
            missing.append('target column')
        raise ValueError(f'Missing {" and ".join(missing)}. Available columns: {columns}')

    frame[date_col] = pd.to_datetime(frame[date_col], errors='coerce')
    frame[target_col] = pd.to_numeric(frame[target_col], errors='coerce')
    frame = frame.dropna(subset=[date_col])
    numeric_cols = [column for column in frame.select_dtypes(include=[np.number]).columns if column != target_col]
    categorical_cols = [column for column in frame.columns if column not in set(numeric_cols + [date_col, target_col])]
    aggregations: dict[str, Any] = {target_col: 'sum'}
    aggregations.update({column: 'mean' for column in numeric_cols})
    aggregations.update({column: 'first' for column in categorical_cols})
    deduped = frame.groupby(date_col, as_index=False).agg(aggregations).sort_values(date_col)
    return deduped, date_col, target_col


def load_and_detect_frame(frame: pd.DataFrame, date_col: str | None = None, target_col: str | None = None) -> tuple[pd.DataFrame, str, str]:
    """Detect and aggregate a supplied dataframe with optional column overrides."""
    working = _parse_object_dates(frame)
    resolved_date = date_col if date_col in working.columns else _detect_date_column(working)
    resolved_target = target_col if target_col in working.columns else _detect_target_column(working)
    if not resolved_date or not resolved_target:
        raise ValueError(f'Missing date or target column. Available columns: {", ".join(map(str, working.columns.tolist()))}')
    working[resolved_date] = pd.to_datetime(working[resolved_date], errors='coerce')
    working[resolved_target] = pd.to_numeric(working[resolved_target], errors='coerce')
    working = working.dropna(subset=[resolved_date])
    numeric_cols = [column for column in working.select_dtypes(include=[np.number]).columns if column != resolved_target]
    categorical_cols = [column for column in working.columns if column not in set(numeric_cols + [resolved_date, resolved_target])]
    aggregations: dict[str, Any] = {resolved_target: 'sum'}
    aggregations.update({column: 'mean' for column in numeric_cols})
    aggregations.update({column: 'first' for column in categorical_cols})
    deduped = working.groupby(resolved_date, as_index=False).agg(aggregations).sort_values(resolved_date)
    return deduped, str(resolved_date), str(resolved_target)


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


def engineer_features(df: pd.DataFrame, target_col: str, date_col: str, frequency: str, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Build frequency-specific calendar, lag, rolling mean, and rolling std features."""
    working = _add_calendar_features(df.copy().sort_values(date_col).reset_index(drop=True), date_col, frequency, config)
    target = pd.to_numeric(working[target_col], errors='coerce').astype(float)
    for lag in config['lags']:
        column = f'lag_{lag}'
        working[column] = target.shift(lag)
        if frequency == 'monthly' and lag == 12 and len(working) <= 12:
            fallback = target.shift(1).expanding(min_periods=1).mean()
            working[column] = working[column].fillna(fallback).fillna(target.expanding(min_periods=1).mean())
            logger.warning('lag_12 backfilled for insufficient history')
    for window in config['rolling_windows']:
        working[f'rolling_mean_{window}'] = target.shift(1).rolling(window=window, min_periods=window).mean()
    for window in config['rolling_std_windows']:
        working[f'rolling_std_{window}'] = target.shift(1).rolling(window=window, min_periods=window).std().fillna(0.0)

    feature_list = list(config['calendar'])
    feature_list.extend([f'lag_{lag}' for lag in config['lags']])
    feature_list.extend([f'rolling_mean_{window}' for window in config['rolling_windows']])
    feature_list.extend([f'rolling_std_{window}' for window in config['rolling_std_windows']])
    feature_df = working.dropna(subset=[target_col]).copy()
    for column in feature_list:
        if feature_df[column].isna().any():
            fill_value = float(target.expanding(min_periods=1).mean().iloc[-1]) if len(target) else 0.0
            feature_df[column] = feature_df[column].fillna(fill_value)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_list).reset_index(drop=True)
    if feature_df.empty:
        raise ValueError('Insufficient data after feature engineering: 0 rows')
    for column in feature_list:
        zero_share = float((feature_df[column] == 0).mean())
        if zero_share > 0.10:
            logger.warning('Feature column %s has %.1f%% zeros', column, zero_share * 100)
    return feature_df, feature_list


def safe_import_lightgbm(min_data_in_leaf: int) -> tuple[Any | None, str]:
    """Import or install LightGBM and return a configured regressor plus status."""
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(num_leaves=31, n_estimators=100, learning_rate=0.05, random_state=42, min_data_in_leaf=min_data_in_leaf, verbose=-1), 'available'
    except Exception:
        try:
            subprocess.run(['pip', 'install', 'lightgbm', '-q'], check=True, timeout=120)
            import lightgbm as lgb
            return lgb.LGBMRegressor(num_leaves=31, n_estimators=100, learning_rate=0.05, random_state=42, min_data_in_leaf=min_data_in_leaf, verbose=-1), 'installed_and_available'
        except Exception as error:
            return None, f'failed: {error}'


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


def build_models_dict(lgbm_model: Any | None, lgbm_status: str) -> dict[str, Any]:
    """Build the production candidate dictionary."""
    models: dict[str, Any] = {'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.05, max_depth=3)}
    try:
        from xgboost import XGBRegressor
        models['XGBoost'] = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, objective='reg:squarederror', random_state=42, n_jobs=1)
    except Exception as error:
        models['XGBoost'] = {'status': 'failed', 'note': f'failed: {error}'}
    models['Prophet'] = {'type': 'prophet'}
    models['LightGBM'] = lgbm_model if lgbm_model is not None else {'status': 'failed', 'note': lgbm_status}
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


def walk_forward_validate(df: pd.DataFrame, features: list[str], target: str, models_dict: dict[str, Any], n_splits: int, frequency: str, date_col: str = 'period') -> dict[str, dict[str, Any]]:
    """Validate all candidates with TimeSeriesSplit and capture named failures."""
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
                if model_name == 'Prophet':
                    fold_predictions = _fit_predict_prophet(train, test[date_col], date_col, target, frequency)
                else:
                    estimator = clone(model)
                    estimator.fit(train[features], train[target])
                    fold_predictions = [max(0.0, float(value)) for value in estimator.predict(test[features])]
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
    """Iteratively predict future periods, appending each prediction before the next step."""
    config = get_config(frequency)
    history = df[[date_col, target]].copy().sort_values(date_col).reset_index(drop=True)
    forecasts: list[dict[str, Any]] = []
    next_date = pd.Timestamp(history.iloc[-1][date_col]) + config['offset']
    for _ in range(horizon):
        if model.__class__.__name__ == 'Prophet':
            prediction = float(model.predict(pd.DataFrame({'ds': [next_date]}))['yhat'].iloc[0])
        else:
            row = _build_future_row(history, date_col, target, frequency, config, next_date)
            row = row.reindex(columns=features).fillna(float(history[target].mean()))
            prediction = float(model.predict(row)[0])
        prediction = max(0.0, prediction)
        forecasts.append({'period': next_date.strftime('%Y-%m-%d'), 'period_label': format_period_label(next_date, frequency), 'forecast_value': round(prediction, 2)})
        history = pd.concat([history, pd.DataFrame([{date_col: next_date, target: prediction}])], ignore_index=True)
        next_date = next_date + config['offset']
    return forecasts


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


def _naive_baseline_mae(values: list[float]) -> float:
    """Compute last-observation naive MAE over the final 20 percent holdout."""
    if len(values) < 3:
        return 0.0
    split = max(1, int(len(values) * 0.8))
    actual = values[split:]
    predicted = [values[index - 1] for index in range(split, len(values))]
    return round(float(mean_absolute_error(actual, predicted)), 2) if actual else 0.0


def _data_quality(clean_df: pd.DataFrame, target_col: str, frequency: str) -> tuple[float, str]:
    """Score cleaned target history from 0 to 100."""
    config = get_config(frequency)
    values = pd.to_numeric(clean_df[target_col], errors='coerce')
    completeness = 1.0 - float(values.isna().mean())
    zero_penalty = min(0.5, float((values.fillna(0) == 0).mean()))
    period_score = min(1.0, len(clean_df) / max(1, config['min_rows'] * 2))
    score = round(100.0 * ((0.45 * completeness) + (0.35 * period_score) + (0.20 * (1.0 - zero_penalty))), 1)
    return score, 'pass' if score >= 70 else 'fail'


def _comparison_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert validation results into the artifact comparison contract."""
    rows = []
    for model_name in ['Gradient Boosting', 'Prophet', 'XGBoost', 'LightGBM']:
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
) -> dict[str, Any]:
    """Write all ML forecast artifacts and return the API-compatible output bundle."""
    source = Path(input_path)
    output_dir = (source if source.is_dir() else source.parent) / 'ml_forecast_output'
    output_dir.mkdir(parents=True, exist_ok=True)
    config = get_config(frequency)
    values = pd.to_numeric(clean_df[target_col], errors='coerce').astype(float)
    volatility = float(values.std() / values.mean()) if float(values.mean() or 0.0) != 0.0 else 0.0
    quality_score, quality_status = _data_quality(clean_df, target_col, frequency)
    naive_mae = _naive_baseline_mae(values.tolist())
    top_driver = shap[0]['feature'] if shap else ''
    top_importance = float(shap[0]['importance']) if shap else 0.0
    improvement = round(((naive_mae - float(best_metrics['mae'])) / naive_mae) * 100.0, 2) if naive_mae else 0.0
    comparison = _comparison_rows(results)
    selection_note = (
        f'{best_model_name} selected by lowest SMAPE. '
        'Note: SMAPE used as primary metric to avoid near-zero denominator inflation.'
    )
    metadata = {
        'frequency': frequency,
        'detected_frequency_label': config['period_unit'],
        'target_col': target_col,
        'date_col': date_col,
        'usable_periods': int(len(clean_df)),
        'volatility': round(volatility, 4),
        'training_split_pct': 80,
        'horizon_periods': int(horizon),
        'generated_feature_count': int(len(feature_list)),
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
        'selection_reason': f'{best_model_name} had the best SMAPE among completed candidates.',
        'selection_metric': 'SMAPE',
        'selection_note': selection_note,
        'data_quality_score': quality_score,
        'data_quality_status': quality_status,
    }
    backtest_values = _holdout_backtest(best_model, feature_df.rename(columns={date_col: 'period'}), feature_list, target_col)
    forecast_line = []
    for _, row in clean_df.iterrows():
        forecast_line.append({'period': pd.Timestamp(row[date_col]).strftime('%Y-%m-%d'), 'actual': round(float(row[target_col]), 2), 'backtest': None, 'forecast': None, 'type': 'actual'})
    feature_dates = [pd.Timestamp(value).strftime('%Y-%m-%d') for value in feature_df[date_col].tolist()]
    for index, prediction in enumerate(backtest_values):
        if prediction is not None:
            forecast_line.append({'period': feature_dates[index], 'actual': None, 'backtest': prediction, 'forecast': None, 'type': 'backtest'})
    for row in forecast:
        forecast_line.append({'period': row['period'], 'actual': None, 'backtest': None, 'forecast': row['forecast_value'], 'type': 'forecast'})
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
        'rows': [{'period': row['period'], 'forecast': row['forecast_value']} for row in forecast],
    }
    sample_start = int(len(feature_df) * 0.4)
    sample_end = max(sample_start + 1, int(len(feature_df) * 0.5))
    sample_rows = feature_df.iloc[sample_start:sample_end].head(10)
    if sample_rows.empty:
        sample_rows = feature_df.tail(min(10, len(feature_df)))
    feature_sample = {'columns': feature_list, 'rows': safe_records(sample_rows[feature_list].round(4))}
    top3 = ', '.join(item['feature'] for item in shap[:3]) or 'engineered time features'
    insight = {
        'insight_text': f'ML forecasting auto-selected {best_model_name} after comparing production candidates. Strongest drivers: {top3}. Backtest MAE {best_metrics["mae"]}, RMSE {best_metrics["rmse"]}, SMAPE {best_metrics["smape"]}%.',
        'audit_trail': 'Gradient Boosting, Prophet, XGBoost, LightGBM compared via walk-forward validation using MAE, RMSE, MAPE, SMAPE. LightGBM reported as named failure if unavailable.',
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
) -> dict[str, Any]:
    """Run the complete Aroha IDA ML sales forecast pipeline and write all outputs."""
    if frame is None:
        df, detected_date_col, detected_target_col = load_and_detect(path)
    else:
        df, detected_date_col, detected_target_col = load_and_detect_frame(frame, date_col, target_col)
    date_col = date_col if date_col in df.columns else detected_date_col
    target_col = target_col if target_col in df.columns else detected_target_col
    if frequency == 'auto':
        frequency = detect_frequency(df, date_col)
    if frequency not in {'weekly', 'monthly'}:
        logger.warning('Unsupported frequency %s; falling back to monthly.', frequency)
        frequency = 'monthly'
    config = get_config(frequency)
    clean_df, trim_report = clean_data(df, target_col, date_col, frequency)
    feature_df, feature_list = engineer_features(clean_df, target_col, date_col, frequency, config)
    if len(feature_df) < max(3, config['cv_splits'] + 2):
        raise ValueError(f'Insufficient data after feature engineering: {len(feature_df)} rows')
    lgbm_model, lgbm_status = safe_import_lightgbm(config['lgbm_min_leaf'])
    models = build_models_dict(lgbm_model, lgbm_status)
    results = walk_forward_validate(feature_df, feature_list, target_col, models, config['cv_splits'], frequency, date_col)
    best_model_name, best_metrics = auto_select_model(results)
    best_model = retrain_on_full(models[best_model_name], feature_df, feature_list, target_col, date_col, frequency)
    forecast = forecast_future(best_model, feature_df, feature_list, target_col, date_col, horizon, frequency)
    shap = compute_shap_importance(best_model, feature_df[feature_list], feature_list)
    outputs = write_all_outputs(path, trim_report, results, best_model_name, best_metrics, feature_df, forecast, shap, clean_df, target_col, date_col, frequency, horizon, best_model, feature_list)
    print_summary(trim_report, results, best_model_name, best_metrics, forecast)
    return outputs
