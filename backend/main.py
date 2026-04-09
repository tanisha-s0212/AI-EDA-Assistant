from __future__ import annotations

import io
import logging
import os
import time
import traceback
import uuid
import warnings
from datetime import date, datetime, time as dt_time
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)
DATASET_DIR = BASE_DIR / 'datasets'
DATASET_DIR.mkdir(exist_ok=True)
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)
TRAINING_N_JOBS = 1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.FileHandler(LOG_DIR / 'backend.log', encoding='utf-8'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def parse_allowed_origins(value: str | None) -> list[str]:
    default_origins = [
        'http://localhost:3000',
        'http://127.0.0.1:3000',
        'http://localhost:3001',
        'http://127.0.0.1:3001',
    ]

    if not value:
        return default_origins

    origins = [origin.strip() for origin in value.split(',') if origin.strip()]
    if '*' in origins:
        return ['*']
    return origins or default_origins


allowed_origins = parse_allowed_origins(os.environ.get('CORS_ALLOWED_ORIGINS'))
allow_credentials = '*' not in allowed_origins

app = FastAPI(title='AI-Assisted EDA & ML Backend', version='3.0.0')
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allow_credentials,
    allow_methods=['*'],
    allow_headers=['*'],
)
router = APIRouter(prefix='/api')
MODEL_CACHE: dict[str, dict[str, Any]] = {}
DATASET_CACHE: dict[str, dict[str, Any]] = {}

ProblemType = Literal['regression', 'classification']
TrainingMode = Literal['fast', 'balanced']

LARGE_DATASET_ROW_THRESHOLD = 20_000
VERY_LARGE_DATASET_ROW_THRESHOLD = 50_000
CV_SAMPLE_LIMIT = 3_000
VERY_LARGE_CV_SAMPLE_LIMIT = 1_500
TRAIN_SAMPLE_LIMIT = 30_000
VERY_LARGE_TRAIN_SAMPLE_LIMIT = 15_000
IMPORTANCE_SAMPLE_LIMIT = 800
VERY_LARGE_IMPORTANCE_SAMPLE_LIMIT = 300
PARQUET_PREVIEW_ROW_LIMIT = 20_000


def normalize_column_name(name: str) -> str:
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in name).strip('_').replace('__', '_')


def dataset_file_path(dataset_id: str, suffix: str = '.parquet') -> Path:
    return DATASET_DIR / f'{dataset_id}{suffix}'


def write_dataset_file(dataset_id: str, content: bytes, suffix: str = '.parquet') -> Path:
    target = dataset_file_path(dataset_id, suffix)
    target.write_bytes(content)
    return target


def write_cached_frame(dataset_id: str, frame: pd.DataFrame) -> Path:
    target = dataset_file_path(dataset_id, '.joblib')
    joblib.dump(frame, target)
    return target


def read_cached_frame(dataset_entry: dict[str, Any], columns: list[str] | None = None, n_rows: int | None = None) -> pd.DataFrame:
    frame_path = dataset_entry.get('frame_path')
    if not frame_path:
        raise HTTPException(status_code=400, detail='Cached dataset frame path is missing. Please upload the file again.')

    try:
        frame = joblib.load(frame_path)
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        if columns is not None:
            frame = frame.loc[:, columns]
        if n_rows is not None and n_rows > 0:
            frame = frame.head(n_rows)
        return frame.copy()
    except Exception as error:
        raise HTTPException(status_code=400, detail=f'Failed to load cached dataset: {error}') from error


def read_cached_parquet(dataset_entry: dict[str, Any], **kwargs: Any) -> pl.DataFrame:
    parquet_path = dataset_entry.get('parquet_path')
    if not parquet_path:
        raise HTTPException(status_code=400, detail='Cached parquet dataset path is missing. Please upload the file again.')

    try:
        return pl.read_parquet(parquet_path, **kwargs)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f'Failed to load cached parquet dataset: {error}') from error


def resolve_requested_columns(requested_columns: list[str], available_columns: list[str]) -> dict[str, str]:
    exact_matches = {column: column for column in available_columns}
    normalized_matches: dict[str, str] = {}
    for column in available_columns:
        normalized = normalize_column_name(column)
        normalized_matches.setdefault(normalized, column)

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for requested in requested_columns:
        if requested in exact_matches:
            resolved[requested] = requested
            continue

        normalized_requested = normalize_column_name(requested)
        matched = normalized_matches.get(normalized_requested)
        if matched is None:
            missing.append(requested)
            continue
        resolved[requested] = matched

    if missing:
        raise HTTPException(status_code=400, detail=f'Missing columns: {missing}')

    return resolved


def sample_training_rows(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
    random_state: int,
    stratify: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= max_rows:
        return X, y

    split_kwargs: dict[str, Any] = {
        'train_size': max_rows / len(X),
        'random_state': random_state,
    }
    if stratify is not None:
        split_kwargs['stratify'] = stratify

    sampled_X, _, sampled_y, _ = train_test_split(X, y, **split_kwargs)
    return sampled_X, sampled_y


def build_training_profile(row_count: int, requested_cv_folds: int, training_mode: TrainingMode) -> dict[str, int | bool | str]:
    profile = {
        'training_mode': training_mode,
        'cv_folds': min(requested_cv_folds, 5),
        'cv_sample_limit': 0,
        'train_sample_limit': 0,
        'importance_sample_limit': 0,
        'importance_repeats': 3,
        'skip_cv_for_large_dataset': False,
    }

    if row_count >= VERY_LARGE_DATASET_ROW_THRESHOLD:
        profile['cv_folds'] = min(requested_cv_folds, 3)
        profile['train_sample_limit'] = VERY_LARGE_TRAIN_SAMPLE_LIMIT
        profile['importance_sample_limit'] = VERY_LARGE_IMPORTANCE_SAMPLE_LIMIT
        profile['importance_repeats'] = 1
        if training_mode == 'fast':
            profile['cv_sample_limit'] = 0
            profile['skip_cv_for_large_dataset'] = True
        else:
            profile['cv_sample_limit'] = max(VERY_LARGE_CV_SAMPLE_LIMIT, 3000)
    elif row_count >= LARGE_DATASET_ROW_THRESHOLD:
        profile['cv_folds'] = min(requested_cv_folds, 4)
        profile['train_sample_limit'] = TRAIN_SAMPLE_LIMIT
        profile['importance_sample_limit'] = IMPORTANCE_SAMPLE_LIMIT
        profile['importance_repeats'] = 2
        if training_mode == 'fast':
            profile['cv_sample_limit'] = 0
            profile['skip_cv_for_large_dataset'] = True
        else:
            profile['cv_sample_limit'] = max(CV_SAMPLE_LIMIT, 5000)

    return profile


class TrainRequest(BaseModel):
    data: list[dict[str, Any]] = Field(default_factory=list)
    dataset_id: str | None = None
    target_column: str
    feature_columns: list[str]
    problem_type: ProblemType
    model_type: str
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = 42
    cv_folds: int = Field(default=5, ge=2, le=10)
    training_mode: TrainingMode = 'balanced'


class PredictRequest(BaseModel):
    model_id: str
    features: dict[str, Any]


class DatasetCacheRequest(BaseModel):
    file_name: str
    data: list[dict[str, Any]] = Field(default_factory=list)


class SalesForecastRequest(BaseModel):
    data: list[dict[str, Any]] = Field(default_factory=list)
    dataset_id: str | None = None
    date_column: str
    target_column: str
    forecast_periods: int = Field(default=3, ge=1, le=24)
    test_percentage: int = Field(default=20, ge=10, le=50)
    test_periods: int | None = Field(default=None, ge=1, le=24)
    lag_periods: int = Field(default=3, ge=1, le=12)
    model_type: str | None = None
    feature_groups: list[str] = Field(default_factory=lambda: ['trend', 'calendar', 'seasonality', 'lags', 'rolling'])


class CleaningLog(BaseModel):
    action: str
    detail: str
    timestamp: str


class CleaningJustificationRequest(BaseModel):
    logs: list[CleaningLog]
    totalRows: int
    totalColumns: int

class ParquetCleaningRequest(BaseModel):
    dataset_id: str
    remove_duplicates: bool = True
    handle_missing: bool = True
    convert_dates: bool = True
    standardize_names: bool = True


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    nonNull: int
    nullCount: int
    uniqueCount: int
    role: str


class PredictionHistoryItem(BaseModel):
    id: str
    prediction: str | float | int
    confidence: float | None = None
    features: dict[str, str | float | int] | None = None
    timestamp: str


class EdaStats(BaseModel):
    numericColumns: list[str] = Field(default_factory=list)
    categoricalColumns: list[str] = Field(default_factory=list)
    stats: dict[str, dict[str, float]] = Field(default_factory=dict)
    correlations: list[dict[str, float | str]] = Field(default_factory=list)



class UploadedModelPayload(BaseModel):
    name: str
    type: str
    target: str
    problem: str
    trainedAt: str
    metrics: dict[str, float] = Field(default_factory=dict)
    features: list[str] = Field(default_factory=list)


class SalesForecastPointPayload(BaseModel):
    period: str
    actual: float | None = None
    predicted: float | None = None


class SalesForecastMetricsPayload(BaseModel):
    mae: float
    rmse: float
    mape: float


class SalesForecastTrainingSummaryPayload(BaseModel):
    model_name: str
    total_periods: int
    train_periods: int
    test_periods: int
    train_percentage: float
    test_percentage: float
    forecast_periods: int
    lag_periods: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    last_observed_period: str


class SalesForecastResultPayload(BaseModel):
    date_column: str
    target_column: str
    frequency: str | None = None
    period_label: str | None = None
    history: list[SalesForecastPointPayload] = Field(default_factory=list)
    test_forecast: list[SalesForecastPointPayload] = Field(default_factory=list)
    future_forecast: list[SalesForecastPointPayload] = Field(default_factory=list)
    metrics: SalesForecastMetricsPayload
    training_summary: SalesForecastTrainingSummaryPayload
    analysis: str


class ReportPayload(BaseModel):
    fileName: str
    totalRows: int
    columns: list[ColumnInfo]
    duplicates: int
    memoryUsage: str
    cleaningLogs: list[CleaningLog]
    cleaningDone: bool
    cleanedRowCount: int
    targetColumn: str | None = None
    problemType: str
    selectedFeatures: list[str]
    selectedModel: str | None = None
    modelMetrics: dict[str, float] | None = None
    featureImportance: list[dict[str, Any]] | None = None
    aiInsights: str | None = None
    uploadedModel: UploadedModelPayload | None = None
    salesForecastResult: SalesForecastResultPayload | None = None
    predictionResult: str | float | int | None = None
    predictionAnalysis: str | None = None
    predictionProbabilities: dict[str, float] | None = None
    predictionHistory: list[PredictionHistoryItem] = Field(default_factory=list)
    edaStats: EdaStats = Field(default_factory=EdaStats)


REGRESSION_MODELS: dict[str, tuple[str, Any, dict[str, Any]]] = {
    'ridge_regression': ('Ridge Regression', Ridge, {'alpha': 1.0}),
    'lasso_regression': ('Lasso Regression', Lasso, {'alpha': 0.001, 'max_iter': 5000}),
    'elasticnet': ('Elastic Net', ElasticNet, {'alpha': 0.001, 'l1_ratio': 0.5, 'max_iter': 5000}),
    'random_forest': ('Random Forest', RandomForestRegressor, {'n_estimators': 100, 'min_samples_leaf': 2, 'n_jobs': TRAINING_N_JOBS}),
    'gradient_boosting': ('Gradient Boosting', GradientBoostingRegressor, {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3}),
    'svr': ('Support Vector Regression', SVR, {'kernel': 'rbf', 'C': 1.0}),
    'decision_tree': ('Decision Tree', DecisionTreeRegressor, {'max_depth': 8}),
    'knn_regressor': ('K-Nearest Neighbors', KNeighborsRegressor, {'n_neighbors': 7, 'weights': 'distance'}),
}

CLASSIFICATION_MODELS: dict[str, tuple[str, Any, dict[str, Any]]] = {
    'logistic_regression': ('Logistic Regression', LogisticRegression, {'max_iter': 2000, 'solver': 'lbfgs'}),
    'random_forest': ('Random Forest', RandomForestClassifier, {'n_estimators': 100, 'min_samples_leaf': 2, 'n_jobs': TRAINING_N_JOBS}),
    'gradient_boosting': ('Gradient Boosting', GradientBoostingClassifier, {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3}),
    'svm': ('Support Vector Machine', SVC, {'kernel': 'rbf', 'C': 1.0, 'probability': True}),
    'decision_tree': ('Decision Tree', DecisionTreeClassifier, {'max_depth': 8}),
    'knn': ('K-Nearest Neighbors', KNeighborsClassifier, {'n_neighbors': 7, 'weights': 'distance'}),
}


def safe_serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: safe_serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_serialize(v) for v in value]
    if isinstance(value, tuple):
        return [safe_serialize(v) for v in value]
    if isinstance(value, (datetime, date, dt_time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


def normalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in normalized.columns:
        series = normalized[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            normalized[column] = series.dt.strftime('%Y-%m-%d %H:%M:%S').where(series.notna(), np.nan).astype(object)
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            cleaned = series.astype(object)
            cleaned = cleaned.where(pd.notna(cleaned), np.nan)
            cleaned = cleaned.replace(r'^\s*$', np.nan, regex=True)
            normalized[column] = cleaned
    return normalized


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric_features = frame.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    categorical_features = [column for column in frame.columns if column not in numeric_features]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                'numeric',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                ]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                'categorical',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10, max_categories=100)),
                ]),
                categorical_features,
            )
        )

    if not transformers:
        raise HTTPException(status_code=400, detail='No usable features found for training.')

    return ColumnTransformer(transformers=transformers)


def build_estimator(problem_type: ProblemType, model_type: str, random_state: int):
    registry = REGRESSION_MODELS if problem_type == 'regression' else CLASSIFICATION_MODELS
    if model_type not in registry:
        raise HTTPException(status_code=400, detail=f"Model '{model_type}' is not available for {problem_type}.")

    model_name, estimator_cls, params = registry[model_type]
    params = dict(params)
    if 'random_state' in estimator_cls().get_params().keys():
        params['random_state'] = random_state
    estimator = estimator_cls(**params)
    return model_name, estimator


def normalize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in normalized.columns:
        if normalized[column].dtype == 'object':
            normalized[column] = normalized[column].replace(r'^\s*$', np.nan, regex=True)
    return normalized


def load_dataset_frame(dataset_id: str | None, data: list[dict[str, Any]], required_columns: list[str]) -> pd.DataFrame:
    if dataset_id:
        dataset_entry = DATASET_CACHE.get(dataset_id)
        if dataset_entry is None:
            raise HTTPException(status_code=400, detail='Cached dataset not found. Please upload the file again.')

        available_columns = list(dataset_entry['columns'])
        resolved_columns = resolve_requested_columns(required_columns, available_columns)
        resolved_selected_columns = [resolved_columns[column] for column in required_columns]

        if dataset_entry.get('frame_path'):
            frame = read_cached_frame(dataset_entry, columns=resolved_selected_columns)
            frame.columns = required_columns
            return normalize_dataframe(frame)

        parquet_frame = read_cached_parquet(dataset_entry, columns=resolved_selected_columns, low_memory=True)
        parquet_frame.columns = required_columns
        return normalize_dataframe(parquet_frame.to_pandas(use_pyarrow_extension_array=False))

    if not data:
        raise HTTPException(status_code=400, detail='Dataset rows are required.')

    frame = normalize_dataframe(pd.DataFrame(data))
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f'Missing columns: {missing_columns}')
    return frame[required_columns].copy()


def build_polars_datetime_expr(column_name: str, dtype: pl.DataType) -> pl.Expr:
    column = pl.col(column_name)
    if dtype in pl.TEMPORAL_DTYPES:
        return column.cast(pl.Datetime, strict=False)

    text_column = column.cast(pl.String, strict=False).str.strip_chars()
    return pl.coalesce([
        text_column.str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S%.f', strict=False),
        text_column.str.strptime(pl.Datetime, '%Y-%m-%d %H:%M:%S', strict=False),
        text_column.str.strptime(pl.Datetime, '%Y-%m-%dT%H:%M:%S%.f', strict=False),
        text_column.str.strptime(pl.Datetime, '%Y-%m-%dT%H:%M:%S', strict=False),
        text_column.str.strptime(pl.Datetime, '%Y-%m-%d', strict=False),
        text_column.str.strptime(pl.Datetime, '%d-%m-%Y', strict=False),
        text_column.str.strptime(pl.Datetime, '%m-%d-%Y', strict=False),
        text_column.str.strptime(pl.Datetime, '%d/%m/%Y', strict=False),
        text_column.str.strptime(pl.Datetime, '%m/%d/%Y', strict=False),
        text_column.str.strptime(pl.Datetime, '%Y/%m/%d', strict=False),
        text_column.str.strptime(pl.Datetime, '%b %d, %Y', strict=False),
        text_column.str.strptime(pl.Datetime, '%B %d, %Y', strict=False),
        text_column.str.strptime(pl.Datetime, '%b-%Y', strict=False),
        text_column.str.strptime(pl.Datetime, '%B-%Y', strict=False),
        text_column.str.strptime(pl.Datetime, '%b %Y', strict=False),
        text_column.str.strptime(pl.Datetime, '%B %Y', strict=False),
        text_column.str.strptime(pl.Datetime, '%Y-%m', strict=False),
        text_column.str.strptime(pl.Datetime, '%m-%Y', strict=False),
    ])


def prepare_sales_series_from_parquet(dataset_entry: dict[str, Any], date_column: str, target_column: str) -> tuple[pd.DataFrame, str, str]:
    parquet_path = dataset_entry.get('parquet_path')
    if not parquet_path:
        raise HTTPException(status_code=400, detail='Cached parquet dataset path is missing. Please upload the file again.')

    try:
        lazy_frame = pl.scan_parquet(parquet_path).select([
            pl.col(date_column).alias(date_column),
            pl.col(target_column).alias(target_column),
        ])
        schema = lazy_frame.collect_schema()
        parsed_date_expr = build_polars_datetime_expr(date_column, schema[date_column])
        sample_dates = lazy_frame.select(parsed_date_expr.alias('__parsed_date')).limit(5000).collect(streaming=True).to_series()
        parsed_sample = pd.to_datetime(pd.Series(sample_dates.to_list()), errors='coerce')
        parsed_sample = parsed_sample.dropna()
        if parsed_sample.empty:
            raise HTTPException(status_code=400, detail='No valid rows remained after parsing the date and sales columns.')

        freq, period_label = infer_sales_time_frequency(parsed_sample)
        period_freq = {'MS': '1mo', 'QS': '1q', 'YS': '1y', 'D': '1d', 'W-MON': '1w'}.get(freq, '1mo')

        aggregated = (
            lazy_frame
            .with_columns([
                parsed_date_expr.alias('__parsed_date'),
                pl.col(target_column).cast(pl.Float64, strict=False).alias('__parsed_sales'),
            ])
            .drop_nulls(['__parsed_date', '__parsed_sales'])
            .group_by_dynamic('__parsed_date', every=period_freq, label='left')
            .agg(pl.col('__parsed_sales').sum().alias('sales'))
            .sort('__parsed_date')
            .collect(streaming=True)
        )
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=f'Failed to prepare parquet sales series: {error}') from error

    if aggregated.height == 0:
        raise HTTPException(status_code=400, detail='No valid rows remained after parsing the date and sales columns.')

    series_frame = aggregated.rename({'__parsed_date': 'period'}).to_pandas(use_pyarrow_extension_array=False)
    full_range = pd.date_range(series_frame['period'].min(), series_frame['period'].max(), freq=freq)
    series_frame = series_frame.set_index('period').reindex(full_range, fill_value=0.0).rename_axis('period').reset_index()
    series_frame['sales'] = series_frame['sales'].astype(float)

    if len(series_frame) < 6:
        raise HTTPException(status_code=400, detail=f'Sales forecasting needs at least 6 {period_label} periods after aggregation.')

    return series_frame, freq, period_label


def infer_sales_time_frequency(dates: pd.Series) -> tuple[str, str]:
    ordered = pd.Series(pd.to_datetime(dates, errors='coerce')).dropna().sort_values().drop_duplicates()
    if len(ordered) < 2:
        return 'MS', 'month'

    deltas = ordered.diff().dropna().dt.total_seconds() / 86400.0
    median_days = float(deltas.median()) if not deltas.empty else 30.0

    if median_days <= 2:
        return 'D', 'day'
    if median_days <= 10:
        return 'W-MON', 'week'
    if median_days <= 45:
        return 'MS', 'month'
    if median_days <= 120:
        return 'QS', 'quarter'
    return 'YS', 'year'


def format_forecast_period(value: pd.Timestamp, period_label: str) -> str:
    timestamp = pd.Timestamp(value)
    if period_label == 'day':
        return timestamp.strftime('%Y-%m-%d')
    if period_label == 'week':
        return f"Week of {timestamp.strftime('%Y-%m-%d')}"
    if period_label == 'month':
        return timestamp.strftime('%Y-%m')
    if period_label == 'quarter':
        quarter = ((timestamp.month - 1) // 3) + 1
        return f"{timestamp.year}-Q{quarter}"
    return timestamp.strftime('%Y')


def prepare_sales_series(frame: pd.DataFrame, date_column: str, target_column: str) -> tuple[pd.DataFrame, str, str]:
    working = frame[[date_column, target_column]].copy()
    working[date_column] = pd.to_datetime(working[date_column], errors='coerce')
    working[target_column] = pd.to_numeric(working[target_column], errors='coerce')
    working = working.dropna(subset=[date_column, target_column])

    if working.empty:
        raise HTTPException(status_code=400, detail='No valid rows remained after parsing the date and sales columns.')

    freq, period_label = infer_sales_time_frequency(working[date_column])
    period_freq = {'MS': 'M', 'QS': 'Q', 'YS': 'Y'}.get(freq, freq)
    period_index = working[date_column].dt.to_period(period_freq).dt.to_timestamp()
    working = working.assign(period=period_index)
    series_frame = working.groupby('period', as_index=False)[target_column].sum().sort_values('period')
    series_frame = series_frame.rename(columns={target_column: 'sales'})

    full_range = pd.date_range(series_frame['period'].min(), series_frame['period'].max(), freq=freq)
    series_frame = series_frame.set_index('period').reindex(full_range, fill_value=0.0).rename_axis('period').reset_index()
    series_frame['sales'] = series_frame['sales'].astype(float)

    if len(series_frame) < 6:
        raise HTTPException(status_code=400, detail=f'Sales forecasting needs at least 6 {period_label} periods after aggregation.')

    return series_frame, freq, period_label


def build_forecast_feature_row(history: list[float], current_period: pd.Timestamp, lag_periods: int) -> dict[str, float]:
    if len(history) < lag_periods:
        raise ValueError('Not enough history to build forecast features.')

    month_number = float(current_period.month)
    quarter_number = float(current_period.quarter)
    day_of_month = float(current_period.day)
    day_of_week = float(current_period.dayofweek)

    row: dict[str, float] = {
        'trend_index': float(len(history) + 1),
        'month_number': month_number,
        'quarter_number': quarter_number,
        'day_of_month': day_of_month,
        'day_of_week': day_of_week,
        'month_sin': float(np.sin(2 * np.pi * month_number / 12)),
        'month_cos': float(np.cos(2 * np.pi * month_number / 12)),
        'quarter_sin': float(np.sin(2 * np.pi * quarter_number / 4)),
        'quarter_cos': float(np.cos(2 * np.pi * quarter_number / 4)),
        'lag_mean': float(np.mean(history[-lag_periods:])),
        'lag_last_3_mean': float(np.mean(history[-min(3, len(history)):])),
    }

    for lag_index in range(1, lag_periods + 1):
        row[f'lag_{lag_index}'] = float(history[-lag_index])

    return row


def build_forecast_training_frame(series_frame: pd.DataFrame, lag_periods: int) -> tuple[pd.DataFrame, pd.Series]:
    rows: list[dict[str, float]] = []
    targets: list[float] = []
    periods = series_frame['period'].tolist()
    values = series_frame['sales'].astype(float).tolist()

    for index in range(lag_periods, len(values)):
        history = values[:index]
        rows.append(build_forecast_feature_row(history, periods[index], lag_periods))
        targets.append(float(values[index]))

    if len(rows) < 3:
        raise HTTPException(status_code=400, detail='Not enough history is available to train the time-series model.')

    return pd.DataFrame(rows), pd.Series(targets)


def recursive_forecast(model: LinearRegression, history: list[float], start_period: pd.Timestamp, periods: int, lag_periods: int, freq: str, period_label: str) -> list[dict[str, Any]]:
    forecasts: list[dict[str, Any]] = []
    running_history = list(history)
    current_period = pd.Timestamp(start_period)

    for _ in range(periods):
        features = build_forecast_feature_row(running_history, current_period, lag_periods)
        prediction = float(model.predict(pd.DataFrame([features]))[0])
        prediction = max(prediction, 0.0)
        forecasts.append({
            'period': format_forecast_period(current_period, period_label),
            'predicted': round(prediction, 2),
        })
        running_history.append(prediction)
        current_period = current_period + pd.tseries.frequencies.to_offset(freq)

    return forecasts



def assess_overfitting(problem_type: ProblemType, metrics: dict[str, Any]) -> dict[str, Any]:
    train_metric_name = 'train_r2' if problem_type == 'regression' else 'train_accuracy'
    test_metric_name = 'test_r2' if problem_type == 'regression' else 'test_accuracy'

    train_score = float(metrics.get(train_metric_name, 0.0) or 0.0)
    test_score = float(metrics.get(test_metric_name, 0.0) or 0.0)
    cv_mean = float(metrics.get('cv_mean', 0.0) or 0.0)
    cv_available = bool(metrics.get('cv_scores'))

    generalization_gap = round(train_score - test_score, 6)
    cv_gap = round(train_score - cv_mean, 6) if cv_available else None

    status = 'healthy'
    explanation = 'Train and validation performance are reasonably aligned.'

    if problem_type == 'regression':
        if train_score >= 0.85 and (generalization_gap >= 0.15 or (cv_gap is not None and cv_gap >= 0.12)):
            status = 'detected'
            explanation = 'Training R2 is much higher than test/CV R2, suggesting the model is memorizing the training set.'
        elif train_score >= 0.75 and (generalization_gap >= 0.08 or (cv_gap is not None and cv_gap >= 0.08)):
            status = 'watch'
            explanation = 'There is a noticeable train-to-validation gap. Review feature leakage, model complexity, or test size.'
    else:
        if train_score >= 0.95 and (generalization_gap >= 0.08 or (cv_gap is not None and cv_gap >= 0.08)):
            status = 'detected'
            explanation = 'Training accuracy is materially above test/CV accuracy, which is a strong overfitting signal.'
        elif train_score >= 0.85 and (generalization_gap >= 0.04 or (cv_gap is not None and cv_gap >= 0.04)):
            status = 'watch'
            explanation = 'The model performs better on training than on held-out data. Monitor for overfitting.'

    if not cv_available and status == 'healthy':
        explanation = 'No strong overfitting signal was found from the train/test comparison.'

    return {
        'status': status,
        'detected': status == 'detected',
        'generalization_gap': generalization_gap,
        'cv_gap': cv_gap,
        'train_score': round(train_score, 6),
        'test_score': round(test_score, 6),
        'explanation': explanation,
    }


def calculate_forecast_metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    if not actual or not predicted:
        return {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0}

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    percentage_errors = [abs((a - p) / a) for a, p in zip(actual, predicted) if a != 0]
    mape = float(np.mean(percentage_errors) * 100) if percentage_errors else 0.0
    return {
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'mape': round(mape, 2),
    }


def make_analysis(problem_type: ProblemType, model_name: str, metrics: dict[str, Any], importances: list[dict[str, Any]]) -> str:
    top_features = ', '.join(item['name'] for item in importances[:3]) or 'No dominant features detected'
    if problem_type == 'regression':
        return (
            f"### {model_name} Summary\n"
            f"- R2: {metrics['primary'].get('R2', 0):.4f}\n"
            f"- RMSE: {metrics['primary'].get('RMSE', 0):.4f}\n"
            f"- MAE: {metrics['primary'].get('MAE', 0):.4f}\n"
            f"- CV Mean: {metrics.get('cv_mean', 0):.4f}\n"
            f"- Top features: {top_features}"
        )
    return (
        f"### {model_name} Summary\n"
        f"- Accuracy: {metrics['primary'].get('Accuracy', 0):.4f}\n"
        f"- Precision: {metrics['primary'].get('Precision', 0):.4f}\n"
        f"- Recall: {metrics['primary'].get('Recall', 0):.4f}\n"
        f"- F1 Score: {metrics['primary'].get('F1 Score', 0):.4f}\n"
        f"- Top features: {top_features}"
    )


def save_model_bundle(model_id: str, bundle: dict[str, Any]) -> None:
    joblib.dump(bundle, MODEL_DIR / f'{model_id}.joblib')
    MODEL_CACHE[model_id] = bundle


def load_model_bundle(model_id: str) -> dict[str, Any]:
    if model_id in MODEL_CACHE:
        return MODEL_CACHE[model_id]
    model_path = MODEL_DIR / f'{model_id}.joblib'
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' was not found.")
    bundle = joblib.load(model_path)
    MODEL_CACHE[model_id] = bundle
    return bundle


def infer_problem_type_from_estimator(estimator: Any) -> ProblemType:
    estimator_type = getattr(estimator, '_estimator_type', None)
    if estimator_type == 'classifier' or hasattr(estimator, 'predict_proba'):
        return 'classification'
    return 'regression'


def normalize_uploaded_bundle(raw_bundle: Any, filename: str) -> dict[str, Any]:
    if isinstance(raw_bundle, dict):
        pipeline = raw_bundle.get('pipeline') or raw_bundle.get('model') or raw_bundle.get('estimator')
        feature_columns = raw_bundle.get('feature_columns') or raw_bundle.get('features') or []
        target_column = raw_bundle.get('target_column') or raw_bundle.get('target') or 'prediction_target'
        problem_type = raw_bundle.get('problem_type')
        model_type = raw_bundle.get('model_type')
        model_name = raw_bundle.get('model_name')
        label_encoder = raw_bundle.get('label_encoder')
        trained_at = raw_bundle.get('trained_at') or datetime.utcnow().isoformat()
    else:
        pipeline = raw_bundle
        feature_columns = getattr(raw_bundle, 'feature_names_in_', [])
        target_column = 'prediction_target'
        problem_type = None
        model_type = type(raw_bundle).__name__.lower()
        model_name = type(raw_bundle).__name__
        label_encoder = None
        trained_at = datetime.utcnow().isoformat()
    if pipeline is None or not hasattr(pipeline, 'predict'):
        raise HTTPException(status_code=400, detail='Uploaded file must contain a scikit-learn compatible model or pipeline with a predict() method.')
    if not problem_type:
        problem_type = infer_problem_type_from_estimator(getattr(pipeline, 'named_steps', {}).get('model', pipeline))
    if not model_name:
        model_name = type(getattr(pipeline, 'named_steps', {}).get('model', pipeline)).__name__
    if not model_type:
        model_type = model_name.lower().replace(' ', '_')
    return {
        'pipeline': pipeline,
        'feature_columns': list(feature_columns),
        'target_column': target_column,
        'problem_type': problem_type,
        'model_type': model_type,
        'model_name': model_name,
        'label_encoder': label_encoder,
        'trained_at': trained_at,
    }


@router.post('/train')
def train_model(request: TrainRequest) -> JSONResponse:
    logger.info(
        'Train request received problem_type=%s model_type=%s dataset_id=%s feature_count=%s test_size=%s cv_folds=%s training_mode=%s',
        request.problem_type,
        request.model_type,
        request.dataset_id,
        len(request.feature_columns),
        request.test_size,
        request.cv_folds,
        request.training_mode,
    )
    selected_columns = [*request.feature_columns, request.target_column]
    data_frame = load_dataset_frame(request.dataset_id, request.data, selected_columns)
    data_frame = data_frame.dropna(subset=[request.target_column])
    if len(data_frame) < 10:
        raise HTTPException(status_code=400, detail='At least 10 valid rows are required for training.')

    X = normalize_feature_frame(data_frame[request.feature_columns].copy())
    y_raw = data_frame[request.target_column].copy()
    label_encoder: LabelEncoder | None = None
    if request.problem_type == 'classification':
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y_raw.astype(str)), index=y_raw.index)
        class_counts = y.value_counts()
        if class_counts.shape[0] < 2:
            raise HTTPException(status_code=400, detail='Classification requires at least two target classes.')
        stratify = y if class_counts.min() >= 2 else None
    else:
        y = pd.to_numeric(y_raw, errors='coerce')
        valid_numeric = y.notna()
        X = X.loc[valid_numeric]
        y = y.loc[valid_numeric]
        stratify = None
        if len(X) < 10:
            raise HTTPException(status_code=400, detail='Regression requires at least 10 numeric target rows.')

    training_profile = build_training_profile(len(X), request.cv_folds, request.training_mode)
    training_rows_available = int(len(X))
    train_sample_limit = int(training_profile['train_sample_limit'])
    if train_sample_limit > 0 and len(X) > train_sample_limit:
        X, y = sample_training_rows(X, y, max_rows=train_sample_limit, random_state=request.random_state, stratify=stratify)
        if request.problem_type == 'classification':
            stratify = y if y.value_counts().min() >= 2 else None
    training_rows_used = int(len(X))

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=request.random_state, stratify=stratify,
        )
    except ValueError as error:
        if request.problem_type == 'classification' and stratify is not None:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=request.test_size, random_state=request.random_state, stratify=None,
                )
            except ValueError as fallback_error:
                raise HTTPException(status_code=400, detail=f'Unable to split the dataset for training: {fallback_error}') from fallback_error
        else:
            raise HTTPException(status_code=400, detail=f'Unable to split the dataset for training: {error}') from error

    model_name, estimator = build_estimator(request.problem_type, request.model_type, request.random_state)
    pipeline = Pipeline([
        ('preprocessor', build_preprocessor(X_train)),
        ('model', estimator),
    ])

    start_time = time.perf_counter()
    try:
        pipeline.fit(X_train, y_train)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Training failed for {model_name}: {error}") from error
    training_time = round(time.perf_counter() - start_time, 4)

    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)
    metrics: dict[str, Any] = {}
    scoring = 'r2'
    if request.problem_type == 'regression':
        rmse = float(np.sqrt(mean_squared_error(y_test, test_predictions)))
        metrics.update({
            'train_r2': round(float(r2_score(y_train, train_predictions)), 6),
            'test_r2': round(float(r2_score(y_test, test_predictions)), 6),
            'test_rmse': round(rmse, 6),
            'test_mae': round(float(mean_absolute_error(y_test, test_predictions)), 6),
        })
        metrics['primary'] = {'R2': metrics['test_r2'], 'RMSE': metrics['test_rmse'], 'MAE': metrics['test_mae']}
    else:
        scoring = 'accuracy'
        metrics.update({
            'train_accuracy': round(float(accuracy_score(y_train, train_predictions)), 6),
            'test_accuracy': round(float(accuracy_score(y_test, test_predictions)), 6),
            'test_precision': round(float(precision_score(y_test, test_predictions, average='weighted', zero_division=0)), 6),
            'test_recall': round(float(recall_score(y_test, test_predictions, average='weighted', zero_division=0)), 6),
            'test_f1': round(float(f1_score(y_test, test_predictions, average='weighted', zero_division=0)), 6),
        })
        metrics['primary'] = {
            'Accuracy': metrics['test_accuracy'],
            'Precision': metrics['test_precision'],
            'Recall': metrics['test_recall'],
            'F1 Score': metrics['test_f1'],
        }

    folds = int(training_profile['cv_folds'])
    try:
        cv_X = X
        cv_y = y
        if bool(training_profile['skip_cv_for_large_dataset']) and len(X) >= LARGE_DATASET_ROW_THRESHOLD:
            raise RuntimeError('Cross-validation skipped for large dataset to keep training responsive.')
        cv_stratify = y if request.problem_type == 'classification' and y.value_counts().min() >= 2 else None
        cv_sample_limit = int(training_profile['cv_sample_limit'])
        if cv_sample_limit > 0:
            cv_X, cv_y = sample_training_rows(X, y, max_rows=cv_sample_limit, random_state=request.random_state, stratify=cv_stratify)
            if request.problem_type == 'classification':
                cv_stratify = cv_y if cv_y.value_counts().min() >= 2 else None
        if request.problem_type == 'classification':
            min_class_size = int(cv_y.value_counts().min())
            folds = max(2, min(folds, min_class_size))
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=request.random_state)
        else:
            cv = KFold(n_splits=folds, shuffle=True, random_state=request.random_state)
        cv_scores = cross_val_score(clone(pipeline), cv_X, cv_y, cv=cv, scoring=scoring, n_jobs=TRAINING_N_JOBS)
        metrics['cv_scores'] = [round(float(score), 6) for score in cv_scores]
        metrics['cv_mean'] = round(float(np.mean(cv_scores)), 6)
        metrics['cv_std'] = round(float(np.std(cv_scores)), 6)
        metrics['cv_rows_evaluated'] = int(len(cv_X))
        metrics['cv_folds_used'] = int(folds)
    except Exception:
        metrics['cv_scores'] = []
        metrics['cv_mean'] = 0.0
        metrics['cv_std'] = 0.0
        metrics['cv_rows_evaluated'] = 0
        metrics['cv_folds_used'] = int(folds)

    try:
        importance_X = X_test
        importance_y = y_test
        if len(X) > 10000:
            raise RuntimeError('Permutation importance skipped for large dataset to keep training responsive.')
        importance_sample_limit = int(training_profile['importance_sample_limit'])
        if importance_sample_limit > 0 and len(X_test) > importance_sample_limit:
            importance_X, importance_y = sample_training_rows(
                X_test, y_test, max_rows=importance_sample_limit, random_state=request.random_state,
                stratify=y_test if request.problem_type == 'classification' and y_test.value_counts().min() >= 2 else None,
            )
        permutation = permutation_importance(
            pipeline, importance_X, importance_y, n_repeats=int(training_profile['importance_repeats']),
            random_state=request.random_state, scoring=scoring, n_jobs=TRAINING_N_JOBS,
        )
        feature_importance = [
            {'name': feature_name, 'importance': round(float(permutation.importances_mean[index]), 6)}
            for index, feature_name in enumerate(request.feature_columns)
        ]
    except Exception:
        feature_importance = [{'name': feature_name, 'importance': 0.0} for feature_name in request.feature_columns]
    feature_importance.sort(key=lambda item: item['importance'], reverse=True)

    if request.problem_type == 'classification' and label_encoder is not None:
        actual_values = label_encoder.inverse_transform(y_test.astype(int))
        predicted_values = label_encoder.inverse_transform(pd.Series(test_predictions).astype(int))
    else:
        actual_values = y_test.tolist()
        predicted_values = pd.Series(test_predictions).tolist()

    sample_predictions = [
        {'actual': safe_serialize(actual_values[index]), 'predicted': safe_serialize(predicted_values[index])}
        for index in range(min(10, len(predicted_values)))
    ]

    overfitting_assessment = assess_overfitting(request.problem_type, metrics)

    model_id = str(uuid.uuid4())[:8]
    model_bundle = {
        'pipeline': pipeline,
        'feature_columns': request.feature_columns,
        'target_column': request.target_column,
        'problem_type': request.problem_type,
        'model_type': request.model_type,
        'model_name': model_name,
        'label_encoder': label_encoder,
        'trained_at': datetime.utcnow().isoformat(),
    }
    save_model_bundle(model_id, model_bundle)
    logger.info('Train request completed successfully model_id=%s model_name=%s', model_id, model_bundle['model_name'])

    response = {
        'model_id': model_id,
        'model_name': model_name,
        'problem_type': request.problem_type,
        'metrics': metrics['primary'],
        'full_metrics': metrics,
        'feature_importance': feature_importance,
        'sample_predictions': sample_predictions,
        'analysis': make_analysis(request.problem_type, model_name, metrics, feature_importance),
        'training_time': training_time,
        'trained_at': model_bundle['trained_at'],
        'cv_scores': metrics.get('cv_scores', []),
        'overfitting_detected': overfitting_assessment['detected'],
        'overfitting_status': overfitting_assessment['status'],
        'overfitting_explanation': overfitting_assessment['explanation'],
        'generalization_gap': overfitting_assessment['generalization_gap'],
        'cv_gap': overfitting_assessment['cv_gap'],
        'optimization': {
            'training_rows_available': training_rows_available,
            'training_rows_used': training_rows_used,
            'training_sampled': training_rows_used < training_rows_available,
            'cv_rows_evaluated': metrics.get('cv_rows_evaluated', len(X)),
            'cv_folds_used': metrics.get('cv_folds_used', folds),
            'cv_sampled': bool(training_profile['cv_sample_limit']) and metrics.get('cv_rows_evaluated', 0) > 0 and metrics.get('cv_rows_evaluated', 0) < training_rows_available,
            'training_mode': str(training_profile['training_mode']),
            'importance_rows_evaluated': int(len(importance_X)) if 'importance_X' in locals() else int(len(X_test)),
            'importance_repeats': int(training_profile['importance_repeats']),
        },
    }
    return JSONResponse(content=safe_serialize(response))


@router.post('/sales-forecast')
def sales_forecast(request: SalesForecastRequest) -> JSONResponse:
    required_columns = [request.date_column, request.target_column]
    if request.dataset_id:
        dataset_entry = DATASET_CACHE.get(request.dataset_id)
        if dataset_entry is None:
            raise HTTPException(status_code=400, detail='Cached dataset not found. Please upload the file again.')
        available_columns = list(dataset_entry['columns'])
        resolved_columns = resolve_requested_columns(required_columns, available_columns)
        resolved_date_column = resolved_columns[request.date_column]
        resolved_target_column = resolved_columns[request.target_column]
        if dataset_entry.get('parquet_path'):
            series_frame, freq, period_label = prepare_sales_series_from_parquet(dataset_entry, resolved_date_column, resolved_target_column)
        else:
            frame = load_dataset_frame(request.dataset_id, request.data, required_columns)
            series_frame, freq, period_label = prepare_sales_series(frame, request.date_column, request.target_column)
    else:
        frame = load_dataset_frame(request.dataset_id, request.data, required_columns)
        series_frame, freq, period_label = prepare_sales_series(frame, request.date_column, request.target_column)

    total_periods = len(series_frame)
    derived_test_periods = int(round(total_periods * (request.test_percentage / 100)))
    requested_test_periods = request.test_periods if request.test_periods is not None else derived_test_periods
    effective_test_periods = min(max(1, requested_test_periods), max(1, total_periods - 4))
    train_periods = total_periods - effective_test_periods
    effective_train_percentage = round((train_periods / total_periods) * 100, 1)
    effective_test_percentage = round((effective_test_periods / total_periods) * 100, 1)
    effective_lag_periods = min(request.lag_periods, max(1, train_periods - 1), max(1, total_periods - 2))
    train_series = series_frame.iloc[:train_periods].copy()
    test_series = series_frame.iloc[train_periods:].copy()

    train_X, train_y = build_forecast_training_frame(train_series, effective_lag_periods)
    model = LinearRegression()
    model.fit(train_X, train_y)

    test_start_period = pd.Timestamp(test_series.iloc[0]['period'])
    historical_train_values = train_series['sales'].astype(float).tolist()
    test_predictions = recursive_forecast(
        model,
        historical_train_values,
        test_start_period,
        effective_test_periods,
        effective_lag_periods,
        freq,
        period_label,
    )

    actual_test_values = test_series['sales'].astype(float).tolist()
    predicted_test_values = [float(item['predicted']) for item in test_predictions]
    metrics = calculate_forecast_metrics(actual_test_values, predicted_test_values)

    full_X, full_y = build_forecast_training_frame(series_frame, effective_lag_periods)
    full_model = LinearRegression()
    full_model.fit(full_X, full_y)
    future_start_period = pd.Timestamp(series_frame.iloc[-1]['period']) + pd.tseries.frequencies.to_offset(freq)
    future_predictions = recursive_forecast(
        full_model,
        series_frame['sales'].astype(float).tolist(),
        future_start_period,
        request.forecast_periods,
        effective_lag_periods,
        freq,
        period_label,
    )

    history = [
        {
            'period': format_forecast_period(pd.Timestamp(row['period']), period_label),
            'actual': round(float(row['sales']), 2),
        }
        for _, row in series_frame.iterrows()
    ]

    test_results = [
        {
            'period': format_forecast_period(pd.Timestamp(test_series.iloc[index]['period']), period_label),
            'actual': round(float(actual_test_values[index]), 2),
            'predicted': round(float(predicted_test_values[index]), 2),
        }
        for index in range(len(test_predictions))
    ]

    plural_label = period_label if request.forecast_periods == 1 else f'{period_label}s'
    analysis = (
        f"Sales forecasting used a time-series regression model trained on {train_periods} historical {plural_label if train_periods != 1 else period_label} "
        f"({effective_train_percentage}% of the dataset) and backtested on {effective_test_periods} {plural_label if effective_test_periods != 1 else period_label} "
        f"({effective_test_percentage}%). The system detected a {period_label}-level pattern from your dataset and projected the next "
        f"{request.forecast_periods} {plural_label}. Backtest MAE is {metrics['mae']}, RMSE is {metrics['rmse']}, and MAPE is {metrics['mape']}%."
    )

    response = {
        'date_column': request.date_column,
        'target_column': request.target_column,
        'frequency': freq,
        'period_label': period_label,
        'history': history,
        'test_forecast': test_results,
        'future_forecast': future_predictions,
        'metrics': metrics,
        'training_summary': {
            'model_name': 'Time-series regression',
            'total_periods': total_periods,
            'train_periods': train_periods,
            'test_periods': effective_test_periods,
            'train_percentage': effective_train_percentage,
            'test_percentage': effective_test_percentage,
            'forecast_periods': request.forecast_periods,
            'lag_periods': effective_lag_periods,
            'train_start': history[0]['period'],
            'train_end': history[train_periods - 1]['period'],
            'test_start': history[train_periods]['period'],
            'test_end': history[-1]['period'],
            'last_observed_period': history[-1]['period'],
        },
        'analysis': analysis,
    }
    return JSONResponse(content=safe_serialize(response))


@router.post('/predict')
def predict(request: PredictRequest) -> JSONResponse:
    bundle = load_model_bundle(request.model_id)
    missing = [feature for feature in bundle['feature_columns'] if request.features.get(feature) in [None, '']]
    if missing:
        raise HTTPException(status_code=400, detail=f'Missing features: {missing}')

    frame = normalize_feature_frame(pd.DataFrame([{feature: request.features.get(feature) for feature in bundle['feature_columns']}]))
    try:
        raw_prediction = bundle['pipeline'].predict(frame)[0]
        prediction: Any = raw_prediction
        if bundle['problem_type'] == 'regression':
            prediction = int(round(float(raw_prediction)))

        payload: dict[str, Any] = {'prediction': safe_serialize(prediction), 'prediction_label': safe_serialize(prediction)}

        label_encoder: LabelEncoder | None = bundle.get('label_encoder')
        if bundle['problem_type'] == 'classification' and label_encoder is not None:
            label = label_encoder.inverse_transform([int(prediction)])[0]
            payload['prediction_label'] = str(label)

            model = bundle['pipeline'].named_steps['model']
            if hasattr(model, 'predict_proba'):
                probabilities = bundle['pipeline'].predict_proba(frame)[0]
                probability_map: dict[str, float] = {}
                for encoded_class, probability in enumerate(probabilities):
                    label_name = label_encoder.inverse_transform([encoded_class])[0]
                    probability_map[str(label_name)] = round(float(probability), 6)
                payload['probabilities'] = probability_map
                payload['confidence'] = round(float(np.max(probabilities)), 6)
                payload['top_class'] = max(probability_map, key=probability_map.get)

        return JSONResponse(content=safe_serialize(payload))
    except HTTPException:
        raise
    except Exception as error:
        logger.exception('Prediction failed model_id=%s', request.model_id)
        raise HTTPException(status_code=400, detail=f'Prediction failed: {error}') from error


@router.post('/upload-model')
async def upload_model(file: UploadFile = File(...)) -> JSONResponse:
    filename = file.filename or 'uploaded_model'
    if not filename.lower().endswith(('.joblib', '.pkl', '.pickle')):
        raise HTTPException(status_code=400, detail='Only .joblib, .pkl, and .pickle model files are supported.')

    try:
        content = await file.read()
        raw_bundle = joblib.load(io.BytesIO(content))
        model_bundle = normalize_uploaded_bundle(raw_bundle, filename)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=f'Failed to load model file: {error}') from error

    model_id = str(uuid.uuid4())[:8]
    save_model_bundle(model_id, model_bundle)
    logger.info('Uploaded model loaded successfully model_id=%s model_name=%s', model_id, model_bundle['model_name'])

    response = {
        'model_id': model_id,
        'model_name': model_bundle['model_name'],
        'model_type': model_bundle['model_type'],
        'problem_type': model_bundle['problem_type'],
        'target_column': model_bundle['target_column'],
        'feature_columns': model_bundle['feature_columns'],
        'trained_at': model_bundle['trained_at'],
        'source_filename': filename,
    }
    return JSONResponse(content=safe_serialize(response))


def build_column_info_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    info: list[dict[str, Any]] = []
    total_rows = len(frame)
    for column in frame.columns:
        series = frame[column]
        non_null = int(series.notna().sum())
        null_count = int(total_rows - non_null)
        unique_count = int(series.nunique(dropna=True))
        role = 'categorical'
        if pd.api.types.is_bool_dtype(series):
            role = 'boolean'
        elif pd.api.types.is_numeric_dtype(series):
            role = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(series):
            role = 'datetime'
        elif unique_count == total_rows and total_rows > 0:
            role = 'identifier'
        info.append({
            'name': str(column),
            'dtype': str(series.dtype),
            'nonNull': non_null,
            'nullCount': null_count,
            'uniqueCount': unique_count,
            'role': role,
        })
    return info


def build_column_info_from_polars_frame(frame: pl.DataFrame) -> list[dict[str, Any]]:
    info: list[dict[str, Any]] = []
    total_rows = frame.height
    for column in frame.columns:
        series = frame.get_column(column)
        non_null = int(series.len() - series.null_count())
        null_count = int(series.null_count())
        unique_count = int(series.n_unique()) if total_rows > 0 else 0
        dtype = series.dtype
        role = 'categorical'
        if dtype == pl.Boolean:
            role = 'boolean'
        elif dtype.is_numeric():
            role = 'numeric'
        elif dtype in pl.TEMPORAL_DTYPES:
            role = 'datetime'
        elif unique_count == total_rows and total_rows > 0:
            role = 'identifier'
        info.append({
            'name': str(column),
            'dtype': str(dtype),
            'nonNull': non_null,
            'nullCount': null_count,
            'uniqueCount': unique_count,
            'role': role,
        })
    return info



def try_parse_datetime_series(series: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors='coerce')
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return None

    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return None

    parsed = pd.to_datetime(sample, errors='coerce')
    success_ratio = float(parsed.notna().mean()) if len(sample) else 0.0
    if success_ratio < 0.6:
        return None
    return pd.to_datetime(series, errors='coerce')


def clean_cached_parquet_dataset(request: ParquetCleaningRequest) -> dict[str, Any]:
    dataset_entry = DATASET_CACHE.get(request.dataset_id)
    if dataset_entry is None:
        raise HTTPException(status_code=400, detail='Cached dataset not found. Please upload the file again.')

    if dataset_entry.get('frame_path'):
        frame = read_cached_frame(dataset_entry)
        frame = normalize_dataframe(pd.DataFrame(frame))
        original_row_count = int(len(frame))
        logs: list[dict[str, Any]] = []

        if request.standardize_names:
            renamed_columns = [normalize_column_name(str(column)) or f'column_{index + 1}' for index, column in enumerate(frame.columns)]
            if renamed_columns != list(frame.columns):
                frame.columns = renamed_columns
                logs.append({
                    'action': 'Standardized Column Names',
                    'detail': 'Normalized column names for easier analysis and modeling.',
                    'timestamp': datetime.utcnow().isoformat(),
                })

        if request.remove_duplicates:
            before = len(frame)
            frame = frame.drop_duplicates().reset_index(drop=True)
            removed = before - len(frame)
            if removed > 0:
                logs.append({
                    'action': 'Removed Duplicates',
                    'detail': f'Removed {removed} duplicate rows.',
                    'timestamp': datetime.utcnow().isoformat(),
                })

        if request.handle_missing:
            filled_columns: list[str] = []
            for column in frame.columns:
                series = frame[column]
                if not series.isna().any():
                    continue
                if pd.api.types.is_numeric_dtype(series):
                    median = series.median()
                    fill_value = 0 if pd.isna(median) else median
                    frame[column] = series.fillna(fill_value)
                else:
                    mode = series.mode(dropna=True)
                    fill_value = mode.iloc[0] if not mode.empty else 'Unknown'
                    frame[column] = series.fillna(fill_value)
                filled_columns.append(str(column))
            if filled_columns:
                logs.append({
                    'action': 'Handled Missing Values',
                    'detail': f'Filled missing values in {len(filled_columns)} column(s).',
                    'timestamp': datetime.utcnow().isoformat(),
                })

        if request.convert_dates:
            converted_columns: list[str] = []
            for column in frame.columns:
                parsed_series = try_parse_datetime_series(frame[column])
                if parsed_series is None or parsed_series.notna().sum() == 0:
                    continue
                frame[column] = parsed_series.dt.strftime('%Y-%m-%d').where(parsed_series.notna(), None)
                converted_columns.append(str(column))
            if converted_columns:
                logs.append({
                    'action': 'Converted Date Columns',
                    'detail': f'Converted {len(converted_columns)} date-like column(s).',
                    'timestamp': datetime.utcnow().isoformat(),
                })

        updated_dataset_path = write_cached_frame(request.dataset_id, frame)
        DATASET_CACHE[request.dataset_id] = {
            'frame_path': str(updated_dataset_path),
            'filename': dataset_entry['filename'],
            'row_count': int(len(frame)),
            'column_count': int(len(frame.columns)),
            'columns': list(frame.columns),
        }
        memory_size = updated_dataset_path.stat().st_size
        preview_frame = frame.head(PARQUET_PREVIEW_ROW_LIMIT)
        duplicate_rows = int(max(0, len(frame) - len(frame.drop_duplicates())))
        return {
            'datasetId': request.dataset_id,
            'data': safe_serialize(preview_frame.to_dict(orient='records')),
            'columns': build_column_info_from_frame(frame),
            'rowCount': int(len(frame)),
            'originalRowCount': original_row_count,
            'loadedRowCount': int(len(preview_frame)),
            'previewLoaded': len(frame) > len(preview_frame),
            'duplicates': duplicate_rows,
            'memoryUsage': f'{memory_size / (1024 * 1024):.2f} MB',
            'logs': logs,
        }

    if not dataset_entry.get('parquet_path'):
        raise HTTPException(status_code=400, detail='Cached dataset storage is missing. Please upload the file again.')

    frame = read_cached_parquet(dataset_entry, low_memory=True)
    original_row_count = int(frame.height)
    logs: list[dict[str, Any]] = []

    if request.standardize_names:
        renamed_columns = [normalize_column_name(str(column)) or f'column_{index + 1}' for index, column in enumerate(frame.columns)]
        if renamed_columns != list(frame.columns):
            frame.columns = renamed_columns
            logs.append({
                'action': 'Standardized Column Names',
                'detail': 'Normalized column names for easier analysis and modeling.',
                'timestamp': datetime.utcnow().isoformat(),
            })

    if request.remove_duplicates:
        before = frame.height
        frame = frame.unique(maintain_order=True)
        removed = before - frame.height
        if removed > 0:
            logs.append({
                'action': 'Removed Duplicates',
                'detail': f'Removed {removed} duplicate rows.',
                'timestamp': datetime.utcnow().isoformat(),
            })

    if request.handle_missing:
        fill_expressions: list[pl.Expr] = []
        filled_columns: list[str] = []
        schema = frame.schema
        for column, dtype in schema.items():
            series = frame.get_column(column)
            if series.null_count() == 0:
                continue
            fill_value: Any | None
            if dtype.is_numeric():
                median_value = series.median()
                fill_value = 0 if median_value is None else median_value
            else:
                mode_frame = frame.select(pl.col(column).drop_nulls().mode().first().alias('mode'))
                fill_value = mode_frame.item(0, 0) if mode_frame.height > 0 else None
                if fill_value is None:
                    if dtype in pl.TEMPORAL_DTYPES:
                        non_null_values = series.drop_nulls()
                        fill_value = non_null_values[0] if non_null_values.len() > 0 else None
                    elif dtype.is_(pl.String):
                        fill_value = 'Unknown'
            if fill_value is None:
                continue
            fill_expressions.append(pl.col(column).fill_null(fill_value).alias(column))
            filled_columns.append(str(column))
        if fill_expressions:
            frame = frame.with_columns(fill_expressions)
            logs.append({
                'action': 'Handled Missing Values',
                'detail': f'Filled missing values in {len(filled_columns)} column(s).',
                'timestamp': datetime.utcnow().isoformat(),
            })

    if request.convert_dates:
        converted_columns: list[str] = []
        date_expressions: list[pl.Expr] = []
        for column, dtype in frame.schema.items():
            parsed_expr = build_polars_datetime_expr(column, dtype)
            sample = frame.select(parsed_expr.alias('__parsed_date')).drop_nulls().head(50).to_series()
            if sample.len() == 0:
                continue
            success_ratio = float(sample.len() / min(50, max(1, frame.select(pl.col(column).drop_nulls().len()).item()))) if frame.height > 0 else 0.0
            if dtype not in pl.TEMPORAL_DTYPES and success_ratio < 0.6:
                continue
            date_expressions.append(
                pl.when(pl.col(column).is_null())
                .then(None)
                .otherwise(parsed_expr.dt.strftime('%Y-%m-%d'))
                .alias(column)
            )
            converted_columns.append(str(column))
        if date_expressions:
            frame = frame.with_columns(date_expressions)
            logs.append({
                'action': 'Converted Date Columns',
                'detail': f'Converted {len(converted_columns)} date-like column(s).',
                'timestamp': datetime.utcnow().isoformat(),
            })

    parquet_buffer = io.BytesIO()
    frame.write_parquet(parquet_buffer)
    updated_dataset_path = write_dataset_file(request.dataset_id, parquet_buffer.getvalue())
    DATASET_CACHE[request.dataset_id] = {
        'parquet_path': str(updated_dataset_path),
        'filename': dataset_entry['filename'],
        'row_count': int(frame.height),
        'column_count': int(len(frame.columns)),
        'columns': list(frame.columns),
    }
    memory_size = updated_dataset_path.stat().st_size

    preview_frame = frame.head(PARQUET_PREVIEW_ROW_LIMIT)
    duplicate_rows = int(max(0, frame.height - frame.unique().height))
    return {
        'datasetId': request.dataset_id,
        'data': safe_serialize(preview_frame.to_dicts()),
        'columns': build_column_info_from_polars_frame(frame),
        'rowCount': int(frame.height),
        'originalRowCount': original_row_count,
        'loadedRowCount': int(preview_frame.height),
        'previewLoaded': frame.height > preview_frame.height,
        'duplicates': duplicate_rows,
        'memoryUsage': f'{memory_size / (1024 * 1024):.2f} MB',
        'logs': logs,
    }


def generate_cleaning_justification(request: CleaningJustificationRequest) -> str:
    summary_lines = [
        f"The dataset has {request.totalRows} rows and {request.totalColumns} columns.",
        'The following cleaning steps were applied to improve data quality:',
    ]
    for log in request.logs:
        summary_lines.append(f"- {log.action}: {log.detail}")
    summary_lines.append('These changes help the downstream analysis and model training use more consistent data.')
    return "\n".join(summary_lines)


def build_report_pdf(payload: ReportPayload) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=34, rightMargin=34, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.white,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#d1fae5'),
    )
    cover_tag_style = ParagraphStyle(
        'CoverTag',
        parent=styles['BodyText'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#ccfbf1'),
        spaceAfter=4,
    )
    cover_meta_label_style = ParagraphStyle(
        'CoverMetaLabel',
        parent=styles['BodyText'],
        fontName='Helvetica-Bold',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#99f6e4'),
    )
    cover_meta_value_style = ParagraphStyle(
        'CoverMetaValue',
        parent=styles['BodyText'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=13,
        textColor=colors.white,
    )
    section_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.white,
        spaceAfter=0,
    )
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=4,
    )
    small_style = ParagraphStyle(
        'ReportSmall',
        parent=body_style,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#475569'),
    )
    card_label_style = ParagraphStyle(
        'CardLabel',
        parent=styles['BodyText'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#0f766e'),
    )
    card_value_style = ParagraphStyle(
        'CardValue',
        parent=styles['BodyText'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=17,
        textColor=colors.HexColor('#111827'),
    )

    elements: list[Any] = []

    def as_paragraph(text: Any, style: ParagraphStyle = body_style) -> Paragraph:
        return Paragraph(str(text).replace('\n', '<br/>'), style)

    def add_paragraph(text: Any, style: ParagraphStyle = body_style) -> None:
        elements.append(as_paragraph(text, style))

    def add_section(title: str, blurb: str | None = None) -> None:
        header = Table([[as_paragraph(title, section_style)]], colWidths=[540])
        header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0f766e')),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ]))
        elements.append(header)
        if blurb:
            info = Table([[as_paragraph(blurb, small_style)]], colWidths=[540])
            info.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0fdfa')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#99f6e4')),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(info)
        elements.append(Spacer(1, 8))

    def add_table(rows: list[list[Any]], widths: list[int] | None = None, header_bg: str = '#0f766e') -> None:
        if not rows:
            return
        normalized: list[list[Any]] = []
        for row_index, row in enumerate(rows):
            style = card_label_style if row_index == 0 else body_style
            normalized.append([as_paragraph(cell, style) for cell in row])
        table = Table(normalized, colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cbd5e1')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(table)

    def add_stat_cards(cards: list[tuple[str, Any]]) -> None:
        row = []
        widths = []
        for label, value in cards:
            card = Table([[as_paragraph(label, card_label_style)], [as_paragraph(value, card_value_style)]], colWidths=[124])
            card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
                ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#cbd5e1')),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            row.append(card)
            widths.append(128)
        wrapper = Table([row], colWidths=widths)
        wrapper.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
        elements.append(wrapper)

    def decorate_page(canvas: Any, doc_obj: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#cbd5e1'))
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, 20, letter[0] - doc.rightMargin, 20)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(doc.leftMargin, 8, f'AI-EDA & ML Workflow Report | {payload.fileName}')
        canvas.drawRightString(letter[0] - doc.rightMargin, 8, f'Page {canvas.getPageNumber()}')
        canvas.restoreState()

    workflow_rows = [
        ['Step', 'Tab', 'Included Details'],
        ['1', 'Upload', f'File {payload.fileName}, {payload.totalRows} rows, {len(payload.columns)} columns, memory {payload.memoryUsage}'],
        ['2', 'Understanding', 'Dataset quality, preview context, and upload profiling details'],
        ['3', 'Cleaning', f'{len(payload.cleaningLogs)} logged operations and cleaned row count {payload.cleanedRowCount}'],
        ['4', 'EDA', f'{len(payload.edaStats.numericColumns)} numeric columns, {len(payload.edaStats.categoricalColumns)} categorical columns, schema, statistics, and correlations'],
        ['5', 'Sales Forecast', 'Time-series training split, backtest metrics, backtest samples, and future forecast' if payload.salesForecastResult else 'No sales forecast run captured'],
        ['6', 'ML', f"Model {payload.selectedModel or 'Not trained'}, problem type {payload.problemType}, metrics and feature importance"],
        ['7', 'Prediction', 'Latest prediction, model context, probabilities, and recent prediction history' if payload.predictionResult is not None else 'No prediction captured'],
    ]

    generated_on = datetime.now().strftime('%d %b %Y, %I:%M %p')
    workflow_coverage = f"{7 if payload.salesForecastResult is not None else 6}/7 sections"
    report_status = 'Complete workflow captured' if payload.predictionResult is not None else 'Workflow summary generated'

    cover_meta = Table([
        [
            Table([
                [as_paragraph('DATASET', cover_meta_label_style), as_paragraph('GENERATED ON', cover_meta_label_style), as_paragraph('WORKFLOW COVERAGE', cover_meta_label_style)],
                [as_paragraph(payload.fileName, cover_meta_value_style), as_paragraph(generated_on, cover_meta_value_style), as_paragraph(workflow_coverage, cover_meta_value_style)],
            ], colWidths=[170, 150, 140]),
        ],
        [
            Table([
                [as_paragraph('REPORT STATUS', cover_meta_label_style), as_paragraph('CLEANED ROWS', cover_meta_label_style), as_paragraph('COLUMNS PROFILED', cover_meta_label_style)],
                [as_paragraph(report_status, cover_meta_value_style), as_paragraph(f'{payload.cleanedRowCount:,}', cover_meta_value_style), as_paragraph(str(len(payload.columns)), cover_meta_value_style)],
            ], colWidths=[170, 150, 140]),
        ],
    ], colWidths=[500])
    cover_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#134e4a')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#5eead4')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#115e59')),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))

    cover = Table([[
        as_paragraph('AI-Assisted EDA & ML Platform', cover_tag_style),
        as_paragraph('Workflow Report', title_style),
        as_paragraph(
            'A complete view of the dataset journey from upload and cleaning to EDA, forecasting, machine learning, and final prediction outputs.',
            subtitle_style,
        ),
        cover_meta,
    ]], colWidths=[540])
    cover.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0f766e')),
        ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor('#14b8a6')),
        ('LEFTPADDING', (0, 0), (-1, -1), 18),
        ('RIGHTPADDING', (0, 0), (-1, -1), 18),
        ('TOPPADDING', (0, 0), (-1, -1), 18),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 18),
    ]))
    elements.append(cover)
    elements.append(Spacer(1, 12))

    add_section('Workflow Coverage', 'This overview mirrors the product tabs so the report reads like the same journey your team followed in the app.')
    add_table(workflow_rows, [35, 90, 415])
    elements.append(Spacer(1, 10))

    add_section('Tab 1: Data Upload', 'Initial dataset intake, scale, and storage footprint at the moment the workflow began.')
    add_stat_cards([
        ('Rows', f'{payload.totalRows:,}'),
        ('Columns', len(payload.columns)),
        ('Duplicates', payload.duplicates),
        ('Memory Usage', payload.memoryUsage),
    ])
    elements.append(Spacer(1, 8))
    add_table([
        ['File Name', 'Cleaned Rows', 'Cleaning Done'],
        [payload.fileName, str(payload.cleanedRowCount), 'Yes' if payload.cleaningDone else 'No'],
    ], [240, 140, 160], header_bg='#115e59')
    elements.append(Spacer(1, 10))

    add_section('Tab 2: Data Understanding', 'This step captures dataset identity, quality checks, preview context, and the initial profiling needed before cleaning and deeper EDA.')
    column_rows = [['Column', 'Type', 'Role', 'Non-null', 'Nulls', 'Unique']]
    for column in payload.columns[:18]:
        column_rows.append([column.name, column.dtype, column.role, str(column.nonNull), str(column.nullCount), str(column.uniqueCount)])
    add_table(column_rows, [165, 70, 80, 60, 50, 50], header_bg='#134e4a')
    if len(payload.columns) > 18:
        add_paragraph(f'Showing the first 18 columns out of {len(payload.columns)} total columns.', small_style)
    elements.append(Spacer(1, 10))

    add_section('Tab 3: Data Cleaning', 'This section records the applied transformations so the report preserves not just the outcome, but also the reasoning trail.')
    add_paragraph(f"Cleaning completed: {'Yes' if payload.cleaningDone else 'No'}. Cleaned row count: {payload.cleanedRowCount}.")
    if payload.cleaningLogs:
        cleaning_rows = [['Action', 'Detail', 'Timestamp']]
        for log in payload.cleaningLogs[:20]:
            cleaning_rows.append([log.action, log.detail, log.timestamp])
        add_table(cleaning_rows, [120, 300, 120], header_bg='#0f766e')
    else:
        add_paragraph('No cleaning steps were recorded for this run.', small_style)
    elements.append(Spacer(1, 10))

    add_section('Tab 4: Exploratory Data Analysis', 'EDA summarizes the dataset schema, descriptive statistics, and strongest numeric relationships for downstream decisions.')
    add_stat_cards([
        ('Numeric Columns', len(payload.edaStats.numericColumns)),
        ('Categorical Columns', len(payload.edaStats.categoricalColumns)),
        ('Correlations', len(payload.edaStats.correlations)),
        ('AI Insight', 'Available' if payload.aiInsights else 'Not captured'),
    ])
    elements.append(Spacer(1, 8))
    if payload.edaStats.numericColumns:
        numeric_rows = [['Numeric Column', 'Mean', 'Std', 'Min', 'Max']]
        for column_name in payload.edaStats.numericColumns[:10]:
            stats = payload.edaStats.stats.get(column_name, {})
            numeric_rows.append([column_name, stats.get('mean', 'N/A'), stats.get('std', 'N/A'), stats.get('min', 'N/A'), stats.get('max', 'N/A')])
        add_table(numeric_rows, [180, 85, 85, 85, 85], header_bg='#115e59')
    if payload.edaStats.correlations:
        elements.append(Spacer(1, 6))
        corr_rows = [['Pair', 'Correlation']]
        for item in payload.edaStats.correlations[:8]:
            corr_rows.append([str(item.get('pair', 'N/A')), str(item.get('correlation', 'N/A'))])
        add_table(corr_rows, [430, 90], header_bg='#115e59')
    if payload.aiInsights:
        elements.append(Spacer(1, 6))
        insight_box = Table([[as_paragraph(payload.aiInsights, body_style)]], colWidths=[540])
        insight_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ecfeff')),
            ('BOX', (0, 0), (-1, -1), 0.6, colors.HexColor('#67e8f9')),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(insight_box)
    elements.append(Spacer(1, 10))

    add_section('Tab 5: Sales Forecast', 'Sales forecasting fits best after EDA because it depends on cleaned, time-aware historical patterns rather than the generic supervised ML pipeline.')
    if payload.salesForecastResult is not None:
        forecast = payload.salesForecastResult
        add_stat_cards([
            ('Forecast Model', forecast.training_summary.model_name),
            ('Split', f"{forecast.training_summary.train_percentage}% / {forecast.training_summary.test_percentage}%"),
            ('Forecast Horizon', forecast.training_summary.forecast_periods),
            ('Period Type', forecast.period_label or forecast.frequency or 'Period'),
        ])
        elements.append(Spacer(1, 8))
        add_table([
            ['Date Column', 'Sales Column', 'Train Window', 'Backtest Window'],
            [forecast.date_column, forecast.target_column, f"{forecast.training_summary.train_start} to {forecast.training_summary.train_end}", f"{forecast.training_summary.test_start} to {forecast.training_summary.test_end}"],
        ], [110, 110, 160, 160], header_bg='#115e59')
        elements.append(Spacer(1, 6))
        add_stat_cards([
            ('MAE', forecast.metrics.mae),
            ('RMSE', forecast.metrics.rmse),
            ('MAPE', f"{forecast.metrics.mape}%"),
            ('Observed Periods', forecast.training_summary.total_periods),
        ])
        elements.append(Spacer(1, 8))
        add_paragraph(forecast.analysis)
        if forecast.test_forecast:
            elements.append(Spacer(1, 6))
            backtest_rows = [['Backtest Period', 'Actual', 'Predicted']]
            for item in forecast.test_forecast[:8]:
                backtest_rows.append([item.period, item.actual if item.actual is not None else 'N/A', item.predicted if item.predicted is not None else 'N/A'])
            add_table(backtest_rows, [190, 160, 160], header_bg='#134e4a')
        if forecast.future_forecast:
            elements.append(Spacer(1, 6))
            future_rows = [['Future Period', 'Forecasted Sales']]
            for item in forecast.future_forecast[:8]:
                future_rows.append([item.period, item.predicted if item.predicted is not None else 'N/A'])
            add_table(future_rows, [270, 270], header_bg='#134e4a')
    else:
        add_paragraph('No sales forecasting run was available for this report.', small_style)
    elements.append(Spacer(1, 10))

    add_section('Tab 6: Machine Learning', 'General machine learning follows forecasting in the workflow because it is a broader predictive branch for supervised models and downstream prediction serving.')
    add_stat_cards([
        ('Target', payload.targetColumn or 'Not selected'),
        ('Problem Type', payload.problemType.title()),
        ('Selected Model', payload.selectedModel or 'Not trained'),
        ('Features', len(payload.selectedFeatures)),
    ])
    elements.append(Spacer(1, 8))
    if payload.selectedFeatures:
        add_paragraph('Selected features: ' + ', '.join(payload.selectedFeatures[:20]))
    if payload.modelMetrics:
        metric_rows = [['Metric', 'Value']]
        for key, value in payload.modelMetrics.items():
            metric_rows.append([key, value])
        add_table(metric_rows, [270, 270], header_bg='#134e4a')
    else:
        add_paragraph('No ML metrics were available.', small_style)
    if payload.featureImportance:
        elements.append(Spacer(1, 6))
        importance_rows = [['Rank', 'Feature', 'Importance']]
        for index, item in enumerate(payload.featureImportance[:12], start=1):
            importance_rows.append([index, item.get('name', 'N/A'), item.get('importance', 'N/A')])
        add_table(importance_rows, [50, 360, 130], header_bg='#134e4a')
    elements.append(Spacer(1, 10))

    add_section('Tab 7: Prediction', 'The report closes with the latest scoring output, supporting model context, and recent prediction history when available.')
    if payload.uploadedModel is not None:
        add_table([
            ['Prediction Model', 'Type', 'Target', 'Problem', 'Trained At'],
            [payload.uploadedModel.name, payload.uploadedModel.type, payload.uploadedModel.target, payload.uploadedModel.problem, payload.uploadedModel.trainedAt],
        ], [130, 90, 120, 70, 130], header_bg='#115e59')
        if payload.uploadedModel.features:
            elements.append(Spacer(1, 6))
            add_paragraph('Prediction model features: ' + ', '.join(payload.uploadedModel.features[:20]))
    if payload.predictionResult is not None:
        elements.append(Spacer(1, 6))
        add_stat_cards([
            ('Latest Prediction', payload.predictionResult),
            ('History Entries', len(payload.predictionHistory)),
            ('Probabilities', 'Available' if payload.predictionProbabilities else 'N/A'),
            ('Prediction Analysis', 'Available' if payload.predictionAnalysis else 'N/A'),
        ])
        elements.append(Spacer(1, 8))
        if payload.predictionAnalysis:
            add_paragraph(payload.predictionAnalysis)
        if payload.predictionProbabilities:
            elements.append(Spacer(1, 6))
            prob_rows = [['Class', 'Probability']]
            for label, probability in list(payload.predictionProbabilities.items())[:10]:
                prob_rows.append([label, f'{round(probability * 100, 2)}%'])
            add_table(prob_rows, [270, 270], header_bg='#134e4a')
        if payload.predictionHistory:
            elements.append(Spacer(1, 6))
            history_rows = [['Timestamp', 'Prediction', 'Confidence']]
            for item in payload.predictionHistory[-8:]:
                confidence = 'N/A' if item.confidence is None else f'{round(item.confidence * 100, 2)}%'
                history_rows.append([item.timestamp, item.prediction, confidence])
            add_table(history_rows, [230, 170, 140], header_bg='#134e4a')
    else:
        add_paragraph('No predictions were generated for this report.', small_style)

    doc.build(elements, onFirstPage=decorate_page, onLaterPages=decorate_page)
    return buffer.getvalue()


@router.post('/cache-dataset')
def cache_dataset(request: DatasetCacheRequest) -> JSONResponse:
    if not request.data:
        raise HTTPException(status_code=400, detail='Dataset rows are required.')

    try:
        data_frame = normalize_dataframe(pd.DataFrame(request.data))
    except Exception as error:
        raise HTTPException(status_code=400, detail=f'Failed to cache dataset: {error}') from error

    if data_frame.empty or data_frame.shape[1] == 0:
        raise HTTPException(status_code=400, detail='Dataset must contain at least one row and one column.')

    dataset_id = str(uuid.uuid4())[:8]
    cached_path = write_cached_frame(dataset_id, data_frame)
    DATASET_CACHE[dataset_id] = {
        'frame_path': str(cached_path),
        'filename': request.file_name,
        'row_count': int(len(data_frame)),
        'column_count': int(len(data_frame.columns)),
        'columns': list(data_frame.columns),
    }

    return JSONResponse(content={
        'datasetId': dataset_id,
        'rowCount': int(len(data_frame)),
        'loadedRowCount': int(len(data_frame)),
        'columnCount': int(len(data_frame.columns)),
        'columns': list(data_frame.columns),
        'previewLoaded': False,
    })


@router.post('/parse-parquet')
async def parse_parquet(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename or not file.filename.lower().endswith('.parquet'):
        raise HTTPException(status_code=400, detail='Only .parquet files are supported.')

    content = await file.read()
    if len(content) > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail='File exceeds 200MB limit.')

    buffer = io.BytesIO(content)

    try:
        parquet_file = pq.ParquetFile(buffer)
        total_rows = int(parquet_file.metadata.num_rows)
        column_count = len(parquet_file.schema.names)
        buffer.seek(0)
        frame = pl.read_parquet(buffer, n_rows=PARQUET_PREVIEW_ROW_LIMIT, low_memory=True)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f'Failed to parse Parquet file: {error}') from error

    dataset_id = str(uuid.uuid4())[:8]
    cached_path = write_dataset_file(dataset_id, content)
    DATASET_CACHE[dataset_id] = {
        'parquet_path': str(cached_path),
        'filename': file.filename,
        'row_count': total_rows,
        'column_count': int(column_count),
        'columns': list(frame.columns),
    }

    rows = frame.to_dicts()
    preview_loaded = total_rows > len(rows)
    return JSONResponse(content={
        'datasetId': dataset_id,
        'data': safe_serialize(rows),
        'columns': frame.columns,
        'columnInfo': build_column_info_from_polars_frame(frame),
        'rowCount': total_rows,
        'loadedRowCount': int(len(rows)),
        'columnCount': int(column_count),
        'previewLoaded': preview_loaded,
        'previewLimit': PARQUET_PREVIEW_ROW_LIMIT,
    })




@router.post('/clean-parquet')
def clean_parquet(request: ParquetCleaningRequest) -> JSONResponse:
    try:
        return JSONResponse(content=clean_cached_parquet_dataset(request))
    except HTTPException:
        raise
    except Exception as error:
        logger.exception('Parquet cleaning failed dataset_id=%s', request.dataset_id)
        raise HTTPException(status_code=400, detail=f'Parquet cleaning failed: {error}') from error

@router.post('/cleaning-justification')
def cleaning_justification(request: CleaningJustificationRequest) -> JSONResponse:
    if not request.logs:
        raise HTTPException(status_code=400, detail='No cleaning logs provided.')
    return JSONResponse(content={'justification': generate_cleaning_justification(request)})


@router.post('/generate-report')
def generate_report(payload: ReportPayload) -> Response:
    try:
        pdf_bytes = build_report_pdf(payload)
    except Exception as error:
        logger.exception('Report generation failed file_name=%s', payload.fileName)
        raise HTTPException(status_code=400, detail=f'Failed to generate report: {error}') from error

    file_stem = payload.fileName.rsplit('.', 1)[0]
    return Response(
        content=pdf_bytes,
        media_type='application/pdf',
        headers={'Content-Disposition': f'attachment; filename="{file_stem}_analysis_report.pdf"'},
    )


@router.get('/health')
def health() -> dict[str, str]:
    return {'status': 'healthy'}


@app.get('/')
def root() -> dict[str, Any]:
    return {'service': 'AI-Assisted EDA & ML Backend', 'docs': '/docs', 'api': '/api'}


app.include_router(router)


if __name__ == '__main__':
    import uvicorn

    port = int(os.environ.get('ML_PORT', '3004'))
    uvicorn.run(app, host='0.0.0.0', port=port)




