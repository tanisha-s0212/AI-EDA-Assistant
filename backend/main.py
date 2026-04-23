from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import secrets
import time
import traceback
import uuid
import warnings
from html import escape
from math import erf, sqrt
from datetime import date, datetime, time as dt_time
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import joblib
import matplotlib
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import psycopg
from fastapi import APIRouter, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from psycopg.rows import dict_row
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)
DATASET_DIR = BASE_DIR / 'datasets'
DATASET_DIR.mkdir(exist_ok=True)
RUNTIME_TEMP_DIR = BASE_DIR / 'tmp'
RUNTIME_TEMP_DIR.mkdir(exist_ok=True)
LOG_DIR = BASE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)
os.environ.setdefault('TMP', str(RUNTIME_TEMP_DIR))
os.environ.setdefault('TEMP', str(RUNTIME_TEMP_DIR))
os.environ.setdefault('TMPDIR', str(RUNTIME_TEMP_DIR))
ACTIVITY_DATABASE_URL = os.environ.get('ACTIVITY_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql://postgres:postgres@localhost:5432/ai_eda_assistant'
ACTIVITY_DB_REQUIRED = os.environ.get('ACTIVITY_DB_REQUIRED', 'false').strip().lower() == 'true'
ACTIVITY_DB_CONNECT_TIMEOUT = int(os.environ.get('ACTIVITY_DB_CONNECT_TIMEOUT', '1'))
TRAINING_N_JOBS = 1
ACTIVITY_DB_AVAILABLE = False
EMAIL_REGEX = re.compile(r'^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$', re.IGNORECASE)
PASSWORD_HASH_ITERATIONS = 600_000
SESSION_COOKIE_NAME = 'ai_eda_session'
SESSION_DURATION_SECONDS = 60 * 60 * 24 * 7
SESSION_MAX_AGE = SESSION_DURATION_SECONDS
SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'false').strip().lower() == 'true'
SESSION_COOKIE_SAMESITE = os.environ.get('SESSION_COOKIE_SAMESITE', 'lax').strip().lower() or 'lax'
SESSION_COOKIE_DOMAIN = os.environ.get('SESSION_COOKIE_DOMAIN') or None
ENABLE_PLOTLY_STATIC_EXPORT = os.environ.get('ENABLE_PLOTLY_STATIC_EXPORT', 'false').strip().lower() == 'true'

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
SESSION_STATE: dict[str, dict[str, Any]] = {}


@app.on_event('startup')
def startup_event() -> None:
    global ACTIVITY_DB_AVAILABLE
    try:
        init_activity_db()
        ACTIVITY_DB_AVAILABLE = True
        logger.info('Activity database is available.')
    except Exception:
        ACTIVITY_DB_AVAILABLE = False
        if ACTIVITY_DB_REQUIRED:
            raise
        logger.exception(
            'Activity database is unavailable. Continuing without persisted activity logging because ACTIVITY_DB_REQUIRED is false.'
        )

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
EDA_ADVANCED_SAMPLE_LIMIT = 5_000
EDA_MAX_MISSINGNESS_COLUMNS = 30
EDA_MISSINGNESS_BUCKETS = 60
UPLOAD_READ_CHUNK_SIZE = 4 * 1024 * 1024
EDA_MAX_NUMERIC_CHARTS = 8
EDA_MAX_CATEGORICAL_CHARTS = 8
EDA_MAX_CATEGORY_BARS = 10
EDA_MAX_INTERACTION_COLUMNS = 40
EDA_MAX_INTERACTION_PAIRS = 3
MAX_UPLOAD_SIZE_BYTES = 512 * 1024 * 1024


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def get_activity_connection() -> psycopg.Connection:
    connection = psycopg.connect(
        ACTIVITY_DATABASE_URL,
        row_factory=dict_row,
        connect_timeout=ACTIVITY_DB_CONNECT_TIMEOUT,
    )
    connection.autocommit = True
    return connection


def init_activity_db() -> None:
    with get_activity_connection() as connection:
        connection.execute(
            '''
            CREATE TABLE IF NOT EXISTS app_users (
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL UNIQUE,
                username TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT NOT NULL
            )
            '''
        )
        try:
            connection.execute('ALTER TABLE app_users ADD COLUMN IF NOT EXISTS password_hash TEXT')
        except Exception:
            logger.exception('Failed to ensure password_hash column on app_users.')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_app_users_email ON app_users (email)')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_app_users_last_login_at ON app_users (last_login_at DESC)')
        connection.execute(
            '''
            CREATE TABLE IF NOT EXISTS app_user_sessions (
                id BIGSERIAL PRIMARY KEY,
                session_id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                session_token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked_at TEXT,
                client_ip TEXT,
                user_agent TEXT
            )
            '''
        )
        connection.execute('CREATE INDEX IF NOT EXISTS idx_app_user_sessions_user_id ON app_user_sessions (user_id)')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_app_user_sessions_expires_at ON app_user_sessions (expires_at DESC)')
        connection.execute(
            '''
            CREATE TABLE IF NOT EXISTS user_activities (
                id BIGSERIAL PRIMARY KEY,
                activity_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                client_session_id TEXT,
                server_session_id TEXT,
                dataset_id TEXT,
                model_id TEXT,
                activity_type TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                api_path TEXT,
                http_method TEXT,
                status_code INTEGER,
                duration_ms REAL,
                file_name TEXT,
                detail TEXT,
                metadata_json TEXT,
                client_ip TEXT,
                user_agent TEXT
            )
            '''
        )
        connection.execute('CREATE INDEX IF NOT EXISTS idx_user_activities_created_at ON user_activities (created_at DESC)')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_user_activities_client_session_id ON user_activities (client_session_id)')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_user_activities_dataset_id ON user_activities (dataset_id)')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_user_activities_server_session_id ON user_activities (server_session_id)')
        connection.execute('CREATE INDEX IF NOT EXISTS idx_user_activities_action ON user_activities (action)')


def sanitize_metadata(metadata: dict[str, Any] | None) -> str | None:
    if not metadata:
        return None
    return json.dumps(safe_serialize(metadata), default=str)


def get_client_session_id(request: Request | None) -> str | None:
    if request is None:
        return None
    return request.headers.get('x-client-session-id') or None


def record_activity(
    *,
    action: str,
    status: str,
    activity_type: str = 'workflow',
    request: Request | None = None,
    dataset_id: str | None = None,
    model_id: str | None = None,
    server_session_id: str | None = None,
    file_name: str | None = None,
    detail: str | None = None,
    metadata: dict[str, Any] | None = None,
    api_path: str | None = None,
    http_method: str | None = None,
    status_code: int | None = None,
    duration_ms: float | None = None,
) -> None:
    global ACTIVITY_DB_AVAILABLE

    if not ACTIVITY_DB_AVAILABLE:
        return

    try:
        with get_activity_connection() as connection:
            connection.execute(
                '''
                INSERT INTO user_activities (
                    activity_id,
                    created_at,
                    client_session_id,
                    server_session_id,
                    dataset_id,
                    model_id,
                    activity_type,
                    action,
                    status,
                    api_path,
                    http_method,
                    status_code,
                    duration_ms,
                    file_name,
                    detail,
                    metadata_json,
                    client_ip,
                    user_agent
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''',
                (
                    uuid.uuid4().hex,
                    utc_now_iso(),
                    get_client_session_id(request),
                    server_session_id,
                    dataset_id,
                    model_id,
                    activity_type,
                    action,
                    status,
                    api_path or (str(request.url.path) if request is not None else None),
                    http_method or (request.method if request is not None else None),
                    status_code,
                    duration_ms,
                    file_name,
                    detail,
                    sanitize_metadata(metadata),
                    request.client.host if request is not None and request.client is not None else None,
                    request.headers.get('user-agent') if request is not None else None,
                ),
            )
    except Exception:
        ACTIVITY_DB_AVAILABLE = False
        logger.exception('Failed to persist activity action=%s status=%s', action, status)


def get_session_id(dataset_id: str | None, session_id: str | None = None) -> str:
    if session_id:
        return session_id
    if dataset_id:
        return dataset_id
    return f'adhoc-{uuid.uuid4().hex[:8]}'


def normalize_email(value: str) -> str:
    return value.strip().lower()


def normalize_username(value: str) -> str:
    return ' '.join(value.strip().split())


def validate_login_payload(username: str, email: str) -> tuple[str, str]:
    normalized_username = normalize_username(username)
    normalized_email = normalize_email(email)

    if len(normalized_username) < 3:
        raise HTTPException(status_code=400, detail='Username must be at least 3 characters long.')
    if len(normalized_username) > 80:
        raise HTTPException(status_code=400, detail='Username must be 80 characters or fewer.')
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._'-]{1,79}", normalized_username):
        raise HTTPException(
            status_code=400,
            detail='Username may contain letters, numbers, spaces, periods, apostrophes, underscores, and hyphens.',
        )
    if not EMAIL_REGEX.fullmatch(normalized_email):
        raise HTTPException(status_code=400, detail='Enter a valid email address.')

    return normalized_username, normalized_email


def validate_password(password: str) -> str:
    if len(password) < 8:
        raise HTTPException(status_code=400, detail='Password must be at least 8 characters long.')
    if len(password) > 128:
        raise HTTPException(status_code=400, detail='Password must be 128 characters or fewer.')
    return password


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived_key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    return f'pbkdf2_sha256${PASSWORD_HASH_ITERATIONS}${base64.b64encode(salt).decode("ascii")}${base64.b64encode(derived_key).decode("ascii")}'


def verify_password(password: str, encoded_hash: str | None) -> bool:
    if not encoded_hash:
        return False

    try:
        algorithm, iterations_text, salt_b64, hash_b64 = encoded_hash.split('$', 3)
        if algorithm != 'pbkdf2_sha256':
            return False
        iterations = int(iterations_text)
        salt = base64.b64decode(salt_b64.encode('ascii'))
        expected_hash = base64.b64decode(hash_b64.encode('ascii'))
    except Exception:
        return False

    candidate_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        iterations,
    )
    return secrets.compare_digest(candidate_hash, expected_hash)


def build_user_payload(row: dict[str, Any]) -> dict[str, str]:
    return {
        'userId': row['user_id'],
        'username': row['username'],
        'email': row['email'],
        'createdAt': row['created_at'],
        'updatedAt': row['updated_at'],
        'lastLoginAt': row['last_login_at'],
    }


def get_user_by_email(email: str) -> dict[str, Any] | None:
    with get_activity_connection() as connection:
        return connection.execute(
            '''
            SELECT user_id, username, email, password_hash, created_at, updated_at, last_login_at
            FROM app_users
            WHERE email = %s
            ''',
            (normalize_email(email),),
        ).fetchone()


def create_authenticated_session(*, user_id: str, request: Request) -> tuple[str, str]:
    session_token = secrets.token_urlsafe(48)
    session_id = uuid.uuid4().hex
    timestamp = utc_now_iso()
    expires_at = datetime.utcfromtimestamp(time.time() + SESSION_DURATION_SECONDS).isoformat()

    with get_activity_connection() as connection:
        connection.execute(
            '''
            INSERT INTO app_user_sessions (
                session_id,
                user_id,
                session_token_hash,
                created_at,
                updated_at,
                expires_at,
                revoked_at,
                client_ip,
                user_agent
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''',
            (
                session_id,
                user_id,
                hash_session_token(session_token),
                timestamp,
                timestamp,
                expires_at,
                None,
                request.client.host if request.client is not None else None,
                request.headers.get('user-agent'),
            ),
        )

    return session_id, session_token


def set_session_cookie(response: Response, session_token: str) -> None:
    same_site = SESSION_COOKIE_SAMESITE if SESSION_COOKIE_SAMESITE in {'lax', 'strict', 'none'} else 'lax'
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_token,
        httponly=True,
        secure=SESSION_COOKIE_SECURE,
        samesite=same_site,
        max_age=SESSION_MAX_AGE,
        path='/',
        domain=SESSION_COOKIE_DOMAIN,
    )


def clear_session_cookie(response: Response) -> None:
    same_site = SESSION_COOKIE_SAMESITE if SESSION_COOKIE_SAMESITE in {'lax', 'strict', 'none'} else 'lax'
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path='/',
        httponly=True,
        samesite=same_site,
        secure=SESSION_COOKIE_SECURE,
        domain=SESSION_COOKIE_DOMAIN,
    )


def revoke_session(session_token: str | None) -> None:
    if not session_token:
        return

    with get_activity_connection() as connection:
        connection.execute(
            '''
            UPDATE app_user_sessions
            SET revoked_at = %s, updated_at = %s
            WHERE session_token_hash = %s AND revoked_at IS NULL
            ''',
            (utc_now_iso(), utc_now_iso(), hash_session_token(session_token)),
        )


def get_authenticated_user(request: Request) -> dict[str, Any]:
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_token:
        raise HTTPException(status_code=401, detail='Authentication required.')

    current_timestamp = utc_now_iso()
    with get_activity_connection() as connection:
        row = connection.execute(
            '''
            SELECT
                u.user_id,
                u.username,
                u.email,
                u.created_at,
                u.updated_at,
                u.last_login_at,
                s.session_id,
                s.expires_at
            FROM app_user_sessions s
            INNER JOIN app_users u ON u.user_id = s.user_id
            WHERE s.session_token_hash = %s
              AND s.revoked_at IS NULL
              AND s.expires_at > %s
            ''',
            (hash_session_token(session_token), current_timestamp),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=401, detail='Session expired or invalid.')

    return row


def create_app_user(*, username: str, email: str, password: str) -> dict[str, str]:
    normalized_username, normalized_email = validate_login_payload(username, email)
    validated_password = validate_password(password)

    if get_user_by_email(normalized_email) is not None:
        raise HTTPException(status_code=409, detail='An account with this email already exists.')

    with get_activity_connection() as connection:
        timestamp = utc_now_iso()
        user_id = uuid.uuid4().hex
        row = connection.execute(
            '''
            INSERT INTO app_users (
                user_id,
                username,
                email,
                password_hash,
                created_at,
                updated_at,
                last_login_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING user_id, username, email, created_at, updated_at, last_login_at
            ''',
            (
                user_id,
                normalized_username,
                normalized_email,
                hash_password(validated_password),
                timestamp,
                timestamp,
                timestamp,
            ),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=500, detail='Failed to store user details.')

    return build_user_payload(row)


def authenticate_user(*, email: str, password: str) -> dict[str, str]:
    normalized_email = normalize_email(email)
    validated_password = validate_password(password)
    row = get_user_by_email(normalized_email)

    if row is None or not verify_password(validated_password, row.get('password_hash')):
        raise HTTPException(status_code=401, detail='Invalid email or password.')

    timestamp = utc_now_iso()
    with get_activity_connection() as connection:
        updated_row = connection.execute(
            '''
            UPDATE app_users
            SET updated_at = %s, last_login_at = %s
            WHERE email = %s
            RETURNING user_id, username, email, created_at, updated_at, last_login_at
            ''',
            (timestamp, timestamp, normalized_email),
        ).fetchone()

    if updated_row is None:
        raise HTTPException(status_code=500, detail='Failed to load authenticated user.')

    return build_user_payload(updated_row)


def ensure_session_state(session_id: str) -> dict[str, Any]:
    if session_id not in SESSION_STATE:
        SESSION_STATE[session_id] = {
            'forecast_steps': {'ts': False, 'ml': False},
            'time_series_result': None,
            'ml_forecast_result': None,
            'updated_at': datetime.utcnow().isoformat(),
        }
    return SESSION_STATE[session_id]


def normalize_column_name(name: str) -> str:
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in name).strip('_').replace('__', '_')


def dataset_file_path(dataset_id: str, suffix: str = '.parquet') -> Path:
    return DATASET_DIR / f'{dataset_id}{suffix}'


def write_dataset_file(dataset_id: str, content: bytes, suffix: str = '.parquet') -> Path:
    target = dataset_file_path(dataset_id, suffix)
    target.write_bytes(content)
    return target


async def write_uploaded_file(upload_file: UploadFile, dataset_id: str, suffix: str) -> tuple[Path, int]:
    target = dataset_file_path(dataset_id, suffix)
    total_bytes = 0

    with target.open('wb') as handle:
        while True:
            chunk = await upload_file.read(UPLOAD_READ_CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                handle.close()
                try:
                    target.unlink(missing_ok=True)
                except Exception:
                    logger.exception('Failed to remove oversized uploaded dataset file %s', target)
                raise HTTPException(status_code=400, detail='File exceeds 512MB limit.')
            handle.write(chunk)

    await upload_file.seek(0)
    return target, total_bytes


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


def read_cached_csv_preview(dataset_entry: dict[str, Any], n_rows: int | None = None) -> pl.DataFrame:
    csv_path = dataset_entry.get('csv_path')
    if not csv_path:
        raise HTTPException(status_code=400, detail='Cached CSV dataset path is missing. Please upload the file again.')

    try:
        separator = '\t' if str(csv_path).lower().endswith('.tsv') else ','
        return pl.read_csv(csv_path, separator=separator, n_rows=n_rows, infer_schema_length=1000, ignore_errors=True)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f'Failed to load cached CSV preview: {error}') from error


def read_cached_csv(dataset_entry: dict[str, Any], columns: list[str] | None = None, n_rows: int | None = None) -> pd.DataFrame:
    csv_path = dataset_entry.get('csv_path')
    if not csv_path:
        raise HTTPException(status_code=400, detail='Cached CSV dataset path is missing. Please upload the file again.')

    try:
        return pd.read_csv(csv_path, usecols=columns, nrows=n_rows, low_memory=True)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f'Failed to load cached CSV dataset: {error}') from error


def read_cached_excel(dataset_entry: dict[str, Any], columns: list[str] | None = None, n_rows: int | None = None) -> pd.DataFrame:
    excel_path = dataset_entry.get('excel_path')
    if not excel_path:
        raise HTTPException(status_code=400, detail='Cached Excel dataset path is missing. Please upload the file again.')

    engine = 'openpyxl' if excel_path.lower().endswith('.xlsx') else 'xlrd'
    try:
        return pd.read_excel(excel_path, engine=engine, usecols=columns, nrows=n_rows)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f'Failed to load cached Excel dataset: {error}') from error


def load_cached_preview(dataset_entry: dict[str, Any], limit: int = PARQUET_PREVIEW_ROW_LIMIT) -> tuple[pd.DataFrame | pl.DataFrame, bool]:
    if dataset_entry.get('parquet_path'):
        return read_cached_parquet(dataset_entry, n_rows=limit, low_memory=True), True
    if dataset_entry.get('csv_path'):
        return read_cached_csv_preview(dataset_entry, n_rows=limit), True
    if dataset_entry.get('excel_path'):
        return read_cached_excel(dataset_entry, n_rows=limit), False
    if dataset_entry.get('frame_path'):
        return read_cached_frame(dataset_entry, n_rows=limit), False
    raise HTTPException(status_code=400, detail='Cached dataset storage is missing. Please upload the file again.')


def load_cached_analysis_sample(dataset_entry: dict[str, Any], limit: int = EDA_ADVANCED_SAMPLE_LIMIT) -> tuple[pd.DataFrame, int]:
    total_rows = int(dataset_entry.get('row_count') or 0)

    if dataset_entry.get('parquet_path'):
        frame = read_cached_parquet(dataset_entry, n_rows=limit, low_memory=True)
        return normalize_dataframe(frame.to_pandas(use_pyarrow_extension_array=False)), total_rows

    if dataset_entry.get('csv_path'):
        return normalize_dataframe(read_cached_csv(dataset_entry, n_rows=limit)), total_rows

    if dataset_entry.get('excel_path'):
        return normalize_dataframe(read_cached_excel(dataset_entry, n_rows=limit)), total_rows

    if dataset_entry.get('frame_path'):
        frame = read_cached_frame(dataset_entry)
        return sample_frame_for_eda(frame, limit), int(len(frame))

    raise HTTPException(status_code=400, detail='Cached dataset storage is missing. Please upload the file again.')


def count_csv_rows(buffer: io.BytesIO, sep: str = ',') -> int:
    buffer.seek(0)
    row_count = 0
    try:
        for chunk in pd.read_csv(buffer, sep=sep, low_memory=True, chunksize=100_000):
            row_count += len(chunk)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f'Failed to determine CSV row count: {error}') from error
    finally:
        buffer.seek(0)
    return row_count


def count_csv_rows_from_path(path: Path, sep: str = ',') -> int:
    row_count = 0
    try:
        for chunk in pd.read_csv(path, sep=sep, low_memory=True, chunksize=100_000):
            row_count += len(chunk)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f'Failed to determine CSV row count: {error}') from error
    return row_count


def count_excel_rows(buffer: io.BytesIO, filename: str) -> int:
    buffer.seek(0)
    suffix = Path(filename).suffix.lower()

    if suffix == '.xlsx':
        try:
            import openpyxl
        except ImportError as error:
            raise HTTPException(status_code=500, detail='openpyxl is required to count rows in .xlsx files.') from error

        workbook = openpyxl.load_workbook(buffer, read_only=True, data_only=True)
        row_count = workbook.active.max_row - 1
        buffer.seek(0)
        return max(0, row_count)

    if suffix == '.xls':
        try:
            import xlrd
        except ImportError as error:
            raise HTTPException(status_code=500, detail='xlrd is required to count rows in .xls files.') from error

        workbook = xlrd.open_workbook(file_contents=buffer.read())
        sheet = workbook.sheet_by_index(0)
        row_count = sheet.nrows - 1
        buffer.seek(0)
        return max(0, row_count)

    buffer.seek(0)
    return 0


def count_excel_rows_from_path(path: Path) -> int:
    suffix = path.suffix.lower()

    if suffix == '.xlsx':
        try:
            import openpyxl
        except ImportError as error:
            raise HTTPException(status_code=500, detail='openpyxl is required to count rows in .xlsx files.') from error

        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        return max(0, workbook.active.max_row - 1)

    if suffix == '.xls':
        try:
            import xlrd
        except ImportError as error:
            raise HTTPException(status_code=500, detail='xlrd is required to count rows in .xls files.') from error

        workbook = xlrd.open_workbook(path)
        sheet = workbook.sheet_by_index(0)
        return max(0, sheet.nrows - 1)

    return 0


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


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class AdvancedEdaRequest(BaseModel):
    data: list[dict[str, Any]] = Field(default_factory=list)
    dataset_id: str | None = None


class SalesForecastRequest(BaseModel):
    data: list[dict[str, Any]] = Field(default_factory=list)
    dataset_id: str | None = None
    session_id: str | None = None
    date_column: str
    target_column: str
    forecast_periods: int = Field(default=3, ge=1, le=24)
    test_percentage: int = Field(default=20, ge=10, le=50)
    test_periods: int | None = Field(default=None, ge=1, le=24)
    lag_periods: int = Field(default=3, ge=1, le=12)
    model_type: str | None = None
    feature_groups: list[str] = Field(default_factory=lambda: ['trend', 'calendar', 'seasonality', 'lags', 'rolling'])


class TimeSeriesForecastRequest(BaseModel):
    data: list[dict[str, Any]] = Field(default_factory=list)
    dataset_id: str | None = None
    session_id: str | None = None
    date_column: str
    target_column: str
    forecast_periods: int = Field(default=3, ge=1, le=24)
    test_percentage: int = Field(default=20, ge=10, le=50)
    model_type: str = Field(default='sarima')


class MlForecastRequest(BaseModel):
    data: list[dict[str, Any]] = Field(default_factory=list)
    dataset_id: str | None = None
    session_id: str | None = None
    date_column: str
    target_column: str
    forecast_periods: int = Field(default=3, ge=1, le=24)
    test_percentage: int = Field(default=20, ge=10, le=50)
    lag_periods: int = Field(default=3, ge=1, le=12)
    model_type: str = Field(default='gradient_boosting')
    feature_groups: list[str] = Field(default_factory=lambda: ['trend', 'calendar', 'lags', 'rolling'])


class CleaningLog(BaseModel):
    action: str
    detail: str
    timestamp: str


class CleaningJustificationRequest(BaseModel):
    logs: list[CleaningLog]
    totalRows: int
    totalColumns: int
    fileName: str | None = None
    loadedRowCount: int | None = None
    previewLoaded: bool = False

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


class ForecastPointPayload(BaseModel):
    period: str
    actual: float | None = None
    predicted: float | None = None
    lower: float | None = None
    upper: float | None = None


class ForecastMetricsPayload(BaseModel):
    mae: float
    rmse: float
    mape: float


class ForecastTrainingSummaryPayload(BaseModel):
    model_name: str
    total_periods: int
    train_periods: int
    test_periods: int
    train_percentage: float
    test_percentage: float
    forecast_periods: int
    lag_periods: int = 0
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    last_observed_period: str


class DatasetProfilePayload(BaseModel):
    detected_frequency: str
    usable_periods: int
    volatility: float
    zero_value_share: float


class StationarityCheckPayload(BaseModel):
    test_name: str
    p_value: float
    verdict: str
    note: str


class TimeSeriesForecastResultPayload(BaseModel):
    date_column: str
    target_column: str
    frequency: str | None = None
    period_label: str | None = None
    dataset_profile: DatasetProfilePayload
    stationarity_check: StationarityCheckPayload
    history: list[ForecastPointPayload] = Field(default_factory=list)
    test_forecast: list[ForecastPointPayload] = Field(default_factory=list)
    future_forecast: list[ForecastPointPayload] = Field(default_factory=list)
    metrics: ForecastMetricsPayload
    training_summary: ForecastTrainingSummaryPayload
    recommended_models: list[dict[str, Any]] = Field(default_factory=list)
    model_details: dict[str, Any] = Field(default_factory=dict)
    analysis: str


class MlForecastResultPayload(BaseModel):
    date_column: str
    target_column: str
    frequency: str | None = None
    period_label: str | None = None
    dataset_profile: DatasetProfilePayload
    generated_features: list[str] = Field(default_factory=list)
    feature_preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    history: list[ForecastPointPayload] = Field(default_factory=list)
    test_forecast: list[ForecastPointPayload] = Field(default_factory=list)
    future_forecast: list[ForecastPointPayload] = Field(default_factory=list)
    metrics: ForecastMetricsPayload
    training_summary: dict[str, Any]
    shap_feature_importance: list[dict[str, Any]] = Field(default_factory=list)
    recommended_models: list[dict[str, Any]] = Field(default_factory=list)
    model_details: dict[str, Any] = Field(default_factory=dict)
    analysis: str


class ReportPayload(BaseModel):
    datasetId: str | None = None
    sessionId: str | None = None
    fileName: str
    totalRows: int
    previewLoaded: bool = False
    loadedRowCount: int | None = None
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
    timeSeriesForecastResult: TimeSeriesForecastResultPayload | None = None
    mlForecastResult: MlForecastResultPayload | None = None
    forecastingStepsCompleted: list[int] = Field(default_factory=list)
    predictionResult: str | float | int | None = None
    predictionAnalysis: str | None = None
    predictionProbabilities: dict[str, float] | None = None
    predictionHistory: list[PredictionHistoryItem] = Field(default_factory=list)
    edaStats: EdaStats = Field(default_factory=EdaStats)


class EdaPdfPayload(BaseModel):
    datasetId: str | None = None
    fileName: str
    totalRows: int
    loadedRowCount: int | None = None
    previewLoaded: bool = False
    columns: list[ColumnInfo] = Field(default_factory=list)
    edaStats: EdaStats = Field(default_factory=EdaStats)
    advancedAnalysis: dict[str, Any] | None = None


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

        if dataset_entry.get('parquet_path'):
            parquet_frame = read_cached_parquet(dataset_entry, columns=resolved_selected_columns, low_memory=True)
            parquet_frame.columns = required_columns
            return normalize_dataframe(parquet_frame.to_pandas(use_pyarrow_extension_array=False))

        if dataset_entry.get('csv_path'):
            frame = read_cached_csv(dataset_entry, columns=resolved_selected_columns)
            frame.columns = required_columns
            return normalize_dataframe(frame)

        if dataset_entry.get('excel_path'):
            frame = read_cached_excel(dataset_entry, columns=resolved_selected_columns)
            frame.columns = required_columns
            return normalize_dataframe(frame)

    if not data:
        raise HTTPException(status_code=400, detail='Dataset rows are required.')

    frame = normalize_dataframe(pd.DataFrame(data))
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f'Missing columns: {missing_columns}')
    return frame[required_columns].copy()


def load_full_dataset_frame(dataset_id: str | None, data: list[dict[str, Any]]) -> pd.DataFrame:
    if dataset_id:
        dataset_entry = DATASET_CACHE.get(dataset_id)
        if dataset_entry is None:
            raise HTTPException(status_code=400, detail='Cached dataset not found. Please upload the file again.')

        if dataset_entry.get('frame_path'):
            return normalize_dataframe(read_cached_frame(dataset_entry))

        if dataset_entry.get('parquet_path'):
            parquet_frame = read_cached_parquet(dataset_entry, low_memory=True)
            return normalize_dataframe(parquet_frame.to_pandas(use_pyarrow_extension_array=False))

        if dataset_entry.get('csv_path'):
            return normalize_dataframe(read_cached_csv(dataset_entry))

        if dataset_entry.get('excel_path'):
            return normalize_dataframe(read_cached_excel(dataset_entry))

    if not data:
        raise HTTPException(status_code=400, detail='Dataset rows are required.')

    frame = normalize_dataframe(pd.DataFrame(data))
    if frame.empty or frame.shape[1] == 0:
        raise HTTPException(status_code=400, detail='Dataset must contain at least one row and one column.')
    return frame


def sample_frame_for_eda(frame: pd.DataFrame, limit: int = EDA_ADVANCED_SAMPLE_LIMIT) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.copy()
    indices = np.linspace(0, len(frame) - 1, num=limit, dtype=int)
    return frame.iloc[indices].reset_index(drop=True).copy()


def safe_numeric_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
    return numeric.astype(float)


def is_identifier_like_name(column_name: str) -> bool:
    normalized = re.sub(r'[^a-z0-9]+', '_', str(column_name).strip().lower())
    return bool(re.search(r'(^|_)(id|uuid|guid|token|hash|key|session)(_|$)', normalized))


def is_identifier_like_categorical(series: pd.Series, column_name: str) -> bool:
    if is_identifier_like_name(column_name):
        return True
    cleaned = series.dropna().astype(str).replace(r'^\s*$', np.nan, regex=True).dropna()
    if cleaned.empty:
        return False
    unique_count = int(cleaned.nunique())
    unique_ratio = unique_count / max(1, len(cleaned))
    average_length = float(cleaned.str.len().mean()) if not cleaned.empty else 0.0
    return unique_count >= 50 and unique_ratio >= 0.85 and average_length >= 8


def is_identifier_like_numeric(series: pd.Series, column_name: str) -> bool:
    if is_identifier_like_name(column_name):
        return True
    values = safe_numeric_series(series)
    if values.empty:
        return False
    unique_count = int(values.nunique())
    unique_ratio = unique_count / max(1, len(values))
    return unique_count >= 50 and unique_ratio >= 0.98


def figure_to_base64(figure: go.Figure, *, width: int = 1400, height: int = 700) -> str | None:
    if not ENABLE_PLOTLY_STATIC_EXPORT:
        return None
    try:
        figure.update_layout(
            template='plotly_white',
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(l=40, r=30, t=60, b=40),
        )
        image_bytes = figure.to_image(format='png', width=width, height=height, scale=2)
        return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
    except Exception:
        logger.exception('Advanced EDA chart rendering failed')
        return None


def matplotlib_figure_to_base64(fig: plt.Figure) -> str | None:
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=180, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('ascii')}"
    except Exception:
        logger.exception('Matplotlib chart rendering failed')
        return None
    finally:
        plt.close(fig)


def build_missingness_chart_matplotlib(matrix: list[list[float]], columns: list[str], y_labels: list[str]) -> str | None:
    try:
        fig_width = max(10, min(20, 4 + len(columns) * 0.7))
        fig_height = max(4.5, min(12, 2.8 + len(y_labels) * 0.12))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(np.array(matrix), aspect='auto', cmap='BuGn', vmin=0, vmax=100)
        ax.set_title('Missingness intensity across row groups', fontsize=12, fontweight='bold')
        ax.set_xlabel('Columns with missing values')
        ax.set_ylabel('Row index percentile groups')
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels(columns, rotation=35, ha='right', fontsize=8)
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=8)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.03)
        colorbar.set_label('Missing %')
        fig.tight_layout()
        return matplotlib_figure_to_base64(fig)
    except Exception:
        logger.exception('Missingness matplotlib fallback failed')
        return None


def build_distribution_chart_matplotlib(frame: pd.DataFrame, selected_columns: list[str]) -> str | None:
    try:
        fig, axes = plt.subplots(len(selected_columns), 2, figsize=(14, max(4, len(selected_columns) * 2.8)))
        axes_array = np.atleast_2d(axes)
        for row_index, column in enumerate(selected_columns):
            values = safe_numeric_series(frame[column])
            hist_ax = axes_array[row_index, 0]
            box_ax = axes_array[row_index, 1]
            if values.empty:
                hist_ax.text(0.5, 0.5, 'No numeric data', ha='center', va='center', fontsize=9, color='#64748b')
                box_ax.text(0.5, 0.5, 'No numeric data', ha='center', va='center', fontsize=9, color='#64748b')
                hist_ax.set_axis_off()
                box_ax.set_axis_off()
                continue
            raw_values = values.to_numpy(dtype=float)
            hist_ax.hist(raw_values, bins=min(40, max(12, int(np.sqrt(raw_values.size)))), density=True, color='#38bdf8', alpha=0.75, edgecolor='white')
            kde = estimate_kde(raw_values)
            if kde is not None:
                hist_ax.plot(kde[0], kde[1], color='#7c3aed', linewidth=2)
            hist_ax.set_title(f'{column} distribution', fontsize=10, fontweight='bold')
            hist_ax.grid(alpha=0.18)
            box_ax.boxplot(raw_values, vert=False, patch_artist=True, boxprops=dict(facecolor='#10b981', alpha=0.55), medianprops=dict(color='#065f46', linewidth=2))
            box_ax.set_title(f'{column} outlier view', fontsize=10, fontweight='bold')
            box_ax.grid(alpha=0.18)
        fig.tight_layout()
        return matplotlib_figure_to_base64(fig)
    except Exception:
        logger.exception('Distribution matplotlib fallback failed')
        return None


def build_distribution_chart_for_column(frame: pd.DataFrame, column: str) -> str | None:
    values = safe_numeric_series(frame[column])
    if values.empty:
        return None
    raw_values = values.to_numpy(dtype=float)
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f'{column} distribution', f'{column} outlier view'),
        horizontal_spacing=0.1,
    )
    figure.add_trace(
        go.Histogram(
            x=raw_values,
            nbinsx=min(40, max(12, int(np.sqrt(raw_values.size)))),
            histnorm='probability density',
            marker=dict(color='rgba(14,165,233,0.78)'),
            name=f'{column} histogram',
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    kde = estimate_kde(raw_values)
    if kde is not None:
        figure.add_trace(
            go.Scatter(
                x=kde[0],
                y=kde[1],
                mode='lines',
                line=dict(color='#7c3aed', width=2),
                name=f'{column} KDE',
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Box(
            x=raw_values,
            orientation='h',
            marker=dict(color='#10b981'),
            line=dict(color='#047857'),
            boxmean='sd',
            name=f'{column} boxplot',
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.update_layout(
        title=f'{column}: distribution and outlier screening',
        title_x=0.5,
        width=1380,
        height=520,
        margin=dict(l=90, r=50, t=90, b=80),
        template='plotly_white',
    )
    figure.update_xaxes(title_text='Value', automargin=True, tickangle=0)
    figure.update_yaxes(automargin=True)
    chart_base64 = figure_to_base64(figure, width=1380, height=520)
    if chart_base64 is not None:
        return chart_base64
    return build_distribution_chart_matplotlib(frame, [column])


def build_categorical_chart_matplotlib(counts: pd.Series, title: str) -> str | None:
    try:
        labels = [str(value) for value in counts.index[::-1]]
        values = counts.values[::-1]
        fig_height = max(4, 1.2 + len(labels) * 0.5)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.barh(labels, values, color='#0ea5e9', alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Count')
        ax.grid(axis='x', alpha=0.18)
        fig.tight_layout()
        return matplotlib_figure_to_base64(fig)
    except Exception:
        logger.exception('Categorical matplotlib fallback failed')
        return None


def build_interaction_chart_matplotlib(x_values: np.ndarray, y_values: np.ndarray, title: str, x_label: str, y_label: str) -> str | None:
    try:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.scatter(x_values, y_values, s=22, alpha=0.7, color='#0ea5e9')
        if np.unique(x_values).size > 1:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            line_x = np.linspace(float(x_values.min()), float(x_values.max()), 100)
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, color='#7c3aed', linewidth=2)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.18)
        fig.tight_layout()
        return matplotlib_figure_to_base64(fig)
    except Exception:
        logger.exception('Interaction matplotlib fallback failed')
        return None


def estimate_kde(values: np.ndarray, point_count: int = 160) -> tuple[np.ndarray, np.ndarray] | None:
    if values.size < 2:
        return None
    std = float(np.std(values))
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if not np.isfinite(std) or std == 0 or min_value == max_value:
        return None
    bandwidth = 1.06 * std * (values.size ** (-1 / 5))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = max((max_value - min_value) / 25, 1e-6)
    x_points = np.linspace(min_value, max_value, point_count)
    scaled = (x_points[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled ** 2).sum(axis=1) / (values.size * bandwidth * np.sqrt(2 * np.pi))
    return x_points, density


def build_missingness_payload(frame: pd.DataFrame) -> dict[str, Any]:
    total_missing = int(frame.isna().sum().sum())
    if total_missing == 0:
        return {
            'status': 'success',
            'message': 'Data Quality: No missing values detected in the dataset.',
            'total_missing': 0,
            'chart_base64': None,
            'columns_analyzed': [],
            'row_groups': 0,
        }

    missing_columns = [str(column) for column in frame.columns if frame[column].isna().any()]
    display_columns = missing_columns[:EDA_MAX_MISSINGNESS_COLUMNS]
    display_frame = frame[display_columns].isna().astype(float).reset_index(drop=True)
    bucket_count = min(EDA_MISSINGNESS_BUCKETS, max(8, len(display_frame)))
    bucket_edges = np.linspace(0, len(display_frame), num=bucket_count + 1, dtype=int)

    matrix: list[list[float]] = []
    y_labels: list[str] = []
    for index in range(bucket_count):
        start = int(bucket_edges[index])
        end = int(bucket_edges[index + 1])
        if end <= start:
            continue
        segment = display_frame.iloc[start:end]
        matrix.append((segment.mean(axis=0) * 100).round(2).tolist())
        start_pct = int(round((start / max(len(display_frame), 1)) * 100))
        end_pct = int(round((end / max(len(display_frame), 1)) * 100))
        y_labels.append(f'{start_pct}-{end_pct}%')

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=display_columns,
            y=y_labels,
            colorscale='Teal',
            colorbar=dict(title='Missing %'),
            hovertemplate='Row Group %{y}<br>Column %{x}<br>Missing %{z:.2f}%<extra></extra>',
        )
    )
    figure.update_layout(
        title='Missingness intensity across row groups',
        xaxis_title='Columns with missing values',
        yaxis_title='Row index percentile groups',
        margin=dict(l=100, r=40, t=80, b=120),
        xaxis=dict(tickangle=-45, automargin=True),
        yaxis=dict(automargin=True),
    )
    chart_base64 = figure_to_base64(
        figure,
        width=max(1100, 180 + len(display_columns) * 85),
        height=max(500, 220 + len(y_labels) * 8),
    )
    if chart_base64 is None:
        chart_base64 = build_missingness_chart_matplotlib(matrix, display_columns, y_labels)
    if chart_base64 is None:
        return {
            'status': 'error',
            'message': 'Missingness visualization could not be generated for this dataset.',
            'total_missing': total_missing,
            'chart_base64': None,
            'columns_analyzed': display_columns,
            'row_groups': len(y_labels),
        }
    note = None
    if len(missing_columns) > len(display_columns):
        note = f'Displaying the first {len(display_columns)} columns with missing values to keep the view stable.'
    return {
        'status': 'chart',
        'message': note,
        'total_missing': total_missing,
        'chart_base64': chart_base64,
        'columns_analyzed': display_columns,
        'row_groups': len(y_labels),
    }


def build_distribution_payload(frame: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = [
        str(column) for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])
    ]
    if not numeric_columns:
        return {
            'status': 'empty',
            'message': 'No numeric columns available for this analysis.',
            'chart_base64': None,
            'columns_analyzed': [],
            'charts': [],
        }

    selected_columns = numeric_columns[:EDA_MAX_NUMERIC_CHARTS]
    chart_payloads = [
        {
            'column': column,
            'chart_base64': build_distribution_chart_for_column(frame, column),
        }
        for column in selected_columns
    ]
    chart_payloads = [payload for payload in chart_payloads if payload['chart_base64']]
    if not chart_payloads:
        return {
            'status': 'error',
            'message': 'Distribution charts could not be generated for this dataset.',
            'chart_base64': None,
            'columns_analyzed': selected_columns,
            'charts': [],
        }
    return {
        'status': 'chart',
        'message': None if len(numeric_columns) <= len(selected_columns) else f'Displaying the first {len(selected_columns)} numeric columns to prevent oversized browser rendering.',
        'chart_base64': chart_payloads[0]['chart_base64'],
        'columns_analyzed': selected_columns,
        'charts': chart_payloads,
    }


def build_categorical_payload(frame: pd.DataFrame) -> dict[str, Any]:
    all_categorical_columns = [
        str(column) for column in frame.columns
        if (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
            or pd.api.types.is_categorical_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
        )
    ]
    categorical_columns = [column for column in all_categorical_columns if not is_identifier_like_categorical(frame[column], column)]
    if not categorical_columns:
        message = 'No categorical columns available for this analysis.'
        if all_categorical_columns:
            message = 'Categorical-looking columns were detected, but they appear to be identifier-like fields and were excluded from charting.'
        return {
            'status': 'empty',
            'message': message,
            'charts': [],
            'warnings': [],
        }

    warnings_payload: list[dict[str, Any]] = []
    for column in categorical_columns:
        unique_count = int(frame[column].dropna().astype(str).replace(r'^\s*$', np.nan, regex=True).dropna().nunique())
        if unique_count > 20:
            warnings_payload.append({
                'column': column,
                'unique_count': unique_count,
                'message': f"High Cardinality: Column '{column}' has {unique_count} unique values. Consider encoding strategies before ML.",
            })

    chart_payloads: list[dict[str, Any]] = []
    for column in categorical_columns[:EDA_MAX_CATEGORICAL_CHARTS]:
        series = frame[column].copy()
        series = series.astype(object).where(pd.notna(series), 'Missing')
        labels = pd.Series(series).astype(str).replace(r'^\s*$', '(blank)', regex=True)
        counts = labels.value_counts().head(EDA_MAX_CATEGORY_BARS)
        if counts.empty:
            continue
        longest_label = max((len(str(value)) for value in counts.index), default=0)
        left_margin = min(420, max(220, 110 + longest_label * 7))
        figure = go.Figure(
            data=go.Bar(
                x=counts.values[::-1],
                y=[str(value) for value in counts.index[::-1]],
                orientation='h',
                marker=dict(color='#0ea5e9'),
                hovertemplate='%{y}<br>Count %{x}<extra></extra>',
            )
        )
        figure.update_layout(
            title=f'Top categories for {column}',
            xaxis_title='Count',
            yaxis_title='Category',
            margin=dict(l=left_margin, r=50, t=80, b=60),
            yaxis=dict(automargin=True, tickfont=dict(size=11)),
            xaxis=dict(automargin=True, tickfont=dict(size=11)),
        )
        chart_base64 = figure_to_base64(figure, width=1440, height=max(480, 140 + len(counts) * 42))
        if chart_base64 is None:
            chart_base64 = build_categorical_chart_matplotlib(counts, f'Top categories for {column}')
        chart_payloads.append({
            'column': column,
            'unique_count': int(labels.nunique()),
            'chart_base64': chart_base64,
        })

    status = 'chart' if chart_payloads else 'error'
    message_parts: list[str] = []
    excluded_identifier_like = len(all_categorical_columns) - len(categorical_columns)
    if excluded_identifier_like > 0:
        message_parts.append(f'Excluded {excluded_identifier_like} identifier-like categorical column{"s" if excluded_identifier_like != 1 else ""} from charting.')
    if len(categorical_columns) > EDA_MAX_CATEGORICAL_CHARTS:
        message_parts.append(f'Displaying the first {EDA_MAX_CATEGORICAL_CHARTS} categorical columns to keep the analysis responsive.')
    message = ' '.join(message_parts) or None
    if status == 'error':
        message = 'Categorical charts could not be generated for this dataset.'
    return {
        'status': status,
        'message': message,
        'charts': chart_payloads,
        'warnings': warnings_payload[:12],
    }


def build_interaction_payload(frame: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = [
        str(column) for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])
    ]
    if len(numeric_columns) < 2:
        return {
            'status': 'empty',
            'message': 'Need at least 2 numeric columns.',
            'plots': [],
        }

    eligible_numeric_columns = [column for column in numeric_columns if not is_identifier_like_numeric(frame[column], column)]
    numeric_frame = frame[eligible_numeric_columns].apply(pd.to_numeric, errors='coerce')
    numeric_frame = numeric_frame.replace([np.inf, -np.inf], np.nan)
    non_constant_columns = [column for column in numeric_frame.columns if numeric_frame[column].dropna().nunique() > 1]
    numeric_frame = numeric_frame[non_constant_columns[:EDA_MAX_INTERACTION_COLUMNS]]
    if numeric_frame.shape[1] < 2:
        excluded_identifier_like = len(numeric_columns) - len(eligible_numeric_columns)
        message = 'Need at least 2 numeric columns.'
        if excluded_identifier_like > 0:
            message = f'Need at least 2 non-identifier numeric columns. Excluded {excluded_identifier_like} identifier-like numeric column{"s" if excluded_identifier_like != 1 else ""}.'
        return {
            'status': 'empty',
            'message': message,
            'plots': [],
        }

    correlation_matrix = numeric_frame.corr().fillna(0)
    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(correlation_matrix.columns):
        for right in correlation_matrix.columns[index + 1:]:
            correlation = float(correlation_matrix.loc[left, right])
            if np.isnan(correlation):
                correlation = 0.0
            pairs.append({'x': str(left), 'y': str(right), 'correlation': correlation})
    pairs.sort(key=lambda item: abs(item['correlation']), reverse=True)

    plots: list[dict[str, Any]] = []
    for pair in pairs[:EDA_MAX_INTERACTION_PAIRS]:
        pair_frame = numeric_frame[[pair['x'], pair['y']]].dropna()
        if len(pair_frame) < 2:
            continue
        x_values = pair_frame[pair['x']].to_numpy(dtype=float)
        y_values = pair_frame[pair['y']].to_numpy(dtype=float)
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                marker=dict(color='#0ea5e9', size=8, opacity=0.7),
                name='Observed values',
                showlegend=False,
            )
        )
        if np.unique(x_values).size > 1:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            line_x = np.linspace(float(x_values.min()), float(x_values.max()), 100)
            line_y = slope * line_x + intercept
            figure.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode='lines',
                    line=dict(color='#7c3aed', width=2),
                    name='OLS trend',
                    showlegend=False,
                )
            )
        figure.update_layout(
            title=f"{pair['x']} vs {pair['y']}",
            xaxis_title=pair['x'],
            yaxis_title=pair['y'],
            margin=dict(l=130, r=50, t=80, b=100),
            xaxis=dict(tickangle=-20, automargin=True, tickfont=dict(size=11)),
            yaxis=dict(automargin=True, tickfont=dict(size=11)),
        )
        chart_base64 = figure_to_base64(figure, width=1400, height=560)
        if chart_base64 is None:
            chart_base64 = build_interaction_chart_matplotlib(x_values, y_values, f"{pair['x']} vs {pair['y']}", pair['x'], pair['y'])
        plots.append({
            'pair': f"{pair['x']} vs {pair['y']}",
            'correlation': round(pair['correlation'], 4),
            'chart_base64': chart_base64,
        })

    if not plots:
        return {
            'status': 'empty',
            'message': 'Need at least 2 numeric columns.',
            'plots': [],
        }
    message_parts: list[str] = []
    excluded_identifier_like = len(numeric_columns) - len(eligible_numeric_columns)
    if excluded_identifier_like > 0:
        message_parts.append(f'Excluded {excluded_identifier_like} identifier-like numeric column{"s" if excluded_identifier_like != 1 else ""} from interaction analysis.')
    if len(non_constant_columns) > EDA_MAX_INTERACTION_COLUMNS:
        message_parts.append(f'Interaction search was capped to the first {EDA_MAX_INTERACTION_COLUMNS} non-constant numeric columns for stability.')
    return {
        'status': 'chart',
        'message': ' '.join(message_parts) or None,
        'plots': plots,
    }


def build_automated_insights(frame: pd.DataFrame) -> dict[str, Any]:
    insights: list[str] = []

    try:
        numeric_columns = [
            str(column) for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])
        ]
    except Exception:
        numeric_columns = []

    for column in numeric_columns[:EDA_MAX_INTERACTION_COLUMNS]:
        try:
            values = safe_numeric_series(frame[column])
            if len(values) < 8:
                continue
            skewness = float(values.skew())
            if np.isfinite(skewness) and abs(skewness) > 1:
                insights.append(f"'{column}' is highly skewed (Skew: {skewness:.2f}). Consider Log/Box-Cox transformation.")
        except Exception:
            continue

    try:
        if len(numeric_columns) >= 2:
            numeric_frame = frame[numeric_columns[:EDA_MAX_INTERACTION_COLUMNS]].apply(pd.to_numeric, errors='coerce')
            numeric_frame = numeric_frame.replace([np.inf, -np.inf], np.nan)
            numeric_frame = numeric_frame[[column for column in numeric_frame.columns if numeric_frame[column].dropna().nunique() > 1]]
            if numeric_frame.shape[1] >= 2:
                corr_matrix = numeric_frame.corr().fillna(0)
                for index, left in enumerate(corr_matrix.columns):
                    for right in corr_matrix.columns[index + 1:]:
                        corr_value = float(corr_matrix.loc[left, right])
                        if abs(corr_value) > 0.95:
                            insights.append(f"'{left}' and '{right}' are highly correlated (>0.95). Consider dropping one to prevent multicollinearity.")
    except Exception:
        pass

    for column in numeric_columns[:EDA_MAX_INTERACTION_COLUMNS]:
        try:
            values = safe_numeric_series(frame[column])
            if len(values) < 8:
                continue
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr <= 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((values < lower) | (values > upper)).sum())
            if outlier_count >= max(3, int(len(values) * 0.01)):
                insights.append(f"Extreme outliers detected in '{column}'.")
        except Exception:
            continue

    deduped_insights = list(dict.fromkeys(insights))
    return {
        'status': 'success',
        'message': 'No major statistical anomalies detected.' if not deduped_insights else None,
        'insights': deduped_insights[:20],
    }


def build_advanced_eda_payload(request: AdvancedEdaRequest) -> dict[str, Any]:
    if request.dataset_id:
        dataset_entry = DATASET_CACHE.get(request.dataset_id)
        if dataset_entry is None:
            raise HTTPException(status_code=400, detail='Cached dataset not found. Please upload the file again.')
        analysis_frame, total_rows = load_cached_analysis_sample(dataset_entry)
        if analysis_frame.empty or analysis_frame.shape[1] == 0:
            raise HTTPException(status_code=400, detail='Dataset must contain at least one row and one column.')
        row_count = total_rows if total_rows > 0 else int(len(analysis_frame))
        column_count = int(dataset_entry.get('column_count') or len(analysis_frame.columns))
    else:
        frame = load_full_dataset_frame(request.dataset_id, request.data)
        if frame.empty or frame.shape[1] == 0:
            raise HTTPException(status_code=400, detail='Dataset must contain at least one row and one column.')
        analysis_frame = sample_frame_for_eda(frame)
        row_count = int(len(frame))
        column_count = int(len(frame.columns))

    return {
        'row_count': row_count,
        'sampled_row_count': int(len(analysis_frame)),
        'column_count': column_count,
        'missingness': build_missingness_payload(analysis_frame),
        'distributions': build_distribution_payload(analysis_frame),
        'categorical': build_categorical_payload(analysis_frame),
        'interactions': build_interaction_payload(analysis_frame),
        'insights': build_automated_insights(analysis_frame),
    }


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


def normal_tail_probability(value: float) -> float:
    return 0.5 * (1 - erf(abs(value) / sqrt(2)))


def compute_stationarity_check(values: list[float]) -> dict[str, Any]:
    if len(values) < 6:
        return {
            'test_name': 'Dickey-Fuller',
            'p_value': 1.0,
            'verdict': 'Insufficient history',
            'note': 'At least 6 periods are needed for a meaningful stationarity diagnostic.',
        }

    y_t = np.array(values[1:], dtype=float)
    y_tm1 = np.array(values[:-1], dtype=float)
    x = np.column_stack([np.ones(len(y_tm1)), y_tm1])
    beta, *_ = np.linalg.lstsq(x, y_t, rcond=None)
    residuals = y_t - x @ beta
    dof = max(len(y_t) - x.shape[1], 1)
    sigma2 = float(np.sum(residuals ** 2) / dof)
    cov = sigma2 * np.linalg.inv(x.T @ x)
    se = float(np.sqrt(max(cov[1, 1], 1e-9)))
    test_stat = float((beta[1] - 1.0) / se)
    p_value = round(min(1.0, max(0.0, 2 * normal_tail_probability(test_stat))), 4)
    stationary = p_value < 0.05
    return {
        'test_name': 'Dickey-Fuller',
        'p_value': p_value,
        'verdict': 'Likely stationary' if stationary else 'Trend or seasonality still present',
        'note': 'The series looks stationary enough for difference-based models.' if stationary else 'The series still shows a trend or seasonal structure, so seasonal statistical models are favored.',
    }


def build_dataset_profile(series_frame: pd.DataFrame, period_label: str) -> dict[str, Any]:
    values = series_frame['sales'].astype(float).tolist()
    mean = float(np.mean(values)) if values else 0.0
    volatility = 0.0 if mean == 0 else float(np.std(values) / abs(mean))
    zero_share = float(sum(value == 0 for value in values) / len(values)) if values else 0.0
    return {
        'detected_frequency': period_label,
        'usable_periods': int(len(series_frame)),
        'volatility': round(volatility, 4),
        'zero_value_share': round(zero_share, 4),
    }


def infer_season_length(period_label: str, total_periods: int) -> int:
    default = {'day': 7, 'week': 4, 'month': 12, 'quarter': 4, 'year': 1}.get(period_label, 1)
    return max(1, min(default, max(1, total_periods // 2)))


def resolve_time_series_model_name(model_type: str) -> str:
    mapping = {'sarima': 'SARIMA', 'prophet': 'Prophet', 'arima': 'ARIMA'}
    return mapping.get(model_type, 'SARIMA')


def statistical_forecast_step(history: list[float], season_length: int, model_type: str) -> float:
    recent = history[-min(3, len(history)):]
    recent_mean = float(np.mean(recent))
    trend = float(np.mean(np.diff(history[-min(5, len(history)):])) if len(history) > 1 else 0.0)
    seasonal_value = float(history[-season_length]) if season_length > 1 and len(history) >= season_length else recent_mean

    if model_type == 'arima':
        prediction = recent_mean + (0.35 * trend)
    elif model_type == 'prophet':
        prediction = (0.55 * recent_mean) + (0.45 * seasonal_value) + (0.65 * trend)
    else:
        prediction = (0.4 * recent_mean) + (0.6 * seasonal_value) + (0.4 * trend)

    return max(0.0, float(prediction))


def build_confidence_bounds(prediction: float, residual_std: float) -> tuple[float, float]:
    interval = 1.96 * residual_std
    lower = max(0.0, prediction - interval)
    upper = max(lower, prediction + interval)
    return round(lower, 2), round(upper, 2)


def build_ml_forecast_feature_row(
    history: list[float],
    current_period: pd.Timestamp,
    lag_periods: int,
    feature_groups: list[str],
) -> dict[str, float]:
    if len(history) < lag_periods:
        raise ValueError('Not enough history to build forecast features.')

    row: dict[str, float] = {}
    if 'trend' in feature_groups:
        row['trend_index'] = float(len(history) + 1)
    if 'calendar' in feature_groups:
        row['month_number'] = float(current_period.month)
        row['quarter_number'] = float(current_period.quarter)
        row['weekday_number'] = float(current_period.dayofweek)
        row['is_month_end'] = float(int(current_period.is_month_end))
    if 'lags' in feature_groups:
        for lag_index in range(1, lag_periods + 1):
            row[f'lag_{lag_index}'] = float(history[-lag_index])
    if 'rolling' in feature_groups:
        row['rolling_mean_3'] = float(np.mean(history[-min(3, len(history)):]))
        row['rolling_mean_6'] = float(np.mean(history[-min(6, len(history)):]))
        row['rolling_std_3'] = float(np.std(history[-min(3, len(history)):]))
    return row


def build_ml_forecast_training_frame(series_frame: pd.DataFrame, lag_periods: int, feature_groups: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    rows: list[dict[str, float]] = []
    targets: list[float] = []
    periods = series_frame['period'].tolist()
    values = series_frame['sales'].astype(float).tolist()
    for index in range(lag_periods, len(values)):
        rows.append(build_ml_forecast_feature_row(values[:index], periods[index], lag_periods, feature_groups))
        targets.append(float(values[index]))
    if len(rows) < 3:
        raise HTTPException(status_code=400, detail='Not enough history is available to train the ML forecasting model.')
    return pd.DataFrame(rows), pd.Series(targets)


def build_forecast_regressor(model_type: str):
    if model_type == 'random_forest':
        return RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=2, n_jobs=TRAINING_N_JOBS)
    if model_type == 'ridge_regression':
        return Ridge(alpha=1.0, random_state=42)
    return GradientBoostingRegressor(random_state=42, n_estimators=140, learning_rate=0.05, max_depth=3)


def recursive_ml_forecast(
    model: Any,
    history: list[float],
    start_period: pd.Timestamp,
    periods: int,
    lag_periods: int,
    feature_groups: list[str],
    freq: str,
    period_label: str,
) -> list[dict[str, Any]]:
    forecasts: list[dict[str, Any]] = []
    running_history = list(history)
    current_period = pd.Timestamp(start_period)
    for _ in range(periods):
        features = build_ml_forecast_feature_row(running_history, current_period, lag_periods, feature_groups)
        prediction = max(0.0, float(model.predict(pd.DataFrame([features]))[0]))
        forecasts.append({'period': format_forecast_period(current_period, period_label), 'predicted': round(prediction, 2)})
        running_history.append(prediction)
        current_period = current_period + pd.tseries.frequencies.to_offset(freq)
    return forecasts


def calculate_shap_like_importance(model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    if hasattr(model, 'feature_importances_'):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, 'coef_'):
        values = np.abs(np.asarray(model.coef_, dtype=float).reshape(-1))
    else:
        values = np.ones(len(feature_names), dtype=float)

    importance = [
        {'name': feature_names[index], 'importance': round(float(value), 4)}
        for index, value in enumerate(values[:len(feature_names)])
    ]
    importance.sort(key=lambda item: item['importance'], reverse=True)
    return importance


def build_time_series_model_recommendations(profile: dict[str, Any], stationarity: dict[str, Any]) -> list[dict[str, Any]]:
    has_seasonality = profile['usable_periods'] >= (12 if profile['detected_frequency'] == 'month' else 8 if profile['detected_frequency'] == 'quarter' else 10)
    return [
        {
            'model_type': 'sarima',
            'model_name': 'SARIMA',
            'recommended': has_seasonality,
            'recommendation_reason': 'Recommended due to detected seasonality.' if has_seasonality else 'Seasonal baseline when recurring cycles are expected.',
        },
        {
            'model_type': 'prophet',
            'model_name': 'Prophet',
            'recommended': not has_seasonality and stationarity['p_value'] >= 0.05,
            'recommendation_reason': 'Recommended when smooth trend movement is more obvious than repeating seasonal blocks.',
        },
        {
            'model_type': 'arima',
            'model_name': 'ARIMA',
            'recommended': stationarity['p_value'] < 0.05,
            'recommendation_reason': 'Useful for shorter or more stable series with weaker seasonal structure.',
        },
    ]


def build_ml_model_recommendations(feature_names: list[str]) -> list[dict[str, Any]]:
    feature_count = len(feature_names)
    return [
        {
            'model_type': 'gradient_boosting',
            'model_name': 'Gradient Boosting',
            'recommended': feature_count >= 6,
            'recommendation_reason': f'Excellent for capturing non-linear patterns across the {feature_count} generated features.',
        },
        {
            'model_type': 'random_forest',
            'model_name': 'Random Forest',
            'recommended': feature_count < 6,
            'recommendation_reason': 'Robust option when feature interactions matter but you want a steady ensemble baseline.',
        },
        {
            'model_type': 'ridge_regression',
            'model_name': 'Ridge Regression',
            'recommended': False,
            'recommendation_reason': 'Simpler baseline when you want linear behavior across generated features.',
        },
    ]


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
def train_model(request: TrainRequest, http_request: Request) -> JSONResponse:
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
    record_activity(
        request=http_request,
        action='train_model',
        status='success',
        dataset_id=request.dataset_id,
        model_id=model_id,
        detail=f'Trained {model_name} for a {request.problem_type} task.',
        metadata={
            'target_column': request.target_column,
            'feature_count': len(request.feature_columns),
            'model_type': request.model_type,
            'training_mode': request.training_mode,
            'primary_metrics': metrics['primary'],
        },
    )
    return JSONResponse(content=safe_serialize(response))


@router.post('/sales-forecast')
def sales_forecast(request: SalesForecastRequest, http_request: Request) -> JSONResponse:
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
    server_session_id = get_session_id(request.dataset_id, request.session_id)
    record_activity(
        request=http_request,
        action='sales_forecast',
        status='success',
        dataset_id=request.dataset_id,
        server_session_id=server_session_id,
        detail=f'Generated {request.forecast_periods} future {period_label} sale forecasts.',
        metadata={
            'date_column': request.date_column,
            'target_column': request.target_column,
            'forecast_periods': request.forecast_periods,
            'lag_periods': effective_lag_periods,
            'metrics': metrics,
        },
    )
    return JSONResponse(content=safe_serialize(response))


@router.post('/forecast/ts/run')
def forecast_time_series(request: TimeSeriesForecastRequest, http_request: Request) -> JSONResponse:
    required_columns = [request.date_column, request.target_column]
    frame = load_dataset_frame(request.dataset_id, request.data, required_columns)
    series_frame, freq, period_label = prepare_sales_series(frame, request.date_column, request.target_column)

    total_periods = len(series_frame)
    effective_test_periods = min(max(1, int(round(total_periods * (request.test_percentage / 100)))), max(1, total_periods - 4))
    train_periods = total_periods - effective_test_periods
    train_series = series_frame.iloc[:train_periods].copy()
    test_series = series_frame.iloc[train_periods:].copy()
    train_values = train_series['sales'].astype(float).tolist()
    test_values = test_series['sales'].astype(float).tolist()

    season_length = infer_season_length(period_label, total_periods)
    stationarity = compute_stationarity_check(series_frame['sales'].astype(float).tolist())
    model_name = resolve_time_series_model_name(request.model_type)
    residual_std = float(np.std(np.diff(train_values))) if len(train_values) > 1 else 0.0
    residual_std = max(residual_std, float(np.std(train_values[-min(4, len(train_values)):])) if train_values else 1.0)

    backtest: list[dict[str, Any]] = []
    running_history = list(train_values)
    current_period = pd.Timestamp(test_series.iloc[0]['period'])
    predicted_test_values: list[float] = []
    for actual in test_values:
        prediction = statistical_forecast_step(running_history, season_length, request.model_type)
        lower, upper = build_confidence_bounds(prediction, residual_std)
        backtest.append({
            'period': format_forecast_period(current_period, period_label),
            'actual': round(float(actual), 2),
            'predicted': round(prediction, 2),
            'lower': lower,
            'upper': upper,
        })
        predicted_test_values.append(round(prediction, 2))
        running_history.append(float(actual))
        current_period = current_period + pd.tseries.frequencies.to_offset(freq)

    future_forecast: list[dict[str, Any]] = []
    future_history = series_frame['sales'].astype(float).tolist()
    current_period = pd.Timestamp(series_frame.iloc[-1]['period']) + pd.tseries.frequencies.to_offset(freq)
    for _ in range(request.forecast_periods):
        prediction = statistical_forecast_step(future_history, season_length, request.model_type)
        lower, upper = build_confidence_bounds(prediction, residual_std)
        future_forecast.append({
            'period': format_forecast_period(current_period, period_label),
            'predicted': round(prediction, 2),
            'lower': lower,
            'upper': upper,
        })
        future_history.append(prediction)
        current_period = current_period + pd.tseries.frequencies.to_offset(freq)

    history = [{'period': format_forecast_period(pd.Timestamp(row['period']), period_label), 'actual': round(float(row['sales']), 2)} for _, row in series_frame.iterrows()]
    metrics = calculate_forecast_metrics(test_values, predicted_test_values)
    profile = build_dataset_profile(series_frame, period_label)
    session_id = get_session_id(request.dataset_id, request.session_id)
    session_state = ensure_session_state(session_id)

    response = {
        'date_column': request.date_column,
        'target_column': request.target_column,
        'frequency': freq,
        'period_label': period_label,
        'dataset_profile': profile,
        'stationarity_check': stationarity,
        'history': history,
        'test_forecast': backtest,
        'future_forecast': future_forecast,
        'metrics': metrics,
        'training_summary': {
            'model_name': model_name,
            'total_periods': total_periods,
            'train_periods': train_periods,
            'test_periods': effective_test_periods,
            'train_percentage': round((train_periods / total_periods) * 100, 1),
            'test_percentage': round((effective_test_periods / total_periods) * 100, 1),
            'forecast_periods': request.forecast_periods,
            'train_start': history[0]['period'],
            'train_end': history[train_periods - 1]['period'],
            'test_start': history[train_periods]['period'],
            'test_end': history[-1]['period'],
            'last_observed_period': history[-1]['period'],
        },
        'recommended_models': build_time_series_model_recommendations(profile, stationarity),
        'model_details': {
            'model_type': request.model_type,
            'model_name': model_name,
            'rationale': f'{model_name} was chosen against a {period_label}-level series with {profile["usable_periods"]} usable periods.',
        },
        'analysis': (
            f'{model_name} forecasted {request.forecast_periods} future {period_label}{"s" if request.forecast_periods != 1 else ""}. '
            f'The series shows {stationarity["verdict"].lower()}, and the backtest produced MAE {metrics["mae"]}, RMSE {metrics["rmse"]}, and MAPE {metrics["mape"]}%.'
        ),
    }

    session_state['forecast_steps']['ts'] = True
    session_state['time_series_result'] = safe_serialize(response)
    session_state['updated_at'] = utc_now_iso()
    record_activity(
        request=http_request,
        action='forecast_time_series',
        status='success',
        dataset_id=request.dataset_id,
        server_session_id=session_id,
        detail=f'Ran {model_name} time-series forecast for {request.forecast_periods} future {period_label} periods.',
        metadata={
            'date_column': request.date_column,
            'target_column': request.target_column,
            'model_type': request.model_type,
            'forecast_periods': request.forecast_periods,
            'metrics': metrics,
        },
    )
    return JSONResponse(content=safe_serialize(response))


@router.post('/forecast/ml/run')
def forecast_ml(request: MlForecastRequest, http_request: Request) -> JSONResponse:
    required_columns = [request.date_column, request.target_column]
    frame = load_dataset_frame(request.dataset_id, request.data, required_columns)
    series_frame, freq, period_label = prepare_sales_series(frame, request.date_column, request.target_column)

    total_periods = len(series_frame)
    effective_test_periods = min(max(1, int(round(total_periods * (request.test_percentage / 100)))), max(1, total_periods - 4))
    train_periods = total_periods - effective_test_periods
    effective_lag_periods = min(request.lag_periods, max(1, train_periods - 1), max(1, total_periods - 2))
    train_series = series_frame.iloc[:train_periods].copy()
    test_series = series_frame.iloc[train_periods:].copy()

    train_X, train_y = build_ml_forecast_training_frame(train_series, effective_lag_periods, request.feature_groups)
    model = build_forecast_regressor(request.model_type)
    model.fit(train_X, train_y)

    test_predictions = recursive_ml_forecast(
        model,
        train_series['sales'].astype(float).tolist(),
        pd.Timestamp(test_series.iloc[0]['period']),
        effective_test_periods,
        effective_lag_periods,
        request.feature_groups,
        freq,
        period_label,
    )
    actual_test_values = test_series['sales'].astype(float).tolist()
    predicted_test_values = [float(item['predicted']) for item in test_predictions]
    metrics = calculate_forecast_metrics(actual_test_values, predicted_test_values)

    full_X, full_y = build_ml_forecast_training_frame(series_frame, effective_lag_periods, request.feature_groups)
    full_model = build_forecast_regressor(request.model_type)
    full_model.fit(full_X, full_y)
    future_predictions = recursive_ml_forecast(
        full_model,
        series_frame['sales'].astype(float).tolist(),
        pd.Timestamp(series_frame.iloc[-1]['period']) + pd.tseries.frequencies.to_offset(freq),
        request.forecast_periods,
        effective_lag_periods,
        request.feature_groups,
        freq,
        period_label,
    )

    history = [{'period': format_forecast_period(pd.Timestamp(row['period']), period_label), 'actual': round(float(row['sales']), 2)} for _, row in series_frame.iterrows()]
    test_results = [
        {
            'period': format_forecast_period(pd.Timestamp(test_series.iloc[index]['period']), period_label),
            'actual': round(float(actual_test_values[index]), 2),
            'predicted': round(float(predicted_test_values[index]), 2),
        }
        for index in range(len(test_predictions))
    ]
    profile = build_dataset_profile(series_frame, period_label)
    importance = calculate_shap_like_importance(full_model, full_X.columns.tolist())
    session_id = get_session_id(request.dataset_id, request.session_id)
    session_state = ensure_session_state(session_id)

    response = {
        'date_column': request.date_column,
        'target_column': request.target_column,
        'frequency': freq,
        'period_label': period_label,
        'dataset_profile': profile,
        'generated_features': full_X.columns.tolist(),
        'feature_preview_rows': safe_serialize(full_X.head(5).round(3).to_dict(orient='records')),
        'history': history,
        'test_forecast': test_results,
        'future_forecast': future_predictions,
        'metrics': metrics,
        'training_summary': {
            'model_name': 'Gradient Boosting' if request.model_type == 'gradient_boosting' else 'Random Forest' if request.model_type == 'random_forest' else 'Ridge Regression',
            'total_periods': total_periods,
            'train_periods': train_periods,
            'test_periods': effective_test_periods,
            'train_percentage': round((train_periods / total_periods) * 100, 1),
            'test_percentage': round((effective_test_periods / total_periods) * 100, 1),
            'forecast_periods': request.forecast_periods,
            'lag_periods': effective_lag_periods,
            'train_start': history[0]['period'],
            'train_end': history[train_periods - 1]['period'],
            'test_start': history[train_periods]['period'],
            'test_end': history[-1]['period'],
            'last_observed_period': history[-1]['period'],
        },
        'shap_feature_importance': importance,
        'recommended_models': build_ml_model_recommendations(full_X.columns.tolist()),
        'model_details': {
            'model_type': request.model_type,
            'model_name': 'Gradient Boosting' if request.model_type == 'gradient_boosting' else 'Random Forest' if request.model_type == 'random_forest' else 'Ridge Regression',
            'rationale': f'The model learned across {len(full_X.columns)} generated forecast features created from time-derived signals.',
        },
        'analysis': (
            f'ML forecasting converted time into {len(full_X.columns)} generated features and trained a {request.model_type.replace("_", " ")} model. '
            f'The strongest drivers were {", ".join(item["name"] for item in importance[:3]) or "the engineered feature set"}, '
            f'with backtest MAE {metrics["mae"]}, RMSE {metrics["rmse"]}, and MAPE {metrics["mape"]}%.'
        ),
    }

    session_state['forecast_steps']['ml'] = True
    session_state['ml_forecast_result'] = safe_serialize(response)
    session_state['updated_at'] = utc_now_iso()
    record_activity(
        request=http_request,
        action='forecast_ml',
        status='success',
        dataset_id=request.dataset_id,
        server_session_id=session_id,
        detail=f'Ran ML forecasting with {request.model_type} over {request.forecast_periods} future {period_label} periods.',
        metadata={
            'date_column': request.date_column,
            'target_column': request.target_column,
            'forecast_periods': request.forecast_periods,
            'lag_periods': effective_lag_periods,
            'feature_groups': request.feature_groups,
            'metrics': metrics,
        },
    )
    return JSONResponse(content=safe_serialize(response))


@router.post('/predict')
def predict(request: PredictRequest, http_request: Request) -> JSONResponse:
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

        record_activity(
            request=http_request,
            action='predict',
            status='success',
            model_id=request.model_id,
            detail='Generated a prediction from a trained model.',
            metadata={
                'feature_count': len(request.features),
                'prediction': payload.get('prediction_label', payload.get('prediction')),
                'confidence': payload.get('confidence'),
            },
        )
        return JSONResponse(content=safe_serialize(payload))
    except HTTPException:
        raise
    except Exception as error:
        logger.exception('Prediction failed model_id=%s', request.model_id)
        raise HTTPException(status_code=400, detail=f'Prediction failed: {error}') from error


@router.post('/upload-model')
async def upload_model(http_request: Request, file: UploadFile = File(...)) -> JSONResponse:
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
    record_activity(
        request=http_request,
        action='upload_model',
        status='success',
        model_id=model_id,
        file_name=filename,
        detail=f'Uploaded model bundle {model_bundle["model_name"]}.',
        metadata={
            'model_type': model_bundle['model_type'],
            'problem_type': model_bundle['problem_type'],
            'feature_count': len(model_bundle['feature_columns']),
        },
    )
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


def clean_cached_dataset(request: ParquetCleaningRequest) -> dict[str, Any]:
    dataset_entry = DATASET_CACHE.get(request.dataset_id)
    if dataset_entry is None:
        raise HTTPException(status_code=400, detail='Cached dataset not found. Please upload the file again.')

    if dataset_entry.get('frame_path') or dataset_entry.get('csv_path') or dataset_entry.get('excel_path'):
        if dataset_entry.get('frame_path'):
            frame = read_cached_frame(dataset_entry)
        elif dataset_entry.get('csv_path'):
            frame = read_cached_csv(dataset_entry)
        else:
            frame = read_cached_excel(dataset_entry)
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
        duplicate_rows = int(max(0, len(frame) - len(frame.drop_duplicates())))
        updated_entry = {
            'frame_path': str(updated_dataset_path),
            'filename': dataset_entry['filename'],
            'row_count': int(len(frame)),
            'column_count': int(len(frame.columns)),
            'columns': list(frame.columns),
            'duplicate_count': duplicate_rows,
        }
        if dataset_entry.get('csv_path'):
            updated_entry['csv_path'] = dataset_entry['csv_path']
        if dataset_entry.get('excel_path'):
            updated_entry['excel_path'] = dataset_entry['excel_path']
        if dataset_entry.get('parquet_path'):
            updated_entry['parquet_path'] = dataset_entry['parquet_path']
        DATASET_CACHE[request.dataset_id] = updated_entry
        memory_size = updated_dataset_path.stat().st_size
        preview_frame = frame.head(PARQUET_PREVIEW_ROW_LIMIT)
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
    duplicate_rows = int(max(0, frame.height - frame.unique().height))
    DATASET_CACHE[request.dataset_id] = {
        'parquet_path': str(updated_dataset_path),
        'filename': dataset_entry['filename'],
        'row_count': int(frame.height),
        'column_count': int(len(frame.columns)),
        'columns': list(frame.columns),
        'duplicate_count': duplicate_rows,
    }
    memory_size = updated_dataset_path.stat().st_size

    preview_frame = frame.head(PARQUET_PREVIEW_ROW_LIMIT)
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
    dataset_label = request.fileName or 'uploaded dataset'
    loaded_rows = request.loadedRowCount or request.totalRows
    scope_line = (
        f"The dataset was uploaded as '{dataset_label}'. A preview of {loaded_rows} rows is currently rendered while cleaning decisions are being applied to the full {request.totalRows}-row dataset."
        if request.previewLoaded and request.totalRows > loaded_rows
        else f"The dataset was uploaded as '{dataset_label}' with {request.totalRows} rows available for direct cleaning review."
    )
    summary_lines = [
        scope_line,
        f"It contains {request.totalColumns} columns, so the cleaning workflow focuses on changes that improve reliability without assuming any specific business domain.",
        'The following cleaning steps were applied to improve data quality:',
    ]
    for log in request.logs:
        summary_lines.append(f"- {log.action}: {log.detail}")
    summary_lines.append('These changes make the uploaded dataset more consistent for EDA, forecasting, machine learning training, and downstream prediction without hard-coding dataset-specific rules.')
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


def build_line_chart_image(
    title: str,
    history: list[dict[str, Any]],
    test_forecast: list[dict[str, Any]],
    future_forecast: list[dict[str, Any]],
    include_interval: bool = False,
) -> Image:
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    all_periods = [item['period'] for item in history]
    all_periods.extend(item['period'] for item in future_forecast if item['period'] not in all_periods)
    x_lookup = {period: index for index, period in enumerate(all_periods)}
    history_periods = [x_lookup[item['period']] for item in history]
    history_values = [float(item.get('actual', 0) or 0) for item in history]
    ax.plot(history_periods, history_values, label='Actual', color='#0f766e', linewidth=2)

    if test_forecast:
      ax.plot([x_lookup[item['period']] for item in test_forecast], [float(item.get('predicted', 0) or 0) for item in test_forecast], label='Backtest', color='#f59e0b', linestyle='--', linewidth=2)
    if future_forecast:
      periods = [x_lookup[item['period']] for item in future_forecast]
      values = [float(item.get('predicted', 0) or 0) for item in future_forecast]
      ax.plot(periods, values, label='Forecast', color='#2563eb', linewidth=2)
      if include_interval:
          lowers = [float(item.get('lower', item.get('predicted', 0)) or 0) for item in future_forecast]
          uppers = [float(item.get('upper', item.get('predicted', 0)) or 0) for item in future_forecast]
          ax.fill_between(periods, lowers, uppers, color='#93c5fd', alpha=0.3, label='95% interval')

    ax.set_title(title)
    ax.set_xticks(list(x_lookup.values()))
    ax.set_xticklabels(all_periods, rotation=35, fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    image_buffer.seek(0)
    return Image(image_buffer, width=480, height=220)


def build_bar_chart_image(title: str, items: list[dict[str, Any]]) -> Image:
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    trimmed = items[:10]
    names = [str(item.get('name', 'Feature')) for item in trimmed][::-1]
    values = [float(item.get('importance', 0) or 0) for item in trimmed][::-1]
    ax.barh(names, values, color='#0f766e')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.2)
    ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout()
    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    image_buffer.seek(0)
    return Image(image_buffer, width=480, height=220)


def build_correlation_chart_image(correlations: list[dict[str, Any]]) -> Image | None:
    if not correlations:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    trimmed = correlations[:8][::-1]
    names = [str(item.get('pair', 'Pair')) for item in trimmed]
    values = [float(item.get('correlation', 0) or 0) for item in trimmed]
    colors_list = ['#0f766e' if value >= 0 else '#dc2626' for value in values]
    ax.barh(names, values, color=colors_list)
    ax.set_title('EDA Correlation Heatmap Summary')
    ax.grid(axis='x', alpha=0.2)
    ax.tick_params(axis='y', labelsize=8)
    fig.tight_layout()
    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    image_buffer.seek(0)
    return Image(image_buffer, width=480, height=220)


def build_image_from_base64(data_uri: str | None, *, max_width: float = 480, max_height: float = 260) -> Image | None:
    if not data_uri:
        return None

    try:
        encoded = data_uri.split(',', 1)[1] if ',' in data_uri else data_uri
        image_bytes = base64.b64decode(encoded)
        image_buffer = io.BytesIO(image_bytes)
        image = Image(image_buffer)
        image.drawWidth = max_width
        image.drawHeight = max_height
        return image
    except Exception:
        logger.exception('Failed to decode base64 image for EDA PDF report.')
        return None


def build_eda_pdf(payload: EdaPdfPayload) -> bytes:
    loaded_row_count = payload.loadedRowCount or payload.totalRows
    preview_mode = payload.previewLoaded and payload.totalRows > loaded_row_count
    advanced = payload.advancedAnalysis or {}

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), leftMargin=30, rightMargin=30, topMargin=26, bottomMargin=24)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('EDA_Title', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=22, leading=26, textColor=colors.HexColor('#0f172a'), spaceAfter=8)
    heading_style = ParagraphStyle('EDA_Heading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=15, leading=18, textColor=colors.HexColor('#0f172a'), spaceAfter=6)
    body_style = ParagraphStyle('EDA_Body', parent=styles['BodyText'], fontName='Helvetica', fontSize=9.2, leading=13, textColor=colors.HexColor('#334155'))
    small_style = ParagraphStyle('EDA_Small', parent=body_style, fontSize=8.2, leading=11, textColor=colors.HexColor('#64748b'))
    label_style = ParagraphStyle('EDA_Label', parent=body_style, fontName='Helvetica-Bold', fontSize=8.2, leading=10, textColor=colors.HexColor('#0f766e'))
    value_style = ParagraphStyle('EDA_Value', parent=body_style, fontName='Helvetica-Bold', fontSize=13, leading=16, textColor=colors.HexColor('#0f172a'))
    elements: list[Any] = []
    page_width = landscape(letter)[0]
    content_width = page_width - 60

    def paragraph(text: Any, style: ParagraphStyle = body_style) -> Paragraph:
        return Paragraph(str(text).replace('\n', '<br/>'), style)

    def add_table(rows: list[list[Any]], widths: list[float], header_bg: str = '#0f766e') -> None:
        normalized = []
        for row_index, row in enumerate(rows):
            row_style = label_style if row_index == 0 else body_style
            normalized.append([paragraph(cell, row_style) for cell in row])
        table = Table(normalized, colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fbff')]),
            ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#dbe4f0')),
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
            card = Table([[paragraph(label, label_style)], [paragraph(value, value_style)]], colWidths=[content_width / max(1, len(cards)) - 8])
            card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fbff')),
                ('BOX', (0, 0), (-1, -1), 0.65, colors.HexColor('#d6e3f1')),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            row.append(card)
            widths.append(content_width / max(1, len(cards)))
        wrapper = Table([row], colWidths=widths)
        wrapper.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
        elements.append(wrapper)

    def add_section(title: str, blurb: str) -> None:
        elements.append(paragraph(title, heading_style))
        elements.append(paragraph(blurb, body_style))
        elements.append(Spacer(1, 8))

    def add_chart_section(title: str, description: str, chart_items: list[tuple[str, str | None]], *, subtitle: str | None = None) -> None:
        add_section(title, description)
        if subtitle:
            elements.append(paragraph(subtitle, small_style))
            elements.append(Spacer(1, 6))
        rendered_any = False
        for chart_title, chart_base64 in chart_items[:4]:
            image = build_image_from_base64(chart_base64)
            elements.append(paragraph(chart_title, label_style))
            if image is not None:
                elements.append(image)
                rendered_any = True
            else:
                elements.append(paragraph('No chart available for this item.', small_style))
            elements.append(Spacer(1, 8))
        if not rendered_any and not chart_items:
            elements.append(paragraph('No chart outputs were available for this section.', small_style))

    def decorate_page(canvas: Any, doc_obj: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#dbe4f0'))
        canvas.line(doc.leftMargin, 18, page_width - doc.rightMargin, 18)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(doc.leftMargin, 8, f'EDA PDF | {payload.fileName}')
        canvas.drawRightString(page_width - doc.rightMargin, 8, f'Page {canvas.getPageNumber()}')
        canvas.restoreState()

    elements.append(paragraph('Exploratory Data Analysis PDF', title_style))
    elements.append(paragraph(
        f'This export captures the EDA tab functionality, working flow, descriptive statistics, relationship analysis, and advanced analytical features for {payload.fileName}.',
        body_style,
    ))
    elements.append(Spacer(1, 10))
    add_stat_cards([
        ('Total Rows', f'{payload.totalRows:,}'),
        ('Rows In Workspace', f'{loaded_row_count:,}'),
        ('Columns', len(payload.columns)),
        ('Numeric Fields', len(payload.edaStats.numericColumns)),
        ('Categorical Fields', len(payload.edaStats.categoricalColumns)),
    ])
    elements.append(Spacer(1, 8))
    elements.append(paragraph(
        f'EDA working mode: {"Preview-backed browser analysis with cached backend dataset support." if preview_mode else "Direct workspace analysis across the full loaded dataset."}',
        small_style,
    ))
    elements.append(Spacer(1, 12))

    add_section('EDA Tab Functional Coverage', 'This PDF mirrors the EDA tab itself: schema review, numeric profiling, correlation discovery, advanced charts, and automated statistical recommendations.')
    add_table([
        ['Feature Area', 'What The EDA Tab Does'],
        ['Dataset Schema', 'Profiles column type, completeness, uniqueness, and inferred role for each field.'],
        ['Statistical Summary', 'Computes count, mean, spread, quartiles, and extrema for numeric columns.'],
        ['Relationships', 'Highlights the strongest positive and negative numeric correlations.'],
        ['Correlation Heatmap', 'Shows matrix-style correlation strength across the leading numeric fields.'],
        ['Advanced Modules', 'Extends the base EDA with missingness, distributions, categorical analysis, interactions, and automated insights.'],
    ], [content_width * 0.24, content_width * 0.72], header_bg='#115e59')
    elements.append(PageBreak())

    add_section('Dataset Schema', 'The EDA tab begins by establishing the structure and quality envelope of the active dataset.')
    schema_rows = [['Column', 'Type', 'Non-Null', 'Missing', 'Unique', 'Role']]
    for column in payload.columns[:24]:
        schema_rows.append([column.name, column.dtype, column.nonNull, column.nullCount, column.uniqueCount, column.role])
    add_table(schema_rows, [content_width * 0.28, content_width * 0.12, content_width * 0.12, content_width * 0.12, content_width * 0.12, content_width * 0.14], header_bg='#115e59')
    if len(payload.columns) > 24:
        elements.append(Spacer(1, 6))
        elements.append(paragraph(f'Showing the first 24 columns out of {len(payload.columns)} total profiled columns.', small_style))
    elements.append(Spacer(1, 10))

    add_section('Statistical Summary', 'Numeric fields are summarized to expose central tendency, spread, and range before cleaning or modeling.')
    numeric_rows = [['Field', 'Mean', 'Std', 'Min', 'Median', 'Max']]
    for field_name in payload.edaStats.numericColumns[:12]:
        stats = payload.edaStats.stats.get(field_name, {})
        numeric_rows.append([
            field_name,
            stats.get('mean', 'N/A'),
            stats.get('std', 'N/A'),
            stats.get('min', 'N/A'),
            stats.get('median', 'N/A'),
            stats.get('max', 'N/A'),
        ])
    add_table(numeric_rows if len(numeric_rows) > 1 else [['Field', 'Mean', 'Std', 'Min', 'Median', 'Max'], ['N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']], [content_width * 0.34, content_width * 0.11, content_width * 0.11, content_width * 0.11, content_width * 0.11, content_width * 0.11], header_bg='#115e59')
    elements.append(PageBreak())

    add_section('Relationships and Correlation Working', 'The EDA tab surfaces the strongest numeric relationships so users can quickly assess signal, redundancy, and interaction behavior.')
    corr_image = build_correlation_chart_image(payload.edaStats.correlations)
    if corr_image is not None:
        elements.append(corr_image)
        elements.append(Spacer(1, 8))
    correlation_rows = [['Pair', 'Correlation']]
    for item in payload.edaStats.correlations[:10]:
        correlation_rows.append([item.get('pair', 'N/A'), item.get('correlation', 'N/A')])
    add_table(correlation_rows if len(correlation_rows) > 1 else [['Pair', 'Correlation'], ['N/A', 'N/A']], [content_width * 0.78, content_width * 0.18], header_bg='#115e59')

    insights = ((advanced.get('insights') or {}).get('insights') if isinstance(advanced.get('insights'), dict) else None) or []
    if insights:
        elements.append(Spacer(1, 10))
        add_section('Automated Insights', 'The advanced EDA layer translates statistical anomalies into plain-language recommendations.')
        insight_rows = [['Insight']]
        for item in insights[:10]:
            insight_rows.append([item])
        add_table(insight_rows, [content_width * 0.96], header_bg='#115e59')
    elements.append(PageBreak())

    missingness = advanced.get('missingness') if isinstance(advanced.get('missingness'), dict) else {}
    distributions = advanced.get('distributions') if isinstance(advanced.get('distributions'), dict) else {}
    categorical = advanced.get('categorical') if isinstance(advanced.get('categorical'), dict) else {}
    interactions = advanced.get('interactions') if isinstance(advanced.get('interactions'), dict) else {}

    add_chart_section(
        'Advanced EDA: Data Quality and Missingness',
        'This section documents how the advanced EDA tab checks missing-value concentration and dataset completeness behavior.',
        [('Missingness Intensity Map', missingness.get('chart_base64'))] if missingness else [],
        subtitle=str(missingness.get('message') or '') if missingness else None,
    )
    add_chart_section(
        'Advanced EDA: Distributions and Outliers',
        'These charts show how the EDA tab evaluates numeric spread, skew, and potential outliers.',
        [(str(item.get('column', 'Distribution')), item.get('chart_base64')) for item in (distributions.get('charts') or []) if isinstance(item, dict)],
        subtitle=str(distributions.get('message') or '') if distributions else None,
    )
    add_chart_section(
        'Advanced EDA: Categorical Features',
        'These plots document top-category behavior and warn about high-cardinality features that may affect ML readiness.',
        [(f"{item.get('column', 'Category')} ({item.get('unique_count', 'N/A')} unique)", item.get('chart_base64')) for item in (categorical.get('charts') or []) if isinstance(item, dict)],
        subtitle=str(categorical.get('message') or '') if categorical else None,
    )
    add_chart_section(
        'Advanced EDA: Key Variable Interactions',
        'These interaction views show the strongest numeric pairings explored by the advanced EDA feature set.',
        [(str(item.get('pair', 'Interaction')), item.get('chart_base64')) for item in (interactions.get('plots') or []) if isinstance(item, dict)],
        subtitle=str(interactions.get('message') or '') if interactions else None,
    )

    doc.build(elements, onFirstPage=decorate_page, onLaterPages=decorate_page)
    return buffer.getvalue()


def build_dynamic_report_pdf(payload: ReportPayload) -> bytes:
    session_id = get_session_id(payload.datasetId, payload.sessionId)
    session_state = ensure_session_state(session_id)
    completed_steps = set(payload.forecastingStepsCompleted)
    if session_state['forecast_steps'].get('ts'):
        completed_steps.add(5)
    if session_state['forecast_steps'].get('ml'):
        completed_steps.add(6)

    ts_result_raw = payload.timeSeriesForecastResult or session_state.get('time_series_result')
    ml_result_raw = payload.mlForecastResult or session_state.get('ml_forecast_result')
    ts_result = ts_result_raw.model_dump() if hasattr(ts_result_raw, 'model_dump') else ts_result_raw
    ml_forecast_result = ml_result_raw.model_dump() if hasattr(ml_result_raw, 'model_dump') else ml_result_raw

    page_size = landscape(letter)
    page_width, page_height = page_size
    content_width = page_width - 64
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=page_size, leftMargin=32, rightMargin=32, topMargin=28, bottomMargin=24)
    styles = getSampleStyleSheet()
    eyebrow_style = ParagraphStyle('IDA_Eyebrow', parent=styles['BodyText'], fontName='Helvetica-Bold', fontSize=9, leading=11, textColor=colors.HexColor('#a5f3fc'), spaceAfter=4)
    hero_title_style = ParagraphStyle('IDA_HeroTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=25, leading=29, textColor=colors.white, spaceAfter=8)
    hero_subtitle_style = ParagraphStyle('IDA_HeroSubtitle', parent=styles['BodyText'], fontName='Helvetica', fontSize=10.5, leading=15, textColor=colors.HexColor('#e0f2fe'))
    title_style = ParagraphStyle('IDA_Title', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=22, leading=25, textColor=colors.HexColor('#0f172a'), spaceAfter=8)
    heading_style = ParagraphStyle('IDA_Heading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=16, leading=19, textColor=colors.HexColor('#0f172a'), spaceAfter=6)
    body_style = ParagraphStyle('IDA_Body', parent=styles['BodyText'], fontName='Helvetica', fontSize=9.2, leading=13.5, textColor=colors.HexColor('#334155'))
    small_style = ParagraphStyle('IDA_Small', parent=body_style, fontSize=8.1, leading=11, textColor=colors.HexColor('#64748b'))
    label_style = ParagraphStyle('IDA_Label', parent=body_style, fontName='Helvetica-Bold', fontSize=8.2, leading=10, textColor=colors.HexColor('#0369a1'))
    value_style = ParagraphStyle('IDA_Value', parent=body_style, fontName='Helvetica-Bold', fontSize=14, leading=17, textColor=colors.HexColor('#0f172a'))
    section_label_style = ParagraphStyle('IDA_SectionLabel', parent=styles['BodyText'], fontName='Helvetica-Bold', fontSize=8.5, leading=10, textColor=colors.HexColor('#0284c7'))
    section_blurb_style = ParagraphStyle('IDA_SectionBlurb', parent=body_style, fontSize=9.4, leading=14, textColor=colors.HexColor('#475569'))
    elements: list[Any] = []

    def as_paragraph(text: Any, style: ParagraphStyle = body_style) -> Paragraph:
        return Paragraph(str(text).replace('\n', '<br/>'), style)

    def add_paragraph(text: Any, style: ParagraphStyle = body_style) -> None:
        elements.append(as_paragraph(text, style))

    def add_table(rows: list[list[Any]], widths: list[int], header_bg: str = '#0f766e') -> None:
        normalized: list[list[Any]] = []
        for row_index, row in enumerate(rows):
            style = label_style if row_index == 0 else body_style
            normalized.append([as_paragraph(cell, style) for cell in row])
        table = Table(normalized, colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_bg)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fbff')]),
            ('GRID', (0, 0), (-1, -1), 0.35, colors.HexColor('#dbe4f0')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 7),
            ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(table)

    def add_stat_cards(cards: list[tuple[str, Any]]) -> None:
        row = []
        widths = []
        for label, value in cards:
            card = Table([[as_paragraph(label, label_style)], [as_paragraph(value, value_style)]], colWidths=[content_width / max(1, len(cards)) - 8])
            card.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fbff')),
                ('BOX', (0, 0), (-1, -1), 0.65, colors.HexColor('#d6e3f1')),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            row.append(card)
            widths.append(content_width / max(1, len(cards)))
        wrapper = Table([row], colWidths=widths)
        wrapper.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
        elements.append(wrapper)

    def add_section(title: str, blurb: str) -> None:
        section_card = Table([[
            Paragraph('WORKFLOW SECTION', section_label_style),
            Paragraph(title, heading_style),
            Paragraph(blurb, section_blurb_style),
        ]], colWidths=[content_width])
        section_card.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
            ('BOX', (0, 0), (-1, -1), 0.65, colors.HexColor('#dbe4f0')),
            ('LEFTPADDING', (0, 0), (-1, -1), 14),
            ('RIGHTPADDING', (0, 0), (-1, -1), 14),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(section_card)
        elements.append(Spacer(1, 8))

    def add_callout(title: str, text: str, tone: str = '#eff6ff', border: str = '#93c5fd') -> None:
        callout = Table([[
            Paragraph(f'<b>{title}</b><br/>{text}', body_style)
        ]], colWidths=[content_width])
        callout.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(tone)),
            ('BOX', (0, 0), (-1, -1), 0.7, colors.HexColor(border)),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
        ]))
        elements.append(callout)

    def role_count(role_name: str) -> int:
        return sum(1 for column in payload.columns if str(column.role).lower() == role_name)

    def metric_text(value: Any) -> str:
        if value is None:
            return 'N/A'
        if isinstance(value, float):
            return f'{value:,.3f}'
        return str(value)

    loaded_row_count = payload.loadedRowCount or payload.totalRows
    preview_mode = payload.previewLoaded and payload.totalRows > loaded_row_count
    workspace_scope = (
        f'{loaded_row_count:,} preview rows were rendered in the browser while the full {payload.totalRows:,}-row dataset remained cached on the backend.'
        if preview_mode
        else f'The full {payload.totalRows:,}-row dataset was loaded directly into the active workspace.'
    )

    def latest_prediction_feature_summary() -> list[list[Any]] | None:
        if not payload.predictionHistory:
            return None
        latest = payload.predictionHistory[-1]
        features = latest.features or {}
        if not features:
            return None
        rows = [['Feature', 'Latest Scored Value']]
        for key, value in list(features.items())[:12]:
            rows.append([key, value])
        return rows

    def decorate_page(canvas: Any, doc_obj: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#dbe4f0'))
        canvas.line(doc.leftMargin, 20, page_width - doc.rightMargin, 20)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(doc.leftMargin, 10, f'Intelligent Data Assistant | {payload.fileName}')
        canvas.drawRightString(page_width - doc.rightMargin, 10, f'Page {canvas.getPageNumber()}')
        canvas.restoreState()

    workflow_rows = [
        ['Step', 'Tab', 'Status', 'Coverage'],
        ['1', 'Upload', 'Completed' if payload.totalRows > 0 else 'Pending', f'{payload.fileName} with {payload.totalRows:,} rows, {len(payload.columns)} columns, and {"preview-backed caching" if preview_mode else "full workspace loading"}'],
        ['2', 'Understanding', 'Completed' if payload.columns else 'Pending', f'Role inference, null counts, unique counts, and schema profiling across {len(payload.columns)} columns'],
        ['3', 'EDA', 'Completed' if payload.columns else 'Pending', f'{len(payload.edaStats.numericColumns)} numeric and {len(payload.edaStats.categoricalColumns)} categorical columns summarized with {len(payload.edaStats.correlations)} sampled correlation signals'],
        ['4', 'Cleaning', 'Completed' if payload.cleaningDone else 'Pending', f'{len(payload.cleaningLogs)} logged operations and {payload.cleanedRowCount:,} cleaned rows retained'],
        ['5', 'Forecast TS', 'Completed' if ts_result else 'Skipped', 'Time-driven forecasting, backtest metrics, horizon outputs, and interval-aware charting'],
        ['6', 'Forecast ML', 'Completed' if ml_forecast_result else 'Skipped', 'Feature-engineered forecasting, SHAP importance, generated features, and projected horizon'],
        ['7', 'ML Assistant', 'Completed' if payload.modelMetrics else 'Pending', f'Model selection, target setup, {len(payload.selectedFeatures)} features, and training metrics'],
        ['8', 'Prediction', 'Completed' if payload.predictionResult is not None else 'Pending', f'{len(payload.predictionHistory)} prediction history entries and latest scoring output'],
    ]
    workflow_status = f"{sum(1 for row in workflow_rows[1:] if row[2] == 'Completed')}/8 core tabs completed"
    forecast_status = ', '.join(name for name, present in [('TS', bool(ts_result)), ('ML', bool(ml_forecast_result))] if present) or 'None'

    generated_on = datetime.now().strftime('%d %b %Y, %I:%M %p')
    cover_card = Table([[
        Paragraph('INTELLIGENT DATA ASSISTANT', eyebrow_style),
        Paragraph('Executive Workflow Report', hero_title_style),
        Paragraph(
            'A presentation-ready summary of the end-to-end analytics journey, designed for stakeholder review, project handoff, and decision-making.',
            hero_subtitle_style,
        ),
        Spacer(1, 6),
        Table([
            [Paragraph('Dataset', label_style), Paragraph('Generated', label_style), Paragraph('Workflow Status', label_style), Paragraph('Forecast Paths', label_style)],
            [Paragraph(payload.fileName, value_style), Paragraph(generated_on, value_style), Paragraph(workflow_status, value_style), Paragraph(forecast_status, value_style)],
        ], colWidths=[content_width * 0.28, content_width * 0.22, content_width * 0.25, content_width * 0.17]),
    ]], colWidths=[content_width])
    cover_card.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#0f766e')),
        ('BOX', (0, 0), (-1, -1), 0.9, colors.HexColor('#14b8a6')),
        ('LEFTPADDING', (0, 0), (-1, -1), 18),
        ('RIGHTPADDING', (0, 0), (-1, -1), 18),
        ('TOPPADDING', (0, 0), (-1, -1), 18),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 18),
    ]))
    elements.append(cover_card)
    elements.append(Spacer(1, 12))
    add_stat_cards([
        ('Rows', f'{payload.totalRows:,}'),
        ('Columns', len(payload.columns)),
        ('Cleaned Rows', f'{payload.cleanedRowCount:,}'),
        ('Workflow', workflow_status),
    ])
    elements.append(Spacer(1, 8))
    add_callout(
        'Executive Summary',
        (
            f'The dataset entered the platform as {payload.fileName}. '
            f'{len(payload.edaStats.numericColumns)} numeric fields and {len(payload.edaStats.categoricalColumns)} categorical fields were profiled during EDA, '
            f'{len(payload.cleaningLogs)} cleaning actions were recorded, '
            f'and the latest workflow outcome is {payload.predictionResult if payload.predictionResult is not None else "still pending prediction"}.'
        ),
        tone='#ecfeff',
        border='#67e8f9',
    )
    elements.append(Spacer(1, 8))
    add_table([
        ['Metric', 'Value'],
        ['Dataset File', payload.fileName],
        ['Workspace Scope', workspace_scope],
        ['Duplicates Detected', payload.duplicates],
        ['Estimated Memory', payload.memoryUsage],
        ['Forecast Paths Run', forecast_status],
        ['Prediction Available', 'Yes' if payload.predictionResult is not None else 'No'],
    ], [220, 280], header_bg='#115e59')
    elements.append(PageBreak())

    add_section('Workflow Coverage Map', 'The report follows the same tab sequence used in the application so the exported file reads like a presentation replay of the full workflow.')
    add_table(workflow_rows, [35, 110, 90, content_width - 235], header_bg='#0f766e')
    elements.append(Spacer(1, 10))

    add_section('Tab 1: Data Upload', 'The upload stage establishes dataset identity, scale, and storage footprint before any cleaning or modeling decisions are made.')
    add_stat_cards([
        ('Dataset ID', payload.datasetId or 'Session only'),
        ('Rows Loaded', f'{payload.totalRows:,}'),
        ('Browser Rows', f'{loaded_row_count:,}'),
        ('Columns Found', len(payload.columns)),
        ('Workspace Mode', 'Preview + cached backend' if preview_mode else 'Full in-browser dataset'),
    ])
    elements.append(Spacer(1, 8))
    add_paragraph(
        f'The uploaded dataset entered the application as {payload.fileName}. {workspace_scope} This report is designed to remain dataset-agnostic, so downstream sections rely on detected roles, computed statistics, and executed workflow steps rather than hard-coded assumptions about specific fields.'
    )
    elements.append(Spacer(1, 10))

    add_section('Tab 2: Data Understanding', 'Profiling converts raw columns into usable metadata by estimating types, inferring roles, and quantifying completeness before transformation.')
    add_stat_cards([
        ('Numeric', role_count('numeric')),
        ('Categorical', role_count('categorical')),
        ('Datetime', role_count('datetime')),
        ('Identifiers', role_count('identifier')),
    ])
    elements.append(Spacer(1, 8))
    understanding_rows = [['Column', 'Type', 'Role', 'Non-null', 'Nulls', 'Unique']]
    for column in payload.columns[:20]:
        understanding_rows.append([column.name, column.dtype, column.role, column.nonNull, column.nullCount, column.uniqueCount])
    add_table(understanding_rows, [content_width * 0.28, content_width * 0.12, content_width * 0.12, content_width * 0.12, content_width * 0.11, content_width * 0.11], header_bg='#134e4a')
    if len(payload.columns) > 20:
        add_paragraph(f'Showing the first 20 columns out of {len(payload.columns)} profiled columns.', small_style)
    elements.append(PageBreak())

    add_section('Tab 3: Exploratory Data Analysis', 'EDA summarizes the dataset structure, descriptive behavior, and strongest relationships so later cleaning and modeling choices have context.')
    add_stat_cards([
        ('Numeric Fields', len(payload.edaStats.numericColumns)),
        ('Categorical Fields', len(payload.edaStats.categoricalColumns)),
        ('Correlations', len(payload.edaStats.correlations)),
        ('AI Insight', 'Captured' if payload.aiInsights else 'Not captured'),
    ])
    elements.append(Spacer(1, 8))
    corr_image = build_correlation_chart_image(payload.edaStats.correlations)
    if corr_image is not None:
        elements.append(corr_image)
        elements.append(Spacer(1, 8))
    numeric_rows = [['Field', 'Mean', 'Std', 'Min', 'Median', 'Max']]
    for field_name in payload.edaStats.numericColumns[:10]:
        stats = payload.edaStats.stats.get(field_name, {})
        numeric_rows.append([
            field_name,
            metric_text(stats.get('mean')),
            metric_text(stats.get('std')),
            metric_text(stats.get('min')),
            metric_text(stats.get('median')),
            metric_text(stats.get('max')),
        ])
    add_table(numeric_rows if len(numeric_rows) > 1 else [['Field', 'Mean', 'Std', 'Min', 'Median', 'Max'], ['N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']], [content_width * 0.3, content_width * 0.11, content_width * 0.11, content_width * 0.11, content_width * 0.11, content_width * 0.11], header_bg='#115e59')
    if payload.edaStats.correlations:
        elements.append(Spacer(1, 8))
        correlation_rows = [['Pair', 'Correlation']]
        for item in payload.edaStats.correlations[:8]:
            correlation_rows.append([item.get('pair', 'N/A'), item.get('correlation', 'N/A')])
        add_table(correlation_rows, [content_width * 0.78, content_width * 0.18], header_bg='#115e59')
    if payload.aiInsights:
        elements.append(Spacer(1, 8))
        add_callout('AI Insight Summary', str(payload.aiInsights), tone='#eff6ff', border='#93c5fd')
    elements.append(PageBreak())

    add_section('Tab 4: Data Cleaning', 'Cleaning follows exploratory analysis and prepares the stable analysis layer used by forecasting, ML training, and final prediction.')
    add_stat_cards([
        ('Cleaning Done', 'Yes' if payload.cleaningDone else 'No'),
        ('Logged Actions', len(payload.cleaningLogs)),
        ('Rows Removed', max(0, payload.totalRows - payload.cleanedRowCount)),
        ('Rows Retained', f'{payload.cleanedRowCount:,}'),
    ])
    elements.append(Spacer(1, 8))
    if payload.cleaningLogs:
        cleaning_rows = [['Action', 'Detail', 'Timestamp']]
        for log in payload.cleaningLogs[:24]:
            cleaning_rows.append([log.action, log.detail, log.timestamp])
        add_table(cleaning_rows, [content_width * 0.18, content_width * 0.57, content_width * 0.18], header_bg='#0f766e')
    else:
        add_paragraph('No cleaning logs were captured for this run. The report still remains valid and summarizes the workflow with the currently available state.', small_style)

    if ts_result:
        elements.append(PageBreak())
        add_section('Tab 5: Forecast TS', 'The time-series forecasting tab models chronology directly and is appropriate when the temporal sequence itself carries the predictive signal.')
        ts_training = ts_result.get('training_summary', {}) or {}
        ts_metrics = ts_result.get('metrics', {}) or {}
        ts_profile = ts_result.get('dataset_profile', {}) or {}
        stationarity = ts_result.get('stationarity_check', {}) or {}
        add_stat_cards([
            ('Model', ts_training.get('model_name', 'N/A')),
            ('Train/Test', f"{ts_training.get('train_percentage', 'N/A')}% / {ts_training.get('test_percentage', 'N/A')}%"),
            ('Horizon', ts_training.get('forecast_periods', 'N/A')),
            ('Frequency', ts_result.get('period_label') or ts_result.get('frequency') or ts_profile.get('detected_frequency', 'Period')),
        ])
        elements.append(Spacer(1, 8))
        add_table([
            ['Field', 'Value'],
            ['Date Column', ts_result.get('date_column', 'N/A')],
            ['Target Column', ts_result.get('target_column', 'N/A')],
            ['Usable Periods', ts_profile.get('usable_periods', 'N/A')],
            ['Volatility', metric_text(ts_profile.get('volatility'))],
            ['Stationarity Verdict', stationarity.get('verdict', 'N/A')],
            ['Stationarity Note', stationarity.get('note', 'N/A')],
            ['MAE / RMSE / MAPE', f"{metric_text(ts_metrics.get('mae'))} / {metric_text(ts_metrics.get('rmse'))} / {metric_text(ts_metrics.get('mape'))}"],
        ], [content_width * 0.28, content_width * 0.68], header_bg='#134e4a')
        elements.append(Spacer(1, 8))
        elements.append(build_line_chart_image('Time Series Forecast', ts_result.get('history', []), ts_result.get('test_forecast', []), ts_result.get('future_forecast', []), include_interval=True))
        if ts_result.get('future_forecast'):
            elements.append(Spacer(1, 8))
            future_rows = [['Future Period', 'Forecast', 'Lower', 'Upper']]
            for item in ts_result.get('future_forecast', [])[:10]:
                future_rows.append([item.get('period', 'N/A'), metric_text(item.get('predicted')), metric_text(item.get('lower')), metric_text(item.get('upper'))])
            add_table(future_rows, [content_width * 0.22, content_width * 0.22, content_width * 0.22, content_width * 0.22], header_bg='#115e59')
        elements.append(Spacer(1, 8))
        add_paragraph(ts_result.get('analysis', 'Time-series forecasting output was recorded for this workflow.'), body_style)

    if ml_forecast_result:
        elements.append(PageBreak())
        add_section('Tab 6: Forecast ML', 'The ML forecasting path transforms time into engineered features, then trains a general-purpose learner to project future periods.')
        ml_training = ml_forecast_result.get('training_summary', {}) or {}
        ml_metrics = ml_forecast_result.get('metrics', {}) or {}
        ml_profile = ml_forecast_result.get('dataset_profile', {}) or {}
        add_stat_cards([
            ('Model', ml_training.get('model_name', 'N/A')),
            ('Generated Features', len(ml_forecast_result.get('generated_features', []))),
            ('Lag Depth', ml_training.get('lag_periods', 'N/A')),
            ('Forecast Horizon', ml_training.get('forecast_periods', 'N/A')),
        ])
        elements.append(Spacer(1, 8))
        add_table([
            ['Field', 'Value'],
            ['Date Column', ml_forecast_result.get('date_column', 'N/A')],
            ['Target Column', ml_forecast_result.get('target_column', 'N/A')],
            ['Detected Frequency', ml_profile.get('detected_frequency', 'N/A')],
            ['Usable Periods', ml_profile.get('usable_periods', 'N/A')],
            ['MAE / RMSE / MAPE', f"{metric_text(ml_metrics.get('mae'))} / {metric_text(ml_metrics.get('rmse'))} / {metric_text(ml_metrics.get('mape'))}"],
        ], [content_width * 0.28, content_width * 0.68], header_bg='#134e4a')
        elements.append(Spacer(1, 8))
        elements.append(build_line_chart_image('ML Forecast', ml_forecast_result.get('history', []), ml_forecast_result.get('test_forecast', []), ml_forecast_result.get('future_forecast', []), include_interval=False))
        shap_items = ml_forecast_result.get('shap_feature_importance', [])
        if shap_items:
            elements.append(Spacer(1, 8))
            elements.append(build_bar_chart_image('SHAP Feature Importance', shap_items))
        feature_rows = [['Generated Feature']]
        for feature in ml_forecast_result.get('generated_features', [])[:16]:
            feature_rows.append([feature])
        elements.append(Spacer(1, 8))
        add_table(feature_rows if len(feature_rows) > 1 else [['Generated Feature'], ['None captured']], [540], header_bg='#115e59')
        preview_rows = ml_forecast_result.get('feature_preview_rows', [])
        if preview_rows:
            preview_columns = list(preview_rows[0].keys())[:6]
            rows = [['Preview Feature Row'] + preview_columns]
            for row_index, item in enumerate(preview_rows[:6], start=1):
                rows.append([f'Row {row_index}'] + [item.get(column_name, 'N/A') for column_name in preview_columns])
            elements.append(Spacer(1, 8))
            add_table(rows, [90] + [max(88, (content_width - 90) / max(1, len(preview_columns)))] * len(preview_columns), header_bg='#134e4a')
        elements.append(Spacer(1, 8))
        add_paragraph(ml_forecast_result.get('analysis', 'ML forecasting output was recorded for this workflow.'), body_style)

    elements.append(PageBreak())
    add_section('Tab 7: ML Assistant', 'This section summarizes the supervised learning branch, including selected target, modeling objective, chosen algorithm, feature set, and performance evidence.')
    add_stat_cards([
        ('Target', payload.targetColumn or 'N/A'),
        ('Problem Type', str(payload.problemType).title()),
        ('Selected Model', payload.selectedModel or 'Not trained'),
        ('Features Used', len(payload.selectedFeatures)),
    ])
    elements.append(Spacer(1, 8))
    if payload.selectedFeatures:
        add_paragraph('Selected features: ' + ', '.join(payload.selectedFeatures[:24]))
    if payload.modelMetrics:
        metric_rows = [['Metric', 'Value']]
        for key, value in payload.modelMetrics.items():
            metric_rows.append([key, metric_text(value)])
        add_table(metric_rows, [content_width * 0.48, content_width * 0.48], header_bg='#134e4a')
    else:
        add_paragraph('No supervised ML training metrics were available in the current session.', small_style)
    if payload.featureImportance:
        elements.append(Spacer(1, 8))
        importance_rows = [['Rank', 'Feature', 'Importance']]
        for index, item in enumerate(payload.featureImportance[:12], start=1):
            importance_rows.append([index, item.get('name', 'N/A'), metric_text(item.get('importance'))])
        add_table(importance_rows, [60, content_width * 0.62, content_width * 0.22], header_bg='#115e59')

    elements.append(PageBreak())
    add_section('Tab 8: Prediction', 'The final tab captures the application outcome by storing the latest inference result, probability breakdowns when available, and recent prediction history.')
    if payload.uploadedModel:
        add_table([
            ['Model Name', 'Type', 'Target', 'Problem', 'Trained At'],
            [payload.uploadedModel.name, payload.uploadedModel.type, payload.uploadedModel.target, payload.uploadedModel.problem, payload.uploadedModel.trainedAt],
        ], [content_width * 0.23, content_width * 0.14, content_width * 0.2, content_width * 0.12, content_width * 0.22], header_bg='#0f766e')
        if payload.uploadedModel.features:
            elements.append(Spacer(1, 8))
            add_paragraph('Prediction-serving model features: ' + ', '.join(payload.uploadedModel.features[:24]))
    elements.append(Spacer(1, 8))
    add_stat_cards([
        ('Latest Prediction', payload.predictionResult if payload.predictionResult is not None else 'N/A'),
        ('History Entries', len(payload.predictionHistory)),
        ('Probabilities', 'Available' if payload.predictionProbabilities else 'N/A'),
        ('Analysis', 'Available' if payload.predictionAnalysis else 'N/A'),
    ])
    elements.append(Spacer(1, 8))
    if payload.predictionAnalysis:
        add_paragraph(payload.predictionAnalysis)
    if payload.predictionProbabilities:
        probability_rows = [['Outcome', 'Probability']]
        for label, probability in list(payload.predictionProbabilities.items())[:10]:
            probability_rows.append([label, f'{round(probability * 100, 2)}%'])
        add_table(probability_rows, [content_width * 0.48, content_width * 0.48], header_bg='#134e4a')
        elements.append(Spacer(1, 8))
    latest_feature_rows = latest_prediction_feature_summary()
    if latest_feature_rows:
        add_table(latest_feature_rows, [content_width * 0.48, content_width * 0.48], header_bg='#115e59')
        elements.append(Spacer(1, 8))
    if payload.predictionHistory:
        history_rows = [['Timestamp', 'Prediction', 'Confidence']]
        for item in payload.predictionHistory[-10:]:
            history_rows.append([item.timestamp, item.prediction, 'N/A' if item.confidence is None else f'{round(item.confidence * 100, 2)}%'])
        add_table(history_rows, [content_width * 0.46, content_width * 0.28, content_width * 0.2], header_bg='#134e4a')
    else:
        add_paragraph('No prediction history was recorded in the current session.', small_style)

    doc.build(elements, onFirstPage=decorate_page, onLaterPages=decorate_page)
    return buffer.getvalue()


def build_dynamic_report_doc(payload: ReportPayload) -> bytes:
    session_id = get_session_id(payload.datasetId, payload.sessionId)
    session_state = ensure_session_state(session_id)
    ts_result_raw = payload.timeSeriesForecastResult or session_state.get('time_series_result')
    ml_result_raw = payload.mlForecastResult or session_state.get('ml_forecast_result')
    ts_result = ts_result_raw.model_dump() if hasattr(ts_result_raw, 'model_dump') else ts_result_raw
    ml_forecast_result = ml_result_raw.model_dump() if hasattr(ml_result_raw, 'model_dump') else ml_result_raw

    def html_table(headers: list[str], rows: list[list[Any]]) -> str:
        head_html = ''.join(f'<th>{escape(str(header))}</th>' for header in headers)
        body_html = ''.join(
            '<tr>' + ''.join(f'<td>{escape(str(cell))}</td>' for cell in row) + '</tr>'
            for row in rows
        )
        return f'<table><thead><tr>{head_html}</tr></thead><tbody>{body_html}</tbody></table>'

    column_rows = [
        [column.name, column.dtype, column.role, column.nonNull, column.nullCount, column.uniqueCount]
        for column in payload.columns[:20]
    ]
    cleaning_rows = [
        [log.action, log.detail, log.timestamp]
        for log in payload.cleaningLogs[:24]
    ] or [['None', 'No cleaning logs captured', 'N/A']]
    metric_rows = [[key, value] for key, value in (payload.modelMetrics or {}).items()] or [['N/A', 'No ML metrics captured']]
    prediction_rows = [
        [item.timestamp, item.prediction, 'N/A' if item.confidence is None else f'{round(item.confidence * 100, 2)}%']
        for item in payload.predictionHistory[-10:]
    ] or [['N/A', 'No prediction history captured', 'N/A']]
    ts_future_rows = [
        [item.get('period', 'N/A'), item.get('predicted', 'N/A'), item.get('lower', 'N/A'), item.get('upper', 'N/A')]
        for item in (ts_result or {}).get('future_forecast', [])[:10]
    ]
    ml_future_rows = [
        [item.get('period', 'N/A'), item.get('predicted', 'N/A')]
        for item in (ml_forecast_result or {}).get('future_forecast', [])[:10]
    ]
    loaded_row_count = payload.loadedRowCount or payload.totalRows
    preview_mode = payload.previewLoaded and payload.totalRows > loaded_row_count
    workflow_rows = [
        ['1', 'Upload', 'Completed' if payload.totalRows > 0 else 'Pending', f'{payload.totalRows:,} total rows; {loaded_row_count:,} browser rows'],
        ['2', 'Understanding', 'Completed' if payload.columns else 'Pending', f'{len(payload.columns)} columns profiled'],
        ['3', 'EDA', 'Completed' if payload.columns else 'Pending', f'{len(payload.edaStats.numericColumns)} numeric, {len(payload.edaStats.categoricalColumns)} categorical, {len(payload.edaStats.correlations)} correlations'],
        ['4', 'Cleaning', 'Completed' if payload.cleaningDone else 'Pending', f'{len(payload.cleaningLogs)} actions, {payload.cleanedRowCount:,} rows retained'],
        ['5', 'Forecast TS', 'Completed' if ts_result else 'Skipped', 'Chronology-first forecasting branch'],
        ['6', 'Forecast ML', 'Completed' if ml_forecast_result else 'Skipped', 'Feature-engineered forecasting branch'],
        ['7', 'ML Assistant', 'Completed' if payload.modelMetrics else 'Pending', f'{len(payload.selectedFeatures)} selected features'],
        ['8', 'Prediction', 'Completed' if payload.predictionResult is not None else 'Pending', f'{len(payload.predictionHistory)} prediction records'],
    ]

    workflow_status = 'Complete workflow captured' if payload.predictionResult is not None else 'Workflow summary generated'
    forecast_status = ', '.join(name for name, present in [('Time Series', bool(ts_result)), ('ML Forecast', bool(ml_forecast_result))] if present) or 'No forecasting branch executed'
    prediction_value = escape(str(payload.predictionResult if payload.predictionResult is not None else 'Pending'))
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{escape(payload.fileName)} Workflow Report</title>
  <style>
    @page {{ size: A4 landscape; margin: 0.6in; }}
    body {{ font-family: Arial, sans-serif; color: #1e293b; margin: 0; line-height: 1.5; background: #f8fafc; }}
    .page {{ page-break-after: always; padding: 8px 0 16px; }}
    .page:last-child {{ page-break-after: auto; }}
    .hero {{
      background: linear-gradient(135deg, #0f766e 0%, #0f172a 100%);
      color: white;
      padding: 28px;
      border-radius: 24px;
      box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
    }}
    .eyebrow {{ font-size: 11px; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; color: #a5f3fc; }}
    .hero h1 {{ margin: 10px 0 8px; font-size: 34px; line-height: 1.05; color: white; }}
    .hero p {{ margin: 0; color: #dbeafe; font-size: 15px; max-width: 860px; }}
    .hero-grid, .stats {{ width: 100%; border-collapse: separate; border-spacing: 12px; margin-top: 20px; }}
    .hero-card, .stat {{
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(165,243,252,0.28);
      border-radius: 18px;
      padding: 14px;
      vertical-align: top;
    }}
    .label {{ font-size: 10px; color: #a5f3fc; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 8px; color: white; }}
    .deck-title {{ font-size: 24px; color: #0f172a; margin: 26px 0 10px; }}
    .summary {{
      background: white;
      border: 1px solid #dbe4f0;
      border-radius: 20px;
      padding: 18px 20px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
      margin-top: 18px;
    }}
    .summary strong {{ color: #0f172a; }}
    .section {{
      background: white;
      border: 1px solid #dbe4f0;
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
      margin-top: 18px;
    }}
    .section-label {{ font-size: 10px; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; color: #0284c7; }}
    h2 {{ color: #0f172a; margin: 8px 0 6px; font-size: 24px; }}
    h3 {{ color: #134e4a; margin: 16px 0 8px; font-size: 18px; }}
    .muted {{ color: #64748b; }}
    .metric-grid {{ width: 100%; border-collapse: separate; border-spacing: 12px; margin: 12px 0 6px; }}
    .metric-card {{
      background: #f8fbff;
      border: 1px solid #dbe4f0;
      border-radius: 18px;
      padding: 14px;
      width: 25%;
      vertical-align: top;
    }}
    table.data {{ width: 100%; border-collapse: collapse; margin: 12px 0 4px; font-size: 13px; }}
    table.data th, table.data td {{ border: 1px solid #dbe4f0; padding: 9px 10px; text-align: left; vertical-align: top; }}
    table.data th {{ background: #0f766e; color: white; }}
    table.data tr:nth-child(even) td {{ background: #f8fbff; }}
    .note {{
      background: linear-gradient(135deg, #eff6ff 0%, #ecfeff 100%);
      border: 1px solid #bfdbfe;
      border-radius: 16px;
      padding: 14px 16px;
      margin: 14px 0 0;
    }}
    .footer-note {{ color: #64748b; font-size: 12px; margin-top: 12px; }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <div class="eyebrow">Intelligent Data Assistant</div>
      <h1>Executive Workflow Report</h1>
      <p>This presentation-style document packages the full analytics workflow into a stakeholder-friendly export suitable for PDF distribution and Word-based revision.</p>
      <table class="hero-grid">
        <tr>
          <td class="hero-card"><div class="label">Dataset</div><div class="value">{escape(payload.fileName)}</div></td>
          <td class="hero-card"><div class="label">Workflow Status</div><div class="value">{escape(workflow_status)}</div></td>
          <td class="hero-card"><div class="label">Forecast Paths</div><div class="value">{escape(forecast_status)}</div></td>
          <td class="hero-card"><div class="label">Prediction</div><div class="value">{prediction_value}</div></td>
        </tr>
      </table>
      <table class="stats">
        <tr>
          <td class="stat"><div class="label">Rows</div><div class="value">{payload.totalRows:,}</div></td>
          <td class="stat"><div class="label">Browser Rows</div><div class="value">{loaded_row_count:,}</div></td>
          <td class="stat"><div class="label">Columns</div><div class="value">{len(payload.columns)}</div></td>
          <td class="stat"><div class="label">Workspace Mode</div><div class="value">{escape('Preview + backend cache' if preview_mode else 'Full dataset')}</div></td>
        </tr>
      </table>
    </div>

    <div class="summary">
      <div class="section-label">Executive Summary</div>
      <h2>Workflow Narrative</h2>
      <p>The uploaded dataset entered the application as <strong>{escape(payload.fileName)}</strong>. The current export reflects the executed workflow path across ingestion, profiling, exploratory analysis, cleaning, forecasting, machine learning, and prediction. This version is intended to read more like a presentation deck than a raw technical dump, so the highlights are surfaced first and the operational details follow as structured tables.</p>
      <div class="note">Estimated memory footprint: {escape(payload.memoryUsage)}. Problem type: {escape(payload.problemType)}. Workspace scope: {escape(f'{loaded_row_count:,} preview rows shown in-browser while the backend kept the full dataset cached.' if preview_mode else f'The full {payload.totalRows:,}-row dataset was available directly in the workspace.')}</div>
    </div>
  </div>

  <div class="page">
    <div class="section">
      <div class="section-label">Coverage</div>
      <h2>Workflow Coverage Map</h2>
      {html_table(['Step', 'Tab', 'Status', 'Coverage'], workflow_rows).replace('<table>', '<table class="data">')}
      <h3>Upload and Understanding</h3>
      <table class="metric-grid">
        <tr>
          <td class="metric-card"><div class="label">Rows Loaded</div><div class="value">{payload.totalRows:,}</div></td>
          <td class="metric-card"><div class="label">Browser Rows</div><div class="value">{loaded_row_count:,}</div></td>
          <td class="metric-card"><div class="label">Columns Profiled</div><div class="value">{len(payload.columns)}</div></td>
          <td class="metric-card"><div class="label">Duplicates</div><div class="value">{payload.duplicates:,}</div></td>
        </tr>
      </table>
      <div class="note">{escape(f'{loaded_row_count:,} preview rows were rendered in the browser while the full {payload.totalRows:,}-row dataset remained cached on the backend.' if preview_mode else f'The full {payload.totalRows:,}-row dataset was loaded directly into the workspace.')}</div>
      {html_table(['Column', 'Type', 'Role', 'Non-null', 'Nulls', 'Unique'], column_rows).replace('<table>', '<table class="data">')}
    </div>
  </div>

  <div class="page">
    <div class="section">
      <div class="section-label">Analysis</div>
      <h2>EDA and Cleaning Overview</h2>
      <table class="metric-grid">
        <tr>
          <td class="metric-card"><div class="label">Numeric Columns</div><div class="value">{len(payload.edaStats.numericColumns)}</div></td>
          <td class="metric-card"><div class="label">Categorical Columns</div><div class="value">{len(payload.edaStats.categoricalColumns)}</div></td>
          <td class="metric-card"><div class="label">Correlation Signals</div><div class="value">{len(payload.edaStats.correlations)}</div></td>
          <td class="metric-card"><div class="label">AI Insight</div><div class="value">{escape('Captured' if payload.aiInsights else 'Not captured')}</div></td>
        </tr>
      </table>
      <div class="note">{escape(payload.aiInsights or 'No AI insight captured for this session.')}</div>
      <h3>Cleaning Trail</h3>
      <p class="muted">Cleaning follows EDA in the current application workflow and prepares the stable dataset used by forecasting, machine learning, and prediction.</p>
      {html_table(['Action', 'Detail', 'Timestamp'], cleaning_rows).replace('<table>', '<table class="data">')}
      <h3>Forecasting Overview</h3>
      <h3>Time Series Forecast</h3>
      <p class="muted">{escape(str((ts_result or {}).get('analysis', 'Time-series forecast was not executed in this session.')))}</p>
      {html_table(['Future Period', 'Forecast', 'Lower', 'Upper'], ts_future_rows or [['N/A', 'N/A', 'N/A', 'N/A']]).replace('<table>', '<table class="data">')}
      <h3>ML Forecast</h3>
      <p class="muted">{escape(str((ml_forecast_result or {}).get('analysis', 'ML forecast was not executed in this session.')))}</p>
      {html_table(['Future Period', 'Forecast'], ml_future_rows or [['N/A', 'N/A']]).replace('<table>', '<table class="data">')}
    </div>
  </div>

  <div class="page">
    <div class="section">
      <div class="section-label">Modeling</div>
      <h2>Machine Learning and Prediction</h2>
      <table class="metric-grid">
        <tr>
          <td class="metric-card"><div class="label">Target Column</div><div class="value">{escape(str(payload.targetColumn or 'N/A'))}</div></td>
          <td class="metric-card"><div class="label">Selected Model</div><div class="value">{escape(str(payload.selectedModel or 'N/A'))}</div></td>
          <td class="metric-card"><div class="label">Feature Count</div><div class="value">{len(payload.selectedFeatures)}</div></td>
          <td class="metric-card"><div class="label">Latest Prediction</div><div class="value">{prediction_value}</div></td>
        </tr>
      </table>
      <h3>ML Metrics</h3>
      {html_table(['Metric', 'Value'], metric_rows).replace('<table>', '<table class="data">')}
      <h3>Prediction Log</h3>
      <p class="muted">{escape(payload.predictionAnalysis or 'No prediction analysis captured.')}</p>
      {html_table(['Timestamp', 'Prediction', 'Confidence'], prediction_rows).replace('<table>', '<table class="data">')}
      <p class="footer-note">This editable export is formatted to feel presentation-ready in Word-compatible tools while remaining easy to revise, annotate, or convert to a formal client deliverable.</p>
    </div>
  </div>
</body>
</html>"""
    return html.encode('utf-8')


@router.post('/cache-dataset')
def cache_dataset(request: DatasetCacheRequest, http_request: Request) -> JSONResponse:
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

    response = {
        'datasetId': dataset_id,
        'rowCount': int(len(data_frame)),
        'loadedRowCount': int(len(data_frame)),
        'columnCount': int(len(data_frame.columns)),
        'columns': list(data_frame.columns),
        'previewLoaded': False,
    }
    record_activity(
        request=http_request,
        action='cache_dataset',
        status='success',
        dataset_id=dataset_id,
        file_name=request.file_name,
        detail=f'Cached {request.file_name} for backend processing.',
        metadata={
            'row_count': int(len(data_frame)),
            'column_count': int(len(data_frame.columns)),
        },
    )
    return JSONResponse(content=response)


@router.get('/dataset-preview')
def get_dataset_preview(
    http_request: Request,
    dataset_id: str = Query(...),
) -> JSONResponse:
    dataset_entry = DATASET_CACHE.get(dataset_id)
    if dataset_entry is None:
        raise HTTPException(status_code=404, detail='Cached dataset not found. Please upload the file again.')

    preview_frame, is_polars_preview = load_cached_preview(dataset_entry, PARQUET_PREVIEW_ROW_LIMIT)
    row_count = int(dataset_entry.get('row_count') or (preview_frame.height if is_polars_preview else len(preview_frame)))
    loaded_row_count = int(preview_frame.height if is_polars_preview else len(preview_frame))
    preview_loaded = row_count > loaded_row_count
    duplicate_rows = int(dataset_entry.get('duplicate_count') or 0)

    if is_polars_preview:
        preview_rows = safe_serialize(preview_frame.to_dicts())
        preview_columns = build_column_info_from_polars_frame(preview_frame)
    else:
        preview_rows = safe_serialize(preview_frame.to_dict(orient='records'))
        preview_columns = build_column_info_from_frame(preview_frame)

    response = {
        'datasetId': dataset_id,
        'fileName': dataset_entry.get('filename'),
        'data': preview_rows,
        'columns': preview_columns,
        'rowCount': row_count,
        'loadedRowCount': loaded_row_count,
        'previewLoaded': preview_loaded,
        'duplicates': duplicate_rows,
    }
    record_activity(
        request=http_request,
        action='load_dataset_preview',
        status='success',
        dataset_id=dataset_id,
        file_name=str(dataset_entry.get('filename') or ''),
        detail='Loaded a cached dataset preview for workspace restore.',
        metadata={
            'row_count': row_count,
            'loaded_row_count': loaded_row_count,
            'preview_loaded': preview_loaded,
        },
    )
    return JSONResponse(content=response)


async def parse_dataset_file(http_request: Request, file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail='A dataset file is required.')

    file_name = file.filename
    lower_file_name = file_name.lower()
    supported_exts = ('.parquet', '.csv', '.tsv', '.xlsx', '.xls')
    if not lower_file_name.endswith(supported_exts):
        raise HTTPException(status_code=400, detail='Only .csv, .tsv, .xlsx, .xls, and .parquet files are supported.')

    dataset_id = str(uuid.uuid4())[:8]
    file_suffix = Path(file_name).suffix.lower()
    cached_path, _ = await write_uploaded_file(file, dataset_id, suffix=file_suffix)
    dataset_entry: dict[str, Any] = {
        'filename': file_name,
        'row_count': 0,
        'column_count': 0,
        'columns': [],
        'duplicate_count': 0,
    }

    try:
        if lower_file_name.endswith('.parquet'):
            parquet_file = pq.ParquetFile(cached_path)
            total_rows = int(parquet_file.metadata.num_rows)
            column_count = len(parquet_file.schema.names)
            frame = pl.read_parquet(cached_path, n_rows=PARQUET_PREVIEW_ROW_LIMIT, low_memory=True)
            dataset_entry.update({'parquet_path': str(cached_path)})
            rows = frame.to_dicts()
            column_info = build_column_info_from_polars_frame(frame)
            preview_duplicate_rows = int(max(0, frame.height - frame.unique().height))
        elif lower_file_name.endswith('.csv') or lower_file_name.endswith('.tsv'):
            sep = '\t' if lower_file_name.endswith('.tsv') else ','
            frame = pl.read_csv(cached_path, separator=sep, n_rows=PARQUET_PREVIEW_ROW_LIMIT, infer_schema_length=1000, ignore_errors=True)
            total_rows = count_csv_rows_from_path(cached_path, sep=sep)
            column_count = frame.width
            dataset_entry.update({'csv_path': str(cached_path)})
            rows = frame.to_dicts()
            column_info = build_column_info_from_polars_frame(frame)
            preview_duplicate_rows = int(max(0, frame.height - frame.unique().height))
        else:
            # Excel workbook
            try:
                preview_frame = pd.read_excel(cached_path, nrows=PARQUET_PREVIEW_ROW_LIMIT)
            except Exception as excel_error:
                raise HTTPException(status_code=400, detail=f'Failed to parse Excel file: {excel_error}') from excel_error

            total_rows = len(preview_frame)
            try:
                total_rows = count_excel_rows_from_path(cached_path)
            except HTTPException:
                total_rows = len(preview_frame)

            frame = preview_frame
            column_count = len(frame.columns)
            dataset_entry.update({'excel_path': str(cached_path)})
            rows = frame.where(pd.notna(frame), None).to_dict(orient='records')
            column_info = build_column_info_from_frame(frame)
            preview_duplicate_rows = int(max(0, len(frame) - len(frame.drop_duplicates())))

        dataset_entry.update({
            'row_count': int(total_rows),
            'column_count': int(column_count),
            'columns': list(frame.columns),
            'duplicate_count': int(preview_duplicate_rows),
        })
        DATASET_CACHE[dataset_id] = dataset_entry

        preview_loaded = int(total_rows) > len(rows)
        response = {
            'datasetId': dataset_id,
            'data': safe_serialize(rows),
            'columns': list(frame.columns),
            'columnInfo': column_info,
            'rowCount': int(total_rows),
            'loadedRowCount': int(len(rows)),
            'columnCount': int(column_count),
            'previewLoaded': preview_loaded,
            'previewLimit': PARQUET_PREVIEW_ROW_LIMIT,
        }
        record_activity(
            request=http_request,
            action='parse_dataset',
            status='success',
            dataset_id=dataset_id,
            file_name=file_name,
            detail=f'Parsed dataset file {file_name}.',
            metadata={
                'row_count': int(total_rows),
                'loaded_row_count': int(len(rows)),
                'column_count': int(column_count),
                'preview_loaded': preview_loaded,
            },
        )
        return JSONResponse(content=response)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=f'Failed to parse dataset file: {error}') from error




@router.post('/parse-dataset')
async def parse_dataset(http_request: Request, file: UploadFile = File(...)) -> JSONResponse:
    return await parse_dataset_file(http_request, file)


@router.post('/parse-parquet')
async def parse_parquet(http_request: Request, file: UploadFile = File(...)) -> JSONResponse:
    return await parse_dataset_file(http_request, file)


@router.post('/clean-dataset')
def clean_dataset(request: ParquetCleaningRequest, http_request: Request) -> JSONResponse:
    try:
        result = clean_cached_dataset(request)
        record_activity(
            request=http_request,
            action='clean_dataset',
            status='success',
            dataset_id=request.dataset_id,
            detail='Cleaned cached dataset and persisted the transformed version.',
            metadata={
                'row_count': result.get('rowCount'),
                'original_row_count': result.get('originalRowCount'),
                'logged_actions': len(result.get('logs', [])),
            },
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception('Dataset cleaning failed dataset_id=%s', request.dataset_id)
        raise HTTPException(status_code=400, detail=f'Dataset cleaning failed: {error}') from error


@router.post('/clean-parquet')
def clean_parquet(request: ParquetCleaningRequest, http_request: Request) -> JSONResponse:
    return clean_dataset(request, http_request)

@router.post('/cleaning-justification')
def cleaning_justification(request: CleaningJustificationRequest, http_request: Request) -> JSONResponse:
    if not request.logs:
        raise HTTPException(status_code=400, detail='No cleaning logs provided.')
    response = {'justification': generate_cleaning_justification(request)}
    record_activity(
        request=http_request,
        action='cleaning_justification',
        status='success',
        detail='Generated AI cleaning justification summary.',
        metadata={
            'log_count': len(request.logs),
            'total_rows': request.totalRows,
            'total_columns': request.totalColumns,
        },
    )
    return JSONResponse(content=response)


@router.post('/eda/advanced')
def advanced_eda(request: AdvancedEdaRequest, http_request: Request) -> JSONResponse:
    try:
        result = safe_serialize(build_advanced_eda_payload(request))
        record_activity(
            request=http_request,
            action='advanced_eda',
            status='success',
            dataset_id=request.dataset_id,
            detail='Generated advanced EDA payload.',
            metadata={'has_cached_dataset': bool(request.dataset_id)},
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception('Advanced EDA generation failed dataset_id=%s', request.dataset_id)
        raise HTTPException(status_code=400, detail=f'Advanced EDA generation failed: {error}') from error


@router.post('/eda/report')
def generate_eda_report(payload: EdaPdfPayload, http_request: Request) -> Response:
    try:
        report_bytes = build_eda_pdf(payload)
    except Exception as error:
        logger.exception('EDA PDF generation failed file_name=%s', payload.fileName)
        raise HTTPException(status_code=400, detail=f'Failed to generate EDA PDF: {error}') from error

    file_stem = ''.join(ch for ch in payload.fileName.rsplit('.', 1)[0] if ch.isalnum() or ch in ('-', '_', ' ')).strip() or 'dataset'
    record_activity(
        request=http_request,
        action='generate_eda_pdf',
        status='success',
        dataset_id=payload.datasetId,
        file_name=payload.fileName,
        detail='Generated the EDA tab PDF export.',
        metadata={
            'total_rows': payload.totalRows,
            'column_count': len(payload.columns),
            'numeric_columns': len(payload.edaStats.numericColumns),
            'categorical_columns': len(payload.edaStats.categoricalColumns),
            'advanced_analysis_available': bool(payload.advancedAnalysis),
        },
    )
    return Response(
        content=report_bytes,
        media_type='application/pdf',
        headers={'Content-Disposition': f'attachment; filename="{file_stem}_eda_report.pdf"'},
    )


@router.post('/report/generate')
@router.post('/generate-report')
def generate_report(payload: ReportPayload, http_request: Request, format: Literal['pdf', 'doc'] = Query(default='pdf')) -> Response:
    try:
        report_bytes = build_dynamic_report_pdf(payload) if format == 'pdf' else build_dynamic_report_doc(payload)
    except Exception as error:
        logger.exception('Report generation failed file_name=%s', payload.fileName)
        raise HTTPException(status_code=400, detail=f'Failed to generate report: {error}') from error

    file_stem = ''.join(ch for ch in payload.fileName.rsplit('.', 1)[0] if ch.isalnum() or ch in ('-', '_', ' ')).strip() or 'dataset'
    server_session_id = get_session_id(payload.datasetId, payload.sessionId)
    record_activity(
        request=http_request,
        action='generate_report',
        status='success',
        dataset_id=payload.datasetId,
        server_session_id=server_session_id,
        file_name=payload.fileName,
        detail=f'Generated a {format.upper()} workflow report.',
        metadata={
            'format': format,
            'cleaning_done': payload.cleaningDone,
            'problem_type': payload.problemType,
            'selected_model': payload.selectedModel,
            'prediction_available': payload.predictionResult is not None,
        },
    )
    return Response(
        content=report_bytes,
        media_type='application/pdf' if format == 'pdf' else 'application/msword',
        headers={'Content-Disposition': f'attachment; filename="{file_stem}_analysis_report.{"pdf" if format == "pdf" else "doc"}"'},
    )


@router.get('/activities')
def list_activities(
    request: Request,
    dataset_id: str | None = Query(default=None),
    client_session_id: str | None = Query(default=None),
    server_session_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> JSONResponse:
    if not ACTIVITY_DB_AVAILABLE:
        return JSONResponse(
            content={
                'activities': [],
                'count': 0,
                'dbAvailable': False,
                'message': 'Activity database is unavailable. Start PostgreSQL to enable persisted activity history.',
            }
        )

    effective_client_session_id = client_session_id or get_client_session_id(request)
    query = '''
        SELECT
            activity_id,
            created_at,
            client_session_id,
            server_session_id,
            dataset_id,
            model_id,
            activity_type,
            action,
            status,
            api_path,
            http_method,
            status_code,
            duration_ms,
            file_name,
            detail,
            metadata_json
        FROM user_activities
        WHERE 1 = 1
    '''
    params: list[Any] = []

    if dataset_id:
        query += ' AND dataset_id = %s'
        params.append(dataset_id)
    if effective_client_session_id:
        query += ' AND client_session_id = %s'
        params.append(effective_client_session_id)
    if server_session_id:
        query += ' AND server_session_id = %s'
        params.append(server_session_id)

    query += ' ORDER BY id DESC LIMIT %s'
    params.append(limit)

    try:
        with get_activity_connection() as connection:
            rows = connection.execute(query, params).fetchall()
    except Exception:
        logger.exception('Failed to query user activities.')
        return JSONResponse(
            content={
                'activities': [],
                'count': 0,
                'dbAvailable': False,
                'message': 'Activity database query failed. Start PostgreSQL to enable persisted activity history.',
            }
        )

    activities = []
    for row in rows:
        metadata_json = row['metadata_json']
        activities.append({
            'activityId': row['activity_id'],
            'createdAt': row['created_at'],
            'clientSessionId': row['client_session_id'],
            'serverSessionId': row['server_session_id'],
            'datasetId': row['dataset_id'],
            'modelId': row['model_id'],
            'activityType': row['activity_type'],
            'action': row['action'],
            'status': row['status'],
            'apiPath': row['api_path'],
            'httpMethod': row['http_method'],
            'statusCode': row['status_code'],
            'durationMs': row['duration_ms'],
            'fileName': row['file_name'],
            'detail': row['detail'],
            'metadata': json.loads(metadata_json) if metadata_json else None,
        })

    return JSONResponse(content={'activities': activities, 'count': len(activities), 'dbAvailable': True})


@router.post('/auth/register')
def register_user(payload: RegisterRequest, request: Request) -> JSONResponse:
    try:
        user = create_app_user(username=payload.username, email=payload.email, password=payload.password)
        _, session_token = create_authenticated_session(user_id=user['userId'], request=request)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception('User registration failed email=%s', payload.email)
        raise HTTPException(status_code=500, detail=f'Failed to register user: {error}') from error

    response = JSONResponse(content={'user': user})
    set_session_cookie(response, session_token)
    record_activity(
        request=request,
        action='user_register',
        status='success',
        activity_type='auth',
        detail=f'User {user["email"]} registered.',
        metadata={
            'user_id': user['userId'],
            'email': user['email'],
            'username': user['username'],
        },
    )
    return response


@router.post('/auth/login')
def login_user(payload: LoginRequest, request: Request) -> JSONResponse:
    try:
        user = authenticate_user(email=payload.email, password=payload.password)
        _, session_token = create_authenticated_session(user_id=user['userId'], request=request)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception('User login failed email=%s', payload.email)
        raise HTTPException(status_code=500, detail=f'Failed to login user: {error}') from error

    response = JSONResponse(content={'user': user})
    set_session_cookie(response, session_token)
    record_activity(
        request=request,
        action='user_login',
        status='success',
        activity_type='auth',
        detail=f'User {user["email"]} signed in.',
        metadata={
            'user_id': user['userId'],
            'email': user['email'],
            'username': user['username'],
        },
    )
    return response


@router.get('/auth/me')
def auth_me(request: Request) -> JSONResponse:
    try:
        user = build_user_payload(get_authenticated_user(request))
    except HTTPException:
        raise
    except Exception as error:
        logger.exception('Auth me failed.')
        raise HTTPException(status_code=500, detail=f'Failed to resolve current session: {error}') from error

    return JSONResponse(content={'user': user})


@router.post('/auth/logout')
def logout_user(request: Request) -> JSONResponse:
    session_token = request.cookies.get(SESSION_COOKIE_NAME)
    if session_token:
        try:
            revoke_session(session_token)
        except Exception:
            logger.exception('Failed to revoke user session during logout.')

    response = JSONResponse(content={'success': True})
    clear_session_cookie(response)
    record_activity(
        request=request,
        action='user_logout',
        status='success',
        activity_type='auth',
        detail='User logged out.',
    )
    return response


@router.get('/health')
def health() -> dict[str, str]:
    return {
        'status': 'healthy',
        'activityDb': 'available' if ACTIVITY_DB_AVAILABLE else 'unavailable',
    }


@app.get('/')
def root() -> dict[str, Any]:
    return {'service': 'AI-Assisted EDA & ML Backend', 'docs': '/docs', 'api': '/api'}


app.include_router(router)


if __name__ == '__main__':
    import uvicorn

    port = int(os.environ.get('ML_PORT', '3004'))
    uvicorn.run(app, host='0.0.0.0', port=port)




