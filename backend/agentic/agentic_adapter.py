from __future__ import annotations

import html
import importlib
import json
import mimetypes
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from agentic.run_artifacts import append_audit_log, read_session_summary, write_decision, write_step_output

agentic_router = APIRouter(prefix='/api/agentic')

AGENTIC_LAYER_ROOT = Path(__file__).resolve().parents[2] / 'agentic-layer'
if str(AGENTIC_LAYER_ROOT) not in sys.path:
    sys.path.append(str(AGENTIC_LAYER_ROOT))

from server.agent import respond as core_chat_respond  # noqa: E402
from server.automation import RUNS_ROOT as CORE_RUNS_ROOT, create_run as core_create_run, record_decision as core_record_decision  # noqa: E402
from server.config import AGENTIC_ROOT as CORE_AGENTIC_ROOT, Settings as CoreSettings  # noqa: E402

CORE_UI_ROOT = CORE_AGENTIC_ROOT / 'ui'
CORE_ACTIVITY_LOG = CORE_AGENTIC_ROOT / 'logs' / 'activity.jsonl'

PIPELINE_STEPS = [
    'Data Understanding',
    'EDA',
    'Data Cleaning',
    'Time Series Forecast',
    'ML Forecast',
    'Loss Forecast',
    'Profit Forecast',
    'ML Assistant',
    'Prediction',
    'Report Generation',
]

# AGENTIC LAYER START
_sessions: dict[str, dict[str, Any]] = {}
_step_executions: dict[str, list[dict[str, Any]]] = {}
_decisions: dict[str, list[dict[str, Any]]] = {}
# AGENTIC LAYER END


class SuggestNextStepsRequest(BaseModel):
    dataset_path: str


class ExecuteStepRequest(BaseModel):
    session_id: str
    step_name: str
    approved_by: str


class DecisionRequest(BaseModel):
    session_id: str
    step_name: str
    decision: str
    reasoning: str = ''


def core_json_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(content={'error': message}, status_code=status_code)


def core_rewrite_json_links(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: core_rewrite_json_links(item) for key, item in value.items()}
    if isinstance(value, list):
        return [core_rewrite_json_links(item) for item in value]
    if isinstance(value, str) and value.startswith('/runs/'):
        return f'/api/agentic/core{value}'
    return value


def core_record_activity(payload: dict[str, Any]) -> None:
    CORE_ACTIVITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    event = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'type': str(payload.get('type', 'ui_event'))[:80],
        'title': str(payload.get('title', ''))[:160],
        'detail': str(payload.get('detail', ''))[:500],
        'session': str(payload.get('session', ''))[:80],
        'mode': str(payload.get('mode', ''))[:40],
    }
    with CORE_ACTIVITY_LOG.open('a', encoding='utf-8') as log_file:
        log_file.write(json.dumps(event, ensure_ascii=True) + '\n')


def core_text_asset(path: Path) -> str:
    text = path.read_text(encoding='utf-8')
    if path.name == 'index.html':
        text = text.replace('href="/styles.css"', 'href="/api/agentic/core/styles.css"')
        text = text.replace('src="/app.js"', 'src="/api/agentic/core/app.js"')
    if path.name == 'app.js':
        replacements = {
            'fetch("/api/activity"': 'fetch("/api/agentic/core/activity"',
            'fetch("/api/health"': 'fetch("/api/agentic/core/health"',
            'fetch("/api/workflow/suggest"': 'fetch("/api/agentic/core/workflow/suggest"',
            'fetch("/api/workflow/decision"': 'fetch("/api/agentic/core/workflow/decision"',
            'fetch("/api/chat"': 'fetch("/api/agentic/core/chat"',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
    if path.name == 'launcher.js':
        text = text.replace('"http://127.0.0.1:5055"', '"/api/agentic/core"')
    return text


def core_file_response(path: Path) -> Response:
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail='File not found')
    content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    if path.suffix.lower() in {'.html', '.js', '.css'}:
        return Response(content=core_text_asset(path), media_type=content_type)
    return Response(content=path.read_bytes(), media_type=content_type)


def agentic_enabled() -> bool:
    return os.environ.get('NEXT_PUBLIC_AGENTIC_ENABLED', 'true').strip().lower() != 'false'


def disabled_response() -> JSONResponse:
    return JSONResponse(content={'error': 'agentic layer disabled', 'status': 503}, status_code=503)


# AGENTIC LAYER START
def is_db_connected() -> bool:
    try:
        backend = get_backend_module()
        with backend.get_activity_connection() as connection:
            connection.execute('SELECT 1')
        return True
    except Exception:
        try:
            backend.logger.warning('Agentic database unavailable; using in-memory fallback store.', exc_info=True)
        except Exception:
            pass
        return False
# AGENTIC LAYER END


def get_backend_module() -> Any:
    for module_name in ('main', 'backend.main', '__main__'):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, 'run_loss_forecast'):
            return module
    return importlib.import_module('main')


def read_dataset_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix == '.tsv':
        return pd.read_csv(path, sep='\t')
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    if suffix == '.parquet':
        return pd.read_parquet(path)
    raise HTTPException(status_code=400, detail='Agentic profiling accepts .csv, .tsv, .xlsx, .xls, and .parquet files.')


def read_dataset_profile(dataset_path: str) -> dict[str, Any]:
    backend = get_backend_module()
    dataset_entry = getattr(backend, 'DATASET_CACHE', {}).get(dataset_path)
    if dataset_entry is not None:
        try:
            frame = backend.load_full_dataset_frame(dataset_path, [])
            return profile_frame(frame, source=dataset_path)
        except Exception as error:
            raise HTTPException(status_code=400, detail=f'Failed to profile cached dataset: {error}') from error

    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f'Dataset path not found: {dataset_path}')

    frame = read_dataset_file(path)
    return profile_frame(frame, source=str(path))


def profile_frame(frame: pd.DataFrame, source: str) -> dict[str, Any]:
    date_columns = []
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            date_columns.append(str(column))
            continue
        parsed = pd.to_datetime(frame[column], errors='coerce')
        if len(parsed) and parsed.notna().mean() >= 0.8:
            date_columns.append(str(column))

    null_counts = {str(column): int(value) for column, value in frame.isna().sum().items()}
    dtypes = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
    numeric_columns = [str(column) for column in frame.select_dtypes(include='number').columns]

    return {
        'path': source,
        'shape': {'rows': int(frame.shape[0]), 'columns': int(frame.shape[1])},
        'nulls': null_counts,
        'dtypes': dtypes,
        'date_columns': date_columns,
        'numeric_columns': numeric_columns,
    }


def build_findings(profile: dict[str, Any]) -> list[str]:
    shape = profile['shape']
    findings = [f"Detected {shape['rows']} rows and {shape['columns']} columns."]
    null_total = sum(int(value) for value in profile['nulls'].values())
    if null_total:
        findings.append(f'Detected {null_total} missing values across the dataset.')
    else:
        findings.append('No missing values detected in the dataset profile.')
    if profile['date_columns']:
        findings.append(f"Detected date-like columns: {', '.join(profile['date_columns'])}.")
    if profile['numeric_columns']:
        findings.append(f"Detected {len(profile['numeric_columns'])} numeric columns for analysis or modeling.")
    return findings


def build_recommendations(profile: dict[str, Any]) -> list[dict[str, Any]]:
    findings = build_findings(profile)
    recommendations = [
        {
            'step': step,
            'reason': reason,
            'findings': findings,
        }
        for step, reason in [
            ('Data Understanding', 'Start by confirming column roles, inferred data types, and basic dataset quality.'),
            ('EDA', 'Generate exploratory statistics and distributions after the dataset structure is known.'),
            ('Data Cleaning', 'Prepare a stable cleaned dataset before forecasting and model training.'),
            ('Time Series Forecast', 'Use the detected date-like and numeric fields for chronological forecasting.'),
            ('ML Forecast', 'Train a feature-engineered forecasting branch for comparison.'),
            ('Loss Forecast', 'Estimate loss exposure after the two forecast branches are available.'),
            ('Profit Forecast', 'Project scenario-based profit after loss exposure is estimated.'),
            ('ML Assistant', 'Train a supervised model using the strongest available target and features.'),
            ('Prediction', 'Run a final prediction from the trained model.'),
            ('Report Generation', 'Compile the full workflow into one downloadable report.'),
        ]
    ]
    return recommendations


def ensure_session(session_id: str) -> dict[str, Any]:
    if session_id not in _sessions:
        _sessions[session_id] = {
            'steps': {step: 'pending' for step in PIPELINE_STEPS},
            'events': [],
            'recommendations': [],
            'created_at': datetime.utcnow().isoformat(),
            'updated_at': datetime.utcnow().isoformat(),
        }
    return _sessions[session_id]


def summarize_response(response: Any) -> str:
    if hasattr(response, 'body'):
        try:
            payload = json.loads(response.body.decode('utf-8'))
        except Exception:
            return 'Step completed and returned a response.'
        if isinstance(payload, dict):
            status = payload.get('status') or 'success'
            keys = ', '.join(sorted(str(key) for key in payload.keys())[:8])
            return f'Step completed with status {status}. Returned fields: {keys}.'
    return 'Step completed.'


def session_dataset_id(session_id: str) -> str:
    session = ensure_session(session_id)
    dataset_id = str(session.get('dataset_path') or session.get('dataset_id') or '')
    if not dataset_id:
        raise HTTPException(status_code=422, detail='Agentic session is missing dataset context. Run Suggest Next Steps again.')
    return dataset_id


def session_frame(session_id: str) -> pd.DataFrame:
    return read_dataset_file_or_cache(session_dataset_id(session_id))


def read_dataset_file_or_cache(dataset_path: str) -> pd.DataFrame:
    backend = get_backend_module()
    dataset_entry = getattr(backend, 'DATASET_CACHE', {}).get(dataset_path)
    if dataset_entry is not None:
        return backend.load_full_dataset_frame(dataset_path, [])
    path = Path(dataset_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f'Dataset path not found: {dataset_path}')
    return read_dataset_file(path)


def preferred_date_column(frame: pd.DataFrame) -> str:
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            return str(column)
        parsed = pd.to_datetime(frame[column], errors='coerce')
        if len(parsed) and parsed.notna().mean() >= 0.8:
            return str(column)
    for column in frame.columns:
        if re.search(r'date|month|period|time', str(column), re.IGNORECASE):
            return str(column)
    raise HTTPException(status_code=422, detail='A date-like column is required for automated forecasting.')


def preferred_target_column(frame: pd.DataFrame) -> str:
    numeric_columns = [str(column) for column in frame.select_dtypes(include='number').columns]
    for pattern in (r'total_taxable_amt|revenue|sales|amount|profit|value|price|cost', r'.+'):
        for column in numeric_columns:
            if re.search(pattern, column, re.IGNORECASE):
                return column
    raise HTTPException(status_code=422, detail='A numeric target column is required for automated forecasting and training.')


def auto_feature_columns(frame: pd.DataFrame, target_column: str) -> list[str]:
    return [str(column) for column in frame.columns if str(column) != target_column and not re.search(r'(^id$|_id$|uuid)', str(column), re.IGNORECASE)][:30]


def infer_problem_type(frame: pd.DataFrame, target_column: str) -> str:
    if target_column in frame.columns and pd.api.types.is_numeric_dtype(frame[target_column]) and frame[target_column].nunique(dropna=True) > 12:
        return 'regression'
    return 'classification'


def record_agentic_execution(session_id: str, step_name: str, status: str, output_summary: str | None, error_message: str | None) -> None:
    execution = {
        'session_id': session_id,
        'step_name': step_name,
        'status': status,
        'started_at': datetime.utcnow().isoformat(),
        'completed_at': datetime.utcnow().isoformat(),
        'output_summary': output_summary,
        'error_message': error_message,
    }
    backend = get_backend_module()
    if not getattr(backend, 'ACTIVITY_DB_AVAILABLE', False):
        _step_executions.setdefault(session_id, []).append(execution)
        append_audit_log(session_id, 'step_execution_recorded', execution)
        return
    # AGENTIC LAYER START
    try:
        with backend.get_activity_connection() as connection:
            connection.execute(
                '''
                INSERT INTO agentic_step_executions (
                    session_id, step_name, status, started_at, completed_at, output_summary, error_message
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''',
                (
                    session_id,
                    step_name,
                    status,
                    execution['started_at'],
                    execution['completed_at'],
                    output_summary,
                    error_message,
                ),
            )
    except Exception:
        backend.logger.warning(
            'Failed to persist agentic execution event for session_id=%s step=%s; using in-memory fallback.',
            session_id,
            step_name,
            exc_info=True,
        )
        _step_executions.setdefault(session_id, []).append(execution)
        append_audit_log(session_id, 'step_execution_recorded', execution)
    # AGENTIC LAYER END


def execute_data_understanding(session_id: str, request: Request) -> Any:
    backend = get_backend_module()
    dataset_id = session_dataset_id(session_id)
    frame = session_frame(session_id)
    payload = {
        'status': 'success',
        'dataset_id': dataset_id,
        'columns': backend.safe_serialize(backend.build_column_info_from_frame(frame)),
        'row_count': int(len(frame)),
        'column_count': int(len(frame.columns)),
        'profile': profile_frame(frame, source=dataset_id),
    }
    append_audit_log(session_id, 'data_understanding_completed', payload)
    return JSONResponse(content=payload)


def execute_eda(session_id: str, request: Request) -> Any:
    backend = get_backend_module()
    payload = backend.AdvancedEdaRequest(dataset_id=session_dataset_id(session_id), data=[])
    return backend.advanced_eda(payload, request)


def execute_data_cleaning(session_id: str, request: Request) -> Any:
    backend = get_backend_module()
    payload = backend.ParquetCleaningRequest(
        dataset_id=session_dataset_id(session_id),
        remove_duplicates=True,
        handle_missing=True,
        convert_dates=True,
        standardize_names=True,
        infer_dtypes=True,
    )
    return backend.clean_dataset(payload, request)


def execute_time_series_forecast(session_id: str, request: Request) -> Any:
    backend = get_backend_module()
    dataset_id = session_dataset_id(session_id)
    frame = session_frame(session_id)
    payload = backend.TimeSeriesForecastRequest(
        dataset_id=dataset_id,
        session_id=dataset_id,
        date_column=preferred_date_column(frame),
        target_column=preferred_target_column(frame),
        forecast_periods=3,
        test_percentage=20,
        model_type='sarima',
    )
    return backend.forecast_time_series(payload, request)


def execute_ml_forecast(session_id: str, request: Request) -> Any:
    backend = get_backend_module()
    dataset_id = session_dataset_id(session_id)
    frame = session_frame(session_id)
    payload = backend.MlForecastRequest(
        dataset_id=dataset_id,
        session_id=dataset_id,
        date_column=preferred_date_column(frame),
        target_column=preferred_target_column(frame),
        forecast_periods=3,
        test_percentage=20,
        lag_periods=3,
        model_type='gradient_boosting',
        feature_groups=['trend', 'calendar', 'lags', 'rolling'],
    )
    return backend.forecast_ml(payload, request)


def execute_loss_forecast(session_id: str, request: Request) -> Any:
    backend = get_backend_module()
    payload = backend.ForecastRunRequest(
        session_id=session_dataset_id(session_id),
        forecast_periods=30,
        confirmed_assumptions=True,
    )
    return backend.run_loss_forecast(payload, request)


def execute_profit_forecast(session_id: str, request: Request) -> Any:
    backend = get_backend_module()
    payload = backend.ForecastRunRequest(
        session_id=session_dataset_id(session_id),
        forecast_periods=30,
        confirmed_assumptions=True,
        scenario_parameters={
            'optimistic': {'revenue': 1.1, 'cogs': 0.97, 'loss': 0.8},
            'baseline': {'revenue': 1.0, 'cogs': 1.0, 'loss': 1.0},
            'pessimistic': {'revenue': 0.9, 'cogs': 1.05, 'loss': 1.2},
        },
    )
    return backend.run_profit_forecast(payload, request)


def execute_ml_assistant(session_id: str, request: Request) -> Any:
    backend = get_backend_module()
    dataset_id = session_dataset_id(session_id)
    frame = session_frame(session_id)
    target_column = preferred_target_column(frame)
    feature_columns = auto_feature_columns(frame, target_column)
    if not feature_columns:
        raise HTTPException(status_code=422, detail='Automated model training needs at least one feature column.')
    problem_type = infer_problem_type(frame, target_column)
    payload = backend.TrainRequest(
        dataset_id=dataset_id,
        data=[],
        target_column=target_column,
        feature_columns=feature_columns,
        problem_type=problem_type,
        model_type='ridge_regression' if problem_type == 'regression' else 'random_forest',
        test_size=0.2,
        random_state=42,
        cv_folds=5,
        training_mode='fast',
    )
    response = backend.train_model(payload, request)
    try:
        data = json.loads(response.body.decode('utf-8'))
        session = ensure_session(session_id)
        session['model_id'] = data.get('model_id')
        session['target_column'] = target_column
        session['feature_columns'] = feature_columns
        session['problem_type'] = problem_type
        session['model_type'] = payload.model_type
    except Exception:
        pass
    return response


def execute_prediction(session_id: str, request: Request) -> Any:
    backend = get_backend_module()
    session = ensure_session(session_id)
    model_id = session.get('model_id')
    if not model_id:
        execute_ml_assistant(session_id, request)
        model_id = ensure_session(session_id).get('model_id')
    if not model_id:
        raise HTTPException(status_code=422, detail='Prediction needs a trained model id.')
    frame = session_frame(session_id)
    feature_columns = session.get('feature_columns') or auto_feature_columns(frame, preferred_target_column(frame))
    sample = frame.head(1).to_dict(orient='records')[0] if not frame.empty else {}
    payload = backend.PredictRequest(model_id=str(model_id), features={column: sample.get(column, 0) for column in feature_columns})
    response = backend.predict(payload, request)
    try:
        data = json.loads(response.body.decode('utf-8'))
        ensure_session(session_id)['prediction_result'] = data.get('prediction_label') or data.get('prediction')
    except Exception:
        pass
    return response


def execute_report_generation(session_id: str, request: Request) -> Any:
    backend = get_backend_module()
    dataset_id = session_dataset_id(session_id)
    frame = session_frame(session_id)
    state = backend.ensure_session_state(dataset_id)
    columns = [
        backend.ColumnInfo(
            name=str(column),
            dtype=str(frame[column].dtype),
            nonNull=int(frame[column].notna().sum()),
            nullCount=int(frame[column].isna().sum()),
            uniqueCount=int(frame[column].nunique(dropna=True)),
            role='numeric' if pd.api.types.is_numeric_dtype(frame[column]) else 'datetime' if pd.api.types.is_datetime64_any_dtype(frame[column]) else 'categorical',
        )
        for column in frame.columns[:80]
    ]
    numeric_columns = [column.name for column in columns if column.role == 'numeric']
    categorical_columns = [column.name for column in columns if column.role == 'categorical']
    session = ensure_session(session_id)
    payload = backend.ReportPayload(
        datasetId=dataset_id,
        sessionId=dataset_id,
        fileName=str(state.get('file_name') or dataset_id),
        totalRows=int(len(frame)),
        previewLoaded=False,
        loadedRowCount=int(len(frame)),
        columns=columns,
        duplicates=int(frame.duplicated().sum()),
        memoryUsage=f'{frame.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB',
        cleaningLogs=[],
        cleaningDone=bool(state.get('cleaning_done')),
        cleanedRowCount=int(len(frame)),
        targetColumn=session.get('target_column') or (preferred_target_column(frame) if numeric_columns else None),
        problemType=str(session.get('problem_type') or 'regression'),
        selectedFeatures=list(session.get('feature_columns') or []),
        selectedModel=session.get('model_type'),
        modelMetrics=None,
        featureImportance=[],
        aiInsights='Generated from the approved IDA Agentic Core workflow.',
        timeSeriesForecastResult=state.get('time_series_result'),
        mlForecastResult=state.get('ml_forecast_result'),
        lossForecast=state.get('loss_forecast_result') or [],
        profitForecast=(state.get('profit_scenarios') or {}).get('baseline', []),
        lossSegments=state.get('loss_segments') or [],
        scenarios=state.get('profit_scenarios'),
        breakevenPeriod=(state.get('breakeven') or {}).get('breakeven_period') if isinstance(state.get('breakeven'), dict) else None,
        reportConfig=backend.ReportConfigPayload(includeLoss=True, includeProfit=True, scenario='baseline'),
        predictionResult=session.get('prediction_result'),
        predictionAnalysis='Automated prediction generated from the approved agentic workflow.' if session.get('prediction_result') is not None else None,
        predictionProbabilities=None,
        predictionHistory=[],
        edaStats=backend.EdaStats(numericColumns=numeric_columns, categoricalColumns=categorical_columns, stats={}, correlations=[]),
    )
    return backend.generate_report(payload, request, format='pdf')


STEP_HANDLERS: dict[str, Callable[[str, Request], Any] | str] = {
    'Data Understanding': execute_data_understanding,
    'EDA': execute_eda,
    'Data Cleaning': execute_data_cleaning,
    'Time Series Forecast': execute_time_series_forecast,
    'ML Forecast': execute_ml_forecast,
    'Loss Forecast': execute_loss_forecast,
    'Profit Forecast': execute_profit_forecast,
    'ML Assistant': execute_ml_assistant,
    'Prediction': execute_prediction,
    'Report Generation': execute_report_generation,
}


@agentic_router.get('/health')
def agentic_health() -> JSONResponse:
    enabled = agentic_enabled()
    db_connected = is_db_connected() if enabled else False
    return JSONResponse(
        content={
            'agentic_enabled': enabled,
            'db_connected': db_connected,
            'db_fallback_active': enabled and not db_connected,
            'providers': {
                'primary': CoreSettings.primary_provider,
                'fallbacks': CoreSettings.fallback_providers,
                'models': CoreSettings.model_switching_map(),
            },
        }
    )


@agentic_router.get('/core/health')
def core_health() -> JSONResponse:
    return JSONResponse(
        content={
            'status': 'ok',
            'providers': {
                'primary': CoreSettings.primary_provider,
                'fallbacks': CoreSettings.fallback_providers,
                'longcat_configured': CoreSettings.provider_configured('longcat'),
                'gemini_configured': CoreSettings.provider_configured('gemini'),
                'groq_configured': CoreSettings.provider_configured('groq'),
                'models': CoreSettings.model_switching_map(),
            },
        }
    )


@agentic_router.post('/core/activity')
async def core_activity(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        core_record_activity(payload if isinstance(payload, dict) else {})
        return JSONResponse(content={'status': 'stored'})
    except json.JSONDecodeError:
        return core_json_error('Invalid JSON body.')
    except Exception as error:
        return core_json_error(str(error), 500)


@agentic_router.post('/core/workflow/suggest')
async def core_workflow_suggest(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        return JSONResponse(content=core_rewrite_json_links(core_create_run(payload if isinstance(payload, dict) else {})))
    except json.JSONDecodeError:
        return core_json_error('Invalid JSON body.')
    except (FileNotFoundError, ValueError) as error:
        return core_json_error(str(error))
    except Exception as error:
        return core_json_error(str(error), 500)


@agentic_router.post('/core/workflow/decision')
async def core_workflow_decision(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        return JSONResponse(content=core_rewrite_json_links(core_record_decision(payload if isinstance(payload, dict) else {})))
    except json.JSONDecodeError:
        return core_json_error('Invalid JSON body.')
    except (FileNotFoundError, ValueError) as error:
        return core_json_error(str(error))
    except Exception as error:
        return core_json_error(str(error), 500)


@agentic_router.post('/core/chat')
async def core_chat(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            payload = {}
        message = str(payload.get('message', '')).strip()
        mode = str(payload.get('mode', 'ask')).strip().lower()
        provider = str(payload.get('provider', 'auto')).strip().lower()
        if not message:
            return core_json_error('Message is required.')
        result = core_chat_respond(message=message, ui_mode=mode, provider=provider)
        return JSONResponse(
            content={
                'answer': result.answer,
                'provider': result.provider,
                'mode': result.mode,
                'fallback_used': result.fallback_used,
            }
        )
    except json.JSONDecodeError:
        return core_json_error('Invalid JSON body.')
    except Exception as error:
        return core_json_error(str(error), 500)


@agentic_router.get('/core/runs/{requested_path:path}')
def core_run_file(requested_path: str) -> Response:
    file_path = (CORE_RUNS_ROOT / requested_path).resolve()
    if not str(file_path).startswith(str(CORE_RUNS_ROOT.resolve())):
        raise HTTPException(status_code=403, detail='Invalid run file path')
    return core_file_response(file_path)


@agentic_router.get('/core')
@agentic_router.get('/core/')
def core_index() -> Response:
    return core_file_response(CORE_UI_ROOT / 'index.html')


@agentic_router.get('/core/{requested_path:path}')
def core_static_file(requested_path: str) -> Response:
    requested = requested_path.strip('/') or 'index.html'
    file_path = (CORE_UI_ROOT / requested).resolve()
    if not str(file_path).startswith(str(CORE_UI_ROOT.resolve())):
        raise HTTPException(status_code=403, detail='Invalid path')
    return core_file_response(file_path)


def prepare_next_recommendation(session_id: str, completed_step: str) -> None:
    session = ensure_session(session_id)
    try:
        current_index = PIPELINE_STEPS.index(completed_step)
    except ValueError:
        return
    for next_step in PIPELINE_STEPS[current_index + 1:]:
        if session['steps'].get(next_step) == 'pending':
            session['recommendations'] = [{
                'step': next_step,
                'reason': 'Previous approved step completed. Review this next pipeline action before execution.',
                'findings': [f'{completed_step} completed successfully.', 'Sequential auto-advance is waiting for explicit approval.'],
            }]
            return
    session['recommendations'] = []


@agentic_router.post('/suggest-next-steps')
def suggest_next_steps(payload: SuggestNextStepsRequest) -> JSONResponse:
    if not agentic_enabled():
        return disabled_response()
    profile = read_dataset_profile(payload.dataset_path)
    session_id = uuid.uuid4().hex
    session = ensure_session(session_id)
    recommendations = build_recommendations(profile)
    session['dataset_path'] = payload.dataset_path
    session['dataset_id'] = payload.dataset_path
    session['profile'] = profile
    session['recommendations'] = recommendations[:1]
    session['updated_at'] = datetime.utcnow().isoformat()
    append_audit_log(session_id, 'recommendations_created', {'profile': profile, 'recommendations': recommendations})
    return JSONResponse(content={'session_id': session_id, 'profile': profile, 'recommendations': recommendations})


@agentic_router.post('/execute-step')
def execute_step(payload: ExecuteStepRequest, request: Request) -> JSONResponse:
    if not agentic_enabled():
        return disabled_response()
    session = ensure_session(payload.session_id)
    handler = STEP_HANDLERS.get(payload.step_name)
    if handler is None:
        raise HTTPException(status_code=404, detail=f'Unknown agentic step: {payload.step_name}')
    if handler == 'not_yet_wired':
        session['steps'][payload.step_name] = 'failed'
        session['updated_at'] = datetime.utcnow().isoformat()
        message = f'{payload.step_name} is not_yet_wired to an existing backend handler.'
        record_agentic_execution(payload.session_id, payload.step_name, 'not_yet_wired', None, message)
        return JSONResponse(content={'status': 'not_yet_wired', 'output_summary': None, 'error': message}, status_code=501)

    session['steps'][payload.step_name] = 'running'
    session['updated_at'] = datetime.utcnow().isoformat()
    try:
        response = handler(payload.session_id, request)
        output_summary = summarize_response(response)
        session['steps'][payload.step_name] = 'completed'
        session['events'].append({
            'step_name': payload.step_name,
            'status': 'completed',
            'approved_by': payload.approved_by,
            'completed_at': datetime.utcnow().isoformat(),
            'output_summary': output_summary,
        })
        write_step_output(payload.session_id, payload.step_name, {'output_summary': output_summary})
        prepare_next_recommendation(payload.session_id, payload.step_name)
        record_agentic_execution(payload.session_id, payload.step_name, 'completed', output_summary, None)
        return JSONResponse(content={'status': 'completed', 'output_summary': output_summary, 'next_recommendations': session['recommendations']})
    except Exception as error:
        session['steps'][payload.step_name] = 'failed'
        session['updated_at'] = datetime.utcnow().isoformat()
        error_message = str(error)
        record_agentic_execution(payload.session_id, payload.step_name, 'failed', None, error_message)
        return JSONResponse(content={'status': 'failed', 'output_summary': None, 'error': error_message}, status_code=500)


@agentic_router.post('/decision')
def record_decision(payload: DecisionRequest) -> JSONResponse:
    if not agentic_enabled():
        return disabled_response()
    if payload.decision not in {'accepted', 'skipped'}:
        raise HTTPException(status_code=400, detail="decision must be 'accepted' or 'skipped'")
    session = ensure_session(payload.session_id)
    if payload.decision == 'skipped':
        session['steps'][payload.step_name] = 'skipped'
        prepare_next_recommendation(payload.session_id, payload.step_name)
    elif payload.decision == 'accepted':
        session['steps'][payload.step_name] = 'completed'
        session['events'].append({
            'step_name': payload.step_name,
            'status': 'completed',
            'approved_by': 'current_user',
            'completed_at': datetime.utcnow().isoformat(),
            'output_summary': payload.reasoning,
        })
        write_step_output(payload.session_id, payload.step_name, {'output_summary': payload.reasoning})
        prepare_next_recommendation(payload.session_id, payload.step_name)
    _decisions.setdefault(payload.session_id, []).append(payload.model_dump())
    write_decision(payload.session_id, payload.step_name, payload.decision, payload.reasoning)
    session['updated_at'] = datetime.utcnow().isoformat()
    return JSONResponse(content={'status': 'recorded', 'steps': session['steps'], 'next_recommendations': session.get('recommendations', [])})


@agentic_router.get('/session/{session_id}/status')
def get_session_status(session_id: str) -> JSONResponse:
    if not agentic_enabled():
        return disabled_response()
    session = ensure_session(session_id)
    return JSONResponse(content={'steps': session['steps'], 'recommendations': session.get('recommendations', []), 'updated_at': session['updated_at']})


@agentic_router.get('/session/{session_id}/report', response_model=None)
def get_session_report(session_id: str) -> HTMLResponse | JSONResponse:
    if not agentic_enabled():
        return disabled_response()
    session = ensure_session(session_id)
    summary = read_session_summary(session_id)
    rows = ''.join(
        f'<tr><td>{html.escape(step)}</td><td>{html.escape(status)}</td></tr>'
        for step, status in session['steps'].items()
    )
    events = ''.join(
        f'<li><strong>{html.escape(str(event.get("step_name", "")))}</strong>: {html.escape(str(event.get("output_summary", "")))}</li>'
        for event in session.get('events', [])
    )
    body = f'''
    <!doctype html>
    <html>
      <head><meta charset="utf-8"><title>IDA Agentic Run Report</title></head>
      <body>
        <h1>IDA Agentic Run Report</h1>
        <p>Session: {html.escape(session_id)}</p>
        <p>Generated: {html.escape(datetime.utcnow().isoformat())}</p>
        <h2>Step Statuses</h2>
        <table border="1" cellspacing="0" cellpadding="6">
          <thead><tr><th>Step</th><th>Status</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        <h2>Execution Events</h2>
        <ul>{events or '<li>No execution events recorded.</li>'}</ul>
        <h2>Artifact Summary</h2>
        <pre>{html.escape(json.dumps(summary, indent=2, default=str))}</pre>
      </body>
    </html>
    '''
    return HTMLResponse(
        content=body,
        headers={'Content-Disposition': f'attachment; filename="agentic_run_{session_id}.html"'},
    )
