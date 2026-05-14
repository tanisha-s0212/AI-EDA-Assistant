from __future__ import annotations

import html
import importlib
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from agentic.run_artifacts import append_audit_log, read_session_summary, write_decision, write_step_output

agentic_router = APIRouter(prefix='/api/agentic')

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


def agentic_enabled() -> bool:
    return os.environ.get('NEXT_PUBLIC_AGENTIC_ENABLED', '').strip().lower() == 'true'


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


def read_csv_profile(dataset_path: str) -> dict[str, Any]:
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
    if path.suffix.lower() != '.csv':
        raise HTTPException(status_code=400, detail='Agentic profiling currently accepts CSV files only.')

    frame = pd.read_csv(path)
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
        findings.append('No missing values detected in the CSV profile.')
    if profile['date_columns']:
        findings.append(f"Detected date-like columns: {', '.join(profile['date_columns'])}.")
    if profile['numeric_columns']:
        findings.append(f"Detected {len(profile['numeric_columns'])} numeric columns for analysis or modeling.")
    return findings


def build_recommendations(profile: dict[str, Any]) -> list[dict[str, Any]]:
    findings = build_findings(profile)
    recommendations = [
        {
            'step': 'Data Understanding',
            'reason': 'Start by confirming column roles, inferred data types, and basic dataset quality.',
            'findings': findings,
        },
        {
            'step': 'EDA',
            'reason': 'Generate exploratory statistics and distributions after the dataset structure is known.',
            'findings': findings,
        },
    ]

    if any(int(value) > 0 for value in profile['nulls'].values()):
        recommendations.append({
            'step': 'Data Cleaning',
            'reason': 'Missing values were detected and should be handled before modeling.',
            'findings': findings,
        })

    if profile['date_columns'] and profile['numeric_columns']:
        recommendations.extend([
            {
                'step': 'Time Series Forecast',
                'reason': 'Date-like and numeric columns are available for time-series forecasting.',
                'findings': findings,
            },
            {
                'step': 'ML Forecast',
                'reason': 'Numeric signals are available for machine-learning forecast workflows.',
                'findings': findings,
            },
        ])

    recommendations.extend([
        {
            'step': 'ML Assistant',
            'reason': 'After analysis and preparation, the existing ML assistant can guide model setup.',
            'findings': findings,
        },
        {
            'step': 'Report Generation',
            'reason': 'A final report should summarize every accepted or skipped step.',
            'findings': findings,
        },
    ])
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


def execute_loss_forecast(session_id: str, request: Request) -> Any:
    backend = get_backend_module()
    payload = backend.ForecastRunRequest(session_id=session_id, forecast_periods=30)
    return backend.run_loss_forecast(payload, request)


def execute_profit_forecast(session_id: str, request: Request) -> Any:
    backend = get_backend_module()
    payload = backend.ForecastRunRequest(session_id=session_id, forecast_periods=30)
    return backend.run_profit_forecast(payload, request)


STEP_HANDLERS: dict[str, Callable[[str, Request], Any] | str] = {
    'Data Understanding': 'not_yet_wired',
    'EDA': 'not_yet_wired',
    'Data Cleaning': 'not_yet_wired',
    'Time Series Forecast': 'not_yet_wired',
    'ML Forecast': 'not_yet_wired',
    'Loss Forecast': execute_loss_forecast,
    'Profit Forecast': execute_profit_forecast,
    'ML Assistant': 'not_yet_wired',
    'Prediction': 'not_yet_wired',
    'Report Generation': 'not_yet_wired',
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
        }
    )


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
    profile = read_csv_profile(payload.dataset_path)
    session_id = uuid.uuid4().hex
    session = ensure_session(session_id)
    recommendations = build_recommendations(profile)
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
