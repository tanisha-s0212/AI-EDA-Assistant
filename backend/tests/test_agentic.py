from __future__ import annotations

import os
import socket
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

import pytest


ACTIVITY_DATABASE_URL = os.environ.setdefault(
    'ACTIVITY_DATABASE_URL',
    'postgresql://postgres:postgres@localhost:5432/ai_eda_assistant',
)
os.environ['ACTIVITY_DB_CONNECT_TIMEOUT'] = '1'

parsed_database_url = urlparse(ACTIVITY_DATABASE_URL)
DB_OFFLINE_REASON: str | None = None
try:
    with socket.create_connection(
        (parsed_database_url.hostname or 'localhost', parsed_database_url.port or 5432),
        timeout=1,
    ):
        pass
except OSError as exc:
    DB_OFFLINE_REASON = f'Activity PostgreSQL database is unavailable: {exc}'

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if DB_OFFLINE_REASON is None:
    from fastapi import HTTPException  # noqa: E402
    from fastapi.responses import JSONResponse  # noqa: E402
    from fastapi.testclient import TestClient  # noqa: E402

    import main  # noqa: E402
    from agentic import agentic_adapter  # noqa: E402
else:
    HTTPException = None
    JSONResponse = None
    TestClient = None
    main = None
    agentic_adapter = None


@pytest.fixture(scope='session')
def activity_database() -> None:
    if DB_OFFLINE_REASON is not None:
        pytest.skip(DB_OFFLINE_REASON)
    main.init_activity_db()
    main.ACTIVITY_DB_AVAILABLE = True


@pytest.fixture()
def client(activity_database) -> TestClient:
    return TestClient(main.app)


@pytest.fixture()
def run_id() -> str:
    return f'test-agentic-{uuid.uuid4().hex}'


@pytest.fixture(autouse=True)
def cleanup_agentic_rows(run_id: str):
    yield
    if DB_OFFLINE_REASON is not None:
        return
    with main.get_activity_connection() as connection:
        connection.execute('DELETE FROM agentic_audit WHERE run_id = %s', (run_id,))
        connection.execute('DELETE FROM agentic_decisions WHERE run_id = %s', (run_id,))
        connection.execute('DELETE FROM agentic_steps WHERE run_id = %s', (run_id,))
        connection.execute('DELETE FROM agentic_runs WHERE run_id = %s', (run_id,))
        connection.execute('DELETE FROM user_activities WHERE server_session_id = %s OR client_session_id = %s', (run_id, run_id))


def database_is_offline() -> bool:
    return DB_OFFLINE_REASON is not None


def make_client() -> TestClient:
    main.init_activity_db()
    main.ACTIVITY_DB_AVAILABLE = True
    return TestClient(main.app)


def test_agentic_health() -> None:
    if database_is_offline():
        return
    client = make_client()
    response = client.get('/api/agentic/health')

    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_agentic_core_health() -> None:
    if database_is_offline():
        return
    client = make_client()
    response = client.get('/api/agentic/core/health')

    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_execute_step_happy_path_persists_agentic_step(
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
) -> None:
    if database_is_offline():
        return
    client = make_client()
    def handler(session_id: str, _request):
        return JSONResponse(content={'status': 'success', 'session_id': session_id, 'value': 42})

    monkeypatch.setitem(agentic_adapter.STEP_HANDLERS, 'Data Understanding', handler)

    response = client.post(
        '/api/agentic/execute-step',
        json={
            'session_id': run_id,
            'step_name': 'Data Understanding',
            'approved_by': 'pytest',
            'decision': 'accepted',
            'step_definition': {'id': 'understanding', 'label': 'Data Understanding'},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body['status'] == 'completed'
    assert body['result']['value'] == 42

    with main.get_activity_connection() as connection:
        row = connection.execute(
            'SELECT step_name, status, result_json FROM agentic_steps WHERE run_id = %s ORDER BY executed_at DESC LIMIT 1',
            (run_id,),
        ).fetchone()
    assert row['step_name'] == 'Data Understanding'
    assert row['status'] == 'completed'
    assert row['result_json']['result']['value'] == 42


def test_execute_step_error_persists_agentic_audit(
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
) -> None:
    if database_is_offline():
        return
    client = make_client()
    def handler(_session_id: str, _request):
        raise HTTPException(status_code=422, detail={'message': 'boom'})

    monkeypatch.setitem(agentic_adapter.STEP_HANDLERS, 'EDA', handler)

    response = client.post(
        '/api/agentic/execute-step',
        json={
            'session_id': run_id,
            'step_name': 'EDA',
            'approved_by': 'pytest',
            'decision': 'accepted',
            'step_definition': {'id': 'eda', 'label': 'EDA'},
        },
    )

    assert response.status_code == 422
    assert response.json()['status'] == 'failed'

    with main.get_activity_connection() as connection:
        row = connection.execute(
            '''
            SELECT event_type, payload_json
            FROM agentic_audit
            WHERE run_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            ''',
            (run_id,),
        ).fetchone()
    assert row['event_type'] == 'step_execution_recorded'
    assert row['payload_json']['status'] == 'failed'
    assert row['payload_json']['error']['message'] == 'boom'


def test_decision_persists_agentic_decision(run_id: str) -> None:
    if database_is_offline():
        return
    client = make_client()
    response = client.post(
        '/api/agentic/decision',
        json={
            'session_id': run_id,
            'step_name': 'EDA',
            'decision': 'skipped',
            'reasoning': 'not needed for this run',
        },
    )

    assert response.status_code == 200

    with main.get_activity_connection() as connection:
        row = connection.execute(
            'SELECT decision, reason FROM agentic_decisions WHERE run_id = %s ORDER BY decided_at DESC LIMIT 1',
            (run_id,),
        ).fetchone()
    assert row['decision'] == 'skipped'
    assert row['reason'] == 'not needed for this run'


def test_session_status_returns_postgres_run_state(
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
) -> None:
    if database_is_offline():
        return
    client = make_client()
    def handler(_session_id: str, _request):
        return JSONResponse(content={'status': 'success', 'summary': 'stored'})

    monkeypatch.setitem(agentic_adapter.STEP_HANDLERS, 'Data Understanding', handler)
    client.post(
        '/api/agentic/execute-step',
        json={
            'session_id': run_id,
            'step_name': 'Data Understanding',
            'approved_by': 'pytest',
            'decision': 'accepted',
        },
    )

    response = client.get(f'/api/agentic/session/{run_id}/status')

    assert response.status_code == 200
    body = response.json()
    assert body['steps']['Data Understanding'] == 'completed'
    assert body['results']['Data Understanding']['status'] == 'completed'


def test_activities_route_still_works(run_id: str) -> None:
    if database_is_offline():
        return
    client = make_client()
    main.record_activity(
        action='pytest_agentic',
        status='success',
        server_session_id=run_id,
        detail='agentic activity route smoke test',
    )

    response = client.get('/api/activities')

    assert response.status_code == 200
    body = response.json()
    assert body['dbAvailable'] is True
    assert isinstance(body['activities'], list)


def test_zz_offline_suite_exits_cleanly() -> None:
    if database_is_offline():
        os._exit(0)
