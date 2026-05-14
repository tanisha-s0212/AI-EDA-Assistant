from __future__ import annotations

import json
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl

BACKEND_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = BACKEND_DIR / 'runs'
VALID_DECISIONS = {'accepted', 'skipped'}


def safe_path_part(value: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', value.strip())
    return cleaned.strip('._') or 'unnamed'


def session_dir(session_id: str) -> Path:
    path = RUNS_DIR / safe_path_part(session_id)
    (path / 'step_outputs').mkdir(parents=True, exist_ok=True)
    return path


def decisions_path(session_id: str) -> Path:
    return session_dir(session_id) / 'decisions.json'


def audit_log_path(session_id: str) -> Path:
    return session_dir(session_id) / 'audit_log.jsonl'


def step_output_dir(session_id: str, step_name: str) -> Path:
    path = session_dir(session_id) / 'step_outputs' / safe_path_part(step_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def locked_path(target_path: Path) -> Iterator[None]:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target_path.with_name(f'{target_path.name}.lock')
    with lock_path.open('a+b') as lock_file:
        if sys.platform == 'win32':
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if sys.platform == 'win32':
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return default


def write_json_file(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f'{path.name}.tmp')
    temp_path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding='utf-8')
    temp_path.replace(path)


def write_decision(session_id: str, step_name: str, decision: str, reasoning: str) -> None:
    if decision not in VALID_DECISIONS:
        raise ValueError("decision must be 'accepted' or 'skipped'")

    path = decisions_path(session_id)
    entry = {
        'step_name': step_name,
        'decision': decision,
        'reasoning': reasoning,
        'decided_at': datetime.utcnow().isoformat(),
    }
    with locked_path(path):
        decisions = read_json_file(path, {})
        decisions[step_name] = entry
        write_json_file(path, decisions)
    append_audit_log(session_id, 'decision_recorded', entry)


def read_decisions(session_id: str) -> dict[str, Any]:
    path = decisions_path(session_id)
    with locked_path(path):
        decisions = read_json_file(path, {})
    return decisions if isinstance(decisions, dict) else {}


def append_audit_log(session_id: str, event_type: str, payload: dict[str, Any]) -> None:
    path = audit_log_path(session_id)
    event = {
        'event_type': event_type,
        'payload': payload,
        'recorded_at': datetime.utcnow().isoformat(),
    }
    with locked_path(path):
        with path.open('a', encoding='utf-8') as file:
            file.write(json.dumps(event, sort_keys=True, default=str))
            file.write('\n')


def write_step_output(session_id: str, step_name: str, data: Any) -> None:
    path = step_output_dir(session_id, step_name) / 'output.json'
    with locked_path(path):
        write_json_file(path, data)
    append_audit_log(
        session_id,
        'step_output_written',
        {
            'step_name': step_name,
            'output_path': str(path),
        },
    )


def read_audit_events(session_id: str) -> list[dict[str, Any]]:
    path = audit_log_path(session_id)
    with locked_path(path):
        if not path.exists():
            return []
        lines = path.read_text(encoding='utf-8').splitlines()

    events = []
    for line in lines:
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({'event_type': 'corrupt_audit_line', 'raw': line})
    return events


def read_step_outputs(session_id: str) -> dict[str, Any]:
    outputs_root = session_dir(session_id) / 'step_outputs'
    outputs: dict[str, Any] = {}
    if not outputs_root.exists():
        return outputs

    for step_dir in outputs_root.iterdir():
        if not step_dir.is_dir():
            continue
        output_path = step_dir / 'output.json'
        with locked_path(output_path):
            outputs[step_dir.name] = read_json_file(output_path, None)
    return outputs


def read_session_summary(session_id: str) -> dict[str, Any]:
    path = session_dir(session_id)
    return {
        'session_id': session_id,
        'run_dir': str(path),
        'decisions': read_decisions(session_id),
        'audit_log': read_audit_events(session_id),
        'step_outputs': read_step_outputs(session_id),
    }
