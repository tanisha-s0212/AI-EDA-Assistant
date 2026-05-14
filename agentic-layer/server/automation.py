from __future__ import annotations

import csv
import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from .config import AGENTIC_ROOT, WORKSPACE_ROOT


RUNS_ROOT = AGENTIC_ROOT / "runs"

WORKFLOW_STEPS = [
    "data_upload",
    "understanding",
    "eda",
    "cleaning",
    "time_series_forecast",
    "ml_forecast",
    "loss_forecast",
    "profit_forecast",
    "ml_assistant",
    "prediction",
    "report",
]


@dataclass
class DatasetProfile:
    file_name: str
    rows_sampled: int
    columns: list[str]
    numeric_columns: list[str]
    missing_counts: dict[str, int]
    notes: list[str]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    return cleaned[:80] or "dataset"


def _json_response(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _safe_workspace_file(relative_path: str) -> Path:
    candidate = (WORKSPACE_ROOT / relative_path).resolve()
    if not str(candidate).startswith(str(WORKSPACE_ROOT.resolve())):
        raise ValueError("Dataset path must stay inside the workspace.")
    if candidate == AGENTIC_ROOT / ".env":
        raise ValueError("The agentic .env file cannot be used as a dataset.")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(relative_path)
    return candidate


def _profile_csv(path: Path, max_rows: int = 500) -> DatasetProfile:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = list(reader.fieldnames or [])
        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            rows.append(row)

    missing_counts = {column: 0 for column in columns}
    numeric_values: dict[str, list[float]] = {column: [] for column in columns}

    for row in rows:
        for column in columns:
            value = (row.get(column) or "").strip()
            if value == "":
                missing_counts[column] += 1
                continue
            try:
                numeric_values[column].append(float(value))
            except ValueError:
                continue

    numeric_columns = [
        column
        for column, values in numeric_values.items()
        if rows and len(values) >= max(1, int(len(rows) * 0.65))
    ]

    notes: list[str] = []
    if not columns:
        notes.append("No header columns were detected in the uploaded file.")
    if any(count > 0 for count in missing_counts.values()):
        notes.append("Missing values were detected and should be reviewed before forecasting.")
    if len(numeric_columns) >= 2:
        averages = {column: round(mean(numeric_values[column]), 4) for column in numeric_columns if numeric_values[column]}
        strongest = sorted(averages.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
        if strongest:
            notes.append("Numeric signals available for EDA and forecasting: " + ", ".join(name for name, _value in strongest))
    if not numeric_columns:
        notes.append("No strong numeric columns were detected in the sample; model steps may need manual target selection.")

    return DatasetProfile(
        file_name=path.name,
        rows_sampled=len(rows),
        columns=columns,
        numeric_columns=numeric_columns,
        missing_counts=missing_counts,
        notes=notes,
    )


def _default_profile(dataset_name: str) -> DatasetProfile:
    return DatasetProfile(
        file_name=dataset_name or "uploaded dataset",
        rows_sampled=0,
        columns=[],
        numeric_columns=[],
        missing_counts={},
        notes=[
            "Dataset content was not directly available to the agentic layer.",
            "Suggestions are based on the confirmed Intelligent Data Assistant workflow.",
        ],
    )


def _suggestions(profile: DatasetProfile) -> list[dict]:
    has_missing = any(count > 0 for count in profile.missing_counts.values())
    has_numeric = bool(profile.numeric_columns)
    return [
        {
            "id": "understanding",
            "title": "Profile Dataset",
            "recommended_action": "Accept and Continue",
            "reason": "Confirm columns, row sample size, missing values, and likely target fields before analysis.",
        },
        {
            "id": "eda",
            "title": "Run EDA Summary",
            "recommended_action": "Accept and Continue",
            "reason": "Generate distributions, missing-value tables, correlation candidates, and quality warnings.",
        },
        {
            "id": "cleaning",
            "title": "Prepare Clean Dataset",
            "recommended_action": "Accept and Continue" if has_missing else "Optional",
            "reason": "Handle missing values, normalize data types, and persist a cleaned version for downstream models.",
        },
        {
            "id": "time_series_forecast",
            "title": "Evaluate Time Series Forecast",
            "recommended_action": "Accept and Continue" if has_numeric else "Skip Until Target Is Selected",
            "reason": "Use numeric/date-like fields to create forecast candidates and store metrics.",
        },
        {
            "id": "ml_forecast",
            "title": "Evaluate ML Forecast",
            "recommended_action": "Accept and Continue" if has_numeric else "Skip Until Target Is Selected",
            "reason": "Train baseline ML forecast candidates and compare performance with the time-series run.",
        },
        {
            "id": "loss_forecast",
            "title": "Estimate Loss Forecast",
            "recommended_action": "Accept and Continue",
            "reason": "Use previous forecast outputs to estimate downside scenarios where the workflow supports it.",
        },
        {
            "id": "profit_forecast",
            "title": "Estimate Profit Forecast",
            "recommended_action": "Accept and Continue",
            "reason": "Use loss and forecast outputs to estimate profit scenarios and explain assumptions.",
        },
        {
            "id": "report",
            "title": "Generate Complete Flow Report",
            "recommended_action": "Accept and Continue",
            "reason": "Package performed steps, assumptions, results, model metrics, and download-ready summaries.",
        },
    ]


def create_run(payload: dict) -> dict:
    dataset_path = str(payload.get("dataset_path", "")).strip()
    dataset_name = str(payload.get("dataset_name", "")).strip()
    profile: DatasetProfile

    if dataset_path:
        safe_path = _safe_workspace_file(dataset_path)
        if safe_path.suffix.lower() == ".csv":
            profile = _profile_csv(safe_path)
        else:
            profile = _default_profile(safe_path.name)
            profile.notes.append("Only CSV files are profiled directly in this local scaffold.")
    else:
        profile = _default_profile(dataset_name)

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{_safe_id(profile.file_name)}-{uuid.uuid4().hex[:6]}"
    run_root = RUNS_ROOT / run_id
    for folder in ("models", "performance", "results", "reports", "decisions"):
        (run_root / folder).mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "created_at": _now(),
        "application": "Intelligent Data Assistant",
        "agent": "IDA Agentic Core",
        "workflow": WORKFLOW_STEPS,
        "dataset": asdict(profile),
        "suggestions": _suggestions(profile),
        "status": "suggested",
    }
    _json_response(run_root / "manifest.json", manifest)
    _json_response(run_root / "results" / "dataset_profile.json", asdict(profile))
    return manifest


def record_decision(payload: dict) -> dict:
    run_id = _safe_id(str(payload.get("run_id", "")).strip())
    step_id = _safe_id(str(payload.get("step_id", "")).strip())
    decision = str(payload.get("decision", "")).strip().lower()

    if not run_id or not step_id:
        raise ValueError("run_id and step_id are required.")
    if decision not in {"accept", "skip"}:
        raise ValueError("decision must be accept or skip.")

    run_root = (RUNS_ROOT / run_id).resolve()
    if not str(run_root).startswith(str(RUNS_ROOT.resolve())):
        raise ValueError("Invalid run_id.")
    if not run_root.exists():
        raise FileNotFoundError(run_id)

    record = {
        "run_id": run_id,
        "step_id": step_id,
        "decision": decision,
        "recorded_at": _now(),
        "status": "queued" if decision == "accept" else "skipped",
        "note": str(payload.get("note", "")).strip()[:500],
    }
    _json_response(run_root / "decisions" / f"{step_id}.json", record)

    if decision == "accept":
        artifact = {
            "step_id": step_id,
            "created_at": _now(),
            "status": "prepared",
            "message": "Step accepted by the user. Main workflow execution can consume this artifact without exposing API keys.",
        }
        _json_response(run_root / "results" / f"{step_id}.json", artifact)

    return record
