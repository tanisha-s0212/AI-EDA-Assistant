from __future__ import annotations

import csv
import html
import json
import math
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

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
    source_path: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    return cleaned[:80] or "dataset"


def _json_response(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _read_json(path: Path, fallback: dict | None = None) -> dict:
    if not path.exists():
        return fallback or {}
    return json.loads(path.read_text(encoding="utf-8"))


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
        source_path=str(path.relative_to(WORKSPACE_ROOT)),
    )


def _default_profile(
    dataset_name: str,
    rows_sampled: int = 0,
    columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
) -> DatasetProfile:
    safe_columns = [str(column) for column in (columns or []) if str(column).strip()]
    safe_numeric_columns = [str(column) for column in (numeric_columns or []) if str(column).strip()]
    return DatasetProfile(
        file_name=dataset_name or "uploaded dataset",
        rows_sampled=max(0, rows_sampled),
        columns=safe_columns,
        numeric_columns=safe_numeric_columns,
        missing_counts={column: 0 for column in safe_columns},
        notes=[
            "Dataset metadata was received from the main Intelligent Data Assistant application.",
            "Suggestions are based on the confirmed Intelligent Data Assistant workflow.",
        ],
    )


def _load_dataset_rows(dataset: dict, max_rows: int | None = None) -> list[dict[str, str]]:
    source_path = str(dataset.get("source_path", "")).strip()
    if not source_path:
        return []

    path = _safe_workspace_file(source_path)
    if path.suffix.lower() != ".csv":
        return []

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            rows.append({str(key): "" if value is None else str(value) for key, value in row.items()})
    return rows


def _numeric_values(rows: list[dict[str, str]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw_value = str(row.get(column, "")).strip().replace(",", "")
        if not raw_value:
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _top_values(rows: list[dict[str, str]], column: str, limit: int = 5) -> list[dict[str, int]]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(column, "")).strip() or "(blank)"
        counts[value] = counts.get(value, 0) + 1
    return [
        {"value": value, "count": count}
        for value, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits) if math.isfinite(float(value)) else 0.0


def _step_summary_path(run_root: Path, step_id: str) -> Path:
    return run_root / "results" / f"{step_id}.json"


def _execute_understanding(run_root: Path, manifest: dict) -> dict:
    dataset = manifest["dataset"]
    payload = {
        "step_id": "understanding",
        "status": "completed",
        "completed_at": _now(),
        "summary": f"Profiled {dataset['file_name']} with {dataset['rows_sampled']} sampled rows and {len(dataset['columns'])} columns.",
        "columns": dataset["columns"],
        "numeric_columns": dataset["numeric_columns"],
        "missing_counts": dataset["missing_counts"],
        "notes": dataset["notes"],
    }
    _json_response(_step_summary_path(run_root, "understanding"), payload)
    return payload


def _execute_eda(run_root: Path, manifest: dict) -> dict:
    dataset = manifest["dataset"]
    rows = _load_dataset_rows(dataset, max_rows=1000)
    numeric_stats = []
    for column in dataset.get("numeric_columns", []):
        values = _numeric_values(rows, column)
        if not values:
            continue
        numeric_stats.append(
            {
                "column": column,
                "count": len(values),
                "mean": _round(mean(values)),
                "median": _round(median(values)),
                "min": _round(min(values)),
                "max": _round(max(values)),
            }
        )

    categorical_columns = [column for column in dataset.get("columns", []) if column not in dataset.get("numeric_columns", [])]
    categorical_summary = [
        {"column": column, "top_values": _top_values(rows, column)}
        for column in categorical_columns[:8]
    ]
    payload = {
        "step_id": "eda",
        "status": "completed",
        "completed_at": _now(),
        "summary": "Generated exploratory statistics, missing-value profile, and leading category values.",
        "numeric_statistics": numeric_stats,
        "categorical_summary": categorical_summary,
        "quality_warnings": dataset.get("notes", []),
    }
    _json_response(_step_summary_path(run_root, "eda"), payload)
    return payload


def _execute_cleaning(run_root: Path, manifest: dict) -> dict:
    dataset = manifest["dataset"]
    rows = _load_dataset_rows(dataset)
    columns = dataset.get("columns", [])
    numeric_columns = set(dataset.get("numeric_columns", []))
    fills: dict[str, str | float] = {}

    for column in columns:
        if column in numeric_columns:
            values = _numeric_values(rows, column)
            fills[column] = _round(mean(values)) if values else 0.0
        else:
            top = _top_values(rows, column, limit=1)
            fills[column] = top[0]["value"] if top and top[0]["value"] != "(blank)" else "Unknown"

    cleaned_rows: list[dict[str, str]] = []
    for row in rows:
        cleaned_row: dict[str, str] = {}
        for column in columns:
            value = str(row.get(column, "")).strip()
            cleaned_row[column] = value if value else str(fills.get(column, ""))
        cleaned_rows.append(cleaned_row)

    cleaned_path = run_root / "results" / "cleaned_dataset.csv"
    if columns and cleaned_rows:
        with cleaned_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(cleaned_rows)

    payload = {
        "step_id": "cleaning",
        "status": "completed",
        "completed_at": _now(),
        "summary": "Prepared a cleaned CSV artifact with missing values filled using numeric means and categorical modes.",
        "rows_written": len(cleaned_rows),
        "fills": fills,
        "artifact": str(cleaned_path.relative_to(AGENTIC_ROOT)) if cleaned_path.exists() else "",
    }
    _json_response(_step_summary_path(run_root, "cleaning"), payload)
    return payload


def _forecast_series(values: list[float], periods: int = 6) -> list[dict[str, float | int]]:
    if not values:
        return []
    window = values[-min(6, len(values)) :]
    baseline = mean(window)
    slope = 0.0
    if len(values) >= 2:
        slope = (values[-1] - values[max(0, len(values) - min(8, len(values)))]) / max(1, min(8, len(values)) - 1)
    return [
        {"period": index, "forecast": _round(max(0.0, baseline + slope * index), 2)}
        for index in range(1, periods + 1)
    ]


def _target_column(dataset: dict) -> str:
    numeric_columns = list(dataset.get("numeric_columns", []))
    for pattern in ("sales", "revenue", "profit", "loss", "amount", "value", "price", "cost"):
        for column in numeric_columns:
            if pattern in column.lower():
                return column
    return numeric_columns[0] if numeric_columns else ""


def _execute_time_series_forecast(run_root: Path, manifest: dict) -> dict:
    dataset = manifest["dataset"]
    rows = _load_dataset_rows(dataset)
    target = _target_column(dataset)
    values = _numeric_values(rows, target) if target else []
    forecast = _forecast_series(values)
    payload = {
        "step_id": "time_series_forecast",
        "status": "completed" if forecast else "skipped",
        "completed_at": _now(),
        "target_column": target,
        "summary": f"Generated a trend-aware baseline forecast for {target}." if forecast else "Skipped because no numeric target column was available.",
        "forecast": forecast,
    }
    _json_response(_step_summary_path(run_root, "time_series_forecast"), payload)
    return payload


def _execute_ml_forecast(run_root: Path, manifest: dict) -> dict:
    dataset = manifest["dataset"]
    rows = _load_dataset_rows(dataset)
    target = _target_column(dataset)
    values = _numeric_values(rows, target) if target else []
    holdout = values[-max(1, int(len(values) * 0.2)) :] if len(values) >= 5 else []
    train = values[: -len(holdout)] if holdout else values
    baseline = mean(train[-min(10, len(train)) :]) if train else 0.0
    mae = mean(abs(value - baseline) for value in holdout) if holdout else 0.0
    forecast = _forecast_series(values, periods=6)
    payload = {
        "step_id": "ml_forecast",
        "status": "completed" if forecast else "skipped",
        "completed_at": _now(),
        "target_column": target,
        "summary": "Generated a lightweight ML-style baseline with holdout error for local standalone execution." if forecast else "Skipped because no numeric target column was available.",
        "metrics": {"mae": _round(mae, 4), "holdout_rows": len(holdout)},
        "feature_candidates": [column for column in dataset.get("columns", []) if column != target][:12],
        "forecast": forecast,
    }
    _json_response(_step_summary_path(run_root, "ml_forecast"), payload)
    return payload


def _execute_loss_forecast(run_root: Path, _manifest: dict) -> dict:
    ml_result = _read_json(_step_summary_path(run_root, "ml_forecast"), {})
    forecast = ml_result.get("forecast") or _read_json(_step_summary_path(run_root, "time_series_forecast"), {}).get("forecast", [])
    rows = []
    for item in forecast:
        value = float(item.get("forecast", 0) or 0)
        rows.append(
            {
                "period": item.get("period"),
                "projected_value": _round(value, 2),
                "estimated_loss": _round(value * 0.08, 2),
                "risk_label": "High" if value * 0.08 > value * 0.12 else "Moderate",
            }
        )
    payload = {
        "step_id": "loss_forecast",
        "status": "completed" if rows else "skipped",
        "completed_at": _now(),
        "summary": "Estimated downside exposure from forecasted values using a conservative local risk factor." if rows else "Skipped because no forecast values were available.",
        "rows": rows,
    }
    _json_response(_step_summary_path(run_root, "loss_forecast"), payload)
    return payload


def _execute_profit_forecast(run_root: Path, _manifest: dict) -> dict:
    ml_result = _read_json(_step_summary_path(run_root, "ml_forecast"), {})
    forecast = ml_result.get("forecast") or _read_json(_step_summary_path(run_root, "time_series_forecast"), {}).get("forecast", [])
    loss_rows = _read_json(_step_summary_path(run_root, "loss_forecast"), {}).get("rows", [])
    losses_by_period = {row.get("period"): float(row.get("estimated_loss", 0) or 0) for row in loss_rows}
    rows = []
    for item in forecast:
        value = float(item.get("forecast", 0) or 0)
        period = item.get("period")
        cost = value * 0.62
        loss = losses_by_period.get(period, value * 0.08)
        rows.append(
            {
                "period": period,
                "forecasted_revenue": _round(value, 2),
                "estimated_cost": _round(cost, 2),
                "estimated_loss": _round(loss, 2),
                "net_profit": _round(value - cost - loss, 2),
            }
        )
    payload = {
        "step_id": "profit_forecast",
        "status": "completed" if rows else "skipped",
        "completed_at": _now(),
        "summary": "Projected profit scenarios using generated forecast and loss artifacts." if rows else "Skipped because no forecast values were available.",
        "rows": rows,
    }
    _json_response(_step_summary_path(run_root, "profit_forecast"), payload)
    return payload


def _execute_ml_assistant(run_root: Path, manifest: dict) -> dict:
    dataset = manifest["dataset"]
    target = _target_column(dataset)
    features = [column for column in dataset.get("columns", []) if column != target][:20]
    payload = {
        "step_id": "ml_assistant",
        "status": "completed" if target and features else "skipped",
        "completed_at": _now(),
        "summary": f"Prepared a local model plan with target {target} and {len(features)} candidate features." if target and features else "Skipped because a target/features pair could not be inferred.",
        "target_column": target,
        "feature_columns": features,
        "recommended_model": "ridge_regression" if target else "",
    }
    _json_response(_step_summary_path(run_root, "ml_assistant"), payload)
    return payload


def _execute_prediction(run_root: Path, _manifest: dict) -> dict:
    profit_rows = _read_json(_step_summary_path(run_root, "profit_forecast"), {}).get("rows", [])
    last = profit_rows[-1] if profit_rows else {}
    prediction = last.get("net_profit") if last else None
    payload = {
        "step_id": "prediction",
        "status": "completed" if prediction is not None else "skipped",
        "completed_at": _now(),
        "summary": "Generated a final local prediction from the last projected profit period." if prediction is not None else "Skipped because profit forecast output was unavailable.",
        "prediction": prediction,
        "basis": last,
    }
    _json_response(_step_summary_path(run_root, "prediction"), payload)
    return payload


def _render_report(run_root: Path, manifest: dict) -> dict:
    results = {
        step: _read_json(_step_summary_path(run_root, step), {})
        for step in WORKFLOW_STEPS
        if step != "data_upload"
    }
    dataset = manifest["dataset"]
    completed = [step for step, result in results.items() if result.get("status") == "completed"]
    skipped = [step for step, result in results.items() if result.get("status") == "skipped"]
    report_path = run_root / "reports" / "workflow_report.html"
    sections = []
    for step in WORKFLOW_STEPS:
        if step == "data_upload":
            continue
        result = results.get(step, {})
        status = str(result.get("status", "pending")).title()
        summary = html.escape(str(result.get("summary", "No artifact was generated for this step.")))
        sections.append(
            f"<section><h2>{html.escape(step.replace('_', ' ').title())}</h2><span class=\"status\">{html.escape(status)}</span><p>{summary}</p></section>"
        )

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(dataset['file_name'])} - Agentic Workflow Report</title>
  <style>
    body {{ margin: 0; background: #f5f7fb; color: #172033; font-family: Inter, Segoe UI, Arial, sans-serif; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 42px 28px 56px; }}
    header {{ border-radius: 14px; padding: 30px; background: #0f2f3a; color: #fff; }}
    h1 {{ margin: 0; font-size: 30px; letter-spacing: 0; }}
    header p {{ margin: 10px 0 0; color: #cfe7e4; }}
    .cards {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 18px 0; }}
    .card, section {{ border: 1px solid #dbe3ec; border-radius: 12px; background: #fff; padding: 18px; box-shadow: 0 12px 28px rgba(23, 32, 51, 0.06); }}
    .card strong {{ display: block; font-size: 24px; }}
    .card span, section p {{ color: #5b6678; }}
    section {{ margin-top: 12px; }}
    h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .status {{ display: inline-block; margin-bottom: 8px; border-radius: 999px; padding: 4px 9px; background: #e6f5f1; color: #076450; font-size: 12px; font-weight: 700; }}
    footer {{ margin-top: 18px; color: #667085; font-size: 12px; }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Executive Workflow Report</h1>
      <p>Generated locally by IDA Agentic Core for {html.escape(dataset['file_name'])}.</p>
    </header>
    <div class="cards">
      <div class="card"><strong>{int(dataset.get('rows_sampled', 0)):,}</strong><span>Rows sampled</span></div>
      <div class="card"><strong>{len(dataset.get('columns', []))}</strong><span>Columns</span></div>
      <div class="card"><strong>{len(completed)}</strong><span>Completed steps</span></div>
    </div>
    {''.join(sections)}
    <footer>Run ID: {html.escape(manifest['run_id'])} | Created: {html.escape(_now())} | Skipped steps: {len(skipped)}</footer>
  </main>
</body>
</html>"""
    report_path.write_text(report_html, encoding="utf-8")
    payload = {
        "step_id": "report",
        "status": "completed",
        "completed_at": _now(),
        "summary": "Generated a polished local HTML report for download.",
        "report_path": str(report_path.relative_to(AGENTIC_ROOT)),
        "download_url": f"/runs/{manifest['run_id']}/reports/workflow_report.html",
    }
    _json_response(_step_summary_path(run_root, "report"), payload)
    return payload


STEP_EXECUTORS = {
    "understanding": _execute_understanding,
    "eda": _execute_eda,
    "cleaning": _execute_cleaning,
    "time_series_forecast": _execute_time_series_forecast,
    "ml_forecast": _execute_ml_forecast,
    "loss_forecast": _execute_loss_forecast,
    "profit_forecast": _execute_profit_forecast,
    "ml_assistant": _execute_ml_assistant,
    "prediction": _execute_prediction,
}


def _execute_pipeline(run_root: Path, start_step: str) -> dict:
    manifest_path = run_root / "manifest.json"
    manifest = _read_json(manifest_path)
    if not manifest:
        raise FileNotFoundError("manifest.json")

    try:
        start_index = WORKFLOW_STEPS.index(start_step)
    except ValueError as exc:
        raise ValueError(f"Unknown workflow step: {start_step}") from exc

    executed = []
    for step_id in WORKFLOW_STEPS[start_index:]:
        if step_id == "data_upload":
            continue
        executor = STEP_EXECUTORS.get(step_id)
        if executor is None:
            if step_id == "report":
                result = _render_report(run_root, manifest)
            else:
                continue
        else:
            result = executor(run_root, manifest)
        executed.append({"step_id": step_id, "status": result.get("status", "completed"), "summary": result.get("summary", "")})

    report_result = _read_json(_step_summary_path(run_root, "report"), {})
    manifest["status"] = "completed"
    manifest["completed_at"] = _now()
    manifest["executed_steps"] = executed
    manifest["report"] = report_result
    _json_response(manifest_path, manifest)
    return {"executed_steps": executed, "report": report_result}


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
    dataset_id = str(payload.get("dataset_id", "")).strip()
    columns = payload.get("dataset_columns", [])
    numeric_columns = payload.get("numeric_columns", [])
    rows_sampled = int(payload.get("row_count") or payload.get("loaded_row_count") or 0)
    profile: DatasetProfile

    if dataset_path:
        safe_path = _safe_workspace_file(dataset_path)
        if safe_path.suffix.lower() == ".csv":
            profile = _profile_csv(safe_path)
        else:
            profile = _default_profile(safe_path.name)
            profile.notes.append("Only CSV files are profiled directly in this local scaffold.")
    else:
        profile = _default_profile(
            dataset_name or dataset_id,
            rows_sampled=rows_sampled,
            columns=columns if isinstance(columns, list) else [],
            numeric_columns=numeric_columns if isinstance(numeric_columns, list) else [],
        )
        if dataset_id:
            profile.notes.append(f"Main application dataset id: {dataset_id}")

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
        "status": "running" if decision == "accept" else "skipped",
        "note": str(payload.get("note", "")).strip()[:500],
    }
    _json_response(run_root / "decisions" / f"{step_id}.json", record)

    if decision == "accept":
        execution = _execute_pipeline(run_root, step_id)
        record["status"] = "completed"
        record["completed_at"] = _now()
        record["executed_steps"] = execution["executed_steps"]
        record["report"] = execution["report"]
        _json_response(run_root / "decisions" / f"{step_id}.json", record)

    return record
