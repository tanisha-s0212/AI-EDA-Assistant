"""Fast-training validation must not reuse the reported holdout for selection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402


def test_fast_candidate_selection_uses_a_separate_final_holdout(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(main, "ACTIVITY_DB_AVAILABLE", False)
    main.MODEL_DIR.mkdir()
    rng = np.random.RandomState(11)
    rows = [
        {"signal_a": float(x), "signal_b": float(rng.normal()),
         "outcome": float(2.5 * x + rng.normal(scale=0.3))}
        for x in range(120)
    ]

    response = TestClient(main.app).post("/api/train", json={
        "data": rows, "target_column": "outcome", "feature_columns": ["signal_a", "signal_b"],
        "problem_type": "regression", "model_type": "auto", "training_mode": "fast",
        "test_size": 0.2,
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success", body
    optimization = body["optimization"]
    assert optimization["candidate_selection_method"] == "separate_training_validation_split"
    assert optimization["final_evaluation_independent"] is True
    assert optimization["final_evaluation_rows"] > 0
    assert optimization["selection_rows"] > 0
    assert optimization["selection_rows"] + optimization["final_evaluation_rows"] <= optimization["training_rows_available"]


def test_cross_validation_selection_reports_an_independent_holdout(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(main, "ACTIVITY_DB_AVAILABLE", False)
    main.MODEL_DIR.mkdir()
    rng = np.random.RandomState(19)
    rows = [
        {"measure_x": float(x), "measure_y": float(rng.normal()),
         "result": float(1.75 * x + rng.normal(scale=0.5))}
        for x in range(150)
    ]

    response = TestClient(main.app).post("/api/train", json={
        "data": rows, "target_column": "result", "feature_columns": ["measure_x", "measure_y"],
        "problem_type": "regression", "model_type": "auto", "training_mode": "balanced",
        "test_size": 0.2,
    })

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success", body
    optimization = body["optimization"]
    assert "cross-validation" in optimization["candidate_selection_method"]
    assert optimization["final_evaluation_independent"] is True
    assert optimization["final_evaluation_rows"] == 30
    assert optimization["selection_rows"] == 120
