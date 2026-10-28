"""The application must not call an external or local language model."""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agentic import agentic_adapter  # noqa: E402


def test_agentic_core_chat_does_not_call_a_model_provider(monkeypatch):
    def forbidden_provider(*_args, **_kwargs):
        raise AssertionError("LLM provider must not be called")

    monkeypatch.setattr(agentic_adapter, "call_provider", forbidden_provider)

    response = agentic_adapter.core_chat_respond(
        "Explain the uploaded data", ui_mode="ask", provider="auto"
    )

    assert response["provider"] == "local"
    assert "disabled" in response["answer"].lower()
    assert "non-llm" in response["answer"].lower()
