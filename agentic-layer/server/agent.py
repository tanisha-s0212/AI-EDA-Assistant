from __future__ import annotations

import re
from dataclasses import dataclass

from .config import Settings
from .providers import ProviderError, call_provider
from .workspace import (
    format_hits,
    list_workspace_tree,
    read_file_excerpt,
    search_priority_files,
    search_workspace,
    workflow_knowledge,
)


SUPPORTED_MODES = {"ask", "plan", "review", "explain", "search", "fast", "balanced", "deep"}


@dataclass
class AgentResponse:
    answer: str
    provider: str
    mode: str
    fallback_used: bool = False


def _model_mode(ui_mode: str) -> str:
    if ui_mode in {"fast", "balanced", "deep"}:
        return ui_mode
    if ui_mode in {"plan", "review"}:
        return "balanced"
    return Settings.default_mode if Settings.default_mode in {"fast", "balanced", "deep"} else "fast"


def _search_terms(message: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-./ ]+", " ", message)
    stop_words = {
        "what",
        "which",
        "where",
        "when",
        "with",
        "about",
        "application",
        "involved",
        "explain",
        "tell",
        "give",
        "plan",
        "user",
        "questions",
    }
    words = [word for word in cleaned.split() if len(word) > 2 and word.lower() not in stop_words]
    aliases = []
    lowered = message.lower()
    if "forecast" in lowered:
        aliases.extend(["forecast_ts", "forecast_ml", "loss_forecast", "profit_forecast", "TimeSeriesForecastTab", "MlForecastTab", "LossForecastTab", "ProfitForecastTab"])
    if "tab" in lowered or "workflow" in lowered:
        aliases.extend(["TabId", "activeTab", "renderTab", "tabs"])
    return " ".join([*words[:8], *aliases])


def _extract_file_reference(message: str) -> str | None:
    match = re.search(r"([\w./\\-]+\.(?:py|js|jsx|ts|tsx|json|md|html|css|yml|yaml|txt))", message)
    if not match:
        return None
    return match.group(1).replace("\\", "/")


def _workspace_context(message: str, ui_mode: str) -> str:
    sections = [
        "Confirmed application workflow knowledge:\n" + workflow_knowledge(),
        "Workspace tree:\n" + list_workspace_tree(),
    ]

    terms = _search_terms(message)
    if ui_mode == "search" or terms:
        priority_hits = search_priority_files(terms)
        general_hits = search_workspace(terms)
        sections.append("Priority application matches:\n" + format_hits(priority_hits))
        sections.append("General workspace matches:\n" + format_hits(general_hits))

    file_reference = _extract_file_reference(message)
    if file_reference:
        try:
            sections.append(f"Excerpt from {file_reference}:\n{read_file_excerpt(file_reference)}")
        except (OSError, ValueError) as exc:
            sections.append(f"File excerpt unavailable for {file_reference}: {exc}")

    return "\n\n".join(sections)


def _system_prompt(ui_mode: str) -> str:
    return f"""You are the standalone Agentic Layer for this local application workspace.

Current mode: {ui_mode}

Rules:
- You may read and summarize the existing application files to answer questions.
- You are read-only for code changes: do not claim that you changed application files.
- Help the user understand, plan, review, and navigate the existing project.
- Keep recommendations compatible with the existing workflow unless the user explicitly asks for a future integration plan.
- When referencing files, include concise relative paths and line hints when available.
- Never reveal or ask for API keys.
- Prefer confirmed workflow knowledge and file-backed evidence over broad inference.
- Keep chat answers clean: use numbered sections or short paragraphs, and avoid raw Markdown bullet markers unless a list is clearly useful.
"""


def _fallback_answer(message: str, ui_mode: str) -> str:
    context = _workspace_context(message, ui_mode)
    return (
        "The cloud model providers are unavailable for this request, so I am answering from local read-only workspace context.\n\n"
        f"{context}\n\n"
        "You can retry later, switch provider mode, or continue using the local workflow context while the provider issue clears."
    )


def respond(message: str, ui_mode: str = "ask", provider: str = "auto") -> AgentResponse:
    ui_mode = ui_mode if ui_mode in SUPPORTED_MODES else "ask"
    model_mode = _model_mode(ui_mode)
    provider = provider.lower()

    context = _workspace_context(message, ui_mode)
    messages = [
        {"role": "system", "content": _system_prompt(ui_mode)},
        {"role": "user", "content": f"User request:\n{message}\n\nLocal workspace context:\n{context}"},
    ]

    providers: list[str]
    if provider in {"gemini", "groq"}:
        providers = [provider]
    else:
        providers = [Settings.primary_provider, Settings.fallback_provider]

    errors: list[str] = []
    for index, current_provider in enumerate(dict.fromkeys(providers)):
        try:
            answer = call_provider(current_provider, messages, model_mode)
            return AgentResponse(answer=answer, provider=current_provider, mode=model_mode, fallback_used=index > 0)
        except ProviderError as exc:
            errors.append(f"{current_provider}: {exc}")

    if errors:
        answer = (
            _fallback_answer(message, ui_mode)
            + "\n\nProvider status summary:\n"
            + "\n".join(f"- {error}" for error in errors)
        )
    else:
        answer = _fallback_answer(message, ui_mode)

    return AgentResponse(answer=answer, provider="local", mode=model_mode, fallback_used=False)
