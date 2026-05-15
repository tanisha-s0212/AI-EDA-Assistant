# IDA Agentic Core

Standalone local agentic assistant core for the AI EDA Assistant workspace.

This layer is intentionally separate from the existing application flow. It runs as its own local browser page and does not modify the current backend, frontend, infrastructure, routes, or workflow.

## What It Does

- Provides the IDA Agentic Core multi-panel workspace.
- Uses LongCat as the primary LLM provider, then Gemini, then Groq.
- Reads the workspace structure for context.
- Searches project text for relevant files and snippets.
- Suggests secure next steps after dataset upload, executes accepted standalone workflow steps, and stores accepted/skipped automation decisions under `agentic-layer/runs/`.
- Generates local artifacts for understanding, EDA, cleaning, forecasts, model planning, prediction, and a download-ready HTML workflow report.
- Helps with explanation, planning, review, and navigation.
- Includes focus modes for Ask, Plan, Review, Explain, and Search workflows.
- Includes prompt shortcuts, activity feed, session reset, answer copy, and workflow-context insertion.

## Current Boundary

- It does not edit application files outside `agentic-layer/`.
- It does not change the existing login-to-report workflow.
- It provides `ui/launcher.js` for a bottom-right main-application launcher when the main app is ready to include it.
- Standalone automation uses local deterministic CSV analysis. The integrated React workspace can still call the main FastAPI workflow for richer app-native execution.

## Run Locally

From the repository root:

```powershell
python agentic-layer/server/app.py
```

Then open:

```text
http://127.0.0.1:5055
```

## Environment

Create a local `.env` file in this folder:

```text
agentic-layer/.env
```

Recommended configuration:

```env
AGENTIC_HOST=127.0.0.1
AGENTIC_PORT=5055
AGENTIC_LOG_LEVEL=info

LLM_PRIMARY_PROVIDER=longcat
LLM_FALLBACK_PROVIDERS=gemini,groq
LLM_DEFAULT_MODE=fast

LONGCAT_BASE_URL=https://api.longcat.chat/openai/v1
LONGCAT_FAST_MODEL=LongCat-Flash-Chat
LONGCAT_BALANCED_MODEL=LongCat-Flash-Chat
LONGCAT_DEEP_MODEL=LongCat-Flash-Thinking

GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_FAST_MODEL=gemini-2.5-flash-lite
GEMINI_BALANCED_MODEL=gemini-2.5-flash
GEMINI_DEEP_MODEL=gemini-2.5-pro

GROQ_API_KEY=your_groq_api_key_here
GROQ_FALLBACK_MODEL=llama-3.3-70b-versatile
GROQ_FAST_MODEL=llama-3.1-8b-instant

LLM_TEMPERATURE=0.2
LLM_MAX_OUTPUT_TOKENS=4096
LLM_REQUEST_TIMEOUT_SECONDS=60
```

The `.env` file is ignored by Git.

## Provider Behavior

Default provider mode is `Auto`.

```text
LongCat -> Gemini -> Groq -> local workspace context
```

If LongCat fails or is unavailable, the server tries Gemini, then Groq. If all hosted providers are unavailable, the layer still returns local workspace tree/search context.

## Main App Launcher

The agentic layer stays separate from the main Intelligent Data Assistant flow. When you are ready to attach the bottom-right launcher to the main app, include this script from the running agentic server:

```html
<script src="http://127.0.0.1:5055/launcher.js"></script>
```

The launcher opens the IDA Agentic Core workspace with a return URL. The workspace includes a `Back to Application` button that navigates back without changing the main workflow.

## Automation API

The agentic layer exposes local endpoints for upload-aware orchestration:

```text
POST /api/workflow/suggest
POST /api/workflow/decision
```

Suggestions use the confirmed flow:

```text
login -> data upload -> understanding -> EDA -> cleaning -> time series forecast -> ML forecast -> loss forecast -> profit forecast -> ML assistant -> prediction -> report
```

Accepted or skipped steps are stored under:

```text
agentic-layer/runs/<run-id>/
```

Each run has separate `models`, `performance`, `results`, `reports`, and `decisions` folders so generated artifacts can be consumed by the main application later without exposing API keys.

When a user clicks `Accept and Continue`, the standalone runner executes the accepted step and the remaining local workflow steps in order, then writes:

```text
agentic-layer/runs/<run-id>/reports/workflow_report.html
```

The UI exposes this as a local `Download local report` link after execution completes.

## Modes

- `Ask`: general project questions.
- `Plan`: implementation planning without editing files.
- `Review`: code review style risk analysis.
- `Explain`: file or module explanation.
- `Search`: project search and navigation.

## Interface Features

- IDA Agentic Core console with a professional workspace layout.
- Focus mode selector for task intent.
- Prompt library for workflow, forecast, report, and review tasks.
- Workspace context panel for frontend, backend, and workflow knowledge areas.
- Activity timeline showing request and response events.
- Dataset automation panel with `Accept and Continue` and `Skip` decisions.
- Composer tools for inserting workflow context, copying the latest answer, and starting a new session.

## Boundary

All implementation files for this layer live under:

```text
agentic-layer/
```

The current application workflow remains separate:

```text
login -> data upload -> understanding -> EDA -> cleaning -> time series forecast -> ML forecast -> loss forecast -> profit forecast -> ML assistant -> prediction -> report
```
