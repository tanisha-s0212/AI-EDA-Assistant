# Agentic Layer

Standalone local assistant layer for the AI EDA Assistant workspace.

This layer is intentionally separate from the existing application flow. It runs as its own local browser page and does not modify the current backend, frontend, infrastructure, routes, or workflow.

## What It Does

- Provides a professional multi-panel agent workspace.
- Uses Gemini as the primary LLM provider.
- Uses Groq as the fallback provider.
- Reads the workspace structure for context.
- Searches project text for relevant files and snippets.
- Helps with explanation, planning, review, and navigation.
- Includes focus modes for Ask, Plan, Review, Explain, and Search workflows.
- Includes prompt shortcuts, activity feed, session reset, answer copy, and workflow-context insertion.

## What It Does Not Do Yet

- It does not edit application files.
- It does not apply patches.
- It does not add buttons or panels inside the existing application UI.
- It does not change the existing login-to-report workflow.

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

LLM_PRIMARY_PROVIDER=gemini
LLM_FALLBACK_PROVIDER=groq
LLM_DEFAULT_MODE=fast

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
Gemini -> Groq -> local workspace context
```

If Gemini fails or reaches quota, the server tries Groq. If both providers are unavailable, the layer still returns local workspace tree/search context.

## Modes

- `Ask`: general project questions.
- `Plan`: implementation planning without editing files.
- `Review`: code review style risk analysis.
- `Explain`: file or module explanation.
- `Search`: project search and navigation.

## Interface Features

- Assistant console with a professional workspace layout.
- Focus mode selector for task intent.
- Prompt library for workflow, forecast, report, and review tasks.
- Workspace context panel for frontend, backend, and workflow knowledge areas.
- Activity timeline showing request and response events.
- Composer tools for inserting workflow context, copying the latest answer, and starting a new session.

## Boundary

All implementation files for this layer live under:

```text
agentic-layer/
```

The current application workflow remains separate:

```text
login -> data upload -> data understanding -> EDA -> data cleaning -> forecasts -> ML assistant -> prediction -> report
```
