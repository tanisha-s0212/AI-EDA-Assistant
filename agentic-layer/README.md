# IDA Agentic Core

IDA Agentic Core is now served by the main FastAPI backend. The main application can open the agentic workspace directly through the existing `/api` backend path, so no separate port-5055 server is required.

## Main Backend Routes

The former standalone server endpoints are mounted under:

```text
/api/agentic/core
```

Key routes:

```text
GET  /api/agentic/core
GET  /api/agentic/core/app.js
GET  /api/agentic/core/styles.css
GET  /api/agentic/core/runs/{path}
GET  /api/agentic/core/health
POST /api/agentic/core/activity
POST /api/agentic/core/chat
POST /api/agentic/core/workflow/suggest
POST /api/agentic/core/workflow/decision
```

The app-native agentic execution routes remain mounted under:

```text
/api/agentic/health
/api/agentic/suggest-next-steps
/api/agentic/execute-step
/api/agentic/decision
/api/agentic/session/{session_id}/status
/api/agentic/session/{session_id}/report
```

## Frontend Launcher

The main React app uses:

```env
NEXT_PUBLIC_AGENTIC_ENABLED=true
NEXT_PUBLIC_AGENTIC_API_BASE=/api/agentic
```

The launcher appears after login and opens the backend-hosted IDA Agentic Core workspace with the active dataset context and a return URL.

## Environment

Provider keys still live in:

```text
agentic-layer/.env
```

Gemini stable model switching is configured with:

```env
GEMINI_STABLE_MODEL=gemini-3.1-flash-lite
GEMINI_FAST_MODEL=gemini-3.1-flash-lite
```

The backend imports the provider configuration and local workflow helpers from `agentic-layer/server`, but serving and API handling happen through `backend/main.py`.

## Workflow

The confirmed Intelligent Data Assistant flow is:

```text
login -> data upload -> understanding -> EDA -> cleaning -> time series forecast -> ML forecast -> loss forecast -> profit forecast -> ML assistant -> prediction -> report
```

Accepted or skipped standalone-core decisions are stored under:

```text
agentic-layer/runs/<run-id>/
```
