# Architecture

```text
Browser
  │
  ▼
Nginx (public Compose endpoint)
  ├── /       → Next.js frontend
  └── /api/*  → FastAPI backend
                    ├── PostgreSQL activity and authentication records
                    ├── named volumes for uploaded datasets and model artifacts
                    └── agentic workspace assets and optional provider clients
```

The Next.js application owns presentation, browser session state, and API requests. The FastAPI application owns file parsing, data profiling, cleaning, ML and forecasting execution, model artifacts, authorization checks, reports, and activity recording. PostgreSQL stores activity and authentication state.

`backend/main.py` is the FastAPI entry point. Its helper modules contain shared dtype inference, supervised-model execution, forecasting, prediction contract validation, sales-domain selection, natural-language analytics, and agentic workflow support. They are imported by application routes and covered by the backend tests; they must be changed with route and regression coverage.

Uploaded datasets and model artifacts are runtime data. Docker deploys them on named volumes, while local development stores them under ignored backend runtime directories. Curated demo artifacts and samples remain versioned because tests and the sample walkthrough rely on them.
