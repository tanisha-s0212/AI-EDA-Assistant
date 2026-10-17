# Intelligent Data Assistant

AI EDA Assistant is a browser-based workspace for uploading tabular datasets,
profiling and cleaning data, running exploratory analysis, training supervised
models, and producing forecast and prediction results.

This project is now split into two deployable layers:

- `frontend/`: Next.js UI running on port `3000`
- `backend/`: FastAPI + scikit-learn service running on port `3004`
- `docker-compose.yml` and `infrastructure/nginx/nginx.conf`: root-level orchestration and routing
- `agentic-layer/`: optional hosted-assistant workspace assets and provider integration
- `samples/`: curated CSV, TSV, Excel, and Parquet examples

## Local Development

1. Frontend
```bash
cd frontend
npm ci
npm run dev
```

2. Backend
```bash
cd backend
python -m pip install -r requirements.txt
python main.py
```

3. Open `http://localhost:3000`

For local frontend development, set `frontend/.env.example` values in a real `.env.local` file if you want a custom API base URL. The default target is `http://127.0.0.1:3004/api`.

## Docker

```bash
docker compose up --build
```

Nginx serves as the public entrypoint on port `8888` by default, proxies `/` to the frontend, and proxies `/api/*` to the backend.
The frontend, backend, and PostgreSQL services are internal to the Compose network. Named volumes preserve PostgreSQL data, uploaded datasets, trained models, and backend logs.

If you want to customize the host port or Compose credentials, copy `.env.example` to `.env` at the repo root. Set a unique `POSTGRES_PASSWORD` before a non-local deployment.

If you want the Next.js server itself to proxy API requests in a non-Docker environment, set `ENABLE_BACKEND_REWRITE=true` and optionally `BACKEND_ORIGIN=http://127.0.0.1:3004`.

## Architecture

- The frontend contains only UI, state management, charts, and API consumption.
- The backend owns training, prediction, parquet parsing with Polars, cleaning explanations, and PDF report generation.
- Trained models are persisted with `joblib` under `backend/models/`.
- User workflow activity is persisted in PostgreSQL through `ACTIVITY_DATABASE_URL`.

## Application Workflow

The guided app flow now follows this sequence:

1. `Upload`
2. `Data Understanding`
3. `Exploratory Data Analysis`
4. `Data Cleaning`
5. `Forecast TS`
6. `Forecast ML`
7. `ML Assistant`
8. `Prediction`
9. `Report`

Notes about the current workflow behavior:

- Large uploads can load a browser preview while the backend caches the full dataset for cleaning, advanced EDA, forecasting, training, and report generation.
- The final report is intended to mirror the same tab sequence the user completed in the application rather than acting as a raw backend dump.
- Forecasting remains optional. The report includes time-series and ML forecast sections only when those paths were executed in the active session.

## Activity Storage

- Every backend-driven user workflow action is now recorded in PostgreSQL, including dataset parsing/caching, cleaning, advanced EDA, forecasting, model training, predictions, model uploads, and report generation.
- The frontend automatically sends a browser session id in the `X-Client-Session-Id` header so activity rows can be grouped by user session even without authentication.
- Query recorded activity with `GET /api/activities`. Optional filters:
  - `dataset_id`
  - `client_session_id`
  - `server_session_id`
  - `limit`
- In Docker, the backend uses the PostgreSQL service on the Compose network.
- For local development, set `ACTIVITY_DATABASE_URL` with the local PostgreSQL credentials.
- If PostgreSQL is optional in your local workflow, you can reduce backend wait time with `ACTIVITY_DB_CONNECT_TIMEOUT=1` in `.env` to fail fast when the database is offline.

## Documentation and validation

See [architecture](docs/ARCHITECTURE.md), [development](docs/DEVELOPMENT.md),
and [deployment](docs/DEPLOYMENT.md) guidance for supported runtime boundaries,
validation commands, security prerequisites, and deployment limitations.
