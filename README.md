# Intelligent Data Assistant

This project is now split into two deployable layers:

- `frontend/`: Next.js UI running on port `3000`
- `backend/`: FastAPI + scikit-learn service running on port `3004`
- `docker-compose.yml` and `infrastructure/nginx/nginx.conf`: root-level orchestration and routing

## Local Development

1. Frontend
```bash
cd frontend
npm install
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
pgAdmin is available on port `5050` by default with login `admin@example.com` / `admin123`.
Its configuration is persisted in a Docker volume, so saved server definitions survive container recreation.
On first startup, it also auto-registers the app PostgreSQL server from `infrastructure/pgadmin/servers.json`.

If you want to customize host ports or Compose credentials, copy `.env.example` to `.env` at the repo root and adjust values like `PGADMIN_PORT`, `NGINX_PORT`, or `POSTGRES_PORT` before starting the stack.

When pgAdmin runs from this Compose stack, add the PostgreSQL server with:

- Host: `postgres`
- Port: `5432`
- Database: `ai_eda_assistant`
- Username: `postgres`
- Password: `postgres`

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
- In Docker, the backend uses `postgresql://postgres:postgres@postgres:5432/ai_eda_assistant` by default.
- For local development, set `ACTIVITY_DATABASE_URL`, for example: `postgresql://postgres:postgres@localhost:5432/ai_eda_assistant`.
