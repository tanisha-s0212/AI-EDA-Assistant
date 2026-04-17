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
pgAdmin is available on port `8080` by default with login `admin@example.com` / `admin123`.
Its configuration is persisted in a Docker volume, so saved server definitions survive container recreation.
On first startup, it also auto-registers the app PostgreSQL server from `infrastructure/pgadmin/servers.json`.

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
