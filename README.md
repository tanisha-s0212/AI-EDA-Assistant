# AI-Assisted EDA & ML Platform

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

If you want the Next.js server itself to proxy API requests in a non-Docker environment, set `ENABLE_BACKEND_REWRITE=true` and optionally `BACKEND_ORIGIN=http://127.0.0.1:3004`.

## Architecture

- The frontend contains only UI, state management, charts, and API consumption.
- The backend owns training, prediction, parquet parsing with Polars, cleaning explanations, and PDF report generation.
- Trained models are persisted with `joblib` under `backend/models/`.
