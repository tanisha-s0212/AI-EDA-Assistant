# Development and validation

Use Python 3.12 and Node.js 22 or later. The backend and frontend can run independently during development.

Create a project virtual environment and use its interpreter for both the backend and tests. From the repository root on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -c "import statsmodels, pmdarima, prophet, xgboost; print('Forecast dependencies available')"
cd backend
..\.venv\Scripts\python.exe main.py
```

In a separate terminal, run backend tests with `..\.venv\Scripts\python.exe -m pytest tests -q` from `backend`. On macOS or Linux, create the environment with `python3.12 -m venv .venv` and use `.venv/bin/python` from the repository root (or `../.venv/bin/python` from `backend`). If the dependency check fails, install the requirements with that same interpreter before starting the backend.

```bash
cd frontend
npm ci
npm run lint
npm run test
npm run build
```

To check the Compose model without starting services:

```bash
docker compose --env-file .env.example config -q
```

The backend suite can run without PostgreSQL; database-dependent behavior is reported as unavailable when the configured database cannot be reached. Do not delete tests or set fake provider keys to hide unavailable external services.

Keep generated data, model artifacts, virtual environments, test caches, build output, logs, and copied environment files out of Git. The sample datasets and demo model are intentional repository fixtures.
