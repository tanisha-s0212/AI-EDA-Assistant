# Development and validation

Use Python 3.12 and Node.js 22 or later. The backend and frontend can run independently during development.

```bash
cd backend
python -m pip install -r requirements.txt
python -m pytest tests -q
```

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
