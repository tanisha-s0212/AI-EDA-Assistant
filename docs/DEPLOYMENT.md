# Deployment

The supplied `docker-compose.yml` supports a single-host deployment:

1. Nginx accepts HTTP on `NGINX_PORT` and routes frontend and API requests.
2. The frontend waits for the FastAPI health endpoint.
3. The backend waits for PostgreSQL and persists datasets, models, and logs to named volumes.

Before exposing the stack outside a trusted network:

- Store `POSTGRES_PASSWORD` and provider credentials in a managed secret store or an untracked deployment environment file.
- Rotate any provider credential previously committed to repository history.
- Put TLS termination in front of Nginx and set `SESSION_COOKIE_SECURE=true`.
- Set exact public origins in `CORS_ALLOWED_ORIGINS` when browsers access the API directly.
- Back up `postgres_data`, `backend_datasets`, and `backend_models` volumes.
- Set resource limits, monitoring, log retention, and an external database or managed backup policy according to the hosting platform.

The repository does not provide certificates, a secret manager, horizontal scaling coordination, automated database migrations, or cloud-specific infrastructure. Those are deployment-environment responsibilities.
