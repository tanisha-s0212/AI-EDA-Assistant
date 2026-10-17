#!/usr/bin/env bash
set -euo pipefail

cd backend
export ACTIVITY_DB_CONNECT_TIMEOUT=1
python -m pytest tests -q

cd ../frontend
npm run test
