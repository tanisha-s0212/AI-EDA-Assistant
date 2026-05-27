#!/usr/bin/env bash
set -euo pipefail

cd backend
export ACTIVITY_DB_CONNECT_TIMEOUT=1
pytest tests/test_agentic.py -v

cd ../frontend
npm run test -- --watchAll=false
