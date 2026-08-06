#!/usr/bin/env bash
set -euo pipefail

cd backend
export ACTIVITY_DB_CONNECT_TIMEOUT=1
pytest tests/test_agentic.py tests/test_cleaning_dates.py tests/test_ml_forecast_aggregation.py tests/test_ts_stationarity_date_freq.py tests/test_report_ist_timestamps.py tests/test_large_dataset_perf.py -v

cd ../frontend
npm run test -- --watchAll=false
