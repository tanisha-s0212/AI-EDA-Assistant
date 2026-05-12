# Application Workflow Knowledge

This file gives the standalone agentic layer read-only knowledge about the existing application workflow. It is documentation inside `agentic-layer` only; it does not modify the application.

## Confirmed Main Workflow

The frontend application defines the workflow tabs in `frontend/src/app/page.tsx`.

Order:

1. Login
2. Data Upload
3. Data Understanding
4. Exploratory Data Analysis
5. Data Cleaning
6. Time Series Forecast
7. Machine Learning Forecast
8. Loss Forecast
9. Profit Forecast
10. ML Assistant
11. Prediction
12. Report

## Confirmed Tab IDs and Components

| Tab ID | Label | Component |
| --- | --- | --- |
| `upload` | Data Upload | `frontend/src/components/tabs/upload-tab.tsx` |
| `understanding` | Data Understanding | `frontend/src/components/tabs/understanding-tab.tsx` |
| `eda` | Exploratory Data Analysis | `frontend/src/components/tabs/eda-tab.tsx` |
| `cleaning` | Data Cleaning | `frontend/src/components/tabs/cleaning-tab.tsx` |
| `forecast_ts` | Time Series Forecast | `frontend/src/components/tabs/time-series-forecast-tab.tsx` |
| `forecast_ml` | Machine Learning Forecast | `frontend/src/components/tabs/ml-forecast-tab.tsx` |
| `loss_forecast` | Loss Forecast | `frontend/src/components/tabs/loss-forecast-tab.tsx` |
| `profit_forecast` | Profit Forecast | `frontend/src/components/tabs/profit-forecast-tab.tsx` |
| `ml` | ML Assistant | `frontend/src/components/tabs/ml-tab.tsx` |
| `prediction` | Prediction | `frontend/src/components/tabs/prediction-tab.tsx` |
| `report` | Report | `frontend/src/components/tabs/report-tab.tsx` |

## Workflow Gating

The step navigation in `frontend/src/app/page.tsx` enables:

- Upload tab always.
- Other tabs after dataset context exists.
- Prediction only after a model is trained.

The user-facing application first shows `frontend/src/components/login-page.tsx` before the workspace.

## Forecast Tab Flow

The forecast sequence is:

1. Time Series Forecast
   - Component: `frontend/src/components/tabs/time-series-forecast-tab.tsx`
   - Stores result in `timeSeriesForecastResult`.

2. Machine Learning Forecast
   - Component: `frontend/src/components/tabs/ml-forecast-tab.tsx`
   - Stores result in `mlForecastResult`.

3. Loss Forecast
   - Component: `frontend/src/components/tabs/loss-forecast-tab.tsx`
   - Requires `timeSeriesForecastResult`, `mlForecastResult`, and `datasetId`.
   - Calls `runLossForecast`.
   - Stores result in `lossForecast`, `lossSegments`, and `lossSummary`.

4. Profit Forecast
   - Component: `frontend/src/components/tabs/profit-forecast-tab.tsx`
   - Requires `datasetId` and existing `lossForecast`.
   - Calls `runProfitForecast`.
   - Stores result in `profitForecast` and `profitScenarios`.

## State and API References

- Shared frontend workflow state: `frontend/src/lib/store.ts`
- Frontend API client: `frontend/src/lib/api.ts`
- Backend API implementation: `backend/main.py`
- Forecast shared types: `frontend/src/types/forecast.ts`

## Agentic Layer Rule

The agentic layer may read and summarize these existing files to answer user questions, but it must not edit them unless the user explicitly approves a future code-modification workflow.
