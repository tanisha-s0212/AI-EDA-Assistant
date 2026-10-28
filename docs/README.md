# Project documentation

- [Architecture](ARCHITECTURE.md) describes runtime boundaries and data flow.
- [Development](DEVELOPMENT.md) describes local setup and validation.
- [Deployment](DEPLOYMENT.md) describes the supported Docker Compose model and operational prerequisites.

Generated presentation and report exports are intentionally excluded from Git.

## Analytics and model review

Assistant supervised-training requests to `POST /api/assistant/ask` now return
`status: needs_confirmation` before training. Review `ml_plan.target`,
`ml_plan.features`, exclusions and warnings, then repeat the request with
`confirm_training: true` and the returned `ml_plan.review_token`. Optionally
send `feature_columns` to specify the predictors. The token is tied to the
dataset, target, features and task, so changes require a fresh review. Existing
API clients that assumed the first request always trained must handle this
review step. Missing prediction inputs are not filled automatically unless the
caller explicitly sets `allow_default_inputs: true`.

`POST /api/ml/feature-review` provides a pre-training review for manual model
selection. Leakage warnings are heuristic and require human judgment about
whether each feature is available at prediction time. The training endpoint
rejects the target or an exact duplicate of it as a feature.

Cleaning saves a new cached snapshot and retains the uploaded source. Numeric
forecast outputs retain model precision; exact decimal values that cannot
round-trip through a JSON number can be serialized as strings. Forecasts and
profit projections are estimates; financial reports disclose mappings and
assumptions and do not represent projected amounts as actual profit.
