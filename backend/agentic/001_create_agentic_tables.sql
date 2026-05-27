CREATE TABLE IF NOT EXISTS agentic_runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agentic_runs_session_id ON agentic_runs (session_id);
CREATE INDEX IF NOT EXISTS idx_agentic_runs_updated_at ON agentic_runs (updated_at DESC);

CREATE TABLE IF NOT EXISTS agentic_steps (
    step_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES agentic_runs(run_id) ON DELETE CASCADE,
    step_name TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json JSONB,
    executed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agentic_steps_run_id ON agentic_steps (run_id);
CREATE INDEX IF NOT EXISTS idx_agentic_steps_executed_at ON agentic_steps (executed_at DESC);

CREATE TABLE IF NOT EXISTS agentic_decisions (
    decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES agentic_runs(run_id) ON DELETE CASCADE,
    step_id TEXT,
    decision TEXT NOT NULL,
    reason TEXT,
    decided_at TEXT NOT NULL
);

ALTER TABLE agentic_decisions ADD COLUMN IF NOT EXISTS decision_id TEXT;
ALTER TABLE agentic_decisions ADD COLUMN IF NOT EXISTS run_id TEXT;
ALTER TABLE agentic_decisions ADD COLUMN IF NOT EXISTS step_id TEXT;
ALTER TABLE agentic_decisions ADD COLUMN IF NOT EXISTS reason TEXT;
ALTER TABLE agentic_decisions ADD COLUMN IF NOT EXISTS decided_at TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_agentic_decisions_decision_id ON agentic_decisions (decision_id);
CREATE INDEX IF NOT EXISTS idx_agentic_decisions_run_id ON agentic_decisions (run_id);
CREATE INDEX IF NOT EXISTS idx_agentic_decisions_decided_at ON agentic_decisions (decided_at DESC);

CREATE TABLE IF NOT EXISTS agentic_audit (
    audit_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES agentic_runs(run_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    payload_json JSONB,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agentic_audit_run_id ON agentic_audit (run_id);
CREATE INDEX IF NOT EXISTS idx_agentic_audit_created_at ON agentic_audit (created_at DESC);
