CREATE TABLE IF NOT EXISTS agentic_sessions (
    session_id UUID PRIMARY KEY,
    created_at TIMESTAMP,
    dataset_path TEXT,
    status TEXT,
    user_id BIGINT REFERENCES app_users(id)
);

CREATE TABLE IF NOT EXISTS agentic_step_executions (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES agentic_sessions(session_id),
    step_name TEXT,
    status TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    output_summary TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS agentic_decisions (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES agentic_sessions(session_id),
    step_name TEXT,
    decision TEXT CHECK (decision IN ('accepted', 'skipped')),
    reasoning TEXT,
    decided_at TIMESTAMP
);
