CREATE TABLE IF NOT EXISTS task (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    connection_mode TEXT NOT NULL CHECK(connection_mode IN ('host_port', 'base_url')),
    host TEXT DEFAULT '',
    port INTEGER DEFAULT 0,
    base_url TEXT DEFAULT '',
    api_key TEXT DEFAULT '',
    request_model TEXT DEFAULT '',
    model_name TEXT DEFAULT '',
    task_type TEXT DEFAULT 'bench' CHECK(task_type IN ('bench', 'eval', 'api_test', 'compat_test', 'multi_turn')),
    model_path TEXT NOT NULL,
    dataset_path TEXT NOT NULL,
    extra_args TEXT DEFAULT '[]',
    monitor_enabled INTEGER NOT NULL DEFAULT 0,
    monitor_host TEXT DEFAULT '',
    monitor_port INTEGER NOT NULL DEFAULT 30100,
    monitor_device INTEGER NOT NULL DEFAULT 0,
    monitor_devices TEXT NOT NULL DEFAULT 'auto',
    monitor_interval_ms INTEGER NOT NULL DEFAULT 100,
    monitor_max_duration INTEGER NOT NULL DEFAULT 7200,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'done', 'failed', 'abort')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE task ADD COLUMN model_name TEXT DEFAULT '';
ALTER TABLE task ADD COLUMN task_type TEXT DEFAULT 'bench';
ALTER TABLE task ADD COLUMN monitor_enabled INTEGER NOT NULL DEFAULT 0;
ALTER TABLE task ADD COLUMN monitor_host TEXT DEFAULT '';
ALTER TABLE task ADD COLUMN monitor_port INTEGER NOT NULL DEFAULT 30100;
ALTER TABLE task ADD COLUMN monitor_device INTEGER NOT NULL DEFAULT 0;
ALTER TABLE task ADD COLUMN monitor_devices TEXT NOT NULL DEFAULT '';
ALTER TABLE task ADD COLUMN monitor_interval_ms INTEGER NOT NULL DEFAULT 100;
ALTER TABLE task ADD COLUMN monitor_max_duration INTEGER NOT NULL DEFAULT 7200;

CREATE TABLE IF NOT EXISTS run (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL REFERENCES task(id) ON DELETE CASCADE,
    concurrency INTEGER NOT NULL DEFAULT 0,
    prompt_multiplier INTEGER NOT NULL DEFAULT 4,
    total_prompts INTEGER DEFAULT 0,
    random_input_len INTEGER NOT NULL DEFAULT 0,
    random_output_len INTEGER NOT NULL DEFAULT 0,
    request_rate REAL NOT NULL DEFAULT 0,
    dataset_name TEXT DEFAULT '',
    limit_val INTEGER DEFAULT 0,
    batch_size INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'done', 'failed', 'abort')),
    raw_output TEXT DEFAULT '',
    parsed_result TEXT DEFAULT '{}',
    monitor_request_id TEXT DEFAULT '',
    monitor_session_id TEXT DEFAULT '',
    monitor_status TEXT NOT NULL DEFAULT 'disabled' CHECK(monitor_status IN (
        'disabled', 'pending', 'running', 'done', 'partial', 'unavailable',
        'start_failed', 'stop_failed', 'timeout'
    )),
    monitor_summary TEXT DEFAULT '{}',
    monitor_samples TEXT DEFAULT '[]',
    monitor_metadata TEXT DEFAULT '{}',
    monitor_error TEXT DEFAULT '',
    monitor_started_at TIMESTAMP,
    monitor_finished_at TIMESTAMP,
    started_at TIMESTAMP,
    finished_at TIMESTAMP
);

ALTER TABLE run ADD COLUMN dataset_name TEXT DEFAULT '';
ALTER TABLE run ADD COLUMN limit_val INTEGER DEFAULT 0;
ALTER TABLE run ADD COLUMN batch_size INTEGER DEFAULT 0;
ALTER TABLE run ADD COLUMN total_prompts INTEGER DEFAULT 0;
ALTER TABLE run ADD COLUMN monitor_request_id TEXT DEFAULT '';
ALTER TABLE run ADD COLUMN monitor_session_id TEXT DEFAULT '';
ALTER TABLE run ADD COLUMN monitor_status TEXT NOT NULL DEFAULT 'disabled';
ALTER TABLE run ADD COLUMN monitor_summary TEXT DEFAULT '{}';
ALTER TABLE run ADD COLUMN monitor_samples TEXT DEFAULT '[]';
ALTER TABLE run ADD COLUMN monitor_metadata TEXT DEFAULT '{}';
ALTER TABLE run ADD COLUMN monitor_error TEXT DEFAULT '';
ALTER TABLE run ADD COLUMN monitor_started_at TIMESTAMP;
ALTER TABLE run ADD COLUMN monitor_finished_at TIMESTAMP;

-- Shared prefix test parameters
ALTER TABLE run ADD COLUMN test_mode TEXT DEFAULT 'normal' CHECK(test_mode IN ('normal', 'shared_prefix', 'multi_turn'));
ALTER TABLE run ADD COLUMN gsp_num_groups INTEGER DEFAULT 0;
ALTER TABLE run ADD COLUMN gsp_system_prompt_len INTEGER DEFAULT 0;
ALTER TABLE run ADD COLUMN gsp_question_len INTEGER DEFAULT 0;
ALTER TABLE run ADD COLUMN gsp_prompts_per_group INTEGER DEFAULT 0;
ALTER TABLE run ADD COLUMN gsp_range_ratio REAL DEFAULT 0.0;
ALTER TABLE run ADD COLUMN gsp_max_concurrency INTEGER DEFAULT 0;

-- Multi-turn test parameters (bench_multiturn.py)
ALTER TABLE run ADD COLUMN mt_num_clients INTEGER DEFAULT 10;
ALTER TABLE run ADD COLUMN mt_num_rounds INTEGER DEFAULT 5;
ALTER TABLE run ADD COLUMN mt_request_length INTEGER DEFAULT 1024;
ALTER TABLE run ADD COLUMN mt_sub_question_input_length INTEGER DEFAULT 256;
ALTER TABLE run ADD COLUMN mt_output_length INTEGER DEFAULT 256;
ALTER TABLE run ADD COLUMN mt_range_ratio REAL DEFAULT 1.0;
ALTER TABLE run ADD COLUMN mt_max_context_length INTEGER DEFAULT 0;
ALTER TABLE run ADD COLUMN mt_init_interval REAL DEFAULT 0.0;
ALTER TABLE run ADD COLUMN mt_total_requests INTEGER DEFAULT 0;
ALTER TABLE run ADD COLUMN mt_seed INTEGER DEFAULT 0;
ALTER TABLE run ADD COLUMN mt_host_header TEXT DEFAULT '';
ALTER TABLE run ADD COLUMN mt_api_format TEXT DEFAULT 'openai';
ALTER TABLE run ADD COLUMN mt_thinking INTEGER DEFAULT 0;
ALTER TABLE run ADD COLUMN mt_enable_session_key INTEGER DEFAULT 1;
ALTER TABLE run ADD COLUMN mt_enable_round_barrier INTEGER DEFAULT 0;
ALTER TABLE run ADD COLUMN mt_workload_mode TEXT DEFAULT 'synthetic';
ALTER TABLE run ADD COLUMN mt_replay_sequential_sessions INTEGER DEFAULT 0;

CREATE TABLE IF NOT EXISTS template (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    config TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API Test configuration (linked to run table)
CREATE TABLE IF NOT EXISTS api_test_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL UNIQUE REFERENCES run(id) ON DELETE CASCADE,
    model TEXT DEFAULT '',
    reasoning_model TEXT DEFAULT '',
    fc_model TEXT DEFAULT '',
    enable_fc_test INTEGER DEFAULT 0,
    enable_n_test INTEGER DEFAULT 0,
    n_value INTEGER DEFAULT 3
);

-- API Test results
CREATE TABLE IF NOT EXISTS api_test_result (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
    test_name TEXT NOT NULL,
    category TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pass', 'fail', 'error')),
    http_code INTEGER,
    response_time_ms INTEGER,
    response_body TEXT,
    error_message TEXT,
    assertion_details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Compat Test configuration (linked to run table)
CREATE TABLE IF NOT EXISTS compat_test_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL UNIQUE REFERENCES run(id) ON DELETE CASCADE,
    test_level TEXT DEFAULT 'standard' CHECK(test_level IN ('quick', 'standard', 'comprehensive')),
    framework TEXT DEFAULT 'standard' CHECK(framework IN ('standard', 'sglang', 'vllm')),
    include_extensions INTEGER DEFAULT 0,
    enable_dangerous INTEGER DEFAULT 0,
    timeout_seconds INTEGER DEFAULT 30,
    max_retries INTEGER DEFAULT 3
);
