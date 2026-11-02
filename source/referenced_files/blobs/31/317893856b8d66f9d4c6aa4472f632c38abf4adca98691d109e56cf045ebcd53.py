import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "benchmark.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"
MONITOR_STATUSES = {
    "disabled", "pending", "running", "done", "partial", "unavailable",
    "start_failed", "stop_failed", "timeout",
}


def _validate_monitor_status(status: str):
    if status not in MONITOR_STATUSES:
        raise ValueError(f"invalid monitor status: {status}")


def _schema_statements() -> list[str]:
    """Return the non-comment SQL statements from schema.sql, split on ';'."""
    with open(SCHEMA_PATH) as f:
        lines = [l for l in f.read().splitlines() if not l.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def _migrate_task_table():
    """Rebuild task/run to the current schema if task's task_type CHECK predates 'multi_turn'.

    SQLite cannot alter CHECK constraints in place, and ALTER TABLE RENAME
    rewrites child foreign keys, so both tables are rebuilt together with
    foreign_keys disabled, preserving all data. Idempotent: no-op once the
    CHECK already contains multi_turn.
    """
    conn = sqlite3.connect(str(DB_PATH), isolation_level=None)  # autocommit so the pragma applies
    try:
        conn.execute("PRAGMA foreign_keys=OFF")
        task_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='task'"
        ).fetchone()
        run_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='run'"
        ).fetchone()
        # Trigger if either table's CHECK predates multi_turn (task_type or test_mode)
        task_ok = bool(task_row) and "multi_turn" in task_row[0]
        run_ok = bool(run_row) and "multi_turn" in run_row[0]
        if task_ok and run_ok:
            return

        stmts = _schema_statements()
        task_create = next(
            (s for s in stmts if s.startswith("CREATE TABLE IF NOT EXISTS task")), None
        )
        if not task_create:
            return
        run_create = next(
            (s for s in stmts if s.startswith("CREATE TABLE IF NOT EXISTS run")), None
        )
        run_alters = [s for s in stmts if s.startswith("ALTER TABLE run")]

        conn.execute("ALTER TABLE task RENAME TO task_old")
        if run_create:
            conn.execute("ALTER TABLE run RENAME TO run_old")
            conn.execute(run_create)
            for s in run_alters:
                try:
                    conn.execute(s)
                except sqlite3.OperationalError:
                    pass  # Ignore "duplicate column" errors (CREATE already includes them)
            run_cols = [r[1] for r in conn.execute("PRAGMA table_info(run)")]
            old_run_cols = [r[1] for r in conn.execute("PRAGMA table_info(run_old)")]
            common = ", ".join(c for c in run_cols if c in old_run_cols)
            conn.execute(f"INSERT INTO run ({common}) SELECT {common} FROM run_old")
            conn.execute("DROP TABLE run_old")
        conn.execute(task_create)
        task_cols = [r[1] for r in conn.execute("PRAGMA table_info(task)")]
        old_task_cols = [r[1] for r in conn.execute("PRAGMA table_info(task_old)")]
        common = ", ".join(c for c in task_cols if c in old_task_cols)
        conn.execute(f"INSERT INTO task ({common}) SELECT {common} FROM task_old")
        conn.execute("DROP TABLE task_old")

        # ALTER TABLE RENAME rewrites foreign keys in OTHER tables to the
        # renamed table names (e.g. multi_turn_config REFERENCES run(id)
        # becomes REFERENCES run_old(id)). Rebuild any such table so its FK
        # points back at the freshly recreated task/run.
        # Materialize first: executing DDL while iterating the sqlite_master
        # cursor keeps it open and DROP raises "database table is locked".
        other_tables = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table'"
            " AND name NOT IN ('task', 'run')"
        ).fetchall()
        for name, sql in other_tables:
            if not sql:
                continue
            # SQLite stores rewritten FK references with quoted identifiers,
            # e.g. REFERENCES "run_old"(id) — match on the unquoted form.
            unquoted = sql.replace('"', "")
            if (
                "REFERENCES task_old(" not in unquoted
                and "REFERENCES run_old(" not in unquoted
            ):
                continue
            new_sql = (
                sql.replace('REFERENCES "task_old"(', "REFERENCES task(")
                .replace("REFERENCES task_old(", "REFERENCES task(")
                .replace('REFERENCES "run_old"(', "REFERENCES run(")
                .replace("REFERENCES run_old(", "REFERENCES run(")
            )
            old_name = f"{name}_mtold"
            conn.execute(f"ALTER TABLE {name} RENAME TO {old_name}")
            conn.execute(new_sql)
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({name})")]
            old_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({old_name})")]
            common = ", ".join(c for c in cols if c in old_cols)
            conn.execute(
                f"INSERT INTO {name} ({common}) SELECT {common} FROM {old_name}"
            )
            conn.execute(f"DROP TABLE {old_name}")
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        for stmt in _schema_statements():
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as exc:
                # schema.sql contains idempotent ALTER statements for upgrading
                # existing benchmark.db files. Only an already-present column is
                # safe to ignore; syntax and other migration errors must surface.
                if stmt.upper().startswith("ALTER TABLE") and "duplicate column name" in str(exc).lower():
                    continue
                raise
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            """UPDATE task SET monitor_devices = CAST(monitor_device AS TEXT)
               WHERE monitor_devices IS NULL OR TRIM(monitor_devices) = ''"""
        )
    _migrate_task_table()


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_task(
    name: str,
    connection_mode: str,
    model_path: str,
    dataset_path: str,
    host: str = "",
    port: int = 0,
    base_url: str = "",
    api_key: str = "",
    request_model: str = "",
    model_name: str = "",
    task_type: str = "bench",
    extra_args: Optional[list] = None,
    monitor_enabled: bool = False,
    monitor_host: str = "",
    monitor_port: int = 30100,
    monitor_device: int = 0,
    monitor_devices: str = "auto",
    monitor_interval_ms: int = 100,
    monitor_max_duration: int = 7200,
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO task (name, connection_mode, host, port, base_url, api_key,
               request_model, model_name, task_type, model_path, dataset_path, extra_args,
               monitor_enabled, monitor_host, monitor_port, monitor_device, monitor_devices,
               monitor_interval_ms, monitor_max_duration)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name, connection_mode, host, port, base_url, api_key,
                request_model, model_name, task_type, model_path, dataset_path,
                json.dumps(extra_args or []),
                int(bool(monitor_enabled)), monitor_host, monitor_port, monitor_device,
                monitor_devices,
                monitor_interval_ms, monitor_max_duration,
            ),
        )
        return cur.lastrowid


def update_task_status(task_id: int, status: str):
    with get_db() as conn:
        conn.execute("UPDATE task SET status = ? WHERE id = ?", (status, task_id))


def get_task(task_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM task WHERE id = ?", (task_id,)).fetchone()
        return dict(row) if row else None


def delete_task(task_id: int):
    """Delete a task and all its runs. CASCADE will handle related records."""
    with get_db() as conn:
        conn.execute("DELETE FROM task WHERE id = ?", (task_id,))


def list_tasks(page: int = 1, per_page: int = 20, task_type: str = None) -> dict:
    with get_db() as conn:
        if task_type:
            total = conn.execute("SELECT COUNT(*) FROM task WHERE task_type = ?", (task_type,)).fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM task WHERE task_type = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (task_type, per_page, (page - 1) * per_page),
            ).fetchall()
        else:
            total = conn.execute("SELECT COUNT(*) FROM task").fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM task ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (per_page, (page - 1) * per_page),
            ).fetchall()
        return {
            "tasks": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "per_page": per_page,
        }


def create_run(
    task_id: int,
    concurrency: int = 0,
    prompt_multiplier: int = 0,
    total_prompts: int = 0,
    random_input_len: int = 0,
    random_output_len: int = 0,
    request_rate: float = 0,
    dataset_name: str = "",
    limit_val: int = 0,
    batch_size: int = 0,
    test_mode: str = "normal",
    gsp_num_groups: int = 0,
    gsp_system_prompt_len: int = 0,
    gsp_question_len: int = 0,
    gsp_prompts_per_group: int = 0,
    gsp_range_ratio: float = 0.0,
    gsp_max_concurrency: int = 0,
    mt_num_clients: int = 0,
    mt_num_rounds: int = 0,
    mt_request_length: int = 0,
    mt_sub_question_input_length: int = 0,
    mt_output_length: int = 0,
    mt_range_ratio: float = 1.0,
    mt_max_context_length: int = 0,
    mt_init_interval: float = 0.0,
    mt_total_requests: int = 0,
    mt_seed: int = 0,
    mt_host_header: str = "",
    mt_api_format: str = "openai",
    mt_thinking: int = 0,
    mt_enable_session_key: int = 1,
    mt_enable_round_barrier: int = 0,
    monitor_status: str = "disabled",
    mt_workload_mode: str = "synthetic",
    mt_replay_sequential_sessions: int = 0,
) -> int:
    _validate_monitor_status(monitor_status)
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO run (task_id, concurrency, prompt_multiplier, total_prompts,
               random_input_len, random_output_len, request_rate,
               dataset_name, limit_val, batch_size, test_mode,
               gsp_num_groups, gsp_system_prompt_len, gsp_question_len, gsp_prompts_per_group,
               gsp_range_ratio, gsp_max_concurrency,
               mt_num_clients, mt_num_rounds, mt_request_length, mt_sub_question_input_length,
               mt_output_length, mt_range_ratio, mt_max_context_length, mt_init_interval,
               mt_total_requests, mt_seed, mt_host_header, mt_api_format, mt_thinking,
               mt_enable_session_key, mt_enable_round_barrier, monitor_status,
               mt_workload_mode, mt_replay_sequential_sessions)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
               ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (task_id, concurrency, prompt_multiplier, total_prompts, random_input_len,
             random_output_len, request_rate, dataset_name, limit_val, batch_size,
             test_mode, gsp_num_groups, gsp_system_prompt_len, gsp_question_len, gsp_prompts_per_group,
             gsp_range_ratio, gsp_max_concurrency,
             mt_num_clients, mt_num_rounds, mt_request_length, mt_sub_question_input_length,
             mt_output_length, mt_range_ratio, mt_max_context_length, mt_init_interval,
             mt_total_requests, mt_seed, mt_host_header, mt_api_format, mt_thinking,
             mt_enable_session_key, mt_enable_round_barrier, monitor_status,
             mt_workload_mode, mt_replay_sequential_sessions),
        )
        return cur.lastrowid


def get_run(run_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM run WHERE id = ?", (run_id,)).fetchone()
        return _deserialize_monitor_fields(dict(row)) if row else None


def get_task_runs(task_id: int, *, include_monitor_samples: bool = False) -> list:
    with get_db() as conn:
        if include_monitor_samples:
            query = "SELECT * FROM run WHERE task_id = ? ORDER BY id"
        else:
            # Avoid reading and JSON-decoding the potentially large time-series
            # column for task polling and other summary-only callers.
            query = """SELECT id, task_id, concurrency, prompt_multiplier, total_prompts,
                       random_input_len, random_output_len, request_rate,
                       dataset_name, limit_val, batch_size, test_mode,
                       gsp_num_groups, gsp_system_prompt_len, gsp_question_len,
                       gsp_prompts_per_group, gsp_range_ratio, gsp_max_concurrency,
                       mt_num_clients, mt_num_rounds, mt_request_length,
                       mt_sub_question_input_length, mt_output_length, mt_range_ratio,
                       mt_max_context_length, mt_init_interval, mt_total_requests,
                       mt_seed, mt_host_header, mt_api_format, mt_thinking,
                       mt_enable_session_key, mt_enable_round_barrier,
                       mt_workload_mode, mt_replay_sequential_sessions,
                       status, raw_output,
                       parsed_result, monitor_request_id, monitor_session_id,
                       monitor_status, monitor_summary, '[]' AS monitor_samples,
                       monitor_metadata, monitor_error, monitor_started_at,
                       monitor_finished_at, started_at, finished_at
                       FROM run WHERE task_id = ? ORDER BY id"""
        rows = conn.execute(query, (task_id,)).fetchall()
        return [_deserialize_monitor_fields(dict(r)) for r in rows]


def get_interrupted_monitor_runs() -> list:
    """Return benchmark runs left active by an unclean platform shutdown."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT run.* FROM run
               JOIN task ON task.id = run.task_id
               WHERE run.status = 'running'
                  OR (task.monitor_enabled = 1 AND run.monitor_status = 'running')"""
        ).fetchall()
        return [_deserialize_monitor_fields(dict(row)) for row in rows]


def _deserialize_monitor_fields(run: dict) -> dict:
    """Expose monitor JSON columns as objects without changing parsed_result."""
    for field, default in (
        ("monitor_summary", {}),
        ("monitor_samples", []),
        ("monitor_metadata", {}),
    ):
        raw = run.get(field)
        if isinstance(raw, str):
            try:
                run[field] = json.loads(raw) if raw else default.copy()
            except (TypeError, json.JSONDecodeError):
                run[field] = default.copy()
    return run


def update_run_monitor_status(
    run_id: int,
    status: str,
    *,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    error: Optional[str] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
):
    _validate_monitor_status(status)
    updates = {"monitor_status": status}
    if request_id is not None:
        updates["monitor_request_id"] = request_id
    if session_id is not None:
        updates["monitor_session_id"] = session_id
    if error is not None:
        updates["monitor_error"] = error
    if started_at is not None:
        updates["monitor_started_at"] = started_at
    if finished_at is not None:
        updates["monitor_finished_at"] = finished_at
    set_clause = ", ".join(f"{key} = ?" for key in updates)
    values = list(updates.values()) + [run_id]
    with get_db() as conn:
        conn.execute(f"UPDATE run SET {set_clause} WHERE id = ?", values)


def save_run_monitor_result(
    run_id: int,
    *,
    status: str,
    summary: Optional[dict] = None,
    samples: Optional[list] = None,
    metadata: Optional[dict] = None,
    session_id: Optional[str] = None,
    error: str = "",
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
):
    _validate_monitor_status(status)
    updates = {
        "monitor_status": status,
        "monitor_summary": json.dumps(summary or {}, ensure_ascii=False),
        "monitor_samples": json.dumps(samples or [], ensure_ascii=False),
        "monitor_metadata": json.dumps(metadata or {}, ensure_ascii=False),
        "monitor_error": error,
    }
    if session_id is not None:
        updates["monitor_session_id"] = session_id
    if started_at is not None:
        updates["monitor_started_at"] = started_at
    if finished_at is not None:
        updates["monitor_finished_at"] = finished_at
    set_clause = ", ".join(f"{key} = ?" for key in updates)
    values = list(updates.values()) + [run_id]
    with get_db() as conn:
        conn.execute(f"UPDATE run SET {set_clause} WHERE id = ?", values)


def update_run(run_id: int, **kwargs):
    allowed = {"status", "raw_output", "parsed_result", "started_at", "finished_at"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [run_id]
    with get_db() as conn:
        conn.execute(f"UPDATE run SET {set_clause} WHERE id = ?", values)


def create_template(name: str, config: dict) -> int:
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO template (name, config) VALUES (?, ?)",
            (name, json.dumps(config, ensure_ascii=False)),
        )
        return cur.lastrowid


def list_templates() -> list:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM template ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_template(template_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM template WHERE id = ?", (template_id,)).fetchone()
        return dict(row) if row else None


def update_template(template_id: int, name: str = None, config: dict = None):
    parts = []
    values = []
    if name is not None:
        parts.append("name = ?")
        values.append(name)
    if config is not None:
        parts.append("config = ?")
        values.append(json.dumps(config, ensure_ascii=False))
    if not parts:
        return
    values.append(template_id)
    with get_db() as conn:
        conn.execute(f"UPDATE template SET {', '.join(parts)} WHERE id = ?", values)


def delete_template(template_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM template WHERE id = ?", (template_id,))


# ---------------------------------------------------------------------------
# API Test config and result operations
# ---------------------------------------------------------------------------

def create_api_test_config(
    run_id: int,
    model: str = "",
    reasoning_model: str = "",
    fc_model: str = "",
    enable_fc_test: int = 0,
    enable_n_test: int = 0,
    n_value: int = 3,
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO api_test_config (run_id, model, reasoning_model, fc_model,
               enable_fc_test, enable_n_test, n_value)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, model, reasoning_model, fc_model, enable_fc_test, enable_n_test, n_value),
        )
        return cur.lastrowid


def get_api_test_config(run_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM api_test_config WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None


def create_api_test_result(
    run_id: int,
    test_name: str,
    category: str,
    status: str,
    http_code: int = 0,
    response_time_ms: int = 0,
    response_body: str = "",
    error_message: str = "",
    assertion_details: str = "",
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO api_test_result (run_id, test_name, category, status,
               http_code, response_time_ms, response_body, error_message, assertion_details)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, test_name, category, status, http_code, response_time_ms,
             response_body, error_message, assertion_details),
        )
        return cur.lastrowid


def get_api_test_results(run_id: int) -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM api_test_result WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Compat Test config operations
# ---------------------------------------------------------------------------

def create_compat_test_config(
    run_id: int,
    test_level: str = "standard",
    framework: str = "standard",
    include_extensions: int = 0,
    enable_dangerous: int = 0,
    timeout_seconds: int = 30,
    max_retries: int = 3,
) -> int:
    with get_db() as conn:
        cur = conn.execute(
            """INSERT INTO compat_test_config (run_id, test_level, framework,
               include_extensions, enable_dangerous, timeout_seconds, max_retries)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run_id, test_level, framework, include_extensions, enable_dangerous,
             timeout_seconds, max_retries),
        )
        return cur.lastrowid


def get_compat_test_config(run_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM compat_test_config WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None
