#!/usr/bin/env python3
"""Safely merge one llm-benchmark-online SQLite database into another.

The source and base databases are never modified. By default this command
performs the merge against a temporary copy (dry run). Pass --apply to create
a new merged database at --output. Task IDs are rebuilt by created_at; run IDs
are rebuilt by started_at, falling back to finished_at and task.created_at.

Examples:
    python3 merge_benchmark_db.py
    python3 merge_benchmark_db.py --source /path/old.db --target ./benchmark.db
    python3 merge_benchmark_db.py --apply
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path("/dfs/liuxing/benchmark.db")
DEFAULT_TARGET = PROJECT_DIR / "benchmark.db"
DEFAULT_OUTPUT = PROJECT_DIR / "benchmark.merged.db"
PARENT_TABLES = ("task", "run")
RUN_CHILD_TABLES = (
    "api_test_config",
    "api_test_result",
    "compat_test_config",
)
KNOWN_TABLES = set(PARENT_TABLES + RUN_CHILD_TABLES + ("template",))


def quote(identifier: str) -> str:
    """Quote an identifier obtained from SQLite schema metadata."""
    return '"' + identifier.replace('"', '""') + '"'


def user_tables(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({quote(table)})")]


def require_compatible_schema(
    source: sqlite3.Connection, target: sqlite3.Connection
) -> None:
    source_tables = user_tables(source)
    target_tables = user_tables(target)
    missing_required = {"task", "run"} - source_tables
    if missing_required:
        raise RuntimeError(
            "source is missing required tables: " + ", ".join(sorted(missing_required))
        )

    missing_target = (source_tables & KNOWN_TABLES) - target_tables
    if missing_target:
        raise RuntimeError(
            "target is missing source tables: " + ", ".join(sorted(missing_target))
        )

    unknown = source_tables - KNOWN_TABLES
    if unknown:
        raise RuntimeError(
            "source contains unsupported tables; update the merger first: "
            + ", ".join(sorted(unknown))
        )

    for table in source_tables & KNOWN_TABLES:
        source_columns = set(columns(source, table))
        target_columns = set(columns(target, table))
        missing_columns = source_columns - target_columns
        if missing_columns:
            raise RuntimeError(
                f"target table {table!r} is missing source columns: "
                + ", ".join(sorted(missing_columns))
            )


def insert_row(
    target: sqlite3.Connection,
    table: str,
    row: sqlite3.Row,
    allowed_columns: Iterable[str],
    replacements: dict[str, object] | None = None,
) -> int:
    replacements = replacements or {}
    selected = [
        name
        for name in allowed_columns
        if name != "id" and (name in row.keys() or name in replacements)
    ]
    values = [replacements.get(name, row[name]) for name in selected]
    names_sql = ", ".join(quote(name) for name in selected)
    marks = ", ".join("?" for _ in selected)
    cursor = target.execute(
        f"INSERT INTO {quote(table)} ({names_sql}) VALUES ({marks})", values
    )
    return int(cursor.lastrowid)


def unique_template_name(target: sqlite3.Connection, original: str) -> str:
    if not target.execute(
        "SELECT 1 FROM template WHERE name = ?", (original,)
    ).fetchone():
        return original
    suffix = 1
    while True:
        candidate = f"{original} (merged {suffix})"
        if not target.execute(
            "SELECT 1 FROM template WHERE name = ?", (candidate,)
        ).fetchone():
            return candidate
        suffix += 1


def timestamp_key(value: object) -> tuple[int, float, str]:
    """Return an old-to-new key for SQLite/ISO timestamps."""
    if value is None or not str(value).strip():
        return (2, 0.0, "")
    text = str(value).strip()
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (0, parsed.timestamp(), "")
    except ValueError:
        return (1, 0.0, text)


def clear_business_data(target: sqlite3.Connection) -> None:
    """Clear only the copied output; both input databases remain read-only."""
    tables = user_tables(target)
    for table in RUN_CHILD_TABLES:
        if table in tables:
            target.execute(f"DELETE FROM {quote(table)}")
    target.execute("DELETE FROM run")
    target.execute("DELETE FROM task")
    if "template" in tables:
        target.execute("DELETE FROM template")
    has_sequence = target.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type='table' AND name='sqlite_sequence'"
    ).fetchone()
    if has_sequence:
        marks = ", ".join("?" for _ in KNOWN_TABLES)
        target.execute(
            f"DELETE FROM sqlite_sequence WHERE name IN ({marks})",
            tuple(sorted(KNOWN_TABLES)),
        )


def merge_chronologically(
    inputs: list[tuple[str, sqlite3.Connection]],
    target: sqlite3.Connection,
) -> dict[str, int]:
    """Rebuild task/run IDs from oldest to newest in a fresh output copy."""
    for _, conn in inputs:
        require_compatible_schema(conn, target)

    target.execute("PRAGMA foreign_keys=ON")
    target.execute("BEGIN IMMEDIATE")
    try:
        clear_business_data(target)

        task_columns = columns(target, "task")
        task_rows = []
        for origin, conn in inputs:
            for row in conn.execute("SELECT * FROM task ORDER BY id"):
                key = (*timestamp_key(row["created_at"]), origin, int(row["id"]))
                task_rows.append((key, origin, row))
        task_rows.sort(key=lambda item: item[0])

        task_id_map: dict[tuple[str, int], int] = {}
        task_time_map: dict[tuple[str, int], object] = {}
        for _, origin, row in task_rows:
            identity = (origin, int(row["id"]))
            task_id_map[identity] = insert_row(target, "task", row, task_columns)
            task_time_map[identity] = row["created_at"]

        run_columns = columns(target, "run")
        run_rows = []
        for origin, conn in inputs:
            for row in conn.execute("SELECT * FROM run ORDER BY id"):
                task_identity = (origin, int(row["task_id"]))
                effective_time = (
                    row["started_at"]
                    or row["finished_at"]
                    or task_time_map[task_identity]
                )
                key = (*timestamp_key(effective_time), origin, int(row["id"]))
                run_rows.append((key, origin, row))
        run_rows.sort(key=lambda item: item[0])

        run_id_map: dict[tuple[str, int], int] = {}
        for _, origin, row in run_rows:
            identity = (origin, int(row["id"]))
            task_identity = (origin, int(row["task_id"]))
            run_id_map[identity] = insert_row(
                target,
                "run",
                row,
                run_columns,
                {"task_id": task_id_map[task_identity]},
            )

        counts = {"task": len(task_id_map), "run": len(run_id_map)}
        for table in RUN_CHILD_TABLES:
            copied = 0
            target_columns = columns(target, table)
            for origin, conn in inputs:
                if table not in user_tables(conn):
                    continue
                for row in conn.execute(
                    f"SELECT * FROM {quote(table)} ORDER BY id"
                ):
                    identity = (origin, int(row["run_id"]))
                    if identity not in run_id_map:
                        raise RuntimeError(
                            f"{origin}.{table} row {row['id']} references "
                            f"missing run {row['run_id']}"
                        )
                    insert_row(
                        target,
                        table,
                        row,
                        target_columns,
                        {"run_id": run_id_map[identity]},
                    )
                    copied += 1
            counts[table] = copied

        template_count = template_renamed = 0
        if "template" in user_tables(target):
            template_columns = columns(target, "template")
            template_rows = []
            for origin, conn in inputs:
                if "template" not in user_tables(conn):
                    continue
                for row in conn.execute("SELECT * FROM template ORDER BY id"):
                    key = (
                        *timestamp_key(row["created_at"]),
                        origin,
                        int(row["id"]),
                    )
                    template_rows.append((key, origin, row))
            template_rows.sort(key=lambda item: item[0])
            for _, _, row in template_rows:
                name = unique_template_name(target, row["name"])
                template_renamed += int(name != row["name"])
                insert_row(
                    target,
                    "template",
                    row,
                    template_columns,
                    {"name": name},
                )
                template_count += 1
        counts["template"] = template_count
        counts["template_renamed"] = template_renamed

        violations = target.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(
                f"foreign-key validation failed with {len(violations)} violation(s)"
            )
        integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
        target.commit()
        return counts
    except Exception:
        target.rollback()
        raise


def open_source(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def copy_database(source: sqlite3.Connection, destination: Path) -> None:
    """Create an SQLite-consistent copy, including any committed WAL data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination) as copied:
        source.backup(copied)


def prepare_output(path: Path) -> sqlite3.Connection:
    """Open an output safely, including on filesystems unsuitable for WAL."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    mode = conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
    if str(mode).lower() != "delete":
        conn.close()
        raise RuntimeError(f"cannot switch output journal mode to DELETE: {mode}")
    conn.execute("PRAGMA synchronous=FULL")
    return conn


def remove_incomplete_output(path: Path) -> None:
    """Remove only files created for a failed output database."""
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        candidate.unlink(missing_ok=True)


def database_fingerprint(path: Path) -> str:
    stat = path.stat()
    text = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help="base database copied into the new merged database",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="create --output (default: dry run without writing a result)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.source.resolve()
    target_path = args.target.resolve()
    output_path = args.output.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source database does not exist: {source_path}")
    if not target_path.is_file():
        raise FileNotFoundError(f"target database does not exist: {target_path}")
    if source_path == target_path:
        raise ValueError("source and target must be different databases")
    if output_path in (source_path, target_path):
        raise ValueError("output must differ from both source and target")

    print(f"source: {source_path} ({database_fingerprint(source_path)})")
    print(f"base:   {target_path} ({database_fingerprint(target_path)})")
    source = open_source(source_path)
    base = open_source(target_path)
    try:
        if args.apply:
            if output_path.exists():
                raise FileExistsError(f"output already exists: {output_path}")
            copy_database(base, output_path)
            merged = prepare_output(output_path)
            try:
                counts = merge_chronologically(
                    [("base", base), ("source", source)], merged
                )
            except Exception:
                merged.close()
                remove_incomplete_output(output_path)
                raise
            else:
                merged.close()
            print(f"output: {output_path}")
            mode = "APPLIED"
        else:
            with tempfile.TemporaryDirectory(prefix="benchmark-db-merge-") as temp:
                dry_path = Path(temp) / "target-dry-run.db"
                copy_database(base, dry_path)
                target = prepare_output(dry_path)
                try:
                    counts = merge_chronologically(
                        [("base", base), ("source", source)], target
                    )
                finally:
                    target.close()
            mode = "DRY RUN"
    finally:
        base.close()
        source.close()

    print(f"{mode} succeeded")
    for table, count in counts.items():
        print(f"  {table}: {count}")
    if not args.apply:
        print("No database was modified. Re-run with --apply to create --output.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
