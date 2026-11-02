"""Compact storage for the GLM online session replay dataset.

The export contains one very large JSON object per session and repeats most of
the message history in every span.  This store compresses each session
independently, which keeps random session access while allowing zlib to remove
the repeated history.  Only fields required to issue the request are retained.
"""

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
import time
import zlib
from datetime import datetime
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/dfs/liuxing/tokenservice_data_export/"
    "2026-08-08-11:00-12:00.out.session.jsonl"
)
CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _configured_store_path() -> Path:
    """Read the replay store path from the project YAML configuration."""
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Missing configuration file: {CONFIG_PATH}")
    try:
        import yaml
        with open(CONFIG_PATH, encoding="utf-8") as stream:
            configured = (yaml.safe_load(stream) or {}).get("replay_dataset_path")
    except ImportError:
        configured = None
        with open(CONFIG_PATH, encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("replay_dataset_path:"):
                    configured = line.partition(":")[2].strip().strip('"\'')
                    break
    if not configured:
        raise RuntimeError("config.yaml must define replay_dataset_path")
    return Path(configured).expanduser()


DEFAULT_STORE = _configured_store_path()


def parse_timestamp(value: str) -> float:
    text = value[:-1] if value.endswith("Z") else value
    timezone = "+00:00" if value.endswith("Z") else ""
    if "." in text:
        prefix, fraction = text.rsplit(".", 1)
        text = prefix + "." + fraction[:6].ljust(6, "0")
    return datetime.fromisoformat(text + timezone).timestamp()


def _is_current(conn: sqlite3.Connection, source: Path) -> bool:
    try:
        values = dict(conn.execute("SELECT key, value FROM metadata"))
    except sqlite3.Error:
        return False
    stat = source.stat()
    return (
        values.get("source_size") == str(stat.st_size)
        and values.get("source_mtime_ns") == str(stat.st_mtime_ns)
        and values.get("format") == "glm-session-replay-v1"
    )


def build_store(source=DEFAULT_SOURCE, destination=DEFAULT_STORE) -> Path:
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        with sqlite3.connect(destination) as conn:
            if _is_current(conn, source):
                return destination

    fd, temporary_name = tempfile.mkstemp(prefix="glm51-replay-", suffix=".sqlite")
    os.close(fd)
    temporary = Path(temporary_name)
    conn = sqlite3.connect(temporary)
    try:
        conn.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                start_ts REAL NOT NULL,
                end_ts REAL NOT NULL,
                span_count INTEGER NOT NULL,
                payload BLOB NOT NULL
            );
            CREATE TABLE events (
                session_id TEXT NOT NULL,
                turn INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                PRIMARY KEY (session_id, turn)
            );
            """
        )
        session_count = span_count = 0
        started = time.perf_counter()
        with open(source, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                session_id = raw.get("session_id") or raw.get("traceid")
                compact_spans = []
                timestamps = []
                for turn, span in enumerate(raw.get("spans") or []):
                    timestamp_text = span.get("@timestamp")
                    if not timestamp_text:
                        continue
                    timestamp = parse_timestamp(timestamp_text)
                    timestamps.append(timestamp)
                    compact_spans.append({
                        "timestamp": timestamp,
                        "spanid": span.get("spanid"),
                        "chat_id": span.get("chat_id"),
                        "request": span.get("request") or {},
                        "output_token": (span.get("metrics") or {}).get("output_token"),
                    })
                if not compact_spans:
                    continue
                payload = zlib.compress(
                    json.dumps(compact_spans, ensure_ascii=False, separators=(",", ":")).encode(),
                    level=6,
                )
                conn.execute(
                    "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
                    (session_id, min(timestamps), max(timestamps), len(compact_spans), payload),
                )
                conn.executemany(
                    "INSERT INTO events VALUES (?, ?, ?)",
                    ((session_id, turn, span["timestamp"])
                     for turn, span in enumerate(compact_spans)),
                )
                session_count += 1
                span_count += len(compact_spans)
                if line_number % 50 == 0:
                    conn.commit()
                    print(
                        f"[replay store] sessions={session_count} spans={span_count} "
                        f"elapsed={time.perf_counter() - started:.1f}s"
                    )
        stat = source.stat()
        conn.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [
                ("format", "glm-session-replay-v1"),
                ("source", str(source)),
                ("source_size", str(stat.st_size)),
                ("source_mtime_ns", str(stat.st_mtime_ns)),
                ("session_count", str(session_count)),
                ("span_count", str(span_count)),
            ],
        )
        conn.execute("CREATE INDEX events_timestamp_idx ON events(timestamp)")
        conn.commit()
    except Exception:
        conn.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        conn.close()
    destination_tmp = Path(str(destination) + f".tmp.{os.getpid()}")
    shutil.copyfile(temporary, destination_tmp)
    os.replace(destination_tmp, destination)
    temporary.unlink(missing_ok=True)
    return destination


class ReplayStore:
    def __init__(self, path=DEFAULT_STORE):
        self.path = Path(path)
        self.conn = sqlite3.connect(
            f"file:{self.path}?mode=ro", uri=True, check_same_thread=False
        )

    def close(self):
        self.conn.close()

    def metadata(self):
        return dict(self.conn.execute("SELECT key, value FROM metadata"))

    def session_ids(self, ordered=False):
        query = "SELECT session_id FROM sessions"
        if ordered:
            query += " ORDER BY start_ts, session_id"
        return [row[0] for row in self.conn.execute(query)]

    def load_session(self, session_id):
        row = self.conn.execute(
            "SELECT span_count, payload FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not row:
            raise KeyError(session_id)
        return json.loads(zlib.decompress(row[1]))

    def events_for_sessions(self, session_ids):
        if not session_ids:
            return []
        result = []
        for start in range(0, len(session_ids), 500):
            batch = session_ids[start:start + 500]
            marks = ",".join("?" for _ in batch)
            result.extend(self.conn.execute(
                f"SELECT session_id, turn, timestamp FROM events "
                f"WHERE session_id IN ({marks})",
                batch,
            ))
        return sorted(result, key=lambda row: (row[2], row[0], row[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", default=str(DEFAULT_STORE))
    arguments = parser.parse_args()
    print(build_store(arguments.source, arguments.output))
