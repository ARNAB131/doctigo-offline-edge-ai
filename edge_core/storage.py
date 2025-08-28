# edge_core/storage.py
from __future__ import annotations
from pathlib import Path
import sqlite3
from contextlib import contextmanager

SCHEMA_VERSION = 1

DDL = [
    # 1) vitals
    """CREATE TABLE IF NOT EXISTS vitals(
        patient_id TEXT NOT NULL,
        ts         TEXT NOT NULL,           -- ISO timestamp
        sensor     TEXT NOT NULL,           -- ECG, SpO2, BP_SYS, BP_DIA, etc.
        value      REAL NOT NULL,
        unit       TEXT DEFAULT '',
        source     TEXT DEFAULT 'edge'
    )""",
    "CREATE INDEX IF NOT EXISTS idx_vitals_pid_ts ON vitals(patient_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_vitals_sensor_ts ON vitals(sensor, ts)",

    # 2) alerts
    """CREATE TABLE IF NOT EXISTS alerts(
        id        TEXT PRIMARY KEY,         -- e.g., uuid4
        patient_id TEXT NOT NULL,
        ts         TEXT NOT NULL,
        type       TEXT NOT NULL,           -- rule/ad/model/etc
        severity   TEXT NOT NULL,           -- info/warn/critical
        payload    TEXT                     -- JSON string
    )""",
    "CREATE INDEX IF NOT EXISTS idx_alerts_pid_ts ON alerts(patient_id, ts)",

    # 3) models registry
    """CREATE TABLE IF NOT EXISTS models(
        name       TEXT NOT NULL,
        version    TEXT NOT NULL,
        sha        TEXT NOT NULL,
        created_ts TEXT NOT NULL,
        meta       TEXT,                    -- JSON
        path       TEXT NOT NULL
    )""",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_models_name_ver ON models(name, version)",

    # 4) audit trail
    """CREATE TABLE IF NOT EXISTS audit(
        ts     TEXT NOT NULL,
        actor  TEXT NOT NULL,               -- username/role
        action TEXT NOT NULL,               -- e.g., 'login','ack_alert'
        details TEXT                        -- JSON/message
    )""",

    # 5) migrations
    """CREATE TABLE IF NOT EXISTS _meta(
        key TEXT PRIMARY KEY,
        val TEXT
    )""",
]

def _apply_pragmas(conn: sqlite3.Connection) -> None:
    # WAL for concurrency; sync=normal for better perf on laptops; foreign_keys off (no FKs here)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=OFF;")

def _get_schema_version(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT val FROM _meta WHERE key='schema_version'")
    row = cur.fetchone()
    return int(row[0]) if row else 0

def _set_schema_version(conn: sqlite3.Connection, ver: int) -> None:
    conn.execute("INSERT OR REPLACE INTO _meta(key,val) VALUES('schema_version', ?)", (str(ver),))

def _migrate(conn: sqlite3.Connection, current: int) -> None:
    # Add future migrations here (current -> SCHEMA_VERSION).
    # Example scaffold:
    # if current < 2:
    #     conn.execute("ALTER TABLE vitals ADD COLUMN quality REAL")
    #     current = 2
    _set_schema_version(conn, SCHEMA_VERSION)

def init_sqlite_if_needed(db_path: str) -> None:
    """Create DB file/folders, apply schema + pragmas + migrations."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _apply_pragmas(conn)
        for stmt in DDL:
            conn.execute(stmt)
        conn.commit()
        ver = _get_schema_version(conn)
        if ver < SCHEMA_VERSION:
            _migrate(conn, ver)
            conn.commit()

@contextmanager
def get_conn(db_path: str):
    """Context manager with PRAGMAs applied."""
    conn = sqlite3.connect(db_path, timeout=10, isolation_level=None)  # autocommit-like
    try:
        _apply_pragmas(conn)
        yield conn
    finally:
        conn.close()
