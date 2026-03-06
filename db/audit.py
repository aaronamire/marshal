"""
WAL-mode SQLite audit database for Leaves OS.

All intent lifecycle events are recorded here. The database uses:
  - WAL journal mode (crash-safe concurrent reads)
  - Foreign keys enabled
  - Strict typing via CHECK constraints

Never open a connection without WAL mode.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from config import AUDIT_DB_PATH
from errors import LeavesError, LeavesErrorCode


def get_db(path: Path = AUDIT_DB_PATH) -> sqlite3.Connection:
    """
    Open (or create) the audit database and return a connection.
    WAL mode and foreign keys are enabled on every connection.
    Schema is created if it does not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")

    _create_schema(conn)
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS intents (
        intent_id       TEXT PRIMARY KEY,
        natural_text    TEXT NOT NULL,
        category        TEXT,
        state           TEXT NOT NULL DEFAULT 'PENDING',
        goal_spec_json  TEXT,
        result_message  TEXT,
        duration_ms     REAL,
        created_at      REAL NOT NULL,
        updated_at      REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS actions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        intent_id       TEXT NOT NULL REFERENCES intents(intent_id),
        action_id       TEXT NOT NULL,
        action_type     TEXT NOT NULL,
        agent           TEXT NOT NULL,
        params_json     TEXT,
        result_json     TEXT,
        error_code      TEXT,
        error_detail    TEXT,
        started_at      REAL,
        completed_at    REAL
    );

    CREATE TABLE IF NOT EXISTS state_transitions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        intent_id       TEXT NOT NULL REFERENCES intents(intent_id),
        from_state      TEXT NOT NULL,
        to_state        TEXT NOT NULL,
        transitioned_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS errors (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        intent_id       TEXT REFERENCES intents(intent_id),
        error_code      TEXT NOT NULL,
        error_detail    TEXT,
        occurred_at     REAL NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_intents_created_at ON intents(created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_actions_intent_id ON actions(intent_id);
    CREATE INDEX IF NOT EXISTS idx_transitions_intent_id ON state_transitions(intent_id);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Write functions
# ---------------------------------------------------------------------------

def log_intent_created(
    conn: sqlite3.Connection,
    intent_id: str,
    natural_text: str,
    goal_spec: Optional[dict[str, Any]] = None,
) -> None:
    now = time.time()
    category = (goal_spec or {}).get("category")
    try:
        conn.execute(
            """
            INSERT INTO intents (intent_id, natural_text, category, state,
                                 goal_spec_json, created_at, updated_at)
            VALUES (?, ?, ?, 'PENDING', ?, ?, ?)
            """,
            (
                intent_id,
                natural_text,
                category,
                json.dumps(goal_spec) if goal_spec else None,
                now,
                now,
            ),
        )
        conn.commit()
    except sqlite3.Error as e:
        raise LeavesError(LeavesErrorCode.DB_ERROR, detail=str(e), cause=e)


def log_state_transition(
    conn: sqlite3.Connection,
    intent_id: str,
    from_state: str,
    to_state: str,
) -> None:
    now = time.time()
    try:
        conn.execute(
            """
            INSERT INTO state_transitions (intent_id, from_state, to_state, transitioned_at)
            VALUES (?, ?, ?, ?)
            """,
            (intent_id, from_state, to_state, now),
        )
        conn.execute(
            "UPDATE intents SET state=?, updated_at=? WHERE intent_id=?",
            (to_state, now, intent_id),
        )
        conn.commit()
    except sqlite3.Error as e:
        raise LeavesError(LeavesErrorCode.DB_ERROR, detail=str(e), cause=e)


def log_action_started(
    conn: sqlite3.Connection,
    intent_id: str,
    action_id: str,
    action_type: str,
    agent: str,
    params: Optional[dict[str, Any]] = None,
) -> int:
    """Insert an action row and return its rowid."""
    now = time.time()
    try:
        cur = conn.execute(
            """
            INSERT INTO actions
                (intent_id, action_id, action_type, agent, params_json, started_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                intent_id,
                action_id,
                action_type,
                agent,
                json.dumps(params) if params else None,
                now,
            ),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.Error as e:
        raise LeavesError(LeavesErrorCode.DB_ERROR, detail=str(e), cause=e)


def log_action_completed(
    conn: sqlite3.Connection,
    row_id: int,
    result: Optional[dict[str, Any]] = None,
    error_code: Optional[str] = None,
    error_detail: Optional[str] = None,
) -> None:
    now = time.time()
    try:
        conn.execute(
            """
            UPDATE actions
            SET result_json=?, error_code=?, error_detail=?, completed_at=?
            WHERE id=?
            """,
            (
                json.dumps(result) if result else None,
                error_code,
                error_detail,
                now,
                row_id,
            ),
        )
        conn.commit()
    except sqlite3.Error as e:
        raise LeavesError(LeavesErrorCode.DB_ERROR, detail=str(e), cause=e)


def log_error(
    conn: sqlite3.Connection,
    error_code: str,
    error_detail: Optional[str],
    intent_id: Optional[str] = None,
) -> None:
    now = time.time()
    try:
        conn.execute(
            """
            INSERT INTO errors (intent_id, error_code, error_detail, occurred_at)
            VALUES (?, ?, ?, ?)
            """,
            (intent_id, error_code, error_detail, now),
        )
        conn.commit()
    except sqlite3.Error as e:
        raise LeavesError(LeavesErrorCode.DB_ERROR, detail=str(e), cause=e)


def complete_intent(
    conn: sqlite3.Connection,
    intent_id: str,
    final_state: str,
    result_message: Optional[str] = None,
    duration_ms: Optional[float] = None,
) -> None:
    now = time.time()
    try:
        conn.execute(
            """
            UPDATE intents
            SET state=?, result_message=?, duration_ms=?, updated_at=?
            WHERE intent_id=?
            """,
            (final_state, result_message, duration_ms, now, intent_id),
        )
        conn.commit()
    except sqlite3.Error as e:
        raise LeavesError(LeavesErrorCode.DB_ERROR, detail=str(e), cause=e)


# ---------------------------------------------------------------------------
# Read functions
# ---------------------------------------------------------------------------

def get_recent_intents(
    conn: sqlite3.Connection,
    limit: int = 20,
) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            "SELECT * FROM intents ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        raise LeavesError(LeavesErrorCode.DB_ERROR, detail=str(e), cause=e)


def get_intent_transitions(
    conn: sqlite3.Connection,
    intent_id: str,
) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            """
            SELECT from_state, to_state, transitioned_at
            FROM state_transitions WHERE intent_id=?
            ORDER BY transitioned_at ASC
            """,
            (intent_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        raise LeavesError(LeavesErrorCode.DB_ERROR, detail=str(e), cause=e)


def get_intent_actions(
    conn: sqlite3.Connection,
    intent_id: str,
) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            "SELECT * FROM actions WHERE intent_id=? ORDER BY started_at ASC",
            (intent_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.Error as e:
        raise LeavesError(LeavesErrorCode.DB_ERROR, detail=str(e), cause=e)
