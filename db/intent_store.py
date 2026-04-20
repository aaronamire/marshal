"""
Persistent intent store — GoalSpecs that survive reboot.

Three trigger types:
- manual:     fires only when user explicitly triggers it
- filesystem: fires when inotify detects matching events on a watched path
- schedule:   fires on a cron-like schedule (stub for now)
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from errors import MarshalError, MarshalErrorCode


def store_persistent_intent(
    db: sqlite3.Connection,
    name: str,
    goalspec: dict[str, Any],
    trigger_type: str,
    trigger_config: Optional[dict[str, Any]] = None,
) -> str:
    """Store a persistent intent and return its ID."""
    intent_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    if trigger_type not in ("manual", "filesystem", "schedule"):
        raise MarshalError(
            MarshalErrorCode.INVALID_INTENT_FORMAT,
            detail=f"Invalid trigger_type: {trigger_type}",
        )

    try:
        db.execute(
            """
            INSERT INTO persistent_intents
                (id, name, goalspec_json, trigger_type, trigger_config, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                intent_id,
                name,
                json.dumps(goalspec),
                trigger_type,
                json.dumps(trigger_config) if trigger_config else None,
                now,
            ),
        )
        db.commit()
    except sqlite3.Error as e:
        raise MarshalError(MarshalErrorCode.DB_ERROR, detail=str(e), cause=e)

    return intent_id


def get_active_intents(
    db: sqlite3.Connection,
    trigger_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return all active persistent intents, optionally filtered by trigger type."""
    try:
        if trigger_type:
            rows = db.execute(
                "SELECT * FROM persistent_intents WHERE active = 1 AND trigger_type = ? "
                "ORDER BY created_at DESC",
                (trigger_type,),
            ).fetchall()
        else:
            rows = db.execute(
                "SELECT * FROM persistent_intents WHERE active = 1 "
                "ORDER BY created_at DESC",
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except sqlite3.Error as e:
        raise MarshalError(MarshalErrorCode.DB_ERROR, detail=str(e), cause=e)


def get_all_intents(db: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all persistent intents (active and inactive)."""
    try:
        rows = db.execute(
            "SELECT * FROM persistent_intents ORDER BY created_at DESC"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except sqlite3.Error as e:
        raise MarshalError(MarshalErrorCode.DB_ERROR, detail=str(e), cause=e)


def get_intent(db: sqlite3.Connection, intent_id: str) -> Optional[dict[str, Any]]:
    """Return a single persistent intent by ID, or None."""
    try:
        row = db.execute(
            "SELECT * FROM persistent_intents WHERE id = ?", (intent_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None
    except sqlite3.Error as e:
        raise MarshalError(MarshalErrorCode.DB_ERROR, detail=str(e), cause=e)


def fire_intent(db: sqlite3.Connection, intent_id: str) -> None:
    """Record that a persistent intent has fired (update last_fired + fire_count)."""
    intent = get_intent(db, intent_id)
    if intent is None:
        raise MarshalError(MarshalErrorCode.INTENT_NOT_FOUND, detail=intent_id)

    now = datetime.now(timezone.utc).isoformat()
    try:
        db.execute(
            "UPDATE persistent_intents SET last_fired = ?, fire_count = fire_count + 1 "
            "WHERE id = ?",
            (now, intent_id),
        )
        db.commit()
    except sqlite3.Error as e:
        raise MarshalError(MarshalErrorCode.DB_ERROR, detail=str(e), cause=e)


def deactivate_intent(db: sqlite3.Connection, intent_id: str) -> None:
    """Pause a persistent intent."""
    intent = get_intent(db, intent_id)
    if intent is None:
        raise MarshalError(MarshalErrorCode.INTENT_NOT_FOUND, detail=intent_id)
    if not intent["active"]:
        raise MarshalError(MarshalErrorCode.INTENT_ALREADY_INACTIVE, detail=intent_id)

    try:
        db.execute(
            "UPDATE persistent_intents SET active = 0 WHERE id = ?", (intent_id,)
        )
        db.commit()
    except sqlite3.Error as e:
        raise MarshalError(MarshalErrorCode.DB_ERROR, detail=str(e), cause=e)


def activate_intent(db: sqlite3.Connection, intent_id: str) -> None:
    """Resume a paused persistent intent."""
    intent = get_intent(db, intent_id)
    if intent is None:
        raise MarshalError(MarshalErrorCode.INTENT_NOT_FOUND, detail=intent_id)

    try:
        db.execute(
            "UPDATE persistent_intents SET active = 1 WHERE id = ?", (intent_id,)
        )
        db.commit()
    except sqlite3.Error as e:
        raise MarshalError(MarshalErrorCode.DB_ERROR, detail=str(e), cause=e)


def delete_intent(db: sqlite3.Connection, intent_id: str) -> None:
    """Permanently delete a persistent intent."""
    intent = get_intent(db, intent_id)
    if intent is None:
        raise MarshalError(MarshalErrorCode.INTENT_NOT_FOUND, detail=intent_id)

    try:
        db.execute("DELETE FROM persistent_intents WHERE id = ?", (intent_id,))
        db.commit()
    except sqlite3.Error as e:
        raise MarshalError(MarshalErrorCode.DB_ERROR, detail=str(e), cause=e)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row to a dict, parsing JSON fields."""
    d = dict(row)
    if d.get("goalspec_json"):
        d["goalspec"] = json.loads(d["goalspec_json"])
    if d.get("trigger_config"):
        d["trigger_config"] = json.loads(d["trigger_config"])
    d["active"] = bool(d.get("active", 0))
    return d
