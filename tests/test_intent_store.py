"""Tests for the persistent intent store (db/intent_store.py)."""
import json
import sqlite3
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.audit import _create_schema
from db.intent_store import (
    activate_intent,
    deactivate_intent,
    delete_intent,
    fire_intent,
    get_active_intents,
    get_all_intents,
    get_intent,
    store_persistent_intent,
)
from errors import LeavesError, LeavesErrorCode


@pytest.fixture
def db():
    """Create an in-memory SQLite database with the full schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    _create_schema(conn)
    return conn


SAMPLE_GOALSPEC = {
    "intent_id": "test-intent-001",
    "natural_text": "organize PDFs in downloads",
    "category": "file_task",
    "actions": [
        {
            "action_id": "a1",
            "type": "QUERY",
            "agent": "file",
            "params": {"path": "~/Downloads", "pattern": "*.pdf"},
        }
    ],
    "authorization": {"resources": ["~/Downloads"], "preview_required": False},
    "metadata": {},
}


class TestStoreAndRetrieve:
    def test_store_returns_uuid(self, db):
        intent_id = store_persistent_intent(
            db, name="organize PDFs", goalspec=SAMPLE_GOALSPEC, trigger_type="manual"
        )
        assert isinstance(intent_id, str)
        assert len(intent_id) == 36  # UUID format

    def test_retrieve_by_id(self, db):
        intent_id = store_persistent_intent(
            db, name="organize PDFs", goalspec=SAMPLE_GOALSPEC, trigger_type="manual"
        )
        result = get_intent(db, intent_id)
        assert result is not None
        assert result["name"] == "organize PDFs"
        assert result["trigger_type"] == "manual"
        assert result["active"] is True
        assert result["fire_count"] == 0
        assert result["goalspec"]["intent_id"] == "test-intent-001"

    def test_retrieve_nonexistent_returns_none(self, db):
        assert get_intent(db, "nonexistent-id") is None

    def test_store_with_trigger_config(self, db):
        config = {"path": "/home/user/Downloads", "events": ["created"]}
        intent_id = store_persistent_intent(
            db,
            name="watch downloads",
            goalspec=SAMPLE_GOALSPEC,
            trigger_type="filesystem",
            trigger_config=config,
        )
        result = get_intent(db, intent_id)
        assert result["trigger_config"]["path"] == "/home/user/Downloads"
        assert result["trigger_config"]["events"] == ["created"]

    def test_store_invalid_trigger_type_raises(self, db):
        with pytest.raises(LeavesError) as exc_info:
            store_persistent_intent(
                db, name="bad", goalspec=SAMPLE_GOALSPEC, trigger_type="invalid"
            )
        assert exc_info.value.code == LeavesErrorCode.INVALID_INTENT_FORMAT


class TestFireIntent:
    def test_fire_increments_count(self, db):
        intent_id = store_persistent_intent(
            db, name="test", goalspec=SAMPLE_GOALSPEC, trigger_type="manual"
        )

        fire_intent(db, intent_id)
        result = get_intent(db, intent_id)
        assert result["fire_count"] == 1
        assert result["last_fired"] is not None

        fire_intent(db, intent_id)
        result = get_intent(db, intent_id)
        assert result["fire_count"] == 2

    def test_fire_nonexistent_raises(self, db):
        with pytest.raises(LeavesError) as exc_info:
            fire_intent(db, "nonexistent")
        assert exc_info.value.code == LeavesErrorCode.INTENT_NOT_FOUND


class TestActivateDeactivate:
    def test_deactivate(self, db):
        intent_id = store_persistent_intent(
            db, name="test", goalspec=SAMPLE_GOALSPEC, trigger_type="manual"
        )
        deactivate_intent(db, intent_id)
        result = get_intent(db, intent_id)
        assert result["active"] is False

    def test_deactivate_already_inactive_raises(self, db):
        intent_id = store_persistent_intent(
            db, name="test", goalspec=SAMPLE_GOALSPEC, trigger_type="manual"
        )
        deactivate_intent(db, intent_id)
        with pytest.raises(LeavesError) as exc_info:
            deactivate_intent(db, intent_id)
        assert exc_info.value.code == LeavesErrorCode.INTENT_ALREADY_INACTIVE

    def test_reactivate(self, db):
        intent_id = store_persistent_intent(
            db, name="test", goalspec=SAMPLE_GOALSPEC, trigger_type="manual"
        )
        deactivate_intent(db, intent_id)
        activate_intent(db, intent_id)
        result = get_intent(db, intent_id)
        assert result["active"] is True

    def test_deactivate_nonexistent_raises(self, db):
        with pytest.raises(LeavesError) as exc_info:
            deactivate_intent(db, "nonexistent")
        assert exc_info.value.code == LeavesErrorCode.INTENT_NOT_FOUND


class TestGetActiveIntents:
    def test_returns_only_active(self, db):
        id1 = store_persistent_intent(
            db, name="active", goalspec=SAMPLE_GOALSPEC, trigger_type="manual"
        )
        id2 = store_persistent_intent(
            db, name="inactive", goalspec=SAMPLE_GOALSPEC, trigger_type="manual"
        )
        deactivate_intent(db, id2)

        active = get_active_intents(db)
        assert len(active) == 1
        assert active[0]["id"] == id1

    def test_filter_by_trigger_type(self, db):
        store_persistent_intent(
            db, name="manual one", goalspec=SAMPLE_GOALSPEC, trigger_type="manual"
        )
        store_persistent_intent(
            db,
            name="fs one",
            goalspec=SAMPLE_GOALSPEC,
            trigger_type="filesystem",
            trigger_config={"path": "/tmp", "events": ["created"]},
        )

        manual = get_active_intents(db, trigger_type="manual")
        assert len(manual) == 1
        assert manual[0]["name"] == "manual one"

        fs = get_active_intents(db, trigger_type="filesystem")
        assert len(fs) == 1
        assert fs[0]["name"] == "fs one"

    def test_empty_when_none_active(self, db):
        assert get_active_intents(db) == []


class TestGetAllIntents:
    def test_returns_active_and_inactive(self, db):
        id1 = store_persistent_intent(
            db, name="a", goalspec=SAMPLE_GOALSPEC, trigger_type="manual"
        )
        id2 = store_persistent_intent(
            db, name="b", goalspec=SAMPLE_GOALSPEC, trigger_type="manual"
        )
        deactivate_intent(db, id2)

        all_intents = get_all_intents(db)
        assert len(all_intents) == 2


class TestDeleteIntent:
    def test_delete_removes_from_db(self, db):
        intent_id = store_persistent_intent(
            db, name="to delete", goalspec=SAMPLE_GOALSPEC, trigger_type="manual"
        )
        delete_intent(db, intent_id)
        assert get_intent(db, intent_id) is None

    def test_delete_nonexistent_raises(self, db):
        with pytest.raises(LeavesError) as exc_info:
            delete_intent(db, "nonexistent")
        assert exc_info.value.code == LeavesErrorCode.INTENT_NOT_FOUND

    def test_delete_removes_from_active_list(self, db):
        intent_id = store_persistent_intent(
            db, name="to delete", goalspec=SAMPLE_GOALSPEC, trigger_type="manual"
        )
        delete_intent(db, intent_id)
        assert get_active_intents(db) == []
