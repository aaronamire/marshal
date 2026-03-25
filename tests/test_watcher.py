"""Tests for the filesystem watcher (cortex/watcher.py)."""
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.audit import _create_schema
from db.intent_store import store_persistent_intent
from cortex.watcher import IntentWatcher


SAMPLE_GOALSPEC = {
    "intent_id": "watcher-test-001",
    "natural_text": "organize PDFs",
    "category": "file_task",
    "actions": [{"action_id": "a1", "type": "QUERY", "agent": "file", "params": {}}],
    "authorization": {"resources": [], "preview_required": False},
    "metadata": {},
}


@pytest.fixture
def db_env(tmp_path):
    """Create a real SQLite DB file + connection (needed for cross-thread access)."""
    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _create_schema(conn)
    return conn, db_file


@pytest.fixture
def db(db_env):
    return db_env[0]


@pytest.fixture
def db_path(db_env):
    return db_env[1]


@pytest.fixture
def watch_dir(tmp_path):
    """Create a directory that will be watched."""
    d = tmp_path / "watched"
    d.mkdir()
    return d


class TestWatcherLifecycle:
    def test_start_stop_no_intents(self, db, db_path):
        on_fire = MagicMock()
        watcher = IntentWatcher(db, on_fire, db_path=db_path)
        watcher.start()
        watcher.stop()
        on_fire.assert_not_called()

    def test_start_stop_with_intent(self, db, db_path, watch_dir):
        store_persistent_intent(
            db, name="test watcher", goalspec=SAMPLE_GOALSPEC,
            trigger_type="filesystem",
            trigger_config={"path": str(watch_dir), "events": ["created"]},
        )
        on_fire = MagicMock()
        watcher = IntentWatcher(db, on_fire, db_path=db_path)
        watcher.start()
        time.sleep(0.2)
        watcher.stop()

    def test_reload_picks_up_new_intents(self, db, db_path, watch_dir):
        on_fire = MagicMock()
        watcher = IntentWatcher(db, on_fire, db_path=db_path)
        watcher.start()

        store_persistent_intent(
            db, name="added later", goalspec=SAMPLE_GOALSPEC,
            trigger_type="filesystem",
            trigger_config={"path": str(watch_dir), "events": ["created"]},
        )
        watcher.reload()
        time.sleep(0.3)

        (watch_dir / "test.txt").write_text("hello")
        time.sleep(1.0)

        watcher.stop()
        assert on_fire.call_count >= 1


class TestWatcherDetection:
    def test_detects_file_creation(self, db, db_path, watch_dir):
        store_persistent_intent(
            db, name="detect create", goalspec=SAMPLE_GOALSPEC,
            trigger_type="filesystem",
            trigger_config={"path": str(watch_dir), "events": ["created"]},
        )

        on_fire = MagicMock()
        watcher = IntentWatcher(db, on_fire, db_path=db_path)
        watcher.start()
        time.sleep(0.3)

        (watch_dir / "newfile.txt").write_text("content")
        time.sleep(1.0)

        watcher.stop()
        assert on_fire.call_count >= 1
        fired_goalspec = on_fire.call_args[0][0]
        assert fired_goalspec["intent_id"] == "watcher-test-001"

    def test_pattern_filter(self, db, db_path, watch_dir):
        store_persistent_intent(
            db, name="pdf only", goalspec=SAMPLE_GOALSPEC,
            trigger_type="filesystem",
            trigger_config={
                "path": str(watch_dir), "events": ["created"], "pattern": "*.pdf",
            },
        )

        on_fire = MagicMock()
        watcher = IntentWatcher(db, on_fire, db_path=db_path)
        watcher.start()
        time.sleep(0.3)

        # Wrong extension — should NOT trigger
        (watch_dir / "readme.txt").write_text("not a PDF")
        time.sleep(0.5)
        assert on_fire.call_count == 0

        # Correct extension — SHOULD trigger
        (watch_dir / "report.pdf").write_bytes(b"%PDF-1.4")
        time.sleep(1.0)

        watcher.stop()
        assert on_fire.call_count >= 1

    def test_ignores_inactive_intents(self, db, db_path, watch_dir):
        from db.intent_store import deactivate_intent

        intent_id = store_persistent_intent(
            db, name="deactivated", goalspec=SAMPLE_GOALSPEC,
            trigger_type="filesystem",
            trigger_config={"path": str(watch_dir), "events": ["created"]},
        )
        deactivate_intent(db, intent_id)

        on_fire = MagicMock()
        watcher = IntentWatcher(db, on_fire, db_path=db_path)
        watcher.start()
        time.sleep(0.3)

        (watch_dir / "file.txt").write_text("ignored")
        time.sleep(1.0)

        watcher.stop()
        on_fire.assert_not_called()

    def test_ignores_directory_events(self, db, db_path, watch_dir):
        store_persistent_intent(
            db, name="files only", goalspec=SAMPLE_GOALSPEC,
            trigger_type="filesystem",
            trigger_config={"path": str(watch_dir), "events": ["created"]},
        )

        on_fire = MagicMock()
        watcher = IntentWatcher(db, on_fire, db_path=db_path)
        watcher.start()
        time.sleep(0.3)

        (watch_dir / "subdir").mkdir()
        time.sleep(1.0)

        watcher.stop()
        on_fire.assert_not_called()


class TestDebounce:
    def test_rapid_events_debounced(self, db, db_path, watch_dir):
        store_persistent_intent(
            db, name="debounce test", goalspec=SAMPLE_GOALSPEC,
            trigger_type="filesystem",
            trigger_config={"path": str(watch_dir), "events": ["created"]},
        )

        on_fire = MagicMock()
        watcher = IntentWatcher(db, on_fire, db_path=db_path)
        watcher.start()
        time.sleep(0.3)

        for i in range(5):
            (watch_dir / f"file{i}.txt").write_text(f"content {i}")
            time.sleep(0.05)

        time.sleep(1.0)
        watcher.stop()

        # Should fire only once due to 5s debounce
        assert on_fire.call_count == 1


class TestFireIntegration:
    def test_fire_updates_db(self, db, db_path, watch_dir):
        from db.intent_store import get_intent

        intent_id = store_persistent_intent(
            db, name="fire test", goalspec=SAMPLE_GOALSPEC,
            trigger_type="filesystem",
            trigger_config={"path": str(watch_dir), "events": ["created"]},
        )

        on_fire = MagicMock()
        watcher = IntentWatcher(db, on_fire, db_path=db_path)
        watcher.start()
        time.sleep(0.3)

        (watch_dir / "trigger.txt").write_text("go")
        time.sleep(1.0)

        watcher.stop()

        # The watcher writes to its own DB connection, so we need to re-read
        # from the file to see the cross-thread write (WAL mode)
        check_conn = sqlite3.connect(str(db_path))
        check_conn.row_factory = sqlite3.Row
        row = check_conn.execute(
            "SELECT fire_count, last_fired FROM persistent_intents WHERE id = ?",
            (intent_id,),
        ).fetchone()
        check_conn.close()
        assert row["fire_count"] == 1
        assert row["last_fired"] is not None
