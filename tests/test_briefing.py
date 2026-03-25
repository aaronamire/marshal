"""Tests for cortex briefing generator and Layer 0 briefing triggers."""
import json
import sqlite3
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.audit import _create_schema
from cortex.briefing import BriefingGenerator, _collapse_home, _fmt_hours, _item_summary
from cortex.knowledge_graph import KnowledgeGraph
from agents.layer0 import match as layer0_match


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    _create_schema(conn)
    return conn


@pytest.fixture
def kg(db, tmp_path):
    return KnowledgeGraph(db, tmp_path / "lance")


@pytest.fixture
def bg(kg):
    return BriefingGenerator(kg)


def _insert_knowledge_item(db, source_path, title, timestamp, source_type="file"):
    """Insert a knowledge_items row directly for testing."""
    import hashlib
    content_hash = hashlib.blake2b(title.encode(), digest_size=16).hexdigest()
    item_id = hashlib.blake2b(
        f"{source_type}:{source_path}:{content_hash}".encode(), digest_size=16
    ).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    metadata = json.dumps({"extension": Path(title).suffix, "size": 100})
    db.execute("""
        INSERT OR REPLACE INTO knowledge_items
        (id, source_type, source_id, source_path, title,
         content_preview, content_hash, metadata_json,
         timestamp, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        item_id, source_type, source_path, source_path, title,
        f"preview of {title}", content_hash, metadata,
        timestamp, now,
    ))
    db.commit()


# ---------------------------------------------------------------------------
# BriefingGenerator
# ---------------------------------------------------------------------------

class TestBriefingEmpty:

    def test_empty_no_data(self, bg):
        briefing = bg.generate(hours=12)
        assert briefing["empty"] is True
        assert briefing["total_changes"] == 0
        assert "No files indexed" in briefing["headline"]

    def test_empty_no_recent_changes(self, db, bg):
        # Insert old item (30 days ago)
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        _insert_knowledge_item(db, "/home/user/old.txt", "old.txt", old_ts)
        briefing = bg.generate(hours=12)
        assert briefing["empty"] is True
        assert "No changes" in briefing["headline"]


class TestBriefingWithData:

    def _populate(self, db, count=5, base_dir="/home/user/dev/project"):
        """Insert recent items."""
        now = datetime.now(timezone.utc)
        for i in range(count):
            ts = (now - timedelta(hours=1, minutes=i)).isoformat()
            path = f"{base_dir}/file{i}.py"
            _insert_knowledge_item(db, path, f"file{i}.py", ts)

    def test_generates_sections(self, db, bg):
        self._populate(db)
        briefing = bg.generate(hours=12)
        assert briefing["empty"] is False
        assert briefing["total_changes"] == 5
        assert len(briefing["sections"]) == 1
        assert briefing["sections"][0]["source_type"] == "file"

    def test_groups_by_directory(self, db, bg):
        now = datetime.now(timezone.utc)
        # 3 files in dir A
        for i in range(3):
            ts = (now - timedelta(hours=1, minutes=i)).isoformat()
            _insert_knowledge_item(db, f"/home/user/dirA/f{i}.py", f"f{i}.py", ts)
        # 3 files in dir B
        for i in range(3):
            ts = (now - timedelta(hours=1, minutes=i)).isoformat()
            _insert_knowledge_item(db, f"/home/user/dirB/f{i}.txt", f"f{i}.txt", ts)

        briefing = bg.generate(hours=12)
        groups = briefing["sections"][0]["groups"]
        dirs = {g["directory"] for g in groups}
        # Both directories should appear as groups (each has >= _MIN_GROUP_SIZE)
        assert any("dirA" in d for d in dirs)
        assert any("dirB" in d for d in dirs)

    def test_collapses_small_groups_to_other(self, db, bg):
        now = datetime.now(timezone.utc)
        # 5 files in one dir (big group)
        for i in range(5):
            ts = (now - timedelta(hours=1, minutes=i)).isoformat()
            _insert_knowledge_item(db, f"/home/user/main/f{i}.py", f"f{i}.py", ts)
        # 1 file in another dir (too small → "other")
        ts = (now - timedelta(hours=1)).isoformat()
        _insert_knowledge_item(db, "/home/user/misc/lone.txt", "lone.txt", ts)

        briefing = bg.generate(hours=12)
        groups = briefing["sections"][0]["groups"]
        group_dirs = {g["directory"] for g in groups}
        assert "other" in group_dirs

    def test_headline_single_source(self, db, bg):
        self._populate(db)
        briefing = bg.generate(hours=12)
        assert "file" in briefing["headline"]
        assert "5" in briefing["headline"]

    def test_headline_multiple_sources(self, db, bg):
        now = datetime.now(timezone.utc)
        ts = (now - timedelta(hours=1)).isoformat()
        _insert_knowledge_item(db, "/home/user/a.py", "a.py", ts, source_type="file")
        _insert_knowledge_item(db, "/home/user/b.py", "b.py", ts, source_type="file")
        _insert_knowledge_item(db, "msg-123", "Email subject", ts, source_type="email")
        _insert_knowledge_item(db, "msg-456", "Another email", ts, source_type="email")

        briefing = bg.generate(hours=12)
        assert "4 changes" in briefing["headline"]

    def test_items_in_groups_have_required_fields(self, db, bg):
        self._populate(db)
        briefing = bg.generate(hours=12)
        for section in briefing["sections"]:
            for group in section["groups"]:
                for item in group["items"]:
                    assert "title" in item
                    assert "path" in item
                    assert "timestamp" in item

    def test_period_hours_in_output(self, db, bg):
        self._populate(db)
        briefing = bg.generate(hours=6)
        assert briefing["period_hours"] == 6

    def test_generated_at_present(self, db, bg):
        self._populate(db)
        briefing = bg.generate(hours=12)
        assert "generated_at" in briefing


class TestBriefingStatus:

    def test_status_empty(self, bg):
        s = bg.status()
        assert s["indexed"] is False
        assert s["total_items"] == 0

    def test_status_with_data(self, db, bg):
        ts = datetime.now(timezone.utc).isoformat()
        _insert_knowledge_item(db, "/tmp/a.txt", "a.txt", ts)
        s = bg.status()
        assert s["indexed"] is True
        assert s["total_items"] == 1


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_collapse_home(self):
        home = str(Path.home())
        assert _collapse_home(f"{home}/Documents") == "~/Documents"
        assert _collapse_home("/etc/config") == "/etc/config"

    def test_fmt_hours(self):
        assert _fmt_hours(1) == "hour"
        assert _fmt_hours(6) == "6 hours"
        assert _fmt_hours(24) == "day"
        assert _fmt_hours(48) == "2 days"

    def test_item_summary(self):
        row = {
            "title": "test.py",
            "source_path": "/home/user/test.py",
            "timestamp": "2026-03-25T10:00:00+00:00",
            "metadata_json": '{"extension": ".py", "size": 500}',
        }
        s = _item_summary(row)
        assert s["title"] == "test.py"
        assert s["extension"] == ".py"

    def test_item_summary_bad_metadata(self):
        row = {
            "title": "test.py",
            "source_path": "/tmp/test.py",
            "timestamp": "2026-03-25T10:00:00+00:00",
            "metadata_json": "not-json",
        }
        s = _item_summary(row)
        assert s["extension"] == ""


# ---------------------------------------------------------------------------
# Layer 0 briefing triggers
# ---------------------------------------------------------------------------

class TestLayer0BriefingTriggers:

    @pytest.mark.parametrize("text", [
        "good morning",
        "Good Morning",
        "morning",
        "briefing",
        "brief me",
        "what changed today",
        "what's changed",
        "what has changed since yesterday",
        "what happened today",
        "what's new",
        "what is new",
        "catch me up",
    ])
    def test_briefing_triggers_match(self, text):
        result = layer0_match(text)
        assert result.matched, f"Expected match for: {text}"
        assert result.agent == "briefing"
        assert result.category == "briefing"
        assert result.action_type == "BRIEFING"

    @pytest.mark.parametrize("text", [
        "good morning sir how are you",  # too verbose, not a clean trigger
        "list ~/Downloads",              # file operation
        "open firefox",                  # app launch
        "how much memory",               # system query
    ])
    def test_non_briefing_not_matched_as_briefing(self, text):
        result = layer0_match(text)
        if result.matched:
            assert result.agent != "briefing"
