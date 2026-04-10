"""Tests for the SessionMemory ring buffer + persistence."""
from __future__ import annotations

import json
import time
from pathlib import Path

from agents.session_memory import SessionMemory


def _gs(action_id: str = "act-1") -> dict:
    return {
        "intent_id": "i-1",
        "actions": [
            {"action_id": action_id, "type": "QUERY", "agent": "file",
             "params": {"path": "~"}},
        ],
    }


def test_record_and_latest(tmp_path: Path):
    sm = SessionMemory(path=tmp_path / "session.jsonl", capacity=4)
    assert sm.latest() is None
    sm.record(
        intent_id="i-1",
        natural_text="find pdfs",
        goal_spec=_gs(),
        results={"act-1": {"files": [{"path": "/x.pdf"}], "count": 1}},
        summary="Found 1 file.",
    )
    last = sm.latest()
    assert last is not None
    assert last.intent_id == "i-1"
    assert last.action_ids == ["act-1"]
    assert last.results["act-1"]["count"] == 1


def test_capacity_evicts_oldest(tmp_path: Path):
    sm = SessionMemory(path=tmp_path / "s.jsonl", capacity=3)
    for i in range(5):
        sm.record(f"i-{i}", f"t{i}", _gs(), {"act-1": {"n": i}}, f"s{i}")
    assert len(sm) == 3
    ids = [t.intent_id for t in sm.recent(10)]
    assert ids == ["i-2", "i-3", "i-4"]


def test_persistence_round_trip(tmp_path: Path):
    p = tmp_path / "s.jsonl"
    a = SessionMemory(path=p, capacity=4)
    a.record("i-1", "first", _gs(), {"act-1": {"v": 1}}, "ok")
    a.record("i-2", "second", _gs(), {"act-1": {"v": 2}}, "ok")
    # Re-load from disk
    b = SessionMemory(path=p, capacity=4)
    assert len(b) == 2
    assert [t.natural_text for t in b.recent(10)] == ["first", "second"]


def test_session_window_drops_stale(tmp_path: Path):
    p = tmp_path / "s.jsonl"
    sm = SessionMemory(path=p, capacity=4, session_window=3600.0)
    sm.record("i-old", "old", _gs(), {"act-1": {}}, "")
    # Rewrite the file with an ancient timestamp
    raw = p.read_text().strip()
    obj = json.loads(raw)
    obj["ts"] = time.time() - 7200  # 2h ago, > 1h window
    p.write_text(json.dumps(obj) + "\n")
    sm2 = SessionMemory(path=p, capacity=4, session_window=3600.0)
    assert len(sm2) == 0


def test_turn_by_back_index(tmp_path: Path):
    sm = SessionMemory(path=tmp_path / "s.jsonl", capacity=4)
    sm.record("i-1", "t1", _gs(), {"act-1": {"v": 1}}, "")
    sm.record("i-2", "t2", _gs(), {"act-1": {"v": 2}}, "")
    sm.record("i-3", "t3", _gs(), {"act-1": {"v": 3}}, "")
    assert sm.turn_by_back_index(1).intent_id == "i-3"
    assert sm.turn_by_back_index(2).intent_id == "i-2"
    assert sm.turn_by_back_index(3).intent_id == "i-1"
    assert sm.turn_by_back_index(4) is None
    assert sm.turn_by_back_index(0) is None


def test_to_prompt_block_empty(tmp_path: Path):
    sm = SessionMemory(path=tmp_path / "s.jsonl", capacity=4)
    assert sm.to_prompt_block() == ""


def test_to_prompt_block_renders(tmp_path: Path):
    sm = SessionMemory(path=tmp_path / "s.jsonl", capacity=4)
    sm.record(
        intent_id="i-1",
        natural_text="find pdfs in ~/Downloads",
        goal_spec=_gs(),
        results={"act-1": {"files": [{"path": "/a.pdf"}], "count": 1}},
        summary="Found 1 file.",
    )
    block = sm.to_prompt_block(n=3)
    assert "<SESSION_HISTORY>" in block
    assert "find pdfs in ~/Downloads" in block
    assert "act-1 file.QUERY" in block
    assert "1 files" in block
    assert "</SESSION_HISTORY>" in block
    assert "$prev" in block  # tutorial line


def test_natural_text_truncated(tmp_path: Path):
    sm = SessionMemory(path=tmp_path / "s.jsonl", capacity=2)
    long = "x" * 5000
    sm.record("i-1", long, _gs(), {"act-1": {}}, "")
    assert len(sm.latest().natural_text) <= 1000


def test_clear_removes_file(tmp_path: Path):
    p = tmp_path / "s.jsonl"
    sm = SessionMemory(path=p, capacity=2)
    sm.record("i-1", "x", _gs(), {"act-1": {}}, "")
    assert p.exists()
    sm.clear()
    assert not p.exists()
    assert len(sm) == 0
