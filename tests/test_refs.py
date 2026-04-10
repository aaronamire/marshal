"""Tests for the $prev[N].action_id.path reference resolver."""
from __future__ import annotations

from pathlib import Path

from agents.refs import resolve_refs
from agents.session_memory import SessionMemory


def _make_session(tmp_path: Path) -> SessionMemory:
    sm = SessionMemory(path=tmp_path / "s.jsonl", capacity=4)
    sm.record(
        intent_id="i-1",
        natural_text="find pdfs in ~/Downloads",
        goal_spec={
            "intent_id": "i-1",
            "actions": [{"action_id": "act-1", "type": "QUERY",
                         "agent": "file", "params": {"path": "~/Downloads"}}],
        },
        results={
            "act-1": {
                "files": [
                    {"path": "/home/u/a.pdf", "size": 100},
                    {"path": "/home/u/b.pdf", "size": 200},
                ],
                "count": 2,
            },
        },
        summary="Found 2 files.",
    )
    return sm


def test_whole_string_resolves_to_typed_value(tmp_path: Path):
    sm = _make_session(tmp_path)
    out = resolve_refs("$prev.act-1.files[0].path", sm)
    assert out == "/home/u/a.pdf"
    assert isinstance(out, str)


def test_whole_string_preserves_int_type(tmp_path: Path):
    sm = _make_session(tmp_path)
    out = resolve_refs("$prev.act-1.files[1].size", sm)
    assert out == 200
    assert isinstance(out, int)


def test_whole_string_returns_dict(tmp_path: Path):
    sm = _make_session(tmp_path)
    out = resolve_refs("$prev.act-1.files[0]", sm)
    assert isinstance(out, dict)
    assert out["path"] == "/home/u/a.pdf"


def test_walks_nested_dict(tmp_path: Path):
    sm = _make_session(tmp_path)
    params = {
        "source": "$prev.act-1.files[0].path",
        "destination": "/tmp/x.pdf",
        "options": {"recursive": False},
    }
    out = resolve_refs(params, sm)
    assert out["source"] == "/home/u/a.pdf"
    assert out["destination"] == "/tmp/x.pdf"
    assert out["options"]["recursive"] is False


def test_walks_list(tmp_path: Path):
    sm = _make_session(tmp_path)
    out = resolve_refs(
        ["$prev.act-1.files[0].path", "$prev.act-1.files[1].path"],
        sm,
    )
    assert out == ["/home/u/a.pdf", "/home/u/b.pdf"]


def test_unresolvable_left_unchanged(tmp_path: Path):
    sm = _make_session(tmp_path)
    s = "$prev.act-99.foo"
    assert resolve_refs(s, sm) == s


def test_out_of_range_back_index(tmp_path: Path):
    sm = _make_session(tmp_path)
    s = "$prev5.act-1.files[0].path"
    assert resolve_refs(s, sm) == s


def test_summary_special_lookup(tmp_path: Path):
    sm = _make_session(tmp_path)
    out = resolve_refs("$prev.summary", sm)
    assert out == "Found 2 files."


def test_inline_substitution(tmp_path: Path):
    sm = _make_session(tmp_path)
    out = resolve_refs("got $prev.act-1.count results", sm)
    assert out == "got 2 results"


def test_no_session_returns_input(tmp_path: Path):
    out = resolve_refs({"path": "$prev.act-1.foo"}, session=None)
    assert out == {"path": "$prev.act-1.foo"}


def test_index_out_of_range(tmp_path: Path):
    sm = _make_session(tmp_path)
    s = "$prev.act-1.files[99].path"
    assert resolve_refs(s, sm) == s


def test_non_string_values_passthrough(tmp_path: Path):
    sm = _make_session(tmp_path)
    assert resolve_refs(42, sm) == 42
    assert resolve_refs(True, sm) is True
    assert resolve_refs(None, sm) is None


def test_prev2_back_index(tmp_path: Path):
    sm = SessionMemory(path=tmp_path / "s.jsonl", capacity=4)
    sm.record("i-1", "t1", {"intent_id": "i-1", "actions": []},
              {"act-1": {"v": "older"}}, "")
    sm.record("i-2", "t2", {"intent_id": "i-2", "actions": []},
              {"act-1": {"v": "newer"}}, "")
    assert resolve_refs("$prev.act-1.v", sm) == "newer"
    assert resolve_refs("$prev2.act-1.v", sm) == "older"
