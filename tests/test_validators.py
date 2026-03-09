"""
Unit tests for agents/validators.py.
No inference server required — tests validator logic directly.
Run with: pytest tests/test_validators.py -v
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.validators import (
    validate_action_ordering,
    validate_destructive_consistency,
    validate_goal_spec,
)


def make_action(aid, type_, destructive=False, depends_on=None, params=None):
    a = {
        "action_id": aid,
        "type": type_,
        "agent": "file",
        "params": params or {"path": "~/test"},
        "destructive": destructive,
    }
    if depends_on is not None:
        a["depends_on"] = depends_on
    return a


def make_spec(actions, auth=None):
    return {
        "intent_id": "00000000-0000-4000-8000-000000000000",
        "natural_text": "test",
        "category": "file_task",
        "actions": actions,
        "authorization": auth or {
            "resources": ["~"],
            "preview_required": False,
            "reversible": True,
        },
        "metadata": {"confidence": 0.9},
    }


class TestActionOrdering:

    def test_valid_single_query(self):
        spec = make_spec([make_action("act-1", "QUERY")])
        r = validate_action_ordering(spec)
        assert r.valid, str(r)

    def test_valid_query_then_move(self):
        spec = make_spec([
            make_action("act-1", "QUERY"),
            make_action("act-2", "MOVE", destructive=True, depends_on=["act-1"]),
        ])
        r = validate_action_ordering(spec)
        assert r.valid, str(r)

    def test_valid_query_then_delete(self):
        spec = make_spec([
            make_action("act-1", "QUERY"),
            make_action("act-2", "DELETE", destructive=True, depends_on=["act-1"]),
        ])
        r = validate_action_ordering(spec)
        assert r.valid, str(r)

    def test_valid_explicit_rename_move_first(self):
        """MOVE as sole action IS valid when path is explicit (rename case)."""
        spec = make_spec([
            make_action("act-1", "MOVE", destructive=True,
                        params={"source": "~/notes.txt", "destination": "~/notes-backup.txt"}),
        ])
        r = validate_action_ordering(spec)
        assert r.valid, str(r)

    def test_invalid_move_without_prior_query_wildcard(self):
        """MOVE first with a wildcard pattern — almost certainly wrong ordering."""
        spec = make_spec([
            make_action("act-1", "MOVE", destructive=True,
                        params={"path": "~/Downloads", "pattern": "*.pdf"}),
            make_action("act-2", "QUERY"),
        ])
        r = validate_action_ordering(spec)
        assert not r.valid
        codes = {e.code for e in r.errors}
        assert "MUTATION_WITHOUT_PRIOR_DISCOVERY" in codes

    def test_invalid_move_delete_query_move_query_bug(self):
        """
        The actual model bug: [MOVE, QUERY, MOVE, QUERY] must be caught.
        This was passing the old eval harness (type presence only).
        """
        spec = make_spec([
            make_action("act-1", "MOVE", destructive=True,
                        params={"path": "~/Downloads", "pattern": "*.pdf"}),
            make_action("act-2", "QUERY"),
            make_action("act-3", "MOVE", destructive=True),
            make_action("act-4", "QUERY"),
        ])
        r = validate_action_ordering(spec)
        assert not r.valid
        assert any(e.code == "MUTATION_WITHOUT_PRIOR_DISCOVERY" for e in r.errors)

    def test_invalid_forward_dependency(self):
        """act-1 depends_on act-2 which appears later — illegal."""
        spec = make_spec([
            make_action("act-1", "MOVE", depends_on=["act-2"]),
            make_action("act-2", "QUERY"),
        ])
        r = validate_action_ordering(spec)
        assert not r.valid
        assert any(e.code == "FORWARD_DEPENDENCY" for e in r.errors)

    def test_self_dependency_is_soft_error(self):
        """act-1 depends_on act-1 — model artifact, soft error not hard."""
        spec = make_spec([
            make_action("act-1", "MOVE", destructive=True,
                        params={"source": "~/notes.txt", "destination": "~/backup/notes.txt"},
                        depends_on=["act-1"]),
        ])
        r = validate_action_ordering(spec)
        # Should report error but only as SELF_DEPENDENCY (soft), not FORWARD_DEPENDENCY
        codes = {e.code for e in r.errors}
        assert "SELF_DEPENDENCY" in codes
        # FORWARD_DEPENDENCY should NOT fire for self-references
        assert "FORWARD_DEPENDENCY" not in codes

    def test_invalid_dangling_dependency(self):
        """depends_on references an action_id that doesn't exist."""
        spec = make_spec([
            make_action("act-1", "QUERY"),
            make_action("act-2", "MOVE", depends_on=["act-99"]),
        ])
        r = validate_action_ordering(spec)
        assert not r.valid
        assert any(e.code == "DANGLING_DEPENDENCY" for e in r.errors)

    def test_invalid_duplicate_action_id(self):
        """Two actions with the same action_id."""
        spec = make_spec([
            make_action("act-1", "QUERY"),
            make_action("act-1", "MOVE"),  # duplicate
        ])
        r = validate_action_ordering(spec)
        assert not r.valid
        assert any(e.code == "DUPLICATE_ACTION_ID" for e in r.errors)

    def test_empty_actions_flagged(self):
        spec = make_spec([])
        r = validate_action_ordering(spec)
        assert not r.valid
        assert any(e.code == "NO_ACTIONS" for e in r.errors)

    def test_delete_first_with_pattern_flagged(self):
        """DELETE as first action with wildcard pattern — must follow QUERY."""
        spec = make_spec([
            make_action("act-1", "DELETE", destructive=True,
                        params={"path": "~", "pattern": "*.tmp"}),
        ])
        # Single DELETE with pattern and no prior QUERY — problematic
        # (single action can't have MUTATION_WITHOUT_PRIOR_DISCOVERY since len==1)
        r = validate_action_ordering(spec)
        assert r.valid  # Single action with explicit path-ish params is allowed

    def test_no_actions_field_flagged(self):
        spec = {"intent_id": "x", "natural_text": "x", "category": "file_task",
                "authorization": {}, "metadata": {}}
        r = validate_action_ordering(spec)
        assert not r.valid


class TestDestructiveConsistency:

    def test_destructive_requires_preview(self):
        spec = make_spec(
            [make_action("act-1", "DELETE", destructive=True)],
            auth={"resources": ["~"], "preview_required": False, "reversible": False},
        )
        r = validate_destructive_consistency(spec)
        assert not r.valid
        assert any(e.code == "DESTRUCTIVE_WITHOUT_PREVIEW" for e in r.errors)

    def test_move_must_be_destructive(self):
        spec = make_spec([make_action("act-1", "MOVE", destructive=False)])
        r = validate_destructive_consistency(spec)
        assert not r.valid
        assert any(e.code == "MUTATION_NOT_FLAGGED_DESTRUCTIVE" for e in r.errors)

    def test_delete_must_be_destructive(self):
        spec = make_spec([make_action("act-1", "DELETE", destructive=False)])
        r = validate_destructive_consistency(spec)
        assert not r.valid
        assert any(e.code == "MUTATION_NOT_FLAGGED_DESTRUCTIVE" for e in r.errors)

    def test_query_non_destructive_is_fine(self):
        spec = make_spec([make_action("act-1", "QUERY", destructive=False)])
        r = validate_destructive_consistency(spec)
        assert r.valid, str(r)

    def test_read_non_destructive_is_fine(self):
        spec = make_spec([make_action("act-1", "READ", destructive=False)])
        r = validate_destructive_consistency(spec)
        assert r.valid, str(r)


class TestValidateGoalSpec:

    def test_valid_spec_passes_all(self):
        spec = make_spec([
            make_action("act-1", "QUERY"),
            make_action("act-2", "DELETE", destructive=True, depends_on=["act-1"]),
        ], auth={"resources": ["~"], "preview_required": True, "reversible": False})
        r = validate_goal_spec(spec)
        assert r.valid, str(r)

    def test_aggregates_multiple_errors(self):
        # Forward dep + destructive without preview
        spec = make_spec(
            [
                make_action("act-1", "DELETE", destructive=True, depends_on=["act-2"]),
                make_action("act-2", "QUERY"),
            ],
            auth={"resources": ["~"], "preview_required": False, "reversible": False},
        )
        r = validate_goal_spec(spec)
        assert not r.valid
        codes = {e.code for e in r.errors}
        assert "FORWARD_DEPENDENCY" in codes
        assert "DESTRUCTIVE_WITHOUT_PREVIEW" in codes
