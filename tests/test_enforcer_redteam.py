"""
Red-team adversarial test suite for agents.enforcer.

Each case constructs a (goal_spec, action) pair that an attacker /
buggy agent might produce, and asserts the enforcer rejects it with
LeavesError(AUTHORIZATION_VIOLATION).

Goal: 20/20 blocked. Run with::

    python3 -m pytest tests/test_enforcer_redteam.py -v

The summary at the bottom prints "N/N blocked" so the result is
quotable in docs and the HN post. A single positive control verifies
the enforcer doesn't just reject everything.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.enforcer import enforce  # noqa: E402
from errors import LeavesError, LeavesErrorCode  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: filesystem layout for path-based tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def safe_tree():
    """A real on-disk directory tree we can authorize and attack."""
    with tempfile.TemporaryDirectory(prefix="leaves-redteam-") as td:
        root = Path(td).resolve()
        safe_dir = root / "safe-dir"
        other_dir = root / "other-dir"
        safe_dir.mkdir()
        other_dir.mkdir()
        (safe_dir / "ok.txt").write_text("ok")
        (safe_dir / "child" / "deep.txt").parent.mkdir()
        (safe_dir / "child" / "deep.txt").write_text("deep")
        (root / "x").mkdir()
        (root / "x" / "safe.txt").write_text("safe")
        (root / "x" / "secrets.txt").write_text("secret")
        (other_dir / "evil.txt").write_text("evil")
        yield {
            "root": root,
            "safe_dir": safe_dir,
            "other_dir": other_dir,
            "safe_file": root / "x" / "safe.txt",
            "sibling_file": root / "x" / "secrets.txt",
            "deep_file": safe_dir / "child" / "deep.txt",
            "evil_file": other_dir / "evil.txt",
        }


def _spec(actions, resources):
    """Minimal valid GoalSpec shell for testing."""
    return {
        "intent_id": "00000000-0000-4000-8000-000000000000",
        "natural_text": "test",
        "category": "file_task",
        "actions": actions,
        "authorization": {
            "resources": resources,
            "preview_required": False,
            "reversible": True,
        },
    }


# ---------------------------------------------------------------------------
# 20 adversarial cases.
# Each builder returns (case_name, goal_spec, action_to_enforce).
# ---------------------------------------------------------------------------


def _cases(tree):
    safe_dir = str(tree["safe_dir"])
    safe_file = str(tree["safe_file"])
    sibling_file = str(tree["sibling_file"])
    evil_file = str(tree["evil_file"])
    safe_dir_dotdot_escape = str(tree["safe_dir"]) + "/../other-dir/evil.txt"

    plan_read_safe_dir = [{
        "action_id": "act-1", "type": "READ", "agent": "file",
        "params": {"path": safe_dir}, "destructive": False,
    }]
    plan_query_only = [{
        "action_id": "act-1", "type": "QUERY", "agent": "file",
        "params": {"path": safe_dir, "pattern": "*.txt"}, "destructive": False,
    }]
    plan_copy_safe = [{
        "action_id": "act-1", "type": "COPY", "agent": "file",
        "params": {"source": safe_file, "destination": safe_file + ".bak"},
        "destructive": True,
    }]
    plan_move_safe = [{
        "action_id": "act-1", "type": "MOVE", "agent": "file",
        "params": {"source": safe_file, "destination": safe_file + ".moved"},
        "destructive": True,
    }]

    cases = []

    # 1. action_id not present in plan
    cases.append((
        "smuggled_action_id",
        _spec(plan_read_safe_dir, [safe_dir]),
        {"action_id": "act-99", "type": "READ", "agent": "file",
         "params": {"path": safe_dir}},
    ))

    # 2. case-mismatched action_id (strings differ → not in plan)
    cases.append((
        "case_mismatch_action_id",
        _spec(plan_read_safe_dir, [safe_dir]),
        {"action_id": "ACT-1", "type": "READ", "agent": "file",
         "params": {"path": safe_dir}},
    ))

    # 3. plan has zero actions; any executed action smuggles in
    cases.append((
        "exec_against_empty_plan",
        _spec([], [safe_dir]),
        {"action_id": "act-1", "type": "READ", "agent": "file",
         "params": {"path": safe_dir}},
    ))

    # 4. type mismatch: plan READ, exec DELETE
    cases.append((
        "type_read_to_delete",
        _spec(plan_read_safe_dir, [safe_dir]),
        {"action_id": "act-1", "type": "DELETE", "agent": "file",
         "params": {"path": safe_file}},
    ))

    # 5. type mismatch: plan QUERY, exec WRITE
    cases.append((
        "type_query_to_write",
        _spec(plan_query_only, [safe_dir]),
        {"action_id": "act-1", "type": "WRITE", "agent": "file",
         "params": {"path": safe_file, "content": "x"}},
    ))

    # 6. type mismatch among destructive types: plan COPY, exec MOVE
    cases.append((
        "type_copy_to_move",
        _spec(plan_copy_safe, [safe_file]),
        {"action_id": "act-1", "type": "MOVE", "agent": "file",
         "params": {"source": safe_file, "destination": safe_file + ".bak"}},
    ))

    # 7. empty type string vs planned READ
    cases.append((
        "type_empty_string",
        _spec(plan_read_safe_dir, [safe_dir]),
        {"action_id": "act-1", "type": "", "agent": "file",
         "params": {"path": safe_dir}},
    ))

    # 8. ../ traversal escape
    cases.append((
        "path_dotdot_traversal",
        _spec(plan_read_safe_dir, [safe_dir]),
        {"action_id": "act-1", "type": "READ", "agent": "file",
         "params": {"path": safe_dir_dotdot_escape}},
    ))

    # 9. absolute path outside authorized dir
    cases.append((
        "path_absolute_outside",
        _spec(plan_read_safe_dir, [safe_dir]),
        {"action_id": "act-1", "type": "READ", "agent": "file",
         "params": {"path": "/etc/passwd"}},
    ))

    # 10. tilde-expansion to a different user's home
    cases.append((
        "path_tilde_other_user",
        _spec(plan_read_safe_dir, [safe_dir]),
        {"action_id": "act-1", "type": "READ", "agent": "file",
         "params": {"path": "~root/.ssh/id_rsa"}},
    ))

    # 11. sibling directory escape (auth /tmp/x/safe-dir, path /tmp/x/other-dir)
    cases.append((
        "path_sibling_dir_escape",
        _spec(plan_read_safe_dir, [safe_dir]),
        {"action_id": "act-1", "type": "READ", "agent": "file",
         "params": {"path": evil_file}},
    ))

    # 12. THE FIXED BUG: auth a single FILE, attacker reads sibling file
    plan_read_safe_file = [{
        "action_id": "act-1", "type": "READ", "agent": "file",
        "params": {"path": safe_file}, "destructive": False,
    }]
    cases.append((
        "path_sibling_file_via_parent_match",
        _spec(plan_read_safe_file, [safe_file]),
        {"action_id": "act-1", "type": "READ", "agent": "file",
         "params": {"path": sibling_file}},
    ))

    # 13. destructive type swapped into a non-destructive plan
    cases.append((
        "delete_in_query_plan",
        _spec(plan_query_only, [safe_dir]),
        {"action_id": "act-1", "type": "DELETE", "agent": "file",
         "params": {"path": safe_file}},
    ))

    # 14. MOVE: source authorized, destination escapes
    cases.append((
        "move_destination_unauthorized",
        _spec(plan_move_safe, [safe_file]),
        {"action_id": "act-1", "type": "MOVE", "agent": "file",
         "params": {"source": safe_file, "destination": "/etc/passwd"}},
    ))

    # 15. COPY: source authorized, destination outside
    cases.append((
        "copy_destination_outside",
        _spec(plan_copy_safe, [safe_file]),
        {"action_id": "act-1", "type": "COPY", "agent": "file",
         "params": {"source": safe_file, "destination": evil_file}},
    ))

    # 16. WRITE to a root-owned path with empty resource list
    plan_write_safe = [{
        "action_id": "act-1", "type": "WRITE", "agent": "file",
        "params": {"path": safe_file}, "destructive": True,
    }]
    cases.append((
        "write_to_root",
        _spec(plan_write_safe, [safe_file]),
        {"action_id": "act-1", "type": "WRITE", "agent": "file",
         "params": {"path": "/etc/shadow", "content": "x"}},
    ))

    # 17. Action has paths but resources = [] — empty must mean "deny",
    # not "allow all". This was a real bypass in the prior enforcer.
    cases.append((
        "no_resources_with_paths",
        _spec(plan_read_safe_dir, []),
        {"action_id": "act-1", "type": "READ", "agent": "file",
         "params": {"path": safe_file}},
    ))

    # 18. authorization key entirely missing from goal_spec
    spec_no_auth = {
        "intent_id": "00000000-0000-4000-8000-000000000000",
        "natural_text": "test",
        "category": "file_task",
        "actions": plan_read_safe_dir,
    }
    cases.append((
        "auth_block_missing",
        spec_no_auth,
        {"action_id": "act-1", "type": "READ", "agent": "file",
         "params": {"path": safe_file}},
    ))

    # 19. params is not a dict — schema-shape attack from a bad agent
    cases.append((
        "params_non_dict",
        _spec(plan_read_safe_dir, [safe_dir]),
        {"action_id": "act-1", "type": "READ", "agent": "file",
         "params": [safe_file]},
    ))

    # 20. Path smuggled via an exotic param key not in the whitelist —
    # caught only by the comprehensive value-scan pass.
    cases.append((
        "path_via_exotic_param_key",
        _spec(plan_read_safe_dir, [safe_dir]),
        {"action_id": "act-1", "type": "READ", "agent": "file",
         "params": {"path": safe_dir, "log_path": "/etc/shadow"}},
    ))

    return cases


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------


@pytest.fixture
def cases(safe_tree):
    return _cases(safe_tree)


@pytest.mark.parametrize("idx", range(20))
def test_enforcer_blocks_adversarial(cases, idx):
    name, goal_spec, action = cases[idx]
    with pytest.raises(LeavesError) as exc_info:
        enforce(action, goal_spec)
    assert exc_info.value.code == LeavesErrorCode.AUTHORIZATION_VIOLATION, (
        f"[{name}] expected AUTHORIZATION_VIOLATION, got {exc_info.value.code}"
    )


# Positive control: enforcer must NOT reject a legitimate action.
def test_positive_control_legitimate_action_passes(safe_tree):
    safe_dir = str(safe_tree["safe_dir"])
    deep = str(safe_tree["deep_file"])
    plan = [{
        "action_id": "act-1", "type": "READ", "agent": "file",
        "params": {"path": safe_dir}, "destructive": False,
    }]
    spec = _spec(plan, [safe_dir])
    action = {"action_id": "act-1", "type": "READ", "agent": "file",
              "params": {"path": deep}}
    enforce(action, spec)  # must not raise


# ---------------------------------------------------------------------------
# Standalone runner: prints "N/N blocked" so it's quotable.
# ---------------------------------------------------------------------------


def _run_summary():
    """For ad-hoc runs: python3 tests/test_enforcer_redteam.py"""
    with tempfile.TemporaryDirectory(prefix="leaves-redteam-") as td:
        root = Path(td).resolve()
        (root / "safe-dir").mkdir()
        (root / "safe-dir" / "ok.txt").write_text("ok")
        (root / "safe-dir" / "child").mkdir()
        (root / "safe-dir" / "child" / "deep.txt").write_text("d")
        (root / "other-dir").mkdir()
        (root / "other-dir" / "evil.txt").write_text("e")
        (root / "x").mkdir()
        (root / "x" / "safe.txt").write_text("s")
        (root / "x" / "secrets.txt").write_text("S")
        tree = {
            "root": root,
            "safe_dir": root / "safe-dir",
            "other_dir": root / "other-dir",
            "safe_file": root / "x" / "safe.txt",
            "sibling_file": root / "x" / "secrets.txt",
            "deep_file": root / "safe-dir" / "child" / "deep.txt",
            "evil_file": root / "other-dir" / "evil.txt",
        }
        cases = _cases(tree)
        blocked = 0
        failures = []
        for name, gs, act in cases:
            try:
                enforce(act, gs)
            except LeavesError as e:
                if e.code == LeavesErrorCode.AUTHORIZATION_VIOLATION:
                    blocked += 1
                else:
                    failures.append((name, f"wrong code: {e.code.value}"))
            else:
                failures.append((name, "NOT blocked (enforcer accepted)"))

        # positive control
        deep = str(tree["deep_file"])
        plan = [{
            "action_id": "act-1", "type": "READ", "agent": "file",
            "params": {"path": str(tree["safe_dir"])}, "destructive": False,
        }]
        gs = _spec(plan, [str(tree["safe_dir"])])
        act = {"action_id": "act-1", "type": "READ", "agent": "file",
               "params": {"path": deep}}
        try:
            enforce(act, gs)
            pos_ok = True
        except LeavesError as e:
            pos_ok = False
            failures.append(("positive_control", f"falsely blocked: {e.detail}"))

        print()
        print("=" * 60)
        print(f"  Red-team enforcer suite: {blocked}/{len(cases)} blocked")
        print(f"  Positive control: {'PASS' if pos_ok else 'FAIL'}")
        print("=" * 60)
        for n, reason in failures:
            print(f"  [FAIL] {n}: {reason}")
        return blocked == len(cases) and pos_ok


if __name__ == "__main__":
    sys.exit(0 if _run_summary() else 1)
