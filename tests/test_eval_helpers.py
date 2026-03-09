"""
Unit tests for eval_suite helper functions.
No inference server required.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.eval_suite import check_action_sequence


def make_spec(types: list[str]) -> dict:
    return {
        "actions": [
            {"type": t, "action_id": f"act-{i+1}"}
            for i, t in enumerate(types)
        ]
    }


class TestCheckActionSequence:

    def test_exact_match_passes(self):
        assert check_action_sequence(make_spec(["QUERY", "MOVE"]), ["QUERY", "MOVE"]) == ""

    def test_extra_query_before_move_passes(self):
        assert check_action_sequence(make_spec(["QUERY", "QUERY", "MOVE"]), ["QUERY", "MOVE"]) == ""

    def test_trailing_extra_passes(self):
        assert check_action_sequence(make_spec(["QUERY", "MOVE", "QUERY"]), ["QUERY", "MOVE"]) == ""

    def test_wrong_order_fails(self):
        err = check_action_sequence(make_spec(["MOVE", "QUERY"]), ["QUERY", "MOVE"])
        assert err != "", "MOVE before QUERY must fail"
        assert "MOVE" in err and "QUERY" in err

    def test_first_occurrence_semantics_regression(self):
        """
        Regression test for the subsequence bug.
        ["MOVE", "QUERY", "MOVE"] with expected ["QUERY", "MOVE"] must FAIL.
        The previous implementation found QUERY at pos 1 then the second MOVE at pos 2,
        falsely passing. First-occurrence uses first MOVE at pos 0 vs first QUERY at pos 1.
        """
        spec = make_spec(["MOVE", "QUERY", "MOVE"])
        err = check_action_sequence(spec, ["QUERY", "MOVE"])
        assert err != "", "Bug not fixed — ['MOVE','QUERY','MOVE'] should fail for ['QUERY','MOVE']"
        assert "MOVE" in err and "QUERY" in err

    def test_delete_exact_passes(self):
        assert check_action_sequence(make_spec(["QUERY", "DELETE"]), ["QUERY", "DELETE"]) == ""

    def test_delete_wrong_order_fails(self):
        err = check_action_sequence(make_spec(["DELETE", "QUERY"]), ["QUERY", "DELETE"])
        assert err != ""

    def test_missing_required_type_fails(self):
        err = check_action_sequence(make_spec(["QUERY"]), ["QUERY", "MOVE"])
        assert err != ""
        assert "MOVE" in err

    def test_empty_actions_fails(self):
        err = check_action_sequence({"actions": []}, ["QUERY", "MOVE"])
        assert err != ""

    def test_single_required_type_always_passes(self):
        # A single required type has no ordering constraint (nothing to compare against)
        assert check_action_sequence(make_spec(["QUERY"]), ["QUERY"]) == ""
        assert check_action_sequence(make_spec(["MOVE", "QUERY"]), ["QUERY"]) == ""
