"""Tests for the tool failure tracker / livelock prevention."""
import pytest
from agents.tool_failure_tracker import ToolFailureTracker
from errors import MarshalError, MarshalErrorCode


def test_escalates_at_threshold():
    tracker = ToolFailureTracker(threshold=3)
    args = {"path": "/home/user/file.txt"}
    tracker.record_failure("fs_delete", args)
    tracker.record_failure("fs_delete", args)
    with pytest.raises(MarshalError) as exc_info:
        tracker.record_failure("fs_delete", args)
    assert exc_info.value.code == MarshalErrorCode.TOOL_FAILURE_ESCALATED


def test_different_args_tracked_separately():
    tracker = ToolFailureTracker(threshold=3)
    args_a = {"path": "/home/user/a.txt"}
    args_b = {"path": "/home/user/b.txt"}
    tracker.record_failure("fs_delete", args_a)
    tracker.record_failure("fs_delete", args_a)
    # args_b has only 1 failure — should not escalate
    tracker.record_failure("fs_delete", args_b)
    assert tracker.failure_count("fs_delete", args_a) == 2
    assert tracker.failure_count("fs_delete", args_b) == 1


def test_failure_count_tracks_correctly():
    tracker = ToolFailureTracker(threshold=5)
    args = {"path": "~/Downloads"}
    tracker.record_failure("fs_list", args)
    tracker.record_failure("fs_list", args)
    assert tracker.failure_count("fs_list", args) == 2
    assert tracker.total_failures() == 2


def test_last_error_accessible():
    tracker = ToolFailureTracker(threshold=5)
    args = {"path": "~/test.txt"}
    err = RuntimeError("disk full")
    tracker.record_failure("fs_write", args, error=err)
    assert tracker.last_error("fs_write", args) is err
    assert tracker.last_error("fs_write", {"path": "other"}) is None
