"""Tests for the intent lifecycle state machine."""
import pytest
from agents.state_machine import IntentLifecycle, IntentState
from errors import MarshalError, MarshalErrorCode


def test_valid_path_to_done():
    lc = IntentLifecycle("test-001")
    lc.transition(IntentState.PARSING)
    lc.transition(IntentState.EXECUTING)
    lc.transition(IntentState.DONE)
    assert lc.state == IntentState.DONE


def test_valid_path_with_auth():
    lc = IntentLifecycle("test-002")
    lc.transition(IntentState.PARSING)
    lc.transition(IntentState.AWAITING_AUTH)
    lc.transition(IntentState.EXECUTING)
    lc.transition(IntentState.DONE)
    assert lc.state == IntentState.DONE


def test_invalid_transition_raises():
    lc = IntentLifecycle("test-003")
    with pytest.raises(MarshalError) as exc_info:
        lc.transition(IntentState.DONE)  # PENDING -> DONE is not allowed
    assert exc_info.value.code == MarshalErrorCode.INVALID_STATE_TRANSITION


def test_terminal_state_no_further_transitions():
    lc = IntentLifecycle("test-004")
    lc.transition(IntentState.PARSING)
    lc.transition(IntentState.EXECUTING)
    lc.transition(IntentState.DONE)
    with pytest.raises(MarshalError) as exc_info:
        lc.transition(IntentState.FAILED)
    assert exc_info.value.code == MarshalErrorCode.INTENT_ALREADY_TERMINAL


def test_cancelled_path():
    lc = IntentLifecycle("test-005")
    lc.transition(IntentState.PARSING)
    lc.transition(IntentState.CANCELLED)
    assert lc.state == IntentState.CANCELLED
    assert lc.is_terminal()


def test_history_records_all_transitions():
    lc = IntentLifecycle("test-006")
    lc.transition(IntentState.PARSING)
    lc.transition(IntentState.AWAITING_AUTH)
    lc.transition(IntentState.EXECUTING)
    lc.transition(IntentState.DONE)

    history = lc.history()
    assert len(history) == 4
    assert history[0][0] == IntentState.PENDING
    assert history[0][1] == IntentState.PARSING
    assert history[-1][1] == IntentState.DONE
