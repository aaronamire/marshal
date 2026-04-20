"""
Intent lifecycle state machine with CAS-equivalent transition guards.
Threading lock is included now for Phase 2 concurrency readiness,
even though Phase 0 is single-threaded.
"""
from __future__ import annotations

import enum
import threading
import time
from typing import Optional

from errors import MarshalError, MarshalErrorCode


class IntentState(enum.Enum):
    PENDING = "PENDING"
    PARSING = "PARSING"
    AWAITING_AUTH = "AWAITING_AUTH"
    EXECUTING = "EXECUTING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Valid transitions: from_state -> set of allowed to_states
_VALID_TRANSITIONS: dict[IntentState, set[IntentState]] = {
    IntentState.PENDING:       {IntentState.PARSING, IntentState.CANCELLED},
    IntentState.PARSING:       {IntentState.AWAITING_AUTH, IntentState.EXECUTING, IntentState.FAILED, IntentState.CANCELLED},
    IntentState.AWAITING_AUTH: {IntentState.EXECUTING, IntentState.CANCELLED},
    IntentState.EXECUTING:     {IntentState.DONE, IntentState.FAILED, IntentState.CANCELLED},
    IntentState.DONE:          set(),   # terminal
    IntentState.FAILED:        set(),   # terminal
    IntentState.CANCELLED:     set(),   # terminal
}

_TERMINAL_STATES = {IntentState.DONE, IntentState.FAILED, IntentState.CANCELLED}


class IntentLifecycle:
    """
    CAS-equivalent state machine for a single intent's lifecycle.
    All state changes MUST go through transition() — never assign _state directly.
    """

    def __init__(self, intent_id: str):
        self.intent_id = intent_id
        self._state = IntentState.PENDING
        self._lock = threading.Lock()
        self._history: list[tuple[IntentState, IntentState, float]] = []
        # (from_state, to_state, timestamp_unix)

    @property
    def state(self) -> IntentState:
        with self._lock:
            return self._state

    def transition(self, to_state: IntentState) -> None:
        """
        Atomically transition to to_state.
        Raises MarshalError if the transition is not valid.
        """
        with self._lock:
            from_state = self._state

            if from_state in _TERMINAL_STATES:
                raise MarshalError(
                    MarshalErrorCode.INTENT_ALREADY_TERMINAL,
                    detail=f"Intent {self.intent_id!r} is already in terminal state {from_state.value}",
                )

            allowed = _VALID_TRANSITIONS.get(from_state, set())
            if to_state not in allowed:
                raise MarshalError(
                    MarshalErrorCode.INVALID_STATE_TRANSITION,
                    detail=(
                        f"Intent {self.intent_id!r}: "
                        f"cannot transition {from_state.value} -> {to_state.value}. "
                        f"Allowed: {[s.value for s in allowed]}"
                    ),
                )

            self._state = to_state
            self._history.append((from_state, to_state, time.time()))

    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    def history(self) -> list[tuple[IntentState, IntentState, float]]:
        """Return a copy of the transition history."""
        with self._lock:
            return list(self._history)

    def __repr__(self) -> str:
        return f"IntentLifecycle(id={self.intent_id!r}, state={self._state.value})"
