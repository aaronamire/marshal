"""
Tool failure tracker — prevents livelock when a tool repeatedly fails
with the same arguments (PhD review Section 7).

If the same (tool_name, args_key) pair fails TOOL_FAILURE_ESCALATION_THRESHOLD
times, raise MarshalError(TOOL_FAILURE_ESCALATED) to abort the intent.
"""
from __future__ import annotations

from typing import Any, Optional

from config import TOOL_FAILURE_ESCALATION_THRESHOLD
from errors import MarshalError, MarshalErrorCode


def _make_args_key(args: dict[str, Any]) -> str:
    """Stable string key for a dict of arguments."""
    return repr(sorted(args.items()))


class ToolFailureTracker:
    """
    Tracks per-(tool, args) failure counts for a single intent execution.
    Escalates to TOOL_FAILURE_ESCALATED when threshold is reached.
    """

    def __init__(self, threshold: int = TOOL_FAILURE_ESCALATION_THRESHOLD):
        self._threshold = threshold
        # (tool_name, args_key) -> (count, last_error)
        self._failures: dict[tuple[str, str], tuple[int, Optional[Exception]]] = {}

    def record_failure(
        self,
        tool_name: str,
        args: dict[str, Any],
        error: Optional[Exception] = None,
    ) -> None:
        """
        Record a failure for tool_name with the given args.
        Raises MarshalError(TOOL_FAILURE_ESCALATED) if threshold is exceeded.
        """
        key = (tool_name, _make_args_key(args))
        count, _ = self._failures.get(key, (0, None))
        count += 1
        self._failures[key] = (count, error)

        if count >= self._threshold:
            raise MarshalError(
                MarshalErrorCode.TOOL_FAILURE_ESCALATED,
                detail=(
                    f"Tool '{tool_name}' failed {count} times with the same arguments. "
                    f"Last error: {error!r}"
                ),
                cause=error,
            )

    def failure_count(self, tool_name: str, args: dict[str, Any]) -> int:
        """Return how many times this tool+args combination has failed."""
        key = (tool_name, _make_args_key(args))
        count, _ = self._failures.get(key, (0, None))
        return count

    def last_error(
        self, tool_name: str, args: dict[str, Any]
    ) -> Optional[Exception]:
        """Return the last recorded error for this tool+args, or None."""
        key = (tool_name, _make_args_key(args))
        _, err = self._failures.get(key, (0, None))
        return err

    def reset(self, tool_name: str, args: dict[str, Any]) -> None:
        """Reset failure count for tool+args (call on success)."""
        key = (tool_name, _make_args_key(args))
        self._failures.pop(key, None)

    def total_failures(self) -> int:
        return sum(c for c, _ in self._failures.values())
