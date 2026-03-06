"""
Leaves OS Agent Daemon — Phase 1 (library module, not yet a daemon process).

Phase 0: orchestration was inlined in leaves.py.
Phase 1: extracted here as AgentCoordinator — clean module with a minimal interface.
Phase 2: this becomes a standalone daemon process with a Unix socket.
Phase 3: cgroup integration and per-intent subprocess isolation.

Public interface (callers use ONLY this):
    coordinator = AgentCoordinator(db)
    results, summary = coordinator.execute(goal_spec, lifecycle)
"""
from __future__ import annotations

from typing import Any

from agents.file_agent import FileAgent
from agents.state_machine import IntentLifecycle, IntentState
from agents.tool_failure_tracker import ToolFailureTracker
from db.audit import log_error, log_state_transition
from errors import LeavesError, LeavesErrorCode

# Map agent type strings -> agent classes
# Phase 1: file only. Phase 2+: expand this map.
_AGENT_MAP: dict[str, type] = {
    "file": FileAgent,
}


class AgentCoordinator:
    """
    Coordinates agent execution for a single intent.

    Phase 1: sequential execution only.
    Phase 2: parallel execution via depends_on DAG.

    One instance per intent. Do not reuse across intents.
    """

    def __init__(self, db):
        self._db = db

    def execute(
        self,
        goal_spec: dict[str, Any],
        lifecycle: IntentLifecycle,
    ) -> tuple[dict[str, Any], str]:
        """
        Execute a validated GoalSpec.

        Returns (results_by_action_id, summary_string).
        Transitions lifecycle to EXECUTING internally.
        Raises LeavesError only on unrecoverable failure (tracker escalation).
        Per-action errors are captured in results and execution continues.
        """
        intent_id = goal_spec["intent_id"]
        actions = goal_spec.get("actions", [])
        tracker = ToolFailureTracker()

        from_state = lifecycle.state.value
        lifecycle.transition(IntentState.EXECUTING)
        log_state_transition(self._db, intent_id, from_state, "EXECUTING")

        results: dict[str, Any] = {}
        failed_ids: list[str] = []

        for action in actions:
            action_id = action.get("action_id", "unknown")
            agent_type = action.get("agent", "")

            # Agent availability check
            if agent_type not in _AGENT_MAP:
                err = LeavesError(
                    LeavesErrorCode.AGENT_NOT_AVAILABLE,
                    detail=(
                        f"Phase 1 supports: {list(_AGENT_MAP.keys())}. "
                        f"Got: '{agent_type}'"
                    ),
                )
                log_error(self._db, err.code.value, err.detail, intent_id)
                results[action_id] = {"error": err.user_message, "skipped": True}
                failed_ids.append(action_id)
                if action.get("on_failure", "abort") == "abort":
                    break
                continue

            # Dependency check
            unmet = [d for d in action.get("depends_on", []) if d in failed_ids]
            if unmet:
                err = LeavesError(
                    LeavesErrorCode.DEPENDENCY_FAILED,
                    detail=f"Action {action_id} skipped: dependency {unmet} failed",
                )
                log_error(self._db, err.code.value, err.detail, intent_id)
                results[action_id] = {"error": err.user_message, "skipped": True}
                failed_ids.append(action_id)
                continue

            # Execute
            agent = _AGENT_MAP[agent_type](intent_id=intent_id, db_conn=self._db)
            try:
                result = agent.execute_action(action)
                tracker.reset(action.get("type", ""), action.get("params", {}))
                results[action_id] = result
            except LeavesError as e:
                # Record failure — may escalate if same tool+args keeps failing
                try:
                    tracker.record_failure(
                        action.get("type", ""),
                        action.get("params", {}),
                        error=e,
                    )
                except LeavesError as escalated:
                    # Livelock detected — abort the entire intent
                    log_error(self._db, escalated.code.value, escalated.detail, intent_id)
                    results[action_id] = {"error": escalated.user_message}
                    failed_ids.append(action_id)
                    raise escalated  # bubble up to handle_intent

                log_error(self._db, e.code.value, e.detail, intent_id)
                results[action_id] = {"error": e.user_message}
                failed_ids.append(action_id)
                if action.get("on_failure", "abort") == "abort":
                    break

        succeeded = len(actions) - len(failed_ids)
        total_files = sum(
            r.get("count", 0) for r in results.values()
            if isinstance(r, dict) and "count" in r
        )

        if not failed_ids:
            summary = f"Completed {len(actions)} action(s)."
            if total_files:
                summary += f" Found {total_files} file(s)."
        else:
            summary = (
                f"{succeeded}/{len(actions)} actions succeeded, "
                f"{len(failed_ids)} failed."
            )

        return results, summary
