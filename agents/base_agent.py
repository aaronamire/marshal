"""
Abstract base class for all Marshal agents.

The current set of implemented agents is declared in agents/registry.py
(file, system, web, audio, network, power, writing). Do not maintain a
duplicate list here — it will drift.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from db.audit import log_action_started, log_action_completed
from errors import MarshalError


class BaseAgent(ABC):
    """
    All agents must:
    - Accept an intent_id for audit logging
    - Implement execute_action(action: dict) -> dict
    - Log every operation via db/audit.py
    - Raise MarshalError (never bare Exception) on failure
    """

    AGENT_TYPE: str = "base"

    def __init__(self, intent_id: str, db_conn):
        self.intent_id = intent_id
        self._db = db_conn

    @abstractmethod
    def execute_action(self, action: dict) -> dict:
        """
        Execute a single GoalSpec action.
        Returns result dict. Raises MarshalError on failure.
        """
        ...

    # ------------------------------------------------------------------
    # Audit helpers (shared by all agents)
    # ------------------------------------------------------------------

    def _audit_start(self, action_id: str, action_type: str, params: dict) -> Optional[int]:
        try:
            return log_action_started(
                self._db,
                intent_id=self.intent_id,
                action_id=action_id,
                action_type=action_type,
                agent=self.AGENT_TYPE,
                params=params,
            )
        except Exception:
            return None  # DB failure is non-fatal for the operation itself

    def _audit_end(
        self,
        row_id: Optional[int],
        result: Optional[dict] = None,
        error: Optional[MarshalError] = None,
    ) -> None:
        if row_id is None:
            return
        try:
            log_action_completed(
                self._db,
                row_id=row_id,
                result=result,
                error_code=error.code.value if error else None,
                error_detail=error.detail if error else None,
            )
        except Exception:
            pass  # DB failure during audit logging is non-fatal
