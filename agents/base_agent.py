"""
Abstract base class for all Leaves OS agents.
Phase 0: Only FileAgent is implemented.
Phase 1+: EmailAgent, WebAgent, SystemAgent.
"""
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    All agents must:
    - Accept an intent_id for audit logging
    - Implement execute_action(action: dict) -> dict
    - Log every operation via db/audit.py
    - Raise LeavesError (never bare Exception) on failure
    """

    AGENT_TYPE: str = "base"

    def __init__(self, intent_id: str, db_conn):
        self.intent_id = intent_id
        self._db = db_conn

    @abstractmethod
    def execute_action(self, action: dict) -> dict:
        """
        Execute a single GoalSpec action.
        Returns result dict. Raises LeavesError on failure.
        """
        ...
