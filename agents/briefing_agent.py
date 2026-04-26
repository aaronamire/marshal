"""
Briefing agent — wraps BriefingGenerator behind the standard agent interface.

Layer 0 routes intents like "briefing", "what changed", "good morning" to
this agent (action_type=BRIEFING). The agent constructs a KnowledgeGraph
from the shared DB + LanceDB store and returns a structured briefing.
"""
from __future__ import annotations

import pathlib

from agents.base_agent import BaseAgent
from cortex.briefing import BriefingGenerator
from cortex.knowledge_graph import KnowledgeGraph
from errors import MarshalError, MarshalErrorCode

_LANCE_PATH = pathlib.Path(__file__).parent.parent / "rag" / ".lancedb"


class BriefingAgent(BaseAgent):
    AGENT_TYPE = "briefing"

    def execute_action(self, action: dict) -> dict:
        action_type = action.get("type", "").upper()
        action_id = action.get("action_id", "unknown")
        params = action.get("params", {}) or {}

        if action_type not in ("BRIEFING", "QUERY"):
            raise MarshalError(
                MarshalErrorCode.AGENT_NOT_AVAILABLE,
                detail=f"BriefingAgent handles BRIEFING/QUERY, got '{action_type}'",
            )

        hours = int(params.get("hours", 12))
        if hours < 1:
            hours = 1
        if hours > 720:
            hours = 720

        row_id = self._audit_start(action_id, action_type, params)
        try:
            kg = KnowledgeGraph(self._db, _LANCE_PATH)
            bg = BriefingGenerator(kg)
            briefing = bg.generate(hours=hours)
            self._audit_end(row_id, briefing)
            return briefing
        except MarshalError:
            raise
        except Exception as e:
            err = MarshalError(MarshalErrorCode.INTERNAL_ERROR, detail=str(e), cause=e)
            self._audit_end(row_id, error=err)
            raise err
