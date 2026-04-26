"""
Central agent registry — the single source of truth for which agents exist,
which categories route to which agents, and what counts as "implemented."

Before this module, three parallel lists drifted apart:

  - agentd.py  _AGENT_MAP              (type name -> class)
  - intent_parser.py  IMPLEMENTED_CATEGORIES  (L1/L2 fast-path check)
  - intent_parser.py  IMPLEMENTED_AGENTS      (post-parse validation)

Every new agent required editing three places, and the user-facing
NOT_IMPLEMENTED error strings drifted to lies ("only file/system/web
supported" long after audio/network/power/writing shipped).

Design: declare each agent once as an AgentEntry. Everything else
(class map, implemented sets, category↔agent mapping, error message
generation) is derived. Adding a new agent is one entry; nothing else
in the project needs to change except the manifest-driven intent_parser
few-shot examples.

Layer 0's `_NOT_IMPL_RULES` is intentionally NOT derived from this
registry — it's a pattern-based fast-path that may flag inputs nobody
has a regex for, even when the matching agent exists. The registry
only decides what the *full* pipeline considers implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


@dataclass(frozen=True)
class AgentEntry:
    """One row in the agent registry."""
    agent: str                       # short name used in GoalSpec.actions[].agent
    category: str                    # GoalSpec.category for this agent's intents
    import_path: str                 # "agents.file_agent"
    class_name: str                  # "FileAgent"
    description: str                 # one-line human summary for error messages


# ---------------------------------------------------------------------------
# Declarations — ADD NEW AGENTS HERE ONLY
# ---------------------------------------------------------------------------

_ENTRIES: tuple[AgentEntry, ...] = (
    AgentEntry(
        agent="file",
        category="file_task",
        import_path="agents.file_agent",
        class_name="FileAgent",
        description="file operations (list, read, write, move, copy, delete)",
    ),
    AgentEntry(
        agent="system",
        category="system_task",
        import_path="agents.system_agent",
        class_name="SystemAgent",
        description="system queries and app launch/terminate",
    ),
    AgentEntry(
        agent="web",
        category="web_task",
        import_path="agents.web_agent",
        class_name="WebAgent",
        description="web search and fetch",
    ),
    AgentEntry(
        agent="audio",
        category="audio_task",
        import_path="agents.audio_agent",
        class_name="AudioAgent",
        description="audio playback and volume control",
    ),
    AgentEntry(
        agent="network",
        category="network_task",
        import_path="agents.network_agent",
        class_name="NetworkAgent",
        description="network status and WiFi",
    ),
    AgentEntry(
        agent="power",
        category="power_task",
        import_path="agents.power_agent",
        class_name="PowerAgent",
        description="power and battery",
    ),
    AgentEntry(
        agent="writing",
        category="writing_task",
        import_path="agents.writing_agent",
        class_name="WritingAgent",
        description="text composition (documents, reports, summaries)",
    ),
    AgentEntry(
        agent="briefing",
        category="briefing",
        import_path="agents.briefing_agent",
        class_name="BriefingAgent",
        description="recent-changes briefing across indexed sources",
    ),
)


# ---------------------------------------------------------------------------
# Lazy import — avoids circular imports between agentd and the agent modules
# ---------------------------------------------------------------------------

def _load_class(entry: AgentEntry) -> "type[BaseAgent]":
    module = __import__(entry.import_path, fromlist=[entry.class_name])
    return getattr(module, entry.class_name)


def agent_classes() -> "dict[str, type[BaseAgent]]":
    """
    Return the live map of agent name -> class. Evaluated lazily so
    importing this module does not trigger loading every agent module.
    """
    return {e.agent: _load_class(e) for e in _ENTRIES}


# ---------------------------------------------------------------------------
# Derived views — all the things callers used to hardcode
# ---------------------------------------------------------------------------

IMPLEMENTED_AGENTS: frozenset[str] = frozenset(e.agent for e in _ENTRIES)
IMPLEMENTED_CATEGORIES: frozenset[str] = frozenset(e.category for e in _ENTRIES)

AGENT_TO_CATEGORY: dict[str, str] = {e.agent: e.category for e in _ENTRIES}
CATEGORY_TO_AGENT: dict[str, str] = {e.category: e.agent for e in _ENTRIES}


def is_agent_implemented(agent: str) -> bool:
    return agent in IMPLEMENTED_AGENTS


def is_category_implemented(category: str) -> bool:
    return category in IMPLEMENTED_CATEGORIES


def supported_summary() -> str:
    """
    Human-readable list of implemented agents, for NOT_IMPLEMENTED error
    messages. Always reflects the current registry — never a hardcoded
    string that drifts.

    Example output:
        "file, system, web, audio, network, power, writing"
    """
    return ", ".join(e.agent for e in _ENTRIES)


def not_implemented_detail(*, category: str | None = None, agent: str | None = None) -> str:
    """
    Build a user-facing NOT_IMPLEMENTED detail string. Callers pass
    whichever of category/agent they have; the message is generated
    from the live registry so it never drifts.
    """
    if category is not None:
        return (
            f"Category '{category}' is not yet implemented. "
            f"Supported agents: {supported_summary()}."
        )
    if agent is not None:
        return (
            f"Agent '{agent}' is not yet implemented. "
            f"Supported agents: {supported_summary()}."
        )
    return f"This request is not yet implemented. Supported agents: {supported_summary()}."
