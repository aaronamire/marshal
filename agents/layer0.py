"""
Layer 0 — sub-millisecond regex pattern matcher.

Handles unambiguous file commands with explicit paths.
Only fires for high-confidence, single-action cases.
Falls through silently (matched=False) for anything ambiguous or multi-step.

Design principles:
  - Conservative: false negatives (fall-through to L1/L2) are fine.
    False positives (wrong match) waste user time and break trust.
  - Only match when an explicit path/glob is present in the input.
  - Never match vague queries ("find large files", "what's in here").
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class Layer0Result:
    matched: bool
    action_type: Optional[str] = None
    params: Optional[dict] = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    is_implemented: bool = True
    agent: str = "file"
    category: str = "file_task"
    preview_required: Optional[bool] = None  # overrides destructive-based default


# ---------------------------------------------------------------------------
# Path / glob sub-patterns
# ---------------------------------------------------------------------------

# Explicit path: starts with ~, /, or ./ — OR is a quoted string.
# Anchored to avoid matching bare words.
_PATH = r'("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[~/\.][^\s]*)'

# Glob pattern: anything containing * — e.g. "*.py", "~/projects/*.log"
_GLOB = r'("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|\S*\*\S*|[~/\.][^\s]*)'


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        return s[1:-1]
    return s


def _norm(s: str) -> str:
    return _unquote(s).strip()


# ---------------------------------------------------------------------------
# Rule table
# ---------------------------------------------------------------------------

_RULES: list = []
# Each entry: (compiled_pattern, action_type, destructive, param_extractor)


def _rule(pattern: str, action_type: str, destructive: bool, extractor) -> None:
    _RULES.append((re.compile(pattern, re.IGNORECASE), action_type, destructive, extractor))


# READ — "read ~/file.txt", "cat /etc/hosts", "show contents of ~/notes.txt"
# "print /tmp/log.txt", "display ~/readme.md"
_rule(
    r'^\s*(?:read|cat|show\s+contents?\s+of|print|display)\s+' + _PATH + r'\s*$',
    "READ", False,
    lambda m: {"path": _norm(m.group(1))},
)

# MOVE — "move ~/a.txt to ~/b.txt", "mv /src /dst", "rename ~/old.txt to ~/new.txt"
_rule(
    r'^\s*(?:move|mv|rename)\s+' + _PATH + r'\s+to\s+' + _PATH + r'\s*$',
    "MOVE", True,
    lambda m: {"source": _norm(m.group(1)), "destination": _norm(m.group(2))},
)

# COPY — "copy ~/a.txt to ~/b.txt", "cp /src /dst"
_rule(
    r'^\s*(?:copy|cp)\s+' + _PATH + r'\s+to\s+' + _PATH + r'\s*$',
    "COPY", False,
    lambda m: {"source": _norm(m.group(1)), "destination": _norm(m.group(2))},
)

# DELETE — "delete ~/tmp/file.txt", "remove /tmp/foo", "rm ~/junk.txt"
_rule(
    r'^\s*(?:delete|remove|rm)\s+' + _PATH + r'\s*$',
    "DELETE", True,
    lambda m: {"path": _norm(m.group(1))},
)

# MOVE (shell-style) — "mv /src /dst" (no "to" keyword)
_rule(
    r'^\s*mv\s+' + _PATH + r'\s+' + _PATH + r'\s*$',
    "MOVE", True,
    lambda m: {"source": _norm(m.group(1)), "destination": _norm(m.group(2))},
)

# COPY (shell-style) — "cp /src /dst" (no "to" keyword)
_rule(
    r'^\s*cp\s+' + _PATH + r'\s+' + _PATH + r'\s*$',
    "COPY", False,
    lambda m: {"source": _norm(m.group(1)), "destination": _norm(m.group(2))},
)

# QUERY (list directory) — "list ~/dir", "ls ~/projects", "list files in ~/dir"
_rule(
    r'^\s*(?:list(?:\s+files?(?:\s+in)?)?|ls)\s+' + _PATH + r'\s*$',
    "QUERY", False,
    lambda m: {"path": _norm(m.group(1)), "search_type": "name"},
)

# QUERY (find by glob) — "find *.py in ~/dev", "find ~/projects/*.log"
_rule(
    r'^\s*find\s+' + _GLOB + r'(?:\s+in\s+' + _PATH + r')?\s*$',
    "QUERY", False,
    lambda m: {
        "pattern": _norm(m.group(1)),
        **({"path": _norm(m.group(2))} if m.group(2) else {}),
        "search_type": "glob",
    },
)


# ---------------------------------------------------------------------------
# System query rules (agent="system", category="system_task")
# These bypass L1+L2 entirely — the SystemAgent uses psutil, no LLM needed.
# ---------------------------------------------------------------------------

_SYSTEM_RULES: list = []
# Each entry: (compiled_pattern, query_type, param_extractor_or_None)


def _sys_rule(pattern: str, query_type: str, extractor=None) -> None:
    _SYSTEM_RULES.append(
        (re.compile(pattern, re.IGNORECASE), query_type, extractor))


# memory / RAM
_sys_rule(r'\b(?:ram|memory)\b', "memory")
_sys_rule(r'\bhow\s+much\s+(?:ram|memory)\b', "memory")

# CPU
_sys_rule(r'\bcpu\s*(?:usage|load|temp|temperature|info)?\b', "cpu")
_sys_rule(r'\bprocessor\b', "cpu")

# disk / storage
_sys_rule(r'\bdisk\s*(?:usage|space|info)?\b', "disk")
_sys_rule(r'\bstorage\s*(?:usage|space|left)?\b', "disk")
_sys_rule(r'\bhow\s+much\s+(?:disk|storage|space)\b', "disk")

# processes
_sys_rule(r'\b(?:running\s+)?processes\b', "processes")
_sys_rule(r'\btop\s+processes\b', "processes")
_sys_rule(r'\bwhat(?:\'s|\s+is)\s+(?:running|using)\b', "processes")

# uptime
_sys_rule(r'\buptime\b', "uptime")
_sys_rule(r'\bhow\s+long\s+.*\b(?:running|on|up)\b', "uptime")


# ---------------------------------------------------------------------------
# App launch / terminate rules (agent="system", category="system_task")
# Launch: WRITE, params.program. Terminate: DELETE, params.target (destructive).
# Conservative: only matches explicit "open/launch/start/run <app>" patterns.
# ---------------------------------------------------------------------------

_APP_LAUNCH_RULES: list = []
_APP_TERMINATE_RULES: list = []

# Common app name: single word, no path chars, lowercase
_APP_NAME = r'(\w[\w\-\.]*)'


def _launch_rule(pattern: str) -> None:
    _APP_LAUNCH_RULES.append(re.compile(pattern, re.IGNORECASE))


def _terminate_rule(pattern: str) -> None:
    _APP_TERMINATE_RULES.append(re.compile(pattern, re.IGNORECASE))


# Launch patterns: "open firefox", "launch gimp", "start htop", "run vlc"
_launch_rule(r'^\s*(?:open|launch|start|run)\s+' + _APP_NAME + r'\s*$')

# Terminate patterns: "close firefox", "kill firefox", "quit vlc", "terminate gimp"
_terminate_rule(r'^\s*(?:close|kill|quit|terminate|stop)\s+' + _APP_NAME + r'\s*$')

# "kill PID 1234" or "kill 1234"
_terminate_rule(r'^\s*kill\s+(?:pid\s+)?(\d+)\s*$')


# ---------------------------------------------------------------------------
# Briefing rules — "good morning", "what changed", "briefing"
# Returns agent="briefing", category="briefing" so callers can dispatch directly.
# ---------------------------------------------------------------------------

_BRIEFING_RULES: list[re.Pattern] = []


def _briefing_rule(pattern: str) -> None:
    _BRIEFING_RULES.append(re.compile(pattern, re.IGNORECASE))


_briefing_rule(r'^\s*(?:good\s+)?morning\s*$')
_briefing_rule(r'^\s*briefing\s*$')
_briefing_rule(r'^\s*brief\s+me\s*$')
_briefing_rule(r'^\s*what(?:\'s|\s+has)?\s+changed\b')
_briefing_rule(r'^\s*what\s+happened\b')
_briefing_rule(r'^\s*what(?:\'s|\s+is)\s+new\b')
_briefing_rule(r'^\s*catch\s+me\s+up\b')
_briefing_rule(r'^\s*status\s+update\s*$')


# ---------------------------------------------------------------------------
# NOT_IMPLEMENTED fast-path patterns
# Matched inputs are flagged is_implemented=False → caller raises immediately.
# Patterns are intentionally broad (no explicit path required).
# ---------------------------------------------------------------------------

_NOT_IMPL_RULES: list[re.Pattern] = []


def _not_impl(pattern: str) -> None:
    _NOT_IMPL_RULES.append(re.compile(pattern, re.IGNORECASE))


# email — write/send/compose/draft + email/message/mail
_not_impl(r'^\s*(?:write|send|compose|draft)\b.+\b(?:email|e-mail|message|mail)\b')
_not_impl(r'^\s*(?:send|compose|draft)\b.+\bto\s+(?!~|/|\.)[\w]')  # "to <person>" not "to ~/path"

# system — hardware controls not handled by SystemAgent
_not_impl(r'\b(?:battery|wifi|wi-fi|brightness|volume)\b')

# web — now implemented by WebAgent (NOT_IMPL patterns removed)

# writing — write a document/report / summarize text
_not_impl(r'\bwrite\s+a\s+(?:document|report|letter|essay|blog\s+post)\b')
_not_impl(r'\bsummariz[ei]\s+(?:this|the)\s+(?:text|document|article|file)\b')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match(user_text: str) -> Layer0Result:
    """
    Try all patterns against user_text. Returns the first match.
    Returns Layer0Result(matched=False) if nothing matches.
    Typical latency: <0.05ms.

    If a NOT_IMPLEMENTED pattern matches, returns matched=True, is_implemented=False.
    The caller should raise LeavesError(NOT_IMPLEMENTED) immediately in that case.
    """
    t0 = time.monotonic()
    for pattern, action_type, destructive, extractor in _RULES:
        m = pattern.match(user_text)
        if m:
            try:
                params = extractor(m)
            except Exception:
                continue  # malformed match — skip, fall through to L1/L2
            latency_ms = (time.monotonic() - t0) * 1000
            return Layer0Result(
                matched=True,
                action_type=action_type,
                params={**params, "destructive": destructive},
                confidence=0.95,
                latency_ms=latency_ms,
                is_implemented=True,
            )
    # System queries — .search() not .match(), since keywords appear mid-sentence
    for pattern, query_type, extractor in _SYSTEM_RULES:
        if pattern.search(user_text):
            params = {"query_type": query_type}
            if extractor:
                try:
                    params.update(extractor(pattern.search(user_text)))
                except Exception:
                    pass
            latency_ms = (time.monotonic() - t0) * 1000
            return Layer0Result(
                matched=True,
                action_type="QUERY",
                params={**params, "destructive": False},
                confidence=0.95,
                latency_ms=latency_ms,
                is_implemented=True,
                agent="system",
                category="system_task",
            )
    # App launch — "open firefox", "launch gimp"
    for pattern in _APP_LAUNCH_RULES:
        m = pattern.match(user_text)
        if m:
            program = m.group(1).strip()
            latency_ms = (time.monotonic() - t0) * 1000
            return Layer0Result(
                matched=True,
                action_type="WRITE",
                params={"program": program, "destructive": False},
                confidence=0.95,
                latency_ms=latency_ms,
                is_implemented=True,
                agent="system",
                category="system_task",
                preview_required=True,
            )
    # App terminate — "close firefox", "kill 1234"
    for pattern in _APP_TERMINATE_RULES:
        m = pattern.match(user_text)
        if m:
            target = m.group(1).strip()
            latency_ms = (time.monotonic() - t0) * 1000
            return Layer0Result(
                matched=True,
                action_type="DELETE",
                params={"target": target, "destructive": True},
                confidence=0.95,
                latency_ms=latency_ms,
                is_implemented=True,
                agent="system",
                category="system_task",
            )
    # Briefing — "good morning", "what changed", "briefing"
    for pattern in _BRIEFING_RULES:
        if pattern.match(user_text):
            latency_ms = (time.monotonic() - t0) * 1000
            return Layer0Result(
                matched=True,
                action_type="BRIEFING",
                params={},
                confidence=0.95,
                latency_ms=latency_ms,
                is_implemented=True,
                agent="briefing",
                category="briefing",
            )
    for pattern in _NOT_IMPL_RULES:
        if pattern.search(user_text):
            latency_ms = (time.monotonic() - t0) * 1000
            return Layer0Result(matched=True, confidence=0.90, latency_ms=latency_ms, is_implemented=False)
    latency_ms = (time.monotonic() - t0) * 1000
    return Layer0Result(matched=False, latency_ms=latency_ms)
