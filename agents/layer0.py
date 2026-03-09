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
# Public API
# ---------------------------------------------------------------------------

def match(user_text: str) -> Layer0Result:
    """
    Try all patterns against user_text. Returns the first match.
    Returns Layer0Result(matched=False) if nothing matches.
    Typical latency: <0.05ms.
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
            )
    latency_ms = (time.monotonic() - t0) * 1000
    return Layer0Result(matched=False, latency_ms=latency_ms)
