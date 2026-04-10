"""
Reference resolution — expand $prev.* references in goalspec params.

The L2 model is encouraged to reference prior turn results via a small
embedded grammar:

    $prev[<N>].<action_id>.<dotted.path[idx]>...

  - `prev`  / `prev1` → most recent completed turn
  - `prev2` / `prev3` → older turns (back-index, 1-based)
  - The first segment after the prev token is an action_id, with two
    special literals:
        summary  → the turn's natural-language summary string
        results  → the entire results dict for the turn
  - Subsequent segments walk dicts and lists. `[N]` indexes a list.

Examples:
    $prev.act-1.files[0].path
    $prev2.act-2.results[3].url
    $prev.summary
    $prev.act-1                 (resolves to the entire result dict)

Resolution semantics:

  - resolve_refs(value, session) walks any nested dict/list/scalar and
    returns a new structure with references expanded.
  - If a STRING value is *exactly* a single reference token (after strip),
    the reference is replaced with the resolved value preserving its
    original Python type (str, int, list, dict, …). This is what makes
    `{"path": "$prev.act-1.files[0].path"}` work end-to-end without
    forcing the path back to a stringified repr.
  - If the string contains a reference inline (mixed with other text),
    each match is substituted as a string (str() coercion).
  - If a reference cannot be resolved (no such turn, missing key, index
    out of range), the original literal text is preserved unchanged.
    The action will then fail at execution time with a meaningful error
    instead of silently doing the wrong thing.

This module is pure: it never touches disk, never logs, never raises.
It's called from the REPL on the parsed GoalSpec *before* the spec is
sent to agentd, so the sandboxed runner sees fully-resolved params and
needs no awareness of session memory at all.
"""
from __future__ import annotations

import re
from typing import Any

# A reference is: $prev | $prevN  followed by zero or more
# .ident or [digits] segments. Identifiers allow letters, digits, _ and -
# (action_ids use a hyphen, e.g. "act-1").
_REF_RE = re.compile(
    r"\$(prev\d*)((?:\.[A-Za-z0-9_\-]+|\[\d+\])*)"
)
# A path step: either ".name" or "[N]"
_STEP_RE = re.compile(r"\.([A-Za-z0-9_\-]+)|\[(\d+)\]")

_SENTINEL = object()


def resolve_refs(value: Any, session: Any) -> Any:
    """
    Recursively walk `value` and resolve any $prev[N].* references using
    `session` (a SessionMemory). Returns a new structure; the input is not
    mutated. Unresolvable references are left as-is.
    """
    if isinstance(value, dict):
        return {k: resolve_refs(v, session) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_refs(v, session) for v in value]
    if isinstance(value, str):
        return _resolve_string(value, session)
    return value


def _resolve_string(s: str, session: Any) -> Any:
    """
    Resolve references inside a single string.

    Two modes:
      - Whole-string: the trimmed string is exactly one reference. We
        return the resolved value preserving its Python type.
      - Inline: one or more references appear mixed with other text. We
        do per-match string substitution.
    """
    stripped = s.strip()
    whole = _REF_RE.fullmatch(stripped)
    if whole is not None:
        resolved = _resolve_match(whole, session)
        if resolved is _SENTINEL:
            return s  # leave unchanged so the user sees the bad ref
        return resolved

    if "$prev" not in s:
        return s

    def _sub(m: re.Match) -> str:
        resolved = _resolve_match(m, session)
        if resolved is _SENTINEL:
            return m.group(0)
        return str(resolved)

    return _REF_RE.sub(_sub, s)


def _resolve_match(m: re.Match, session: Any) -> Any:
    """
    Resolve a single regex match to a Python value, or _SENTINEL if the
    reference can't be resolved.
    """
    head = m.group(1)            # "prev" or "prevN"
    tail = m.group(2) or ""      # ".act-1.files[0].path" or ""

    # Back-index: prev / prev1 → 1, prev2 → 2, …
    if head == "prev":
        back = 1
    else:
        try:
            back = int(head[4:])
        except ValueError:
            return _SENTINEL
        if back < 1:
            return _SENTINEL

    turn = session.turn_by_back_index(back) if session is not None else None
    if turn is None:
        return _SENTINEL

    steps = _STEP_RE.findall(tail)
    if not steps:
        # Bare $prev with no path — return the entire results dict.
        return turn.results

    # First step is special: it can be "summary", "results", or an action_id.
    first_name, first_idx = steps[0]
    if first_idx:
        # $prev[0] is meaningless — turns aren't a list at the top level.
        return _SENTINEL

    if first_name == "summary":
        cursor: Any = turn.summary
    elif first_name == "results":
        cursor = turn.results
    else:
        # Treat as action_id lookup in the results dict.
        if first_name not in turn.results:
            return _SENTINEL
        cursor = turn.results[first_name]

    for name, idx in steps[1:]:
        if idx:
            # List index
            try:
                i = int(idx)
            except ValueError:
                return _SENTINEL
            if not isinstance(cursor, list) or i < 0 or i >= len(cursor):
                return _SENTINEL
            cursor = cursor[i]
        else:
            # Dict key
            if not isinstance(cursor, dict) or name not in cursor:
                return _SENTINEL
            cursor = cursor[name]

    return cursor
