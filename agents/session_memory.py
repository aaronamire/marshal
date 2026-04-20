"""
Session memory — bounded ring buffer of recent intents and their results.

The OS is otherwise stateless across intents: every parse() call sees only the
current user_text, with no recollection of what was said five seconds ago.
SessionMemory closes that gap. It records each completed turn (intent_id,
natural_text, goal_spec, results, summary, timestamp), persists to disk so the
user can quit and re-enter the REPL within a session window, and exposes:

  - recent(n)         — last N turns chronologically (most recent last)
  - latest()          — most recent turn or None
  - to_prompt_block() — formatted history block for the L2 system prompt
  - resolve_ref(s)    — resolve a "$prev.act-1.files[0].path" reference

Reference grammar (used by agents/refs.py):
    $prev[<N>].<action_id>.<dotted.path[idx]>...

  - prev   = prev1 = most recent completed turn
  - prev2  = the turn before that, etc.
  - The first path segment is an action_id (or the literal "summary" / "results")
  - Subsequent segments walk dicts/lists; "[N]" indexes a list

Persistence: ~/.marshal/session.jsonl. Rewritten on every record() so the file
always reflects the in-memory ring (newline-delimited JSON, one Turn per line).
On load, turns older than `session_window` seconds (default 1h) are dropped —
this is what defines a "session" of conversation.

Thread safety: SessionMemory is touched only from the REPL main thread.
No lock.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Defaults — small constants, kept here so callers can read them.
DEFAULT_CAPACITY: int = 16
DEFAULT_SESSION_WINDOW_SECONDS: float = 3600.0  # 1 hour idle = new session
DEFAULT_PATH: Path = Path.home() / ".marshal" / "session.jsonl"

# Caps to keep persistence bounded even if the user pastes huge inputs.
_MAX_NATURAL_TEXT: int = 1000
_MAX_SUMMARY: int = 500


@dataclass
class Turn:
    """One completed intent + its outcome."""
    intent_id: str
    natural_text: str
    goal_spec: dict
    results: dict
    summary: str
    ts: float
    # Pre-computed list of action_ids in order, for the prompt block —
    # avoids re-walking goal_spec at every render.
    action_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Turn":
        return cls(
            intent_id=d.get("intent_id", ""),
            natural_text=d.get("natural_text", ""),
            goal_spec=d.get("goal_spec", {}) or {},
            results=d.get("results", {}) or {},
            summary=d.get("summary", ""),
            ts=float(d.get("ts", 0.0)),
            action_ids=list(d.get("action_ids") or []),
        )

    def to_dict(self) -> dict:
        return asdict(self)


class SessionMemory:
    """
    Bounded ring buffer of recent turns, persisted to JSONL.

    Turns older than `session_window` are evicted at load time. Capacity caps
    the in-memory ring; record() drops the oldest turn when full. Persistence
    is best-effort — IO errors are logged once and ignored thereafter.
    """

    def __init__(
        self,
        path: Path | None = None,
        capacity: int = DEFAULT_CAPACITY,
        session_window: float = DEFAULT_SESSION_WINDOW_SECONDS,
    ):
        self._path = path if path is not None else DEFAULT_PATH
        self._capacity = max(1, int(capacity))
        self._window = float(session_window)
        self._turns: list[Turn] = []
        self._io_failed = False
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Read the JSONL file, drop stale turns, populate the ring."""
        if not self._path.exists():
            return
        try:
            now = time.time()
            cutoff = now - self._window
            kept: list[Turn] = []
            for line in self._path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                turn = Turn.from_dict(obj)
                if turn.ts >= cutoff:
                    kept.append(turn)
            # Keep only the most recent `capacity` after the time filter.
            kept.sort(key=lambda t: t.ts)
            if len(kept) > self._capacity:
                kept = kept[-self._capacity:]
            self._turns = kept
        except OSError:
            self._io_failed = True

    def _persist(self) -> None:
        if self._io_failed:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            with tmp.open("w") as f:
                for t in self._turns:
                    f.write(json.dumps(t.to_dict(), default=str) + "\n")
            tmp.replace(self._path)
        except OSError:
            self._io_failed = True

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def record(
        self,
        intent_id: str,
        natural_text: str,
        goal_spec: dict,
        results: dict,
        summary: str,
    ) -> Turn:
        """Append a completed turn to the ring and persist."""
        action_ids = [
            a.get("action_id", "")
            for a in (goal_spec.get("actions") or [])
            if isinstance(a, dict)
        ]
        turn = Turn(
            intent_id=intent_id,
            natural_text=natural_text[:_MAX_NATURAL_TEXT],
            goal_spec=goal_spec,
            results=results,
            summary=summary[:_MAX_SUMMARY],
            ts=time.time(),
            action_ids=action_ids,
        )
        self._turns.append(turn)
        if len(self._turns) > self._capacity:
            self._turns = self._turns[-self._capacity:]
        self._persist()
        return turn

    def clear(self) -> None:
        """Drop all turns and remove the persisted file."""
        self._turns = []
        try:
            if self._path.exists():
                self._path.unlink()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._turns)

    def latest(self) -> Turn | None:
        return self._turns[-1] if self._turns else None

    def recent(self, n: int = 5) -> list[Turn]:
        """Return the most recent `n` turns in chronological order."""
        if n <= 0:
            return []
        return list(self._turns[-n:])

    def turn_by_back_index(self, back: int) -> Turn | None:
        """
        back=1 -> most recent turn (latest), back=2 -> the one before, etc.
        Returns None if `back` is out of range.
        """
        if back < 1 or back > len(self._turns):
            return None
        return self._turns[-back]

    def to_prompt_block(self, n: int = 3) -> str:
        """
        Format the last `n` turns as a compact context block to inject into
        the L2 system prompt. Empty string if there are no turns.

        Format (per turn):
            [T-1 22s ago] "find pdfs in ~/Downloads"
              act-1 file.QUERY -> 12 files
              summary: Completed 1 action(s). Found 12 file(s).

        The model can reference these via $prev.act-1.* in subsequent params.
        """
        if not self._turns:
            return ""
        now = time.time()
        slice_ = self._turns[-n:]
        lines: list[str] = ["<SESSION_HISTORY>"]
        # T-1 = most recent. T-N = oldest in slice.
        # Iterate so the most recent is labeled T-1.
        for offset, turn in enumerate(reversed(slice_), start=1):
            age = max(0.0, now - turn.ts)
            age_str = _fmt_age(age)
            lines.append(f"[T-{offset} {age_str} ago] {turn.natural_text!r}")
            for aid in turn.action_ids:
                action = _find_action(turn.goal_spec, aid)
                if not action:
                    continue
                atype = action.get("type", "?")
                agent = action.get("agent", "?")
                preview = _result_preview(turn.results.get(aid))
                lines.append(f"  {aid} {agent}.{atype} -> {preview}")
            if turn.summary:
                lines.append(f"  summary: {turn.summary}")
        lines.append("</SESSION_HISTORY>")
        lines.append(
            "You may reference prior turn results in action params with the "
            "$prev[<N>].<action_id>.<path> syntax — e.g., "
            "$prev.act-1.files[0].path. The OS will resolve these before "
            "execution."
        )
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _find_action(goal_spec: dict, action_id: str) -> dict | None:
    for a in goal_spec.get("actions") or []:
        if isinstance(a, dict) and a.get("action_id") == action_id:
            return a
    return None


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    return f"{int(seconds / 3600)}h"


def _result_preview(value: Any) -> str:
    """A one-line preview of an action result for the prompt block."""
    if value is None:
        return "(no result)"
    if isinstance(value, dict):
        if "error" in value:
            return f"ERROR {value.get('error', '')[:60]}"
        if "files" in value and isinstance(value["files"], list):
            return f"{value.get('count', len(value['files']))} files"
        if "results" in value and isinstance(value["results"], list):
            return f"{value.get('result_count', len(value['results']))} results"
        if "content" in value and isinstance(value["content"], str):
            n = value.get("content_length", len(value["content"]))
            return f"{n} chars"
        # Fall through: show a few key=val pairs
        pairs = []
        for k, v in list(value.items())[:3]:
            if isinstance(v, (str, int, float, bool)):
                pairs.append(f"{k}={v}")
        return "{" + ", ".join(pairs) + "}" if pairs else "{...}"
    if isinstance(value, list):
        return f"[{len(value)}]"
    return str(value)[:80]
