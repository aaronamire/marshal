#!/usr/bin/env python3
"""
Leaves OS Phase 1 — Main CLI entry point.

Two-stage pipeline:
  Layer 1 (3-8ms):  sklearn classifier → instant category feedback
  Layer 2 (26s+):   Llama GoalSpec generation → full execution plan

Orchestration delegated to agentd.AgentCoordinator.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from agentd import AgentCoordinator
from agents.intent_parser import IntentParser
from agents.state_machine import IntentLifecycle, IntentState
from config import (
    APP_NAME, APP_VERSION,
    LEAVES_PRIMARY_COLOR, LEAVES_SUCCESS_COLOR,
    LEAVES_ERROR_COLOR, LEAVES_WARNING_COLOR, LEAVES_DIM_COLOR,
)
from db.audit import (
    get_db, log_intent_created, log_state_transition,
    complete_intent, log_error, get_recent_intents,
    get_intent_transitions, get_intent_actions,
)
from errors import LeavesError

console = Console()

# Module-level singletons — initialized once in main()
_db = None
_parser: IntentParser | None = None


def _get_db():
    global _db
    if _db is None:
        _db = get_db()
    return _db


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _banner() -> None:
    title = Text(f" {APP_NAME} ", style=f"bold {LEAVES_PRIMARY_COLOR}")
    subtitle = Text(f"v{APP_VERSION} · Phase 1", style=LEAVES_DIM_COLOR)
    console.print(Panel(
        f"{title}\n{subtitle}",
        border_style=LEAVES_PRIMARY_COLOR,
        padding=(0, 2),
    ))
    console.print()


def _show_error(msg: str) -> None:
    console.print(f"[{LEAVES_ERROR_COLOR}]Error:[/{LEAVES_ERROR_COLOR}] {msg}")


def _show_warning(msg: str) -> None:
    console.print(f"[{LEAVES_WARNING_COLOR}]Warning:[/{LEAVES_WARNING_COLOR}] {msg}")


def _render_file_list(result: dict) -> None:
    files = result.get("files", [])
    path = result.get("path", "?")
    count = result.get("count", 0)

    if not files:
        console.print(f"[{LEAVES_DIM_COLOR}]No files found in {path}[/{LEAVES_DIM_COLOR}]")
        return

    table = Table(
        title=f"{count} file(s) in {path}",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style=f"bold {LEAVES_PRIMARY_COLOR}",
    )
    table.add_column("Name", style="white")
    table.add_column("Size", justify="right", style=LEAVES_DIM_COLOR)
    table.add_column("Modified", style=LEAVES_DIM_COLOR)
    table.add_column("Type", style=LEAVES_DIM_COLOR)

    for f in files:
        size = _fmt_size(f.get("size_bytes", 0))
        mtime = _fmt_time(f.get("mtime", 0))
        ftype = "dir" if f.get("is_dir") else "file"
        table.add_row(f["name"], size, mtime, ftype)

    console.print(table)


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _fmt_time(ts: float) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Authorization dialog
# ---------------------------------------------------------------------------

def _show_auth_dialog(goal_spec: dict) -> bool:
    """
    Show confirmation dialog for destructive intents.
    ALWAYS shown when preview_required=True — not advisory.
    """
    auth = goal_spec.get("authorization", {})
    resources = auth.get("resources", [])
    reversible = auth.get("reversible", True)

    console.print()
    console.print(Panel(
        f"[bold {LEAVES_WARNING_COLOR}]Authorization Required[/bold {LEAVES_WARNING_COLOR}]\n\n"
        f"Intent: [white]{goal_spec['natural_text']}[/white]\n"
        f"Resources: {', '.join(resources) or 'none'}\n"
        f"Reversible: {'yes' if reversible else '[red]NO[/red]'}\n\n"
        "Actions:\n" + "\n".join(
            f"  [{LEAVES_DIM_COLOR}]{a['action_id']}[/{LEAVES_DIM_COLOR}] "
            f"[yellow]{a['type']}[/yellow] ({a.get('agent', '?')})"
            for a in goal_spec.get("actions", [])
        ),
        border_style=LEAVES_WARNING_COLOR,
        title="Confirm",
    ))

    try:
        answer = input("Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    return answer in ("y", "yes")


# ---------------------------------------------------------------------------
# Core intent handler
# ---------------------------------------------------------------------------

def handle_intent(user_text: str) -> None:
    """
    Full intent lifecycle:
      PENDING → PARSING → [AWAITING_AUTH] → EXECUTING → DONE | FAILED | CANCELLED

    Layer 1 fires the on_classified callback immediately (inside parser.parse()).
    Layer 2 (Llama) runs and returns the full GoalSpec.
    Execution is delegated to AgentCoordinator.
    """
    db = _get_db()
    parser = _parser

    # --- server availability ---
    if not parser._client.is_available():
        _show_error(
            "Inference server is not running.\n"
            "  Start it: bash scripts/start-inference.sh"
        )
        return

    t_start = time.monotonic()

    # Layer 1 callback — fires immediately from inside parser.parse()
    def on_classified(result):
        color = LEAVES_PRIMARY_COLOR if result.is_confident else LEAVES_WARNING_COLOR
        console.print(
            f"  [bold {color}]◆[/bold {color}] "
            f"{result.category} · {result.confidence:.0%} · {result.latency_ms:.0f}ms"
        )

    parser._on_classified = on_classified

    # --- PARSING (includes Layer 1 callback + Layer 2 LLM) ---
    lifecycle = IntentLifecycle(intent_id="pending")
    try:
        lifecycle.transition(IntentState.PARSING)
        with console.status(f"[{LEAVES_DIM_COLOR}]Generating plan…[/{LEAVES_DIM_COLOR}]"):
            goal_spec = parser.parse(user_text)

        intent_id = goal_spec["intent_id"]
        lifecycle = IntentLifecycle(intent_id=intent_id)
        lifecycle.transition(IntentState.PARSING)

        log_intent_created(db, intent_id, user_text, goal_spec)
        log_state_transition(db, intent_id, "PENDING", "PARSING")

        confidence = goal_spec.get("metadata", {}).get("confidence", 0.0)
        latency = goal_spec.get("metadata", {}).get("parse_latency_ms", 0)
        console.print(
            f"  [{LEAVES_DIM_COLOR}]Plan: "
            f"{len(goal_spec.get('actions', []))} action(s) via {goal_spec['category']} "
            f"· L2 confidence {confidence:.0%} · {latency:.0f}ms[/{LEAVES_DIM_COLOR}]"
        )

    except LeavesError as e:
        _show_error(e.user_message)
        log_error(_get_db(), e.code.value, e.detail)
        return

    # --- AUTHORIZATION ---
    auth = goal_spec.get("authorization", {})
    if auth.get("preview_required", False):
        lifecycle.transition(IntentState.AWAITING_AUTH)
        log_state_transition(db, intent_id, "PARSING", "AWAITING_AUTH")

        if not _show_auth_dialog(goal_spec):
            lifecycle.transition(IntentState.CANCELLED)
            log_state_transition(db, intent_id, "AWAITING_AUTH", "CANCELLED")
            complete_intent(db, intent_id, "CANCELLED", "User cancelled")
            _show_warning("Cancelled.")
            return
        # Coordinator will log AWAITING_AUTH → EXECUTING via from_state capture
    else:
        # No auth needed — log PARSING → EXECUTING (coordinator sees PARSING state)
        pass

    # --- EXECUTING (via AgentCoordinator) ---
    coordinator = AgentCoordinator(db)
    try:
        results, summary = coordinator.execute(goal_spec, lifecycle)
    except LeavesError as e:
        _show_error(e.user_message)
        log_state_transition(db, intent_id, "EXECUTING", "FAILED")
        complete_intent(
            db, intent_id, "FAILED", e.detail,
            duration_ms=(time.monotonic() - t_start) * 1000,
        )
        return

    # --- Render results ---
    for action in goal_spec.get("actions", []):
        action_id = action.get("action_id", "unknown")
        result = results.get(action_id, {})

        if isinstance(result, dict) and "error" in result:
            _show_error(result["error"])
            continue

        atype = action.get("type", "").upper()
        if atype in ("QUERY", "READ") and isinstance(result, dict) and "files" in result:
            _render_file_list(result)
        elif atype == "READ" and isinstance(result, dict) and "content" in result:
            console.print(Panel(
                result["content"][:2000],
                title=result.get("path", ""),
                border_style=LEAVES_PRIMARY_COLOR,
            ))
        elif result:
            console.print(
                f"  [{LEAVES_SUCCESS_COLOR}]✓[/{LEAVES_SUCCESS_COLOR}] {result}"
            )

    # --- DONE ---
    lifecycle.transition(IntentState.DONE)
    log_state_transition(db, intent_id, "EXECUTING", "DONE")
    duration_ms = (time.monotonic() - t_start) * 1000
    complete_intent(db, intent_id, "DONE", summary, duration_ms=duration_ms)

    reversible = auth.get("reversible", True)
    rev_str = "reversible" if reversible else "[red]irreversible[/red]"
    console.print(
        f"  [{LEAVES_DIM_COLOR}]Done in {duration_ms:.0f}ms · {rev_str} · logged[/{LEAVES_DIM_COLOR}]"
    )


# ---------------------------------------------------------------------------
# Built-in commands
# ---------------------------------------------------------------------------

def cmd_history() -> None:
    db = _get_db()
    intents = get_recent_intents(db, limit=20)
    if not intents:
        console.print(f"[{LEAVES_DIM_COLOR}]No history yet.[/{LEAVES_DIM_COLOR}]")
        return

    table = Table(
        title="Recent Intents",
        box=box.SIMPLE_HEAD,
        header_style=f"bold {LEAVES_PRIMARY_COLOR}",
    )
    table.add_column("#", style=LEAVES_DIM_COLOR, justify="right")
    table.add_column("Intent", style="white", max_width=50)
    table.add_column("Category", style=LEAVES_DIM_COLOR)
    table.add_column("State", style="white")
    table.add_column("Duration", style=LEAVES_DIM_COLOR, justify="right")
    table.add_column("Time", style=LEAVES_DIM_COLOR)

    state_styles = {
        "DONE": LEAVES_SUCCESS_COLOR,
        "FAILED": LEAVES_ERROR_COLOR,
        "CANCELLED": LEAVES_WARNING_COLOR,
    }

    for i, row in enumerate(intents, 1):
        state = row["state"]
        color = state_styles.get(state, "white")
        state_str = f"[{color}]{state}[/{color}]"
        dur = f"{row['duration_ms']:.0f}ms" if row["duration_ms"] else "—"
        table.add_row(
            str(i), row["natural_text"][:50],
            row["category"] or "—", state_str,
            dur, _fmt_time(row["created_at"]),
        )

    console.print(table)


def cmd_detail(intent_id_prefix: str) -> None:
    db = _get_db()
    intents = get_recent_intents(db, limit=100)
    matches = [i for i in intents if i["intent_id"].startswith(intent_id_prefix)]

    if not matches:
        _show_error(f"No intent found with prefix '{intent_id_prefix}'")
        return

    intent = matches[0]
    dur = f"{intent['duration_ms']:.0f}ms" if intent["duration_ms"] else "—"
    console.print(Panel(
        f"[bold]Intent:[/bold] {intent['natural_text']}\n"
        f"[bold]ID:[/bold] {intent['intent_id']}\n"
        f"[bold]Category:[/bold] {intent['category']}\n"
        f"[bold]State:[/bold] {intent['state']}\n"
        f"[bold]Duration:[/bold] {dur}",
        title="Intent Detail",
        border_style=LEAVES_PRIMARY_COLOR,
    ))

    transitions = get_intent_transitions(db, intent["intent_id"])
    if transitions:
        t_table = Table(title="State Transitions", box=box.SIMPLE_HEAD)
        t_table.add_column("From", style=LEAVES_DIM_COLOR)
        t_table.add_column("To", style="white")
        t_table.add_column("At", style=LEAVES_DIM_COLOR)
        for t in transitions:
            t_table.add_row(t["from_state"], t["to_state"], _fmt_time(t["transitioned_at"]))
        console.print(t_table)

    actions = get_intent_actions(db, intent["intent_id"])
    if actions:
        a_table = Table(title="Actions", box=box.SIMPLE_HEAD)
        a_table.add_column("ID", style=LEAVES_DIM_COLOR)
        a_table.add_column("Type", style="white")
        a_table.add_column("Agent", style=LEAVES_DIM_COLOR)
        a_table.add_column("Status", style="white")
        for a in actions:
            if a["error_code"]:
                status = f"[{LEAVES_ERROR_COLOR}]{a['error_code']}[/{LEAVES_ERROR_COLOR}]"
            else:
                status = f"[{LEAVES_SUCCESS_COLOR}]OK[/{LEAVES_SUCCESS_COLOR}]"
            a_table.add_row(a["action_id"], a["action_type"], a["agent"], status)
        console.print(a_table)


def cmd_help() -> None:
    console.print(Panel(
        "[bold]Commands:[/bold]\n\n"
        "  [white]<natural language>[/white]  — execute an intent\n"
        "  [white]history[/white]             — show recent intents\n"
        "  [white]detail <id-prefix>[/white]  — show intent audit trail\n"
        "  [white]help[/white]                — show this help\n"
        "  [white]quit[/white] / [white]exit[/white]          — exit\n\n"
        "[bold]Examples:[/bold]\n"
        "  find all PDFs in my Downloads folder\n"
        "  list Python files in ~/dev\n"
        "  show me files modified today in ~",
        title="Leaves OS Help",
        border_style=LEAVES_PRIMARY_COLOR,
    ))


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def repl() -> None:
    global _parser

    _banner()

    # Try to load Layer 1 classifier
    classifier = None
    try:
        from agents.classifier import IntentClassifier
        classifier = IntentClassifier()
        console.print(f"[{LEAVES_DIM_COLOR}]Layer 1 classifier: loaded (1.8ms avg)[/{LEAVES_DIM_COLOR}]")
    except FileNotFoundError:
        console.print(
            f"[{LEAVES_WARNING_COLOR}]Layer 1 classifier not found.[/{LEAVES_WARNING_COLOR}] "
            f"[{LEAVES_DIM_COLOR}]Run: python3 scripts/train_classifier.py[/{LEAVES_DIM_COLOR}]"
        )
    except Exception as e:
        console.print(f"[{LEAVES_DIM_COLOR}]Layer 1 unavailable: {e}[/{LEAVES_DIM_COLOR}]")

    _parser = IntentParser(classifier=classifier)

    if not _parser._client.is_available():
        console.print(
            f"[{LEAVES_WARNING_COLOR}]Inference server not running.[/{LEAVES_WARNING_COLOR}] "
            f"Start: bash scripts/start-inference.sh"
        )
    console.print()

    while True:
        try:
            raw = input("[leaves] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break

        if not raw:
            continue

        lower = raw.lower()
        if lower in ("quit", "exit", "q"):
            console.print("Goodbye.")
            break
        elif lower in ("help", "?"):
            cmd_help()
        elif lower in ("history", "hist"):
            cmd_history()
        elif lower.startswith("detail "):
            cmd_detail(raw[7:].strip())
        else:
            handle_intent(raw)

        console.print()


if __name__ == "__main__":
    repl()
