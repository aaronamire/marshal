#!/usr/bin/env python3
"""
Leaves OS Phase 0 — Main CLI entry point.

Wires together:
  - IntentParser (NL -> GoalSpec)
  - IntentLifecycle (state machine)
  - FileAgent (execution)
  - ToolFailureTracker (livelock prevention)
  - db/audit.py (logging)
  - rich (display)
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from agents.file_agent import FileAgent
from agents.intent_parser import IntentParser
from agents.state_machine import IntentLifecycle, IntentState
from agents.tool_failure_tracker import ToolFailureTracker
from config import (
    APP_NAME, APP_VERSION,
    LEAVES_PRIMARY_COLOR, LEAVES_SUCCESS_COLOR,
    LEAVES_ERROR_COLOR, LEAVES_WARNING_COLOR, LEAVES_DIM_COLOR,
)
from db.audit import (
    get_db, log_intent_created, log_state_transition,
    complete_intent, log_error, get_recent_intents, get_intent_transitions,
    get_intent_actions,
)
from errors import LeavesError, LeavesErrorCode

console = Console()
_db = None
_parser = None


def _get_db():
    global _db
    if _db is None:
        _db = get_db()
    return _db


def _get_parser():
    global _parser
    if _parser is None:
        _parser = IntentParser()
    return _parser


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _banner() -> None:
    title = Text(f" {APP_NAME} ", style=f"bold {LEAVES_PRIMARY_COLOR}")
    subtitle = Text(f"v{APP_VERSION} · Phase 0", style=LEAVES_DIM_COLOR)
    console.print(Panel(
        f"{title}\n{subtitle}",
        border_style=LEAVES_PRIMARY_COLOR,
        padding=(0, 2),
    ))
    console.print()


def _show_error(msg: str) -> None:
    console.print(f"[{LEAVES_ERROR_COLOR}]Error:[/{LEAVES_ERROR_COLOR}] {msg}")


def _show_success(msg: str) -> None:
    console.print(f"[{LEAVES_SUCCESS_COLOR}]{msg}[/{LEAVES_SUCCESS_COLOR}]")


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
    Display the authorization dialog for destructive intents.
    Returns True if the user confirms, False to cancel.
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
        f"Actions:\n" + "\n".join(
            f"  [{LEAVES_DIM_COLOR}]{a['action_id']}[/{LEAVES_DIM_COLOR}] "
            f"[yellow]{a['type']}[/yellow] "
            f"({a.get('agent', '?')})"
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
    PENDING -> PARSING -> [AWAITING_AUTH] -> EXECUTING -> DONE | FAILED | CANCELLED
    """
    db = _get_db()
    parser = _get_parser()

    # --- check inference server is available ---
    if not parser._client.is_available():
        _show_error(
            "Inference server is not running.\n"
            "  Start it with:  bash scripts/start-inference.sh\n"
            "  Then try again."
        )
        return

    # --- create lifecycle ---
    lifecycle = IntentLifecycle(intent_id="pending")
    t_start = time.monotonic()

    # --- PARSING ---
    try:
        lifecycle.transition(IntentState.PARSING)
        console.print(f"[{LEAVES_DIM_COLOR}]Parsing intent...[/{LEAVES_DIM_COLOR}]")

        goal_spec = parser.parse(user_text)
        intent_id = goal_spec["intent_id"]
        # Recreate lifecycle with real UUID
        lifecycle = IntentLifecycle(intent_id=intent_id)
        lifecycle.transition(IntentState.PARSING)

        log_intent_created(db, intent_id, user_text, goal_spec)
        log_state_transition(db, intent_id, "PENDING", "PARSING")

        confidence = goal_spec.get("metadata", {}).get("confidence", 0.0)
        latency = goal_spec.get("metadata", {}).get("parse_latency_ms", 0)
        console.print(
            f"[{LEAVES_DIM_COLOR}]Parsed: category={goal_spec['category']} "
            f"confidence={confidence:.0%} latency={latency:.0f}ms[/{LEAVES_DIM_COLOR}]"
        )

    except LeavesError as e:
        _show_error(e.user_message)
        log_error(db, e.code.value, e.detail)
        return

    # --- AUTHORIZATION (if required) ---
    auth = goal_spec.get("authorization", {})
    if auth.get("preview_required", False):
        lifecycle.transition(IntentState.AWAITING_AUTH)
        log_state_transition(db, intent_id, "PARSING", "AWAITING_AUTH")

        confirmed = _show_auth_dialog(goal_spec)
        if not confirmed:
            lifecycle.transition(IntentState.CANCELLED)
            log_state_transition(db, intent_id, "AWAITING_AUTH", "CANCELLED")
            complete_intent(db, intent_id, "CANCELLED", "User cancelled at auth dialog")
            _show_warning("Cancelled.")
            return

        log_state_transition(db, intent_id, "AWAITING_AUTH", "EXECUTING")
    else:
        log_state_transition(db, intent_id, "PARSING", "EXECUTING")

    # --- EXECUTING ---
    lifecycle.transition(IntentState.EXECUTING)
    failure_tracker = ToolFailureTracker()
    agent = FileAgent(intent_id=intent_id, db_conn=db)

    action_results = []
    for action in goal_spec.get("actions", []):
        if action.get("agent") != "file":
            _show_warning(f"Agent '{action.get('agent')}' not implemented in Phase 0. Skipping.")
            continue

        try:
            result = agent.execute_action(action)
            failure_tracker.reset(action.get("type", ""), action.get("params", {}))
            action_results.append((action, result, None))

            # Render results
            atype = action.get("type", "").upper()
            if atype in ("QUERY", "READ") and isinstance(result, dict) and "files" in result:
                _render_file_list(result)
            elif atype == "READ" and isinstance(result, dict) and "content" in result:
                console.print(Panel(
                    result["content"][:2000],
                    title=result.get("path", ""),
                    border_style=LEAVES_PRIMARY_COLOR,
                ))
            else:
                console.print(f"[{LEAVES_SUCCESS_COLOR}]Done:[/{LEAVES_SUCCESS_COLOR}] {result}")

        except LeavesError as e:
            action_results.append((action, None, e))
            try:
                failure_tracker.record_failure(
                    action.get("type", ""),
                    action.get("params", {}),
                    error=e,
                )
            except LeavesError as escalated:
                _show_error(escalated.user_message)
                lifecycle.transition(IntentState.FAILED)
                log_state_transition(db, intent_id, "EXECUTING", "FAILED")
                complete_intent(
                    db, intent_id, "FAILED", escalated.detail,
                    duration_ms=(time.monotonic() - t_start) * 1000,
                )
                return

            _show_error(e.user_message)
            # Continue with remaining actions

    # --- DONE ---
    lifecycle.transition(IntentState.DONE)
    log_state_transition(db, intent_id, "EXECUTING", "DONE")
    duration_ms = (time.monotonic() - t_start) * 1000
    complete_intent(
        db, intent_id, "DONE",
        f"Completed {len(action_results)} action(s)",
        duration_ms=duration_ms,
    )
    console.print(
        f"[{LEAVES_DIM_COLOR}]Completed in {duration_ms:.0f}ms[/{LEAVES_DIM_COLOR}]"
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
        state_styled = f"[{state_styles.get(state, 'white')}]{state}[/{state_styles.get(state, 'white')}]"
        dur = f"{row['duration_ms']:.0f}ms" if row["duration_ms"] else "—"
        ts = _fmt_time(row["created_at"])
        table.add_row(str(i), row["natural_text"][:50], row["category"] or "—", state_styled, dur, ts)

    console.print(table)


def cmd_detail(intent_id_prefix: str) -> None:
    db = _get_db()
    intents = get_recent_intents(db, limit=100)
    matches = [i for i in intents if i["intent_id"].startswith(intent_id_prefix)]

    if not matches:
        _show_error(f"No intent found with prefix '{intent_id_prefix}'")
        return

    intent = matches[0]
    console.print(Panel(
        f"[bold]Intent:[/bold] {intent['natural_text']}\n"
        f"[bold]ID:[/bold] {intent['intent_id']}\n"
        f"[bold]Category:[/bold] {intent['category']}\n"
        f"[bold]State:[/bold] {intent['state']}\n"
        f"[bold]Duration:[/bold] {intent['duration_ms']:.0f}ms" if intent["duration_ms"] else "—",
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
            status = f"[{LEAVES_ERROR_COLOR}]{a['error_code']}[/{LEAVES_ERROR_COLOR}]" if a["error_code"] else f"[{LEAVES_SUCCESS_COLOR}]OK[/{LEAVES_SUCCESS_COLOR}]"
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
    _banner()

    # Warm check
    parser = _get_parser()
    if not parser._client.is_available():
        console.print(
            f"[{LEAVES_WARNING_COLOR}]Inference server not running.[/{LEAVES_WARNING_COLOR}] "
            f"Start it with:  bash scripts/start-inference.sh"
        )
        console.print()

    while True:
        try:
            raw = input(f"[leaves] ").strip()
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
