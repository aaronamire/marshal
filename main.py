#!/usr/bin/env python3
"""
Marshal — Main CLI entry point.

Three-layer intent pipeline:
  Layer 0 (<0.1ms): regex matcher → unambiguous commands skip LLM entirely
  Layer 1 (3-8ms):  sklearn classifier → instant category feedback
  Layer 2 (26s+):   Llama GoalSpec generation → full execution plan

Orchestration delegated to agentd.AgentCoordinator.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import socket as _socket
import subprocess
import sys
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
from agents.refs import resolve_refs
from agents.session_memory import SessionMemory
from agents.state_machine import IntentLifecycle, IntentState
from config import (
    APP_NAME, APP_VERSION,
    MARSHAL_PRIMARY_COLOR, MARSHAL_SUCCESS_COLOR,
    MARSHAL_ERROR_COLOR, MARSHAL_WARNING_COLOR, MARSHAL_DIM_COLOR,
)
from db.audit import (
    get_db, log_intent_created, log_state_transition,
    complete_intent, log_error, get_recent_intents,
    get_intent_transitions, get_intent_actions,
)
from cortex.briefing import BriefingGenerator
from cortex.indexer import CortexIndexer
from cortex.adapters.filesystem import FilesystemAdapter
from db.intent_store import (
    store_persistent_intent, get_active_intents, get_intent,
    deactivate_intent, delete_intent,
)
from errors import MarshalError, MarshalErrorCode

_LANCE_PATH = pathlib.Path(__file__).parent / "rag" / ".lancedb"

# Verbose mode: dump GoalSpec JSON after every parse and include full
# MarshalError code + detail on failures. Set via --verbose CLI flag or
# MARSHAL_VERBOSE=1 env var (the env var is honored even when the REPL is
# launched with no argv, e.g. from a systemd unit).
_VERBOSE = os.environ.get("MARSHAL_VERBOSE", "").lower() in ("1", "true", "yes")

console = Console()

# ---------------------------------------------------------------------------
# agentd socket client
# ---------------------------------------------------------------------------

_AGENTD_SOCK = pathlib.Path.home() / ".marshal" / "agentd.sock"


def _ensure_agentd() -> bool:
    """Return True if agentd socket is available, spawning the daemon if needed."""
    if _AGENTD_SOCK.exists():
        return True
    # Spawn agentd as a background process.
    agentd_path = pathlib.Path(__file__).with_name("agentd.py")
    subprocess.Popen(
        [sys.executable, str(agentd_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait up to 2 seconds for the socket to appear.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if _AGENTD_SOCK.exists():
            return True
        time.sleep(0.05)
    return False


def _send_goalspec(
    goal_spec: dict,
    from_state: str,
    on_event=None,
) -> tuple[dict, str]:
    """
    Send a GoalSpec to agentd and return (results, summary).

    If `on_event` is provided, the request opts into the streaming protocol:
    agentd will write zero or more {"event": ...} frames to the socket as
    channel messages arrive from the runner, followed by a single
    {"final": true, ...} frame. `on_event` is called once per event frame.

    Raises MarshalError if the daemon reports an execution failure.
    Raises OSError / other exceptions on communication failure
    (caller falls back).
    """
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    s.settimeout(120.0)
    s.connect(str(_AGENTD_SOCK))
    try:
        request = {
            "goal_spec": goal_spec,
            "from_state": from_state,
        }
        if on_event is not None:
            request["stream_events"] = True
        s.sendall(json.dumps(request).encode() + b"\n")

        # Read newline-delimited JSON frames until we see one with final=true.
        buf = b""
        final_resp: dict | None = None
        while final_resp is None:
            # Pull at least one complete line.
            while b"\n" not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    raise OSError("agentd closed connection without final frame")
                buf += chunk
            line, buf = buf.split(b"\n", 1)
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            if frame.get("final"):
                final_resp = frame
                break
            event = frame.get("event")
            if event is not None and on_event is not None:
                try:
                    on_event(event)
                except Exception:
                    pass
    finally:
        s.close()

    assert final_resp is not None  # loop only exits when set
    if final_resp.get("ok"):
        return final_resp["results"], final_resp["summary"]

    # Daemon reported an execution failure — re-raise as MarshalError.
    code_str = final_resp.get("code", "INTERNAL_ERROR")
    try:
        code = MarshalErrorCode[code_str]
    except KeyError:
        code = MarshalErrorCode.INTERNAL_ERROR
    raise MarshalError(
        code, detail=final_resp.get("detail") or final_resp.get("error"))


def _send_cancel(intent_id: str) -> bool:
    """
    Open a fresh socket to agentd and send a cancel frame for `intent_id`.

    Used by the SIGINT handler during EXECUTING. Best-effort: any failure
    is silently swallowed because we are inside a signal handler context
    and the worst case is the user has to wait for the in-flight intent
    to finish naturally.
    """
    try:
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(str(_AGENTD_SOCK))
        try:
            s.sendall(json.dumps({"_cancel": intent_id}).encode() + b"\n")
            line = s.recv(4096)
            if not line:
                return False
            resp = json.loads(line.split(b"\n", 1)[0])
            return bool(resp.get("_cancelled"))
        finally:
            s.close()
    except (OSError, ValueError, json.JSONDecodeError):
        return False


# Module-level singletons — initialized once in main()
_db = None
_parser: IntentParser | None = None
_session: SessionMemory | None = None
# The intent currently being executed; SIGINT handler reads this so it
# knows what to ask agentd to cancel. Set just before _send_goalspec,
# cleared on the post-execution path.
_inflight_intent_id: str | None = None


def _get_db():
    global _db
    if _db is None:
        _db = get_db()
    return _db


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _banner() -> None:
    title = Text(f" {APP_NAME} ", style=f"bold {MARSHAL_PRIMARY_COLOR}")
    subtitle = Text(f"v{APP_VERSION}", style=MARSHAL_DIM_COLOR)
    console.print(Panel(
        f"{title}\n{subtitle}",
        border_style=MARSHAL_PRIMARY_COLOR,
        padding=(0, 2),
    ))
    console.print()


def _ensure_tier_and_show() -> None:
    """
    Print the active model tier, running a first-run hardware probe if none
    has been saved. Any error here is non-fatal — the REPL must still start.
    """
    try:
        import hardware
    except ImportError:
        return
    try:
        if hardware.active_tier_name() is None:
            profile = hardware.probe()
            decision = hardware.select_tier(profile, bench=None)
            hardware.write_decision(decision, hardware.CONFIG_PATH)
            console.print(
                f"[{MARSHAL_DIM_COLOR}]First-run hardware check: "
                f"chose '{decision.chosen}' tier for {profile.total_ram_gb:.0f}GB RAM "
                f"({'laptop' if profile.is_laptop else 'desktop'})"
                f"[/{MARSHAL_DIM_COLOR}]"
            )
            tier_name = decision.chosen
        else:
            tier_name = hardware.active_tier_name() or "standard"
        tier = hardware.TIER_BY_NAME.get(tier_name)
        if tier is not None:
            console.print(
                f"[{MARSHAL_DIM_COLOR}]Model tier: {tier.name} "
                f"({tier.param_billions:.1f}B, {tier.model_file})"
                f"[/{MARSHAL_DIM_COLOR}]"
            )
    except Exception as e:
        console.print(
            f"[{MARSHAL_DIM_COLOR}]Hardware probe skipped: {e}[/{MARSHAL_DIM_COLOR}]"
        )


def _show_error(msg: str) -> None:
    console.print(f"[{MARSHAL_ERROR_COLOR}]Error:[/{MARSHAL_ERROR_COLOR}] {msg}")


def _show_marshal_error(e: MarshalError) -> None:
    """Render a MarshalError with the user-facing message; in verbose mode
    also print the error code and the structured detail (which often
    contains the offending value, e.g. the raw model output that failed
    to parse, or the path that was outside the authorization scope)."""
    _show_error(e.user_message)
    if _VERBOSE:
        console.print(f"  [{MARSHAL_DIM_COLOR}]code:   {e.code.value}[/{MARSHAL_DIM_COLOR}]")
        if e.detail:
            console.print(f"  [{MARSHAL_DIM_COLOR}]detail: {e.detail}[/{MARSHAL_DIM_COLOR}]")
        if e.cause is not None:
            console.print(f"  [{MARSHAL_DIM_COLOR}]cause:  {type(e.cause).__name__}: {e.cause}[/{MARSHAL_DIM_COLOR}]")


def _dump_goal_spec(goal_spec: dict) -> None:
    """Pretty-print the full GoalSpec in verbose mode so power users can see
    exactly what the model produced and what the enforcer will check
    against."""
    if not _VERBOSE:
        return
    rendered = json.dumps(goal_spec, indent=2, sort_keys=False)
    console.print(f"[{MARSHAL_DIM_COLOR}]GoalSpec:[/{MARSHAL_DIM_COLOR}]")
    console.print(f"[{MARSHAL_DIM_COLOR}]{rendered}[/{MARSHAL_DIM_COLOR}]")


def _show_warning(msg: str) -> None:
    console.print(f"[{MARSHAL_WARNING_COLOR}]Warning:[/{MARSHAL_WARNING_COLOR}] {msg}")


def _render_file_list(result: dict) -> None:
    files = result.get("files", [])
    path = result.get("path", "?")
    count = result.get("count", 0)

    if not files:
        console.print(f"[{MARSHAL_DIM_COLOR}]No files found in {path}[/{MARSHAL_DIM_COLOR}]")
        return

    table = Table(
        title=f"{count} file(s) in {path}",
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style=f"bold {MARSHAL_PRIMARY_COLOR}",
    )
    table.add_column("Name", style="white")
    table.add_column("Size", justify="right", style=MARSHAL_DIM_COLOR)
    table.add_column("Modified", style=MARSHAL_DIM_COLOR)
    table.add_column("Type", style=MARSHAL_DIM_COLOR)

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


def _fmt_uptime(seconds: float) -> str:
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    mins = int((seconds % 3600) // 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{mins}m")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# System result renderers
# ---------------------------------------------------------------------------

def _render_cpu(result: dict) -> None:
    usage = result.get("usage_percent", 0)
    freq = result.get("freq_mhz")
    cores = result.get("cores")
    temp = result.get("temp_celsius")

    lines = [f"  Usage: [white]{usage}%[/white]"]
    if freq:
        lines.append(f"  Frequency: [white]{freq:.0f} MHz[/white]")
    if cores:
        lines.append(f"  Cores: [white]{cores}[/white]")
    if temp is not None:
        color = MARSHAL_SUCCESS_COLOR if temp < 70 else MARSHAL_WARNING_COLOR if temp < 85 else MARSHAL_ERROR_COLOR
        lines.append(f"  Temperature: [{color}]{temp}°C[/{color}]")
    console.print(Panel("\n".join(lines), title="CPU", border_style=MARSHAL_PRIMARY_COLOR))


def _render_memory(result: dict) -> None:
    pct = result.get("percent", 0)
    total = result.get("total_gb", 0)
    used = result.get("used_gb", 0)
    avail = result.get("available_gb", 0)

    color = MARSHAL_SUCCESS_COLOR if pct < 70 else MARSHAL_WARNING_COLOR if pct < 90 else MARSHAL_ERROR_COLOR
    console.print(Panel(
        f"  Used: [{color}]{used:.1f} GB / {total:.1f} GB ({pct}%)[/{color}]\n"
        f"  Available: [white]{avail:.1f} GB[/white]",
        title="Memory", border_style=MARSHAL_PRIMARY_COLOR,
    ))


def _render_disk(result: dict) -> None:
    pct = result.get("percent", 0)
    total = result.get("total_gb", 0)
    used = result.get("used_gb", 0)
    free = result.get("free_gb", 0)
    path = result.get("path", "/")

    color = MARSHAL_SUCCESS_COLOR if pct < 70 else MARSHAL_WARNING_COLOR if pct < 90 else MARSHAL_ERROR_COLOR
    console.print(Panel(
        f"  Used: [{color}]{used:.1f} GB / {total:.1f} GB ({pct}%)[/{color}]\n"
        f"  Free: [white]{free:.1f} GB[/white]",
        title=f"Disk ({path})", border_style=MARSHAL_PRIMARY_COLOR,
    ))


def _render_processes(result: dict) -> None:
    procs = result.get("processes", [])
    if not procs:
        console.print(f"[{MARSHAL_DIM_COLOR}]No processes found.[/{MARSHAL_DIM_COLOR}]")
        return

    table = Table(
        title=f"Top {len(procs)} Processes",
        box=box.SIMPLE_HEAD,
        header_style=f"bold {MARSHAL_PRIMARY_COLOR}",
    )
    table.add_column("PID", style=MARSHAL_DIM_COLOR, justify="right")
    table.add_column("Name", style="white")
    table.add_column("CPU%", justify="right")
    table.add_column("Memory", justify="right", style=MARSHAL_DIM_COLOR)
    table.add_column("Status", style=MARSHAL_DIM_COLOR)

    for p in procs:
        cpu = p.get("cpu_percent", 0)
        cpu_color = MARSHAL_ERROR_COLOR if cpu > 50 else MARSHAL_WARNING_COLOR if cpu > 20 else "white"
        table.add_row(
            str(p["pid"]),
            p["name"],
            f"[{cpu_color}]{cpu}[/{cpu_color}]",
            f"{p.get('memory_mb', 0):.0f} MB",
            p.get("status", ""),
        )
    console.print(table)


def _render_uptime(result: dict) -> None:
    secs = result.get("uptime_seconds", 0)
    boot = result.get("boot_time_iso", "")
    console.print(
        f"  [{MARSHAL_PRIMARY_COLOR}]Uptime:[/{MARSHAL_PRIMARY_COLOR}] "
        f"[white]{_fmt_uptime(secs)}[/white]  "
        f"[{MARSHAL_DIM_COLOR}](booted {boot})[/{MARSHAL_DIM_COLOR}]"
    )


def _render_system_result(result: dict) -> None:
    """Route system agent results to the appropriate renderer."""
    if "usage_percent" in result and "freq_mhz" in result:
        _render_cpu(result)
    elif "total_gb" in result and "available_gb" in result:
        _render_memory(result)
    elif "total_gb" in result and "free_gb" in result:
        _render_disk(result)
    elif "processes" in result:
        _render_processes(result)
    elif "uptime_seconds" in result:
        _render_uptime(result)
    elif "launched" in result:
        prog = result["launched"]
        pid = result.get("pid", "?")
        console.print(
            f"  [{MARSHAL_SUCCESS_COLOR}]Launched[/{MARSHAL_SUCCESS_COLOR}] "
            f"[white]{prog}[/white] [{MARSHAL_DIM_COLOR}](PID {pid})[/{MARSHAL_DIM_COLOR}]"
        )
    elif "terminated" in result:
        target = result.get("target", "?")
        count = result.get("count", 0)
        for info in result.get("terminated", []):
            sig = info.get("signal", "?")
            console.print(
                f"  [{MARSHAL_WARNING_COLOR}]Terminated[/{MARSHAL_WARNING_COLOR}] "
                f"[white]{info.get('name', target)}[/white] "
                f"[{MARSHAL_DIM_COLOR}](PID {info.get('pid', '?')}, {sig})[/{MARSHAL_DIM_COLOR}]"
            )
    else:
        console.print(f"  [{MARSHAL_SUCCESS_COLOR}]✓[/{MARSHAL_SUCCESS_COLOR}] {result}")


# ---------------------------------------------------------------------------
# Web result renderers
# ---------------------------------------------------------------------------

def _render_web_search(result: dict) -> None:
    query = result.get("query", "")
    results_list = result.get("results", [])
    err = result.get("error")

    if err:
        _show_error(f"Web search failed: {err}")
        return

    if not results_list:
        console.print(f"[{MARSHAL_DIM_COLOR}]No results for '{query}'[/{MARSHAL_DIM_COLOR}]")
        return

    console.print(f"  [{MARSHAL_PRIMARY_COLOR}]Search:[/{MARSHAL_PRIMARY_COLOR}] {query}")
    for i, r in enumerate(results_list, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        console.print(f"  [{MARSHAL_DIM_COLOR}]{i}.[/{MARSHAL_DIM_COLOR}] [bold white]{title}[/bold white]")
        if url:
            console.print(f"     [{MARSHAL_DIM_COLOR}]{url}[/{MARSHAL_DIM_COLOR}]")
        if snippet:
            console.print(f"     {snippet[:200]}")


def _render_web_fetch(result: dict) -> None:
    url = result.get("url", "")
    title = result.get("title", "")
    content = result.get("content", "")
    err = result.get("error")

    if err:
        _show_error(f"Fetch failed: {err}")
        return

    header = title or url
    console.print(Panel(
        content[:3000],
        title=header,
        border_style=MARSHAL_PRIMARY_COLOR,
    ))


def _render_web_result(result: dict) -> None:
    """Route web agent results to the appropriate renderer."""
    if "results" in result:
        _render_web_search(result)
    elif "content" in result and "url" in result:
        _render_web_fetch(result)
    elif "error" in result:
        _show_error(result["error"])
    else:
        console.print(f"  [{MARSHAL_SUCCESS_COLOR}]✓[/{MARSHAL_SUCCESS_COLOR}] {result}")


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
        f"[bold {MARSHAL_WARNING_COLOR}]Authorization Required[/bold {MARSHAL_WARNING_COLOR}]\n\n"
        f"Intent: [white]{goal_spec['natural_text']}[/white]\n"
        f"Resources: {', '.join(resources) or 'none'}\n"
        f"Reversible: {'yes' if reversible else '[red]NO[/red]'}\n\n"
        "Actions:\n" + "\n".join(
            f"  [{MARSHAL_DIM_COLOR}]{a['action_id']}[/{MARSHAL_DIM_COLOR}] "
            f"[yellow]{a['type']}[/yellow] ({a.get('agent', '?')})"
            for a in goal_spec.get("actions", [])
        ),
        border_style=MARSHAL_WARNING_COLOR,
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
    session = _session

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
        color = MARSHAL_PRIMARY_COLOR if result.is_confident else MARSHAL_WARNING_COLOR
        console.print(
            f"  [bold {color}]◆[/bold {color}] "
            f"{result.category} · {result.confidence:.0%} · {result.latency_ms:.0f}ms"
        )

    parser._on_classified = on_classified

    # --- PARSING (includes Layer 1 callback + Layer 2 LLM) ---
    lifecycle = IntentLifecycle(intent_id="pending")
    try:
        lifecycle.transition(IntentState.PARSING)
        history_block = session.to_prompt_block() if session is not None else None
        with console.status(f"[{MARSHAL_DIM_COLOR}]Generating plan…[/{MARSHAL_DIM_COLOR}]"):
            goal_spec = parser.parse(user_text, session_history=history_block)

        # Resolve $prev[N].action_id.path references in action params
        # against session memory BEFORE the spec marshal this process.
        # The runner subprocess sees fully-expanded params.
        if session is not None and len(session) > 0:
            for action in goal_spec.get("actions", []):
                if isinstance(action.get("params"), dict):
                    action["params"] = resolve_refs(action["params"], session)

        intent_id = goal_spec["intent_id"]
        lifecycle = IntentLifecycle(intent_id=intent_id)
        lifecycle.transition(IntentState.PARSING)

        log_intent_created(db, intent_id, user_text, goal_spec)
        log_state_transition(db, intent_id, "PENDING", "PARSING")

        confidence = goal_spec.get("metadata", {}).get("confidence", 0.0)
        latency = goal_spec.get("metadata", {}).get("parse_latency_ms", 0)
        meta = goal_spec.get("metadata", {})
        cached = meta.get("tokens_cached", 0)
        computed = meta.get("prompt_tokens_computed", 0)
        cache_tag = ""
        if cached or computed:
            total_prompt = cached + computed
            hit_pct = (cached / total_prompt * 100) if total_prompt else 0
            cache_tag = f" · cache {hit_pct:.0f}%"
        console.print(
            f"  [{MARSHAL_DIM_COLOR}]Plan: "
            f"{len(goal_spec.get('actions', []))} action(s) via {goal_spec['category']} "
            f"· L2 confidence {confidence:.0%} · {latency:.0f}ms"
            f"{cache_tag}[/{MARSHAL_DIM_COLOR}]"
        )
        _dump_goal_spec(goal_spec)

    except MarshalError as e:
        _show_marshal_error(e)
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

    # --- EXECUTING (via agentd socket; in-process fallback) ---
    from_state_str = lifecycle.state.value

    def _on_progress_event(event: dict) -> None:
        """Render a single channel event as a dim status line."""
        kind = event.get("kind", "?")
        aid = event.get("action_id", "?")
        data = event.get("data")
        if kind == "progress":
            label = ""
            if isinstance(data, dict):
                if "pct" in data:
                    label = f"{data['pct']}%"
                elif "message" in data:
                    label = str(data["message"])
                else:
                    label = ", ".join(f"{k}={v}" for k, v in data.items())
            console.print(
                f"[{MARSHAL_DIM_COLOR}]  ↳ {aid} {label}[/{MARSHAL_DIM_COLOR}]")
        elif kind == "partial":
            preview = ""
            if isinstance(data, dict):
                preview = data.get("path") or data.get("message") or ""
            elif isinstance(data, str):
                preview = data
            console.print(
                f"[{MARSHAL_DIM_COLOR}]  ⋯ {aid} {preview}[/{MARSHAL_DIM_COLOR}]")
        elif kind == "log":
            console.print(
                f"[{MARSHAL_DIM_COLOR}]  · {aid} {data}[/{MARSHAL_DIM_COLOR}]")

    # Install a SIGINT handler for the duration of the EXECUTING window.
    # Ctrl-C asks agentd to cancel the in-flight intent rather than tearing
    # down the REPL. The handler is removed in the finally block.
    global _inflight_intent_id
    _inflight_intent_id = intent_id
    _prev_sigint = signal.getsignal(signal.SIGINT)

    def _on_sigint(_signum, _frame):
        target = _inflight_intent_id
        if target:
            ok = _send_cancel(target)
            tag = "cancel sent" if ok else "cancel failed"
            console.print(
                f"[{MARSHAL_WARNING_COLOR}]  ⌃C — {tag}[/{MARSHAL_WARNING_COLOR}]"
            )

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):
        pass  # not on main thread / unsupported

    try:
        if _ensure_agentd():
            try:
                results, summary = _send_goalspec(
                    goal_spec, from_state_str, on_event=_on_progress_event)
                # Daemon advanced its own lifecycle to EXECUTING; mirror that here
                # so subsequent transitions (→DONE / →FAILED) remain valid.
                lifecycle.transition(IntentState.EXECUTING)
            except MarshalError:
                lifecycle.transition(IntentState.EXECUTING)
                raise
            except Exception:
                _show_warning("agentd socket error — running in-process.")
                results, summary = AgentCoordinator(db).execute(goal_spec, lifecycle)
        else:
            _show_warning("agentd unavailable — running in-process.")
            results, summary = AgentCoordinator(db).execute(goal_spec, lifecycle)
    except MarshalError as e:
        _show_marshal_error(e)
        log_state_transition(db, intent_id, "EXECUTING", "FAILED")
        complete_intent(
            db, intent_id, "FAILED", e.detail,
            duration_ms=(time.monotonic() - t_start) * 1000,
        )
        return
    finally:
        _inflight_intent_id = None
        try:
            signal.signal(signal.SIGINT, _prev_sigint)
        except (ValueError, OSError, TypeError):
            pass

    # --- Render results ---
    for action in goal_spec.get("actions", []):
        action_id = action.get("action_id", "unknown")
        result = results.get(action_id, {})
        agent = action.get("agent", "")

        if isinstance(result, dict) and "error" in result:
            _show_error(result["error"])
            continue

        if not isinstance(result, dict):
            if result:
                console.print(
                    f"  [{MARSHAL_SUCCESS_COLOR}]✓[/{MARSHAL_SUCCESS_COLOR}] {result}"
                )
            continue

        # Route to agent-specific renderer
        if agent == "system":
            _render_system_result(result)
        elif agent == "web":
            _render_web_result(result)
        elif "files" in result:
            _render_file_list(result)
        elif "content" in result and "path" in result:
            console.print(Panel(
                result["content"][:2000],
                title=result.get("path", ""),
                border_style=MARSHAL_PRIMARY_COLOR,
            ))
        elif result:
            console.print(
                f"  [{MARSHAL_SUCCESS_COLOR}]✓[/{MARSHAL_SUCCESS_COLOR}] {result}"
            )

    # --- DONE ---
    lifecycle.transition(IntentState.DONE)
    log_state_transition(db, intent_id, "EXECUTING", "DONE")
    duration_ms = (time.monotonic() - t_start) * 1000
    complete_intent(db, intent_id, "DONE", summary, duration_ms=duration_ms)

    # Record the completed turn so future intents can $prev-reference it.
    if session is not None:
        try:
            session.record(
                intent_id=intent_id,
                natural_text=user_text,
                goal_spec=goal_spec,
                results=results,
                summary=summary,
            )
        except Exception:
            pass  # Memory is best-effort; never block the user

    reversible = auth.get("reversible", True)
    rev_str = "reversible" if reversible else "[red]irreversible[/red]"
    console.print(
        f"  [{MARSHAL_DIM_COLOR}]Done in {duration_ms:.0f}ms · {rev_str} · logged[/{MARSHAL_DIM_COLOR}]"
    )


# ---------------------------------------------------------------------------
# Built-in commands
# ---------------------------------------------------------------------------

def cmd_history() -> None:
    db = _get_db()
    intents = get_recent_intents(db, limit=20)
    if not intents:
        console.print(f"[{MARSHAL_DIM_COLOR}]No history yet.[/{MARSHAL_DIM_COLOR}]")
        return

    table = Table(
        title="Recent Intents",
        box=box.SIMPLE_HEAD,
        header_style=f"bold {MARSHAL_PRIMARY_COLOR}",
    )
    table.add_column("#", style=MARSHAL_DIM_COLOR, justify="right")
    table.add_column("Intent", style="white", max_width=50)
    table.add_column("Category", style=MARSHAL_DIM_COLOR)
    table.add_column("State", style="white")
    table.add_column("Duration", style=MARSHAL_DIM_COLOR, justify="right")
    table.add_column("Time", style=MARSHAL_DIM_COLOR)

    state_styles = {
        "DONE": MARSHAL_SUCCESS_COLOR,
        "FAILED": MARSHAL_ERROR_COLOR,
        "CANCELLED": MARSHAL_WARNING_COLOR,
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
        border_style=MARSHAL_PRIMARY_COLOR,
    ))

    transitions = get_intent_transitions(db, intent["intent_id"])
    if transitions:
        t_table = Table(title="State Transitions", box=box.SIMPLE_HEAD)
        t_table.add_column("From", style=MARSHAL_DIM_COLOR)
        t_table.add_column("To", style="white")
        t_table.add_column("At", style=MARSHAL_DIM_COLOR)
        for t in transitions:
            t_table.add_row(t["from_state"], t["to_state"], _fmt_time(t["transitioned_at"]))
        console.print(t_table)

    actions = get_intent_actions(db, intent["intent_id"])
    if actions:
        a_table = Table(title="Actions", box=box.SIMPLE_HEAD)
        a_table.add_column("ID", style=MARSHAL_DIM_COLOR)
        a_table.add_column("Type", style="white")
        a_table.add_column("Agent", style=MARSHAL_DIM_COLOR)
        a_table.add_column("Status", style="white")
        for a in actions:
            if a["error_code"]:
                status = f"[{MARSHAL_ERROR_COLOR}]{a['error_code']}[/{MARSHAL_ERROR_COLOR}]"
            else:
                status = f"[{MARSHAL_SUCCESS_COLOR}]OK[/{MARSHAL_SUCCESS_COLOR}]"
            a_table.add_row(a["action_id"], a["action_type"], a["agent"], status)
        console.print(a_table)


def cmd_help() -> None:
    console.print(Panel(
        "[bold]Commands:[/bold]\n\n"
        "  [white]<natural language>[/white]        — execute an intent\n"
        "  [white]history[/white]                   — show recent intents\n"
        "  [white]detail <id-prefix>[/white]        — show intent audit trail\n"
        "  [white]search <query>[/white]             — semantic search over indexed files\n"
        "  [white]briefing[/white]                  — show what changed recently\n"
        "  [white]watch <text> on <path>[/white]    — create a filesystem watcher\n"
        "  [white]watchers[/white]                  — list active watchers\n"
        "  [white]unwatch <id-prefix>[/white]       — deactivate a watcher\n"
        "  [white]model[/white]                     — show the active model tier\n"
        "  [white]model detect[/white]              — re-probe hardware and re-select tier\n"
        "  [white]model <tier>[/white]              — force a tier (tiny|standard|pro|max)\n"
        "  [white]model download <tier>[/white]    — download the GGUF for a tier\n"
        "  [white]switch <tier>[/white]             — alias for 'model <tier>' (also: fast|best)\n"
        "  [white]inference on / off / status[/white] — control the local LLM server\n"
        "  [white]marshal-exit[/white]              — exit Marshal entirely (terminal-only)\n"
        "  [white]help[/white]                      — show this help\n"
        "  [white]quit[/white] / [white]exit[/white]                — exit\n\n"
        "[bold]Examples:[/bold]\n"
        "  find all PDFs in my Downloads folder\n"
        "  how much disk space do I have left\n"
        "  open firefox\n"
        "  watch organize PDFs on ~/Downloads\n"
        "  search the web for Python tutorials\n"
        "  move ~/Desktop/screenshot.png to ~/Pictures",
        title="Marshal Help",
        border_style=MARSHAL_PRIMARY_COLOR,
    ))


# ---------------------------------------------------------------------------
# Persistent intent commands
# ---------------------------------------------------------------------------

def _notify_watcher_reload() -> None:
    """Tell agentd to reload filesystem watches. Best-effort."""
    try:
        s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        s.settimeout(3.0)
        s.connect(str(_AGENTD_SOCK))
        s.sendall(json.dumps({"_reload_watcher": True}).encode() + b"\n")
        s.recv(1024)
        s.close()
    except Exception:
        pass


def cmd_watch(raw: str) -> None:
    """
    Parse 'watch <intent text> on <path>' and create a filesystem-triggered
    persistent intent.
    """
    db = _get_db()
    parser = _parser

    # Split on ' on ' (last occurrence to handle intents containing 'on')
    parts = raw.rsplit(" on ", 1)
    if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
        _show_error("Usage: watch <intent text> on <path>")
        console.print(f"  [{MARSHAL_DIM_COLOR}]Example: watch organize PDFs on ~/Downloads[/{MARSHAL_DIM_COLOR}]")
        return

    intent_text = parts[0].strip()
    watch_path = parts[1].strip()

    # Expand ~ and validate path
    expanded = pathlib.Path(watch_path).expanduser().resolve()
    if not expanded.exists():
        _show_error(f"Path does not exist: {expanded}")
        return
    if not expanded.is_dir():
        _show_error(f"Path is not a directory: {expanded}")
        return

    # Parse the intent to get a GoalSpec
    if not parser._client.is_available():
        _show_error("Inference server not running. Start: bash scripts/start-inference.sh")
        return

    try:
        with console.status(f"[{MARSHAL_DIM_COLOR}]Parsing intent…[/{MARSHAL_DIM_COLOR}]"):
            goal_spec = parser.parse(intent_text)
    except MarshalError as e:
        _show_marshal_error(e)
        return

    # Store as persistent intent
    trigger_config = {
        "path": str(expanded),
        "events": ["created"],
    }
    name = intent_text[:60]

    try:
        intent_id = store_persistent_intent(
            db, name=name, goalspec=goal_spec,
            trigger_type="filesystem", trigger_config=trigger_config,
        )
    except MarshalError as e:
        _show_marshal_error(e)
        return

    # Notify agentd to reload watches
    _notify_watcher_reload()

    console.print(
        f"  [{MARSHAL_SUCCESS_COLOR}]✓[/{MARSHAL_SUCCESS_COLOR}] "
        f"Watcher created: [white]{name}[/white]"
    )
    console.print(
        f"  [{MARSHAL_DIM_COLOR}]Watching: {expanded}[/{MARSHAL_DIM_COLOR}]"
    )
    console.print(
        f"  [{MARSHAL_DIM_COLOR}]ID: {intent_id[:8]}…[/{MARSHAL_DIM_COLOR}]"
    )


def cmd_watchers() -> None:
    """List all active persistent intents."""
    db = _get_db()
    intents = get_active_intents(db)

    if not intents:
        console.print(f"[{MARSHAL_DIM_COLOR}]No active watchers.[/{MARSHAL_DIM_COLOR}]")
        return

    table = Table(
        title="Active Watchers",
        box=box.SIMPLE_HEAD,
        header_style=f"bold {MARSHAL_PRIMARY_COLOR}",
    )
    table.add_column("ID", style=MARSHAL_DIM_COLOR)
    table.add_column("Name", style="white")
    table.add_column("Trigger", style=MARSHAL_DIM_COLOR)
    table.add_column("Path", style=MARSHAL_DIM_COLOR)
    table.add_column("Fires", justify="right", style=MARSHAL_DIM_COLOR)
    table.add_column("Last Fired", style=MARSHAL_DIM_COLOR)

    for i in intents:
        trigger_conf = i.get("trigger_config") or {}
        path = trigger_conf.get("path", "—")
        last = i.get("last_fired") or "never"
        if last != "never":
            last = last[:16]  # trim to datetime
        table.add_row(
            i["id"][:8] + "…",
            i["name"][:40],
            i["trigger_type"],
            str(path),
            str(i["fire_count"]),
            last,
        )

    console.print(table)


def cmd_unwatch(id_prefix: str) -> None:
    """Deactivate a persistent intent by ID prefix."""
    db = _get_db()
    intents = get_active_intents(db)
    matches = [i for i in intents if i["id"].startswith(id_prefix)]

    if not matches:
        _show_error(f"No active watcher found with prefix '{id_prefix}'")
        return

    if len(matches) > 1:
        _show_warning(f"Multiple matches for '{id_prefix}' — provide more characters.")
        return

    intent = matches[0]
    try:
        deactivate_intent(db, intent["id"])
    except MarshalError as e:
        _show_marshal_error(e)
        return

    console.print(
        f"  [{MARSHAL_WARNING_COLOR}]Unwatched:[/{MARSHAL_WARNING_COLOR}] "
        f"[white]{intent['name']}[/white]"
    )


# ---------------------------------------------------------------------------
# Cortex search
# ---------------------------------------------------------------------------

_indexer: CortexIndexer | None = None


def _get_indexer() -> CortexIndexer:
    global _indexer
    if _indexer is None:
        _indexer = CortexIndexer(_get_db(), _LANCE_PATH)
        _indexer.register(FilesystemAdapter())
    return _indexer


def cmd_search(query: str) -> None:
    """Semantic search over indexed files."""
    if not query:
        _show_error("Usage: search <query>")
        return

    indexer = _get_indexer()
    with console.status(f"[{MARSHAL_DIM_COLOR}]Searching…[/{MARSHAL_DIM_COLOR}]"):
        results = indexer.search(query, top_k=10)

    if not results:
        console.print(f"[{MARSHAL_DIM_COLOR}]No results for '{query}'[/{MARSHAL_DIM_COLOR}]")
        return

    table = Table(
        title=f"Search: {query}",
        box=box.SIMPLE_HEAD,
        header_style=f"bold {MARSHAL_PRIMARY_COLOR}",
    )
    table.add_column("#", style=MARSHAL_DIM_COLOR, justify="right")
    table.add_column("Title", style="white", max_width=40)
    table.add_column("Type", style=MARSHAL_DIM_COLOR)
    table.add_column("Score", justify="right", style=MARSHAL_DIM_COLOR)
    table.add_column("Path", style=MARSHAL_DIM_COLOR, max_width=50)

    for i, r in enumerate(results, 1):
        dist = r.get("_distance", 0)
        score = f"{1 - dist:.2f}" if dist else "—"
        # source_id is the file path for filesystem items
        path = r.get("source_type", "")
        title = r.get("title", "—")
        content_preview = r.get("content", "")[:80].replace("\n", " ")
        table.add_row(
            str(i), title, r.get("source_type", "—"), score, content_preview
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Briefing
# ---------------------------------------------------------------------------

# Patterns that trigger briefing from natural language (subset of L0 rules)
import re as _re
_BRIEFING_PATTERNS = [
    _re.compile(r'^\s*(?:good\s+)?morning\s*$', _re.IGNORECASE),
    _re.compile(r'^\s*brief(?:ing)?\s+me\s*$', _re.IGNORECASE),
    _re.compile(r'^\s*what(?:\'s|\s+has)\s+changed\b', _re.IGNORECASE),
    _re.compile(r'^\s*what\s+happened\b', _re.IGNORECASE),
    _re.compile(r'^\s*what(?:\'s|\s+is)\s+new\b', _re.IGNORECASE),
    _re.compile(r'^\s*catch\s+me\s+up\b', _re.IGNORECASE),
]


def _is_briefing_trigger(text: str) -> bool:
    return any(p.match(text) for p in _BRIEFING_PATTERNS)


def cmd_briefing(hours: int = 12) -> None:
    """Generate and display a briefing of recent changes."""
    indexer = _get_indexer()
    bg = BriefingGenerator(indexer._kg)

    with console.status(f"[{MARSHAL_DIM_COLOR}]Generating briefing…[/{MARSHAL_DIM_COLOR}]"):
        briefing = bg.generate(hours=hours)

    if briefing["empty"]:
        console.print(
            f"  [{MARSHAL_DIM_COLOR}]{briefing['headline']}[/{MARSHAL_DIM_COLOR}]"
        )
        return

    # Headline
    console.print(
        f"  [{MARSHAL_PRIMARY_COLOR}]{briefing['headline']}[/{MARSHAL_PRIMARY_COLOR}]"
    )
    console.print()

    # Sections
    for section in briefing["sections"]:
        src = section["source_type"]
        count = section["count"]
        console.print(
            f"  [bold white]{src}[/bold white] "
            f"[{MARSHAL_DIM_COLOR}]({count} changed)[/{MARSHAL_DIM_COLOR}]"
        )

        for group in section["groups"]:
            directory = group["directory"]
            g_count = group["count"]
            console.print(
                f"    [{MARSHAL_DIM_COLOR}]📁[/{MARSHAL_DIM_COLOR}] "
                f"[white]{directory}[/white] "
                f"[{MARSHAL_DIM_COLOR}]({g_count})[/{MARSHAL_DIM_COLOR}]"
            )
            for item in group["items"]:
                ext = item.get("extension", "")
                console.print(
                    f"      [{MARSHAL_DIM_COLOR}]·[/{MARSHAL_DIM_COLOR}] "
                    f"{item['title']}"
                    f"[{MARSHAL_DIM_COLOR}]{ext}[/{MARSHAL_DIM_COLOR}]"
                    if not ext or ext in item['title'] else
                    f"      [{MARSHAL_DIM_COLOR}]·[/{MARSHAL_DIM_COLOR}] "
                    f"{item['title']}"
                )
        console.print()


# ---------------------------------------------------------------------------
# Model tier commands
# ---------------------------------------------------------------------------

def cmd_model(arg: str) -> None:
    """
    Dispatch 'model ...' sub-commands.

    Empty:           show the active tier and whether its GGUF is on disk.
    'detect':        re-probe hardware, re-select, persist, print.
    'download <t>':  fetch the GGUF for <t> via huggingface-cli (with consent).
    '<tier>':        force a tier by name.
    """
    try:
        import hardware
    except ImportError:
        _show_error("hardware module unavailable")
        return

    parts = arg.split(maxsplit=1)
    sub = parts[0] if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    if sub == "":
        _cmd_model_show(hardware)
    elif sub == "detect":
        _cmd_model_detect(hardware)
    elif sub == "download":
        if not rest:
            _show_error("Usage: model download <tier>")
            return
        _cmd_model_download(hardware, rest.strip())
    elif sub in hardware.TIER_BY_NAME:
        _cmd_model_force(hardware, sub)
    else:
        _show_error(
            f"Unknown model sub-command '{sub}'. "
            f"Try: model | model detect | model <tier> | model download <tier>"
        )


def _cmd_model_show(hardware) -> None:
    tier_name = hardware.active_tier_name() or "standard"
    tier = hardware.TIER_BY_NAME.get(tier_name)
    if tier is None:
        _show_error(f"saved tier '{tier_name}' is not in the catalog — run 'model detect'")
        return

    # Preferred file vs. what the resolver would actually serve. If the two
    # differ, the user is running a degraded tier and should know.
    preferred_path = hardware.DEFAULT_MODELS_DIR / tier.model_file
    resolved = hardware.resolve_model_path()
    preferred_installed = preferred_path.exists()

    if preferred_installed:
        status = (
            f"[{MARSHAL_SUCCESS_COLOR}]installed[/{MARSHAL_SUCCESS_COLOR}] "
            f"at {preferred_path}"
        )
    elif resolved is not None:
        status = (
            f"[{MARSHAL_WARNING_COLOR}]preferred model not downloaded — "
            f"serving {resolved.name} instead[/{MARSHAL_WARNING_COLOR}] "
            f"(run 'model download {tier.name}' to upgrade)"
        )
    else:
        status = (
            f"[{MARSHAL_ERROR_COLOR}]no model installed[/{MARSHAL_ERROR_COLOR}] — "
            f"run 'model download {tier.name}'"
        )

    console.print(
        f"Preferred tier : [{MARSHAL_PRIMARY_COLOR}]{tier.name}[/{MARSHAL_PRIMARY_COLOR}] "
        f"({tier.param_billions:.1f}B)\n"
        f"Preferred file : {tier.model_file}\n"
        f"Status         : {status}\n"
        f"Description    : {tier.description}"
    )


def _cmd_model_detect(hardware) -> None:
    profile = hardware.probe()
    bench = hardware.calibrate()  # None if server unreachable; that's fine
    decision = hardware.select_tier(profile, bench)
    hardware.write_decision(decision, hardware.CONFIG_PATH)
    console.print(
        f"Re-selected [{MARSHAL_PRIMARY_COLOR}]{decision.chosen}[/{MARSHAL_PRIMARY_COLOR}]: "
        f"{decision.reason}"
    )


def _cmd_model_force(hardware, tier_name: str) -> None:
    """Save a forced tier choice (bypassing auto-detection)."""
    profile = hardware.probe()
    forced = hardware.TIER_BY_NAME[tier_name]
    decision = hardware.TierDecision(
        chosen=forced.name,
        reason=f"forced from REPL: 'model {tier_name}'",
        profile=profile,
        bench=None,
        estimated_gen_tok_s={},
        disqualified={},
        timestamp=hardware._utc_timestamp(),
    )
    hardware.write_decision(decision, hardware.CONFIG_PATH)
    console.print(
        f"Tier set to [{MARSHAL_PRIMARY_COLOR}]{tier_name}[/{MARSHAL_PRIMARY_COLOR}] "
        f"({forced.param_billions:.1f}B, {forced.model_file}).\n"
        f"[{MARSHAL_DIM_COLOR}]Restart the inference server to load the new model: "
        f"bash scripts/start-inference.sh[/{MARSHAL_DIM_COLOR}]"
    )


def _cmd_model_download(hardware, tier_name: str) -> None:
    tier = hardware.TIER_BY_NAME.get(tier_name)
    if tier is None:
        _show_error(f"unknown tier '{tier_name}' (choices: "
                    f"{', '.join(t.name for t in hardware.TIERS)})")
        return

    def consent(t) -> bool:
        try:
            ans = input(
                f"Download {t.hf_filename} from {t.hf_repo} "
                f"(~{t.model_size_gb:.1f}GB)? [y/N] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return False
        return ans in ("y", "yes")

    def progress(msg: str) -> None:
        console.print(f"[{MARSHAL_DIM_COLOR}]{msg}[/{MARSHAL_DIM_COLOR}]")

    try:
        path = hardware.ensure_tier_model(
            tier, consent_fn=consent, on_progress=progress,
        )
    except hardware.ModelDownloadError as e:
        _show_error(str(e))
        return

    if path is None:
        console.print(f"[{MARSHAL_DIM_COLOR}]download declined[/{MARSHAL_DIM_COLOR}]")
        return
    console.print(
        f"[{MARSHAL_SUCCESS_COLOR}]✓[/{MARSHAL_SUCCESS_COLOR}] {path}\n"
        f"[{MARSHAL_DIM_COLOR}]Restart the inference server to load it: "
        f"bash scripts/start-inference.sh[/{MARSHAL_DIM_COLOR}]"
    )


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def repl() -> None:
    global _parser, _session

    _banner()
    _ensure_tier_and_show()

    # Session memory: persists turns across REPL invocations within a
    # 1h window so the model can $prev-reference prior results.
    _session = SessionMemory()
    if len(_session) > 0:
        console.print(
            f"[{MARSHAL_DIM_COLOR}]Session: resumed {len(_session)} prior turn(s)[/{MARSHAL_DIM_COLOR}]"
        )

    # Try to load Layer 1 classifier
    classifier = None
    try:
        from agents.classifier import IntentClassifier
        classifier = IntentClassifier()
        console.print(f"[{MARSHAL_DIM_COLOR}]Layer 1 classifier: loaded (1.8ms avg)[/{MARSHAL_DIM_COLOR}]")
    except FileNotFoundError:
        console.print(
            f"[{MARSHAL_WARNING_COLOR}]Layer 1 classifier not found.[/{MARSHAL_WARNING_COLOR}] "
            f"[{MARSHAL_DIM_COLOR}]Run: python3 scripts/train_classifier.py[/{MARSHAL_DIM_COLOR}]"
        )
    except Exception as e:
        console.print(f"[{MARSHAL_DIM_COLOR}]Layer 1 unavailable: {e}[/{MARSHAL_DIM_COLOR}]")

    _parser = IntentParser(classifier=classifier)

    if not _parser._client.is_available():
        console.print(
            f"[{MARSHAL_WARNING_COLOR}]Inference server not running.[/{MARSHAL_WARNING_COLOR}] "
            f"Start: bash scripts/start-inference.sh"
        )
    console.print()

    while True:
        try:
            raw = input("[marshal] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break

        if not raw:
            continue

        lower = raw.lower()
        if lower in ("quit", "exit", "q"):
            console.print("Goodbye.")
            break
        elif lower in ("help", "?", "commands"):
            cmd_help()
        elif lower in ("history", "hist"):
            cmd_history()
        elif lower.startswith("detail "):
            cmd_detail(raw[7:].strip())
        elif lower.startswith("watch ") and " on " in lower:
            cmd_watch(raw[6:].strip())
        elif lower in ("watchers", "watches"):
            cmd_watchers()
        elif lower.startswith("unwatch "):
            cmd_unwatch(raw[8:].strip())
        elif lower.startswith("search "):
            cmd_search(raw[7:].strip())
        elif lower == "model":
            cmd_model("")
        elif lower.startswith("model "):
            cmd_model(raw[6:].strip())
        elif lower.startswith("switch "):
            # Alias: 'switch <tier>' === 'model <tier>'. Same handler — same
            # tier.json write — but exposed under the verb users actually
            # reach for ("switch to fast" reads more naturally than
            # "model fast").
            cmd_model(raw[7:].strip())
        elif lower in ("briefing", "morning"):
            cmd_briefing()
        elif lower == "verbose":
            global _VERBOSE
            _VERBOSE = not _VERBOSE
            console.print(f"  [{MARSHAL_DIM_COLOR}]verbose mode: {'on' if _VERBOSE else 'off'}[/{MARSHAL_DIM_COLOR}]")
        elif _is_briefing_trigger(raw):
            cmd_briefing()
        else:
            handle_intent(raw)

        console.print()


def main() -> None:
    """CLI entry. Parses args, sets _VERBOSE, then runs the REPL."""
    global _VERBOSE
    p = argparse.ArgumentParser(
        prog="marshal",
        description="Marshal — local AI agents that can't escape their plan.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Dump full GoalSpec after each parse and show error code/detail on failures. "
             "Also enabled by MARSHAL_VERBOSE=1.",
    )
    p.add_argument(
        "--version", action="version",
        version=f"{APP_NAME} {APP_VERSION}",
    )
    args = p.parse_args()
    if args.verbose:
        _VERBOSE = True
    repl()


if __name__ == "__main__":
    main()
