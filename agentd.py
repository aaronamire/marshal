"""
Leaves OS Agent Daemon.

Phase 1: AgentCoordinator as a library module (imported by leaves.py).
Phase 2: Unix socket daemon — run `python3 agentd.py` to start.
         leaves.py connects via ~/.leaves/agentd.sock (newline-delimited JSON).
Phase 3: cgroup integration and per-intent subprocess isolation.

Public interface (callers use ONLY this):
    coordinator = AgentCoordinator(db)
    results, summary = coordinator.execute(goal_spec, lifecycle)
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import signal
import sys
from typing import Any

from agents.file_agent import FileAgent
from agents.system_agent import SystemAgent
from agents.web_agent import WebAgent
from agents.audio_agent import AudioAgent
from agents.network_agent import NetworkAgent
from agents.power_agent import PowerAgent
from agents.writing_agent import WritingAgent
from agents.state_machine import IntentLifecycle, IntentState
from agents.tool_failure_tracker import ToolFailureTracker
from db.audit import get_db, log_error, log_intent_created, log_state_transition, complete_intent
from errors import LeavesError, LeavesErrorCode

# Map agent type strings -> agent classes
_AGENT_MAP: dict[str, type] = {
    "file": FileAgent,
    "system": SystemAgent,
    "web": WebAgent,
    "audio": AudioAgent,
    "network": NetworkAgent,
    "power": PowerAgent,
    "writing": WritingAgent,
}


class AgentCoordinator:
    """
    Coordinates agent execution for a single intent.

    Phase 1: sequential execution only.
    Phase 2: parallel execution via depends_on DAG.

    One instance per intent. Do not reuse across intents.
    """

    def __init__(self, db):
        self._db = db

    def execute(
        self,
        goal_spec: dict[str, Any],
        lifecycle: IntentLifecycle,
    ) -> tuple[dict[str, Any], str]:
        """
        Execute a validated GoalSpec.

        Returns (results_by_action_id, summary_string).
        Transitions lifecycle to EXECUTING internally.
        Raises LeavesError only on unrecoverable failure (tracker escalation).
        Per-action errors are captured in results and execution continues.
        """
        intent_id = goal_spec["intent_id"]
        actions = goal_spec.get("actions", [])
        tracker = ToolFailureTracker()

        from_state = lifecycle.state.value
        lifecycle.transition(IntentState.EXECUTING)
        log_state_transition(self._db, intent_id, from_state, "EXECUTING")

        results: dict[str, Any] = {}
        failed_ids: list[str] = []

        # Inject compositor WAYLAND_DISPLAY into each action so agents can
        # read it directly without relying on os.environ (belt-and-suspenders).
        _wl_meta = goal_spec.get("metadata", {}).get("wayland_display")

        for action in actions:
            if _wl_meta:
                action["_wayland_display"] = _wl_meta
            action_id = action.get("action_id", "unknown")
            agent_type = action.get("agent", "")

            # Agent availability check
            if agent_type not in _AGENT_MAP:
                err = LeavesError(
                    LeavesErrorCode.AGENT_NOT_AVAILABLE,
                    detail=(
                        f"Available agents: {list(_AGENT_MAP.keys())}. "
                        f"Got: '{agent_type}'"
                    ),
                )
                log_error(self._db, err.code.value, err.detail, intent_id)
                results[action_id] = {"error": err.user_message, "skipped": True}
                failed_ids.append(action_id)
                if action.get("on_failure", "abort") == "abort":
                    break
                continue

            # Dependency check
            unmet = [d for d in action.get("depends_on", []) if d in failed_ids]
            if unmet:
                err = LeavesError(
                    LeavesErrorCode.DEPENDENCY_FAILED,
                    detail=f"Action {action_id} skipped: dependency {unmet} failed",
                )
                log_error(self._db, err.code.value, err.detail, intent_id)
                results[action_id] = {"error": err.user_message, "skipped": True}
                failed_ids.append(action_id)
                continue

            # Execute
            agent = _AGENT_MAP[agent_type](intent_id=intent_id, db_conn=self._db)
            try:
                result = agent.execute_action(action)
                tracker.reset(action.get("type", ""), action.get("params", {}))
                results[action_id] = result
            except LeavesError as e:
                # Record failure — may escalate if same tool+args keeps failing
                try:
                    tracker.record_failure(
                        action.get("type", ""),
                        action.get("params", {}),
                        error=e,
                    )
                except LeavesError as escalated:
                    # Livelock detected — abort the entire intent
                    log_error(self._db, escalated.code.value, escalated.detail, intent_id)
                    results[action_id] = {"error": escalated.user_message}
                    failed_ids.append(action_id)
                    raise escalated  # bubble up to handle_intent

                log_error(self._db, e.code.value, e.detail, intent_id)
                results[action_id] = {"error": e.user_message}
                failed_ids.append(action_id)
                if action.get("on_failure", "abort") == "abort":
                    break

        succeeded = len(actions) - len(failed_ids)
        total_files = sum(
            r.get("count", 0) for r in results.values()
            if isinstance(r, dict) and "count" in r
        )

        if not failed_ids:
            summary = f"Completed {len(actions)} action(s)."
            if total_files:
                summary += f" Found {total_files} file(s)."
        else:
            summary = (
                f"{succeeded}/{len(actions)} actions succeeded, "
                f"{len(failed_ids)} failed."
            )

        return results, summary


# ---------------------------------------------------------------------------
# Unix socket daemon  (only active when run as __main__)
# ---------------------------------------------------------------------------

_SOCK_PATH = pathlib.Path.home() / ".leaves" / "agentd.sock"

# cgroup v2 resource limits — initialized in _run_server()
_CGROUP_ROOT = pathlib.Path("/sys/fs/cgroup/leaves")
_CGROUP_AVAILABLE = False


def _is_app_launch(goal_spec: dict) -> bool:
    """True when the goal only launches apps (system WRITE actions)."""
    actions = goal_spec.get("actions", [])
    return bool(actions) and all(
        a.get("agent") == "system" and a.get("type", "").upper() == "WRITE"
        for a in actions
    )


async def _run_sandboxed(goal_spec: dict, from_state: str) -> tuple[dict, str]:
    """
    Spawn agents/sandboxed_runner.py as a fresh subprocess.

    The subprocess applies Landlock to itself before importing project code,
    restricting its FS access to the intent's authorized resources.
    Communication is JSON over stdin/stdout.

    If cgroup v2 is available, the subprocess is placed in a per-intent
    cgroup with memory, CPU, and PID limits.

    App-launch intents (system WRITE) skip cgroup entirely — the launched
    GUI app must outlive the runner and needs unrestricted resources.
    """
    runner = pathlib.Path(__file__).parent / "agents" / "sandboxed_runner.py"
    payload = json.dumps({"goal_spec": goal_spec,
                          "from_state": from_state}).encode() + b"\n"
    intent_id = goal_spec.get("intent_id", "unknown")[:8]
    cgroup_path = _CGROUP_ROOT / f"intent-{intent_id}"
    cgroup_applied = False
    is_launch = _is_app_launch(goal_spec)

    # Build subprocess env: inject WAYLAND_DISPLAY from goal_spec metadata
    # so the runner (and any app it launches) starts with the CORRECT socket
    # even before the runner's own os.environ override kicks in.
    sub_env = os.environ.copy()
    wl_disp = goal_spec.get("metadata", {}).get("wayland_display")
    if wl_disp:
        sub_env["WAYLAND_DISPLAY"] = wl_disp

    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(runner),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=sub_env,
    )

    # Skip cgroup for app launches — launched apps need unrestricted
    # resources and must not be killed when the runner exits.
    if _CGROUP_AVAILABLE and not is_launch:
        try:
            cgroup_path.mkdir(parents=False, exist_ok=True)
            (cgroup_path / "memory.max").write_text("536870912")    # 512MB
            (cgroup_path / "memory.swap.max").write_text("0")       # no swap
            (cgroup_path / "cpu.max").write_text("50000 100000")    # 50% 1 core
            (cgroup_path / "pids.max").write_text("32")             # no fork bombs
            (cgroup_path / "cgroup.procs").write_text(str(proc.pid))
            cgroup_applied = True
        except (PermissionError, FileNotFoundError, OSError) as e:
            print(f"agentd: cgroup setup failed: {e}", flush=True)

    try:
        stdout, _stderr = await asyncio.wait_for(
            proc.communicate(payload), timeout=120.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise LeavesError(LeavesErrorCode.INFERENCE_TIMEOUT,
                          detail="sandboxed runner timed out")
    finally:
        # cgroup cleanup — must happen AFTER process exits
        if cgroup_applied and cgroup_path.exists():
            try:
                # Kill any lingering processes
                procs_file = cgroup_path / "cgroup.procs"
                if procs_file.exists():
                    pids = procs_file.read_text().strip().split()
                    for pid_str in pids:
                        if pid_str.strip():
                            try:
                                os.kill(int(pid_str), signal.SIGKILL)
                            except (ProcessLookupError, ValueError):
                                pass
                    await asyncio.sleep(0.05)
                cgroup_path.rmdir()
            except (FileNotFoundError, OSError):
                pass  # already cleaned up

    if proc.returncode != 0:
        err_text = _stderr.decode(errors="replace").strip()
        raise LeavesError(
            LeavesErrorCode.INTERNAL_ERROR,
            detail=f"runner exited {proc.returncode}: {err_text[:300]}",
        )

    try:
        resp = json.loads(stdout.split(b"\n")[0])
    except Exception as exc:
        raise LeavesError(LeavesErrorCode.INTERNAL_ERROR,
                          detail=f"runner output not JSON: {exc}")

    if resp.get("ok"):
        return resp["results"], resp["summary"]

    code_str = resp.get("code", "INTERNAL_ERROR")
    try:
        code = LeavesErrorCode[code_str]
    except KeyError:
        code = LeavesErrorCode.INTERNAL_ERROR
    raise LeavesError(code, detail=resp.get("detail") or resp.get("error"))


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        line = await reader.readline()
        if not line:
            return

        msg = json.loads(line)

        # --- ping ---
        if msg.get("_ping"):
            writer.write(json.dumps({"_pong": True}).encode() + b"\n")
            await writer.drain()
            return

        # --- reload watcher ---
        if msg.get("_reload_watcher"):
            if _watcher is not None:
                _watcher.reload()
            writer.write(json.dumps({"_reloaded": True}).encode() + b"\n")
            await writer.drain()
            return

        # --- trigger re-index ---
        if msg.get("_reindex"):
            if _indexer is not None:
                loop = asyncio.get_event_loop()
                full = msg.get("full", False)
                if full:
                    asyncio.ensure_future(
                        loop.run_in_executor(None, _indexer.run_once)
                    )
                else:
                    asyncio.ensure_future(
                        loop.run_in_executor(None, _indexer.run_incremental)
                    )
            writer.write(json.dumps({"_reindex_started": True}).encode() + b"\n")
            await writer.drain()
            return

        # --- execute GoalSpec ---
        goal_spec      = msg["goal_spec"]
        from_state_str = msg.get("from_state", "PARSING")

        try:
            results, summary = await _run_sandboxed(goal_spec, from_state_str)
            response: dict = {"ok": True, "results": results, "summary": summary}
        except LeavesError as e:
            response = {
                "ok":     False,
                "code":   e.code.value,
                "error":  e.user_message,
                "detail": e.detail,
            }

    except Exception as exc:
        response = {"ok": False, "code": "INTERNAL_ERROR", "error": str(exc), "detail": None}

    try:
        writer.write(json.dumps(response).encode() + b"\n")
        await writer.drain()
    except Exception:
        pass
    finally:
        writer.close()
        await writer.wait_closed()


_watcher = None   # IntentWatcher instance, set in _run_server
_indexer = None   # CortexIndexer instance, set in _run_server
_event_watcher = None  # CompositorEventWatcher instance, set in _run_server

_INDEX_INTERVAL_SECONDS = 1800  # 30 minutes between incremental re-indexes


async def _background_indexing(indexer) -> None:
    """
    Run initial full index, then incremental every 30 minutes.
    Runs in a thread pool to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()

    # Initial full index
    try:
        print("agentd: starting initial index (background)…", flush=True)
        stats = await loop.run_in_executor(None, indexer.run_once)
        print(
            f"agentd: initial index complete — "
            f"{stats['total_scanned']} scanned, "
            f"{stats['total_upserted']} indexed "
            f"in {stats.get('elapsed_seconds', 0):.1f}s",
            flush=True,
        )
    except Exception as e:
        print(f"agentd: initial index failed: {e}", flush=True)

    # Periodic incremental
    while True:
        await asyncio.sleep(_INDEX_INTERVAL_SECONDS)
        try:
            stats = await loop.run_in_executor(None, indexer.run_incremental)
            if stats["total_upserted"] > 0:
                print(
                    f"agentd: incremental index — {stats['total_upserted']} updated",
                    flush=True,
                )
        except Exception as e:
            print(f"agentd: incremental index failed: {e}", flush=True)


def _fire_goalspec_sync(goal_spec: dict) -> None:
    """
    Execute a GoalSpec from the watcher thread (synchronous).

    Runs through AgentCoordinator directly (in-process, no socket round-trip).
    Uses its own DB connection since this runs in the watcher thread.
    """
    import time as _time

    db = get_db()
    intent_id = goal_spec.get("intent_id", "unknown")
    natural_text = goal_spec.get("natural_text", "")

    print(f"agentd: watcher firing intent {intent_id[:8]} — {natural_text}", flush=True)
    t_start = _time.monotonic()

    try:
        log_intent_created(db, intent_id, natural_text, goal_spec)
        lifecycle = IntentLifecycle(intent_id=intent_id)
        lifecycle.transition(IntentState.PARSING)
        log_state_transition(db, intent_id, "PENDING", "PARSING")

        coordinator = AgentCoordinator(db)
        results, summary = coordinator.execute(goal_spec, lifecycle)

        lifecycle.transition(IntentState.DONE)
        log_state_transition(db, intent_id, "EXECUTING", "DONE")
        duration_ms = (_time.monotonic() - t_start) * 1000
        complete_intent(db, intent_id, "DONE", summary, duration_ms=duration_ms)
        print(f"agentd: watcher intent done in {duration_ms:.0f}ms — {summary}", flush=True)

    except LeavesError as e:
        duration_ms = (_time.monotonic() - t_start) * 1000
        complete_intent(db, intent_id, "FAILED", e.detail, duration_ms=duration_ms)
        print(f"agentd: watcher intent failed — {e.user_message}", flush=True)
    except Exception as e:
        print(f"agentd: watcher intent error — {e}", flush=True)


async def _on_terminal_error(
    app_id: str, exit_code: int, scrollback_path: pathlib.Path,
) -> None:
    """Synthesize a proactive intent card when a terminal exits with error."""
    scrollback = ""
    try:
        if scrollback_path.exists():
            scrollback = scrollback_path.read_text(errors="replace")[-2000:]
    except OSError:
        pass

    if not scrollback:
        return

    intent_text = (
        f"The terminal ({app_id}) just exited with error code {exit_code}. "
        f"The last output was:\n{scrollback}\n"
        f"What went wrong and what should I do next?"
    )

    import uuid as _uuid
    import time as _time

    goal_spec = {
        "intent_id": str(_uuid.uuid4()),
        "natural_text": intent_text,
        "category": "system_task",
        "actions": [{
            "action_id": "act-1",
            "type": "QUERY",
            "agent": "system",
            "params": {"query_type": "diagnostics", "exit_code": exit_code},
            "destructive": False,
        }],
        "authorization": {
            "resources": ["~"],
            "preview_required": True,
            "reversible": True,
        },
        "metadata": {
            "confidence": 0.90,
            "parse_latency_ms": 0.0,
            "model": "proactive-terminal-error",
            "proactive": True,
            "source_app": app_id,
        },
    }

    db = get_db()
    log_intent_created(db, goal_spec["intent_id"], intent_text, goal_spec)
    print(
        f"agentd: proactive intent created for {app_id} "
        f"exit code {exit_code}",
        flush=True,
    )


async def _run_server() -> None:
    global _CGROUP_AVAILABLE, _watcher, _event_watcher

    # One-time cgroup v2 parent setup
    try:
        if pathlib.Path("/sys/fs/cgroup/cgroup.controllers").exists():
            _CGROUP_ROOT.mkdir(parents=False, exist_ok=True)
            _CGROUP_AVAILABLE = True
            print("agentd: cgroup v2 resource limits enabled", flush=True)
        else:
            print("agentd: cgroup v2 not mounted — limits disabled", flush=True)
    except PermissionError:
        print("agentd: no cgroup write permission — limits disabled", flush=True)

    _SOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _SOCK_PATH.exists():
        _SOCK_PATH.unlink()

    # Start filesystem watcher for persistent intents
    try:
        from cortex.watcher import IntentWatcher
        watcher_db = get_db()
        _watcher = IntentWatcher(watcher_db, _fire_goalspec_sync)
        _watcher.start()
        print("agentd: filesystem watcher started", flush=True)
    except Exception as e:
        print(f"agentd: filesystem watcher failed to start: {e}", flush=True)
        _watcher = None

    # Start cortex indexer — background thread for initial + periodic indexing
    try:
        from cortex.indexer import CortexIndexer
        from cortex.adapters.filesystem import FilesystemAdapter
        indexer_db = get_db()
        lance_path = pathlib.Path(__file__).parent / "rag" / ".lancedb"
        _indexer = CortexIndexer(indexer_db, lance_path)
        _indexer.register(FilesystemAdapter())
        print("agentd: cortex indexer registered", flush=True)
    except Exception as e:
        print(f"agentd: cortex indexer failed to initialize: {e}", flush=True)
        _indexer = None

    # Start compositor event watcher
    try:
        from compositor.event_watcher import CompositorEventWatcher
        _event_watcher = CompositorEventWatcher()
        _event_watcher.on_nonzero_exit(_on_terminal_error)
        await _event_watcher.start()
        print("agentd: compositor event watcher started", flush=True)
    except Exception as e:
        print(f"agentd: compositor event watcher failed: {e}", flush=True)
        _event_watcher = None

    server = await asyncio.start_unix_server(_handle_client, path=str(_SOCK_PATH))
    print(f"agentd listening on {_SOCK_PATH}", flush=True)

    # Schedule background indexing
    if _indexer is not None:
        asyncio.get_event_loop().create_task(_background_indexing(_indexer))

    try:
        async with server:
            await server.serve_forever()
    finally:
        if _event_watcher is not None:
            await _event_watcher.stop()
        if _watcher is not None:
            _watcher.stop()
        if _SOCK_PATH.exists():
            _SOCK_PATH.unlink()


if __name__ == "__main__":
    # Ensure the project root is on sys.path so relative imports work when
    # agentd.py is spawned as a subprocess from any working directory.
    _project_root = str(pathlib.Path(__file__).parent.resolve())
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    asyncio.run(_run_server())
