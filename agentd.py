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
from agents.state_machine import IntentLifecycle, IntentState
from agents.tool_failure_tracker import ToolFailureTracker
from db.audit import get_db, log_error, log_intent_created, log_state_transition, complete_intent
from errors import LeavesError, LeavesErrorCode

# Map agent type strings -> agent classes
_AGENT_MAP: dict[str, type] = {
    "file": FileAgent,
    "system": SystemAgent,
    "web": WebAgent,
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

        for action in actions:
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


async def _run_sandboxed(goal_spec: dict, from_state: str) -> tuple[dict, str]:
    """
    Spawn agents/sandboxed_runner.py as a fresh subprocess.

    The subprocess applies Landlock to itself before importing project code,
    restricting its FS access to the intent's authorized resources.
    Communication is JSON over stdin/stdout.

    If cgroup v2 is available, the subprocess is placed in a per-intent
    cgroup with memory, CPU, and PID limits.
    """
    runner = pathlib.Path(__file__).parent / "agents" / "sandboxed_runner.py"
    payload = json.dumps({"goal_spec": goal_spec,
                          "from_state": from_state}).encode() + b"\n"
    intent_id = goal_spec.get("intent_id", "unknown")[:8]
    cgroup_path = _CGROUP_ROOT / f"intent-{intent_id}"
    cgroup_applied = False

    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(runner),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Apply cgroup limits after spawn (must have PID)
    if _CGROUP_AVAILABLE:
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


async def _run_server() -> None:
    global _CGROUP_AVAILABLE, _watcher

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

    server = await asyncio.start_unix_server(_handle_client, path=str(_SOCK_PATH))
    print(f"agentd listening on {_SOCK_PATH}", flush=True)

    # Schedule background indexing
    if _indexer is not None:
        asyncio.get_event_loop().create_task(_background_indexing(_indexer))

    try:
        async with server:
            await server.serve_forever()
    finally:
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
