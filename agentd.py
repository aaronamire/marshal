"""
Leaves OS Agent Daemon.

Phase 1: AgentCoordinator as a library module (imported by leaves.py).
Phase 2: Unix socket daemon + parallel DAG execution.
         leaves.py connects via ~/.leaves/agentd.sock (newline-delimited JSON).
         Multi-action intents run concurrently via depends_on DAG scheduling.
Phase 3: cgroup integration and per-intent subprocess isolation.

Public interface (callers use ONLY this):
    coordinator = AgentCoordinator(db)
    results, summary = coordinator.execute(goal_spec, lifecycle)
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import pathlib
import queue
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agents.channel import ActionChannel, ChannelMessage
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

# Per-agent-type cache: does execute_action accept a `channel` kwarg?
# Lets new agents opt into ActionChannel by adding `channel=None` to their
# signature without forcing every legacy agent to be updated at once.
_AGENT_CHANNEL_SUPPORT: dict[str, bool] = {}


def _agent_accepts_channel(agent_type: str, agent_cls: type) -> bool:
    cached = _AGENT_CHANNEL_SUPPORT.get(agent_type)
    if cached is not None:
        return cached
    try:
        sig = inspect.signature(agent_cls.execute_action)
        accepts = "channel" in sig.parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        )
    except (TypeError, ValueError):
        accepts = False
    _AGENT_CHANNEL_SUPPORT[agent_type] = accepts
    return accepts


class AgentCoordinator:
    """
    Coordinates agent execution for a single intent.

    Single-action intents run directly (zero overhead).
    Multi-action intents use DAG-scheduled parallel execution:
    actions with satisfied depends_on run concurrently in a thread pool.

    Stream dependency edges (depends_on items of the form
    {"id": ..., "mode": "stream"}) let downstream actions start as soon as
    their upstream parent starts emitting `partial` messages, enabling
    pipelined execution.

    `on_channel_message`, if supplied, is called by the DAG drain loop for
    every non-terminal channel message (progress / log / partial). It runs
    on the main scheduler thread, so callbacks must be cheap and non-blocking.
    Use this hook to forward live progress to a UI or to a parent process.

    One instance per intent. Do not reuse across intents.
    """

    def __init__(self, db, on_channel_message=None, cancel_event=None):
        self._db = db
        self._on_channel_message = on_channel_message
        # Optional threading.Event polled by the DAG scheduler. When set
        # mid-flight the loop transitions into the abort path: in-flight
        # actions get their channels cancelled and EOS'd, no new actions
        # are submitted, pending actions are marked skipped, and the
        # caller receives a partial result. The runner subprocess sets
        # this event in response to SIGUSR1 from agentd.
        self._cancel_event = cancel_event

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

        # Pre-inject compositor WAYLAND_DISPLAY into each action
        _wl_meta = goal_spec.get("metadata", {}).get("wayland_display")
        if _wl_meta:
            for action in actions:
                action["_wayland_display"] = _wl_meta

        # Single-action fast path — no thread pool overhead
        if len(actions) <= 1:
            results, failed_ids = self._execute_sequential(
                intent_id, actions, tracker)
        else:
            results, failed_ids = self._execute_dag(
                intent_id, actions, tracker)

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

    # ------------------------------------------------------------------
    # Sequential execution (0-1 actions)
    # ------------------------------------------------------------------

    def _execute_sequential(
        self,
        intent_id: str,
        actions: list[dict],
        tracker: ToolFailureTracker,
    ) -> tuple[dict[str, Any], list[str]]:
        results: dict[str, Any] = {}
        failed_ids: list[str] = []

        for action in actions:
            if self._cancel_event is not None and self._cancel_event.is_set():
                results[action.get("action_id", "unknown")] = {
                    "error": "Cancelled by user",
                    "skipped": True,
                }
                failed_ids.append(action.get("action_id", "unknown"))
                continue

            action_id = action.get("action_id", "unknown")
            agent_type = action.get("agent", "")

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
                break

            agent = _AGENT_MAP[agent_type](
                intent_id=intent_id, db_conn=self._db)
            try:
                result = agent.execute_action(action)
                tracker.reset(action.get("type", ""), action.get("params", {}))
                results[action_id] = result
            except LeavesError as e:
                try:
                    tracker.record_failure(
                        action.get("type", ""),
                        action.get("params", {}),
                        error=e,
                    )
                except LeavesError as escalated:
                    log_error(self._db, escalated.code.value,
                              escalated.detail, intent_id)
                    results[action_id] = {"error": escalated.user_message}
                    failed_ids.append(action_id)
                    raise escalated

                log_error(self._db, e.code.value, e.detail, intent_id)
                results[action_id] = {"error": e.user_message}
                failed_ids.append(action_id)

        return results, failed_ids

    # ------------------------------------------------------------------
    # DAG-scheduled parallel execution (2+ actions)
    # ------------------------------------------------------------------

    def _execute_dag(
        self,
        intent_id: str,
        actions: list[dict],
        tracker: ToolFailureTracker,
    ) -> tuple[dict[str, Any], list[str]]:
        """
        Execute actions respecting depends_on ordering, with support for
        full and stream dependency edges.

        Edge modes:
          - "full" (default, also: bare action_id string): downstream waits
            for upstream's terminal done before starting.
          - "stream": downstream may start as soon as upstream STARTS;
            upstream's `partial` messages are forwarded to downstream's
            inbox, and an EOS sentinel is delivered when upstream finishes.
            Late subscribers catch up via a per-parent buffer of partials.

        All shared state is mutated only on the main thread inside the
        message handler — workers communicate exclusively through the
        outbox queue, so no locks are needed.

        If the coordinator was constructed with an `on_channel_message`
        hook, every non-terminal channel message is forwarded to it on the
        main thread. The hook should be cheap and non-blocking.
        """
        action_map = {a["action_id"]: a for a in actions}

        # Parse depends_on into full vs stream sets, and build the inverse
        # subscriber map (parent -> set of stream-subscribed children).
        deps_full: dict[str, set[str]] = {}
        deps_stream: dict[str, set[str]] = {}
        subscribers: dict[str, set[str]] = {}

        for a in actions:
            aid = a["action_id"]
            deps_full[aid] = set()
            deps_stream[aid] = set()
            for d in a.get("depends_on", []):
                if isinstance(d, str):
                    deps_full[aid].add(d)
                    continue
                if not isinstance(d, dict) or "id" not in d:
                    continue
                parent = d["id"]
                mode = d.get("mode", "full")
                if mode == "stream":
                    deps_stream[aid].add(parent)
                    subscribers.setdefault(parent, set()).add(aid)
                else:
                    deps_full[aid].add(parent)

        results: dict[str, Any] = {}
        failed_ids: set[str] = set()
        abort = False

        # Shared outbox: every channel pushes here, main thread drains.
        outbox: queue.Queue[ChannelMessage] = queue.Queue()
        channels: dict[str, ActionChannel] = {}  # action_id -> live channel

        # Per-parent stream buffer + completion flag, used to deliver
        # already-emitted `partial` messages (and EOS) to subscribers that
        # haven't been submitted yet at the time the parent emits.
        stream_buffers: dict[str, list[ChannelMessage]] = {}
        stream_done: dict[str, bool] = {}

        # Thread-local DB connections (SQLite forbids cross-thread sharing)
        _tls = threading.local()

        def _worker_db():
            if not hasattr(_tls, "db"):
                _tls.db = get_db()
            return _tls.db

        def _run_action(action: dict, channel: ActionChannel) -> None:
            """
            Worker entry point. Resolves the agent, runs it, and emits a
            terminal message (done/error) to the outbox. Agents may emit
            non-terminal progress messages through `channel` while running.
            """
            aid = action["action_id"]
            agent_type = action.get("agent", "")
            try:
                if agent_type not in _AGENT_MAP:
                    raise LeavesError(
                        LeavesErrorCode.AGENT_NOT_AVAILABLE,
                        detail=(
                            f"Available agents: {list(_AGENT_MAP.keys())}. "
                            f"Got: '{agent_type}'"
                        ),
                    )
                agent_cls = _AGENT_MAP[agent_type]
                db = _worker_db()
                agent = agent_cls(intent_id=intent_id, db_conn=db)
                if _agent_accepts_channel(agent_type, agent_cls):
                    result = agent.execute_action(action, channel=channel)
                else:
                    result = agent.execute_action(action)
            except LeavesError as e:
                outbox.put(ChannelMessage(aid, "error", e))
                return
            except Exception as e:  # noqa: BLE001 — boundary
                outbox.put(ChannelMessage(
                    aid, "error",
                    LeavesError(
                        LeavesErrorCode.INTERNAL_ERROR,
                        detail=f"worker crashed: {e}",
                    ),
                ))
                return
            outbox.put(ChannelMessage(aid, "done", result))

        def _cascade_dep_failures() -> None:
            changed = True
            while changed:
                changed = False
                for aid in list(pending):
                    all_deps = deps_full[aid] | deps_stream[aid]
                    dep_failed = all_deps & failed_ids
                    if not dep_failed:
                        continue
                    err = LeavesError(
                        LeavesErrorCode.DEPENDENCY_FAILED,
                        detail=(
                            f"Action {aid} skipped: dependency "
                            f"{sorted(dep_failed)} failed"
                        ),
                    )
                    log_error(self._db, err.code.value,
                              err.detail, intent_id)
                    results[aid] = {
                        "error": err.user_message, "skipped": True,
                    }
                    failed_ids.add(aid)
                    pending.discard(aid)
                    changed = True

        def _is_ready(aid: str) -> bool:
            # Full deps must be in results (i.e., parent has finished).
            if deps_full[aid] - results.keys():
                return False
            # Stream deps are satisfied when the parent has either started
            # (currently in channels) or already finished (in results).
            started_or_done = set(channels.keys()) | set(results.keys())
            if deps_stream[aid] - started_or_done:
                return False
            return True

        def _submit(aid: str) -> None:
            ch = ActionChannel(aid, outbox)
            channels[aid] = ch
            # Replay any buffered upstream partials and EOS into the new
            # channel's inbox so a late subscriber doesn't miss them.
            for parent in deps_stream[aid]:
                for buffered in stream_buffers.get(parent, ()):
                    ch._push_inbox(buffered)
                if stream_done.get(parent):
                    ch._push_eos()
            pool.submit(_run_action, action_map[aid], ch)

        def _close_stream_to_subscribers(parent_aid: str) -> None:
            """Mark a parent's stream as finished and EOS its live subscribers."""
            stream_done[parent_aid] = True
            for sub_aid in subscribers.get(parent_aid, ()):
                ch = channels.get(sub_aid)
                if ch is not None:
                    ch._push_eos()

        pending: set[str] = set(action_map.keys())

        # Cancel polling: if a cancel event was supplied, the scheduler
        # uses a bounded outbox.get() so cancel takes effect within
        # _CANCEL_POLL_S of the user pressing Ctrl-C. Without a cancel
        # event the loop blocks indefinitely (zero overhead).
        _CANCEL_POLL_S = 0.25
        _cancel = self._cancel_event

        with ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="agent",
        ) as pool:
            while pending or channels:
                _cascade_dep_failures()

                if _cancel is not None and _cancel.is_set():
                    abort = True

                if abort:
                    # Mark remaining pending as skipped; signal cancel to
                    # any still-running siblings. EOS any in-flight stream
                    # subscribers so they don't block forever on recv().
                    for aid in list(pending):
                        results[aid] = {
                            "error": "Aborted due to prior failure",
                            "skipped": True,
                        }
                        failed_ids.add(aid)
                    pending.clear()
                    for ch in channels.values():
                        ch.cancel()
                        ch._push_eos()
                    if not channels:
                        break
                else:
                    # Submit every newly-ready action. Loop with a `changed`
                    # flag so chains of stream deps (A → B via stream, where
                    # both become ready in the same pass) get submitted in
                    # one go before we block on outbox.get().
                    submitted_any = True
                    while submitted_any:
                        submitted_any = False
                        for aid in list(pending):
                            if _is_ready(aid):
                                pending.discard(aid)
                                _submit(aid)
                                submitted_any = True

                    # Cycle: nothing in flight but pending remains.
                    if not channels:
                        for aid in list(pending):
                            results[aid] = {
                                "error": "Unreachable: circular dependency",
                                "skipped": True,
                            }
                            failed_ids.add(aid)
                        pending.clear()
                        break

                # Block until the next channel message. When a cancel
                # event is wired, use a bounded poll so Ctrl-C takes
                # effect even when no agent is producing messages.
                try:
                    if _cancel is not None:
                        msg = outbox.get(timeout=_CANCEL_POLL_S)
                    else:
                        msg = outbox.get()
                except queue.Empty:
                    continue
                aid = msg.action_id
                kind = msg.kind

                if kind == "done":
                    if aid not in channels:
                        continue  # stale — should not happen in phase 1
                    channels.pop(aid, None)
                    action = action_map[aid]
                    tracker.reset(
                        action.get("type", ""),
                        action.get("params", {}),
                    )
                    results[aid] = msg.data
                    _close_stream_to_subscribers(aid)

                elif kind == "error":
                    if aid not in channels:
                        continue
                    channels.pop(aid, None)
                    action = action_map[aid]
                    e: LeavesError = msg.data
                    try:
                        tracker.record_failure(
                            action.get("type", ""),
                            action.get("params", {}),
                            error=e,
                        )
                    except LeavesError as escalated:
                        log_error(
                            self._db, escalated.code.value,
                            escalated.detail, intent_id,
                        )
                        results[aid] = {"error": escalated.user_message}
                        failed_ids.add(aid)
                        _close_stream_to_subscribers(aid)
                        raise escalated

                    log_error(self._db, e.code.value, e.detail, intent_id)
                    results[aid] = {"error": e.user_message}
                    failed_ids.add(aid)
                    _close_stream_to_subscribers(aid)
                    if action.get("on_failure", "abort") == "abort":
                        abort = True

                elif kind == "partial":
                    # Buffer for late subscribers, forward to live ones.
                    stream_buffers.setdefault(aid, []).append(msg)
                    for sub_aid in subscribers.get(aid, ()):
                        sub_ch = channels.get(sub_aid)
                        if sub_ch is not None:
                            sub_ch._push_inbox(msg)
                    if self._on_channel_message is not None:
                        try:
                            self._on_channel_message(msg)
                        except Exception:  # noqa: BLE001 — hook is best-effort
                            pass

                else:
                    # progress / log — forward to hook only.
                    if self._on_channel_message is not None:
                        try:
                            self._on_channel_message(msg)
                        except Exception:  # noqa: BLE001
                            pass

        return results, list(failed_ids)


# ---------------------------------------------------------------------------
# Unix socket daemon  (only active when run as __main__)
# ---------------------------------------------------------------------------

_SOCK_PATH = pathlib.Path.home() / ".leaves" / "agentd.sock"

# cgroup v2 resource limits — initialized in _run_server()
_CGROUP_ROOT = pathlib.Path("/sys/fs/cgroup/leaves")
_CGROUP_AVAILABLE = False

# Map of in-flight runner subprocesses keyed by full intent_id. The
# {"_cancel": "<intent_id>"} socket message looks up the matching proc
# and SIGUSR1's it. Mutated only from the asyncio main thread inside
# _run_sandboxed and _handle_client, so no lock is needed.
_INFLIGHT_PROCS: dict[str, "asyncio.subprocess.Process"] = {}


def _is_app_launch(goal_spec: dict) -> bool:
    """True when the goal only launches apps (system WRITE actions)."""
    actions = goal_spec.get("actions", [])
    return bool(actions) and all(
        a.get("agent") == "system" and a.get("type", "").upper() == "WRITE"
        for a in actions
    )


async def _read_event_pipe(fd: int, on_event) -> None:
    """
    Read newline-delimited JSON event frames from `fd` until EOF.
    Each frame is decoded and passed to `on_event` (which may be sync
    or async). Errors are swallowed — events are best-effort.
    """
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    pipe_obj = os.fdopen(fd, "rb", 0)
    try:
        await loop.connect_read_pipe(lambda: protocol, pipe_obj)
    except Exception:
        pipe_obj.close()
        return
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                res = on_event(event)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass


async def _run_sandboxed(
    goal_spec: dict,
    from_state: str,
    on_event=None,
) -> tuple[dict, str]:
    """
    Spawn agents/sandboxed_runner.py as a fresh subprocess.

    The subprocess applies Landlock to itself before importing project code,
    restricting its FS access to the intent's authorized resources.
    Communication is JSON over stdin/stdout.

    If `on_event` is provided, a sidecar pipe is opened and its write end is
    passed to the subprocess via `LEAVES_EVENT_FD`. The subprocess writes
    JSON channel-message frames to that fd as it executes; this function
    reads them concurrently and forwards each to `on_event`. Pre-opened
    pipes are not subject to Landlock (which restricts paths, not fds), so
    the sidecar works even when the runner sandbox is active.

    If cgroup v2 is available, the subprocess is placed in a per-intent
    cgroup with memory, CPU, and PID limits.

    App-launch intents (system WRITE) skip cgroup entirely — the launched
    GUI app must outlive the runner and needs unrestricted resources.
    """
    runner = pathlib.Path(__file__).parent / "agents" / "sandboxed_runner.py"
    payload = json.dumps({"goal_spec": goal_spec,
                          "from_state": from_state}).encode() + b"\n"
    full_intent_id = goal_spec.get("intent_id", "unknown")
    intent_id = full_intent_id[:8]
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

    # Sidecar event pipe (only when a consumer is interested in events).
    event_r = -1
    event_w = -1
    pass_fds: tuple[int, ...] = ()
    event_task: asyncio.Task | None = None
    if on_event is not None:
        event_r, event_w = os.pipe()
        os.set_inheritable(event_w, True)
        pass_fds = (event_w,)
        sub_env["LEAVES_EVENT_FD"] = str(event_w)

    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(runner),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=sub_env,
        pass_fds=pass_fds,
    )

    # Register this proc so cancel-by-intent-id can find it. The entry
    # is removed in the finally block below regardless of how the
    # subprocess exits (success, timeout, error, signal).
    _INFLIGHT_PROCS[full_intent_id] = proc

    # Parent no longer needs the write end; the child has its own copy.
    # Closing here is what eventually delivers EOF to the reader when the
    # child exits (or closes its fd).
    if event_w >= 0:
        os.close(event_w)
        event_w = -1
        event_task = asyncio.create_task(_read_event_pipe(event_r, on_event))

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
        if event_task is not None:
            event_task.cancel()
        raise LeavesError(LeavesErrorCode.INFERENCE_TIMEOUT,
                          detail="sandboxed runner timed out")
    finally:
        # Always deregister so a stale entry can't accumulate.
        _INFLIGHT_PROCS.pop(full_intent_id, None)
        # Subprocess closed its end of the event pipe when it exited; the
        # reader task should drain remaining frames and finish on its own.
        if event_task is not None:
            try:
                await asyncio.wait_for(event_task, timeout=1.0)
            except asyncio.TimeoutError:
                event_task.cancel()
            except Exception:  # noqa: BLE001
                pass
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

        # --- cancel an in-flight intent ---
        if "_cancel" in msg:
            target_id = msg["_cancel"]
            proc = _INFLIGHT_PROCS.get(target_id)
            if proc is None:
                writer.write(json.dumps(
                    {"_cancelled": False, "reason": "not_inflight"}
                ).encode() + b"\n")
            else:
                try:
                    proc.send_signal(signal.SIGUSR1)
                    cancelled = True
                except (ProcessLookupError, PermissionError, OSError):
                    cancelled = False
                writer.write(json.dumps(
                    {"_cancelled": cancelled}
                ).encode() + b"\n")
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
        wants_events   = bool(msg.get("stream_events"))

        # Live event forwarder: writes one frame per channel message to
        # the client. The client opts in by setting stream_events=true in
        # its request — clients that don't opt in skip the sidecar pipe
        # entirely (zero overhead).
        async def _forward_event(event: dict) -> None:
            try:
                writer.write(
                    json.dumps({"event": event}).encode() + b"\n")
                await writer.drain()
            except Exception:  # noqa: BLE001
                pass

        try:
            results, summary = await _run_sandboxed(
                goal_spec, from_state_str,
                on_event=_forward_event if wants_events else None,
            )
            response: dict = {
                "final": True, "ok": True,
                "results": results, "summary": summary,
            }
        except LeavesError as e:
            response = {
                "final":  True,
                "ok":     False,
                "code":   e.code.value,
                "error":  e.user_message,
                "detail": e.detail,
            }

    except Exception as exc:
        response = {
            "final": True, "ok": False,
            "code": "INTERNAL_ERROR", "error": str(exc), "detail": None,
        }

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
