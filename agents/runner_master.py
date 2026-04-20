#!/usr/bin/env python3
"""
Leaves OS runner-pool master.

Long-lived process that pre-imports the agent stack and forks per-intent
worker children. Eliminates the ~150-300ms CPython startup + module-import
cost paid by the cold-path `sandboxed_runner.py` subprocess: warm workers
inherit the parent's loaded modules via copy-on-write and only pay for
fork() (~5ms) + Landlock setup (~1ms).

Architecture:

    runner-master  (long-lived; project code already imported, NOT Landlocked)
        │
        ├── per accept(): fork() → worker child
        │       ├── worker tells agentd its pid
        │       ├── waits for "cgroup_ready" ack from agentd
        │       ├── applies Landlock with the intent's authorized paths
        │       ├── runs AgentCoordinator
        │       ├── streams `event` frames + final `result` frame
        │       └── exits
        └── parent closes the connection fd, returns to accept()

Why pre-fork is safe here:
  • Landlock is applied INSIDE the child after fork(). Per-intent FS scope
    is preserved; the parent never holds the restriction.
  • Children get fresh DB handles (forked SQLite handles are unsafe — we
    re-open via get_db() in the child).
  • Address spaces diverge on first write — one intent's data cannot leak
    to another via shared memory.

Protocol on each per-connection socket (newline-delimited JSON):

    Client → Server (one frame):
        {"goal_spec": {...}, "from_state": "..."}

    Server → Client (multiple frames, then close):
        {"_kind": "worker_pid", "pid": <int>}
        ── client may now send {"_kind": "cgroup_ready"} ──
        {"_kind": "event", "event": {...}}        # zero or more
        {"_kind": "result", "ok": true, "results": ..., "summary": ..., "sandbox": ...}

Socket path: ~/.leaves/runner-pool.sock (0o600, same uid as agentd).
"""
from __future__ import annotations

import json
import os
import pathlib
import signal
import socket
import sys
from typing import Any

# Ensure project root is importable before we eagerly load the heavy modules.
_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Landlock primitives — stdlib-only, safe to import pre-fork.
from agents.sandboxed_runner import apply_landlock  # noqa: E402

_SOCK_PATH = pathlib.Path.home() / ".leaves" / "runner-pool.sock"

# Cap how long the worker waits for the cgroup-ready ack. agentd's cgroup
# write is ~1ms; if we don't hear back in 5s, agentd has crashed and the
# worker should bail rather than hang.
_ACK_TIMEOUT_S = 5.0


def _eager_import() -> None:
    """
    Force-load every project module the worker child will need so each fork
    inherits them via COW. Runs ONCE at master startup; any exception here
    crashes the master loudly, which is correct (a broken module never
    becomes a silent runtime error inside the worker).
    """
    # The agent registry transitively pulls in every agent class and their
    # heavy deps (psutil, requests, beautifulsoup, lancedb, anthropic, …).
    from agents.registry import agent_classes
    agent_classes()
    # Coordinator + state-machine + cancel signal handler.
    from agents.state_machine import IntentLifecycle, IntentState  # noqa: F401
    from agents.cancel import cancel_event, install_handler  # noqa: F401
    from agentd import AgentCoordinator  # noqa: F401
    # DB layer.
    from db.audit import get_db  # noqa: F401
    # Errors + observability so worker rejections increment the same counters.
    from errors import LeavesError, LeavesErrorCode  # noqa: F401
    from observability import configure_logging, intents_total  # noqa: F401
    # Inference client (loads `requests`).
    from inference.client import LocalLlamaCppBackend  # noqa: F401


# ─── socket framing helpers ────────────────────────────────────────────────


def _read_one_line(sock: socket.socket, timeout: float | None = None) -> bytes:
    """
    Read until the first '\n' or EOF. Returns the bytes BEFORE the newline.
    Empty return on closed peer.
    """
    if timeout is not None:
        sock.settimeout(timeout)
    buf = bytearray()
    try:
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf.extend(chunk)
            nl = buf.find(b"\n")
            if nl >= 0:
                return bytes(buf[:nl])
    except (TimeoutError, OSError):
        return bytes(buf).split(b"\n", 1)[0]
    finally:
        if timeout is not None:
            sock.settimeout(None)
    return bytes(buf)


def _send_frame(sock: socket.socket, obj: dict[str, Any]) -> None:
    sock.sendall((json.dumps(obj) + "\n").encode())


def _is_json_safe(x: Any) -> bool:
    try:
        json.dumps(x)
        return True
    except (TypeError, ValueError):
        return False


# ─── child execution path ──────────────────────────────────────────────────


def _handle_in_child(conn: socket.socket, request: dict) -> None:
    """
    Runs in the forked worker. The parent has already validated the request
    frame and passed it as `request`; this function applies Landlock and
    executes the GoalSpec, streaming events + the final result back over
    `conn`.
    """
    # Re-imports are no-ops since the modules are already in sys.modules,
    # but the names need to be in local scope.
    from agents.cancel import cancel_event, install_handler as install_cancel_handler
    from agents.state_machine import IntentLifecycle, IntentState
    from agentd import AgentCoordinator
    from db.audit import get_db
    from errors import LeavesError

    goal_spec = request["goal_spec"]
    from_state_str = request.get("from_state", "PARSING")

    # 1. Tell the parent (agentd) our pid so it can do the cgroup write.
    _send_frame(conn, {"_kind": "worker_pid", "pid": os.getpid()})

    # 2. Wait for cgroup_ready ack. If agentd's cgroup setup failed, it will
    #    still send the ack — the security boundary is Landlock, not cgroup,
    #    so we proceed regardless.
    ack_line = _read_one_line(conn, timeout=_ACK_TIMEOUT_S)
    try:
        ack = json.loads(ack_line) if ack_line else {}
    except json.JSONDecodeError:
        ack = {}
    if ack.get("_kind") != "cgroup_ready":
        # Don't fail — agentd may just be on an older protocol. Proceed.
        pass

    # 3. WAYLAND_DISPLAY override (parity with cold path).
    wl_display = goal_spec.get("metadata", {}).get("wayland_display")
    if wl_display:
        os.environ["WAYLAND_DISPLAY"] = wl_display

    # 4. Apply Landlock. App-launch intents skip — the launched GUI must not
    #    inherit our FS restrictions. Same logic as sandboxed_runner._main().
    is_launch = bool(goal_spec.get("actions")) and all(
        a.get("agent") == "system" and a.get("type", "").upper() == "WRITE"
        for a in goal_spec.get("actions", [])
    )
    if not is_launch:
        sandbox_status = apply_landlock(goal_spec)
    else:
        sandbox_status = {
            "active": False,
            "reason": "launch_skipped",
            "authorized_resources": [
                str(r)
                for r in goal_spec.get("authorization", {}).get("resources", [])
                if r
            ],
        }

    # 5. Cancel handler (SIGUSR1 → cancel_event) — must be installed on the
    #    main thread BEFORE AgentCoordinator's worker pool spins up so its
    #    threads inherit a process where the signal is wired.
    install_cancel_handler()

    # 6. Lifecycle to pre-execution state.
    intent_id = goal_spec["intent_id"]
    lifecycle = IntentLifecycle(intent_id=intent_id)
    lifecycle.transition(IntentState.PARSING)
    if from_state_str == "AWAITING_AUTH":
        lifecycle.transition(IntentState.AWAITING_AUTH)

    # 7. Channel-message hook: stream `event` frames over the same socket.
    def _on_channel_message(msg) -> None:
        try:
            data = msg.data if _is_json_safe(msg.data) else repr(msg.data)
        except Exception:
            data = None
        try:
            _send_frame(conn, {
                "_kind": "event",
                "event": {
                    "action_id": msg.action_id,
                    "kind": msg.kind,
                    "data": data,
                },
            })
        except (OSError, ValueError):
            pass

    # 8. Re-open the audit DB. Inheriting a parent's SQLite handle across
    #    fork() is documented-unsafe; get_db() returns a fresh per-process
    #    connection, but we must NOT reuse any handle that may have been
    #    cached in the parent before fork. The eager import path doesn't
    #    open any DB connections, so this is the first open in this process.
    db = get_db()

    try:
        coordinator = AgentCoordinator(
            db,
            on_channel_message=_on_channel_message,
            cancel_event=cancel_event,
        )
        results, summary = coordinator.execute(goal_spec, lifecycle)
        if cancel_event.is_set():
            summary = (summary or "") + " (cancelled by user)"
        result_frame: dict = {
            "_kind": "result",
            "ok": True,
            "results": results,
            "summary": summary,
            "sandbox": sandbox_status,
        }
    except LeavesError as e:
        result_frame = {
            "_kind": "result",
            "ok": False,
            "code": e.code.value,
            "error": e.user_message,
            "detail": e.detail,
        }
    except Exception as e:  # noqa: BLE001 — boundary
        result_frame = {
            "_kind": "result",
            "ok": False,
            "code": "INTERNAL_ERROR",
            "error": str(e),
            "detail": None,
        }

    try:
        _send_frame(conn, result_frame)
    except OSError:
        pass


# ─── parent accept loop ────────────────────────────────────────────────────


def _serve() -> None:
    _SOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _SOCK_PATH.exists():
        _SOCK_PATH.unlink()

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(_SOCK_PATH))
    # Same-uid only — defense-in-depth alongside the SO_PEERCRED checks
    # agentd applies to its own socket. See security audit F-2/F-3.
    os.chmod(_SOCK_PATH, 0o600)
    sock.listen(64)

    print("runner-pool: listening", flush=True)

    # Auto-reap exited children — we never wait4() them ourselves and we
    # don't want zombies. SIG_IGN on SIGCHLD is the POSIX-blessed reaper.
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    while True:
        try:
            conn, _ = sock.accept()
        except KeyboardInterrupt:
            break
        except OSError:
            continue

        # Read the request frame in the PARENT before fork(). Two reasons:
        #  1. A malformed request can be rejected without spawning a child.
        #  2. The child starts with the request already in memory — no
        #     blocking read needed before it can do useful work.
        try:
            req_line = _read_one_line(conn, timeout=5.0)
            request = json.loads(req_line) if req_line else None
        except (json.JSONDecodeError, OSError):
            request = None

        if not isinstance(request, dict) or "goal_spec" not in request:
            try:
                _send_frame(conn, {
                    "_kind": "result",
                    "ok": False,
                    "code": "INTERNAL_ERROR",
                    "error": "malformed request frame",
                })
            except OSError:
                pass
            conn.close()
            continue

        pid = os.fork()
        if pid == 0:
            # Child: drop the listening socket so it doesn't leak across
            # workers, handle the connection, exit.
            try:
                sock.close()
            except OSError:
                pass
            try:
                _handle_in_child(conn, request)
            except Exception as e:  # noqa: BLE001 — boundary
                try:
                    _send_frame(conn, {
                        "_kind": "result",
                        "ok": False,
                        "code": "INTERNAL_ERROR",
                        "error": f"worker crash: {e}",
                    })
                except OSError:
                    pass
            finally:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                conn.close()
            os._exit(0)
        else:
            # Parent: close our copy of the connection fd and accept the next.
            conn.close()


if __name__ == "__main__":
    _eager_import()
    _serve()
