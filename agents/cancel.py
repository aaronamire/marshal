"""
Cooperative cancellation for the runner subprocess.

The OS supports user-initiated cancel of an in-flight intent. The control
path is:

    REPL Ctrl-C
        ↓ (sends {"_cancel": <intent_id>} on a fresh agentd socket)
    agentd._handle_client
        ↓ (looks up the in-flight asyncio subprocess and SIGUSR1s it)
    runner subprocess (PID inside the Landlock sandbox)
        ↓ (SIGUSR1 handler sets a process-wide threading.Event)
    AgentCoordinator scheduler loop
        ↓ (polls the event between outbox.get() ticks; treats it as abort)
    All in-flight agents
        ↓ (their channels are .cancel()'d, EOS pushed; no new actions submit)
    runner exits cleanly with a partial result + "cancelled" summary

Why SIGUSR1 instead of a control fd? The Landlock sandbox restricts paths,
not signals or pre-opened fds, so a signal works just as well as a pipe and
costs less new code. SIGUSR1 was chosen because SIGINT is already used by
asyncio for the daemon's own shutdown and we don't want them to collide.

This module exports:
    cancel_event       — process-wide threading.Event, set by the handler
    install_handler()  — register SIGUSR1 → set the event
    is_cancelled()     — helper for cheap polling

Importing this module is side-effect free; you must call install_handler()
explicitly. The runner does so at startup before importing AgentCoordinator.
"""
from __future__ import annotations

import signal
import threading

# Process-wide cancel flag. Once set, it stays set — the runner subprocess
# is one-shot per intent, so there is no need for a reset path.
cancel_event: threading.Event = threading.Event()


def _handler(_signum: int, _frame) -> None:
    cancel_event.set()


def install_handler() -> None:
    """
    Install the SIGUSR1 → cancel_event handler. Idempotent. Silently
    skipped on platforms that don't expose SIGUSR1 (Windows).
    """
    try:
        signal.signal(signal.SIGUSR1, _handler)
    except (AttributeError, ValueError, OSError):
        # AttributeError: SIGUSR1 missing on Windows
        # ValueError: not on main thread
        # OSError: handler install failed
        pass


def is_cancelled() -> bool:
    return cancel_event.is_set()
