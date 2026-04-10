"""
Bidirectional channel for parallel DAG execution.

Each running action is given an ActionChannel by the scheduler. Through it
the agent can:
  - emit(kind, data)        — push a non-terminal message to the scheduler
                              (kind ∈ progress | log | partial)
  - cancelled()             — cooperative cancellation poll
  - recv(timeout)           — pull the next inbox message (or None on EOS)
  - consume()               — iterator over inbox messages until EOS

The scheduler can:
  - cancel()                — request cooperative cancel (sets the event)
  - _push_inbox(msg)        — deliver an upstream stream-edge message
  - _push_eos()             — deliver end-of-stream sentinel

Streaming edges:
  When action B has a `stream` dependency on action A, the scheduler
  forwards every `partial` message A emits into B's inbox, then pushes
  an EOS sentinel (None) when A finishes. B reads the stream via
  `for msg in channel.consume(): ...` — this lets pipelines start
  producing output before the upstream action has finished.

Terminal "done" / "error" messages are emitted by the agentd worker wrapper
itself, not by agents — emit() rejects reserved kinds so an agent can never
forge a completion message.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any

# Kinds an agent is allowed to emit through ActionChannel.emit().
ALLOWED_AGENT_KINDS = frozenset({"progress", "log", "partial"})

# Terminal kinds — emitted by the worker wrapper, never by agents directly.
TERMINAL_KINDS = frozenset({"done", "error"})


@dataclass(frozen=True)
class ChannelMessage:
    """One message flowing from a worker to the scheduler's drain loop."""
    action_id: str
    kind: str
    data: Any = None


class ActionChannel:
    """
    Per-action bidirectional channel.

    The outbox is shared across every channel in a single DAG execution;
    the scheduler demultiplexes by action_id. The inbox is per-channel and
    is used for streaming-edge delivery: upstream `partial` messages are
    forwarded by the scheduler, terminated by an EOS sentinel (None).

    Cancellation is per-channel so the scheduler can target individual
    in-flight actions.
    """

    __slots__ = ("action_id", "_outbox", "_inbox", "_cancel")

    def __init__(
        self,
        action_id: str,
        outbox: "queue.Queue[ChannelMessage]",
    ):
        self.action_id = action_id
        self._outbox = outbox
        # Sentinel for end-of-stream is `None`. Inbox accepts ChannelMessage
        # or None.
        self._inbox: "queue.Queue[ChannelMessage | None]" = queue.Queue()
        self._cancel = threading.Event()

    # ------------------------------------------------------------------
    # Agent-facing API — outbound (worker → scheduler)
    # ------------------------------------------------------------------

    def emit(self, kind: str, data: Any = None) -> None:
        """
        Push a non-terminal message to the scheduler.

        Raises ValueError if kind is reserved (done/error) or unknown — the
        worker wrapper owns terminal status, agents must not forge it.
        """
        if kind not in ALLOWED_AGENT_KINDS:
            raise ValueError(
                f"ActionChannel.emit: kind {kind!r} is not permitted "
                f"for agents (allowed: {sorted(ALLOWED_AGENT_KINDS)})"
            )
        self._outbox.put(ChannelMessage(self.action_id, kind, data))

    def cancelled(self) -> bool:
        """Cooperative cancellation flag. Long-running agents should poll this."""
        return self._cancel.is_set()

    # ------------------------------------------------------------------
    # Agent-facing API — inbound (upstream stream-edge → worker)
    # ------------------------------------------------------------------

    def recv(self, timeout: float | None = None) -> "ChannelMessage | None":
        """
        Pull the next inbox message. Returns None on end-of-stream.

        Raises queue.Empty if `timeout` is given and elapses with no message.
        Blocks indefinitely when timeout is None.
        """
        return self._inbox.get(timeout=timeout)

    def consume(self):
        """
        Iterate over upstream stream-edge messages until end-of-stream.

        Usage in a stream-consumer agent:
            for msg in channel.consume():
                process(msg.data)
        """
        while True:
            msg = self._inbox.get()
            if msg is None:  # EOS sentinel
                return
            yield msg

    # ------------------------------------------------------------------
    # Scheduler-facing API
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cooperative cancellation. Idempotent."""
        self._cancel.set()

    def _push_inbox(self, msg: "ChannelMessage") -> None:
        """Scheduler-only: forward an upstream stream-edge message."""
        self._inbox.put(msg)

    def _push_eos(self) -> None:
        """Scheduler-only: signal end-of-stream from all upstream parents."""
        self._inbox.put(None)
