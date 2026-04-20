"""
Tests for AgentCoordinator._execute_dag — the outbox-driven parallel
DAG executor introduced with ActionChannel.

These tests register a controllable mock agent into _AGENT_MAP and stub
out the audit DB so they run in milliseconds with no SQLite dependency.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import agentd  # noqa: E402
from agentd import AgentCoordinator  # noqa: E402
from agents.base_agent import BaseAgent  # noqa: E402
from agents.channel import ActionChannel  # noqa: E402
from agents.state_machine import IntentLifecycle, IntentState  # noqa: E402
from errors import MarshalError, MarshalErrorCode  # noqa: E402


# ---------------------------------------------------------------------------
# Mock agent — driven by a per-action plan stored in a class-level registry.
# ---------------------------------------------------------------------------

class _MockAgent(BaseAgent):
    """
    Test double. Looks up its behavior by action_id in _PLAN.

    Plan entry shape:
        {
            "delay": float,                     # seconds before returning
            "result": dict | None,              # success result
            "error": MarshalError | None,        # raise this instead
            "emit_progress": bool,              # call channel.emit() once
            "started_event": threading.Event,   # set when execution begins
            "block_until": threading.Event,     # block on this before returning
            "observed_cancel": list,            # appended True if cancelled() was True
        }
    """

    AGENT_TYPE = "mock"
    _PLAN: dict = {}

    def execute_action(self, action: dict, channel: ActionChannel | None = None) -> dict:
        aid = action["action_id"]
        plan = self._PLAN.get(aid, {})

        ev = plan.get("started_event")
        if ev is not None:
            ev.set()

        if plan.get("emit_progress") and channel is not None:
            channel.emit("progress", {"pct": 50})

        delay = plan.get("delay", 0)
        if delay:
            time.sleep(delay)

        block = plan.get("block_until")
        if block is not None:
            # Poll cancel cooperatively while waiting.
            while not block.is_set():
                if channel is not None and channel.cancelled():
                    plan.setdefault("observed_cancel", []).append(True)
                    break
                time.sleep(0.005)

        if plan.get("error"):
            raise plan["error"]

        return plan.get("result", {"ok": True, "aid": aid})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _stub_audit_and_register_mock(monkeypatch):
    """Wire _MockAgent into agentd's registry and silence SQLite I/O."""
    fake_db = MagicMock()
    monkeypatch.setitem(agentd._AGENT_MAP, "mock", _MockAgent)
    monkeypatch.setattr(agentd, "get_db", lambda: fake_db)
    monkeypatch.setattr(agentd, "log_error", lambda *a, **k: None)
    monkeypatch.setattr(agentd, "log_state_transition", lambda *a, **k: None)
    # Force re-detection of channel support for mock — the cache may be
    # populated from a prior test run if the test order changes.
    agentd._AGENT_CHANNEL_SUPPORT.pop("mock", None)
    yield
    _MockAgent._PLAN = {}
    agentd._AGENT_CHANNEL_SUPPORT.pop("mock", None)


def _make_lifecycle() -> IntentLifecycle:
    lc = IntentLifecycle(intent_id="t-intent")
    lc.transition(IntentState.PARSING)
    return lc


def _action(action_id: str, depends_on=None, on_failure="abort") -> dict:
    return {
        "action_id": action_id,
        "agent": "mock",
        "type": "QUERY",
        "params": {},
        "depends_on": depends_on or [],
        "on_failure": on_failure,
    }


def _goal(*actions: dict) -> dict:
    return {"intent_id": "t-intent", "actions": list(actions)}


def _coord() -> AgentCoordinator:
    return AgentCoordinator(db=MagicMock())


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------

class TestParallelDagSuccess:

    def test_two_independent_actions_run_in_parallel(self):
        """Two independent ~50ms actions should finish in <90ms wall clock."""
        ev_a = threading.Event()
        ev_b = threading.Event()
        _MockAgent._PLAN = {
            "a": {"delay": 0.05, "started_event": ev_a},
            "b": {"delay": 0.05, "started_event": ev_b},
        }
        gs = _goal(_action("a"), _action("b"))

        t0 = time.monotonic()
        results, summary = _coord().execute(gs, _make_lifecycle())
        elapsed = time.monotonic() - t0

        assert results["a"]["ok"] is True
        assert results["b"]["ok"] is True
        assert "Completed 2" in summary
        assert elapsed < 0.09, f"actions ran sequentially: {elapsed:.3f}s"
        assert ev_a.is_set() and ev_b.is_set()

    def test_dependency_chain_executes_in_order(self):
        """B must observe A's result already in the results dict when it runs."""
        a_finished = threading.Event()
        b_saw_a_done = threading.Event()

        class _OrderingAgent(BaseAgent):
            AGENT_TYPE = "mock"

            def execute_action(self, action, channel=None):
                aid = action["action_id"]
                if aid == "a":
                    time.sleep(0.02)
                    a_finished.set()
                    return {"aid": "a"}
                if aid == "b":
                    # If a wasn't finished before b started, ordering broke.
                    if a_finished.is_set():
                        b_saw_a_done.set()
                    return {"aid": "b"}
                return {}

        with patch.dict(agentd._AGENT_MAP, {"mock": _OrderingAgent}):
            agentd._AGENT_CHANNEL_SUPPORT.pop("mock", None)
            gs = _goal(_action("a"), _action("b", depends_on=["a"]))
            results, _ = _coord().execute(gs, _make_lifecycle())

        assert results["a"] == {"aid": "a"}
        assert results["b"] == {"aid": "b"}
        assert b_saw_a_done.is_set(), "b ran before a finished"

    def test_progress_emit_does_not_break_execution(self):
        # Two actions to force the DAG path (sequential path is for <= 1).
        _MockAgent._PLAN = {
            "a": {"emit_progress": True},
            "b": {"emit_progress": True},
        }
        results, _ = _coord().execute(
            _goal(_action("a"), _action("b")), _make_lifecycle())
        assert results["a"]["ok"] is True
        assert results["b"]["ok"] is True


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

class TestDagFailureHandling:

    def test_dependency_failure_cascades_skip(self):
        _MockAgent._PLAN = {
            "a": {"error": MarshalError(
                MarshalErrorCode.FILE_NOT_FOUND, detail="missing")},
            "b": {},  # depends on a
            "c": {},  # depends on b
        }
        gs = _goal(
            _action("a", on_failure="continue"),
            _action("b", depends_on=["a"]),
            _action("c", depends_on=["b"]),
        )
        results, summary = _coord().execute(gs, _make_lifecycle())

        assert "error" in results["a"]
        assert results["b"].get("skipped") is True
        assert results["c"].get("skipped") is True
        assert "3 failed" in summary

    def test_abort_skips_pending_and_signals_inflight_cancel(self):
        """
        Action 'fail' errors with on_failure=abort. Independent action
        'long' is still in flight; the executor must signal cancel and
        the long action observes channel.cancelled() == True.
        """
        block = threading.Event()
        long_started = threading.Event()
        observed: list = []
        _MockAgent._PLAN = {
            "fail": {
                "delay": 0.01,
                "error": MarshalError(
                    MarshalErrorCode.FILE_READ_ERROR, detail="boom"),
            },
            "long": {
                "started_event": long_started,
                "block_until": block,
                "observed_cancel": observed,
            },
            "later": {},  # never runs because abort fires before submit
        }
        gs = _goal(
            _action("fail", on_failure="abort"),
            _action("long", on_failure="continue"),
            _action("later", depends_on=["fail"]),
        )

        # Safety: unblock 'long' after a short delay so the test can't hang
        # if cancellation is broken.
        threading.Timer(0.5, block.set).start()

        results, _ = _coord().execute(gs, _make_lifecycle())

        assert long_started.is_set()
        assert observed == [True], (
            "long action should have observed channel.cancelled() == True")
        assert "error" in results["fail"]
        assert results["later"].get("skipped") is True

    def test_cycle_detection_marks_unreachable(self):
        _MockAgent._PLAN = {"a": {}, "b": {}}
        gs = _goal(
            _action("a", depends_on=["b"]),
            _action("b", depends_on=["a"]),
        )
        results, _ = _coord().execute(gs, _make_lifecycle())
        assert "circular" in results["a"]["error"].lower() or \
               "circular" in results["b"]["error"].lower()
        assert results["a"].get("skipped") is True
        assert results["b"].get("skipped") is True

    def test_unknown_agent_type_yields_error(self):
        _MockAgent._PLAN = {"a": {}, "b": {}}
        bad = _action("b")
        bad["agent"] = "no_such_agent"
        gs = _goal(_action("a"), bad)

        results, _ = _coord().execute(gs, _make_lifecycle())
        assert results["a"]["ok"] is True
        assert "error" in results["b"]


# ---------------------------------------------------------------------------
# Channel-level unit tests
# ---------------------------------------------------------------------------

class TestActionChannel:

    def test_emit_rejects_reserved_kinds(self):
        import queue as _q
        ch = ActionChannel("a", _q.Queue())
        with pytest.raises(ValueError, match="not permitted"):
            ch.emit("done", {})
        with pytest.raises(ValueError, match="not permitted"):
            ch.emit("error", {})

    def test_emit_rejects_unknown_kinds(self):
        import queue as _q
        ch = ActionChannel("a", _q.Queue())
        with pytest.raises(ValueError, match="not permitted"):
            ch.emit("garbage", {})

    def test_cancel_is_observable(self):
        import queue as _q
        ch = ActionChannel("a", _q.Queue())
        assert ch.cancelled() is False
        ch.cancel()
        assert ch.cancelled() is True

    def test_emit_pushes_message_to_outbox(self):
        import queue as _q
        outbox: _q.Queue = _q.Queue()
        ch = ActionChannel("a-1", outbox)
        ch.emit("progress", {"pct": 25})
        msg = outbox.get_nowait()
        assert msg.action_id == "a-1"
        assert msg.kind == "progress"
        assert msg.data == {"pct": 25}

    def test_inbox_recv_and_consume(self):
        import queue as _q
        from agents.channel import ChannelMessage as _CM
        ch = ActionChannel("c", _q.Queue())
        ch._push_inbox(_CM("p", "partial", {"i": 1}))
        ch._push_inbox(_CM("p", "partial", {"i": 2}))
        ch._push_eos()

        items = list(ch.consume())
        assert [m.data for m in items] == [{"i": 1}, {"i": 2}]

    def test_recv_returns_none_on_eos(self):
        import queue as _q
        ch = ActionChannel("c", _q.Queue())
        ch._push_eos()
        assert ch.recv() is None


# ---------------------------------------------------------------------------
# Streaming dependency edges
# ---------------------------------------------------------------------------

class TestStreamingEdges:

    def test_producer_consumer_pipeline(self):
        """B with stream-dep on A should receive every partial A emits."""
        consumed: list = []

        class _ProdCons(BaseAgent):
            AGENT_TYPE = "mock"

            def execute_action(self, action, channel=None):
                aid = action["action_id"]
                if aid == "prod":
                    for i in range(5):
                        channel.emit("partial", {"i": i})
                        time.sleep(0.005)
                    return {"produced": 5}
                if aid == "cons":
                    for msg in channel.consume():
                        consumed.append(msg.data["i"])
                    return {"consumed": list(consumed)}
                return {}

        with patch.dict(agentd._AGENT_MAP, {"mock": _ProdCons}):
            agentd._AGENT_CHANNEL_SUPPORT.pop("mock", None)
            gs = _goal(
                _action("prod"),
                {
                    "action_id": "cons",
                    "agent": "mock",
                    "type": "QUERY",
                    "params": {},
                    "depends_on": [{"id": "prod", "mode": "stream"}],
                    "on_failure": "abort",
                },
            )
            results, _ = _coord().execute(gs, _make_lifecycle())

        assert results["prod"] == {"produced": 5}
        assert results["cons"]["consumed"] == [0, 1, 2, 3, 4]

    def test_late_subscriber_receives_buffered_partials(self):
        """
        If the producer finishes before the consumer is submitted (e.g.
        thread scheduling races), the consumer must still receive every
        partial via the per-parent buffer + EOS at submit time.
        """
        consumed: list = []

        class _RaceAgents(BaseAgent):
            AGENT_TYPE = "mock"

            def execute_action(self, action, channel=None):
                aid = action["action_id"]
                if aid == "fast_prod":
                    # Emit 3 partials immediately and exit before the
                    # consumer's worker has even started.
                    for i in range(3):
                        channel.emit("partial", {"i": i})
                    return {"produced": 3}
                if aid == "slow_cons":
                    # Sleep first, then consume — by the time we read,
                    # the producer has long finished.
                    time.sleep(0.05)
                    for msg in channel.consume():
                        consumed.append(msg.data["i"])
                    return {"consumed": list(consumed)}
                return {}

        with patch.dict(agentd._AGENT_MAP, {"mock": _RaceAgents}):
            agentd._AGENT_CHANNEL_SUPPORT.pop("mock", None)
            # Force prod to be submitted first via a sequential dep — but
            # we want stream semantics, so use a full-dep proxy: actually
            # the simplest race is just two independents where we can't
            # control order. Either way, the buffer must save us.
            gs = _goal(
                _action("fast_prod"),
                {
                    "action_id": "slow_cons",
                    "agent": "mock",
                    "type": "QUERY",
                    "params": {},
                    "depends_on": [
                        {"id": "fast_prod", "mode": "stream"},
                    ],
                    "on_failure": "abort",
                },
            )
            results, _ = _coord().execute(gs, _make_lifecycle())

        assert results["fast_prod"] == {"produced": 3}
        assert results["slow_cons"]["consumed"] == [0, 1, 2]

    def test_stream_dep_failure_eos_propagates(self):
        """
        When a stream parent errors mid-execution, the consumer's inbox
        receives EOS so its consume() loop terminates cleanly. The
        consumer reports whatever it managed to process — the scheduler
        does not fabricate a failure, since the consumer may legitimately
        be useful even with a partial stream.
        """
        cons_started = threading.Event()
        prod_release = threading.Event()
        consumed: list = []

        class _FailingProd(BaseAgent):
            AGENT_TYPE = "mock"

            def execute_action(self, action, channel=None):
                aid = action["action_id"]
                if aid == "prod":
                    # Wait until consumer is definitely running, emit one
                    # partial, then fail. Stream-edge semantics submit
                    # both concurrently, so we want a deterministic order.
                    cons_started.wait(timeout=1.0)
                    channel.emit("partial", {"i": 0})
                    time.sleep(0.01)
                    raise MarshalError(
                        MarshalErrorCode.FILE_READ_ERROR, detail="boom")
                if aid == "cons":
                    cons_started.set()
                    for msg in channel.consume():
                        consumed.append(msg.data["i"])
                    return {"consumed": list(consumed), "saw_eos": True}
                return {}

        with patch.dict(agentd._AGENT_MAP, {"mock": _FailingProd}):
            agentd._AGENT_CHANNEL_SUPPORT.pop("mock", None)
            gs = _goal(
                _action("prod", on_failure="continue"),
                {
                    "action_id": "cons",
                    "agent": "mock",
                    "type": "QUERY",
                    "params": {},
                    "depends_on": [{"id": "prod", "mode": "stream"}],
                    "on_failure": "continue",
                },
            )
            results, _ = _coord().execute(gs, _make_lifecycle())

        del prod_release  # unused, kept for symmetry with other helpers
        assert "error" in results["prod"]
        # Consumer received the one partial that landed before the error,
        # then EOS unblocked consume().
        assert results["cons"]["saw_eos"] is True
        assert results["cons"]["consumed"] == [0]

    def test_stream_dep_string_form_still_works_as_full(self):
        """Bare-string depends_on items must keep their full-dep semantics."""
        order: list = []

        class _Order(BaseAgent):
            AGENT_TYPE = "mock"

            def execute_action(self, action, channel=None):
                order.append(action["action_id"])
                return {"aid": action["action_id"]}

        with patch.dict(agentd._AGENT_MAP, {"mock": _Order}):
            agentd._AGENT_CHANNEL_SUPPORT.pop("mock", None)
            gs = _goal(
                _action("a"),
                _action("b", depends_on=["a"]),
            )
            results, _ = _coord().execute(gs, _make_lifecycle())

        assert order == ["a", "b"]
        assert results["a"]["aid"] == "a"
        assert results["b"]["aid"] == "b"


# ---------------------------------------------------------------------------
# on_channel_message hook
# ---------------------------------------------------------------------------

class TestChannelMessageHook:

    def test_hook_receives_progress_and_partial(self):
        seen: list = []

        class _Emitter(BaseAgent):
            AGENT_TYPE = "mock"

            def execute_action(self, action, channel=None):
                channel.emit("progress", {"pct": 10})
                channel.emit("partial", {"chunk": "x"})
                channel.emit("log", "hi")
                return {"ok": True}

        with patch.dict(agentd._AGENT_MAP, {"mock": _Emitter}):
            agentd._AGENT_CHANNEL_SUPPORT.pop("mock", None)
            coord = AgentCoordinator(
                db=MagicMock(),
                on_channel_message=lambda m: seen.append((m.kind, m.data)),
            )
            # Two actions to force the DAG path.
            gs = _goal(_action("a"), _action("b"))
            coord.execute(gs, _make_lifecycle())

        kinds = [k for (k, _d) in seen]
        assert "progress" in kinds
        assert "partial" in kinds
        assert "log" in kinds
        # Each kind appears at least twice (one per action).
        assert kinds.count("progress") >= 2

    def test_hook_exceptions_do_not_break_execution(self):
        class _Emitter(BaseAgent):
            AGENT_TYPE = "mock"

            def execute_action(self, action, channel=None):
                channel.emit("progress", {"pct": 50})
                return {"ok": True}

        def _bad_hook(_msg):
            raise RuntimeError("hook is broken")

        with patch.dict(agentd._AGENT_MAP, {"mock": _Emitter}):
            agentd._AGENT_CHANNEL_SUPPORT.pop("mock", None)
            coord = AgentCoordinator(
                db=MagicMock(), on_channel_message=_bad_hook)
            gs = _goal(_action("a"), _action("b"))
            results, _ = coord.execute(gs, _make_lifecycle())

        assert results["a"] == {"ok": True}
        assert results["b"] == {"ok": True}
