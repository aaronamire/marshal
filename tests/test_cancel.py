"""
Tests for cooperative cancellation in AgentCoordinator.

These tests cover the in-process side of the cancel pipeline:

  - threading.Event passed to AgentCoordinator → DAG scheduler aborts
    promptly when the event is set mid-flight.
  - The sequential path (single-action intents) checks the event
    between actions.

The signal-handler side (sandboxed_runner SIGUSR1 → cancel_event) and
the agentd socket-side (_cancel message → SIGUSR1 to inflight proc)
are exercised end-to-end via the existing socket integration tests
when those are run, but pure-cancel here keeps the unit suite fast.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import agentd  # noqa: E402
from agentd import AgentCoordinator  # noqa: E402
from agents.base_agent import BaseAgent  # noqa: E402
from agents.channel import ActionChannel  # noqa: E402
from agents.state_machine import IntentLifecycle, IntentState  # noqa: E402


class _SlowAgent(BaseAgent):
    AGENT_TYPE = "mock"
    _PLAN: dict = {}

    def execute_action(self, action: dict, channel: ActionChannel | None = None) -> dict:
        aid = action["action_id"]
        plan = self._PLAN.get(aid, {})
        started = plan.get("started_event")
        if started is not None:
            started.set()
        # Cooperatively poll cancellation while sleeping.
        deadline = time.monotonic() + plan.get("max_wait", 5.0)
        while time.monotonic() < deadline:
            if channel is not None and channel.cancelled():
                plan.setdefault("observed_cancel", []).append(True)
                return {"cancelled": True, "aid": aid}
            time.sleep(0.01)
        return {"ok": True, "aid": aid}


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    fake_db = MagicMock()
    monkeypatch.setitem(agentd._AGENT_MAP, "mock", _SlowAgent)
    monkeypatch.setattr(agentd, "get_db", lambda: fake_db)
    monkeypatch.setattr(agentd, "log_error", lambda *a, **k: None)
    monkeypatch.setattr(agentd, "log_state_transition", lambda *a, **k: None)
    agentd._AGENT_CHANNEL_SUPPORT.pop("mock", None)
    yield
    _SlowAgent._PLAN = {}
    agentd._AGENT_CHANNEL_SUPPORT.pop("mock", None)


def _make_lifecycle() -> IntentLifecycle:
    lc = IntentLifecycle(intent_id="t-cancel")
    lc.transition(IntentState.PARSING)
    return lc


def _action(aid: str, depends_on=None) -> dict:
    return {
        "action_id": aid,
        "agent": "mock",
        "type": "QUERY",
        "params": {},
        "depends_on": depends_on or [],
        "on_failure": "abort",
    }


def _goal(*actions) -> dict:
    return {"intent_id": "t-cancel", "actions": list(actions)}


# ---------------------------------------------------------------------------
# DAG path
# ---------------------------------------------------------------------------

def test_dag_aborts_when_cancel_event_set_midflight():
    """
    Two parallel slow actions; the cancel event fires after both have
    started. Both should observe the cancellation through their channels
    and the scheduler should return promptly.
    """
    started_a = threading.Event()
    started_b = threading.Event()
    _SlowAgent._PLAN = {
        "a": {"started_event": started_a, "max_wait": 5.0},
        "b": {"started_event": started_b, "max_wait": 5.0},
    }
    cancel_evt = threading.Event()
    coord = AgentCoordinator(db=MagicMock(), cancel_event=cancel_evt)
    gs = _goal(_action("a"), _action("b"))

    timer = threading.Timer(0.1, cancel_evt.set)
    timer.start()
    try:
        t0 = time.monotonic()
        results, _summary = coord.execute(gs, _make_lifecycle())
        elapsed = time.monotonic() - t0
    finally:
        timer.cancel()

    # Cancel should land in well under the 5s slow-path budget.
    assert elapsed < 2.0, f"cancel did not abort: {elapsed:.2f}s"
    # Both actions started; both observed the cancellation.
    assert started_a.is_set() and started_b.is_set()
    obs_a = _SlowAgent._PLAN["a"].get("observed_cancel", [])
    obs_b = _SlowAgent._PLAN["b"].get("observed_cancel", [])
    assert obs_a == [True]
    assert obs_b == [True]
    assert results["a"] == {"cancelled": True, "aid": "a"}
    assert results["b"] == {"cancelled": True, "aid": "b"}


def test_dag_skips_pending_actions_when_cancelled_early():
    """
    A → B chain. Cancel fires while A is running. B should never start;
    it must be marked skipped in results.
    """
    started_a = threading.Event()
    _SlowAgent._PLAN = {
        "a": {"started_event": started_a, "max_wait": 5.0},
        "b": {"max_wait": 5.0},
    }
    cancel_evt = threading.Event()
    coord = AgentCoordinator(db=MagicMock(), cancel_event=cancel_evt)
    gs = _goal(_action("a"), _action("b", depends_on=["a"]))

    threading.Timer(0.1, cancel_evt.set).start()
    results, _summary = coord.execute(gs, _make_lifecycle())

    assert started_a.is_set()
    # B never ran — marked skipped by the abort-cascade path.
    assert "skipped" in results.get("b", {}) or "error" in results.get("b", {})
    assert results["b"].get("skipped") is True


def test_dag_without_cancel_event_runs_normally():
    """When no cancel_event is wired, the loop must remain unaffected."""
    _SlowAgent._PLAN = {"a": {"max_wait": 0.05}, "b": {"max_wait": 0.05}}
    coord = AgentCoordinator(db=MagicMock())  # no cancel_event
    results, _summary = coord.execute(
        _goal(_action("a"), _action("b")), _make_lifecycle())
    assert results["a"] == {"ok": True, "aid": "a"}
    assert results["b"] == {"ok": True, "aid": "b"}


# ---------------------------------------------------------------------------
# Sequential path
# ---------------------------------------------------------------------------

def test_sequential_path_unaffected_when_cancel_not_set():
    """The sequential branch (single-action) runs to completion when no cancel."""
    _SlowAgent._PLAN = {"a": {"max_wait": 0.02}}
    cancel_evt = threading.Event()
    coord = AgentCoordinator(db=MagicMock(), cancel_event=cancel_evt)
    results, _summary = coord.execute(_goal(_action("a")), _make_lifecycle())
    assert results["a"] == {"ok": True, "aid": "a"}


# ---------------------------------------------------------------------------
# Cancel module
# ---------------------------------------------------------------------------

def test_install_handler_idempotent_and_event_settable():
    """Smoke test the cancel module API."""
    from agents import cancel as cancel_mod
    cancel_mod.install_handler()
    cancel_mod.install_handler()  # twice OK
    cancel_mod.cancel_event.clear()
    assert cancel_mod.is_cancelled() is False
    cancel_mod.cancel_event.set()
    assert cancel_mod.is_cancelled() is True
    cancel_mod.cancel_event.clear()
