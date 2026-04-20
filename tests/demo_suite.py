"""
Demo suite — guaranteed-working intents with asserted p95 latency budgets.

Two modes, controlled by MARSHAL_DEMO_MODE:

  stub  (default in CI)
        Only intents that resolve via Layer 0 (regex). No inference server
        required. Latency budgets are sub-millisecond. This is what runs on
        every push and gates the CI build.

  full  (run locally before recording the demo)
        All stub cases plus the L2 intents that exercise the full pipeline
        (sklearn classifier + llama.cpp inference). Requires a running
        llama-server. Latency budgets are seconds, not milliseconds.

Each case is run N times and the p95 latency is checked against a budget.
A failure here means the demo will look slow or wrong on stage — we treat
that the same as a broken feature.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.intent_parser import IntentParser
from inference.client import InferenceClient


DEMO_MODE = os.environ.get("MARSHAL_DEMO_MODE", "stub").lower()
RUNS_PER_CASE = int(os.environ.get("MARSHAL_DEMO_RUNS", "10" if DEMO_MODE == "stub" else "3"))


@dataclass(frozen=True)
class DemoCase:
    label: str
    intent: str
    expected_category: str
    expected_action_type: str
    budget_ms: float          # p95 must be <= this
    requires_l2: bool = False  # if True, skipped in stub mode


# ---------------------------------------------------------------------------
# L0 cases — no inference server required. Rock-solid intents we can demo on
# any laptop, anywhere.
#
# The budget here is end-to-end parse time, not raw regex time. Even when L0
# short-circuits, the parser still runs jsonschema validation and semantic
# checks (~3-5ms locally). 15ms gives 3-5x headroom for shared CI runners.
# ---------------------------------------------------------------------------

L0_BUDGET_MS = 15.0

L0_CASES: list[DemoCase] = [
    DemoCase(
        label="file-query-glob",
        intent="find ~/Downloads/*.pdf",
        expected_category="file_task",
        expected_action_type="QUERY",
        budget_ms=L0_BUDGET_MS,
    ),
    DemoCase(
        label="file-query-list",
        intent="list ~/dev",
        expected_category="file_task",
        expected_action_type="QUERY",
        budget_ms=L0_BUDGET_MS,
    ),
    DemoCase(
        label="file-read",
        intent="read ~/README.md",
        expected_category="file_task",
        expected_action_type="READ",
        budget_ms=L0_BUDGET_MS,
    ),
    DemoCase(
        label="system-cpu",
        intent="what's my CPU usage",
        expected_category="system_task",
        expected_action_type="QUERY",
        budget_ms=L0_BUDGET_MS,
    ),
    DemoCase(
        label="system-memory",
        intent="how much memory is free",
        expected_category="system_task",
        expected_action_type="QUERY",
        budget_ms=L0_BUDGET_MS,
    ),
    DemoCase(
        label="system-launch",
        intent="open firefox",
        expected_category="system_task",
        expected_action_type="WRITE",
        budget_ms=L0_BUDGET_MS,
    ),
    DemoCase(
        label="power-battery",
        intent="battery status",
        expected_category="power_task",
        expected_action_type="QUERY",
        budget_ms=L0_BUDGET_MS,
    ),
]


# ---------------------------------------------------------------------------
# L2 cases — require llama-server. Skipped in stub mode.
# Budgets are wall-clock-honest and assume a warm KV cache (agentd warms it
# at boot). Cold-start runs will exceed these — that is acceptable because
# the demo is always rehearsed with the daemon already up.
# ---------------------------------------------------------------------------

L2_CASES: list[DemoCase] = [
    DemoCase(
        label="file-organize",
        intent="move all PDFs from ~/Downloads to ~/Documents",
        expected_category="file_task",
        expected_action_type="MOVE",
        budget_ms=8000.0,
        requires_l2=True,
    ),
    DemoCase(
        label="file-cleanup",
        intent="delete all .tmp files in /tmp",
        expected_category="file_task",
        expected_action_type="DELETE",
        budget_ms=8000.0,
        requires_l2=True,
    ),
]


def _select_cases() -> list[DemoCase]:
    if DEMO_MODE == "stub":
        return list(L0_CASES)
    return list(L0_CASES) + list(L2_CASES)


CASES = _select_cases()


# ---------------------------------------------------------------------------
# Module-level parser — instantiated once, reused across cases.
# In stub mode we disable RAG and the sklearn classifier so the test has
# zero external dependencies (no LanceDB, no joblib model, no inference).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def parser() -> IntentParser:
    if DEMO_MODE == "stub":
        return IntentParser(classifier=None, use_gbnf=False, use_rag=False)
    return IntentParser()


@pytest.fixture(scope="module", autouse=True)
def _check_inference_available(parser: IntentParser) -> None:
    """In full mode, skip the whole module if no inference server is reachable."""
    if DEMO_MODE == "stub":
        return
    client: InferenceClient = parser._client
    if not client.is_available():
        pytest.skip("MARSHAL_DEMO_MODE=full but no inference server reachable")


# ---------------------------------------------------------------------------
# The actual tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
def test_demo_intent(parser: IntentParser, case: DemoCase) -> None:
    """
    Each demo intent must:
      1. Parse to the expected category.
      2. Emit at least one action of the expected type.
      3. Hit its p95 latency budget over RUNS_PER_CASE runs.
    """
    latencies_ms: list[float] = []
    last_spec: dict | None = None

    for _ in range(RUNS_PER_CASE):
        t0 = time.monotonic()
        spec = parser.parse(case.intent)
        latencies_ms.append((time.monotonic() - t0) * 1000)
        last_spec = spec

    assert last_spec is not None
    assert last_spec["category"] == case.expected_category, (
        f"category mismatch for {case.label!r}: "
        f"got {last_spec['category']!r}, expected {case.expected_category!r}"
    )

    actual_types = [a.get("type") for a in last_spec.get("actions", [])]
    assert case.expected_action_type in actual_types, (
        f"action type mismatch for {case.label!r}: "
        f"got {actual_types}, expected to contain {case.expected_action_type!r}"
    )

    p95 = _percentile(latencies_ms, 95)
    assert p95 <= case.budget_ms, (
        f"p95 latency budget exceeded for {case.label!r}: "
        f"{p95:.2f}ms > {case.budget_ms:.2f}ms "
        f"(samples: {[f'{x:.2f}' for x in latencies_ms]})"
    )


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, matching numpy's default."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def test_demo_mode_advertised() -> None:
    """Sanity: print which mode ran so CI logs make the choice obvious."""
    print(f"\nMARSHAL_DEMO_MODE={DEMO_MODE}  cases={len(CASES)}  runs/case={RUNS_PER_CASE}")
    assert DEMO_MODE in ("stub", "full"), f"unknown MARSHAL_DEMO_MODE: {DEMO_MODE!r}"


# Allow running directly: `python tests/demo_suite.py`
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
