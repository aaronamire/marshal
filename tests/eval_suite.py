#!/usr/bin/env python3
"""
Leaves OS Phase 0 eval harness.

Checks three distinct correctness levels for each test case:
  1. Schema validity   — output passes jsonschema (table stakes)
  2. Category correct  — goal_spec["category"] matches expected
  3. Action type ok    — at least one action has an expected type

Usage:
  python3 tests/eval_suite.py            # GBNF on (default)
  python3 tests/eval_suite.py --no-gbnf  # GBNF off (baseline comparison)
  python3 tests/eval_suite.py --timeout 30

Phase 0 passes when:
  schema_validity    >= 95%
  action_type_ok     >= 80%
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.intent_parser import IntentParser
from errors import LeavesError, LeavesErrorCode

# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

@dataclass
class Case:
    label: str
    intent: str
    expected_category: str            # exact string or "" to skip check
    expected_action_types: list[str]  # at least one action must have this type
    expect_not_implemented: bool = False  # True for unimplemented categories

    # Multi-action: minimum number of actions expected
    min_actions: int = 1


CASES: list[Case] = [
    # --- QUERY (find / list / search) ---
    Case(
        label="find PDFs",
        intent="find all PDFs in my Downloads folder",
        expected_category="file_task",
        expected_action_types=["QUERY"],
    ),
    Case(
        label="list Python files",
        intent="list Python files in ~/dev",
        expected_category="file_task",
        expected_action_types=["QUERY"],
    ),
    Case(
        label="find large files",
        intent="find all files larger than 100MB in my home directory",
        expected_category="file_task",
        expected_action_types=["QUERY"],
    ),

    # --- READ (read file contents) ---
    Case(
        label="read file",
        intent="show me the contents of ~/README.md",
        expected_category="file_task",
        expected_action_types=["READ"],
    ),

    # --- MOVE (rename / move) ---
    Case(
        label="rename file",
        intent="rename ~/notes.txt to ~/notes-backup.txt",
        expected_category="file_task",
        expected_action_types=["MOVE"],
    ),
    Case(
        label="move files",
        intent="move all PDFs from ~/Downloads to ~/archive/pdfs",
        expected_category="file_task",
        expected_action_types=["MOVE"],
    ),

    # --- COPY ---
    Case(
        label="copy file",
        intent="copy my config.py to ~/backup/config.py",
        expected_category="file_task",
        expected_action_types=["COPY"],
    ),

    # --- DELETE ---
    Case(
        label="delete tmp files",
        intent="delete all .tmp files in my home directory",
        expected_category="file_task",
        expected_action_types=["DELETE"],
    ),
    Case(
        label="delete by pattern",
        intent="delete all log files in /var/log older than 7 days",
        expected_category="file_task",
        expected_action_types=["DELETE"],
    ),

    # --- Multi-action (QUERY then MOVE) ---
    Case(
        label="find-then-move",
        intent="find all PDFs in my Downloads folder and move them to ~/Documents",
        expected_category="file_task",
        expected_action_types=["QUERY", "MOVE"],
        min_actions=2,
    ),

    # --- NOT_IMPLEMENTED: unimplemented categories ---
    Case(
        label="email (not impl)",
        intent="write an email to my boss about the project update",
        expected_category="email_task",
        expected_action_types=[],
        expect_not_implemented=True,
    ),
    Case(
        label="system (not impl)",
        intent="show me what processes are using the most memory",
        expected_category="system_task",
        expected_action_types=[],
        expect_not_implemented=True,
    ),
]


# ---------------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------------

@dataclass
class Result:
    label: str
    intent: str
    passed_schema: bool = False
    passed_category: bool = False
    passed_action_type: bool = False
    passed_not_impl: bool = False   # for expect_not_implemented cases
    is_not_impl_case: bool = False
    latency_ms: float = 0.0
    error: str = ""
    goal_spec: Optional[dict] = None


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def evaluate(case: Case, parser: IntentParser, timeout_s: int) -> Result:
    result = Result(
        label=case.label,
        intent=case.intent,
        is_not_impl_case=case.expect_not_implemented,
    )

    t0 = time.monotonic()
    try:
        gs = parser.parse(case.intent)
        result.latency_ms = (time.monotonic() - t0) * 1000
        result.goal_spec = gs

        # 1. Schema validity — if parse() returned without raising, jsonschema passed
        result.passed_schema = True

        # 2. Category correctness
        if case.expected_category:
            result.passed_category = (gs.get("category") == case.expected_category)
        else:
            result.passed_category = True

        # 3. Action type correctness
        if not case.expected_action_types:
            # No action type expectation — schema pass counts
            result.passed_action_type = True
        else:
            actual_types = {a.get("type") for a in gs.get("actions", [])}
            # All expected types must appear in at least one action
            result.passed_action_type = all(
                t in actual_types for t in case.expected_action_types
            )
            # Also check min_actions
            if len(gs.get("actions", [])) < case.min_actions:
                result.passed_action_type = False

        # NOT_IMPLEMENTED cases that parsed are failing
        if case.expect_not_implemented:
            result.passed_not_impl = False
            result.passed_schema = False  # parsing was supposed to fail gracefully
            result.error = "Expected NOT_IMPLEMENTED but got a parsed GoalSpec"

    except LeavesError as e:
        result.latency_ms = (time.monotonic() - t0) * 1000
        if case.expect_not_implemented and e.code == LeavesErrorCode.NOT_IMPLEMENTED:
            # Correct: unimplemented category correctly rejected
            result.passed_not_impl = True
            result.passed_schema = True  # the model did produce parseable JSON
            result.passed_category = True
            result.passed_action_type = True
        else:
            result.error = f"[{e.code.value}] {e.user_message}"

    except Exception as e:
        result.latency_ms = (time.monotonic() - t0) * 1000
        result.error = f"{type(e).__name__}: {e}"

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_result(r: Result, case: Case, verbose: bool) -> None:
    if r.is_not_impl_case:
        ok = "PASS" if r.passed_not_impl else "FAIL"
        tag = "not-impl"
    else:
        all_pass = r.passed_schema and r.passed_category and r.passed_action_type
        ok = "PASS" if all_pass else "FAIL"
        checks = []
        if not r.passed_schema:     checks.append("schema")
        if not r.passed_category:   checks.append("category")
        if not r.passed_action_type: checks.append("action_type")
        tag = ",".join(checks) if checks else "ok"

    lat = f"{r.latency_ms:.0f}ms"
    print(f"  [{ok}] {r.label:<20s} {tag:<20s} {lat}")

    if r.error:
        print(f"         error: {r.error}")
    elif verbose and r.goal_spec:
        actions = r.goal_spec.get("actions", [])
        types = " + ".join(a.get("type", "?") for a in actions)
        cat = r.goal_spec.get("category", "?")
        conf = r.goal_spec.get("metadata", {}).get("confidence", 0)
        print(f"         {cat}  [{types}]  conf={conf:.0%}")


def print_summary(results: list[Result], cases: list[Case], use_gbnf: bool) -> int:
    mode = "GBNF ON" if use_gbnf else "GBNF OFF (baseline)"
    print(f"\n{'='*60}")
    print(f"Results — {mode}")
    print(f"{'='*60}")

    # Separate regular vs not-impl cases
    regular = [(r, c) for r, c in zip(results, cases) if not c.expect_not_implemented]
    not_impl = [(r, c) for r, c in zip(results, cases) if c.expect_not_implemented]

    n_reg = len(regular)
    schema_pass     = sum(1 for r, _ in regular if r.passed_schema)
    category_pass   = sum(1 for r, _ in regular if r.passed_category)
    action_pass     = sum(1 for r, _ in regular if r.passed_action_type)
    all_pass        = sum(1 for r, _ in regular if r.passed_schema and r.passed_category and r.passed_action_type)
    not_impl_pass   = sum(1 for r, _ in not_impl if r.passed_not_impl)

    def pct(n, d): return f"{n}/{d} ({100*n//d if d else 0}%)"

    print(f"\n  Regular intents ({n_reg} cases):")
    print(f"    Schema validity :  {pct(schema_pass, n_reg)}")
    print(f"    Category correct:  {pct(category_pass, n_reg)}")
    print(f"    Action type ok  :  {pct(action_pass, n_reg)}")
    print(f"    All checks pass :  {pct(all_pass, n_reg)}")

    if not_impl:
        print(f"\n  Not-implemented intents ({len(not_impl)} cases):")
        print(f"    Correctly rejected: {pct(not_impl_pass, len(not_impl))}")

    # Phase 0 gate
    schema_pct  = 100 * schema_pass  // n_reg if n_reg else 0
    action_pct  = 100 * action_pass  // n_reg if n_reg else 0
    schema_gate = schema_pct >= 95
    action_gate = action_pct >= 80

    print(f"\n  Phase 0 gate:")
    print(f"    schema_validity >= 95%  : {'PASS' if schema_gate else 'FAIL'} ({schema_pct}%)")
    print(f"    action_type_ok  >= 80%  : {'PASS' if action_gate else 'FAIL'} ({action_pct}%)")

    if schema_gate and action_gate:
        print(f"\n  *** PHASE 0 COMPLETE — eval harness passed ***")
        return 0
    else:
        print(f"\n  Phase 0 not complete yet.")
        return 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Leaves OS Phase 0 eval harness")
    parser.add_argument("--no-gbnf", action="store_true",
                        help="Disable GBNF grammar (baseline comparison)")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Per-intent timeout in seconds (default: 60)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show category/action details for each result")
    parser.add_argument("--case", type=str, default=None,
                        help="Run only cases whose label contains this string")
    args = parser.parse_args()

    use_gbnf = not args.no_gbnf
    mode = "GBNF ON" if use_gbnf else "GBNF OFF"
    print(f"Leaves OS eval harness — {mode}")
    print(f"Timeout: {args.timeout}s per intent\n")

    intent_parser = IntentParser(use_gbnf=use_gbnf)

    if not intent_parser._client.is_available():
        print("ERROR: Inference server not running. Start with: bash scripts/start-inference.sh")
        return 1

    cases = CASES
    if args.case:
        cases = [c for c in CASES if args.case.lower() in c.label.lower()]
        if not cases:
            print(f"No cases matching '{args.case}'")
            return 1

    results = []
    print(f"Running {len(cases)} test cases...\n")

    for case in cases:
        r = evaluate(case, intent_parser, args.timeout)
        results.append(r)
        print_result(r, case, args.verbose)

    return print_summary(results, cases, use_gbnf)


if __name__ == "__main__":
    sys.exit(main())
