#!/usr/bin/env python3
"""
Benchmark warm vs cold runner-path latency.

Fires N identical "list .py files in cwd" intents through agentd's Unix
socket and records end-to-end wall time. Run twice — once with the
runner-pool socket present (warm path) and once with it removed (cold
fallback) — and compare p50/p95/min.

Usage:
    # Warm (default — runner-pool master must be up via agentd):
    python scripts/bench_runner_paths.py --n 20

    # Force cold by removing the pool socket before each request:
    python scripts/bench_runner_paths.py --n 20 --force-cold

    # Compare both in one invocation:
    python scripts/bench_runner_paths.py --n 20 --compare

The script avoids the inference path entirely: it submits a pre-built
GoalSpec directly so the timing reflects sandbox+executor overhead, not
LLM latency. That isolates exactly what the warm-pool change improves.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import statistics
import sys
import time
import uuid

# Make `db.audit` importable when run from any cwd.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_AGENTD_SOCK = pathlib.Path.home() / ".leaves" / "agentd.sock"
_POOL_SOCK = pathlib.Path.home() / ".leaves" / "runner-pool.sock"


def _make_goalspec() -> dict:
    return {
        "intent_id": str(uuid.uuid4()),
        "natural_text": "list .py files in cwd",
        "category": "file_task",
        "actions": [{
            "action_id": "act-1",
            "type": "QUERY",
            "agent": "file",
            "params": {
                "path": str(pathlib.Path.cwd()),
                "pattern": "*.py",
                "recursive": False,
            },
            "destructive": False,
            "depends_on": [],
        }],
        "authorization": {
            "resources": [str(pathlib.Path.cwd())],
            "preview_required": False,
            "reversible": True,
        },
        "metadata": {
            "confidence": 1.0,
            "parse_latency_ms": 0.0,
            "model": "bench-harness",
        },
    }


async def _one_request() -> tuple[float, bool]:
    """Send one intent, return (elapsed_seconds, ok)."""
    spec = _make_goalspec()

    # Pre-create the intents row so the worker's audit-log inserts (which
    # FK back to intents.intent_id) don't blow up. Mirrors the path
    # leaves.py / api.server take before submitting to agentd.
    from db.audit import get_db, log_intent_created
    db = get_db()
    log_intent_created(db, spec["intent_id"], spec["natural_text"], spec)

    payload = (json.dumps({"goal_spec": spec, "from_state": "PARSING"})
               .encode() + b"\n")

    t0 = time.perf_counter()
    reader, writer = await asyncio.open_unix_connection(str(_AGENTD_SOCK))
    try:
        writer.write(payload)
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=60.0)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    elapsed = time.perf_counter() - t0

    if not line:
        return elapsed, False
    try:
        resp = json.loads(line)
    except json.JSONDecodeError:
        return elapsed, False
    return elapsed, bool(resp.get("ok"))


async def _run_n(n: int, label: str) -> list[float]:
    print(f"\n=== {label} ({n} requests) ===")
    times: list[float] = []
    for i in range(n):
        try:
            elapsed, ok = await _one_request()
        except Exception as e:
            print(f"  [{i+1}/{n}] error: {e}")
            continue
        status = "ok" if ok else "FAIL"
        times.append(elapsed)
        print(f"  [{i+1}/{n}] {elapsed*1000:7.1f}ms  {status}")
    return times


def _print_stats(label: str, times: list[float]) -> None:
    if not times:
        print(f"\n{label}: no successful samples")
        return
    ms = [t * 1000 for t in times]
    ms.sort()
    p50 = statistics.median(ms)
    p95 = ms[int(len(ms) * 0.95)] if len(ms) > 1 else ms[0]
    print(f"\n{label} (n={len(ms)}):")
    print(f"  min:  {min(ms):7.1f}ms")
    print(f"  p50:  {p50:7.1f}ms")
    print(f"  p95:  {p95:7.1f}ms")
    print(f"  max:  {max(ms):7.1f}ms")
    print(f"  mean: {statistics.mean(ms):7.1f}ms")


async def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20,
                    help="number of requests per path (default: 20)")
    ap.add_argument("--force-cold", action="store_true",
                    help="hide the pool socket before each request")
    ap.add_argument("--compare", action="store_true",
                    help="run warm then cold and print a comparison table")
    args = ap.parse_args()

    if not _AGENTD_SOCK.exists():
        print(f"agentd socket not found at {_AGENTD_SOCK}; start agentd first",
              file=sys.stderr)
        return 1

    # Warm-up: one call so the in-process bits (DB, watchers) are settled.
    try:
        await _one_request()
    except Exception:
        pass

    if args.compare:
        if not _POOL_SOCK.exists():
            print("warning: pool socket missing — warm path will fall back",
                  file=sys.stderr)
        warm = await _run_n(args.n, "WARM (runner-pool present)")

        # Force cold by renaming the pool socket aside for the duration.
        moved = False
        backup = _POOL_SOCK.with_suffix(".sock.bench-disabled")
        if _POOL_SOCK.exists():
            try:
                os.rename(_POOL_SOCK, backup)
                moved = True
            except OSError as e:
                print(f"could not move pool socket: {e}", file=sys.stderr)
        try:
            cold = await _run_n(args.n, "COLD (runner-pool hidden)")
        finally:
            if moved:
                try:
                    os.rename(backup, _POOL_SOCK)
                except OSError:
                    pass

        _print_stats("WARM", warm)
        _print_stats("COLD", cold)

        if warm and cold:
            warm_p50 = statistics.median([t * 1000 for t in warm])
            cold_p50 = statistics.median([t * 1000 for t in cold])
            delta = cold_p50 - warm_p50
            speedup = (cold_p50 / warm_p50) if warm_p50 > 0 else 0
            print(
                f"\nSpeedup (cold p50 / warm p50): "
                f"{speedup:.2f}×  (saved {delta:.1f}ms per intent)"
            )
        return 0

    label = "COLD (forced)" if args.force_cold else "warm/cold (auto)"
    if args.force_cold and _POOL_SOCK.exists():
        backup = _POOL_SOCK.with_suffix(".sock.bench-disabled")
        try:
            os.rename(_POOL_SOCK, backup)
        except OSError:
            pass
        try:
            times = await _run_n(args.n, label)
        finally:
            try:
                os.rename(backup, _POOL_SOCK)
            except OSError:
                pass
    else:
        times = await _run_n(args.n, label)

    _print_stats(label, times)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
