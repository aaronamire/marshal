# Leaves OS — Phase 1 Final Results
Date: 2026-03-11
Model: Qwen2.5-3B-Instruct Q4_K_M
Threads: 2 (Kaby Lake physical cores)
Grammar cap: 3 actions
Layer 0: enabled
Eval cases: 29 (24 file + 5 NOT_IMPLEMENTED)

Schema validity  : 23/24 (95%) PASS
Category correct : 23/24 (95%)
Action type ok   : 22/24 (91%) PASS
Action ordering  : 23/24 (95%) PASS
NOT_IMPLEMENTED  : 5/5  (100%)

*** PHASE 1 GATE PASSED ***

Known failures:
- "read file": INFERENCE_TIMEOUT (90s) — thermal cold-start, same as Phase 0
- "delete tmp files": model generated [DELETE] only, missing prior QUERY

Phase 0 → Phase 1 delta:
- find-then-move: FAIL → PASS (Qwen2.5-3B fixed ordering)
- action_ordering: 80% → 95%
- action_type_ok: 90% → 91%
