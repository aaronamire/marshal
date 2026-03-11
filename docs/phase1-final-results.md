Phase 1 eval — 2026-03-11
Regular intents (24 cases):
  Schema validity:  23/24 (95%)  PASS
  Category correct: 23/24 (95%)
  Action type ok:   22/24 (91%)  PASS
  Action ordering:  23/24 (95%)  PASS
  All checks pass:  22/24 (91%)

Not-implemented (5 cases): 5/5 (100%)

PHASE 1 GATE PASSED

Known failures:
  - "read file" — INFERENCE_TIMEOUT (thermal cold-start)
  - "delete tmp files" — model generated [DELETE] only, missing prior QUERY

Post-eval fixes applied:
  - L0 NOT_IMPLEMENTED fast-path for email/system/web/writing
  - NOT_IMPLEMENTED regex false positive on "write X to ~/path" corrected

RAG: deferred — disk quota prevented lancedb install; fails gracefully
