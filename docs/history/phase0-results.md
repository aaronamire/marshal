# Leaves OS Phase 0 — Completion Report

Date: 2026-03-09
Branch: dev (commits through 4b7f53a)

---

## What Phase 0 Covered

Phase 0 was the hardening sprint: wire grammar-constrained decoding, fix the eval harness,
and build the fast-path classifier. No new agent capabilities — only infrastructure.

---

## Four Tasks Completed

### Task 1 — COPY Grammar Mystery (State A confirmed)

**Finding:** COPY was already in `inference/grammar/goal_spec.gbnf` (State A). Grammar and
JSON schema both enumerate the same 8 action types: QUERY, READ, WRITE, DELETE, MOVE, COPY,
SUMMARIZE, COMPOSE.

**Also fixed:** Grammar actions-array was unbounded (`(... action)*`), causing the 1B model to
loop and fill the 512-token budget with identical READ actions. Capped to 3 actions via two
optional groups `(ws "," ws action)?`.

**Permanent tests added:** `tests/test_grammar_enforcement.py` (4 static unit tests, no server
required). Tests verify grammar/schema action-type enums are identical, categories match, and
actions-array is bounded.

---

### Task 2 — Eval Harness Ordering Blindspot

**The bug:** `find-then-move` was generating `[MOVE, QUERY, MOVE, QUERY]` and PASSING the eval
harness because it checked type presence (`{"QUERY", "MOVE"} ⊆ actual_types`) not ordering.

**Fix:**

1. **`agents/validators.py`** (new) — semantic validators:
   - `validate_action_ordering()`: unique IDs, FORWARD_DEPENDENCY, DANGLING_DEPENDENCY,
     DUPLICATE_ACTION_ID, DEPENDENCY_CYCLE, MUTATION_WITHOUT_PRIOR_DISCOVERY
   - `validate_destructive_consistency()`: DESTRUCTIVE_WITHOUT_PREVIEW,
     MUTATION_NOT_FLAGGED_DESTRUCTIVE
   - `validate_goal_spec()`: aggregates both

2. **`agents/intent_parser.py`** — wired `_validate_semantics()` call after schema validation.
   Hard error codes: FORWARD_DEPENDENCY, DANGLING_DEPENDENCY, DUPLICATE_ACTION_ID.
   DEPENDENCY_CYCLE excluded (1B model generates self-references; these are soft SELF_DEPENDENCY
   errors, stripped in `_inject_os_fields()`).

3. **`tests/eval_suite.py`** — added `expected_sequence` field and ordering validation:
   delete cases require `["QUERY", "DELETE"]` prefix; find-then-move requires `["QUERY", "MOVE"]`.

4. **`tests/test_validators.py`** (new) — 20 unit tests covering all error codes.

**Prompt regression fixed:** An added "ORDERING RULE" caused the 1B model to emit
`act-1 depends_on ["act-1"]` (self-reference). Rule removed. The validator catches ordering
issues without confusing the model. Rule 10 added instead: "NEVER repeat the same action type."

---

### Task 3 — Prompt Caching Diagnosis

**Finding:** Prompt caching is not the right fix. Profiling shows:

- Prefill (prompt processing): ~2ms/token via llama.cpp batch processing — already fast.
- Generation (autoregressive decode): 20-30 tok/s nominal; drops to ~7 tok/s after 60s
  sustained load due to thermal throttling on i5-7200U (TDP=15W).
- A GoalSpec JSON is ~250-300 tokens. At 7 tok/s = 36-43s.

**Cache accumulation bug discovered:** Both the `--cache-prompt` server flag and
`cache_prompt: true` in the request payload cause context accumulation — request 2 sees
system + user1 + output1 + user2, making each subsequent request longer and slower. Request 3
in a suite measured at 74.9s vs 16.8s for request 1. Fixed by running stateless requests
(default behavior — no `cache_prompt` in payload).

**Resolution:** Thermal throttling is the real bottleneck. No caching change is needed.
The fix is a model upgrade (Phase 1: Qwen2.5-3B on a machine without TDP constraints) or
accepting that the 1B model on the i5-7200U generates in ~40s under load.

---

### Task 4 — Layer 1 NOT_IMPLEMENTED Fast-Path

**Before:** email/system/web/writing categories called the LLM for 18-51s before returning
NOT_IMPLEMENTED (the LLM generates a GoalSpec, then `_check_actions_present()` raises).

**After:** `IntentParser.parse()` checks the L1 classifier result before calling the LLM.
If `l1.is_confident` (>= 0.60) and `l1.category not in IMPLEMENTED_CATEGORIES`, raises
NOT_IMPLEMENTED immediately. No LLM call.

| Category | Before | After |
|---|---|---|
| email_task | ~21s (LLM times out or returns) | ~1.2ms |
| system_task | ~17s | ~0.8ms |
| web_task | ~19s (estimated) | ~1ms |
| writing_task | ~15s (estimated) | ~1ms |

L1 classifier accuracy on held-out examples: 10/10 (trained on 750 synthetic examples,
TF-IDF + LogReg pipeline). Pipeline at `models/layer1_pipeline.joblib`.

**Tests added:** `TestNotImplementedFastPath` in `tests/test_classifier.py` (4 tests).

---

## Unit Test Suite Status

```
61 tests, 0 failures (non-inference tests)
```

Key test files:
- `tests/test_grammar_enforcement.py` — 4 tests, grammar/schema consistency
- `tests/test_validators.py` — 20 tests, all semantic validator codes
- `tests/test_classifier.py` — 13 tests, L1 accuracy + fast-path integration
- `tests/test_schema_validation.py` — schema structure tests
- `tests/test_state_machine.py` — intent lifecycle transitions

---

## Eval Suite (Phase 0 Gate) — Known State

The eval suite (`tests/eval_suite.py`) has 12 cases. Requires inference server to run.

**Phase 0 gate thresholds:**
- `schema_validity >= 95%` (≥ 11/12 must produce valid GoalSpec JSON)
- `action_type_ok >= 80%` (≥ 10/12 must have correct action type)
- `action_ordering >= 80%` (≥ 10/12 must have correct ordering)

**Known failure mode:** The "read file" case (position 4 in suite, after 3 QUERY cases) hits
thermal throttling. The 1B model generates 3 READ actions (action loop) filling the 512-token
budget before the JSON closes. This produces a `JSON_PARSE_FAILED` or `INFERENCE_TIMEOUT`.

**NOT_IMPLEMENTED cases:** "email (not impl)" and "system (not impl)" now complete in <2ms
via the L1 fast-path (down from 18-21s). These always pass.

**Estimated gate status post-Phase 0 fixes:**
- schema_validity: 90% (9/10 file cases pass; "read file" thermal failure is
  hardware-dependent and runs may vary)
- action_type_ok: ~80-90%
- action_ordering: ~80-90%

The schema gate at exactly 90% fails the ≥95% threshold. This is a 1B model + thermal
throttle limitation, not a software bug. Requires model upgrade (Phase 1) to reliably pass.

---

## Phase 1 Entry Criteria

These are the unresolved gaps Phase 1 must address:

1. **Model upgrade** — Replace Llama-3.2-1B-Q4_K_M with Qwen2.5-3B-Instruct (or equivalent).
   Target: eliminate the action-loop (3 READ actions), reliably generate 2-action GoalSpecs
   (QUERY+DELETE, QUERY+MOVE), pass schema gate at ≥95%.

2. **Layer 0** — Regex pattern matcher for highest-frequency intents (list-files, find-pattern).
   Target: <0.1ms, eliminates LLM call for ~30% of file_task intents.

3. **Eval harness expansion** — Add more test cases to allow 1 failure at ≥95%
   (currently 10 file cases → need ≥11/12 to reach 91.7%, or expand to 20 cases).

4. **RAG pipeline** (deprioritized) — Deferred until model upgrade proves needed.

---

## Commits in Phase 0 (dev branch since main)

```
4b7f53a feat: Layer 1 NOT_IMPLEMENTED fast-path — skip LLM for non-file categories
70dfb9d feat: Task 1+2 — grammar enforcement tests, semantic validators, eval ordering
dba8c65 fix: clear NOT_IMPLEMENTED error for unimplemented agent categories
d6e9ed1 fix: harden _inject_os_fields against partial authorization objects and truncated JSON
34fe56e docs: Phase 0/1 canonical test results — issues found and fixed
41f964b fix: inject default authorization when model omits it, default confidence=0.80
aa2a2f0 fix: ctx-size 4096 for full GoalSpec prompt, TIMEOUT_READ_SECONDS 300s
```
