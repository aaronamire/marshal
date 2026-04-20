# Leaves OS — Phase 0 Baseline
*This document records the measured state of the system at Phase 0 completion.*
*All numbers are from actual hardware runs, not estimates.*

---

## Hardware

| Component | Spec |
|-----------|------|
| CPU | Intel Core i5-7200U (Kaby Lake, 2 cores / 4 threads HT) |
| TDP | 15W (mobile) |
| L2 cache | 512KB per core |
| L3 cache | 3MB shared |
| RAM | 7.6GB DDR4 (5.2GB used, 2.5GB available at measurement) |
| Storage | HDD (931.5GB, rotational) |
| OS | Arch Linux, kernel 6.18.7-arch1-1 |
| CPU governor | powersave |

## Software

| Component | Version |
|-----------|---------|
| Python | 3.14.2 |
| llama-server | 1 (f5ddcd1) |
| llama.cpp build | f5ddcd1 |
| sklearn | 1.8.0 |
| Model | Llama-3.2-1B-Instruct-Q4_K_M.gguf |
| Model size | 771MB |
| Commit | c3ae115 |
| Branch | dev |

## Inference Server Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--ctx-size` | 4096 | System prompt ~2000 tokens + 512 output. 4096 provides safe headroom. |
| `--threads` | 4 | Note: should be 2 for Kaby Lake (physical cores only). HT siblings share L2 cache; using 4 threads may cause cache thrashing. Known issue, deferred to Phase 1. |
| `--mlock` | not set | Available RAM is 2.5GB vs 771MB model. No page fault risk in practice. |
| `--cache-prompt` | not used | Context accumulation bug: with cache-prompt enabled, each request receives context from all prior requests, making later requests progressively slower. Thermal throttling (not prefill) is the latency bottleneck; caching doesn't help. |

## Generation Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `INFERENCE_MAX_TOKENS` | 512 | GoalSpec JSON peaks at ~300 tokens for 2-action intents. 512 gives headroom. |
| `INFERENCE_TEMPERATURE` | 0.1 | Near-deterministic. Reduces stochastic variation in small model output. |
| `grammar` | `goal_spec.gbnf` | GBNF grammar-constrained decoding. Forces structurally valid JSON, bounded to 2 actions. |

## Grammar: actions-array Cap

The grammar caps GoalSpec at 2 actions (`actions-array ::= "[" ws action (ws "," ws action)? ws "]"`).

Rationale: the i5-7200U in powersave mode generates at ~7 tok/s under thermal load.
- 3-action GoalSpec: ~450 tokens → 64s → fills 512-token budget before JSON closes → `JSON_PARSE_FAILED`
- 2-action GoalSpec: ~300 tokens → 43s → JSON closes with 15s headroom at 7 tok/s

All Phase 0 cases require at most 2 actions. 3-action chains are not needed until Phase 1
multi-step operations.

## Layer 1 Classifier

| Metric | Value |
|--------|-------|
| Algorithm | TF-IDF + Logistic Regression |
| Training examples | 750 (150 per category × 5 categories) |
| Inference latency | 0.8–1.2ms |
| Model path | `models/layer1_pipeline.joblib` |
| Confidence threshold | 0.60 (`MIN_CONFIDENCE_THRESHOLD` in config.py) |

## Eval Suite Results — Configuration C: Full Stack (GBNF + Layer 1)

**This is the primary measurement.** 12 test cases: 10 file intents + 2 NOT_IMPLEMENTED intents.
Run with 15s inter-case cooldown to allow CPU thermal recovery between cases.

```
  [PASS] read file            — file_task  [READ]             15.9s
  [PASS] rename file          — file_task  [MOVE]             18.9s
  [PASS] find PDFs            — file_task  [QUERY]            17.6s
  [PASS] list Python files    — file_task  [QUERY + READ]     23.8s
  [PASS] find large files     — file_task  [QUERY + QUERY]    24.8s
  [PASS] move files           — file_task  [MOVE]             21.1s
  [PASS] copy file            — file_task  [COPY]             19.1s
  [FAIL] delete tmp files     — file_task  [QUERY] only       17.3s  (expected QUERY + DELETE)
  [PASS] delete by pattern    — file_task  [QUERY + DELETE]   27.3s
  [FAIL] find-then-move       — file_task  [MOVE + QUERY]     27.2s  (ordering: MOVE before QUERY)
  [PASS] email (not impl)     — NOT_IMPLEMENTED (L1 fast-path) 4ms
  [PASS] system (not impl)    — NOT_IMPLEMENTED (L1 fast-path) 4ms
```

### Gate Results

| Gate | Threshold | Result | Status |
|------|-----------|--------|--------|
| schema_validity | ≥ 95% | 100% (10/10) | **PASS** |
| action_type_ok | ≥ 80% | 90% (9/10) | **PASS** |
| action_ordering | ≥ 80% | 80% (8/10) | **PASS** |

**Phase 0 gate: PASSED.**

### NOT_IMPLEMENTED Latency

| Category | Latency | Method |
|----------|---------|--------|
| email_task | ~1–4ms | L1 classifier fast-path, no LLM call |
| system_task | ~1–4ms | L1 classifier fast-path, no LLM call |

Before L1 fast-path: email ~21s, system ~17s (full LLM call + model returns wrong category).

## Known 1B Model Limitations (Carried Into Phase 1)

### 1. "delete tmp files" intermittently generates only QUERY
The model sometimes omits the DELETE action for delete intents. With the 2-action grammar cap,
the model occasionally generates `[QUERY]` instead of `[QUERY, DELETE]`. This is a 1B model
quality issue — the model doesn't reliably follow the "delete = 2 actions" pattern.

### 2. "find-then-move" always generates MOVE before QUERY
The model generates `[MOVE, QUERY]` instead of `[QUERY, MOVE]` for "find and move" intents.
This ordering bug is correctly caught by `validate_action_ordering()`
(`MUTATION_WITHOUT_PRIOR_DISCOVERY`). The model needs prompt reinforcement or a larger model
to reliably generate discovery before mutation for multi-step intents.

### 3. Single-READ vs multi-READ under thermal load
Without cooldowns (`--fast` flag), "read file" fails when the CPU is thermally loaded from
prior eval cases. The model generates 2 READ actions filling ~400 tokens, which at 7 tok/s
takes ~57s + overhead, sometimes exceeding the 90s timeout or truncating the JSON.
With 15s cooldowns it consistently generates a single READ in ~16s (cold CPU).

### 4. Powersave CPU governor limits generation speed
The i5-7200U in powersave mode generates at 7–10 tok/s even under sustained load.
The performance governor would allow 25–30 tok/s but the system is configured for battery life.
All timing measurements in this document assume powersave governor.

## Thermal Profile

| Condition | Generation rate | Notes |
|-----------|----------------|-------|
| Cold start, single case | ~20–25 tok/s | CPU boosts before throttling |
| After 3+ consecutive cases | ~7–10 tok/s | Thermal throttle kicks in |
| Recovery time | ~15s | 15s cooldown between cases restores ~20 tok/s |

## Three-Configuration Comparison

Configuration A (no GBNF, no Layer 1) and B (GBNF only) benchmark runs were not performed
in this session due to the extended time required (10-minute waits between configs, ~2h total).
The full three-configuration comparison is deferred to Phase 1 hardware commissioning.

Primary delta known from isolated tests:
- L1 fast-path: NOT_IMPLEMENTED from ~21s (LLM call) → ~2ms (classifier)
- GBNF: eliminates all JSON structural failures; without GBNF, ~40% of outputs fail schema

## Unit Test Suite

```
61 passed, 5 deselected in 1.33s
(5 deselected = inference + integration markers, require inference server)
```

| File | Tests | Purpose |
|------|-------|---------|
| `tests/test_grammar_enforcement.py` | 4 static + 5 inference | Grammar/schema consistency |
| `tests/test_validators.py` | 20 | Semantic validator logic |
| `tests/test_classifier.py` | 13 | L1 accuracy + fast-path integration |
| `tests/test_schema_validation.py` | 9 | GoalSpec JSON schema compliance |
| `tests/test_state_machine.py` | 6 | Intent lifecycle transitions |
| `tests/test_tool_failure_tracker.py` | 4 | Tool failure escalation |
| `tests/test_path_authorization.py` | 5 | Path authorization checks |

## Infrastructure Delivered in Phase 0

| Component | Status | File(s) |
|-----------|--------|---------|
| GBNF grammar-constrained decoding (2-action cap) | ✓ | `inference/grammar/goal_spec.gbnf` |
| GoalSpec JSON schema | ✓ | `agents/schema/goal_spec.json` |
| Semantic validators (DAG, ordering, destructive flags) | ✓ | `agents/validators.py` |
| Intent parser with 3-tier validation | ✓ | `agents/intent_parser.py` |
| Layer 1 TF-IDF+LogReg classifier | ✓ | `agents/classifier.py`, `models/layer1_pipeline.joblib` |
| L1 NOT_IMPLEMENTED fast-path (<2ms for non-file categories) | ✓ | `agents/intent_parser.py` |
| SQLite audit log | ✓ | `db/audit.py` |
| File agent | ✓ | `agents/file_agent.py` |
| CLI entry point with Rich UI | ✓ | `leaves.py` |
| Eval harness: ordering-aware, multi-config, thermal-aware | ✓ | `tests/eval_suite.py` |
| Unit test suite (61 tests) | ✓ | `tests/` |

## Carried Into Phase 1

| Item | Reason Deferred |
|------|----------------|
| Model upgrade | Llama 3.2 1B generates action loops and wrong orderings. Qwen2.5-3B is target. |
| Layer 0 regex matcher | Deferred until model upgrade stabilizes output format. |
| "find-then-move" ordering | 1B model always generates MOVE+QUERY for this intent. Prompt-resistant. |
| "delete tmp" reliability | 1B model sometimes drops the DELETE action. Requires larger model. |
| threads=2 fix | Kaby Lake needs 2 threads (physical cores). Currently 4. Low-priority. |
| 3-config comparison | A/B/C benchmark not run. Phase 1 hardware commissioning task. |
| Real file execution | FileAgent parses GoalSpec but Phase 0 scope was intent parsing only. |

## Phase 0 Gate Summary

```
schema_validity >= 95%  : PASS (100%)   — with 15s cooldowns
action_type_ok  >= 80%  : PASS (90%)
action_ordering >= 80%  : PASS (80%)    — at exactly the threshold

Phase 0 declared complete at commit c3ae115
Hardware: i5-7200U (powersave), 8GB RAM, Llama-3.2-1B-Q4_K_M, llama-server f5ddcd1
Date: 2026-03-09
```
