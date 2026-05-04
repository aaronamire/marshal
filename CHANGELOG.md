# Changelog

All notable changes to Marshal are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-04-29 — Demo release

### Added
- **Briefing as a first-class agent** — `agents/briefing_agent.py` wraps
  `cortex/briefing.py` behind the standard `BaseAgent` interface, registered
  under the `briefing` agent + `briefing` category. Layer 0 already
  recognized phrases like `briefing`, `good morning`, `what changed`; they
  now actually execute end-to-end instead of returning NOT_IMPLEMENTED.
  Schema (`agents/schema/goal_spec.json`) and grammar
  (`inference/grammar/goal_spec.gbnf`) updated with the new `BRIEFING`
  action type and `briefing` agent/category enums.
- **In-OS process manager and file manager** — typing `processes`,
  `process manager`, `task manager`, `ps`, `files`, or `file manager` in
  the compositor's intent bar opens a tabular listing rendered with the
  existing card pipeline. Process listing shows PID/name/CPU%/MEM/user
  (sorted by CPU desc, top 25) and walks the user toward `kill <name>`
  for termination. File listing shows name/size/mtime in human units.
  Both bypass the LLM via Layer 0 fast-paths.
- **Compositor exit shortcut** — typing `exit`, `quit`, or `:q` in the
  intent bar triggers a clean `wl_display_terminate()` via the existing
  wakeup-pipe channel. Joins Ctrl+Alt+Backspace as a non-emergency exit.
- **Compositor history shortcut** — typing `history` or `hist` reloads
  `/v1/history?limit=50` into the feed instead of forwarding to the LLM
  (which used to misclassify the bare word as a `system_task`).
- **Well-known folder fast-paths** — Layer 0 catches phrasings like
  "summarize my recent downloads" / "list documents" / "show my pictures"
  and routes them to `FileAgent.QUERY` against the right `~/<Folder>`,
  sorted by mtime desc. Closes the LLM-misroute that was sending these
  to `WebAgent.QUERY` and producing web search results for filesystem
  queries.

### Changed
- **KV-cache warmup** is now awaited (with a 300s ceiling) before agentd
  accepts user connections, eliminating the cold-start race that caused
  the first user intent to time out. The warmup primes only the bare
  ChatML system header — the longest prefix every L2 request shares —
  so cache_prompt actually pays off; the previous version included a
  per-query RAG block that broke the prefix match. Also bypasses
  `IntentParser` construction so HuggingFace metadata downloads don't
  steal CPU during the prefill.
- **`scripts/start-session.sh`** detects whether marshal-*.service units
  are loaded; if not, falls back to direct background launches with logs
  in `~/.marshal/logs/`. New `reset_standalone_leftovers` step kills
  stale daemons from a prior crashed run before binding new sockets.
- **Inference timeouts** — `TIMEOUT_READ_SECONDS` 90s → 180s for
  steady-state requests; warmup uses its own dedicated 240s timeout.
- **Result rendering** in `api/server.py:_format_result_text`:
  - Web results now list titles/URLs/snippets (was: count-only `5 web result(s)`).
  - Web fetches show title + URL + 400-char preview.
  - Briefings surface item filenames when the change set is small (≤5).
  - Process listing is now tabular with columns and a `… N more` footer.
  - File listing is now tabular with NAME/SIZE/MODIFIED columns.
- **Compositor input routing** — `search ...` no longer auto-routes to
  cortex search; it goes through the full intent pipeline so the LLM
  can choose between web and cortex. `find ...` remains the explicit
  cortex prefix.
- **Default standard-tier model** is now `goalspec_qwen25_3b_q4km.gguf`
  (the fine-tuned GoalSpec model) rather than the base Qwen2.5-3B.
- **Briefing groups of size 1** are now named directories instead of
  being collapsed into "other" — fixes the opaque single-file briefing.
- **Compositor logo** in the taskbar replaced with a simple stroked
  circle (`draw_logo_glyph`); the leaf glyph is retained for the
  empty-state placeholder.

### Fixed
- **Internal-error on large file listings** — `agentd._run_sandboxed`
  opened its Unix connection to the runner pool with the default 64 KB
  StreamReader buffer; a 500-file listing produced ~90 KB and
  `LimitOverrunError` masqueraded as `INTERNAL_ERROR`. Lifted to 16 MB,
  matching the API→agentd direction.
- **`launch terminal` silently no-op** — `system_agent` was launching
  apps with `stdout=stderr=PIPE` then closing the parent's ends, which
  delivered SIGPIPE to any child that wrote startup diagnostics
  (including marshal-terminal). Replaced with `stdin=DEVNULL` plus
  `stdout/stderr` to a per-app log under `~/.marshal/logs/launches/`.
  In-tree binaries (`marshal-terminal`, `marshal-compositor`) are now
  resolved from the project's builddir when not in PATH.
- **Stale meson builddir after `leaves-os` → `marshal` rename** —
  documented in bootstrap notes; resolution is `rm -rf builddir &&
  meson setup builddir`. The terminal binary's `meson.build` already
  produced `marshal-terminal`; only the cached path needed wiping.
- **`ddgs not installed`** — replaced legacy `duckduckgo-search>=4.0`
  with the renamed-upstream `ddgs>=9.0` in `pyproject.toml` and
  `requirements.txt`. `agents/web_agent.py` was already importing from
  `ddgs`.
- **Watchdog dependency** — declared `watchdog>=4.0` (was used by
  `cortex/watcher.py` but never listed; agentd boot logged the failure
  and persistent-FS-trigger intents silently never fired).
- **Wlroots packaging on Arch** — `bootstrap.sh --with-compositor` now
  installs `wlroots0.18` (matches the version pinned in
  `compositor/meson.build`); the unversioned `wlroots` package no
  longer exists in Arch repos.

### Security
- Security audit pass on enforcer + sandbox; full report in
  `docs/security-audit-2026-04-18.md`. Fixes shipped in this release:
  - **F-1 (HIGH)** — agentd now validates `intent_id` against the
    UUID-v4 regex at the socket trust boundary. Previously, a same-uid
    process could submit `intent_id = "/../foo"` and cause cgroup-path
    traversal under `/sys/fs/cgroup/marshal`.
  - **F-2 (MED)** — `~/.marshal/agentd.sock` is `chmod 0o600` immediately
    after bind, regardless of parent-directory perms.
  - **F-3 (MED)** — added `SO_PEERCRED` peer-uid check in `_handle_client`;
    the daemon now refuses connections from any uid other than its own.
  - **F-4 (MED)** — API CORS no longer advertises credentials, restricts
    methods to GET/POST/DELETE, and a Host-header allowlist middleware
    rejects DNS-rebound requests with 421 Misdirected Request.
  - **F-5 (LOW)** — removed dead step-4 in `agents/enforcer.py` whose
    check was already subsumed by step-2's type-equality requirement.
  - **F-6 (LOW)** — `_looks_like_path` now catches bare-relative paths
    in non-whitelisted param keys (`{"log_target": "etc/shadow"}` is
    rejected).
- New tests: `tests/test_security_hardening.py` — 29 cases mapping 1:1
  to audit findings.

### Added
- **4500-pair v2 training corpus** (`data/goalspec_training_v2.jsonl`,
  generated by `scripts/gen_training_v2.py`) — closes the 0-pair gaps for
  audio/network/power and the thin coverage for system/web that the
  current fine-tuned model was trained without. Distribution:
  - 1000 file_task (READ/QUERY/DELETE/MOVE/COPY/WRITE single-action)
  - 700 system_task (12 query phrasings × 5 query_types + 50 apps × 12
    launch verbs + 50 apps × 13 terminate verbs)
  - 700 web_task (54 search topics × 16 verbs + 25 URLs × 16 fetch verbs)
  - 400 audio_task / 400 network_task / 400 power_task — first training
    coverage for these three agents (previous corpus had 0 pairs each)
  - 300 writing_task (31 topics × 16 verbs × 10 formats)
  - 200 multi-action DAGs (find+delete, find+move, find+copy with
    `depends_on` edges) — teaches the model to emit QUERY before
    MUTATION
  - 200 NOT_IMPL email refusals (matches L0 fast-path category)
  - 200 ambiguous/long-winded phrasings (polite hedges, multi-clause
    framings) — robustness against real-world phrasing diversity
  - Every pair is JSON-schema-validated against
    `agents/schema/goal_spec.json` before write; deduplicated by
    case-folded user text (185 raw duplicates rejected).
- **L0 web search/fetch + file phrasing expansion** (`agents/layer0.py`) —
  intents that previously fell through to the L2 LLM path now match L0 in
  under 1ms, saving ~800-1500ms of inference per intent. New patterns:
  - **Web search** (agent=web, query_type=search): `google X`, `look up X`,
    `search the web for X`, `search for X`, `web search X`. Path-in-query
    guard (`look up ~/file.txt`) falls through so file finds still go to
    the right place.
  - **Web fetch** (agent=web, query_type=fetch): `fetch <url>`, `load <url>`,
    `scrape <url>`, `download <url>`, `get the page at <url>`. `open <url>`
    requires an explicit `https?://` scheme so `open firefox` keeps routing
    to APP_LAUNCH.
  - **File READ alternates**: `view ~/x.py`, `open ~/x.py` (path-prefixed
    only), `what's in ~/x.txt` / `what is in ~/x.txt`.
  - **File WRITE (empty)**: `touch ~/x.txt`, `create file ~/x.md`,
    `create a file at ~/notes/x.txt`.
- 23 new `tests/test_layer0.py` cases covering every new phrasing plus the
  app-launch / web-fetch disambiguation.
- **Warm runner pool** (`agents/runner_master.py`) — long-lived prefork
  master that pre-imports the agent stack and forks workers per intent.
  Eliminates the ~150-300ms CPython startup + ~250ms agent-stack import
  cost paid by every cold-path subprocess. Workers inherit modules via
  copy-on-write and apply Landlock inside the fork (so per-intent FS
  scope is preserved), then receive their cgroup placement via a small
  pid → ack handshake before executing. agentd's `_run_sandboxed` now
  dispatches to the pool first and falls back to the cold-path
  subprocess on transport failure (master crashed, socket missing).
  Domain errors from the worker propagate normally — no double-charge.
  - Bench: **24ms warm p50 vs 452ms cold p50 = 18.5× speedup, 427ms
    saved per intent** on a `file QUERY` GoalSpec (`scripts/bench_runner_paths.py`).
- `marshal_runner_path_total{path}` Counter exposed at `/v1/metrics`,
  labelled `warm` or `cold` so dispatch ratios are observable.
- `observability.py` — JSON log formatter and a tiny in-process metrics
  registry (Counter + Histogram, no `prometheus_client` dep). Exposes
  three named series:
    - `marshal_intents_total{status,category}` — incremented on every
      terminal intent disposition in agentd.
    - `marshal_inference_latency_ms` — histogram of `LocalLlamaCppBackend`
      `/completion` latencies, with buckets tuned for the CPU path.
    - `marshal_enforcer_rejections_total{reason}` — incremented on every
      `AUTHORIZATION_VIOLATION` raise inside `agents/enforcer.enforce()`,
      labelled by which check fired.
- `/v1/metrics` endpoint serving the registry in Prometheus 0.0.4 text
  format. `curl http://127.0.0.1:8765/v1/metrics`.
- agentd's 20 ad-hoc `print(..., flush=True)` calls replaced with
  structured `logging` calls; `extra={...}` carries intent_id, durations,
  and error fields as first-class JSON keys instead of string-formatted
  noise.
- `tests/test_observability.py` — 15 tests covering the formatter,
  counter/histogram primitives, Prometheus rendering, and the HTTP
  endpoint.
- Inference KV cache warmup at agentd boot (`_warm_inference_kv_cache`).
  Moves the ~30s cold-prefill cost out of the user's first request.
- `MARSHAL_ANTHROPIC_MODEL` env var to override the default Claude model
  used by `RemoteAnthropicBackend` without code changes.
- `research/` directory with `README.md` clearly marking unshipped designs.
- `tests/demo_suite.py` — guaranteed-working intents with asserted p95
  latency budgets. `MARSHAL_DEMO_MODE=stub` (CI default) exercises the L0
  regex path only; `MARSHAL_DEMO_MODE=full` adds Layer-2 inference cases.
- `pytest-timeout` dev dep with a 30s default per-test budget. Slow
  inference tests now fail loudly instead of hanging the suite.

### Changed
- Default `pytest` invocation now deselects the `inference` marker. Run
  `pytest -m inference` explicitly when llama-server is up.
- Consolidated pytest config: removed `pytest.ini` in favor of
  `[tool.pytest.ini_options]` in `pyproject.toml`.
- `marshal` REPL gained `--verbose` / `-v` (and `MARSHAL_VERBOSE=1` env var)
  that dumps the full GoalSpec after every parse and shows the
  `MarshalError` code + structured detail on failures. The `verbose` REPL
  command toggles it at runtime. Replaces the previous "user message
  only" surface that hid the offending value.
- `marshal --version` prints the package version (was previously hidden).
- `[project.scripts]` re-enabled: `pip install -e .` now installs a
  `marshal` console script (was deferred until `main()` existed).
- 10 new `web_task` examples in `rag/seed_examples.jsonl` covering
  natural-language search phrasings ("tell me about X", "explain Y",
  "google Z") and fetch-style intents ("open <url>", "scrape <url>",
  "load <url>"). Closes the gap where the fine-tune saw 0 web_task pairs.

### Changed
- README rewritten with a clear pitch, architecture diagram, and an honest
  "what this is / what this is not" framing.
- `RemoteAnthropicBackend` default model bumped from the retired
  `claude-sonnet-4-20250514` to `claude-sonnet-4-6`.

### Moved
- `docs/agent-authority-protocol.md` → `research/agent-authority-protocol.md`.
  This is an unshipped design exploration, not a feature of the product.
- `docs/phase0-*.md`, `docs/phase1-*.md`, and the corresponding canonical
  test result files → `docs/history/` to keep the docs root focused on
  current state.

### Removed
- Stray binary files at the repo root (`chromatic1.jpeg`, a WhatsApp image).

## [0.3.0] — 2026-04-14

Phase 2 release. See `docs/phase2-final-results.md` for the eval gate.

### Added
- Fine-tuned Qwen-2.5-3B GoalSpec model with GBNF grammar-constrained
  decoding (37/38 schema+category+action_type pass, 38/38 ordering).
- `WritingAgent` with local + remote (Anthropic) backend selection.
- DAG executor with parallel multi-action scheduling and stream-mode
  dependency edges.
- Compositor event watcher and Cairo-rendered AI panel cards with state
  transitions (PENDING / AWAITING_CONFIRM / EXECUTING / DONE / FAILED /
  CANCELLED).
- `agents/enforcer.py` runtime action-contract enforcer with path-traversal
  resolution, symlink following, and destructive-action gating.
- Landlock + cgroup v2 isolation per agent worker in `agentd.py`.
- SQLite WAL audit log with replay; replay refuses destructive actions.
- Hybrid dense+sparse RAG store (LanceDB + BM25, 59 seed examples).

## Earlier phases

See `docs/history/` for Phase 0 and Phase 1 release notes and gate results.
