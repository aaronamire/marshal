# Marshal

**Local agents that can't escape their plan.**

Every action is validated against a signed GoalSpec at the dispatch boundary,
sandboxed with Landlock + cgroups, and logged to a replayable audit DB.
Non-destructive actions are safe to replay; destructive ones can't be.

## What this is

A local-first AI agent runtime for Linux. Natural language goes in, a validated
`GoalSpec` (category, actions, authorized resources) comes out, and a runtime
enforcer checks every dispatched action against the plan before it touches the
system.

- **Three-layer intent parser:** regex fast path (<0.1ms), sklearn classifier
  (3-8ms), then a fine-tuned Qwen-2.5-3B with GBNF grammar constrained
  decoding for the fallback.
- **Runtime contract enforcer:** path-traversal, symlink resolution, destructive
  action gating, authorization-scope checks. Agents cannot act outside the plan.
- **Kernel-level isolation:** agent workers run under Landlock with per-intent
  cgroup resource limits.
- **Audit log with replay:** SQLite WAL. Every intent, every state transition,
  every action, every error. Replay refuses destructive actions by design.
- **Wayland compositor** (optional) with a dedicated AI panel rendered via Cairo.
- **BYOK remote escalation:** set `MARSHAL_ANTHROPIC_KEY` to let the writing
  agent call Claude for long-form content. Core planning stays local.

## Quick start

```bash
python3 -m venv .os
source .os/bin/activate
pip install -r requirements.txt
./scripts/start-inference.sh &   # llama.cpp server on :8080
python3 main.py                # REPL
# or
python3 agentd.py                # background daemon (~/.marshal/agentd.sock)
uvicorn api.server:app --port 8765  # HTTP API
```

## Architecture

```
user text
   │
   ▼
IntentParser ── Layer 0 regex ──┐
   │                             │
   ├─ Layer 1 TF-IDF classifier ─┤──► GoalSpec (schema-validated JSON)
   │                             │
   └─ Layer 2 llama.cpp + GBNF ──┘
                                     │
                                     ▼
                              enforcer.enforce(action, goal_spec)
                                     │
                                     ▼
                              AgentCoordinator (DAG-scheduled)
                                     │
                          ┌──────────┼──────────┐
                          ▼          ▼          ▼
                     file_agent  system_agent  web_agent … (sandboxed)
                                     │
                                     ▼
                              SQLite audit log
```

Key files: `agents/intent_parser.py`, `agents/enforcer.py`, `agentd.py`,
`inference/client.py`, `db/audit.py`, `api/server.py`.

## Status

Phase 2 complete (2026-04-14). Eval suite: 37/38 schema+category+action_type
(97%), 38/38 action ordering (100%) across 24 regular + 5 not-implemented cases.

Implemented agents: `file`, `system` (incl. app launch/terminate), `web`,
`writing`, `audio`, `network`, `power`.

Not yet implemented: `email`, Layer 3 remote GoalSpec fallback, the
always-on background daemon (`marshald`), the agent-authority protocol.

## Remote inference (optional)

```bash
export MARSHAL_ANTHROPIC_KEY=sk-ant-...
```

When set, `WritingAgent` escalates to Claude for prose generation and falls
back to the local model if the API is unavailable. Intent parsing stays
local to keep p50 latency predictable.

## License

TBD.
