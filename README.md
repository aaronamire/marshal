# Marshal

**Local agents that can't escape their plan.**

Every action is validated against a signed `GoalSpec` at the dispatch
boundary, sandboxed with Landlock + cgroups, and logged to a replayable
audit DB. Non-destructive actions are safe to replay; destructive ones
can't be.

## What it is

A local-first AI agent runtime for Linux. Natural language goes in, a
schema-validated `GoalSpec` (category, actions, authorized resources)
comes out, and a runtime enforcer checks every dispatched action against
that plan before it touches the system.

- **Three-layer intent parser:** regex fast path (<0.1ms), sklearn
  classifier (3–8ms), then a fine-tuned Qwen-2.5-3B with GBNF
  grammar-constrained decoding for the fallback.
- **Runtime contract enforcer:** path-traversal resolution, symlink
  following, destructive-action gating, authorization-scope checks.
  Agents cannot act outside the plan.
- **Kernel-level isolation:** agent workers run under Landlock with
  per-intent cgroup v2 resource limits.
- **Replayable audit log:** SQLite WAL — every intent, every state
  transition, every action, every error. Replay refuses destructive
  actions by design.
- **Wayland compositor** (optional) with a dedicated AI panel rendered
  via Cairo.
- **BYOK remote escalation:** set `MARSHAL_ANTHROPIC_KEY` to let the
  writing agent call Claude for long-form content. Core planning stays
  local.

## Quick start

One command bootstraps everything (system deps, llama.cpp build, venv,
GoalSpec model download, optional compositor + systemd units):

```bash
./bootstrap.sh --with-compositor   # add --with-systemd to install user units
```

Then launch the full session (inference + agentd + API + compositor):

```bash
./scripts/start-session.sh
```

`start-session.sh` works from any TTY and falls back to standalone mode
if the systemd units aren't installed.

Or run the components individually:

```bash
source .os/bin/activate
./scripts/start-inference.sh &     # llama.cpp on :8080
python3 agentd.py &                # ~/.marshal/agentd.sock
uvicorn api.server:app --port 8765 &
python3 main.py                    # REPL
```

### Demo intents

Once the compositor is running, type any of these into the intent bar:

| Type this | What happens |
| --- | --- |
| `briefing` | Recent-changes briefing across indexed sources |
| `processes` / `task manager` / `ps` | Tabular process listing |
| `files` / `file manager` | Tabular file listing for `~/` |
| `summarize my recent downloads` | File listing for `~/Downloads` by mtime |
| `launch terminal` | Spawns marshal-terminal on the marshal compositor |
| `kill marshal-inference` | Terminates the named process |
| `volume up` / `mute` | Audio actions via PipeWire/PulseAudio |
| `cpu` / `ram` / `uptime` | System info via psutil |
| `search for the latest news on local LLMs` | Web search (DuckDuckGo) |
| `find report.pdf` | Local cortex semantic search |
| `history` / `hist` | Reload the past-intents feed |
| `exit` / `quit` / `:q` | Leave the compositor cleanly |

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

## Remote inference (optional)

```bash
export MARSHAL_ANTHROPIC_KEY=sk-ant-...
```

When set, `WritingAgent` escalates to Claude for prose generation and
falls back to the local model if the API is unavailable. Intent parsing
stays local.

## License

Apache-2.0 — see [LICENSE](LICENSE).
