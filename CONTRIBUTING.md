# Contributing to Marshal

Thanks for considering a contribution. Marshal runs untrusted user input
through a model and dispatches actions on the host. Every change touches
either correctness, safety, or both — please treat patches accordingly.

## Before you start

- Open an issue for non-trivial changes. A 5-minute design conversation
  prevents most rejected PRs.
- Read `SECURITY.md` if your change touches `agents/enforcer.py`,
  `agents/sandboxed_runner.py`, `agentd.py`, or any path that handles
  user-supplied parameters.
- Run `pytest tests/ -q` locally before opening a PR. CI runs the full
  suite plus the demo suite.

## Developer Certificate of Origin (DCO)

Every commit must be signed off:

```bash
git commit -s -m "your message"
```

The `-s` adds a `Signed-off-by` trailer asserting that you have the right
to submit the change under the project's Apache-2.0 license. We use DCO
instead of a CLA to keep contribution friction low.

## Pull Request Checklist

- [ ] Tests pass locally (`pytest tests/ -q`)
- [ ] New behavior has tests
- [ ] Public API changes are documented in `CHANGELOG.md` under
      `[Unreleased]`
- [ ] Security-path changes (enforcer, sandbox, IPC) include a red-team
      case in `tests/test_enforcer_redteam.py` or equivalent
- [ ] Commit messages follow the existing terse, present-tense style
      (e.g. "add KV cache warmup", not "added warmup feature")
- [ ] DCO sign-off on every commit

## Code Style

- Python: standard library first, third-party next, local imports last.
  Type hints on public functions. No bare `except:`. Use `MarshalError`
  for typed failures.
- C (compositor): match existing style — 8-column tabs, K&R braces,
  snake_case identifiers, headers in the order shown in `compositor.c`.
- Comments only when the *why* is non-obvious. Don't restate the *what*.

## Test Categories

- **Unit** (`tests/test_*.py`) — small, fast, hermetic.
- **Integration** (`tests/test_integration.py`) — full pipeline through
  `IntentParser` → `enforcer` → `AgentCoordinator` → audit log.
- **Red team** (`tests/test_enforcer_redteam.py`) — adversarial GoalSpecs
  that should be rejected. Add a case for every reported security issue.
- **Demo** (`tests/demo_suite.py`) — guaranteed intents with latency
  budgets. Keep this list short and rock-solid.

## What we do not accept

- Patches that bypass the enforcer "just for this case"
- Patches that disable Landlock / cgroup limits without a flag and a
  documented use case
- Patches that add a remote network call from inside the GoalSpec parsing
  hot path
- Patches that introduce a new agent without a test module
- Patches that depend on a not-yet-published upstream library version

## License

By contributing, you agree that your contribution is licensed under the
Apache License 2.0, the same license as the project.
