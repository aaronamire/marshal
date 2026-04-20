# Security Policy

Leaves OS executes user-issued natural language as system actions. Every code
path through the agent dispatch boundary is in scope for security review.

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Email `security@leaves.dev` (or the maintainer at the email listed in
`pyproject.toml`) with:

1. A description of the issue
2. Steps to reproduce, including the GoalSpec or intent that triggered it
3. The audit log row (`leaves_audit.db`) for the affected intent if available
4. Your assessment of severity and impact

You will receive an acknowledgement within 72 hours. We aim to triage within
7 days and ship a fix within 30 days for high/critical findings, 90 days for
medium/low.

## Scope

In scope:
- The runtime enforcer (`agents/enforcer.py`) — any path that allows an
  action outside the GoalSpec's authorized resources, any path-traversal
  bypass, any destructive-action gating bypass.
- Sandboxing (`agents/sandboxed_runner.py`, `agentd.py`) — any escape from
  the Landlock ruleset or cgroup limits, any privilege elevation.
- The intent parser (`agents/intent_parser.py`) — any prompt injection that
  causes the model to emit a GoalSpec contradicting the user's stated intent
  (the `<USER_INTENT>` boundary is the trust line).
- The audit log (`db/audit.py`) — any way to fabricate, suppress, or modify
  log rows from outside the daemon.
- The HTTP API (`api/server.py`) — anything reachable from a non-loopback
  origin, any auth bypass, any injection vector.

Out of scope (for now):
- Denial of service via legitimate agent calls (e.g., requesting a 1M-token
  generation). Rate limits land in a future release.
- Issues in upstream dependencies (llama.cpp, wlroots, FastAPI) — report to
  those projects directly.
- Issues in the optional `RemoteAnthropicBackend` requiring a stolen API key.

## Threat Model

Leaves assumes:
- The user controls the host and trusts the operating system kernel.
- The local llama.cpp model is treated as untrusted output — the enforcer
  re-validates everything the model produces.
- User input via the REPL or HTTP API is treated as untrusted instruction
  data. The system prompt is the only trusted source of instructions.
- The Anthropic remote backend is trusted only with the prompt content
  the user explicitly sends through the writing agent.

Leaves does NOT defend against:
- A compromised host kernel
- A compromised llama.cpp build (verify checksums)
- A user with shell access running arbitrary processes outside the agent
  boundary
- Social engineering of the user to type a malicious intent

## Disclosure Timeline

We follow a 90-day coordinated disclosure window. After a fix ships and
users have had reasonable time to update, we publish an advisory in
`SECURITY-ADVISORIES.md` crediting the reporter (unless they prefer
anonymity).

## Hall of Fame

Reporters who find valid vulnerabilities are listed here with their consent.

_(none yet)_
