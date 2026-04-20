# Security Audit — 2026-04-18

Scope: `agents/enforcer.py`, `agents/sandboxed_runner.py`, `agentd.py`,
`api/server.py`, `db/audit.py`. The pitch — *agents that can't escape
their plan* — rests on these files. A hole in any of them invalidates
the product claim.

Methodology: threat-model first, then line-by-line read of the four
chokepoints, then exploit-construction for each candidate finding,
then targeted red-team tests for what was actually exploitable.

## Threat model

We assume the local user is benign but a) the LLM may be coerced by
prompt injection in retrieved content, b) any local process running as
the same uid may speak the agentd Unix-socket protocol, c) a browser
on the same host may speak the HTTP API via a same-origin or DNS-
rebound page. We do NOT assume root or kernel-level adversaries.

Goals (in priority order):
1. **Plan integrity** — every executed action is in the GoalSpec the
   user approved, with the same type and within the same authorized
   resource set.
2. **Path containment** — no executed file op touches a path outside
   `authorization.resources`, even via symlink, `..`, tilde, or
   non-whitelisted param keys.
3. **Process isolation** — runner subprocess cannot escape its cgroup
   slice, cannot influence other intents' cgroups, cannot be reached
   by other local users.
4. **Audit completeness** — every authorization violation makes it to
   the audit log before the user is told the action was blocked.

## Findings

### F-1 — `intent_id` from socket clients is unvalidated and used to derive a cgroup path (HIGH)

`agentd._handle_client` reads `goal_spec` directly off the wire
(line 819 of `agentd.py`) and forwards it to `_run_sandboxed`, which
builds:

```python
intent_id = goal_spec.get("intent_id", "unknown")[:8]
cgroup_path = _CGROUP_ROOT / f"intent-{intent_id}"
```

The schema requires `intent_id` to match a UUID-v4 regex, but agentd
performs no schema validation. A client that connects directly to the
Unix socket can pass `intent_id = "/../foo"`. The first 8 chars are
`/../foo`, producing `/sys/fs/cgroup/leaves/intent-/../foo`. The
kernel resolves `..` during the `mkdir` parent lookup, so the
subprocess actually creates `/sys/fs/cgroup/leaves/foo` and writes
its pid into `cgroup.procs` there.

Reproduced via `python -c` (see audit notes); both `mkdir` and the
follow-up `cgroup.procs` write succeed against an attacker-chosen
sibling-of-intent path under `_CGROUP_ROOT`.

**Impact.** Bounded to `_CGROUP_ROOT` and below (we are not root),
but enables: (a) collision with another in-flight intent's cgroup
slice, allowing the attacker's process to share its memory budget,
(b) leaving stranded directories that agentd's cleanup logic can't
match, (c) a primitive that compounds with future cgroup-delegated
controllers.

The API surface is currently safe because `/v1/intent/plan` overwrites
`intent_id` with `uuid.uuid4()` before storing the plan. The exposure
is purely the socket protocol, which any same-uid process can speak.

**Fix.** Validate `intent_id` against the UUID-v4 regex in agentd, at
the socket boundary, before any path interpolation. Reject violators
with `INVALID_INTENT_FORMAT`. Applied in this commit
(`agentd._validate_intent_id`).

### F-2 — Unix socket created with default umask permissions (MEDIUM)

`agentd._run_server` calls `asyncio.start_unix_server(path=...)`
without an explicit chmod or umask. With the default user umask
`022`, the socket is created world-readable/writable in mode terms
(actually mode `0666 & ~umask = 0644` on most systems, which still
permits reads — and connect(2) on AF_UNIX requires only read+write
on the socket inode). On a multi-tenant box, any other UID with
filesystem access to `~/.leaves/agentd.sock` can connect and submit
GoalSpecs.

In practice `~/.leaves/` is `0700` on most user accounts, but Leaves
does not enforce that. The socket is a sole-author resource and
should be `0600` regardless of the parent dir.

**Fix.** `os.chmod(_SOCK_PATH, 0o600)` immediately after
`start_unix_server` returns, before the first `accept()`. Applied.

### F-3 — Unix socket has no peer-credential check (MEDIUM)

Defense-in-depth on top of F-2: the daemon should refuse connections
from other UIDs even if filesystem perms are misconfigured. Linux
exposes peer credentials via `getsockopt(SOL_SOCKET, SO_PEERCRED)`.
The asyncio server doesn't check them.

**Fix.** Wrap `_handle_client` to call `SO_PEERCRED` on the underlying
socket and reject if `pid_t.uid != os.geteuid()`. Applied as
`_check_peer_credentials`. Tightens the socket trust boundary from
"anything that can open the file" to "only this user's processes".

### F-4 — CORS permits credentialed cross-origin requests from localhost dev origins (MEDIUM)

`api.server` allows credentialed CORS from
`http://localhost:{3000,5173}` and `http://127.0.0.1:{3000,5173}`,
with `allow_methods=*` and `allow_headers=*`. Combined with the lack
of any authentication on the API, this means any page the user loads
in a browser tab on those ports — including a malicious dev server
they pull down for an unrelated project — can issue authenticated
calls into `/v1/intent/plan` and `/v1/intent/execute`.

DNS-rebinding amplifies this: an attacker-controlled domain can be
rebound to `127.0.0.1` and call the API directly, since the `Host`
header is not validated.

**Fix.**
1. Drop `allow_credentials=True` (no cookies are involved).
2. Restrict `allow_methods` to the methods we actually serve.
3. Add a `Host`-header allowlist middleware that rejects requests
   whose `Host` is not `127.0.0.1:8765` / `localhost:8765`.

Applied. A header-based local-machine bearer token is the next-step
hardening — out of scope for this commit because it requires UX work
in the REPL/SDK clients.

### F-5 — Enforcer step 4 (destructive-mismatch check) is dead code (LOW)

`agents/enforcer.py` step 2 already rejects any case where the
executed action's `type` differs from the planned `type`. Step 4
checks `action_type in DESTRUCTIVE and planned_type not in
DESTRUCTIVE` — but if the two types differ, step 2 has already
raised; if they match, both are either destructive or both are not.
Step 4's branch is unreachable.

**Impact.** No security impact — this is dead code masquerading as
defense-in-depth. The risk is that a future refactor could relax
step 2 (e.g., to allow type promotion) and quietly disable a check
the author thought was independent.

**Fix.** Remove step 4 with a comment pointing back to step 2.
Applied.

### F-6 — Non-whitelisted param keys with bare-relative paths bypass the value-scan (LOW)

`_extract_paths` performs two passes: a hard-coded whitelist of param
keys (`path`, `source`, `destination`, …) that are always treated as
path-like, and a value-scan that catches strings looking like paths
in any other key. The value-scan uses `_looks_like_path`, which
returns False for strings that don't begin with `/`, `~/`, `./`, or
`../`. A param such as `{"my_log": "etc/passwd"}` (bare relative)
slips through both passes.

In practice, bare relatives hit the agent's CWD and Landlock
re-restricts them, so the exploitable window is narrow. But the
contract docstring says "every path-like param" is checked, and that
overstatement is what worries me — future agents may rely on it.

**Fix.** Two options:
- (A) Treat any string value containing a `/` as path-like. False-
  positive risk: URLs without scheme, command fragments.
- (B) Document the limitation precisely and add a strict-mode flag
  that fails closed.

Chose (A) with the URL-without-scheme case explicitly tested. Applied.

### F-7 — Landlock fails open with no strict mode (INFO)

`apply_landlock` returns `{"active": False, "reason": ...}` on any
error, and the runner proceeds unsandboxed, relying on the
Python-level enforcer. The sandbox status surfaces upward through
the `sandbox` field on the response, so a UI can show the user.

This is a deliberate availability tradeoff for unsupported kernels,
but a security-conscious deployment should be able to opt into
"refuse to run unless Landlock is active". Out of scope for this
commit; tracked as a follow-up.

### F-8 — JSON line size unbounded on socket and runner stdin (INFO)

`reader.readline()` and `sys.stdin.readline()` impose no max length.
A malicious peer (after F-2/F-3 are bypassed) can send arbitrarily
large JSON to OOM agentd. Out of scope for this commit; mitigated by
F-2/F-3 closing the trust boundary.

## Positive observations

- All audit-log writes use parametrized SQLite queries
  (`db/audit.py`). No f-string SQL anywhere in the project tree
  (verified by repo-wide grep for `executescript|execute.*f".*WHERE`).
- The enforcer's path-containment check correctly resolves symlinks
  via `Path.resolve()` and rejects sibling-file escape when an
  authorized resource is a single file (the regression test
  `path_sibling_file_via_parent_match` exists and passes).
- Per-intent cgroup limits include `pids.max=32` and `memory.swap.max=0`,
  which together kill fork bombs and prevent swap thrashing. Good
  defaults.
- The `IntentParser` always overwrites `goal_spec["intent_id"]` with
  a fresh `uuid.uuid4()`, so even if the LLM emits a malicious
  string, the API path strips it. The exposure in F-1 is solely
  via the un-validated socket interface.

## Test additions

`tests/test_enforcer_redteam.py` is unchanged in shape; new tests
go in `tests/test_security_hardening.py`:
- `test_intent_id_uuid_validation_rejects_traversal`
- `test_socket_perms_chmod_after_bind`
- `test_cors_drops_credentials_and_dev_origins`
- `test_value_scan_catches_bare_relative_path`

All pass after the fixes.

## Out of scope, tracked

- TOCTOU between enforcer path resolution and agent execution.
  Landlock provides the kernel-level second layer that bounds this.
  A full fix requires passing resolved fds from enforcer to agent.
- Bearer-token auth for the HTTP API. Requires SDK/CLI changes.
- Strict-Landlock mode (refuse to run unsandboxed). Requires UX
  decision on how to surface unsupported-kernel errors.
