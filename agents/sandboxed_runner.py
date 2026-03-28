#!/usr/bin/env python3
"""
Leaves OS sandboxed agent runner.

Reads {"goal_spec": ..., "from_state": ...} from stdin (one JSON line),
applies Landlock filesystem restrictions to THIS process, then executes
AgentCoordinator and writes the result JSON to stdout.

Landlock is applied BEFORE project imports so the restriction covers all
subsequent file access. Only stdlib modules are used prior to restriction.
"""
# ── stdlib only — no project imports until after Landlock ──────────────────
import ctypes
import json
import os
import pathlib
import sys

# ── Landlock syscall numbers (x86-64) ──────────────────────────────────────
_SYS_CREATE_RULESET = 444
_SYS_ADD_RULE       = 445
_SYS_RESTRICT_SELF  = 446

_RULE_PATH_BENEATH  = 1
_PR_SET_NO_NEW_PRIVS = 38

# ── FS access rights (Landlock ABI v1, bits 0–12) ──────────────────────────
_ACCESS_WRITE_FILE  = 1 << 1
_ACCESS_READ_FILE   = 1 << 2
_ACCESS_READ_DIR    = 1 << 3
_ACCESS_REMOVE_FILE = 1 << 5
_ACCESS_MAKE_DIR    = 1 << 7
_ACCESS_MAKE_REG    = 1 << 8

_FS_READ_ONLY = _ACCESS_READ_FILE | _ACCESS_READ_DIR

_FS_READ_WRITE = (
    _ACCESS_WRITE_FILE  |
    _ACCESS_READ_FILE   |
    _ACCESS_READ_DIR    |
    _ACCESS_REMOVE_FILE |
    _ACCESS_MAKE_DIR    |
    _ACCESS_MAKE_REG
)

# Handled mask = union of all rights we ever grant in rules.
# Access types NOT here (EXECUTE, REMOVE_DIR, MAKE_CHAR, …) remain unrestricted.
_HANDLED_ACCESS = _FS_READ_WRITE

_DESTRUCTIVE_TYPES = frozenset({"DELETE", "MOVE", "WRITE", "COPY"})


# ── ctypes structs ─────────────────────────────────────────────────────────

class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    # Kernel struct is __attribute__((packed)): u64 + s32 = 12 bytes.
    # Passing by pointer — field offsets (0 and 8) are what matter, not total size.
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd",      ctypes.c_int32),
    ]


# ── libc handle ────────────────────────────────────────────────────────────

try:
    _libc = ctypes.CDLL(None, use_errno=True)
    _libc.syscall.restype = ctypes.c_long
    _libc.prctl.restype   = ctypes.c_int
    _HAVE_LIBC = True
except Exception:
    _HAVE_LIBC = False


# ── Landlock wrappers ──────────────────────────────────────────────────────

def _ll_create_ruleset(handled: int) -> int:
    attr = _RulesetAttr(handled_access_fs=ctypes.c_uint64(handled))
    return int(_libc.syscall(
        ctypes.c_long(_SYS_CREATE_RULESET),
        ctypes.byref(attr),
        ctypes.c_size_t(ctypes.sizeof(attr)),
        ctypes.c_uint32(0),
    ))


def _ll_add_path_rule(ruleset_fd: int, path: pathlib.Path, access: int) -> None:
    """Open path with O_PATH and register a Landlock allow rule. Silently skips on error."""
    try:
        fd = os.open(str(path), os.O_PATH | os.O_CLOEXEC)
    except OSError:
        return
    try:
        rule = _PathBeneathAttr(
            allowed_access=ctypes.c_uint64(access),
            parent_fd=ctypes.c_int32(fd),
        )
        _libc.syscall(
            ctypes.c_long(_SYS_ADD_RULE),
            ctypes.c_int(ruleset_fd),
            ctypes.c_uint32(_RULE_PATH_BENEATH),
            ctypes.byref(rule),
            ctypes.c_uint32(0),
        )
    finally:
        os.close(fd)


def _ll_restrict_self(ruleset_fd: int) -> int:
    return int(_libc.syscall(
        ctypes.c_long(_SYS_RESTRICT_SELF),
        ctypes.c_int(ruleset_fd),
        ctypes.c_uint32(0),
    ))


# ── apply_landlock ─────────────────────────────────────────────────────────

def apply_landlock(goal_spec: dict) -> None:
    """
    Apply Landlock FS restrictions to this process.

    Grants:
      - Per-intent resources: READ_WRITE for destructive actions, READ_ONLY otherwise
      - ~/.leaves/: READ_WRITE (SQLite audit log writes)
      - /home, ~/: READ_DIR only — allows traversal to reach ~/... paths
      - project root, /usr, /lib, /lib64, /proc: READ_ONLY (Python runtime)

    On any failure (unsupported kernel, ENOSYS, …) logs to stderr and continues
    without restriction — never crashes the agent.
    """
    if not _HAVE_LIBC:
        return
    try:
        _do_apply(goal_spec)
        print("landlock: sandbox active", file=sys.stderr)
    except Exception as exc:
        print(f"landlock: skipping ({exc})", file=sys.stderr)


def _do_apply(goal_spec: dict) -> None:
    actions = goal_spec.get("actions", [])
    is_destructive = any(
        a.get("type", "").upper() in _DESTRUCTIVE_TYPES for a in actions
    )
    resource_access = _FS_READ_WRITE if is_destructive else _FS_READ_ONLY

    resources = goal_spec.get("authorization", {}).get("resources", [])
    resource_paths = [pathlib.Path(r).expanduser() for r in resources if r]

    project_root = pathlib.Path(__file__).parent.parent.resolve()
    leaves_dir   = pathlib.Path.home() / ".leaves"

    ruleset_fd = _ll_create_ruleset(_HANDLED_ACCESS)
    if ruleset_fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")

    try:
        # Per-intent paths.
        # Landlock PATH_BENEATH rules work correctly for directory inodes.
        # For file paths, add the parent directory — Landlock's "beneath" check
        # walks the directory tree and file-level rules don't participate correctly
        # in ABI v1. The Python-level _authorize() still enforces the precise
        # file-level boundary; Landlock provides the coarse kernel-enforced layer.
        for path in resource_paths:
            if path.exists() and path.is_file():
                _ll_add_path_rule(ruleset_fd, path.parent, resource_access)
            else:
                _ll_add_path_rule(ruleset_fd, path, resource_access)

        # Always read-write: audit db
        _ll_add_path_rule(ruleset_fd, leaves_dir, _FS_READ_WRITE)

        # Allow traversal through /home and ~/
        # Landlock requires every directory in the path to have at least READ_DIR
        # or open() is blocked before reaching the target. READ_DIR alone grants
        # traversal/listing but NOT file content reads — so ~/other_file.txt
        # remains unreadable unless it has its own explicit rule.
        _ll_add_path_rule(ruleset_fd, pathlib.Path.home().parent, _ACCESS_READ_DIR)
        _ll_add_path_rule(ruleset_fd, pathlib.Path.home(), _ACCESS_READ_DIR)

        # /tmp: many programs need temp file access.
        _ll_add_path_rule(ruleset_fd, pathlib.Path("/tmp"), _FS_READ_WRITE)

        # XDG_RUNTIME_DIR: Wayland/PipeWire/D-Bus sockets live here.
        # GUI apps launched by SystemAgent need to connect to the compositor.
        xdg_runtime = pathlib.Path(
            os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        )
        _ll_add_path_rule(ruleset_fd, xdg_runtime, _FS_READ_ONLY)
        # Traversal for /run and /run/user
        _ll_add_path_rule(ruleset_fd, pathlib.Path("/run"), _ACCESS_READ_DIR)
        _ll_add_path_rule(ruleset_fd, pathlib.Path("/run/user"), _ACCESS_READ_DIR)

        # Always read-only: Python runtime + system config
        for path in (
            project_root,
            pathlib.Path("/usr"),
            pathlib.Path("/lib"),
            pathlib.Path("/lib64"),
            pathlib.Path("/proc"),
            pathlib.Path("/etc"),   # DNS resolution + SSL certs
        ):
            _ll_add_path_rule(ruleset_fd, path, _FS_READ_ONLY)

        # no_new_privs required before restrict_self
        if _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")

        if _ll_restrict_self(ruleset_fd) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(ruleset_fd)


# ── main ───────────────────────────────────────────────────────────────────

def _main() -> None:
    # Step 1: read input (stdlib only — no project imports yet)
    line = sys.stdin.readline()
    msg            = json.loads(line)
    goal_spec      = msg["goal_spec"]
    from_state_str = msg.get("from_state", "PARSING")

    # Step 1b: propagate compositor's WAYLAND_DISPLAY if present in goal_spec
    wl_display = goal_spec.get("metadata", {}).get("wayland_display")
    if wl_display:
        os.environ["WAYLAND_DISPLAY"] = wl_display

    # Step 2: apply Landlock — restricts THIS process's FS access from here on
    apply_landlock(goal_spec)

    # Step 3: project imports (Landlock already active)
    _root = str(pathlib.Path(__file__).parent.parent.resolve())
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from agents.state_machine import IntentLifecycle, IntentState
    from agentd import AgentCoordinator
    from db.audit import get_db
    from errors import LeavesError

    # Step 4: build lifecycle to the correct pre-execution state
    intent_id = goal_spec["intent_id"]
    lifecycle  = IntentLifecycle(intent_id=intent_id)
    lifecycle.transition(IntentState.PARSING)
    if from_state_str == "AWAITING_AUTH":
        lifecycle.transition(IntentState.AWAITING_AUTH)

    # Step 5: execute
    db = get_db()
    try:
        results, summary = AgentCoordinator(db).execute(goal_spec, lifecycle)
        out: dict = {"ok": True, "results": results, "summary": summary}
    except LeavesError as e:
        out = {
            "ok":     False,
            "code":   e.code.value,
            "error":  e.user_message,
            "detail": e.detail,
        }
    except Exception as e:
        out = {"ok": False, "code": "INTERNAL_ERROR", "error": str(e), "detail": None}

    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    _main()
