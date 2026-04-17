"""
Runtime action-contract enforcer.

Validates every action against the GoalSpec's authorization bundle
BEFORE dispatching to an agent. This is the chokepoint that guarantees
agents cannot execute actions outside their planned scope.

Checks (in order):
  1. Action type is in the GoalSpec's planned action list (action_id match).
  2. Action type matches what was planned for that action_id.
  3. Every path-like param (whitelisted keys + string values that look
     like absolute filesystem paths) resolves to within
     authorization.resources. Symlinks are followed via Path.resolve().
  4. Destructive actions (DELETE, MOVE, WRITE, COPY) are only allowed
     when the planned type is also destructive.
  5. If the action references any path, authorization.resources MUST
     be non-empty — empty-resources is treated as "no paths allowed",
     not as "all paths allowed".

Authorization model for paths:
  - An authorized resource that exists and is a directory grants access
    to itself and any descendant.
  - An authorized resource that exists and is a file grants access to
    that exact path only — sibling files are NOT authorized.
  - An authorized resource that does NOT exist (e.g. a target write
    path) grants access to that exact path only. To write into a
    not-yet-created directory, authorize the parent directory.

On any violation: raises LeavesError(AUTHORIZATION_VIOLATION). The
caller is expected to audit-log the failure and surface a typed error
to the user / compositor.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from errors import LeavesError, LeavesErrorCode

_DESTRUCTIVE_TYPES = frozenset({"DELETE", "MOVE", "WRITE", "COPY"})

# Whitelisted param keys that are *always* treated as path-like, even
# if the value doesn't lead with /, ~, ./, ../. Agents conventionally
# use these names for filesystem paths.
_PATH_PARAM_KEYS = (
    "path", "source", "destination", "src", "dest", "target",
    "file", "filepath", "file_path", "filename", "output", "output_path",
)

# Schemes treated as non-filesystem and skipped during the
# "any string value that looks like a path" scan.
_NON_FS_SCHEMES = ("http://", "https://", "ftp://", "ftps://", "file://", "data:")


def enforce(action: dict[str, Any], goal_spec: dict[str, Any]) -> None:
    """
    Validate a single action against the GoalSpec. Raises
    LeavesError(AUTHORIZATION_VIOLATION) on any contract breach.

    Must be called BEFORE the agent executes the action.
    """
    action_id = action.get("action_id", "unknown")
    action_type = (action.get("type") or "").upper()
    params = action.get("params") or {}
    if not isinstance(params, dict):
        raise LeavesError(
            LeavesErrorCode.AUTHORIZATION_VIOLATION,
            detail=f"Action '{action_id}' has non-dict params: {type(params).__name__}",
        )

    planned_actions = goal_spec.get("actions") or []
    auth = goal_spec.get("authorization") or {}
    resources = auth.get("resources") or []

    # 1. Action must be in the plan
    planned_ids = {a.get("action_id") for a in planned_actions}
    if action_id not in planned_ids:
        raise LeavesError(
            LeavesErrorCode.AUTHORIZATION_VIOLATION,
            detail=(
                f"Action '{action_id}' is not in the planned GoalSpec. "
                f"Planned: {sorted(i for i in planned_ids if i)}"
            ),
        )

    # 2. Action type must match what was planned for this action_id
    planned_action = next(
        (a for a in planned_actions if a.get("action_id") == action_id), None
    )
    planned_type = (planned_action or {}).get("type", "").upper()
    if planned_action is None or action_type != planned_type:
        raise LeavesError(
            LeavesErrorCode.AUTHORIZATION_VIOLATION,
            detail=(
                f"Action '{action_id}' type mismatch: "
                f"planned '{planned_type}', got '{action_type}'"
            ),
        )

    # 3. All path-like params must be within authorized resources
    action_paths = _extract_paths(params)
    if action_paths:
        if not resources:
            raise LeavesError(
                LeavesErrorCode.AUTHORIZATION_VIOLATION,
                detail=(
                    f"Action '{action_id}' references paths "
                    f"{[raw for raw, _ in action_paths]} but the GoalSpec "
                    f"declares no authorized resources."
                ),
            )
        authorized = _resolve_resources(resources)
        for raw_path, resolved in action_paths:
            if not _path_within_authorized(resolved, authorized):
                raise LeavesError(
                    LeavesErrorCode.AUTHORIZATION_VIOLATION,
                    detail=(
                        f"Action '{action_id}' references path '{raw_path}' "
                        f"which resolves to '{resolved}' — outside authorized "
                        f"resources: {resources}"
                    ),
                )

    # 4. Destructive action type must match plan
    if action_type in _DESTRUCTIVE_TYPES and planned_type not in _DESTRUCTIVE_TYPES:
        raise LeavesError(
            LeavesErrorCode.AUTHORIZATION_VIOLATION,
            detail=(
                f"Action '{action_id}' is destructive ({action_type}) "
                f"but plan specifies non-destructive '{planned_type}'"
            ),
        )


def _looks_like_path(val: str) -> bool:
    """Cheap heuristic: does this string look like a filesystem path?"""
    if not val or len(val) > 4096:
        return False
    if any(val.startswith(s) for s in _NON_FS_SCHEMES):
        return False
    # Absolute, home-relative, or explicitly relative paths.
    return val.startswith(("/", "~/", "./", "../")) or val in ("~", ".", "..")


def _extract_paths(params: dict) -> list[tuple[str, Path]]:
    """
    Extract and resolve all path-like parameter values.

    Two passes:
      - Whitelist pass: known path-bearing keys are *always* checked,
        even if the value doesn't trigger _looks_like_path (covers
        bare filenames in 'path' etc.).
      - Scan pass: every other string value gets checked only if it
        looks path-like, catching agent-specific keys we don't know
        about (e.g. 'log_file', 'config_path').
    """
    pairs: list[tuple[str, Path]] = []
    seen: set[str] = set()

    for key in _PATH_PARAM_KEYS:
        val = params.get(key)
        if isinstance(val, str) and val and val not in seen:
            seen.add(val)
            resolved = _safe_resolve(val)
            if resolved is not None:
                pairs.append((val, resolved))

    for key, val in params.items():
        if key in _PATH_PARAM_KEYS or not isinstance(val, str):
            continue
        if val in seen or not _looks_like_path(val):
            continue
        seen.add(val)
        resolved = _safe_resolve(val)
        if resolved is not None:
            pairs.append((val, resolved))

    return pairs


def _safe_resolve(val: str) -> Path | None:
    try:
        return Path(val).expanduser().resolve()
    except (ValueError, OSError, RuntimeError):
        return None


def _resolve_resources(resources: list) -> list[Path]:
    resolved: list[Path] = []
    for r in resources:
        if not isinstance(r, str) or not r:
            continue
        p = _safe_resolve(r)
        if p is not None:
            resolved.append(p)
    return resolved


def _path_within_authorized(path: Path, authorized: list[Path]) -> bool:
    """
    True iff `path` is reachable via any authorized resource.

    Rules:
      - exact equality always wins.
      - if the authorized path exists and is a directory, any descendant
        is allowed.
      - if the authorized path is a file (existing) or does not yet
        exist, only the exact path is allowed. Sibling files in the
        same parent dir are NOT authorized — that was a real bug in
        the previous implementation that let any file in /home/x/
        through when /home/x/safe.txt was authorized.
    """
    for auth_path in authorized:
        if path == auth_path:
            return True
        try:
            if auth_path.is_dir():
                path.relative_to(auth_path)
                return True
        except (ValueError, OSError):
            continue
    return False
