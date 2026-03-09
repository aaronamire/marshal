"""
Semantic validators for GoalSpec beyond JSON schema compliance.

JSON schema validates structure. These validators check semantics:
- Action ordering (depends_on DAG is valid and respected)
- Mutation safety (destructive actions follow their QUERY)
- Reference integrity (depends_on references valid action_ids)
- No cycles in the dependency graph
- Destructive-flag consistency with authorization block

Called by intent_parser.py after schema validation.
Also used directly by the eval harness for test-case verification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Action types that modify filesystem state.
MUTATION_TYPES = frozenset({"MOVE", "DELETE", "WRITE", "COPY", "TRANSFORM", "COMMUNICATE", "EXECUTE"})

# Action types that discover/read state without modifying it.
DISCOVERY_TYPES = frozenset({"QUERY", "READ"})


@dataclass
class ValidationError:
    code: str
    message: str
    action_id: Optional[str] = None


@dataclass
class ValidationResult:
    valid: bool = True
    errors: list[ValidationError] = field(default_factory=list)

    def add_error(self, code: str, message: str, action_id: str = None):
        self.errors.append(ValidationError(code, message, action_id))
        self.valid = False

    def __str__(self) -> str:
        if self.valid:
            return "valid"
        return "; ".join(f"[{e.code}] {e.message}" for e in self.errors)


def validate_action_ordering(goal_spec: dict) -> ValidationResult:
    """
    Validate that actions are ordered correctly and dependencies are respected.

    Rules enforced:
    1. action_ids are unique within the spec
    2. depends_on references only action_ids that exist
    3. depends_on references only action_ids that appear EARLIER in the list
       (list order = execution order; forward references are illegal)
    4. No dependency cycles
    5. Multi-action specs where first action is a mutation without an explicit
       specific path are flagged (probable ordering error)
    """
    result = ValidationResult()
    actions = goal_spec.get("actions", [])

    if not actions:
        result.add_error("NO_ACTIONS", "GoalSpec has no actions")
        return result

    # Build action_id → position index
    id_to_pos: dict[str, int] = {}
    for pos, action in enumerate(actions):
        aid = action.get("action_id", "")
        if not aid:
            result.add_error("MISSING_ACTION_ID", f"Action at position {pos} has no action_id")
            continue
        if aid in id_to_pos:
            result.add_error("DUPLICATE_ACTION_ID", f"action_id '{aid}' appears twice", aid)
        id_to_pos[aid] = pos

    # Validate depends_on: existence and ordering
    for pos, action in enumerate(actions):
        aid = action.get("action_id", f"pos_{pos}")
        for dep in action.get("depends_on", []):
            if dep == aid:
                # Self-reference: model error (e.g. act-1 depends_on act-1).
                # Meaningless but not execution-dangerous — soft error only.
                result.add_error(
                    "SELF_DEPENDENCY",
                    f"Action '{aid}' depends_on itself — ignored.",
                    aid,
                )
            elif dep not in id_to_pos:
                result.add_error(
                    "DANGLING_DEPENDENCY",
                    f"Action '{aid}' depends_on '{dep}' which does not exist",
                    aid,
                )
            elif id_to_pos[dep] >= pos:
                result.add_error(
                    "FORWARD_DEPENDENCY",
                    f"Action '{aid}' at position {pos} depends_on '{dep}' "
                    f"at position {id_to_pos[dep]} — dependencies must appear earlier "
                    f"in the list (list order = execution order).",
                    aid,
                )

    # Cycle detection via DFS
    adj: dict[str, list[str]] = {
        a.get("action_id", ""): a.get("depends_on", [])
        for a in actions if a.get("action_id")
    }

    def _has_cycle(node: str, visited: set, stack: set) -> bool:
        visited.add(node)
        stack.add(node)
        for dep in adj.get(node, []):
            if dep not in visited:
                if _has_cycle(dep, visited, stack):
                    return True
            elif dep in stack:
                return True
        stack.discard(node)
        return False

    visited: set = set()
    for aid in adj:
        if aid not in visited:
            if _has_cycle(aid, visited, set()):
                result.add_error("DEPENDENCY_CYCLE", f"Cycle detected involving '{aid}'", aid)

    # Semantic ordering: mutation as first action without explicit path
    if len(actions) > 1:
        first = actions[0]
        first_type = first.get("type", "")
        first_params = first.get("params", {})

        if first_type in MUTATION_TYPES:
            path = first_params.get("path", first_params.get("source", ""))
            pattern = first_params.get("pattern", "")
            # An explicit single-file path (no wildcards, not home dir alone)
            is_explicit = (
                path
                and not pattern
                and path not in {"~", "/", ""}
                and not path.endswith(("/", "*"))
            )
            if not is_explicit:
                actual = " → ".join(a.get("type", "?") for a in actions)
                result.add_error(
                    "MUTATION_WITHOUT_PRIOR_DISCOVERY",
                    f"First action '{first.get('action_id')}' is '{first_type}' "
                    f"(mutation) with no prior QUERY to identify target files. "
                    f"Correct pattern: QUERY → {first_type}. Got: {actual}",
                    first.get("action_id"),
                )

    return result


def validate_destructive_consistency(goal_spec: dict) -> ValidationResult:
    """
    Validate that destructive flags are consistent with the authorization block.

    Rules:
    - If any action is destructive=True, authorization.preview_required must be True
    - DELETE and MOVE actions must be flagged destructive
    """
    result = ValidationResult()
    actions = goal_spec.get("actions", [])
    auth = goal_spec.get("authorization", {})

    has_destructive = any(a.get("destructive", False) for a in actions)
    if has_destructive and not auth.get("preview_required", False):
        result.add_error(
            "DESTRUCTIVE_WITHOUT_PREVIEW",
            "A destructive action exists but authorization.preview_required=False. "
            "Destructive operations must require user preview.",
        )

    for action in actions:
        t = action.get("type", "")
        if t in {"DELETE", "MOVE"} and not action.get("destructive", False):
            result.add_error(
                "MUTATION_NOT_FLAGGED_DESTRUCTIVE",
                f"Action '{action.get('action_id')}' has type '{t}' "
                f"but destructive=False. {t} operations modify filesystem state.",
                action.get("action_id"),
            )

    return result


def validate_goal_spec(goal_spec: dict) -> ValidationResult:
    """
    Run all semantic validators and return an aggregated result.
    Called by intent_parser.py after JSON schema validation.
    """
    result = ValidationResult()

    for sub in (validate_action_ordering(goal_spec), validate_destructive_consistency(goal_spec)):
        if not sub.valid:
            for err in sub.errors:
                result.add_error(err.code, err.message, err.action_id)

    return result
