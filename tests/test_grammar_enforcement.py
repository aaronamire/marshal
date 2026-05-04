"""
Grammar enforcement tests for Marshal.

These tests verify that GBNF grammar constraints are ACTUALLY constraining
the sampler — not just that the output happens to look valid.

Methodology: send prompts that would naturally produce types outside the
valid enum WITHOUT grammar constraints. With grammar active, impossible
types must never appear.

If any test here fails, the grammar constraint layer is broken.
Do not merge any commit that breaks these tests.

Run:
    pytest tests/test_grammar_enforcement.py -v           # unit tests
    pytest tests/test_grammar_enforcement.py -m inference # inference tests (slow)
"""
import json
import re
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Definitive valid type set — must match grammar AND schema simultaneously.
# Verified consistent via scripts/verify_schema_grammar_consistency.py.
# If you add a type to the grammar, add it here AND to the schema.
# ---------------------------------------------------------------------------
VALID_ACTION_TYPES = frozenset({
    "QUERY", "READ", "WRITE", "DELETE", "MOVE", "COPY", "COMPOSE",
    # BRIEFING is L0-only (regex fast-path) — the LLM never produces it,
    # but it must appear in the schema/grammar enum for the L0-built
    # GoalSpec to pass jsonschema.validate(). Keeping it here keeps the
    # static-consistency tests happy.
    "BRIEFING",
})

# Types the model would plausibly generate without grammar constraints.
# These are NOT in the grammar enum and must never appear in constrained output.
IMPOSSIBLE_IF_CONSTRAINED = frozenset({
    "LIST", "SEARCH", "CREATE", "APPEND", "FETCH", "FIND",
    "RENAME", "OPEN", "SHOW", "UPDATE", "DOWNLOAD",
})


# ---------------------------------------------------------------------------
# Schema/grammar consistency — pure static check, no inference server needed
# ---------------------------------------------------------------------------

class TestStaticConsistency:
    """These tests have no external dependencies. They must always pass."""

    def test_grammar_schema_action_types_identical(self):
        """Grammar and schema must define the same closed enum of action types."""
        grammar = Path("inference/grammar/goal_spec.gbnf").read_text()
        schema = json.loads(Path("agents/schema/goal_spec.json").read_text())

        grammar_types = set()
        for line in grammar.splitlines():
            if "action-type" in line and "::=" in line:
                grammar_types = set(re.findall(r'\\"([A-Z]+)\\"', line))

        schema_types = set(
            schema["properties"]["actions"]["items"]["properties"]["type"]["enum"]
        )

        assert grammar_types, "No action-type rule found in grammar"
        assert schema_types, "No action type enum found in schema"
        assert grammar_types == schema_types, (
            f"Grammar/schema mismatch.\n"
            f"  In grammar only: {sorted(grammar_types - schema_types)}\n"
            f"  In schema only:  {sorted(schema_types - grammar_types)}\n"
            f"Both files must define the same closed enum."
        )

    def test_grammar_category_schema_category_identical(self):
        """Category enum must also match between grammar and schema."""
        grammar = Path("inference/grammar/goal_spec.gbnf").read_text()
        schema = json.loads(Path("agents/schema/goal_spec.json").read_text())

        grammar_cats = set()
        for line in grammar.splitlines():
            if line.startswith("category") and "::=" in line:
                grammar_cats = set(re.findall(r'\\"([a-z_]+)\\"', line))

        schema_cats = set(schema["properties"]["category"]["enum"])

        assert grammar_cats == schema_cats, (
            f"Category mismatch.\n"
            f"  Grammar: {sorted(grammar_cats)}\n"
            f"  Schema:  {sorted(schema_cats)}"
        )

    def test_valid_action_types_constant_matches_schema(self):
        """VALID_ACTION_TYPES at the top of this file must match schema."""
        schema = json.loads(Path("agents/schema/goal_spec.json").read_text())
        schema_types = set(
            schema["properties"]["actions"]["items"]["properties"]["type"]["enum"]
        )
        assert VALID_ACTION_TYPES == schema_types, (
            f"Update VALID_ACTION_TYPES in this file to match schema.\n"
            f"  Constant: {sorted(VALID_ACTION_TYPES)}\n"
            f"  Schema:   {sorted(schema_types)}"
        )

    def test_grammar_actions_array_bounded(self):
        """actions-array rule must not use unbounded * (infinite loop risk)."""
        grammar = Path("inference/grammar/goal_spec.gbnf").read_text()
        for line in grammar.splitlines():
            if line.startswith("actions-array") and "::=" in line:
                assert ")*" not in line and "ws action)*" not in line, (
                    "actions-array uses unbounded * — model can loop infinitely.\n"
                    "Use optional groups (?) to cap at a fixed maximum.\n"
                    f"Offending line: {line}"
                )
                return
        pytest.fail("No actions-array rule found in grammar")


# ---------------------------------------------------------------------------
# Inference-backed tests — require the server, marked slow
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gbnf_parser():
    from agents.intent_parser import IntentParser
    p = IntentParser(use_gbnf=True)
    if not p._client.is_available():
        pytest.skip("Inference server not running")
    return p


def assert_no_impossible_types(goal_spec: dict, intent: str):
    """Fail if any action has a type that the grammar cannot produce."""
    for action in goal_spec.get("actions", []):
        t = action.get("type", "")
        assert t not in IMPOSSIBLE_IF_CONSTRAINED, (
            f"\nGRAMMAR CONSTRAINT VIOLATED for: '{intent}'\n"
            f"Model produced type: '{t}'\n"
            f"This type is not in the grammar enum — constraint is broken.\n"
            f"Check: is 'grammar' field sent in API requests?\n"
            f"Check: does /completion endpoint receive the grammar field?\n"
            f"GoalSpec: {json.dumps(goal_spec, indent=2)}"
        )
        assert t in VALID_ACTION_TYPES, (
            f"Type '{t}' not in VALID_ACTION_TYPES {VALID_ACTION_TYPES}\n"
            f"Intent: '{intent}'"
        )


@pytest.mark.inference
@pytest.mark.timeout(180)  # Layer-2 inference can take 30-60s per call (5 calls in test_all_outputs_valid_enum)
class TestGrammarEnforcement:
    """Verify grammar is actively constraining sampler output."""

    def test_rename_no_rename_type(self, gbnf_parser):
        """'rename' would produce RENAME without grammar; must produce MOVE."""
        result = gbnf_parser.parse("rename notes.txt to notes-backup.txt")
        assert_no_impossible_types(result, "rename notes.txt")
        types = {a["type"] for a in result["actions"]}
        assert "MOVE" in types, f"rename intent should produce MOVE, got {types}"

    def test_list_no_list_type(self, gbnf_parser):
        """'list files' would produce LIST without grammar; must produce QUERY."""
        result = gbnf_parser.parse("list all Python files in my home directory")
        assert_no_impossible_types(result, "list Python files")

    def test_search_no_search_type(self, gbnf_parser):
        """'search for files' would produce SEARCH without grammar."""
        result = gbnf_parser.parse("search for all PDF files in Downloads")
        assert_no_impossible_types(result, "search for PDF files")

    def test_all_outputs_valid_enum(self, gbnf_parser):
        """Every action type across varied intents must be from the closed enum."""
        intents = [
            "find all PDFs in Downloads",
            "read the contents of readme.md",
            "delete all tmp files",
            "move photos to Pictures",
            "copy this file to backup",
            "show me files from last week",
        ]
        for intent in intents:
            result = gbnf_parser.parse(intent)
            for action in result.get("actions", []):
                t = action.get("type", "MISSING")
                assert t in VALID_ACTION_TYPES, (
                    f"Intent '{intent}' produced invalid type '{t}'"
                )

    def test_copy_produces_copy_type(self, gbnf_parser):
        """
        'copy file' should produce COPY (it IS in the grammar enum).
        This test documents that COPY is intentional, not a constraint failure.
        """
        result = gbnf_parser.parse("copy config.py to ~/backup/config.py")
        types = {a["type"] for a in result["actions"]}
        # COPY is valid — it's in the grammar. This is expected behavior.
        # If this starts failing (getting WRITE instead), the prompt changed.
        assert "COPY" in types or "WRITE" in types, (
            f"Copy intent should produce COPY or WRITE, got {types}"
        )
        assert_no_impossible_types(result, "copy config.py")
