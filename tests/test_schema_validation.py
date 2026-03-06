"""Tests for GoalSpec JSON schema validation."""
import json
import uuid
import pytest
import jsonschema
from pathlib import Path

SCHEMA = json.loads(Path("agents/schema/goal_spec.json").read_text())

VALID_MINIMAL = {
    "intent_id": "550e8400-e29b-41d4-a716-446655440000",
    "natural_text": "find all PDFs in Downloads",
    "category": "file_task",
    "actions": [{
        "action_id": "act-1",
        "type": "QUERY",
        "agent": "file",
        "params": {"path": "~/Downloads", "pattern": "*.pdf"},
        "destructive": False
    }],
    "authorization": {
        "resources": ["~/Downloads"],
        "preview_required": False,
        "reversible": True
    }
}


def _make(**overrides):
    import copy
    d = copy.deepcopy(VALID_MINIMAL)
    d.update(overrides)
    return d


def test_valid_minimal_intent():
    jsonschema.validate(VALID_MINIMAL, SCHEMA)


def test_valid_with_metadata():
    spec = _make(metadata={"confidence": 0.95, "parse_latency_ms": 312.5})
    jsonschema.validate(spec, SCHEMA)


def test_invalid_uuid_rejected():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_make(intent_id="not-a-uuid"), SCHEMA)


def test_missing_natural_text_rejected():
    spec = dict(VALID_MINIMAL)
    del spec["natural_text"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(spec, SCHEMA)


def test_empty_actions_rejected():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_make(actions=[]), SCHEMA)


def test_unknown_agent_rejected():
    import copy
    spec = copy.deepcopy(VALID_MINIMAL)
    spec["actions"][0]["agent"] = "robot"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(spec, SCHEMA)


def test_unknown_action_type_rejected():
    import copy
    spec = copy.deepcopy(VALID_MINIMAL)
    spec["actions"][0]["type"] = "EXPLODE"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(spec, SCHEMA)


def test_extra_fields_rejected():
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_make(malicious="injection"), SCHEMA)


def test_invalid_action_id_format_rejected():
    import copy
    spec = copy.deepcopy(VALID_MINIMAL)
    spec["actions"][0]["action_id"] = "action_1"  # wrong format
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(spec, SCHEMA)


def test_confidence_out_of_range_rejected():
    spec = _make(metadata={"confidence": 1.5})
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(spec, SCHEMA)
