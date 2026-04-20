"""
End-to-end integration test for the Marshal pipeline.

Tests the full flow: parse → audit → execute → history → detail → replay.
Uses real SQLite audit DB but mocks inference and agent execution.
Verifies that all pipeline seams (parser → DB → API → agentd → audit) hold.
"""
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def audit_db(tmp_path):
    from db.audit import get_db
    return get_db(tmp_path / "integration.db")


@pytest.fixture
def read_goalspec():
    return {
        "intent_id": "integ-read-001",
        "natural_text": "list files in ~/Downloads",
        "category": "file_task",
        "actions": [
            {
                "action_id": "a1",
                "type": "QUERY",
                "agent": "file",
                "params": {"path": "~/Downloads", "pattern": "*"},
            }
        ],
        "authorization": {
            "resources": ["~/Downloads"],
            "preview_required": False,
            "reversible": True,
        },
        "metadata": {"parse_latency_ms": 42.0},
    }


@pytest.fixture
def delete_goalspec():
    return {
        "intent_id": "integ-del-001",
        "natural_text": "delete old logs in ~/logs",
        "category": "file_task",
        "actions": [
            {
                "action_id": "a1",
                "type": "DELETE",
                "agent": "file",
                "params": {"path": "~/logs", "pattern": "*.log"},
            }
        ],
        "authorization": {
            "resources": ["~/logs"],
            "preview_required": True,
            "reversible": False,
        },
        "metadata": {"parse_latency_ms": 55.0},
    }


@pytest.fixture
def api_client(audit_db, read_goalspec):
    mock_parser = MagicMock()
    mock_parser.parse.return_value = read_goalspec

    with patch("api.server._get_parser", return_value=mock_parser), \
         patch("api.server._get_db", return_value=audit_db), \
         patch("api.server._fetch_session_context", new_callable=AsyncMock, return_value=None), \
         patch("api.server._SOCK_PATH", MagicMock(exists=MagicMock(return_value=True))), \
         patch("api.server._inference_healthy", True):
        from api.server import app
        transport = httpx.ASGITransport(app=app)
        c = httpx.AsyncClient(transport=transport, base_url="http://test")
        yield c


# ---------------------------------------------------------------------------
# Full pipeline: plan → execute → history → detail → replay
# ---------------------------------------------------------------------------

class TestFullPipeline:

    @pytest.mark.asyncio
    async def test_plan_execute_history_detail_replay(
        self, api_client, audit_db, read_goalspec
    ):
        # 1. Plan
        plan_resp = await api_client.post(
            "/v1/intent/plan", json={"text": "list files in ~/Downloads"})
        assert plan_resp.status_code == 200
        plan_data = plan_resp.json()
        assert plan_data["status"] == "planned"
        intent_id = plan_data["intent_id"]
        assert intent_id == "integ-read-001"

        # 2. Execute
        exec_results = {"a1": {"count": 3, "files": [
            {"name": "a.txt", "path": "~/Downloads/a.txt"},
            {"name": "b.pdf", "path": "~/Downloads/b.pdf"},
            {"name": "c.zip", "path": "~/Downloads/c.zip"},
        ]}}
        exec_sandbox = {"active": True, "reason": "landlock",
                        "authorized_resources": ["~/Downloads"]}
        with patch("api.server._send_goalspec", new_callable=AsyncMock,
                   return_value=(exec_results, "3 files found", exec_sandbox)):
            exec_resp = await api_client.post(
                "/v1/intent/execute", json={"intent_id": intent_id})
        assert exec_resp.status_code == 200
        exec_data = exec_resp.json()
        assert exec_data["status"] == "done"
        assert exec_data["sandbox_active"] is True

        # 3. History lists it
        hist_resp = await api_client.get("/v1/history")
        hist = hist_resp.json()
        assert any(h["intent_id"] == intent_id for h in hist)

        # 4. Detail returns audit trace
        detail_resp = await api_client.get(f"/v1/history/{intent_id}/detail")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["intent_id"] == intent_id
        assert detail["state"] == "DONE"
        assert detail["goal_spec"] is not None
        assert detail["goal_spec"]["category"] == "file_task"

        # 5. Replay (QUERY is non-destructive, should succeed)
        replay_results = {"a1": {"count": 3}}
        replay_sandbox = {"active": True, "reason": "landlock",
                          "authorized_resources": ["~/Downloads"]}
        with patch("api.server._send_goalspec", new_callable=AsyncMock,
                   return_value=(replay_results, "3 files", replay_sandbox)):
            replay_resp = await api_client.post(
                f"/v1/history/{intent_id}/replay")
        assert replay_resp.status_code == 200
        replay_data = replay_resp.json()
        assert replay_data["status"] == "done"
        assert replay_data["matches_state"] is True
        assert replay_data["replay"]["intent_id"] != intent_id

    @pytest.mark.asyncio
    async def test_destructive_replay_refused(
        self, api_client, audit_db, delete_goalspec
    ):
        mock_parser = MagicMock()
        mock_parser.parse.return_value = delete_goalspec
        with patch("api.server._get_parser", return_value=mock_parser):
            await api_client.post(
                "/v1/intent/plan", json={"text": "delete old logs"})

        exec_results = {"a1": {"deleted_count": 5}}
        exec_sandbox = {"active": True, "reason": "landlock",
                        "authorized_resources": ["~/logs"]}
        with patch("api.server._send_goalspec", new_callable=AsyncMock,
                   return_value=(exec_results, "5 deleted", exec_sandbox)):
            await api_client.post(
                "/v1/intent/execute",
                json={"intent_id": "integ-del-001"})

        replay_resp = await api_client.post(
            "/v1/history/integ-del-001/replay")
        data = replay_resp.json()
        assert data["status"] == "refused"
        assert data["reason"] == "destructive_replay_blocked"


# ---------------------------------------------------------------------------
# Audit DB integrity
# ---------------------------------------------------------------------------

class TestAuditIntegrity:

    def test_intent_lifecycle_in_db(self, audit_db):
        from db.audit import (
            log_intent_created, log_state_transition, log_action_started,
            log_action_completed, complete_intent, get_recent_intents,
            get_intent_actions, get_intent_transitions,
        )

        iid = "audit-test-001"
        gs = {
            "intent_id": iid,
            "natural_text": "test intent",
            "category": "file_task",
            "actions": [{"action_id": "a1", "type": "QUERY", "agent": "file"}],
        }

        log_intent_created(audit_db, iid, "test intent", gs)
        log_state_transition(audit_db, iid, "PENDING", "EXECUTING")

        row_id = log_action_started(audit_db, iid, "a1", "QUERY", "file",
                                    {"path": "~"})
        assert row_id is not None
        log_action_completed(audit_db, row_id,
                             result={"count": 0, "files": []})

        log_state_transition(audit_db, iid, "EXECUTING", "DONE")
        complete_intent(audit_db, iid, "DONE", "0 files found", 150.0)

        # Verify DB state
        intents = get_recent_intents(audit_db, 10)
        assert len(intents) == 1
        assert intents[0]["intent_id"] == iid
        assert intents[0]["state"] == "DONE"
        assert intents[0]["duration_ms"] == 150.0

        actions = get_intent_actions(audit_db, iid)
        assert len(actions) == 1
        assert actions[0]["action_type"] == "QUERY"
        assert actions[0]["error_code"] is None
        result = json.loads(actions[0]["result_json"])
        assert result["count"] == 0

        trans = get_intent_transitions(audit_db, iid)
        assert len(trans) == 2
        assert trans[0]["from_state"] == "PENDING"
        assert trans[0]["to_state"] == "EXECUTING"
        assert trans[1]["from_state"] == "EXECUTING"
        assert trans[1]["to_state"] == "DONE"

    def test_goalspec_stored_and_retrievable(self, audit_db):
        from db.audit import log_intent_created, complete_intent
        gs = {
            "intent_id": "gs-store-test",
            "natural_text": "read file",
            "category": "file_task",
            "actions": [{"action_id": "a1", "type": "READ", "agent": "file",
                         "params": {"path": "~/test.txt"}}],
            "authorization": {"resources": ["~"]},
        }
        log_intent_created(audit_db, "gs-store-test", "read file", gs)
        complete_intent(audit_db, "gs-store-test", "DONE", "read ok", 50.0)

        row = audit_db.execute(
            "SELECT goal_spec_json FROM intents WHERE intent_id = ?",
            ("gs-store-test",)).fetchone()
        stored = json.loads(row["goal_spec_json"])
        assert stored["intent_id"] == "gs-store-test"
        assert stored["actions"][0]["params"]["path"] == "~/test.txt"


# ---------------------------------------------------------------------------
# Enforcer → Agent → Audit chain
# ---------------------------------------------------------------------------

class TestEnforcerAgentAudit:

    def test_enforcer_blocks_unauthorized_path(self, audit_db):
        from agents.enforcer import enforce
        from errors import MarshalError, MarshalErrorCode

        goal_spec = {
            "intent_id": "enforcer-test",
            "authorization": {"resources": ["/home/user/safe"]},
            "actions": [{"action_id": "a1", "type": "READ", "agent": "file",
                         "params": {"path": "/etc/shadow"}}],
        }
        with pytest.raises(MarshalError) as exc:
            enforce(goal_spec["actions"][0], goal_spec)
        assert exc.value.code == MarshalErrorCode.AUTHORIZATION_VIOLATION

    def test_enforcer_allows_authorized_path(self, audit_db, tmp_path):
        from agents.enforcer import enforce
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        (safe_dir / "file.txt").write_text("ok")

        goal_spec = {
            "intent_id": "enforcer-pass",
            "authorization": {"resources": [str(safe_dir)]},
            "actions": [{"action_id": "a1", "type": "READ", "agent": "file",
                         "params": {"path": str(safe_dir / "file.txt")}}],
        }
        enforce(goal_spec["actions"][0], goal_spec)
