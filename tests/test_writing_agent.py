"""
Unit tests for WritingAgent — text composition via inference.
Remote and local inference are mocked; no real API calls.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.writing_agent import WritingAgent
from errors import LeavesError, LeavesErrorCode
from inference.client import InferenceResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn():
    conn = MagicMock()
    conn.execute = MagicMock(return_value=MagicMock(lastrowid=1))
    return conn


def _mock_response(content="Generated text here."):
    return InferenceResponse(
        content=content,
        latency_ms=150.0,
        tokens_predicted=20,
        model="mock-model",
    )


def _action(topic, fmt="text", path=None):
    params = {"topic": topic, "format": fmt}
    if path:
        params["path"] = path
    return {"action_id": "act-1", "type": "COMPOSE", "params": params}


# ---------------------------------------------------------------------------
# COMPOSE — successful
# ---------------------------------------------------------------------------

class TestComposeSuccess:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    @patch("agents.writing_agent.RemoteAnthropicBackend")
    @patch("agents.writing_agent.InferenceClient")
    def test_compose_returns_content(self, MockClient, MockRemote, _s, _e, db_conn):
        mock_remote = MockRemote.return_value
        mock_remote.is_available.return_value = False
        mock_client = MockClient.return_value
        mock_client.complete.return_value = _mock_response("A great essay.")
        agent = WritingAgent("test-w1", db_conn)
        result = agent.execute_action(_action("write an essay about AI"))
        assert result["content"] == "A great essay."
        assert result["topic"] == "write an essay about AI"
        assert result["format"] == "text"

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    @patch("agents.writing_agent.RemoteAnthropicBackend")
    @patch("agents.writing_agent.InferenceClient")
    def test_compose_uses_remote_when_available(self, MockClient, MockRemote, _s, _e, db_conn):
        mock_remote = MockRemote.return_value
        mock_remote.is_available.return_value = True
        mock_client = MockClient.return_value
        mock_client.complete.return_value = _mock_response("Remote output")
        agent = WritingAgent("test-w2", db_conn)
        result = agent.execute_action(_action("write a poem"))
        MockClient.assert_called_once_with(backend=mock_remote)
        assert result["content"] == "Remote output"

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    @patch("agents.writing_agent.RemoteAnthropicBackend")
    @patch("agents.writing_agent.InferenceClient")
    def test_compose_writes_to_file(self, MockClient, MockRemote, _s, _e, db_conn, tmp_path):
        mock_remote = MockRemote.return_value
        mock_remote.is_available.return_value = False
        mock_client = MockClient.return_value
        mock_client.complete.return_value = _mock_response("File content")
        agent = WritingAgent("test-w3", db_conn)
        target = str(tmp_path / "output.txt")
        result = agent.execute_action(_action("write notes", path=target))
        assert result["path"] == target
        assert result["bytes_written"] == len("File content".encode())
        assert Path(target).read_text() == "File content"

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    @patch("agents.writing_agent.RemoteAnthropicBackend")
    @patch("agents.writing_agent.InferenceClient")
    def test_compose_creates_parent_dirs(self, MockClient, MockRemote, _s, _e, db_conn, tmp_path):
        mock_remote = MockRemote.return_value
        mock_remote.is_available.return_value = False
        mock_client = MockClient.return_value
        mock_client.complete.return_value = _mock_response("deep content")
        agent = WritingAgent("test-w4", db_conn)
        target = str(tmp_path / "sub" / "dir" / "output.txt")
        result = agent.execute_action(_action("write report", path=target))
        assert Path(target).read_text() == "deep content"

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    @patch("agents.writing_agent.RemoteAnthropicBackend")
    @patch("agents.writing_agent.InferenceClient")
    def test_write_action_type(self, MockClient, MockRemote, _s, _e, db_conn):
        mock_remote = MockRemote.return_value
        mock_remote.is_available.return_value = False
        mock_client = MockClient.return_value
        mock_client.complete.return_value = _mock_response("Written.")
        agent = WritingAgent("test-w5", db_conn)
        result = agent.execute_action({
            "action_id": "act-1", "type": "WRITE",
            "params": {"topic": "a letter"}})
        assert result["content"] == "Written."


# ---------------------------------------------------------------------------
# COMPOSE — errors
# ---------------------------------------------------------------------------

class TestComposeErrors:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_missing_topic_raises(self, _s, _e, db_conn):
        agent = WritingAgent("test-err1", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action({"action_id": "a1", "type": "COMPOSE",
                                  "params": {"format": "text"}})
        assert "topic" in exc.value.detail.lower()

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_empty_topic_raises(self, _s, _e, db_conn):
        agent = WritingAgent("test-err2", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action(_action(""))
        assert "topic" in exc.value.detail.lower()

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_unsupported_action_type_raises(self, _s, _e, db_conn):
        agent = WritingAgent("test-err3", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action({"action_id": "a1", "type": "DELETE",
                                  "params": {"topic": "anything"}})
        assert exc.value.code == LeavesErrorCode.AGENT_NOT_AVAILABLE

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    @patch("agents.writing_agent.RemoteAnthropicBackend")
    @patch("agents.writing_agent.InferenceClient")
    def test_inference_error_propagates(self, MockClient, MockRemote, _s, _e, db_conn):
        mock_remote = MockRemote.return_value
        mock_remote.is_available.return_value = False
        mock_client = MockClient.return_value
        mock_client.complete.side_effect = LeavesError(
            LeavesErrorCode.INFERENCE_UNAVAILABLE, detail="server down")
        agent = WritingAgent("test-err4", db_conn)
        with pytest.raises(LeavesError) as exc:
            agent.execute_action(_action("write something"))
        assert exc.value.code == LeavesErrorCode.INFERENCE_UNAVAILABLE


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

class TestMeta:

    def test_agent_type_is_writing(self, db_conn):
        with patch("agents.base_agent.log_action_started"), \
             patch("agents.base_agent.log_action_completed"):
            agent = WritingAgent("test-m1", db_conn)
        assert agent.AGENT_TYPE == "writing"

    def test_inherits_base_agent(self, db_conn):
        from agents.base_agent import BaseAgent
        with patch("agents.base_agent.log_action_started"), \
             patch("agents.base_agent.log_action_completed"):
            agent = WritingAgent("test-m2", db_conn)
        assert isinstance(agent, BaseAgent)
