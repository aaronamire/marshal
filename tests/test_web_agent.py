"""
Unit tests for WebAgent — search and fetch via DuckDuckGo + requests.
All HTTP calls are mocked; no real network requests are made.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.web_agent import WebAgent
from errors import MarshalError, MarshalErrorCode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn():
    """Stub DB connection — audit logging is non-fatal so a mock suffices."""
    conn = MagicMock()
    conn.execute = MagicMock(return_value=MagicMock(lastrowid=1))
    return conn


@pytest.fixture
def agent(db_conn):
    """WebAgent with mocked audit DB."""
    with patch("agents.base_agent.log_action_started", return_value=1), \
         patch("agents.base_agent.log_action_completed"):
        a = WebAgent(intent_id="test-intent-001", db_conn=db_conn)
    return a


def _action(action_type="QUERY", query_type=None, **extra):
    """Helper to build an action dict."""
    params = dict(extra)
    if query_type is not None:
        params["query_type"] = query_type
    return {"action_id": "act-1", "type": action_type, "params": params}


# ---------------------------------------------------------------------------
# QUERY / search — successful
# ---------------------------------------------------------------------------

class TestSearchSuccess:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_search_returns_results(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-001", db_conn)
        ddgs_instance = MagicMock()
        ddgs_instance.__enter__ = MagicMock(return_value=ddgs_instance)
        ddgs_instance.__exit__ = MagicMock(return_value=False)
        ddgs_instance.text.return_value = [
            {"title": "Python docs", "href": "https://python.org", "body": "Welcome to Python"},
            {"title": "PyPI", "href": "https://pypi.org", "body": "The Python Package Index"},
        ]

        with patch("agents.web_agent.DDGS", return_value=ddgs_instance, create=True) as mock_ddgs:
            # Patch the import inside _web_search
            with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=MagicMock(return_value=ddgs_instance))}):
                # Directly call _web_search to avoid import indirection
                result = agent._web_search("python programming")

        assert result["query"] == "python programming"
        assert result["result_count"] == 2
        assert len(result["results"]) == 2
        assert result["results"][0]["title"] == "Python docs"
        assert result["results"][0]["url"] == "https://python.org"
        assert result["results"][0]["snippet"] == "Welcome to Python"

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_search_no_results(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-002", db_conn)
        ddgs_instance = MagicMock()
        ddgs_instance.__enter__ = MagicMock(return_value=ddgs_instance)
        ddgs_instance.__exit__ = MagicMock(return_value=False)
        ddgs_instance.text.return_value = []

        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=MagicMock(return_value=ddgs_instance))}):
            result = agent._web_search("xyzzy nonexistent query 9999")

        assert result["query"] == "xyzzy nonexistent query 9999"
        assert result["result_count"] == 0
        assert result["results"] == []

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_search_snippet_truncated_to_300(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-003", db_conn)
        long_body = "A" * 500
        ddgs_instance = MagicMock()
        ddgs_instance.__enter__ = MagicMock(return_value=ddgs_instance)
        ddgs_instance.__exit__ = MagicMock(return_value=False)
        ddgs_instance.text.return_value = [
            {"title": "Long", "href": "https://example.com", "body": long_body},
        ]

        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=MagicMock(return_value=ddgs_instance))}):
            result = agent._web_search("long body test")

        assert len(result["results"][0]["snippet"]) == 300


# ---------------------------------------------------------------------------
# QUERY / search — errors
# ---------------------------------------------------------------------------

class TestSearchErrors:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_empty_query_raises(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-err-1", db_conn)
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(_action("QUERY", query_type="search", query=""))
        assert exc_info.value.code == MarshalErrorCode.INFERENCE_BAD_RESPONSE
        assert "No query" in exc_info.value.detail

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_missing_query_param_raises(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-err-2", db_conn)
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action({"action_id": "a1", "type": "QUERY",
                                  "params": {"query_type": "search"}})
        assert exc_info.value.code == MarshalErrorCode.INFERENCE_BAD_RESPONSE

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_ddgs_api_failure_returns_error_dict(self, _audit_s, _audit_e, db_conn):
        """DuckDuckGo raising inside _web_search returns error dict, not exception."""
        agent = WebAgent("test-err-3", db_conn)
        ddgs_instance = MagicMock()
        ddgs_instance.__enter__ = MagicMock(return_value=ddgs_instance)
        ddgs_instance.__exit__ = MagicMock(return_value=False)
        ddgs_instance.text.side_effect = Exception("DuckDuckGo rate limit")

        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=MagicMock(return_value=ddgs_instance))}):
            result = agent._web_search("test query")

        assert "error" in result
        assert "rate limit" in result["error"]
        assert result["results"] == []

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_ddgs_not_installed_returns_error(self, _audit_s, _audit_e, db_conn):
        """When ddgs module is not importable, _web_search returns an error dict."""
        agent = WebAgent("test-err-4", db_conn)
        # Temporarily hide the ddgs module
        import importlib
        with patch.dict("sys.modules", {"ddgs": None}):
            # Force ImportError by making the import fail
            result = agent._web_search("anything")
        assert "error" in result
        assert "ddgs" in result["error"].lower() or "not installed" in result["error"].lower()


# ---------------------------------------------------------------------------
# QUERY / fetch — successful
# ---------------------------------------------------------------------------

class TestFetchSuccess:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_fetch_returns_content(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-fetch-1", db_conn)
        html = """
        <html>
        <head><title>Test Page</title></head>
        <body>
        <nav>Navigation</nav>
        <main><p>Hello world content here.</p></main>
        <footer>Footer stuff</footer>
        </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = agent._web_fetch("https://example.com")

        assert result["url"] == "https://example.com"
        assert result["title"] == "Test Page"
        assert "Hello world" in result["content"]
        # nav and footer should be stripped
        assert "Navigation" not in result["content"]
        assert "Footer stuff" not in result["content"]
        assert "content_length" in result

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_fetch_adds_https_scheme(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-fetch-2", db_conn)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><head><title>T</title></head><body>Content</body></html>"
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = agent._web_fetch("example.com")

        assert result["url"] == "https://example.com"
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert call_url == "https://example.com"

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_fetch_truncates_long_content(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-fetch-3", db_conn)
        long_text = "A" * 5000
        html = f"<html><head><title>Big</title></head><body><p>{long_text}</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            result = agent._web_fetch("https://example.com/big")

        assert "[truncated]" in result["content"]
        # content_length counts up to the truncation point
        assert result["content_length"] <= 3100  # 3000 + "[truncated]" + newline

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_fetch_strips_script_and_style(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-fetch-4", db_conn)
        html = """
        <html><head><title>T</title>
        <style>body { color: red; }</style>
        </head><body>
        <script>alert('xss')</script>
        <p>Actual content</p>
        </body></html>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            result = agent._web_fetch("https://example.com")

        assert "alert" not in result["content"]
        assert "color: red" not in result["content"]
        assert "Actual content" in result["content"]


# ---------------------------------------------------------------------------
# QUERY / fetch — errors
# ---------------------------------------------------------------------------

class TestFetchErrors:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_empty_url_raises(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-ferr-1", db_conn)
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(_action("QUERY", query_type="fetch", url=""))
        assert exc_info.value.code == MarshalErrorCode.INFERENCE_BAD_RESPONSE
        assert "No URL" in exc_info.value.detail

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_missing_url_param_raises(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-ferr-2", db_conn)
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action({"action_id": "a1", "type": "QUERY",
                                  "params": {"query_type": "fetch"}})
        assert exc_info.value.code == MarshalErrorCode.INFERENCE_BAD_RESPONSE

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_timeout_returns_error_dict(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-ferr-3", db_conn)
        import requests as real_requests

        with patch("requests.get", side_effect=real_requests.exceptions.Timeout("timed out")):
            result = agent._web_fetch("https://slow-site.example.com")

        assert "error" in result
        assert "timed out" in result["error"].lower()
        assert result["url"] == "https://slow-site.example.com"

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_http_error_returns_status(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-ferr-4", db_conn)
        import requests as real_requests

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        http_err = real_requests.exceptions.HTTPError(response=mock_resp)
        mock_resp.raise_for_status.side_effect = http_err

        with patch("requests.get", return_value=mock_resp):
            result = agent._web_fetch("https://example.com/missing")

        assert "error" in result
        assert "404" in result["error"]

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_connection_error_returns_error_dict(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-ferr-5", db_conn)
        import requests as real_requests

        with patch("requests.get", side_effect=real_requests.exceptions.ConnectionError("DNS failed")):
            result = agent._web_fetch("https://nonexistent.invalid")

        assert "error" in result
        assert result["url"] == "https://nonexistent.invalid"


# ---------------------------------------------------------------------------
# Action dispatch / edge cases
# ---------------------------------------------------------------------------

class TestActionDispatch:

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_invalid_action_type_raises(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-disp-1", db_conn)
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(_action("WRITE", query_type="search", query="test"))
        assert exc_info.value.code == MarshalErrorCode.AGENT_NOT_AVAILABLE
        assert "WRITE" in exc_info.value.detail

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_delete_action_type_raises(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-disp-2", db_conn)
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(_action("DELETE", query_type="fetch", url="http://x.com"))
        assert exc_info.value.code == MarshalErrorCode.AGENT_NOT_AVAILABLE

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_read_action_type_raises(self, _audit_s, _audit_e, db_conn):
        agent = WebAgent("test-disp-3", db_conn)
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action(_action("READ", query_type="search", query="test"))
        assert exc_info.value.code == MarshalErrorCode.AGENT_NOT_AVAILABLE

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_missing_query_type_defaults_to_search(self, _audit_s, _audit_e, db_conn):
        """When query_type is omitted, execute_action defaults to 'search'."""
        agent = WebAgent("test-disp-4", db_conn)
        ddgs_instance = MagicMock()
        ddgs_instance.__enter__ = MagicMock(return_value=ddgs_instance)
        ddgs_instance.__exit__ = MagicMock(return_value=False)
        ddgs_instance.text.return_value = [
            {"title": "Result", "href": "https://r.com", "body": "Snippet"},
        ]

        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=MagicMock(return_value=ddgs_instance))}):
            result = agent.execute_action(
                {"action_id": "a1", "type": "QUERY",
                 "params": {"query": "default search test"}})

        assert result["query"] == "default search test"
        assert result["result_count"] == 1

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_unknown_query_type_falls_through_to_search(self, _audit_s, _audit_e, db_conn):
        """Unknown query_type (e.g. 'summarize') falls through to search branch."""
        agent = WebAgent("test-disp-5", db_conn)
        ddgs_instance = MagicMock()
        ddgs_instance.__enter__ = MagicMock(return_value=ddgs_instance)
        ddgs_instance.__exit__ = MagicMock(return_value=False)
        ddgs_instance.text.return_value = []

        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=MagicMock(return_value=ddgs_instance))}):
            result = agent.execute_action(
                {"action_id": "a1", "type": "QUERY",
                 "params": {"query_type": "summarize", "query": "test"}})

        assert result["result_count"] == 0
        assert result["results"] == []

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_missing_params_entirely(self, _audit_s, _audit_e, db_conn):
        """Action with no params at all — defaults to search, empty query raises."""
        agent = WebAgent("test-disp-6", db_conn)
        with pytest.raises(MarshalError) as exc_info:
            agent.execute_action({"action_id": "a1", "type": "QUERY", "params": {}})
        assert exc_info.value.code == MarshalErrorCode.INFERENCE_BAD_RESPONSE

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_action_type_case_insensitive(self, _audit_s, _audit_e, db_conn):
        """action type 'query' (lowercase) should work."""
        agent = WebAgent("test-disp-7", db_conn)
        ddgs_instance = MagicMock()
        ddgs_instance.__enter__ = MagicMock(return_value=ddgs_instance)
        ddgs_instance.__exit__ = MagicMock(return_value=False)
        ddgs_instance.text.return_value = []

        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=MagicMock(return_value=ddgs_instance))}):
            result = agent.execute_action(
                {"action_id": "a1", "type": "query",
                 "params": {"query_type": "search", "query": "test"}})

        assert "results" in result

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_search_uses_path_as_fallback_query(self, _audit_s, _audit_e, db_conn):
        """When 'query' param is missing, 'path' is used as fallback."""
        agent = WebAgent("test-disp-8", db_conn)
        ddgs_instance = MagicMock()
        ddgs_instance.__enter__ = MagicMock(return_value=ddgs_instance)
        ddgs_instance.__exit__ = MagicMock(return_value=False)
        ddgs_instance.text.return_value = [
            {"title": "R", "href": "https://r.com", "body": "B"},
        ]

        with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=MagicMock(return_value=ddgs_instance))}):
            result = agent.execute_action(
                {"action_id": "a1", "type": "QUERY",
                 "params": {"query_type": "search", "path": "python tutorials"}})

        assert result["query"] == "python tutorials"

    @patch("agents.base_agent.log_action_completed")
    @patch("agents.base_agent.log_action_started", return_value=1)
    def test_fetch_uses_path_as_fallback_url(self, _audit_s, _audit_e, db_conn):
        """When 'url' param is missing, 'path' is used as fallback."""
        agent = WebAgent("test-disp-9", db_conn)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><head><title>T</title></head><body>Ok</body></html>"
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            result = agent.execute_action(
                {"action_id": "a1", "type": "QUERY",
                 "params": {"query_type": "fetch", "path": "https://example.com"}})

        assert result["url"] == "https://example.com"


# ---------------------------------------------------------------------------
# Audit logging integration
# ---------------------------------------------------------------------------

class TestAuditLogging:

    def test_successful_search_calls_audit_start_and_end(self, db_conn):
        with patch("agents.base_agent.log_action_started", return_value=42) as mock_start, \
             patch("agents.base_agent.log_action_completed") as mock_end:
            agent = WebAgent("test-audit-1", db_conn)

            ddgs_instance = MagicMock()
            ddgs_instance.__enter__ = MagicMock(return_value=ddgs_instance)
            ddgs_instance.__exit__ = MagicMock(return_value=False)
            ddgs_instance.text.return_value = []

            with patch.dict("sys.modules", {"ddgs": MagicMock(DDGS=MagicMock(return_value=ddgs_instance))}):
                agent.execute_action(_action("QUERY", query_type="search", query="test"))

            mock_start.assert_called_once()
            mock_end.assert_called_once()
            # audit_end should be called with the row_id and result (no error)
            end_call = mock_end.call_args
            assert end_call.kwargs.get("row_id") == 42 or end_call[1].get("row_id") == 42

    def test_fetch_error_calls_audit_end(self, db_conn):
        import requests as real_requests

        with patch("agents.base_agent.log_action_started", return_value=7) as mock_start, \
             patch("agents.base_agent.log_action_completed") as mock_end:
            agent = WebAgent("test-audit-2", db_conn)

            with patch("requests.get", side_effect=real_requests.exceptions.Timeout("slow")):
                result = agent.execute_action(
                    _action("QUERY", query_type="fetch", url="https://slow.example.com"))

            # Even on timeout (which returns error dict, not exception), audit completes
            mock_start.assert_called_once()
            mock_end.assert_called_once()


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

class TestAgentMeta:

    def test_agent_type_is_web(self, db_conn):
        with patch("agents.base_agent.log_action_started"), \
             patch("agents.base_agent.log_action_completed"):
            agent = WebAgent("test-meta-1", db_conn)
        assert agent.AGENT_TYPE == "web"

    def test_inherits_base_agent(self, db_conn):
        from agents.base_agent import BaseAgent
        with patch("agents.base_agent.log_action_started"), \
             patch("agents.base_agent.log_action_completed"):
            agent = WebAgent("test-meta-2", db_conn)
        assert isinstance(agent, BaseAgent)
