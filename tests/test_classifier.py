"""
Tests for Layer 1 intent classifier and its NOT_IMPLEMENTED fast-path integration.
Skipped automatically if model not trained yet.
"""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

PIPELINE_PATH = Path("models/layer1_pipeline.joblib")

pytestmark = pytest.mark.skipif(
    not PIPELINE_PATH.exists(),
    reason="Layer 1 model not trained. Run: python3 scripts/train_classifier.py",
)

from agents.classifier import IntentClassifier, ClassificationResult


@pytest.fixture(scope="module")
def clf():
    return IntentClassifier()


def test_file_task_classified(clf):
    r = clf.classify("find all PDFs in my Downloads folder")
    assert r.category == "file_task"
    assert r.confidence > 0.80


def test_email_task_classified(clf):
    r = clf.classify("find emails from last week")
    assert r.category == "email_task"
    assert r.confidence > 0.60


def test_web_task_classified(clf):
    r = clf.classify("search the web for Python tutorials")
    assert r.category == "web_task"
    assert r.confidence > 0.60


def test_system_task_classified(clf):
    r = clf.classify("show me how much RAM I am using")
    assert r.category == "system_task"
    assert r.confidence > 0.60


def test_writing_task_classified(clf):
    r = clf.classify("write a summary of my project")
    assert r.category == "writing_task"
    assert r.confidence > 0.60


def test_latency_under_50ms(clf):
    times = []
    for _ in range(20):
        t0 = time.monotonic()
        clf.classify("find all Python files in my home directory")
        times.append((time.monotonic() - t0) * 1000)
    avg = sum(times) / len(times)
    assert avg < 50, f"Average latency {avg:.1f}ms exceeds 50ms"


def test_empty_input_no_crash(clf):
    r = clf.classify("")
    assert r.category == "unknown"
    assert r.confidence == 0.0


def test_is_confident_flag(clf):
    r = clf.classify("find my PDFs")
    assert isinstance(r.is_confident, bool)


def test_all_scores_sum_to_one(clf):
    r = clf.classify("find my PDFs")
    assert "file_task" in r.all_scores
    assert "email_task" in r.all_scores
    assert abs(sum(r.all_scores.values()) - 1.0) < 0.01


class TestNotImplementedFastPath:
    """
    Verify that the L1 classifier fast-path in IntentParser skips the LLM
    for non-file categories (saves 18-51s per request).
    """

    @pytest.fixture
    def parser_with_mock_llm(self):
        from agents.classifier import IntentClassifier
        from agents.intent_parser import IntentParser
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        clf = IntentClassifier()
        return IntentParser(client=mock_client, classifier=clf), mock_client

    def test_email_fast_path_skips_llm(self, parser_with_mock_llm):
        parser, mock_client = parser_with_mock_llm
        from errors import MarshalError, MarshalErrorCode
        with pytest.raises(MarshalError) as exc_info:
            parser.parse("send an email to Alice about the project")
        assert exc_info.value.code == MarshalErrorCode.NOT_IMPLEMENTED
        assert not mock_client.complete.called, "LLM must not be called for email_task"

    def test_system_fast_path_skips_llm(self, parser_with_mock_llm):
        """System queries are handled by Layer 0 regex → SystemAgent.
        The LLM must never be called."""
        parser, mock_client = parser_with_mock_llm
        result = parser.parse("how much RAM is my computer using")
        assert result["category"] == "system_task"
        assert result["actions"][0]["agent"] == "system"
        assert result["actions"][0]["params"]["query_type"] == "memory"
        assert not mock_client.complete.called, "LLM must not be called for system_task"

    def test_file_task_reaches_llm(self, parser_with_mock_llm):
        parser, mock_client = parser_with_mock_llm
        mock_client.complete.side_effect = RuntimeError("mock LLM")
        with pytest.raises(RuntimeError, match="mock LLM"):
            parser.parse("find all PDFs in my Downloads folder")
        assert mock_client.complete.called, "LLM must be called for file_task"

    def test_fast_path_latency_under_10ms(self, parser_with_mock_llm):
        parser, mock_client = parser_with_mock_llm
        from errors import MarshalError
        t0 = time.monotonic()
        try:
            parser.parse("send an email to my boss")
        except MarshalError:
            pass
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 10, f"Fast-path took {elapsed_ms:.1f}ms — should be <10ms"
