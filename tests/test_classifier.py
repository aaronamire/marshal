"""
Tests for Layer 1 intent classifier.
Skipped automatically if model not trained yet.
"""
import time
from pathlib import Path

import pytest

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
