"""
Tests for the observability module: JSON log formatter, Counter,
Histogram, Prometheus text rendering, and the /v1/metrics HTTP surface.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from observability import (  # noqa: E402
    Counter,
    Histogram,
    JsonLogFormatter,
    enforcer_rejections_total,
    inference_latency_ms,
    intents_total,
    render_prometheus_text,
)


# ---------------------------------------------------------------------------
# JSON log formatter
# ---------------------------------------------------------------------------


class TestJsonLogFormatter:
    def test_basic_record_emits_one_json_object(self):
        rec = logging.LogRecord(
            name="agentd", level=logging.INFO, pathname=__file__, lineno=1,
            msg="hello %s", args=("world",), exc_info=None,
        )
        out = JsonLogFormatter().format(rec)
        obj = json.loads(out)
        assert obj["msg"] == "hello world"
        assert obj["level"] == "INFO"
        assert obj["logger"] == "agentd"
        assert "ts" in obj and obj["ts"].endswith("Z")

    def test_extra_fields_are_merged(self):
        rec = logging.LogRecord(
            name="x", level=logging.WARNING, pathname=__file__, lineno=1,
            msg="boom", args=(), exc_info=None,
        )
        rec.__dict__["intent_id"] = "abc12345"
        rec.__dict__["duration_ms"] = 42.0
        obj = json.loads(JsonLogFormatter().format(rec))
        assert obj["intent_id"] == "abc12345"
        assert obj["duration_ms"] == 42.0

    def test_non_json_serializable_extra_falls_back_to_repr(self):
        class Weird:
            def __repr__(self):
                return "<weird>"

        rec = logging.LogRecord(
            name="x", level=logging.INFO, pathname=__file__, lineno=1,
            msg="m", args=(), exc_info=None,
        )
        rec.__dict__["thing"] = Weird()
        obj = json.loads(JsonLogFormatter().format(rec))
        assert obj["thing"] == "<weird>"

    def test_exception_info_is_included(self):
        try:
            raise ValueError("nope")
        except ValueError:
            rec = logging.LogRecord(
                name="x", level=logging.ERROR, pathname=__file__, lineno=1,
                msg="caught", args=(), exc_info=sys.exc_info(),
            )
        obj = json.loads(JsonLogFormatter().format(rec))
        assert "exc" in obj
        assert "ValueError" in obj["exc"]


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------


class TestCounter:
    def test_inc_with_labels(self):
        c = Counter("test_ctr", "help")
        c.inc(status="done", category="file_task")
        c.inc(status="done", category="file_task")
        c.inc(status="failed", category="web_task")
        out = "\n".join(c.render())
        assert 'test_ctr{category="file_task",status="done"} 2' in out
        assert 'test_ctr{category="web_task",status="failed"} 1' in out

    def test_empty_counter_renders_zero(self):
        c = Counter("empty_ctr", "h")
        out = "\n".join(c.render())
        assert "empty_ctr 0" in out

    def test_label_value_escaping(self):
        c = Counter("esc_ctr", "h")
        c.inc(reason='has "quote" and \\ slash')
        out = "\n".join(c.render())
        # Verify the raw line is parseable — no unescaped quote in label value
        assert 'esc_ctr{reason="has \\"quote\\" and \\\\ slash"} 1' in out

    def test_help_and_type_lines_present(self):
        c = Counter("hctr", "the help text")
        c.inc()
        out = c.render()
        assert out[0] == "# HELP hctr the help text"
        assert out[1] == "# TYPE hctr counter"


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


class TestHistogram:
    def test_buckets_are_cumulative(self):
        h = Histogram("hg", "h", buckets=[10, 100, 1000])
        for v in [5, 50, 500, 5000]:
            h.observe(v)
        out = "\n".join(h.render())
        assert 'hg_bucket{le="10"} 1' in out
        assert 'hg_bucket{le="100"} 2' in out
        assert 'hg_bucket{le="1000"} 3' in out
        assert 'hg_bucket{le="+Inf"} 4' in out
        assert "hg_sum 5555" in out
        assert "hg_count 4" in out

    def test_value_at_bucket_edge_counts_in_that_bucket(self):
        h = Histogram("edge", "h", buckets=[100])
        h.observe(100)
        out = "\n".join(h.render())
        assert 'edge_bucket{le="100"} 1' in out
        assert 'edge_bucket{le="+Inf"} 1' in out

    def test_empty_histogram_renders_zero_counts(self):
        h = Histogram("empty_hg", "h", buckets=[1, 10])
        out = "\n".join(h.render())
        assert 'empty_hg_bucket{le="1"} 0' in out
        assert 'empty_hg_bucket{le="+Inf"} 0' in out
        assert "empty_hg_count 0" in out


# ---------------------------------------------------------------------------
# Prometheus text rendering
# ---------------------------------------------------------------------------


class TestPrometheusText:
    def test_render_includes_all_named_metrics(self):
        text = render_prometheus_text()
        assert "leaves_intents_total" in text
        assert "leaves_inference_latency_ms_bucket" in text
        assert "leaves_inference_latency_ms_sum" in text
        assert "leaves_inference_latency_ms_count" in text
        assert "leaves_enforcer_rejections_total" in text
        assert text.endswith("\n"), "exposition must end with a newline"

    def test_round_trip_after_observation(self):
        intents_total.inc(status="done", category="file_task")
        inference_latency_ms.observe(750.0)
        enforcer_rejections_total.inc(reason="path_outside_resources")
        text = render_prometheus_text()
        # One HELP + one TYPE per registered metric block.
        named_metrics = [
            "leaves_intents_total",
            "leaves_inference_latency_ms",
            "leaves_enforcer_rejections_total",
            "leaves_runner_path_total",
        ]
        assert text.count("# HELP ") == len(named_metrics)
        assert text.count("# TYPE ") == len(named_metrics)
        for m in named_metrics:
            assert f"# HELP {m} " in text
            assert f"# TYPE {m} " in text


# ---------------------------------------------------------------------------
# /v1/metrics HTTP surface
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from api.server import app
        return TestClient(app)

    def test_metrics_endpoint_returns_prometheus_format(self, client):
        resp = client.get("/v1/metrics")
        assert resp.status_code == 200
        ct = resp.headers["content-type"]
        assert "text/plain" in ct
        assert "version=0.0.4" in ct
        body = resp.text
        assert "leaves_intents_total" in body
        assert "leaves_enforcer_rejections_total" in body
