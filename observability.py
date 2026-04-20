"""
Observability primitives for Marshal.

Two surfaces:

  1. configure_logging(level) — installs a stderr handler that emits one
     JSON object per log record. Replaces the ad-hoc `print(..., flush=True)`
     scattered across agentd; gives every line a timestamp, level, logger
     name, and any structured `extra` keys the call site supplies.

  2. metrics — a tiny in-process registry (Counter + Histogram) and a
     `render_prometheus_text()` that emits the Prometheus 0.0.4 text
     exposition format. We do NOT pull in prometheus_client; the surface
     we need is small enough that a 60-line implementation is cheaper
     than a dependency.

Named metrics exposed at /v1/metrics:

    marshal_intents_total{status,category}        Counter
    marshal_inference_latency_ms_bucket{le}       Histogram (+ _sum, _count)
    marshal_enforcer_rejections_total{reason}     Counter

Buckets for inference latency are tuned for the local llama.cpp path on
CPU: median request lands around 800-1500ms after KV warmup, so the
buckets cluster there and stretch out to the ~30s cold-start tail.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from typing import Any


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

# Standard LogRecord attributes — anything else on the record is treated as
# a structured `extra` and merged into the emitted JSON object.
_RESERVED_LOGRECORD_ATTRS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
})


class JsonLogFormatter(logging.Formatter):
    """One JSON object per log record, no pretty-printing."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            ) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOGRECORD_ATTRS or key.startswith("_"):
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


_logging_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """
    Install the JSON formatter on the root logger's stderr handler.
    Idempotent — calling it twice does not double-up handlers.
    """
    global _logging_configured
    if _logging_configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    _logging_configured = True


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def _label_key(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Hashable, order-stable key for a labels dict."""
    return tuple(sorted(labels.items()))


def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{_escape(v)}"' for k, v in labels)
    return "{" + inner + "}"


def _escape(s: str) -> str:
    # Prometheus label-value escaping: backslash, double-quote, newline.
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Counter:
    """Monotonically increasing counter, labelled."""

    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help = help_text
        self._values: dict[tuple[tuple[str, str], ...], float] = {}
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = _label_key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def render(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} counter",
        ]
        with self._lock:
            items = list(self._values.items())
        if not items:
            # Emit a zero so scrapers see the series exists.
            lines.append(f"{self.name} 0")
            return lines
        for key, val in sorted(items):
            lines.append(f"{self.name}{_format_labels(key)} {val:g}")
        return lines


class Histogram:
    """
    Fixed-bucket histogram. Cumulative bucket counts plus _sum and _count,
    matching the Prometheus exposition contract. No labels — we only
    instrument inference latency, which is one stream.
    """

    def __init__(self, name: str, help_text: str, buckets: list[float]):
        self.name = name
        self.help = help_text
        self._buckets = sorted(buckets)
        self._counts = [0] * len(self._buckets)
        self._inf_count = 0
        self._sum = 0.0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._inf_count += 1
            for i, edge in enumerate(self._buckets):
                if value <= edge:
                    self._counts[i] += 1

    def render(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.help}",
            f"# TYPE {self.name} histogram",
        ]
        with self._lock:
            edges = list(self._buckets)
            counts = list(self._counts)
            inf_count = self._inf_count
            total = self._sum
        for edge, count in zip(edges, counts):
            lines.append(
                f'{self.name}_bucket{{le="{edge:g}"}} {count}'
            )
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {inf_count}')
        lines.append(f"{self.name}_sum {total:g}")
        lines.append(f"{self.name}_count {inf_count}")
        return lines


# ---------------------------------------------------------------------------
# Named instances — import these directly at call sites.
# ---------------------------------------------------------------------------

intents_total = Counter(
    "marshal_intents_total",
    "Total intents executed by agentd, labelled by terminal status and category.",
)

# Buckets in milliseconds. Tuned for CPU llama.cpp where the warm-cache p50
# is ~800-1500ms and the cold-start tail can hit 30s.
inference_latency_ms = Histogram(
    "marshal_inference_latency_ms",
    "Latency of inference backend completion calls, in milliseconds.",
    buckets=[50, 100, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000],
)

enforcer_rejections_total = Counter(
    "marshal_enforcer_rejections_total",
    "Action-contract enforcer rejections, labelled by rejection reason.",
)

runner_path_total = Counter(
    "marshal_runner_path_total",
    "Intents executed by runner path: warm (pre-forked pool) vs cold (fresh subprocess).",
)


def render_prometheus_text() -> str:
    """
    Return the full /v1/metrics body in Prometheus 0.0.4 text format.

    Each metric block ends with a newline; the body terminates with a
    final newline so `curl ... | wc -l` matches the line count Prometheus
    expects.
    """
    blocks: list[list[str]] = [
        intents_total.render(),
        inference_latency_ms.render(),
        enforcer_rejections_total.render(),
        runner_path_total.render(),
    ]
    return "\n".join("\n".join(block) for block in blocks) + "\n"
