"""
Layer 1 intent classifier.
Wraps the trained sklearn pipeline for use in the two-stage intent pipeline.

CONTRACT:
- Always resident in memory after startup (loaded once)
- Never raises — returns confidence=0.0 / category='unknown' on any error
- Inference: 1-8ms on i5-7200U
- No torch, no GPU, no CUDA deps
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib

from config import MIN_CONFIDENCE_THRESHOLD

PIPELINE_PATH = Path(__file__).parent.parent / "models" / "layer1_pipeline.joblib"


@dataclass
class ClassificationResult:
    category: str
    confidence: float
    latency_ms: float
    all_scores: dict[str, float] = field(default_factory=dict)
    is_confident: bool = False


_UNKNOWN = ClassificationResult(
    category="unknown", confidence=0.0, latency_ms=0.0,
    all_scores={}, is_confident=False
)


class IntentClassifier:
    """
    Layer 1 classifier. Instantiate once at startup, reuse across all intents.
    Thread-safe for reads (joblib pipeline is read-only after load).
    """

    def __init__(self, pipeline_path: Path = PIPELINE_PATH):
        self._pipeline = None
        self._pipeline_path = pipeline_path
        self._load()

    def _load(self) -> None:
        if not self._pipeline_path.exists():
            raise FileNotFoundError(
                f"Layer 1 classifier not found at {self._pipeline_path}.\n"
                f"Train it: python3 scripts/train_classifier.py"
            )
        self._pipeline = joblib.load(self._pipeline_path)

    @property
    def is_available(self) -> bool:
        return self._pipeline is not None

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify intent text into a category.
        Never raises — returns confidence=0.0 on any error.
        """
        if not text or not text.strip():
            return _UNKNOWN

        t0 = time.monotonic()
        try:
            proba = self._pipeline.predict_proba([text])[0]
            classes = self._pipeline.classes_
            scores = dict(zip(classes, proba.tolist()))
            best = max(scores, key=scores.get)
            conf = scores[best]
        except Exception:
            return _UNKNOWN

        latency_ms = (time.monotonic() - t0) * 1000
        return ClassificationResult(
            category=best,
            confidence=round(conf, 4),
            latency_ms=round(latency_ms, 2),
            all_scores={k: round(v, 4) for k, v in scores.items()},
            is_confident=conf >= MIN_CONFIDENCE_THRESHOLD,
        )
