"""
Location-transparent inference client for Leaves OS.

Uses an InferenceBackend ABC so that Phase 1 can swap in a remote API
backend (Anthropic, OpenAI) without touching any calling code.

Phase 0: LocalLlamaCppBackend only.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import requests

from config import (
    INFERENCE_SERVER_URL,
    INFERENCE_TEMPERATURE,
    INFERENCE_MAX_TOKENS,
    INFERENCE_STOP_TOKENS,
    TIMEOUT_CONNECT_SECONDS,
    TIMEOUT_READ_SECONDS,
    TIMEOUT_HARD_SECONDS,
)
from errors import LeavesError, LeavesErrorCode


@dataclass
class InferenceRequest:
    prompt: str
    temperature: float = INFERENCE_TEMPERATURE
    max_tokens: int = INFERENCE_MAX_TOKENS
    stop_tokens: list[str] = field(default_factory=lambda: list(INFERENCE_STOP_TOKENS))
    grammar: Optional[str] = None  # GBNF grammar string; None = unconstrained
    stream: bool = False


@dataclass
class InferenceResponse:
    content: str
    latency_ms: float
    tokens_predicted: int = 0
    model: str = "unknown"


class InferenceBackend(ABC):
    """Abstract base — swap implementations without touching callers."""

    @abstractmethod
    def complete(self, request: InferenceRequest) -> InferenceResponse:
        """Send a completion request and return the response."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend is reachable right now."""
        ...


class LocalLlamaCppBackend(InferenceBackend):
    """
    Talks to a locally running llama.cpp server via its OpenAI-compatible HTTP API.
    Server endpoint: POST /v1/completions  (grammar field only works here, not /completion)
    Health check:    GET  /health
    """

    def __init__(self, base_url: str = INFERENCE_SERVER_URL):
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def is_available(self) -> bool:
        try:
            r = self._session.get(
                f"{self._base_url}/health",
                timeout=TIMEOUT_CONNECT_SECONDS,
            )
            return r.status_code == 200
        except Exception:
            return False

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        if not self.is_available():
            raise LeavesError(
                LeavesErrorCode.INFERENCE_UNAVAILABLE,
                detail=f"llama.cpp server not reachable at {self._base_url}",
            )

        # /v1/completions uses OpenAI field names (max_tokens, not n_predict)
        # and is the only endpoint that honours the per-request "grammar" field.
        payload = {
            "model": "local",
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stop": request.stop_tokens,
            "stream": request.stream,
        }
        if request.grammar is not None:
            payload["grammar"] = request.grammar

        t0 = time.monotonic()
        try:
            r = self._session.post(
                f"{self._base_url}/v1/completions",
                json=payload,
                timeout=(TIMEOUT_CONNECT_SECONDS, TIMEOUT_READ_SECONDS),
            )
        except requests.exceptions.Timeout:
            raise LeavesError(
                LeavesErrorCode.INFERENCE_TIMEOUT,
                detail=f"llama.cpp server timed out after {TIMEOUT_READ_SECONDS}s",
            )
        except requests.exceptions.ConnectionError as e:
            raise LeavesError(
                LeavesErrorCode.INFERENCE_UNAVAILABLE,
                detail=str(e),
                cause=e,
            )
        latency_ms = (time.monotonic() - t0) * 1000

        if r.status_code != 200:
            raise LeavesError(
                LeavesErrorCode.INFERENCE_BAD_RESPONSE,
                detail=f"HTTP {r.status_code}: {r.text[:200]}",
            )

        try:
            data = r.json()
        except Exception as e:
            raise LeavesError(
                LeavesErrorCode.INFERENCE_BAD_RESPONSE,
                detail=f"Could not decode JSON from server: {e}",
                cause=e,
            )

        # OpenAI response format: {"choices": [{"text": "...", ...}], "usage": {...}}
        try:
            content = data["choices"][0]["text"]
        except (KeyError, IndexError):
            raise LeavesError(
                LeavesErrorCode.INFERENCE_BAD_RESPONSE,
                detail=f"Unexpected response structure: {str(data)[:200]}",
            )
        if not content:
            raise LeavesError(
                LeavesErrorCode.INFERENCE_BAD_RESPONSE,
                detail="Server returned empty text in choices[0].",
            )

        model_name = data.get("model", "llama.cpp")
        tokens_predicted = data.get("usage", {}).get("completion_tokens", 0)

        return InferenceResponse(
            content=content,
            latency_ms=latency_ms,
            tokens_predicted=tokens_predicted,
            model=model_name,
        )


class InferenceClient:
    """
    Thin wrapper that selects the active backend.
    Phase 0: always uses LocalLlamaCppBackend.
    Phase 1: add RemoteAnthropicBackend as fallback.
    """

    def __init__(self, backend: Optional[InferenceBackend] = None):
        self._backend = backend or LocalLlamaCppBackend()

    def is_available(self) -> bool:
        return self._backend.is_available()

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        return self._backend.complete(request)
