"""
Location-transparent inference client for Leaves OS.

Uses an InferenceBackend ABC so that callers never know which backend
(local llama.cpp, remote API) is serving the request.

Phase 0-2: LocalLlamaCppBackend via native /completion endpoint.
           Prompt caching (cache_prompt=true) reuses the KV cache for the
           common system-prompt prefix across requests, cutting prefill from
           ~30s to ~2-4s on the i5-7200U.
Phase 3:   add RemoteAPIBackend as fallback.
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
    tokens_cached: int = 0     # prompt tokens reused from KV cache
    prompt_tokens: int = 0     # prompt tokens actually computed (after cache hit)
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
    Talks to a locally running llama.cpp server via its native HTTP API.
    Server endpoint: POST /completion  (GBNF grammar + cache_prompt support)
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

        # Native /completion endpoint — supports cache_prompt for KV cache
        # reuse across requests (the system prompt prefix is identical every
        # time, so only the RAG examples + user intent need fresh prefill).
        # Also supports grammar for GBNF-constrained decoding.
        payload: dict = {
            "prompt": request.prompt,
            "n_predict": request.max_tokens,
            "temperature": request.temperature,
            "stop": request.stop_tokens,
            "stream": request.stream,
            "cache_prompt": True,
        }
        if request.grammar is not None:
            payload["grammar"] = request.grammar

        t0 = time.monotonic()
        try:
            r = self._session.post(
                f"{self._base_url}/completion",
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

        # Native response: {"content": "...", "tokens_cached": N, "timings": {...}}
        content = data.get("content")
        if content is None:
            raise LeavesError(
                LeavesErrorCode.INFERENCE_BAD_RESPONSE,
                detail=f"Unexpected response structure: {str(data)[:200]}",
            )
        if not content:
            raise LeavesError(
                LeavesErrorCode.INFERENCE_BAD_RESPONSE,
                detail="Server returned empty content.",
            )

        timings = data.get("timings", {})
        tokens_cached = data.get("tokens_cached", 0)
        tokens_predicted = timings.get("predicted_n", 0)
        prompt_tokens = timings.get("prompt_n", 0)
        model_name = data.get("model", "llama.cpp")

        return InferenceResponse(
            content=content,
            latency_ms=latency_ms,
            tokens_predicted=tokens_predicted,
            tokens_cached=tokens_cached,
            prompt_tokens=prompt_tokens,
            model=model_name,
        )


class InferenceClient:
    """
    Thin wrapper that selects the active backend.
    Phase 0-2: LocalLlamaCppBackend with prompt caching.
    Phase 3:   add RemoteAPIBackend as fallback.
    """

    def __init__(self, backend: Optional[InferenceBackend] = None):
        self._backend = backend or LocalLlamaCppBackend()

    def is_available(self) -> bool:
        return self._backend.is_available()

    def complete(self, request: InferenceRequest) -> InferenceResponse:
        return self._backend.complete(request)
