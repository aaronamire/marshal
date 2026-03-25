"""
CortexIndexer — orchestrates knowledge graph population from adapters.

Registers available adapters, runs full or incremental scans,
and feeds items into KnowledgeGraph in batches.

Safety: checks llama-server health before embedding batches to avoid
competing for RAM on constrained hardware.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from config import INFERENCE_SERVER_URL
from cortex.adapters.base import BaseAdapter, KnowledgeItem
from cortex.knowledge_graph import KnowledgeGraph

log = logging.getLogger("cortex.indexer")

_BATCH_SIZE = 64
_LLAMA_HEALTH_URL = f"{INFERENCE_SERVER_URL}/health"
_LLAMA_HEALTH_TIMEOUT = 2  # seconds


class CortexIndexer:
    """
    Scans registered adapters and upserts items into the KnowledgeGraph.

    Usage:
        indexer = CortexIndexer(db, lance_path)
        indexer.register(FilesystemAdapter())
        stats = indexer.run_once()       # full re-index
        stats = indexer.run_incremental() # only changes since last run
    """

    def __init__(self, db: sqlite3.Connection, lance_path: Path):
        self._kg = KnowledgeGraph(db, lance_path)
        self._adapters: list[BaseAdapter] = []
        self._last_run: Optional[str] = None  # ISO8601

    def register(self, adapter: BaseAdapter) -> None:
        if adapter.is_available():
            self._adapters.append(adapter)
            log.info("Registered adapter: %s", adapter.source_type())
        else:
            log.warning("Adapter unavailable, skipping: %s", adapter.source_type())

    def run_once(self) -> dict:
        """Full re-index from all adapters. Returns stats."""
        return self._run(since=None)

    def run_incremental(self, since: Optional[str] = None) -> dict:
        """Index only items modified since last run (or given timestamp)."""
        cutoff = since or self._last_run
        return self._run(since=cutoff)

    def _run(self, since: Optional[str] = None) -> dict:
        if not self._adapters:
            log.warning("No adapters registered")
            return {"total_scanned": 0, "total_upserted": 0, "adapters": {}}

        start = time.monotonic()
        stats = {"total_scanned": 0, "total_upserted": 0, "adapters": {}}

        for adapter in self._adapters:
            src = adapter.source_type()
            log.info("Scanning %s (since=%s)", src, since)

            scanned = 0
            upserted = 0
            batch: list[dict] = []

            try:
                for item in adapter.scan(since=since):
                    scanned += 1
                    batch.append(self._item_to_dict(item))

                    if len(batch) >= _BATCH_SIZE:
                        if self._llama_busy():
                            log.info("llama-server busy, pausing embed batch")
                            self._wait_for_llama_idle()
                        upserted += self._kg.upsert_batch(batch)
                        batch = []

                # Flush remaining
                if batch:
                    if self._llama_busy():
                        self._wait_for_llama_idle()
                    upserted += self._kg.upsert_batch(batch)

            except Exception:
                log.exception("Error scanning adapter %s", src)

            stats["adapters"][src] = {"scanned": scanned, "upserted": upserted}
            stats["total_scanned"] += scanned
            stats["total_upserted"] += upserted
            log.info("Adapter %s: scanned=%d upserted=%d", src, scanned, upserted)

        elapsed = time.monotonic() - start
        stats["elapsed_seconds"] = round(elapsed, 2)
        self._last_run = datetime.now(timezone.utc).isoformat()
        log.info("Indexing complete: %d scanned, %d upserted in %.1fs",
                 stats["total_scanned"], stats["total_upserted"], elapsed)
        return stats

    def search(self, query: str, **kwargs) -> list[dict]:
        """Proxy to KnowledgeGraph.semantic_search."""
        return self._kg.semantic_search(query, **kwargs)

    def timeline(self, hours: int = 24, **kwargs) -> list[dict]:
        """Proxy to KnowledgeGraph.temporal_query."""
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        return self._kg.temporal_query(since=cutoff, **kwargs)

    def status(self) -> dict:
        """Return KnowledgeGraph stats + last run time."""
        s = self._kg.stats()
        s["last_run"] = self._last_run
        return s

    @staticmethod
    def _item_to_dict(item: KnowledgeItem) -> dict:
        item_id = KnowledgeGraph.make_id(
            item.source_type, item.source_id, item.content_hash
        )
        return {
            "id": item_id,
            "source_type": item.source_type,
            "source_id": item.source_id,
            "source_path": item.source_path,
            "title": item.title,
            "content": item.content,
            "content_hash": item.content_hash,
            "metadata": item.metadata,
            "timestamp": item.timestamp,
        }

    @staticmethod
    def _llama_busy() -> bool:
        """Check if llama-server is actively processing (avoid RAM contention)."""
        try:
            r = requests.get(_LLAMA_HEALTH_URL, timeout=_LLAMA_HEALTH_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                # llama.cpp /health returns {"status": "ok"} when idle,
                # {"status": "loading model"} or slots_idle < slots_total when busy
                if data.get("status") == "ok":
                    return False
                return True
            # Server up but non-200 → probably busy
            return True
        except (requests.ConnectionError, requests.Timeout):
            # Server not running → not competing for RAM
            return False

    @staticmethod
    def _wait_for_llama_idle(max_wait: int = 60, poll_interval: int = 5) -> None:
        """Block until llama-server is idle or max_wait exceeded."""
        waited = 0
        while waited < max_wait:
            time.sleep(poll_interval)
            waited += poll_interval
            if not CortexIndexer._llama_busy():
                return
        log.warning("llama-server still busy after %ds, proceeding anyway", max_wait)
