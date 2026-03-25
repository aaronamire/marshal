"""Abstract base class for Cortex data source adapters."""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class KnowledgeItem:
    """A single item to be indexed in the knowledge graph."""
    source_type: str     # "file", "email", "browser", "calendar", "git"
    source_id: str       # unique within source type (filepath, message-id, URL)
    source_path: str     # human-readable location
    title: str
    content: str         # plain text
    content_hash: str    # BLAKE2b hex — skip re-embedding if unchanged
    metadata: dict       # source-specific structured data
    timestamp: str       # ISO8601 — when the item was created/modified at source


class BaseAdapter(ABC):
    """
    Each adapter scans one data source and yields KnowledgeItems.

    Contract:
    - scan() is idempotent
    - scan() yields items lazily — never load entire source into memory
    - content_hash enables skip-if-unchanged in KnowledgeGraph.upsert_batch()
    - All timestamps are UTC ISO8601
    - All paths are absolute (~ expanded)
    """

    @abstractmethod
    def source_type(self) -> str:
        """Unique identifier for this source type."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the data source exists on this system."""

    @abstractmethod
    def scan(self, since: Optional[str] = None) -> Iterator[KnowledgeItem]:
        """
        Yield all items modified since `since` (ISO8601).
        If since is None, yield all items (full re-index).
        """

    @staticmethod
    def hash_content(content: str) -> str:
        """BLAKE2b hash of content for dedup."""
        return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()
