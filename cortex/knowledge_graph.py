"""
Knowledge graph — unified index of user's files for semantic search.

Dual storage:
  - LanceDB: ANN vector search over embeddings (semantic queries)
  - SQLite: structured queries (date ranges, source types, path prefixes)

Reuses the all-MiniLM-L6-v2 model from rag/store.py (384-dim, ~80MB RAM).
Same LanceDB instance (rag/.lancedb/), new table "knowledge_items".
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("cortex.knowledge_graph")

_LANCE_TABLE = "knowledge_items"
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_DIM = 384
_BATCH_SIZE = 64
_MAX_CONTENT_LEN = 4096  # chars — truncate before embedding


class KnowledgeGraph:
    """
    Manages the unified knowledge store across data sources.

    Thread safety: writes go through a lock. Reads are concurrent.
    """

    def __init__(self, db: sqlite3.Connection, lance_path: Path):
        # Extract the on-disk path so we can open thread-local connections on
        # demand. Indexing runs in a threadpool executor (see agentd.py
        # _background_indexing → loop.run_in_executor) but sqlite3 connections
        # are thread-bound by default. Previously this stored a single
        # connection and raised ProgrammingError ("SQLite objects created in a
        # thread can only be used in that same thread") on every upsert,
        # silently losing all indexed rows.
        rows = db.execute("PRAGMA database_list").fetchall()
        path_str = ""
        for r in rows:
            try:
                name = r["name"]
            except Exception:
                name = r[1]
            if name == "main":
                try:
                    path_str = r["file"]
                except Exception:
                    path_str = r[2]
                break
        # Empty path means :memory: DB (or an attached-db quirk). In-memory
        # connections can't be reopened, so fall back to sharing the passed
        # connection across threads. check_same_thread=False plus the
        # existing _write_lock keeps this safe for the test harness.
        if path_str:
            self._db_path = Path(path_str)
            self._shared_db = None
        else:
            self._db_path = None
            self._shared_db = db
        self._tls = threading.local()
        self._lance_path = lance_path
        self._model = None  # lazy
        self._table = None  # lazy
        self._write_lock = threading.Lock()

    def _db_conn(self) -> sqlite3.Connection:
        """Return a sqlite3 connection bound to the current thread."""
        if self._shared_db is not None:
            return self._shared_db
        conn = getattr(self._tls, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        # Mirror db/audit.py: indexer writes contend with agentd + api writers.
        conn.execute("PRAGMA busy_timeout=10000")
        self._tls.conn = conn
        return conn

    def _ensure_model(self):
        if self._model is None:
            from rag.store import RagStore
            self._model = RagStore._get_or_load_model()
        return self._model

    def _ensure_table(self):
        if self._table is None:
            import lancedb
            db = lancedb.connect(str(self._lance_path))
            # lancedb's list_tables() returns a ListTablesResponse wrapper
            # whose names live on .tables; bare membership tests against the
            # wrapper silently return False. Unwrap before checking.
            table_names = db.list_tables() if hasattr(db, 'list_tables') else list(db)
            if hasattr(table_names, "tables"):
                table_names = table_names.tables
            if _LANCE_TABLE in table_names:
                self._table = db.open_table(_LANCE_TABLE)
            else:
                # Create with a placeholder row, then delete it
                self._table = db.create_table(_LANCE_TABLE, data=[{
                    "id": "__placeholder__",
                    "source_type": "",
                    "source_id": "",
                    "title": "",
                    "content": "",
                    "content_hash": "",
                    "timestamp": "",
                    "indexed_at": "",
                    "vector": [0.0] * _EMBED_DIM,
                }])
                self._table.delete('id = "__placeholder__"')
        return self._table

    def upsert_batch(self, items: list[dict[str, Any]]) -> int:
        """
        Upsert a batch of knowledge items.
        Skips items where content_hash matches existing entry.
        Returns count of actually embedded/updated items.
        """
        if not items:
            return 0

        model = self._ensure_model()
        table = self._ensure_table()

        # Check which items are new or changed
        to_embed = []
        for item in items:
            row = self._db_conn().execute(
                "SELECT content_hash FROM knowledge_items WHERE id = ?",
                (item["id"],),
            ).fetchone()
            if row and row[0] == item["content_hash"]:
                continue  # unchanged
            to_embed.append(item)

        if not to_embed:
            return 0

        # Batch embed
        texts = [
            f"{item.get('title', '')} {item.get('content', '')[:_MAX_CONTENT_LEN]}"
            for item in to_embed
        ]
        embeddings = model.encode(texts, batch_size=_BATCH_SIZE, show_progress_bar=False)

        with self._write_lock:
            lance_rows = []
            now = datetime.now(timezone.utc).isoformat()

            for item, emb in zip(to_embed, embeddings):
                lance_rows.append({
                    "id": item["id"],
                    "source_type": item.get("source_type", ""),
                    "source_id": item.get("source_id", ""),
                    "title": item.get("title", ""),
                    "content": item.get("content", "")[:_MAX_CONTENT_LEN],
                    "content_hash": item["content_hash"],
                    "timestamp": item.get("timestamp", now),
                    "indexed_at": now,
                    "vector": emb.tolist(),
                })

                # Upsert SQLite structured index
                self._db_conn().execute("""
                    INSERT OR REPLACE INTO knowledge_items
                    (id, source_type, source_id, source_path, title,
                     content_preview, content_hash, metadata_json,
                     timestamp, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item["id"],
                    item.get("source_type", ""),
                    item.get("source_id", ""),
                    item.get("source_path", ""),
                    item.get("title", ""),
                    item.get("content", "")[:500],
                    item["content_hash"],
                    json.dumps(item.get("metadata", {})),
                    item.get("timestamp", now),
                    now,
                ))

            # Commit sqlite BEFORE LanceDB writes. The LanceDB delete/add
            # below can take seconds per batch (disk I/O) — if we hold the
            # sqlite writer transaction across it, every other writer
            # (agentd + api /v1/intent/execute) blocks past busy_timeout
            # and crashes with "database is locked".
            self._db_conn().commit()

            # LanceDB: delete existing then add
            ids_to_replace = [r["id"] for r in lance_rows]
            for rid in ids_to_replace:
                try:
                    table.delete(f'id = "{rid}"')
                except Exception:
                    pass
            table.add(lance_rows)

        log.info("Upserted %d items (%d skipped)", len(to_embed), len(items) - len(to_embed))
        return len(to_embed)

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        source_types: Optional[list[str]] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Hybrid semantic + structured search."""
        model = self._ensure_model()
        table = self._ensure_table()

        query_emb = model.encode([query], show_progress_bar=False)[0].tolist()

        # LanceDB ANN search — over-retrieve for filtering
        results = table.search(query_emb).limit(top_k * 3).to_list()

        # Apply structured filters
        filtered = []
        for r in results:
            if source_types and r.get("source_type") not in source_types:
                continue
            if after and r.get("timestamp", "") < after:
                continue
            if before and r.get("timestamp", "") > before:
                continue
            filtered.append({
                "id": r.get("id"),
                "source_type": r.get("source_type"),
                "title": r.get("title"),
                "content": r.get("content", ""),
                "timestamp": r.get("timestamp"),
                "_distance": r.get("_distance", 0),
            })

        return filtered[:top_k]

    def temporal_query(
        self,
        since: str,
        until: Optional[str] = None,
        source_types: Optional[list[str]] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """What changed in a time range? Returns items ordered by timestamp desc."""
        sql = "SELECT * FROM knowledge_items WHERE timestamp >= ?"
        params: list = [since]

        if until:
            sql += " AND timestamp <= ?"
            params.append(until)
        if source_types:
            placeholders = ",".join("?" * len(source_types))
            sql += f" AND source_type IN ({placeholders})"
            params.extend(source_types)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._db_conn().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        """Return indexing statistics per source type."""
        rows = self._db_conn().execute(
            "SELECT source_type, COUNT(*) as count, MAX(indexed_at) as last_indexed "
            "FROM knowledge_items GROUP BY source_type"
        ).fetchall()
        return {
            row["source_type"]: {"count": row["count"], "last_indexed": row["last_indexed"]}
            for row in rows
        }

    @staticmethod
    def make_id(source_type: str, source_id: str, content_hash: str) -> str:
        """Deterministic ID for dedup."""
        raw = f"{source_type}:{source_id}:{content_hash}"
        return hashlib.blake2b(raw.encode(), digest_size=16).hexdigest()
