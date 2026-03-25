"""Tests for cortex knowledge graph, filesystem adapter, and indexer."""
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.audit import _create_schema
from cortex.adapters.base import BaseAdapter, KnowledgeItem
from cortex.adapters.filesystem import (
    FilesystemAdapter,
    INDEXABLE_EXTENSIONS,
    SKIP_DIRS,
    MAX_FILE_SIZE,
)
from cortex.knowledge_graph import KnowledgeGraph
from cortex.indexer import CortexIndexer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    _create_schema(conn)
    return conn


@pytest.fixture
def tmp_tree(tmp_path):
    """Create a small file tree for testing the filesystem adapter."""
    (tmp_path / "hello.txt").write_text("Hello world")
    (tmp_path / "notes.md").write_text("# Notes\nSome notes here")
    (tmp_path / "script.py").write_text("print('hi')")
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0")  # not indexable
    (tmp_path / "empty.txt").write_text("")  # empty — should be skipped

    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "data.json").write_text('{"key": "value"}')

    # Skip dirs
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text("gitconfig")

    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "mod.cpython.pyc").write_text("bytecode")

    return tmp_path


# ---------------------------------------------------------------------------
# FilesystemAdapter
# ---------------------------------------------------------------------------

class TestFilesystemAdapter:

    def test_scan_finds_text_files(self, tmp_tree):
        adapter = FilesystemAdapter(root=tmp_tree)
        items = list(adapter.scan())
        titles = {item.title for item in items}

        assert "hello.txt" in titles
        assert "notes.md" in titles
        assert "script.py" in titles
        assert "data.json" in titles

    def test_scan_skips_non_indexable(self, tmp_tree):
        adapter = FilesystemAdapter(root=tmp_tree)
        items = list(adapter.scan())
        titles = {item.title for item in items}
        assert "photo.jpg" not in titles

    def test_scan_skips_empty_files(self, tmp_tree):
        adapter = FilesystemAdapter(root=tmp_tree)
        items = list(adapter.scan())
        titles = {item.title for item in items}
        assert "empty.txt" not in titles

    def test_scan_skips_git_and_pycache(self, tmp_tree):
        adapter = FilesystemAdapter(root=tmp_tree)
        items = list(adapter.scan())
        paths = {item.source_path for item in items}
        assert not any(".git" in p for p in paths)
        assert not any("__pycache__" in p for p in paths)

    def test_scan_since_filters_by_mtime(self, tmp_tree):
        adapter = FilesystemAdapter(root=tmp_tree)
        # Use a future timestamp — should return nothing
        items = list(adapter.scan(since="2099-01-01T00:00:00+00:00"))
        assert len(items) == 0

    def test_content_hash_deterministic(self, tmp_tree):
        adapter = FilesystemAdapter(root=tmp_tree)
        items1 = {i.title: i.content_hash for i in adapter.scan()}
        items2 = {i.title: i.content_hash for i in adapter.scan()}
        for title in items1:
            assert items1[title] == items2[title]

    def test_source_type(self, tmp_tree):
        adapter = FilesystemAdapter(root=tmp_tree)
        assert adapter.source_type() == "file"

    def test_is_available(self, tmp_tree):
        adapter = FilesystemAdapter(root=tmp_tree)
        assert adapter.is_available()

    def test_unavailable_root(self, tmp_path):
        adapter = FilesystemAdapter(root=tmp_path / "nonexistent")
        assert not adapter.is_available()

    def test_large_file_skipped(self, tmp_tree):
        big = tmp_tree / "big.txt"
        big.write_text("x" * (MAX_FILE_SIZE + 1))
        adapter = FilesystemAdapter(root=tmp_tree)
        items = list(adapter.scan())
        titles = {i.title for i in items}
        assert "big.txt" not in titles

    def test_item_fields(self, tmp_tree):
        adapter = FilesystemAdapter(root=tmp_tree)
        items = list(adapter.scan())
        txt = next(i for i in items if i.title == "hello.txt")
        assert txt.source_type == "file"
        assert txt.content == "Hello world"
        assert "size" in txt.metadata
        assert "mtime" in txt.metadata
        assert "extension" in txt.metadata
        assert txt.metadata["extension"] == ".txt"


# ---------------------------------------------------------------------------
# KnowledgeGraph (mocked embedding model)
# ---------------------------------------------------------------------------

class TestKnowledgeGraph:

    @pytest.fixture
    def mock_model(self):
        import numpy as np
        model = MagicMock()
        model.encode = MagicMock(
            side_effect=lambda texts, **kw: np.random.rand(len(texts), 384).astype("float32")
        )
        return model

    @pytest.fixture
    def kg(self, db, tmp_path, mock_model):
        kg = KnowledgeGraph(db, tmp_path / "lance")
        with patch.object(kg, "_ensure_model", return_value=mock_model):
            yield kg

    def _make_item(self, idx: int) -> dict:
        content = f"content for item {idx}"
        return {
            "id": f"item-{idx}",
            "source_type": "file",
            "source_id": f"/tmp/file{idx}.txt",
            "source_path": f"/tmp/file{idx}.txt",
            "title": f"file{idx}.txt",
            "content": content,
            "content_hash": BaseAdapter.hash_content(content),
            "metadata": {"size": 100},
            "timestamp": "2026-03-25T00:00:00+00:00",
        }

    def test_upsert_batch_inserts(self, kg, mock_model):
        with patch.object(kg, "_ensure_model", return_value=mock_model):
            items = [self._make_item(i) for i in range(3)]
            count = kg.upsert_batch(items)
            assert count == 3

    def test_upsert_batch_skips_unchanged(self, kg, mock_model):
        with patch.object(kg, "_ensure_model", return_value=mock_model):
            items = [self._make_item(0)]
            kg.upsert_batch(items)
            # Second time — same content_hash
            count = kg.upsert_batch(items)
            assert count == 0

    def test_upsert_batch_empty(self, kg):
        assert kg.upsert_batch([]) == 0

    def test_make_id_deterministic(self):
        id1 = KnowledgeGraph.make_id("file", "/tmp/a.txt", "abc123")
        id2 = KnowledgeGraph.make_id("file", "/tmp/a.txt", "abc123")
        assert id1 == id2

    def test_make_id_different_for_different_content(self):
        id1 = KnowledgeGraph.make_id("file", "/tmp/a.txt", "abc123")
        id2 = KnowledgeGraph.make_id("file", "/tmp/a.txt", "def456")
        assert id1 != id2

    def test_stats_empty(self, kg):
        assert kg.stats() == {}

    def test_stats_after_insert(self, kg, mock_model):
        with patch.object(kg, "_ensure_model", return_value=mock_model):
            items = [self._make_item(0)]
            kg.upsert_batch(items)
            s = kg.stats()
            assert "file" in s
            assert s["file"]["count"] == 1


# ---------------------------------------------------------------------------
# CortexIndexer
# ---------------------------------------------------------------------------

class TestCortexIndexer:

    @pytest.fixture
    def indexer(self, db, tmp_path):
        lance_path = tmp_path / "lance"
        indexer = CortexIndexer(db, lance_path)
        return indexer

    def test_register_available_adapter(self, indexer, tmp_tree):
        adapter = FilesystemAdapter(root=tmp_tree)
        indexer.register(adapter)
        assert len(indexer._adapters) == 1

    def test_register_unavailable_adapter(self, indexer, tmp_path):
        adapter = FilesystemAdapter(root=tmp_path / "nonexistent")
        indexer.register(adapter)
        assert len(indexer._adapters) == 0

    def test_run_no_adapters(self, indexer):
        stats = indexer.run_once()
        assert stats["total_scanned"] == 0
        assert stats["total_upserted"] == 0

    @patch("cortex.indexer.CortexIndexer._llama_busy", return_value=False)
    def test_run_once_with_mock_kg(self, _busy, indexer, tmp_tree):
        """Run indexer with mocked embedding to verify adapter→KG pipeline."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            side_effect=lambda texts, **kw: np.random.rand(len(texts), 384).astype("float32")
        )

        adapter = FilesystemAdapter(root=tmp_tree)
        indexer.register(adapter)

        with patch.object(indexer._kg, "_ensure_model", return_value=mock_model):
            stats = indexer.run_once()

        assert stats["total_scanned"] >= 4  # hello.txt, notes.md, script.py, data.json
        assert stats["total_upserted"] >= 4
        assert "file" in stats["adapters"]
        assert stats["elapsed_seconds"] >= 0

    @patch("cortex.indexer.CortexIndexer._llama_busy", return_value=False)
    def test_incremental_after_full(self, _busy, indexer, tmp_tree):
        """Incremental run after full should upsert 0 (nothing changed)."""
        import numpy as np

        mock_model = MagicMock()
        mock_model.encode = MagicMock(
            side_effect=lambda texts, **kw: np.random.rand(len(texts), 384).astype("float32")
        )

        adapter = FilesystemAdapter(root=tmp_tree)
        indexer.register(adapter)

        with patch.object(indexer._kg, "_ensure_model", return_value=mock_model):
            indexer.run_once()
            stats = indexer.run_incremental()

        # All items already indexed with same content_hash → 0 upserted
        assert stats["total_upserted"] == 0

    def test_llama_busy_server_down(self):
        """When llama-server is unreachable, _llama_busy returns False."""
        assert CortexIndexer._llama_busy() is False

    def test_status_initial(self, indexer):
        s = indexer.status()
        assert s["last_run"] is None

    def test_item_to_dict(self):
        item = KnowledgeItem(
            source_type="file",
            source_id="/tmp/test.txt",
            source_path="/tmp/test.txt",
            title="test.txt",
            content="hello",
            content_hash="abc123",
            metadata={"size": 5},
            timestamp="2026-03-25T00:00:00+00:00",
        )
        d = CortexIndexer._item_to_dict(item)
        assert d["source_type"] == "file"
        assert d["title"] == "test.txt"
        assert d["content"] == "hello"
        assert "id" in d  # generated by make_id
