"""
Unit tests for the RAG store.
Requires: lancedb, sentence-transformers, rank-bm25.
Skipped automatically if any of these are missing.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip the entire module if RAG dependencies are missing
try:
    import lancedb  # noqa: F401
    import rank_bm25  # noqa: F401
    import sentence_transformers  # noqa: F401
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _RAG_AVAILABLE,
    reason="lancedb, sentence-transformers, or rank-bm25 not installed",
)


@pytest.fixture(scope="module")
def store():
    """Build a fresh RagStore from seed examples (once per test module)."""
    from rag.store import RagStore
    s = RagStore.load(rebuild=True)
    assert s is not None, "RagStore.load() returned None — check seed_examples.jsonl"
    return s


class TestSeedLoad:

    def test_store_loads(self, store):
        assert store is not None

    def test_corpus_nonempty(self, store):
        assert len(store._corpus) > 0

    def test_all_examples_have_intent(self, store):
        for ex in store._corpus:
            assert "intent" in ex and ex["intent"]


class TestRetrieval:

    def test_read_query_returns_read_example(self, store):
        results = store.retrieve("show me the contents of a file", top_k=3)
        assert len(results) > 0
        types = [a["type"] for ex in results for a in ex.get("actions", [])]
        assert "READ" in types, f"Expected READ in results, got types: {types}"

    def test_move_query_returns_move_example(self, store):
        results = store.retrieve("rename my notes file", top_k=3)
        assert len(results) > 0
        types = [a["type"] for ex in results for a in ex.get("actions", [])]
        assert "MOVE" in types, f"Expected MOVE in results, got: {types}"

    def test_delete_query_returns_delete_example(self, store):
        results = store.retrieve("remove temporary files", top_k=3)
        assert len(results) > 0
        types = [a["type"] for ex in results for a in ex.get("actions", [])]
        assert "DELETE" in types or "QUERY" in types

    def test_email_query_returns_not_implemented(self, store):
        results = store.retrieve("write an email to my manager", top_k=3)
        assert len(results) > 0
        categories = [ex.get("category") for ex in results]
        assert "email_task" in categories

    def test_top_k_respected(self, store):
        for k in (1, 2, 3):
            results = store.retrieve("find python files", top_k=k)
            assert len(results) <= k

    def test_returns_list(self, store):
        results = store.retrieve("list files in my downloads", top_k=3)
        assert isinstance(results, list)


class TestFormatExamples:

    def test_empty_examples_returns_empty_string(self, store):
        assert store.format_examples([]) == ""

    def test_format_contains_intent(self, store):
        results = store.retrieve("rename file", top_k=1)
        if results:
            snippet = store.format_examples(results)
            assert "Intent:" in snippet
            assert results[0]["intent"] in snippet

    def test_format_contains_action_type(self, store):
        results = store.retrieve("copy my config to backup", top_k=1)
        if results:
            snippet = store.format_examples(results)
            # Should mention the action type or NOT_IMPLEMENTED
            assert any(t in snippet for t in ["READ", "MOVE", "COPY", "DELETE",
                                               "QUERY", "WRITE", "NOT_IMPLEMENTED"])

    def test_format_has_header_and_footer(self, store):
        results = store.retrieve("read a file", top_k=1)
        if results:
            snippet = store.format_examples(results)
            assert "FEW-SHOT EXAMPLES" in snippet
            assert "END EXAMPLES" in snippet


class TestRrfFusion:

    def test_rrf_prefers_items_in_both_lists(self):
        from rag.store import _rrf_fuse
        # Item 5 appears first in both — should win
        dense = [(5, {"id": 5}), (1, {"id": 1}), (2, {"id": 2})]
        sparse = [(5, {"id": 5}), (3, {"id": 3}), (4, {"id": 4})]
        fused = _rrf_fuse(dense, sparse)
        assert fused[0]["id"] == 5

    def test_rrf_returns_all_unique_items(self):
        from rag.store import _rrf_fuse
        dense = [(0, {"x": 0}), (1, {"x": 1})]
        sparse = [(2, {"x": 2}), (1, {"x": 1})]
        fused = _rrf_fuse(dense, sparse)
        assert len(fused) == 3  # 0, 1, 2 unique items
