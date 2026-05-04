"""
RAG (Retrieval-Augmented Generation) pipeline for Marshal.

Architecture:
  Dense:  all-MiniLM-L6-v2 embeddings → LanceDB ANN search
  Sparse: BM25 (rank-bm25)
  Fusion: Reciprocal Rank Fusion (k=60)

The retrieved examples are injected into the intent_parser system prompt
as compact few-shot demonstrations, improving action type selection and
output format compliance for the local 3B model.

Designed for robustness: falls through gracefully if LanceDB, the embedding
model, or seed_examples.jsonl are unavailable. The L2 LLM still works without
RAG — RAG is a quality enhancer, not a hard dependency.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SEED_PATH = Path(__file__).parent / "seed_examples.jsonl"
_DB_PATH = Path(__file__).parent / ".lancedb"
_TABLE_NAME = "intent_examples"
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TOP_K = 3      # number of examples to retrieve per query
_RRF_K = 60     # RRF constant (standard: 60)


def _rrf_fuse(dense_ranked: list, sparse_ranked: list, k: int = _RRF_K) -> list:
    """
    Reciprocal Rank Fusion over two ranked lists of (doc_id, example) pairs.
    Returns examples ordered by descending fused score.
    """
    scores: dict[int, float] = {}
    for rank, (doc_id, _) in enumerate(dense_ranked, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, (doc_id, _) in enumerate(sparse_ranked, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Collect examples in fused order
    all_examples: dict[int, dict] = {}
    for doc_id, ex in dense_ranked + sparse_ranked:
        all_examples[doc_id] = ex

    return [
        all_examples[doc_id]
        for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])
    ]


class RagStore:
    """
    Hybrid dense+sparse retriever backed by LanceDB and BM25.

    Usage:
        store = RagStore.load()       # returns None if unavailable
        if store:
            examples = store.retrieve("find all PDFs in Downloads", top_k=3)
            snippet = store.format_examples(examples)
            # inject snippet into system prompt
    """

    def __init__(self, table, bm25_index, corpus: list[dict]):
        self._table = table
        self._bm25 = bm25_index
        self._corpus = corpus  # list of example dicts in insertion order

    @classmethod
    def load(cls, rebuild: bool = False) -> Optional["RagStore"]:
        """
        Load or build the RAG store from seed_examples.jsonl.
        Returns None on any failure — never raises.
        """
        try:
            return cls._load_impl(rebuild=rebuild)
        except Exception as e:
            logger.warning("RAG store unavailable: %s", e)
            return None

    @classmethod
    def _load_impl(cls, rebuild: bool) -> "RagStore":
        import lancedb
        from rank_bm25 import BM25Okapi

        examples = cls._read_seed()
        texts = [ex["intent"] for ex in examples]
        tokenized = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenized)

        # Build or open LanceDB table.
        # list_tables() returns a ListTablesResponse (lancedb >=0.x); the
        # actual list of names is on .tables. Membership tests on the bare
        # response object always evaluate False — that's the trap that
        # silently caused "table already exists" failures when the legacy
        # table_names() call was naively swapped for list_tables().
        db = lancedb.connect(str(_DB_PATH))
        existing_tables = db.list_tables()
        if hasattr(existing_tables, "tables"):
            existing_tables = existing_tables.tables
        if rebuild or _TABLE_NAME not in existing_tables:
            # Load model once — reuse for both building index and later retrieval
            model = cls._get_or_load_model()
            embeddings = model.encode(texts, show_progress_bar=False).tolist()
            data = [
                {
                    "id": i,
                    "intent": ex["intent"],
                    "vector": emb,
                    "category": ex["category"],
                    "actions_json": json.dumps(ex.get("actions", [])),
                }
                for i, (ex, emb) in enumerate(zip(examples, embeddings))
            ]
            if _TABLE_NAME in existing_tables:
                db.drop_table(_TABLE_NAME)
            table = db.create_table(_TABLE_NAME, data=data)
        else:
            table = db.open_table(_TABLE_NAME)
            # Ensure model is cached for retrieval — no-op if already loaded
            cls._get_or_load_model()

        return cls(table, bm25, examples)

    @classmethod
    def _get_or_load_model(cls) -> object:
        """
        Load the embedding model exactly once per process.
        Caches at class level — subsequent calls return the cached instance.
        This is the single source of truth for model instantiation.
        """
        from sentence_transformers import SentenceTransformer
        if not hasattr(cls, "_embed_model") or cls._embed_model is None:
            logger.debug("Loading embedding model: %s", _EMBED_MODEL)
            cls._embed_model = SentenceTransformer(_EMBED_MODEL)
        return cls._embed_model

    @staticmethod
    def _read_seed() -> list[dict]:
        if not _SEED_PATH.exists():
            raise FileNotFoundError(f"Seed examples not found: {_SEED_PATH}")
        examples = []
        with _SEED_PATH.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        if not examples:
            raise ValueError("seed_examples.jsonl is empty")
        return examples

    def retrieve(self, query: str, top_k: int = _TOP_K) -> list[dict]:
        """
        Retrieve the top_k most relevant examples for a query.
        Uses RRF fusion of dense (LanceDB ANN) and sparse (BM25) results.
        """
        n = min(top_k * 3, len(self._corpus))  # over-retrieve for fusion

        # Dense retrieval
        try:
            model = self._get_or_load_model()
            q_vec = model.encode([query], show_progress_bar=False)[0].tolist()
            dense_rows = self._table.search(q_vec).limit(n).to_list()
            dense_ranked = [
                (row["id"], self._corpus[row["id"]])
                for row in dense_rows
                if row["id"] < len(self._corpus)
            ]
        except Exception as e:
            logger.debug("Dense retrieval failed: %s", e)
            dense_ranked = []

        # Sparse retrieval (BM25)
        try:
            tokens = query.lower().split()
            scores = self._bm25.get_scores(tokens)
            sparse_ids = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
            sparse_ranked = [(i, self._corpus[i]) for i in sparse_ids]
        except Exception as e:
            logger.debug("BM25 retrieval failed: %s", e)
            sparse_ranked = []

        if not dense_ranked and not sparse_ranked:
            return []

        fused = _rrf_fuse(dense_ranked, sparse_ranked)
        return fused[:top_k]

    def format_examples(self, examples: list[dict]) -> str:
        """
        Format retrieved examples as a compact few-shot block for injection
        into the system prompt. Each example shows intent → action type + params.
        Kept compact (<30 tokens/example) to stay within context budget.
        """
        if not examples:
            return ""
        lines = ["--- FEW-SHOT EXAMPLES (retrieved) ---"]
        for ex in examples:
            actions = ex.get("actions", [])
            if actions:
                action_summary = " → ".join(
                    f"{a['type']}({', '.join(f'{k}={v!r}' for k, v in a.get('params', {}).items() if k != 'search_type')})"
                    for a in actions
                )
            else:
                action_summary = f"NOT_IMPLEMENTED ({ex.get('category', 'unknown')})"
            lines.append(f'Intent: "{ex["intent"]}"')
            lines.append(f"→ category={ex.get('category', '?')} | {action_summary}")
        lines.append("--- END EXAMPLES ---")
        return "\n".join(lines)
