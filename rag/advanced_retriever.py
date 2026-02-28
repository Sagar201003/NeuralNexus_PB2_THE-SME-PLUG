"""
advanced_retriever.py
---------------------
Hybrid retrieval with Reciprocal Rank Fusion (RRF).

Combines:
  1. BM25 keyword retrieval (top-K sparse matches)
  2. Dense vector retrieval (top-K semantic matches via ChromaDB)
  3. RRF fusion to merge both ranked lists into a single unified ranking

This is the core "Advanced RAG" retriever — far superior to naive
single-method retrieval for domain-specific queries.
"""

import os
from typing import Optional

from loguru import logger

from rag.vector_store import VectorStoreManager
from rag.ingestion import BM25Index


class AdvancedRetriever:
    """
    Hybrid BM25 + Dense retriever with Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = Σ 1 / (k + rank_i(d))
    where k=60 (standard), and i ∈ {BM25, Dense}
    """

    RRF_K: int = 60  # standard RRF constant

    def __init__(
        self,
        vector_store: VectorStoreManager | None = None,
        bm25_index: BM25Index | None = None,
    ):
        self.vector_store = vector_store or VectorStoreManager()
        self.bm25 = bm25_index or BM25Index()
        logger.debug("AdvancedRetriever initialized (Hybrid BM25 + Dense + RRF).")

    def set_stores(self, vector_store: VectorStoreManager, bm25_index: BM25Index) -> None:
        """Set/replace vector store and BM25 index (used during bootstrap)."""
        self.vector_store = vector_store
        self.bm25 = bm25_index

    # ── Main retrieval pipeline ───────────────────────────────────────────────

    def retrieve(
        self,
        original_query: str,
        hyde_query: str | None = None,
        domain_id: str = "",
        top_k_bm25: int = 20,
        top_k_dense: int = 20,
        top_k_final: int = 20,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> list[dict]:
        """
        Perform hybrid retrieval with RRF fusion.

        Args:
            original_query: The user's raw query (used for BM25).
            hyde_query: HyDE-expanded query (used for dense). Falls back to original.
            domain_id: Which capsule's collection to search.
            top_k_bm25: Number of BM25 results to fetch.
            top_k_dense: Number of dense results to fetch.
            top_k_final: Number of final fused results to return.
            embedding_model: Model name for dense embeddings.

        Returns:
            Sorted list of chunk dicts with unified RRF scores.
        """
        dense_query = hyde_query or original_query

        # ── BM25 Retrieval ────────────────────────────────────────────────────
        bm25_results = []
        if self.bm25.has_index(domain_id):
            bm25_results = self.bm25.search(domain_id, original_query, top_k=top_k_bm25)
            logger.debug(f"BM25 returned {len(bm25_results)} results for '{domain_id}'")
        else:
            logger.debug(f"No BM25 index for '{domain_id}', skipping sparse retrieval.")

        # ── Dense Retrieval ───────────────────────────────────────────────────
        dense_results = self.vector_store.query(
            domain_id=domain_id,
            query_text=dense_query,
            top_k=top_k_dense,
            embedding_model=embedding_model,
        )
        logger.debug(f"Dense returned {len(dense_results)} results for '{domain_id}'")

        # ── Reciprocal Rank Fusion ────────────────────────────────────────────
        fused = self._rrf_fuse(bm25_results, dense_results)

        # Sort by fused RRF score descending, take top-k
        fused_sorted = sorted(fused, key=lambda x: x["rrf_score"], reverse=True)[:top_k_final]

        logger.info(
            f"[Hybrid Retrieval] BM25={len(bm25_results)} + Dense={len(dense_results)} "
            f"→ RRF fused={len(fused_sorted)} chunks"
        )
        return fused_sorted

    # ── RRF Fusion ────────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        bm25_results: list[dict],
        dense_results: list[dict],
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.
        Merges two ranked lists into one using RRF scores.
        """
        chunk_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}
        k = self.RRF_K

        # Score BM25 results
        for rank, chunk in enumerate(bm25_results, start=1):
            cid = chunk.get("chunk_id", f"bm25_{rank}")
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunk_data:
                chunk_data[cid] = chunk.copy()
                chunk_data[cid]["retrieval_methods"] = []
            chunk_data[cid]["retrieval_methods"].append("bm25")

        # Score Dense results
        for rank, chunk in enumerate(dense_results, start=1):
            cid = chunk.get("chunk_id", f"dense_{rank}")
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in chunk_data:
                chunk_data[cid] = chunk.copy()
                chunk_data[cid]["retrieval_methods"] = []
            chunk_data[cid]["retrieval_methods"].append("dense")

        # Merge scores into chunk data
        fused = []
        for cid, score in chunk_scores.items():
            data = chunk_data[cid]
            data["rrf_score"] = score
            data["retrieval_methods"] = list(set(data.get("retrieval_methods", [])))
            fused.append(data)

        return fused

    # ── Dense-only fallback ───────────────────────────────────────────────────

    def retrieve_dense_only(
        self,
        query: str,
        domain_id: str,
        top_k: int = 10,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> list[dict]:
        """Fallback: dense-only retrieval without BM25 or RRF."""
        return self.vector_store.query(
            domain_id=domain_id,
            query_text=query,
            top_k=top_k,
            embedding_model=embedding_model,
        )
