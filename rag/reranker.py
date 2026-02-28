"""
reranker.py
-----------
Cross-encoder reranker using FlashRank.

Takes top-N chunks from the hybrid retriever and reranks them using a
cross-encoder model that jointly encodes (query, passage) pairs,
producing much more accurate relevance scores than bi-encoder similarity.

Pipeline position:
  BM25 + Dense → RRF (top-20) → Cross-Encoder Reranker → Top-5 context chunks → LLM
"""

from loguru import logger


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using FlashRank's cross-encoder.
    Falls back to original ranking if FlashRank is unavailable.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self._model_name = model_name
        self._ranker = None
        self._initialized = False
        logger.debug(f"CrossEncoderReranker initialized (model={model_name}).")

    def _init_ranker(self):
        """Lazy-load the reranker model."""
        if self._initialized:
            return
        try:
            from flashrank import Ranker, RerankRequest
            self._ranker = Ranker(model_name=self._model_name)
            self._initialized = True
            logger.info(f"FlashRank model loaded: {self._model_name}")
        except ImportError:
            logger.warning("FlashRank not installed. Reranking disabled — using original scores.")
            self._initialized = True
        except Exception as e:
            logger.warning(f"FlashRank init failed: {e}. Reranking disabled.")
            self._initialized = True

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank retrieved chunks and return the top_k most relevant.

        Args:
            query: User query string.
            chunks: List of chunk dicts from the retriever (must have "text" key).
            top_k: How many chunks to return after reranking.

        Returns:
            list of top_k chunk dicts, sorted by cross-encoder relevance.
        """
        if not chunks:
            return []

        self._init_ranker()

        if self._ranker is None:
            # Fallback: return top_k by existing score
            logger.debug("Using fallback ranking (no reranker model).")
            return sorted(
                chunks,
                key=lambda x: x.get("rrf_score", x.get("score", 0)),
                reverse=True,
            )[:top_k]

        try:
            from flashrank import RerankRequest

            # Build passages for FlashRank
            passages = []
            for i, chunk in enumerate(chunks):
                passages.append({
                    "id": i,
                    "text": chunk.get("text", ""),
                    "meta": {"chunk_id": chunk.get("chunk_id", f"chunk_{i}")},
                })

            rerank_request = RerankRequest(query=query, passages=passages)
            reranked = self._ranker.rerank(rerank_request)

            # Map back to original chunk dicts with cross-encoder scores
            id_to_chunk = {i: chunk for i, chunk in enumerate(chunks)}
            results = []
            for item in reranked[:top_k]:
                idx = item["id"]
                chunk_data = id_to_chunk[idx].copy()
                chunk_data["rerank_score"] = float(item["score"])
                results.append(chunk_data)

            logger.info(
                f"[Reranker] {len(chunks)} chunks → top {len(results)} "
                f"(scores: {[round(r['rerank_score'], 3) for r in results]})"
            )
            return results

        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original ranking.")
            return sorted(
                chunks,
                key=lambda x: x.get("rrf_score", x.get("score", 0)),
                reverse=True,
            )[:top_k]
