"""
vector_store.py
---------------
ChromaDB-based vector store manager.
Each DNA capsule gets its own isolated collection to prevent
knowledge bleed between domains.
"""

import os
from pathlib import Path
from typing import Optional

import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger

# Auto-detect GPU
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VectorStoreManager:
    """
    Manages per-capsule ChromaDB collections.
    Provides add, query, delete, and collection lifecycle methods.
    """

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._embedder: SentenceTransformer | None = None
        self._current_model_name: str | None = None
        logger.info(f"VectorStoreManager initialized | persist_dir={self.persist_dir} | device={_DEVICE}")

    # ── Embedder Management ───────────────────────────────────────────────────

    def _load_embedder(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
        """Lazy-load and cache the embedding model. Uses GPU if available."""
        if self._embedder is None or self._current_model_name != model_name:
            logger.info(f"Loading embedding model: {model_name} (device={_DEVICE})")
            self._embedder = SentenceTransformer(model_name, device=_DEVICE)
            self._current_model_name = model_name
        return self._embedder

    # ── Collection CRUD ───────────────────────────────────────────────────────

    def get_or_create_collection(self, domain_id: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection for a given domain."""
        collection_name = f"sme_{domain_id}"
        collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "domain": domain_id},
        )
        logger.debug(f"Collection ready: {collection_name} ({collection.count()} docs)")
        return collection

    def delete_collection(self, domain_id: str) -> None:
        """Delete a capsule's entire vector collection."""
        collection_name = f"sme_{domain_id}"
        try:
            self._client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Could not delete collection {collection_name}: {e}")

    def list_collections(self) -> list[str]:
        """List all domain collections."""
        return [c.name for c in self._client.list_collections()]

    # ── Add Documents ─────────────────────────────────────────────────────────

    def add_documents(
        self,
        domain_id: str,
        chunks: list[dict],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> int:
        """
        Add document chunks to a domain's vector collection.

        Args:
            domain_id: Capsule domain identifier.
            chunks: List of dicts, each with keys:
                - "text": chunk text content
                - "source": source file name
                - "chunk_id": unique chunk identifier
                - (optional) other metadata fields
            embedding_model: HuggingFace model name for embeddings.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        collection = self.get_or_create_collection(domain_id)
        embedder = self._load_embedder(embedding_model)

        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {
                "source": c.get("source", "unknown"),
                "domain": domain_id,
                "chunk_index": i,
                **{k: v for k, v in c.items() if k not in ("text", "chunk_id", "embedding")},
            }
            for i, c in enumerate(chunks)
        ]

        # Embed all texts
        embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings_list = [emb.tolist() for emb in embeddings]

        # Upsert in batches (ChromaDB batch limit ~5461)
        batch_size = 5000
        added = 0
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings_list[start:end],
                metadatas=metadatas[start:end],
            )
            added += len(ids[start:end])

        logger.success(f"Added {added} chunks to collection 'sme_{domain_id}' (total: {collection.count()})")
        return added

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        domain_id: str,
        query_text: str,
        top_k: int = 20,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        where: dict | None = None,
    ) -> list[dict]:
        """
        Dense vector query against a domain's collection.

        Returns list of dicts: [{"text", "source", "score", "chunk_id", ...}]
        """
        collection = self.get_or_create_collection(domain_id)
        if collection.count() == 0:
            logger.warning(f"Empty collection for domain '{domain_id}'")
            return []

        embedder = self._load_embedder(embedding_model)
        query_emb = embedder.encode([query_text], normalize_embeddings=True)[0].tolist()

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        # Flatten Chroma's nested list structure
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        output = []
        for text, meta, dist, chunk_id in zip(docs, metas, dists, ids):
            output.append({
                "text": text,
                "source": meta.get("source", "unknown"),
                "score": 1.0 - dist,  # cosine distance → similarity
                "chunk_id": chunk_id,
                "domain": domain_id,
                **{k: v for k, v in meta.items() if k not in ("source", "domain")},
            })

        return output

    # ── Stats ─────────────────────────────────────────────────────────────────

    def collection_stats(self, domain_id: str) -> dict:
        collection = self.get_or_create_collection(domain_id)
        return {
            "domain_id": domain_id,
            "collection_name": f"sme_{domain_id}",
            "document_count": collection.count(),
        }
