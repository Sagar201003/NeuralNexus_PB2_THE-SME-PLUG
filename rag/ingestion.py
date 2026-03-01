"""
ingestion.py
------------
Document ingestion pipeline for populating capsule knowledge bases.

Pipeline: Source files (PDF/DOCX/TXT/HTML)
    → Parse with unstructured/pymupdf
    → Semantic chunking (with overlap)
    → Embed & store in ChromaDB via VectorStoreManager

Also builds BM25 index for hybrid retrieval.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional

from loguru import logger

from rag.vector_store import VectorStoreManager


# ─────────────────────────────────────────────────────────────────────────────
# Text splitter with semantic chunking
# ─────────────────────────────────────────────────────────────────────────────

class SemanticChunker:
    """Split documents into overlapping chunks preserving sentence boundaries."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, source: str = "unknown") -> list[dict]:
        """Split text into chunks with metadata."""
        if not text.strip():
            return []

        # Sentence-level splitting for better boundaries
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence)
            if current_len + sent_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_id = self._make_id(source, chunk_text, len(chunks))
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "chunk_id": chunk_id,
                    "chunk_index": len(chunks),
                })

                # Overlap: keep last N chars worth of sentences
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)
                current_chunk = overlap_sentences
                current_len = overlap_len

            current_chunk.append(sentence)
            current_len += sent_len

        # Final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = self._make_id(source, chunk_text, len(chunks))
            chunks.append({
                "text": chunk_text,
                "source": source,
                "chunk_id": chunk_id,
                "chunk_index": len(chunks),
            })

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Basic sentence splitting on periods, exclamation, and question marks."""
        import re
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw if s.strip()]

    @staticmethod
    def _make_id(source: str, text: str, idx: int) -> str:
        h = hashlib.md5(f"{source}:{text[:100]}:{idx}".encode()).hexdigest()[:12]
        return f"{Path(source).stem}_{idx}_{h}"


# ─────────────────────────────────────────────────────────────────────────────
# File parsers (multi-format)
# ─────────────────────────────────────────────────────────────────────────────

class FileParser:
    """Parse various file types into plain text."""

    @staticmethod
    def parse(file_path: str) -> str:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in (".txt", ".md", ".csv"):
            return path.read_text(encoding="utf-8", errors="ignore")

        if suffix == ".pdf":
            return FileParser._parse_pdf(file_path)

        if suffix in (".docx",):
            return FileParser._parse_docx(file_path)

        if suffix in (".html", ".htm"):
            return FileParser._parse_html(file_path)

        if suffix == ".json":
            return FileParser._parse_json(file_path)

        # Fallback: try as text
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            logger.warning(f"Could not parse file: {file_path}")
            return ""

    @staticmethod
    def _parse_pdf(path: str) -> str:
        try:
            import fitz  # pymupdf
            doc = fitz.open(path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except ImportError:
            try:
                from pypdf import PdfReader
                reader = PdfReader(path)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                logger.error("No PDF parser available. Install pymupdf or pypdf.")
                return ""

    @staticmethod
    def _parse_docx(path: str) -> str:
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            logger.error("python-docx not installed.")
            return ""

    @staticmethod
    def _parse_html(path: str) -> str:
        try:
            from bs4 import BeautifulSoup
            with open(path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "lxml")
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            logger.error("beautifulsoup4 not installed.")
            return ""

    @staticmethod
    def _parse_json(path: str) -> str:
        """Convert structured JSON (e.g. cricket match data) into readable text."""
        import json as _json
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception as e:
            logger.warning(f"JSON parse error for {path}: {e}")
            return ""

        # Cricket match JSON (Cricsheet format)
        if "info" in data and "innings" in data:
            return FileParser._cricket_match_to_text(data, Path(path).stem)

        # Generic JSON: flatten to key-value text
        return _json.dumps(data, indent=2, ensure_ascii=False)

    @staticmethod
    def _cricket_match_to_text(data: dict, match_id: str) -> str:
        """Convert a Cricsheet match JSON into a compact text summary for RAG."""
        info = data.get("info", {})
        lines = []

        teams = info.get("teams", [])
        lines.append(f"Match ID: {match_id}")
        lines.append(f"Match: {' vs '.join(teams)}")
        lines.append(f"Match Type: {info.get('match_type', 'Unknown')}")
        lines.append(f"Gender: {info.get('gender', 'Unknown')}")

        dates = info.get("dates", [])
        if dates:
            lines.append(f"Date: {dates[0]}" + (f" to {dates[-1]}" if len(dates) > 1 else ""))

        lines.append(f"Venue: {info.get('venue', 'Unknown')}")
        lines.append(f"City: {info.get('city', 'Unknown')}")
        lines.append(f"Season: {info.get('season', 'Unknown')}")

        event = info.get("event", {})
        if event:
            lines.append(f"Event: {event.get('name', '')}")
            if "match_number" in event:
                lines.append(f"Match Number: {event['match_number']}")

        toss = info.get("toss", {})
        if toss:
            lines.append(f"Toss: {toss.get('winner', '?')} won and chose to {toss.get('decision', '?')}")

        outcome = info.get("outcome", {})
        if "winner" in outcome:
            margin = outcome.get("by", {})
            margin_str = ", ".join(f"{v} {k}" for k, v in margin.items()) if margin else ""
            lines.append(f"Result: {outcome['winner']} won" + (f" by {margin_str}" if margin_str else ""))
        elif "result" in outcome:
            lines.append(f"Result: {outcome['result']}")

        pom = info.get("player_of_match", [])
        if pom:
            lines.append(f"Player of the Match: {', '.join(pom)}")

        # Players
        players = info.get("players", {})
        for team_name, roster in players.items():
            lines.append(f"{team_name} squad: {', '.join(roster)}")

        # Innings summaries
        for i, inn in enumerate(data.get("innings", []), 1):
            team = inn.get("team", f"Innings {i}")
            overs = inn.get("overs", [])
            total_runs = 0
            total_wickets = 0
            for over in overs:
                for delivery in over.get("deliveries", []):
                    runs = delivery.get("runs", {})
                    total_runs += runs.get("total", 0)
                    if "wickets" in delivery:
                        total_wickets += len(delivery["wickets"])
            overs_bowled = f"{overs[-1]['over']}.{len(overs[-1].get('deliveries', []))}" if overs else "0"
            lines.append(f"Innings {i} — {team}: {total_runs}/{total_wickets} in {overs_bowled} overs")

        # Officials
        officials = info.get("officials", {})
        umpires = officials.get("umpires", [])
        if umpires:
            lines.append(f"Umpires: {', '.join(umpires)}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# BM25 Index Builder (for hybrid retrieval)
# ─────────────────────────────────────────────────────────────────────────────

class BM25Index:
    """
    In-memory BM25 index for keyword-based retrieval.
    Stored per-domain alongside the vector store.
    """

    def __init__(self):
        self._indices: dict[str, object] = {}      # domain_id → BM25Okapi
        self._corpora: dict[str, list[dict]] = {}   # domain_id → chunk list

    def build(self, domain_id: str, chunks: list[dict]) -> None:
        """Build BM25 index from chunk dicts."""
        from rank_bm25 import BM25Okapi

        tokenized = [c["text"].lower().split() for c in chunks]
        self._indices[domain_id] = BM25Okapi(tokenized)
        self._corpora[domain_id] = chunks
        logger.info(f"BM25 index built for '{domain_id}': {len(chunks)} chunks")

    def search(self, domain_id: str, query: str, top_k: int = 20) -> list[dict]:
        """Return top-k BM25-ranked chunks."""
        if domain_id not in self._indices:
            logger.warning(f"No BM25 index for '{domain_id}'")
            return []

        bm25 = self._indices[domain_id]
        corpus = self._corpora[domain_id]
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        # Get top-k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = corpus[idx].copy()
                chunk["bm25_score"] = float(scores[idx])
                results.append(chunk)

        return results

    def has_index(self, domain_id: str) -> bool:
        return domain_id in self._indices


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion pipeline
# ─────────────────────────────────────────────────────────────────────────────

class DocumentIngestionPipeline:
    """
    End-to-end ingestion: scans knowledge_dir, parses files, chunks,
    embeds into ChromaDB, and builds BM25 index.
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        bm25_index: BM25Index | None = None,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25_index or BM25Index()
        self.parser = FileParser()
        self.chunker = SemanticChunker()

    def ingest_capsule(
        self,
        domain_id: str,
        knowledge_dir: str,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> dict:
        """
        Ingest all documents from a capsule's knowledge directory.
        Supports incremental ingestion — only new/unseen files are processed.

        Returns summary dict: {files_found, files_new, chunks_created, chunks_stored}
        """
        knowledge_path = Path(knowledge_dir)
        if not knowledge_path.exists():
            logger.warning(f"Knowledge dir not found: {knowledge_dir}")
            return {"files_found": 0, "files_new": 0, "chunks_created": 0, "chunks_stored": 0}

        self.chunker = SemanticChunker(chunk_size, chunk_overlap)

        # Discover files (including extensionless PDFs)
        supported = {".txt", ".md", ".pdf", ".docx", ".html", ".htm", ".csv", ".json"}
        files = []
        for f in knowledge_path.rglob("*"):
            if f.is_dir():
                continue
            if f.suffix.lower() in supported:
                files.append(f)
            elif f.suffix == "":  # extensionless — check if it's a PDF by magic bytes
                try:
                    with open(f, "rb") as fh:
                        header = fh.read(4)
                    if header == b"%PDF":
                        files.append(f)
                except Exception:
                    pass

        logger.info(f"Found {len(files)} files for domain '{domain_id}'")

        # ── Smart incremental ingestion: skip already-indexed files ────
        already_indexed = set()
        try:
            collection = self.vector_store.get_or_create_collection(domain_id)
            if collection.count() > 0:
                # Sample existing source names from the collection
                existing = collection.get(limit=collection.count(), include=["metadatas"])
                for meta in existing.get("metadatas", []):
                    src = meta.get("source", "")
                    if src:
                        already_indexed.add(src)
        except Exception as e:
            logger.warning(f"Could not check existing sources: {e}")

        new_files = [f for f in files if f.name not in already_indexed]
        skipped = len(files) - len(new_files)
        if skipped > 0:
            logger.info(f"⏭️  Skipping {skipped} already-indexed files for '{domain_id}'")
        if not new_files:
            logger.info(f"No new files to ingest for '{domain_id}'")
            return {"files_found": len(files), "files_new": 0, "chunks_created": 0, "chunks_stored": 0}

        logger.info(f"Ingesting {len(new_files)} NEW files for domain '{domain_id}'")

        all_chunks = []
        for file_path in new_files:
            text = self.parser.parse(str(file_path))
            if not text.strip():
                continue
            chunks = self.chunker.chunk(text, source=file_path.name)
            all_chunks.extend(chunks)
            logger.debug(f"  {file_path.name}: {len(chunks)} chunks")

        if not all_chunks:
            logger.warning(f"No content extracted from new files for domain '{domain_id}'")
            return {"files_found": len(files), "files_new": len(new_files), "chunks_created": 0, "chunks_stored": 0}

        # Store in vector DB
        stored = self.vector_store.add_documents(
            domain_id=domain_id,
            chunks=all_chunks,
            embedding_model=embedding_model,
        )

        # Rebuild BM25 index (includes all chunks — existing + new)
        self.bm25.build(domain_id, all_chunks)

        summary = {
            "files_found": len(files),
            "files_new": len(new_files),
            "chunks_created": len(all_chunks),
            "chunks_stored": stored,
        }
        logger.success(f"Ingestion complete for '{domain_id}': {summary}")
        return summary
