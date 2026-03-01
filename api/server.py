"""
api/server.py
-------------
FastAPI application setup with lifespan events.
Pre-loads all capsules on startup.

Run: uvicorn api.server:app --reload --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from api.routes import router, set_expert_core, set_vector_store
from core.expert_core import ExpertCore
from rag.vector_store import VectorStoreManager
from rag.ingestion import DocumentIngestionPipeline, BM25Index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: bootstrap ExpertCore and ingest all capsule knowledge."""
    logger.info("🚀 Starting SME-PLUG API Server...")

    # Initialize subsystems
    vector_store = VectorStoreManager()
    bm25_index = BM25Index()

    expert_core = ExpertCore()

    # Wire up the retriever with stores
    expert_core.retriever.set_stores(vector_store, bm25_index)

    # Bootstrap domain router (loads capsules, computes seed embeddings)
    expert_core.bootstrap()

    # Ingest all capsule knowledge (incremental — only new files processed)
    pipeline = DocumentIngestionPipeline(vector_store, bm25_index)
    for domain_id, capsule in expert_core.router._capsules.items():
        if capsule.knowledge_dir:
            pipeline.ingest_capsule(
                domain_id=domain_id,
                knowledge_dir=capsule.knowledge_dir,
                chunk_size=capsule.rag.chunk_size,
                chunk_overlap=capsule.rag.chunk_overlap,
                embedding_model=capsule.rag.embedding_model,
            )

    # Make ExpertCore globally accessible via routes
    set_expert_core(expert_core)
    set_vector_store(vector_store)

    logger.success("✅ SME-PLUG API ready!")
    yield
    logger.info("🛑 Shutting down SME-PLUG API Server...")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="SME-PLUG API",
    description=(
        "Universal Subject Matter Expert Plugin API. "
        "Hot-swappable domain expertise with Advanced RAG, "
        "decision trees, and source-of-truth citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (open for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="", tags=["SME-PLUG"])

# Serve UI
ui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui")
app.mount("/ui", StaticFiles(directory=ui_dir), name="ui")


@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(os.path.join(ui_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.server:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "true").lower() == "true",
    )
