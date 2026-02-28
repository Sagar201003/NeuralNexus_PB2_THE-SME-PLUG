"""
api/routes.py
-------------
FastAPI routes for the SME-PLUG API server.

Endpoints:
  POST /query           — Query ExpertCore with auto domain detection
  GET  /capsules        — List all loaded capsules
  POST /capsule/ingest  — Ingest documents for a capsule
  GET  /health          — Health check
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models import (
    QueryRequest,
    ExpertResponseModel,
    ClarificationResponseModel,
    CapsuleInfoModel,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    CitationItem,
)
from core.expert_core import ExpertCore
from core.confidence_gate import ExpertResponse, ClarificationResponse
from rag.vector_store import VectorStoreManager
from rag.ingestion import DocumentIngestionPipeline


router = APIRouter()

# Global ExpertCore instance (initialized in server.py lifespan)
_expert_core: ExpertCore | None = None
_vector_store: VectorStoreManager | None = None


def set_expert_core(ec: ExpertCore) -> None:
    global _expert_core
    _expert_core = ec


def set_vector_store(vs: VectorStoreManager) -> None:
    global _vector_store
    _vector_store = vs


def get_expert_core() -> ExpertCore:
    if _expert_core is None:
        raise HTTPException(status_code=503, detail="ExpertCore not initialized")
    return _expert_core


# ── POST /query ──────────────────────────────────────────────────────────────

@router.post("/query", response_model=ExpertResponseModel | ClarificationResponseModel)
async def query_expert(request: QueryRequest):
    """
    Query the SME-PLUG with auto domain detection.

    Optionally force a specific domain via `force_domain`.
    Returns expert response with citations, decision tree path,
    and guardrail warnings.
    """
    ec = get_expert_core()

    try:
        result = ec.query(request.query, force_domain=request.force_domain)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    if isinstance(result, ClarificationResponse):
        return ClarificationResponseModel(
            message=result.message,
            suggested_domains=result.suggested_domains,
            original_query=result.original_query,
        )

    if isinstance(result, ExpertResponse):
        return ExpertResponseModel(
            domain_id=result.domain_id,
            domain_name=result.domain_name,
            answer=result.answer,
            citations=[
                CitationItem(
                    source=c.get("source", "unknown"),
                    excerpt=c.get("excerpt", ""),
                    relevance_score=c.get("relevance_score", 0.0),
                )
                for c in result.citations
            ],
            decision_tree_path=result.decision_tree_path,
            confidence_score=result.confidence_score,
            detection_layer=result.detection_layer,
            guardrail_passed=result.guardrail_passed,
            guardrail_warnings=result.guardrail_warnings,
            multi_domain=result.multi_domain,
            fallback_used=result.fallback_used,
        )


# ── GET /capsules ────────────────────────────────────────────────────────────

@router.get("/capsules", response_model=list[CapsuleInfoModel])
async def list_capsules():
    """List all loaded DNA capsules."""
    ec = get_expert_core()
    capsules = ec.list_capsules()
    return [CapsuleInfoModel(**c) for c in capsules]


# ── POST /capsule/ingest ─────────────────────────────────────────────────────

@router.post("/capsule/ingest", response_model=IngestResponse)
async def ingest_capsule(request: IngestRequest):
    """Ingest documents for a specific capsule into the vector store."""
    ec = get_expert_core()

    try:
        capsule = ec.loader.load(request.domain_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Capsule '{request.domain_id}' not found.")

    if not capsule.knowledge_dir:
        raise HTTPException(status_code=400, detail=f"No knowledge_dir set for '{request.domain_id}'.")

    vs = _vector_store or VectorStoreManager()
    pipeline = DocumentIngestionPipeline(vs)

    summary = pipeline.ingest_capsule(
        domain_id=request.domain_id,
        knowledge_dir=capsule.knowledge_dir,
        chunk_size=capsule.rag.chunk_size,
        chunk_overlap=capsule.rag.chunk_overlap,
        embedding_model=capsule.rag.embedding_model,
    )

    return IngestResponse(
        domain_id=request.domain_id,
        files_found=summary["files_found"],
        chunks_created=summary["chunks_created"],
        chunks_stored=summary["chunks_stored"],
    )


# ── GET /health ──────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    ec = get_expert_core()
    capsules = ec.list_capsules()
    return HealthResponse(
        status="healthy",
        capsules_loaded=len(capsules),
        capsule_ids=[c["domain_id"] for c in capsules],
    )
