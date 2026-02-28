"""
api/models.py
-------------
Pydantic request/response models for the FastAPI server.
"""

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question or task", min_length=1)
    force_domain: Optional[str] = Field(None, description="Bypass domain detection and use this capsule")


class CitationItem(BaseModel):
    source: str
    excerpt: str
    relevance_score: float


class ExpertResponseModel(BaseModel):
    domain_id: str
    domain_name: str
    answer: str
    citations: list[CitationItem]
    decision_tree_path: list[str]
    confidence_score: float
    detection_layer: int
    guardrail_passed: bool
    guardrail_warnings: list[str]
    multi_domain: bool
    fallback_used: bool


class ClarificationResponseModel(BaseModel):
    message: str
    suggested_domains: list[str]
    original_query: str


class CapsuleInfoModel(BaseModel):
    domain_id: str
    name: str
    version: str
    description: str


class HealthResponse(BaseModel):
    status: str
    capsules_loaded: int
    capsule_ids: list[str]


class IngestRequest(BaseModel):
    domain_id: str = Field(..., description="Domain slug to ingest documents for")


class IngestResponse(BaseModel):
    domain_id: str
    files_found: int
    chunks_created: int
    chunks_stored: int
