"""
capsule_loader.py
-----------------
Loads and validates DNA capsule YAML configurations.
Each capsule defines expert persona, triggers, decision tree ref,
RAG settings, validation rules, and guardrail config.
"""

import os
import json
from pathlib import Path
from typing import Any, Optional
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field, validator
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models for capsule schema validation
# ─────────────────────────────────────────────────────────────────────────────

class TriggerConfig(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    regex_patterns: list[str] = Field(default_factory=list)
    min_keyword_hits: int = 2
    seed_queries: list[str] = Field(default_factory=list, description="50 representative queries for embedding similarity (L3 detection)")


class RAGConfig(BaseModel):
    top_k_bm25: int = 20
    top_k_dense: int = 20
    top_k_reranked: int = 5
    use_hyde: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class GuardrailConfig(BaseModel):
    min_confidence: float = 0.55
    require_citations: bool = True
    hallucination_threshold: float = 0.35
    domain_validation_rules: list[dict[str, Any]] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)


class CapsuleConfig(BaseModel):
    """Full validated DNA capsule configuration."""
    name: str
    version: str = "1.0.0"
    domain_id: str                          # unique slug, e.g. "structural_engineering"
    description: str
    expert_persona: str                     # injected system prompt
    triggers: TriggerConfig
    rag: RAGConfig = Field(default_factory=RAGConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    knowledge_dir: Optional[str] = None     # path to knowledge docs
    decision_tree_path: Optional[str] = None
    sources_of_truth: list[str] = Field(default_factory=list)  # e.g. ["AISC 360-22", "IS 456:2000"]
    fallback_message: str = "I cannot verify this with sufficient confidence. Please consult a certified {domain} expert."

    @validator("domain_id")
    def domain_id_slugify(cls, v: str) -> str:
        return v.lower().replace(" ", "_").replace("-", "_")


# ─────────────────────────────────────────────────────────────────────────────
# CapsuleLoader: discovers, loads, and caches capsules
# ─────────────────────────────────────────────────────────────────────────────

class CapsuleLoader:
    """
    Discovers all capsule directories, validates their YAML configs,
    and caches CapsuleConfig instances for fast hot-swapping.
    """

    def __init__(self, capsules_dir: str | None = None):
        self.capsules_dir = Path(capsules_dir or os.getenv("CAPSULES_DIR", "./capsules"))
        self._cache: dict[str, CapsuleConfig] = {}
        self._decision_tree_cache: dict[str, dict] = {}
        logger.info(f"CapsuleLoader initialized. Capsules dir: {self.capsules_dir}")

    def discover_all(self) -> list[str]:
        """Return list of all available domain_ids by scanning capsule directories."""
        if not self.capsules_dir.exists():
            logger.warning(f"Capsules directory not found: {self.capsules_dir}")
            return []
        domain_ids = []
        for entry in sorted(self.capsules_dir.iterdir()):
            if entry.is_dir() and (entry / "capsule.yaml").exists():
                domain_ids.append(entry.name)
        logger.info(f"Discovered {len(domain_ids)} capsules: {domain_ids}")
        return domain_ids

    def load(self, domain_id: str) -> CapsuleConfig:
        """Load and validate a capsule by domain_id. Cached after first load."""
        if domain_id in self._cache:
            return self._cache[domain_id]

        capsule_path = self.capsules_dir / domain_id / "capsule.yaml"
        if not capsule_path.exists():
            raise FileNotFoundError(f"Capsule not found: {capsule_path}")

        with open(capsule_path, "r") as f:
            raw = yaml.safe_load(f)

        # Resolve relative paths to absolute
        capsule_dir = str(self.capsules_dir / domain_id)
        if raw.get("knowledge_dir") and not Path(raw["knowledge_dir"]).is_absolute():
            raw["knowledge_dir"] = str(self.capsules_dir / domain_id / raw["knowledge_dir"])
        if raw.get("decision_tree_path") and not Path(raw["decision_tree_path"]).is_absolute():
            raw["decision_tree_path"] = str(self.capsules_dir / domain_id / raw["decision_tree_path"])

        config = CapsuleConfig(**raw)
        self._cache[domain_id] = config
        logger.success(f"Loaded capsule: [{domain_id}] v{config.version}")
        return config

    def load_all(self) -> dict[str, CapsuleConfig]:
        """Load all discovered capsules and return as {domain_id: CapsuleConfig} dict."""
        all_capsules = {}
        for domain_id in self.discover_all():
            try:
                all_capsules[domain_id] = self.load(domain_id)
            except Exception as e:
                logger.error(f"Failed to load capsule '{domain_id}': {e}")
        return all_capsules

    def load_decision_tree(self, domain_id: str) -> dict:
        """Load the decision tree JSON for a given capsule."""
        if domain_id in self._decision_tree_cache:
            return self._decision_tree_cache[domain_id]

        capsule = self.load(domain_id)
        if not capsule.decision_tree_path:
            return {}

        dt_path = Path(capsule.decision_tree_path)
        if not dt_path.exists():
            logger.warning(f"Decision tree not found: {dt_path}")
            return {}

        with open(dt_path, "r") as f:
            tree = json.load(f)

        self._decision_tree_cache[domain_id] = tree
        return tree

    def invalidate(self, domain_id: str) -> None:
        """Force reload a capsule (hot-swap)."""
        self._cache.pop(domain_id, None)
        self._decision_tree_cache.pop(domain_id, None)
        logger.info(f"Invalidated capsule cache for: {domain_id}")

    def invalidate_all(self) -> None:
        self._cache.clear()
        self._decision_tree_cache.clear()
        logger.info("All capsule caches invalidated.")
