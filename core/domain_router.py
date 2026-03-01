"""
domain_router.py
----------------
3-Layer Cascade Domain Detection Engine.

Layer 1: Keyword / Regex matching (< 5ms, no API calls)
Layer 2: Zero-shot LLM classifier (Groq Llama 3.1 structured output)
Layer 3: Embedding cosine similarity against capsule seed queries

Falls back to ClarificationRequest if confidence < threshold.
Supports multi-capsule routing for cross-domain queries.
"""

import re
import os
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from loguru import logger
from groq import Groq
from sentence_transformers import SentenceTransformer

# Auto-detect GPU
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from core.capsule_loader import CapsuleLoader, CapsuleConfig


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DomainMatch:
    domain_id: str
    confidence: float
    detection_layer: int          # 1, 2, or 3
    capsule: CapsuleConfig | None = None

@dataclass
class DomainResult:
    matches: list[DomainMatch]    # sorted desc by confidence
    needs_clarification: bool = False
    clarification_message: str = ""
    multi_domain: bool = False    # True if cross-domain query detected

    @property
    def primary(self) -> DomainMatch | None:
        return self.matches[0] if self.matches else None


# ─────────────────────────────────────────────────────────────────────────────
# DomainRouter
# ─────────────────────────────────────────────────────────────────────────────

class DomainRouter:
    

    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
    MULTI_DOMAIN_THRESHOLD = float(os.getenv("MULTI_DOMAIN_THRESHOLD", "0.70"))  # both domains above this → multi

    def __init__(self, capsule_loader: CapsuleLoader):
        self.loader = capsule_loader
        self._llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._embedder: SentenceTransformer | None = None
        self._seed_embeddings: dict[str, np.ndarray] = {}   # domain_id → matrix
        self._capsules: dict[str, CapsuleConfig] = {}
        self._compiled_triggers: dict[str, list[re.Pattern]] = {}

    # ── Bootstrap ─────────────────────────────────────────────────────────────

    def bootstrap(self) -> None:
        """Load all capsules and pre-compute seed embeddings for L3."""
        self._capsules = self.loader.load_all()
        if not self._capsules:
            logger.warning("No capsules loaded. Domain router will always return clarification.")
            return

        logger.info(f"Bootstrapping domain router with {len(self._capsules)} capsules...")

        # Pre-compile regex patterns per capsule (L1)
        for domain_id, cfg in self._capsules.items():
            self._compiled_triggers[domain_id] = [
                re.compile(p, re.IGNORECASE)
                for p in cfg.triggers.regex_patterns
            ]

        all_have_seeds = any(cfg.triggers.seed_queries for cfg in self._capsules.values())
        if all_have_seeds:
            logger.info(f"Computing seed query embeddings for L3 detection (device={_DEVICE})...")
            self._embedder = SentenceTransformer(
                next(iter(self._capsules.values())).rag.embedding_model,
                device=_DEVICE,
            )
            for domain_id, cfg in self._capsules.items():
                if cfg.triggers.seed_queries:
                    embs = self._embedder.encode(
                        cfg.triggers.seed_queries, normalize_embeddings=True
                    )
                    self._seed_embeddings[domain_id] = np.array(embs)
                    logger.debug(f"  [{domain_id}] {len(cfg.triggers.seed_queries)} seed embeddings computed.")

        logger.success("Domain router ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def route(self, query: str) -> DomainResult:
        """
        Main routing method. Returns DomainResult with matched domains,
        confidence scores, which detection layer fired, and clarification if needed.
        """
        logger.info(f"Routing query: '{query[:80]}...' " if len(query) > 80 else f"Routing: '{query}'")

        # Layer 1: Keyword/Regex
        l1_matches = self._layer1_keyword(query)
        if l1_matches and l1_matches[0].confidence >= self.CONFIDENCE_THRESHOLD:
            logger.info(f"[L1] Domain detected: {l1_matches[0].domain_id} (conf={l1_matches[0].confidence:.2f})")
            return self._build_result(l1_matches)

        # Layer 2: Zero-shot LLM
        l2_matches = self._layer2_llm(query)
        if l2_matches and l2_matches[0].confidence >= self.CONFIDENCE_THRESHOLD:
            logger.info(f"[L2] Domain detected: {l2_matches[0].domain_id} (conf={l2_matches[0].confidence:.2f})")
            return self._build_result(l2_matches)

        if self._seed_embeddings:
            l3_matches = self._layer3_embedding(query)
            if l3_matches and l3_matches[0].confidence >= self.CONFIDENCE_THRESHOLD:
                logger.info(f"[L3] Domain detected: {l3_matches[0].domain_id} (conf={l3_matches[0].confidence:.2f})")
                return self._build_result(l3_matches)

        logger.warning("All layers failed to detect domain with sufficient confidence.")
        available = ", ".join(
            f"'{cfg.name}'" for cfg in self._capsules.values()
        )
        return DomainResult(
            matches=[],
            needs_clarification=True,
            clarification_message=(
                f"I can act as an expert in: {available}. "
                "Which domain best fits your question?"
            ),
        )


    def _layer1_keyword(self, query: str) -> list[DomainMatch]:
        query_lower = query.lower()
        results = []

        for domain_id, cfg in self._capsules.items():
            score = 0.0
            max_score = cfg.triggers.min_keyword_hits + len(self._compiled_triggers.get(domain_id, []))
            if max_score == 0:
                continue

            # Keyword hits
            kw_hits = sum(1 for kw in cfg.triggers.keywords if kw.lower() in query_lower)
            # Regex hits
            rx_hits = sum(1 for p in self._compiled_triggers.get(domain_id, []) if p.search(query))

            if kw_hits >= cfg.triggers.min_keyword_hits or rx_hits > 0:
                # Confidence: normalized hit count, capped at 0.95
                raw_conf = min(0.95, (kw_hits + rx_hits * 2) / max(max_score, 1))
                score = max(raw_conf, 0.60)  # floor at 0.60 if threshold passed
                results.append(DomainMatch(
                    domain_id=domain_id,
                    confidence=score,
                    detection_layer=1,
                    capsule=cfg,
                ))

        return sorted(results, key=lambda x: x.confidence, reverse=True)


    def _layer2_llm(self, query: str) -> list[DomainMatch]:
        domain_list = list(self._capsules.keys())
        if not domain_list:
            return []

        system = (
            "You are a domain classifier. Given a user query, classify it into one of the available domains. "
            "Respond ONLY with a valid JSON object — no markdown, no explanation."
        )
        user = (
            f"Available domains: {json.dumps(domain_list)}\n\n"
            f"Query: \"{query}\"\n\n"
            "Respond with JSON: "
            "{\"domain\": \"<domain_id or null>\", \"confidence\": <0.0-1.0>, "
            "\"secondary_domain\": \"<domain_id or null>\", \"secondary_confidence\": <0.0-1.0>}"
        )

        try:
            response = self._llm.chat.completions.create(
                model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"L2 LLM classifier failed: {e}")
            return []

        results = []
        primary_domain = data.get("domain")
        if primary_domain and primary_domain in self._capsules:
            results.append(DomainMatch(
                domain_id=primary_domain,
                confidence=float(data.get("confidence", 0.0)),
                detection_layer=2,
                capsule=self._capsules[primary_domain],
            ))
        secondary_domain = data.get("secondary_domain")
        if secondary_domain and secondary_domain in self._capsules:
            results.append(DomainMatch(
                domain_id=secondary_domain,
                confidence=float(data.get("secondary_confidence", 0.0)),
                detection_layer=2,
                capsule=self._capsules[secondary_domain],
            ))

        return sorted(results, key=lambda x: x.confidence, reverse=True)


    def _layer3_embedding(self, query: str) -> list[DomainMatch]:
        if not self._embedder or not self._seed_embeddings:
            return []

        query_emb = self._embedder.encode([query], normalize_embeddings=True)[0]
        results = []

        for domain_id, seed_matrix in self._seed_embeddings.items():
            # Cosine similarity = dot product (since both are normalized)
            similarities = seed_matrix @ query_emb
            avg_top5 = float(np.sort(similarities)[-5:].mean())  # mean of top-5 seed matches
            results.append(DomainMatch(
                domain_id=domain_id,
                confidence=avg_top5,
                detection_layer=3,
                capsule=self._capsules.get(domain_id),
            ))

        return sorted(results, key=lambda x: x.confidence, reverse=True)


    def _build_result(self, matches: list[DomainMatch]) -> DomainResult:
        """Determine if this is a multi-domain query based on top-2 confidence scores."""
        multi = (
            len(matches) >= 2
            and matches[0].confidence >= self.MULTI_DOMAIN_THRESHOLD
            and matches[1].confidence >= self.MULTI_DOMAIN_THRESHOLD
        )
        if multi:
            logger.info(
                f"Multi-domain query detected: "
                f"{matches[0].domain_id} + {matches[1].domain_id}"
            )
        return DomainResult(matches=matches, multi_domain=multi)
