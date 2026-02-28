"""
confidence_gate.py
------------------
Evaluates confidence scores from the domain router and decides:
 - Pass: route to capsule with high confidence
 - Clarify: ask user to pick a domain
 - Multi-domain: parallelize across top N capsules
 - Block: refuse with expert fallback message

Also formats all expert responses with proper structure.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from core.capsule_loader import CapsuleConfig


# ─────────────────────────────────────────────────────────────────────────────
# Response types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExpertResponse:
    """Fully structured expert response from ExpertCore."""
    domain_id: str
    domain_name: str
    answer: str
    citations: list[dict[str, str]]          # [{source, excerpt, relevance_score}]
    decision_tree_path: list[str]             # nodes traversed
    confidence_score: float
    detection_layer: int                      # which layer identified domain
    guardrail_passed: bool = True
    guardrail_warnings: list[str] = field(default_factory=list)
    multi_domain: bool = False
    fallback_used: bool = False
    raw_context_chunks: list[str] = field(default_factory=list)


@dataclass
class ClarificationResponse:
    """Returned when domain confidence is too low."""
    message: str
    suggested_domains: list[str]
    original_query: str


# ─────────────────────────────────────────────────────────────────────────────
# ConfidenceGate
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceGate:
    """
    Gates all domain routing decisions based on confidence thresholds.
    Provides structured response formatting and fallback handling.
    """

    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
    MULTI_DOMAIN_THRESHOLD: float = float(os.getenv("MULTI_DOMAIN_THRESHOLD", "0.70"))

    def __init__(self):
        logger.debug(
            f"ConfidenceGate initialized | "
            f"threshold={self.CONFIDENCE_THRESHOLD} | "
            f"multi_domain_threshold={self.MULTI_DOMAIN_THRESHOLD}"
        )

    def evaluate(
        self,
        confidence: float,
        domain_id: str,
        capsule: CapsuleConfig,
        query: str,
        detection_layer: int,
    ) -> dict[str, Any]:
        """
        Returns decision dict:
        {
          "decision": "pass" | "clarify" | "block",
          "domain_id": ...,
          "confidence": ...,
        }
        """
        if confidence >= self.CONFIDENCE_THRESHOLD:
            logger.info(f"Gate PASS: {domain_id} (conf={confidence:.2f}, layer={detection_layer})")
            return {"decision": "pass", "domain_id": domain_id, "confidence": confidence}

        logger.warning(f"Gate BLOCK: {domain_id} (conf={confidence:.2f} below threshold)")
        return {
            "decision": "clarify",
            "domain_id": domain_id,
            "confidence": confidence,
            "fallback_message": capsule.fallback_message.format(domain=capsule.name),
        }

    def format_response(
        self,
        raw_answer: str,
        citations: list[dict],
        capsule: CapsuleConfig,
        decision_layer: int,
        confidence: float,
        decision_tree_path: list[str] | None = None,
        guardrail_warnings: list[str] | None = None,
        multi_domain: bool = False,
        raw_chunks: list[str] | None = None,
    ) -> ExpertResponse:
        """Package all response components into a structured ExpertResponse."""
        return ExpertResponse(
            domain_id=capsule.domain_id,
            domain_name=capsule.name,
            answer=raw_answer,
            citations=citations or [],
            decision_tree_path=decision_tree_path or [],
            confidence_score=confidence,
            detection_layer=decision_layer,
            guardrail_passed=len(guardrail_warnings or []) == 0,
            guardrail_warnings=guardrail_warnings or [],
            multi_domain=multi_domain,
            raw_context_chunks=raw_chunks or [],
        )

    def build_clarification(
        self,
        message: str,
        available_domains: list[str],
        query: str,
    ) -> ClarificationResponse:
        return ClarificationResponse(
            message=message,
            suggested_domains=available_domains,
            original_query=query,
        )

    def format_fallback(self, capsule: CapsuleConfig, query: str) -> ExpertResponse:
        """Returns a safe fallback response when guardrails block the answer."""
        fallback_msg = (
            f"⚠️ **Confidence too low to provide a reliable expert answer.**\n\n"
            f"{capsule.fallback_message.format(domain=capsule.name)}\n\n"
            f"*Your query has been logged. Please consult a verified expert.*"
        )
        return ExpertResponse(
            domain_id=capsule.domain_id,
            domain_name=capsule.name,
            answer=fallback_msg,
            citations=[],
            decision_tree_path=[],
            confidence_score=0.0,
            detection_layer=0,
            guardrail_passed=False,
            guardrail_warnings=["Low confidence — fallback triggered"],
            fallback_used=True,
        )
