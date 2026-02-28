"""
expert_core.py
--------------
Main ExpertCore class — the single entrypoint for the SME-PLUG.

Usage:
    ec = ExpertCore()
    ec.bootstrap()
    result = ec.query("Is this beam design safe for 500 kN load?")

Or inject into any agent:
    from adapters.langchain_adapter import LangChainAdapter
    adapter = LangChainAdapter(ec)
    adapter.inject(my_langchain_agent)
"""

import os
import json
from typing import Any, Optional, Union

import networkx as nx
from loguru import logger

from core.capsule_loader import CapsuleLoader, CapsuleConfig
from core.domain_router import DomainRouter, DomainResult
from core.confidence_gate import ConfidenceGate, ExpertResponse, ClarificationResponse
from rag.advanced_retriever import AdvancedRetriever
from rag.hyde_engine import HyDEEngine
from rag.reranker import CrossEncoderReranker
from guardrails.hallucination_detector import HallucinationDetector
from guardrails.citation_enforcer import CitationEnforcer
from guardrails.output_validator import OutputValidator


class ExpertCore:
    """
    Universal SME Plugin — hot-swappable domain expertise for any AI agent.

    Architecture:
      Query
        → DomainRouter (3-layer detection)
        → ConfidenceGate (pass / clarify / block)
        → CapsuleLoader (load DNA capsule)
        → AdvancedRetriever (HyDE + Hybrid BM25+Dense + RRF + Reranker)
        → DecisionTree walker
        → LLM call with expert persona + context + decision path
        → Guardrails (hallucination check + citation enforcement + domain validation)
        → ExpertResponse
    """

    def __init__(
        self,
        capsules_dir: str | None = None,
        groq_api_key: str | None = None,
    ):
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key

        self.loader = CapsuleLoader(capsules_dir)
        self.router = DomainRouter(self.loader)
        self.gate = ConfidenceGate()
        self.retriever = AdvancedRetriever()
        self.hyde = HyDEEngine()
        self.reranker = CrossEncoderReranker()
        self.hallucination_detector = HallucinationDetector()
        self.citation_enforcer = CitationEnforcer()
        self.output_validator = OutputValidator()
        self._bootstrapped = False
        logger.info("ExpertCore initialized.")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def bootstrap(self) -> None:
        """
        Pre-load all capsules, compute seed embeddings, warm up vector stores.
        Call once before serving queries (e.g., on app startup).
        """
        logger.info("Bootstrapping ExpertCore...")
        self.router.bootstrap()
        self._bootstrapped = True
        logger.success("ExpertCore ready. Capsules loaded: " + str(list(self.router._capsules.keys())))

    def hot_swap(self, domain_id: str) -> None:
        """Reload a single capsule without restarting (true hot-swap)."""
        self.loader.invalidate(domain_id)
        self.router.bootstrap()
        logger.success(f"Hot-swapped capsule: {domain_id}")

    # ── Core query pipeline ───────────────────────────────────────────────────

    def query(
        self,
        query: str,
        force_domain: str | None = None,
    ) -> Union[ExpertResponse, ClarificationResponse]:
        """
        Process a user query through the full ExpertCore pipeline.

        Args:
            query: The user's question or task.
            force_domain: Bypass domain detection and use this capsule directly.

        Returns:
            ExpertResponse (success) or ClarificationResponse (low confidence).
        """
        if not self._bootstrapped:
            self.bootstrap()

        # ── Step 1: Domain Detection ─────────────────────────────────────────
        if force_domain:
            capsule = self.loader.load(force_domain)
            domain_result = None
            confidence = 1.0
            detection_layer = 0
        else:
            domain_result: DomainResult = self.router.route(query)

            if domain_result.needs_clarification:
                return self.gate.build_clarification(
                    message=domain_result.clarification_message,
                    available_domains=list(self.router._capsules.keys()),
                    query=query,
                )

            primary = domain_result.primary
            capsule = primary.capsule
            confidence = primary.confidence
            detection_layer = primary.detection_layer

        # ── Step 2: HyDE query expansion ────────────────────────────────────
        logger.info(f"[ExpertCore] Running HyDE for domain: {capsule.domain_id}")
        hyde_query = self.hyde.expand(query, capsule) if capsule.rag.use_hyde else query

        # ── Step 3: Advanced Retrieval (BM25 + Dense + RRF) ─────────────────
        logger.info("[ExpertCore] Retrieving context chunks (hybrid retrieval)...")
        raw_chunks = self.retriever.retrieve(
            original_query=query,
            hyde_query=hyde_query,
            domain_id=capsule.domain_id,
            top_k_bm25=capsule.rag.top_k_bm25,
            top_k_dense=capsule.rag.top_k_dense,
        )

        # ── Step 4: Cross-encoder Reranking ─────────────────────────────────
        logger.info(f"[ExpertCore] Reranking {len(raw_chunks)} chunks → top {capsule.rag.top_k_reranked}...")
        reranked_chunks = self.reranker.rerank(
            query=query,
            chunks=raw_chunks,
            top_k=capsule.rag.top_k_reranked,
        )

        # ── Step 5: Decision Tree Traversal ──────────────────────────────────
        dt_path = self._walk_decision_tree(capsule.domain_id, query, reranked_chunks)

        # ── Step 6: Expert LLM Call ──────────────────────────────────────────
        logger.info("[ExpertCore] Calling LLM with expert context...")
        raw_answer = self._call_expert_llm(
            query=query,
            capsule=capsule,
            context_chunks=reranked_chunks,
            decision_path=dt_path,
        )

        # ── Step 7: Guardrails ───────────────────────────────────────────────
        hallucination_warnings = self.hallucination_detector.check(
            answer=raw_answer,
            context_chunks=reranked_chunks,
        )
        cited_answer, citations = self.citation_enforcer.enforce(
            answer=raw_answer,
            chunks=reranked_chunks,
            sources_of_truth=capsule.sources_of_truth,
        )
        validation_warnings = self.output_validator.validate(
            answer=cited_answer,
            capsule=capsule,
        )

        all_warnings = hallucination_warnings + validation_warnings

        # ── Step 8: Format and return ────────────────────────────────────────
        return self.gate.format_response(
            raw_answer=cited_answer,
            citations=citations,
            capsule=capsule,
            decision_layer=detection_layer,
            confidence=confidence,
            decision_tree_path=dt_path,
            guardrail_warnings=all_warnings,
            multi_domain=getattr(domain_result, "multi_domain", False) if domain_result else False,
            raw_chunks=[c.get("text", "") for c in reranked_chunks],
        )

    # ── Decision Tree Walker ──────────────────────────────────────────────────

    def _walk_decision_tree(
        self,
        domain_id: str,
        query: str,
        context_chunks: list[dict],
    ) -> list[str]:
        """
        Walk the domain's decision tree (NetworkX DAG) and return the path
        of node labels traversed. Enriches LLM reasoning with structured logic.
        """
        tree_data = self.loader.load_decision_tree(domain_id)
        if not tree_data:
            return []

        try:
            G = nx.DiGraph()
            for node in tree_data.get("nodes", []):
                G.add_node(node["id"], label=node["label"], question=node.get("question", ""))
            for edge in tree_data.get("edges", []):
                G.add_edge(edge["from"], edge["to"], condition=edge.get("condition", ""))

            # Simple traversal: follow the longest path from root
            root = tree_data.get("root", "start")
            if root not in G:
                return []

            path_nodes = []
            current = root
            visited = set()
            while current and current not in visited:
                visited.add(current)
                label = G.nodes[current].get("label", current)
                path_nodes.append(label)
                successors = list(G.successors(current))
                current = successors[0] if successors else None

            return path_nodes

        except Exception as e:
            logger.warning(f"Decision tree walk failed: {e}")
            return []

    # ── Expert LLM Call ───────────────────────────────────────────────────────

    def _call_expert_llm(
        self,
        query: str,
        capsule: CapsuleConfig,
        context_chunks: list[dict],
        decision_path: list[str],
    ) -> str:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        context_text = "\n\n---\n".join(
            f"[Source: {c.get('source', 'Unknown')}]\n{c.get('text', '')}"
            for c in context_chunks
        )

        decision_text = (
            "\n".join(f"  {i+1}. {step}" for i, step in enumerate(decision_path))
            if decision_path else "  No structured decision tree available."
        )

        sources_text = ", ".join(capsule.sources_of_truth) if capsule.sources_of_truth else "domain knowledge base"

        system_prompt = f"""{capsule.expert_persona}

REASONING PROTOCOL — you MUST follow this exact reasoning path:
{decision_text}

CONTEXT (retrieved from verified sources: {sources_text}):
{context_text}

RESPONSE RULES:
1. Answer ONLY based on the provided context. Do NOT hallucinate facts not present in context.
2. Every factual claim MUST reference its source: write [Source: <source_name>] inline.
3. If context is insufficient, say "Insufficient verified information — please consult a certified {capsule.name} expert."
4. Use domain-specific terminology precisely.
5. Structure your response with clear sections if complex.
"""

        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        return response.choices[0].message.content

    # ── Utility ───────────────────────────────────────────────────────────────

    def list_capsules(self) -> list[dict[str, str]]:
        """Return a summary of all loaded capsules."""
        return [
            {
                "domain_id": cfg.domain_id,
                "name": cfg.name,
                "version": cfg.version,
                "description": cfg.description,
            }
            for cfg in self.router._capsules.values()
        ]

    def __repr__(self) -> str:
        caps = list(self.router._capsules.keys()) if self._bootstrapped else "not bootstrapped"
        return f"<ExpertCore capsules={caps}>"
