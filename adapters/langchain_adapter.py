"""
langchain_adapter.py
--------------------
LangChain framework adapter for ExpertCore.

Provides:
  - Custom LangChain Tool wrapping ExpertCore.query()
  - Custom Retriever using advanced RAG pipeline
  - Callback handler for injecting expert context
"""

from typing import Any, Optional

from loguru import logger

from adapters.base_adapter import BaseAdapter
from core.expert_core import ExpertCore
from core.confidence_gate import ExpertResponse, ClarificationResponse


class LangChainAdapter(BaseAdapter):
    """
    Integrates ExpertCore into LangChain agents as a custom Tool
    and/or custom Retriever.
    """

    def __init__(self, expert_core: ExpertCore):
        super().__init__(expert_core)
        logger.info("LangChainAdapter initialized.")

    def inject(self, agent: Any) -> Any:
        """
        Inject ExpertCore as a tool into a LangChain agent.
        Works with AgentExecutor, create_openai_tools_agent, etc.
        """
        tool = self.create_tool()

        # If agent has a .tools attribute, append our tool
        if hasattr(agent, "tools"):
            if isinstance(agent.tools, list):
                agent.tools.append(tool)
            logger.success("ExpertCore tool injected into LangChain agent.")
        else:
            logger.warning("Could not inject tool — agent has no 'tools' attribute. Use create_tool() manually.")

        return agent

    def wrap_prompt(self, prompt: str, domain_id: str | None = None) -> str:
        """
        Enhance a prompt by querying ExpertCore and prepending expert context.
        Useful for simple chain-based workflows (not agent-based).
        """
        result = self.expert_core.query(prompt, force_domain=domain_id)

        if isinstance(result, ClarificationResponse):
            return f"[ExpertCore] {result.message}\n\nOriginal query: {prompt}"

        if isinstance(result, ExpertResponse):
            expert_context = (
                f"[ExpertCore — {result.domain_name}]\n"
                f"Expert Analysis:\n{result.answer}\n\n"
                f"Decision Path: {' → '.join(result.decision_tree_path)}\n"
                f"Confidence: {result.confidence_score:.2f}\n"
                f"---\n\n"
                f"Original Query: {prompt}"
            )
            return expert_context

        return prompt

    def create_tool(self) -> Any:
        """
        Create a LangChain-compatible Tool wrapping ExpertCore.
        """
        try:
            from langchain_core.tools import Tool
        except ImportError:
            from langchain.tools import Tool

        def _run(query: str) -> str:
            result = self.expert_core.query(query)
            if isinstance(result, ClarificationResponse):
                return f"Need clarification: {result.message}"
            if isinstance(result, ExpertResponse):
                output = f"🔬 Domain: {result.domain_name}\n"
                output += f"📊 Confidence: {result.confidence_score:.2f} (Layer {result.detection_layer})\n"
                output += f"🌳 Decision Path: {' → '.join(result.decision_tree_path)}\n\n"
                output += result.answer
                if result.guardrail_warnings:
                    output += f"\n\n⚠️ Warnings: {'; '.join(result.guardrail_warnings)}"
                return output
            return str(result)

        return Tool(
            name="sme_expert",
            description=(
                "Subject Matter Expert tool. Use this when a question requires deep domain expertise "
                "in Structural Engineering, Cybersecurity, or Legal/Contract analysis. "
                "Provides expert answers with source citations and structured reasoning."
            ),
            func=_run,
        )

    def create_retriever(self) -> Any:
        """
        Create a LangChain-compatible BaseRetriever that uses ExpertCore's
        advanced RAG pipeline (HyDE + Hybrid + Reranker).
        """
        try:
            from langchain_core.retrievers import BaseRetriever
            from langchain_core.documents import Document
            from langchain_core.callbacks import CallbackManagerForRetrieverRun
        except ImportError:
            logger.error("langchain-core not installed. Cannot create retriever.")
            return None

        expert_core = self.expert_core

        class ExpertCoreRetriever(BaseRetriever):
            """Custom retriever backed by ExpertCore's advanced RAG."""

            class Config:
                arbitrary_types_allowed = True

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun | None = None,
            ) -> list[Document]:
                # Use ExpertCore's retriever directly
                result = expert_core.query(query)
                if isinstance(result, ExpertResponse) and result.raw_context_chunks:
                    return [
                        Document(page_content=chunk, metadata={"source": "ExpertCore"})
                        for chunk in result.raw_context_chunks
                    ]
                return []

        return ExpertCoreRetriever()
