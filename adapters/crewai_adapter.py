"""
crewai_adapter.py
-----------------
CrewAI framework adapter for ExpertCore.

Provides a CrewAI-compatible Tool and Agent wrapper.
"""

from typing import Any, Optional

from loguru import logger

from adapters.base_adapter import BaseAdapter
from core.expert_core import ExpertCore
from core.confidence_gate import ExpertResponse, ClarificationResponse


class CrewAIAdapter(BaseAdapter):
    """
    Integrates ExpertCore into CrewAI as a custom Tool
    that any CrewAI Agent can utilize.
    """

    def __init__(self, expert_core: ExpertCore):
        super().__init__(expert_core)
        logger.info("CrewAIAdapter initialized.")

    def inject(self, agent: Any) -> Any:
        """
        Inject ExpertCore as a tool into a CrewAI Agent.
        """
        tool = self.create_tool()

        if hasattr(agent, "tools"):
            if isinstance(agent.tools, list):
                agent.tools.append(tool)
            logger.success("ExpertCore tool injected into CrewAI agent.")
        else:
            logger.warning("Could not inject tool — agent has no 'tools' attribute.")

        return agent

    def wrap_prompt(self, prompt: str, domain_id: str | None = None) -> str:
        """Wrap a prompt with expert context from ExpertCore."""
        result = self.expert_core.query(prompt, force_domain=domain_id)

        if isinstance(result, ClarificationResponse):
            return f"[SME-PLUG] {result.message}\n\nOriginal: {prompt}"

        if isinstance(result, ExpertResponse):
            return (
                f"[SME-PLUG Expert: {result.domain_name}]\n"
                f"{result.answer}\n\n"
                f"Confidence: {result.confidence_score:.2f}\n"
                f"---\nOriginal: {prompt}"
            )

        return prompt

    def create_tool(self) -> Any:
        """
        Create a CrewAI-compatible Tool wrapping ExpertCore.query().
        """
        try:
            from crewai.tools import tool as crewai_tool
        except ImportError:
            logger.warning("CrewAI not installed. Returning a callable wrapper instead.")
            return self._fallback_tool()

        expert_core = self.expert_core

        @crewai_tool("SME Expert Consultant")
        def sme_expert_tool(query: str) -> str:
            """
            Subject Matter Expert tool. Provides expert-level analysis with
            source citations for Structural Engineering, Cybersecurity, or Legal domains.
            Use this when a question requires deep domain expertise.
            """
            result = expert_core.query(query)

            if isinstance(result, ClarificationResponse):
                return f"Need clarification: {result.message}"

            if isinstance(result, ExpertResponse):
                output = f"🔬 Domain: {result.domain_name}\n"
                output += f"📊 Confidence: {result.confidence_score:.2f}\n"
                output += f"🌳 Reasoning: {' → '.join(result.decision_tree_path)}\n\n"
                output += result.answer
                if result.guardrail_warnings:
                    output += f"\n\n⚠️ Warnings: {'; '.join(result.guardrail_warnings)}"
                return output

            return str(result)

        return sme_expert_tool

    def _fallback_tool(self) -> Any:
        """Fallback for when CrewAI is not installed — returns a simple callable."""
        def sme_tool(query: str) -> str:
            result = self.expert_core.query(query)
            if isinstance(result, ExpertResponse):
                return result.answer
            if isinstance(result, ClarificationResponse):
                return result.message
            return str(result)

        sme_tool.__name__ = "sme_expert"
        sme_tool.__doc__ = "Subject Matter Expert tool for domain-specific queries."
        return sme_tool

    def create_expert_agent(
        self,
        role: str = "Domain Expert",
        goal: str = "Provide expert-level analysis with source citations",
        backstory: str | None = None,
    ) -> Any:
        """
        Create a complete CrewAI Agent pre-loaded with ExpertCore tools.
        """
        try:
            from crewai import Agent
        except ImportError:
            logger.error("CrewAI not installed. Cannot create Agent.")
            return None

        if backstory is None:
            backstory = (
                "You are a domain expert powered by SME-PLUG. "
                "You have access to specialized knowledge bases, structured decision trees, "
                "and source-of-truth citations. Always use your SME Expert tool for domain-specific questions."
            )

        tool = self.create_tool()

        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=[tool],
            verbose=True,
        )
