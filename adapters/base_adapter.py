"""
base_adapter.py
---------------
Abstract base adapter defining the interface for framework integrations.
All framework-specific adapters inherit from this.
"""

from abc import ABC, abstractmethod
from typing import Any

from core.expert_core import ExpertCore


class BaseAdapter(ABC):
    """
    Universal adapter interface for integrating ExpertCore
    into any AI agent framework.
    """

    def __init__(self, expert_core: ExpertCore):
        self.expert_core = expert_core

    @abstractmethod
    def inject(self, agent: Any) -> Any:
        """
        Inject ExpertCore into the given agent.
        Returns the enhanced agent.
        """
        ...

    @abstractmethod
    def wrap_prompt(self, prompt: str, domain_id: str | None = None) -> str:
        """
        Wrap/enhance a prompt with expert context from ExpertCore.
        Returns the enriched prompt.
        """
        ...

    @abstractmethod
    def create_tool(self) -> Any:
        """
        Create a framework-native tool that wraps ExpertCore.query().
        """
        ...
