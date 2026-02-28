"""
hyde_engine.py
--------------
HyDE: Hypothetical Document Embeddings

Instead of embedding the raw query, HyDE asks the LLM to generate a
"hypothetical expert answer" first, then embeds THAT for retrieval.
This dramatically improves recall for technical/domain queries because
the hypothetical answer uses domain vocabulary that better matches
the stored document chunks.

Pipeline:
  Query → LLM generates hypothetical answer → Embed hypothetical → Retrieve
"""

import os
from typing import Optional

from loguru import logger
from groq import Groq

from core.capsule_loader import CapsuleConfig


class HyDEEngine:
    """
    Generates hypothetical expert documents for embedding-based retrieval.
    Uses the capsule's expert persona to produce domain-specific hypotheticals.
    """

    def __init__(self):
        self._llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
        logger.debug("HyDEEngine initialized.")

    def expand(self, query: str, capsule: CapsuleConfig) -> str:
        """
        Given a user query and a capsule config, generate a hypothetical
        expert document/answer that would contain the relevant information.

        Returns the hypothetical text (used for embedding-based retrieval).
        """
        system_prompt = (
            f"{capsule.expert_persona}\n\n"
            "You are writing a reference document that would perfectly answer the following question. "
            "Write a detailed, technical, factual paragraph (150-250 words) as if it were an excerpt "
            "from an authoritative reference book or specification document. "
            "Use precise domain terminology. Include specific values, standards, and procedures where applicable. "
            "Do NOT say 'I', do NOT ask questions, do NOT refuse — just write the reference text."
        )

        try:
            response = self._llm.chat.completions.create(
                model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Write a reference document excerpt for: {query}"},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            hyde_text = response.choices[0].message.content.strip()
            logger.info(f"[HyDE] Generated hypothetical document ({len(hyde_text)} chars)")
            return hyde_text

        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}. Falling back to original query.")
            return query

    def expand_multi(self, query: str, capsule: CapsuleConfig, n: int = 3) -> list[str]:
        """
        Generate multiple hypothetical documents for the same query.
        More hypotheticals = better recall coverage (at cost of more embeddings).
        """
        hypotheticals = []
        for i in range(n):
            system_prompt = (
                f"{capsule.expert_persona}\n\n"
                f"You are writing reference document #{i+1} of {n} that would answer the question below. "
                "Each document should approach the topic from a slightly different angle or level of detail. "
                "Write a detailed technical paragraph (150-250 words) as if from an authoritative reference. "
                "Use precise domain terminology. Do NOT say 'I', do NOT ask questions."
            )

            try:
                response = self._llm.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Write reference excerpt #{i+1} for: {query}"},
                    ],
                    temperature=0.5 + (i * 0.15),
                    max_tokens=400,
                )
                hypotheticals.append(response.choices[0].message.content.strip())
            except Exception as e:
                logger.warning(f"HyDE multi-generation #{i+1} failed: {e}")

        logger.info(f"[HyDE] Generated {len(hypotheticals)} hypothetical documents")
        return hypotheticals if hypotheticals else [query]
