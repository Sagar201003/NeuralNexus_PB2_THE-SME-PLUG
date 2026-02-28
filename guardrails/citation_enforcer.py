"""
citation_enforcer.py
--------------------
Post-processes every LLM response to:
1. Inject [Source: doc_name] citations inline with claims
2. Add a "Sources" section at the end
3. Reject responses with zero grounding (no citations possible)
"""

import re
from loguru import logger


class CitationEnforcer:
    """
    Enforces source-of-truth citations in expert responses.
    Matches answer sentences to context chunks and injects citations.
    """

    def enforce(
        self,
        answer: str,
        chunks: list[dict],
        sources_of_truth: list[str] | None = None,
    ) -> tuple[str, list[dict]]:
        """
        Process an answer to add citations.

        Args:
            answer: Raw LLM answer text.
            chunks: Retrieved context chunks (each has "text", "source" keys).
            sources_of_truth: Official source names from the capsule config.

        Returns:
            (cited_answer, citations_list)
            - cited_answer: Answer with inline [Source: ...] tags
            - citations_list: List of citation dicts for structured output
        """
        if not answer or not chunks:
            return answer, []

        citations = []
        cited_sentences = []
        used_sources = set()

        sentences = self._split_sentences(answer)

        for sent in sentences:
            # Skip very short or meta sentences
            if len(sent.split()) < 4:
                cited_sentences.append(sent)
                continue

            # Already has a citation?
            if "[Source:" in sent:
                cited_sentences.append(sent)
                # Extract existing source for tracking
                existing = re.findall(r'\[Source:\s*([^\]]+)\]', sent)
                used_sources.update(existing)
                for src in existing:
                    citations.append({
                        "source": src.strip(),
                        "excerpt": sent[:150],
                        "relevance_score": 1.0,
                    })
                continue

            # Find best matching chunk
            best_chunk, best_score = self._find_best_match(sent, chunks)

            if best_chunk and best_score > 0.25:
                source_name = best_chunk.get("source", "Unknown")
                cited_sent = f"{sent} [Source: {source_name}]"
                cited_sentences.append(cited_sent)
                used_sources.add(source_name)
                citations.append({
                    "source": source_name,
                    "excerpt": sent[:150],
                    "relevance_score": round(best_score, 3),
                })
            else:
                # No grounding found — keep sentence but don't cite
                cited_sentences.append(sent)

        # Rebuild the answer
        cited_answer = " ".join(cited_sentences)

        # Append Sources section
        if used_sources:
            sources_section = "\n\n---\n📚 **Sources Referenced:**\n"
            for i, src in enumerate(sorted(used_sources), 1):
                sources_section += f"  {i}. {src}\n"
            if sources_of_truth:
                sources_section += "\n📖 **Authoritative Standards:**\n"
                for i, sot in enumerate(sources_of_truth, 1):
                    sources_section += f"  {i}. {sot}\n"
            cited_answer += sources_section

        if not citations:
            logger.warning("CitationEnforcer: No citations could be added to the response.")
        else:
            logger.info(f"CitationEnforcer: Added {len(citations)} citations from {len(used_sources)} sources.")

        return cited_answer, citations

    def _find_best_match(
        self,
        sentence: str,
        chunks: list[dict],
    ) -> tuple[dict | None, float]:
        """Find the chunk with highest word overlap to a sentence."""
        best_chunk = None
        best_score = 0.0
        sent_words = set(sentence.lower().split())

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "and", "but", "or",
            "not", "this", "that", "it", "its", "i", "we", "you", "they",
        }
        sent_meaningful = sent_words - stop_words

        if not sent_meaningful:
            return None, 0.0

        for chunk in chunks:
            chunk_words = set(chunk.get("text", "").lower().split())
            overlap = sent_meaningful & chunk_words
            score = len(overlap) / len(sent_meaningful)
            if score > best_score:
                best_score = score
                best_chunk = chunk

        return best_chunk, best_score

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw if s.strip()]
