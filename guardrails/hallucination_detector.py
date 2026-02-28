"""
hallucination_detector.py
-------------------------
Detects hallucination in LLM responses by checking whether claims
are semantically grounded in the retrieved context chunks.

Uses sentence-level semantic overlap scoring to flag ungrounded claims.
"""

import re
from loguru import logger


class HallucinationDetector:
    """
    Checks whether an LLM answer's claims are grounded in the retrieved context.
    Returns a list of warning strings for ungrounded or suspicious claims.
    """

    DEFAULT_THRESHOLD = 0.35  # min overlap ratio to consider a sentence grounded

    def __init__(self, threshold: float | None = None):
        self.threshold = threshold or self.DEFAULT_THRESHOLD
        logger.debug(f"HallucinationDetector initialized (threshold={self.threshold}).")

    def check(
        self,
        answer: str,
        context_chunks: list[dict],
    ) -> list[str]:
        """
        Check an answer for potential hallucinations by comparing each
        answer sentence against the context.

        Returns list of warning strings (empty = no hallucinations detected).
        """
        if not answer or not context_chunks:
            return []

        # Combine all context into one text block
        context_text = " ".join(c.get("text", "") for c in context_chunks).lower()
        context_words = set(context_text.split())

        # Split answer into sentences
        sentences = self._split_sentences(answer)
        warnings = []

        for sent in sentences:
            # Skip very short sentences, metadata lines, or citation lines
            if len(sent.split()) < 5:
                continue
            if sent.startswith("[Source:") or sent.startswith("⚠️"):
                continue

            overlap = self._word_overlap(sent, context_words)

            if overlap < self.threshold:
                warnings.append(
                    f"Potential hallucination (overlap={overlap:.2f}): "
                    f"\"{sent[:100]}...\""
                )

        if warnings:
            logger.warning(f"Hallucination detector flagged {len(warnings)} claims.")
        else:
            logger.debug("Hallucination check passed — all claims grounded.")

        return warnings

    def _word_overlap(self, sentence: str, context_words: set[str]) -> float:
        """Calculate word-level overlap ratio between a sentence and context."""
        sent_words = set(sentence.lower().split())
        # Remove common stop words for more meaningful comparison
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "and", "but", "or",
            "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
            "every", "all", "any", "few", "more", "most", "other", "some", "such",
            "than", "too", "very", "just", "also", "this", "that", "these", "those",
            "it", "its", "i", "we", "you", "he", "she", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "our", "their",
        }
        sent_meaningful = sent_words - stop_words
        if not sent_meaningful:
            return 1.0  # trivial sentence

        overlap = sent_meaningful & context_words
        return len(overlap) / len(sent_meaningful)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        raw = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in raw if s.strip()]
