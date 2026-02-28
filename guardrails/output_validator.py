"""
output_validator.py
-------------------
Domain-specific output validation using rules defined in each capsule.
Checks for:
  - Required patterns (units, formats, standards references)
  - Forbidden phrases (overconfident or dangerous language)
  - Domain-specific format rules
"""

import re
from loguru import logger
from core.capsule_loader import CapsuleConfig


class OutputValidator:
    """
    Validates expert responses against domain-specific rules defined
    in the capsule's guardrails configuration.
    """

    def validate(
        self,
        answer: str,
        capsule: CapsuleConfig,
    ) -> list[str]:
        """
        Run all domain validation rules against the answer.

        Returns list of warning strings (empty = all validations passed).
        """
        if not answer:
            return ["Empty answer received."]

        warnings = []

        # ── Check forbidden phrases ───────────────────────────────────────
        for phrase in capsule.guardrails.forbidden_phrases:
            if phrase.lower() in answer.lower():
                warnings.append(
                    f"Forbidden phrase detected: \"{phrase}\" — "
                    "Expert responses should not use uncertain/dangerous language."
                )

        # ── Check domain validation rules ─────────────────────────────────
        for rule in capsule.guardrails.domain_validation_rules:
            rule_name = rule.get("rule", "unnamed")
            pattern = rule.get("pattern", "")
            message = rule.get("message", f"Validation rule '{rule_name}' check.")

            if pattern:
                try:
                    if not re.search(pattern, answer, re.IGNORECASE):
                        # Pattern not found when it should be present
                        warnings.append(f"[{rule_name}] {message}")
                except re.error as e:
                    logger.warning(f"Invalid regex in rule '{rule_name}': {e}")

        # ── Check citation requirement ────────────────────────────────────
        if capsule.guardrails.require_citations:
            if "[Source:" not in answer and "Sources Referenced" not in answer:
                warnings.append(
                    "No citations found in response. "
                    "Expert answers must reference their sources."
                )

        if warnings:
            logger.warning(f"OutputValidator: {len(warnings)} issues found for domain '{capsule.domain_id}'.")
        else:
            logger.debug(f"OutputValidator: All checks passed for domain '{capsule.domain_id}'.")

        return warnings
