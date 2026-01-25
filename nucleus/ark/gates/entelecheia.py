#!/usr/bin/env python3
"""
entelecheia.py - EntelecheiaGate: Purpose enforcement

LTL: □ (commit → ◇ witnessed) ∧ □ (cosmetic → rejected)

Every change must demonstrably advance a goal.
Purposeless churn is rejected.
"""

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger("ARK.Entelecheia")


@dataclass
class EntelecheiaContext:
    """Context for Entelecheia gate evaluation."""
    purpose: str = ""
    spec_ref: Optional[str] = None
    is_cosmetic: bool = False
    liveness_gain: float = 0.0
    diff: str = ""  # Git diff for spec validation


class EntelecheiaGate:
    """
    Axiom 2: Entelecheia (Purpose)
    
    Every code change must serve a clear purpose.
    Cosmetic refactoring without liveness gain is rejected.
    Now with LTL spec validation.
    """
    
    # Cosmetic-only indicators
    COSMETIC_PATTERNS = [
        "formatting",
        "whitespace",
        "style",
        "lint",
        "typo",
    ]
    
    def __init__(self):
        self._ltl_validator = None
    
    @property
    def ltl_validator(self):
        """Lazy-load LTL validator."""
        if self._ltl_validator is None:
            try:
                from nucleus.ark.specs.ltl_validator import LTLValidator
                self._ltl_validator = LTLValidator()
            except ImportError:
                logger.warning("LTL validator not available")
        return self._ltl_validator
    
    def name(self) -> str:
        return "Entelecheia"
    
    def check(self, context: EntelecheiaContext) -> bool:
        """
        Returns True if the commit serves a purpose (passes Entelecheia gate).
        
        Rules:
        1. Must have a non-empty purpose or spec_ref
        2. Purely cosmetic changes are allowed ONLY with explicit purpose
        3. Changes with liveness_gain > 0 always pass
        4. If spec_ref is provided, diff must satisfy the spec
        """
        # Check for explicit purpose
        has_purpose = bool(context.purpose.strip())
        has_spec = bool(context.spec_ref)
        has_liveness = context.liveness_gain > 0
        
        # If there's liveness gain, always pass
        if has_liveness:
            return True
        
        # If cosmetic without explicit purpose, reject
        if context.is_cosmetic or self._is_cosmetic_purpose(context.purpose):
            if not has_spec:
                return False
        
        # If spec_ref provided, validate against LTL spec
        if has_spec and self.ltl_validator and context.diff:
            passed, reason = self.ltl_validator.validate(
                context.spec_ref, context.diff, context.purpose
            )
            if not passed:
                logger.warning(f"LTL validation failed: {reason}")
                return False
        
        # Must have some form of purpose
        return has_purpose or has_spec
    
    def _is_cosmetic_purpose(self, purpose: str) -> bool:
        """Check if purpose describes cosmetic-only changes."""
        purpose_lower = purpose.lower()
        for pattern in self.COSMETIC_PATTERNS:
            if pattern in purpose_lower and len(purpose_lower) < 50:
                return True
        return False
    
    def ltl_formula(self) -> str:
        """Return the LTL formula for this gate."""
        return "□ (commit → purpose_achieved)"
