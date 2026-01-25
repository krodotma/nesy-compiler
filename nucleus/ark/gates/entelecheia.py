#!/usr/bin/env python3
"""
entelecheia.py - EntelecheiaGate: Purpose enforcement

LTL: □ (commit → ◇ witnessed) ∧ □ (cosmetic → rejected)

Every change must demonstrably advance a goal.
Purposeless churn is rejected.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EntelecheiaContext:
    """Context for Entelecheia gate evaluation."""
    purpose: str = ""
    spec_ref: Optional[str] = None
    is_cosmetic: bool = False
    liveness_gain: float = 0.0


class EntelecheiaGate:
    """
    Axiom 2: Entelecheia (Purpose)
    
    Every code change must serve a clear purpose.
    Cosmetic refactoring without liveness gain is rejected.
    """
    
    # Cosmetic-only indicators
    COSMETIC_PATTERNS = [
        "formatting",
        "whitespace",
        "style",
        "lint",
        "typo",
    ]
    
    def name(self) -> str:
        return "Entelecheia"
    
    def check(self, context: EntelecheiaContext) -> bool:
        """
        Returns True if the commit serves a purpose (passes Entelecheia gate).
        
        Rules:
        1. Must have a non-empty purpose or spec_ref
        2. Purely cosmetic changes are allowed ONLY with explicit purpose
        3. Changes with liveness_gain > 0 always pass
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
