#!/usr/bin/env python3
"""
homeostasis.py - HomeostasisGate: System stability enforcement

LTL: □ (H* > T_max → ○ stabilize) ∧ □ (stable → ¬grow)

When system entropy exceeds threshold, growth halts
and stabilization begins.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class HomeostasisContext:
    """Context for Homeostasis gate evaluation."""
    entropy: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.7
    is_stabilization_commit: bool = False


class HomeostasisGate:
    """
    Axiom 3: Homeostasis (Correction)
    
    Monitors the 8-dimensional Entropy Vector (H*).
    When drift exceeds threshold, triggers stabilization.
    """
    
    def name(self) -> str:
        return "Homeostasis"
    
    def check(self, context: HomeostasisContext) -> bool:
        """
        Returns True if homeostasis is maintained (passes gate).
        
        Rules:
        1. If H* < threshold, allow all commits
        2. If H* > threshold, only stabilization commits allowed
        3. Stabilization = reducing entropy, fixing bugs, cleanup
        """
        h_total = self._calculate_total_entropy(context.entropy)
        
        # System is stable, allow normal operation
        if h_total < context.threshold:
            return True
        
        # System is unstable - only allow stabilization
        if context.is_stabilization_commit:
            return True
        
        # Reject growth during instability
        return False
    
    def _calculate_total_entropy(self, entropy: Dict[str, float]) -> float:
        """Calculate total entropy as weighted average of H* vector."""
        if not entropy:
            return 0.5  # Default to mid-range
        
        # Weights for each entropy dimension
        weights = {
            "h_struct": 1.0,   # Structural complexity
            "h_doc": 0.5,      # Documentation gaps
            "h_type": 0.8,     # Type safety
            "h_test": 0.9,     # Test coverage
            "h_deps": 0.7,     # Dependency sprawl
            "h_churn": 1.0,    # Code churn
            "h_debt": 0.9,     # Technical debt
            "h_align": 0.8,    # Spec alignment
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for key, value in entropy.items():
            weight = weights.get(key, 1.0)
            weighted_sum += value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def entropy_status(self, entropy: Dict[str, float]) -> Dict[str, str]:
        """Get human-readable status for each entropy dimension."""
        status = {}
        for key, value in entropy.items():
            if value < 0.3:
                status[key] = "✅ Low"
            elif value < 0.6:
                status[key] = "🟡 Medium"
            else:
                status[key] = "🔴 High"
        return status
    
    def ltl_formula(self) -> str:
        """Return the LTL formula for this gate."""
        return "□ (unstable → ○ stabilize)"
