#!/usr/bin/env python3
"""
inertia.py - InertiaGate: Stability preservation

LTL: □ (mutation(node) ∧ high_inertia(node) → formal_proof(mutation))

High-centrality nodes in the dependency DAG resist change
unless formally verified or witnessed.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from abc import ABC, abstractmethod


@dataclass
class InertiaContext:
    """Context for Inertia gate evaluation."""
    files: List[str] = field(default_factory=list)
    high_inertia_threshold: float = 0.8
    inertia_scores: dict = field(default_factory=dict)
    has_formal_proof: bool = False
    has_witness: bool = False


class InertiaGate:
    """
    Axiom 1: Inertia (Stability)
    
    High-inertia components require formal proof or witness for changes.
    Uses PageRank-like centrality to identify critical components.
    """
    
    # Known high-inertia patterns (nucleus core)
    HIGH_INERTIA_PATTERNS = [
        "world_router.py",
        "agent_bus.py",
        "ohm.py",
        "dna/",
        "ribosome/",
        "__init__.py",
    ]
    
    def name(self) -> str:
        return "Inertia"
    
    def check(self, context: InertiaContext) -> bool:
        """
        Returns True if the commit is allowed (passes Inertia gate).
        
        Rules:
        1. Changes to high-inertia files require witness or formal proof
        2. New files are allowed (low inertia by default)
        3. Low-inertia files always pass
        """
        high_inertia_files = []
        
        for file in context.files:
            if self._is_high_inertia(file, context):
                high_inertia_files.append(file)
        
        if not high_inertia_files:
            return True  # No high-inertia files affected
        
        # High-inertia files require proof or witness
        if context.has_formal_proof or context.has_witness:
            return True
        
        # Reject: high-inertia without verification
        return False
    
    def _is_high_inertia(self, filepath: str, context: InertiaContext) -> bool:
        """Check if a file is high-inertia based on patterns and scores."""
        # Check explicit inertia scores
        if filepath in context.inertia_scores:
            return context.inertia_scores[filepath] > context.high_inertia_threshold
        
        # Check known high-inertia patterns
        for pattern in self.HIGH_INERTIA_PATTERNS:
            if pattern in filepath:
                return True
        
        return False
    
    def ltl_formula(self) -> str:
        """Return the LTL formula for this gate."""
        return "□ (mutate ∧ I > T → verified)"
