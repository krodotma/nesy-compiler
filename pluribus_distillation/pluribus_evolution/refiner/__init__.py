"""
pluribus_evolution.refiner

Proposes improvements to the primary pluribus trunk based on observer findings.

Components:
- proposal_generator: Generates refactoring proposals
- manifold_optimizer: Optimizes vector/embedding space
- axiom_evolver: Evolves axiom definitions
"""

from __future__ import annotations

__all__ = [
    "ProposalGenerator",
    "ManifoldOptimizer",
    "AxiomEvolver",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "ProposalGenerator":
        from .proposal_generator import ProposalGenerator
        return ProposalGenerator
    if name == "ManifoldOptimizer":
        from .manifold_optimizer import ManifoldOptimizer
        return ManifoldOptimizer
    if name == "AxiomEvolver":
        from .axiom_evolver import AxiomEvolver
        return AxiomEvolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
