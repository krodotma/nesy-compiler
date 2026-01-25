"""
Theia Program Synthesis — CGP, EGGP, Metagrammars.

From graph_evolutionary_programming_distillation.md:

P1 (Critical):
    CGP for AST Evolution — Mature, neutral drift proven essential
    
P2 (Important):
    EGGP for Refactoring — Graph programs = refactoring rules
    MAGE for Multi-Lang — Type coherence maps to typed ASTs
    
Key insight: Neutral drift allows inactive gene mutations to accumulate,
potentially activating beneficial variations later.
"""

from theia.synthesis.cgp import (
    CGPGenome,
    CGPNode,
    FUNCTION_SET,
    cgp_evolve,
    symbolic_regression_fitness,
)

from theia.synthesis.eggp import (
    GraphNode,
    GraphEdge,
    GraphProgram,
    GraphGrammar,
    EGGPEvolver,
    Metagrammar,
)

from theia.synthesis.metagrammar import (
    TypeMapping,
    ASTPattern,
    TransformRule,
    MetagrammarRegistry,
    create_default_registry,
)

__all__ = [
    # CGP
    "CGPGenome",
    "CGPNode",
    "FUNCTION_SET",
    "cgp_evolve",
    "symbolic_regression_fitness",
    # EGGP
    "GraphNode",
    "GraphEdge",
    "GraphProgram",
    "GraphGrammar",
    "EGGPEvolver",
    "Metagrammar",
    # Metagrammar
    "TypeMapping",
    "ASTPattern",
    "TransformRule",
    "MetagrammarRegistry",
    "create_default_registry",
]


