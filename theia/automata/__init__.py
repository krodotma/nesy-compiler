"""
Theia Automata Layer (L4) — Dual Neurosymbolic Automata (DNA).

A = (Q, Σ, δ, q₀, F, ρ) where ρ: Q → Mod(E)
ρ = reentrant modification functor

DUAL structure:
    • Automaton coalgebra: Q → F(Q) [execution]
    • Algebra: F*(E) → E [constraint backprop]
    
Coalgebraic reentry: Higher domains can modify lower energy landscapes.
"""

from theia.automata.coalgebra import (
    StateType,
    State,
    Modification,
    Coalgebra,
    DNAAutomaton,
    create_simple_dna,
    create_browser_dna,
)

from theia.automata.reentry import (
    EnergyModification,
    ModificationFunctor,
    ReentryController,
    SelfTeacher,
)

__all__ = [
    # Coalgebra
    "StateType",
    "State",
    "Modification",
    "Coalgebra",
    "DNAAutomaton",
    "create_simple_dna",
    "create_browser_dna",
    # Reentry
    "EnergyModification",
    "ModificationFunctor",
    "ReentryController",
    "SelfTeacher",
]


