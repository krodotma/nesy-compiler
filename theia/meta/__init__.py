"""
Theia Meta Layer (L5) — Reflexive Domain Ω.

Ω ≅ [Ω → Ω]_⊥  (Scott domain / reflexive object)
Internal topos logic: self-referential inference

Sheaf cohomology: H*(Stack, Ω)
Global consistency of local computation.

Fixed-point semantics: Self-referential statements have well-defined meaning.
The system can:
    - Model its own uncertainty
    - Predict its own failures
    - Propose modifications to itself
"""

from theia.meta.omega import (
    OmegaState,
    OmegaDomain,
    MetacognitiveState,
    Metacognition,
    OmegaModifier,
)

from theia.meta.sheaf import (
    Stalk,
    LocalSection,
    ConsistencyChecker,
    compute_cech_cohomology,
)

__all__ = [
    # Omega
    "OmegaState",
    "OmegaDomain",
    "MetacognitiveState",
    "Metacognition",
    "OmegaModifier",
    # Sheaf
    "Stalk",
    "LocalSection",
    "ConsistencyChecker",
    "compute_cech_cohomology",
]
