"""
Theia Crystallization Layer (L3) — Birkhoff Polytope Dynamics.

Sinkhorn operator: S_β = D_r ∘ D_c [row/col normalization]
Fixed point: S_β^∞(M) ∈ B_n
β → ∞: Geometric quantization to vertices (permutations)

Collapse modes by face lattice:
    - codim 0 (interior): full superposition
    - codim k (k-face): partial crystallization
    - codim max (vertex): deterministic permutation

Birkhoff-von Neumann: M = Σᵢ λᵢ Pᵢ (convex decomposition)
"""

from theia.crystallize.sinkhorn import (
    sinkhorn,
    sinkhorn_log_stable,
    hungarian_from_sinkhorn,
    distance_to_birkhoff,
    distance_to_vertex,
    crystallization_pressure,
)

from theia.crystallize.polytope import (
    Face,
    BirkhoffPolytope,
    FaceLattice,
)

__all__ = [
    # Sinkhorn
    "sinkhorn",
    "sinkhorn_log_stable",
    "hungarian_from_sinkhorn",
    "distance_to_birkhoff",
    "distance_to_vertex",
    "crystallization_pressure",
    # Polytope
    "Face",
    "BirkhoffPolytope",
    "FaceLattice",
]


