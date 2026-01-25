"""
Theia Geometric Substrate (L1) — Spherical-Hyperbolic Duality.

The S^n ⊣ H^n adjunction provides:
    - S^n: Local angular similarity (what transformers do)
    - H^n: Global hierarchical structure (tree-like)
    - Möbius group: Acts on both (conformal equivalence)
"""

from theia.geometric.spherical import (
    project_sphere,
    geodesic_distance,
    cosine_similarity,
    spherical_mean,
    random_sphere,
    slerp,
)

from theia.geometric.hyperbolic import (
    project_poincare,
    mobius_add,
    hyperbolic_distance,
    exp_map,
    log_map,
    hyperbolic_mean,
)

__all__ = [
    # Spherical
    "project_sphere",
    "geodesic_distance",
    "cosine_similarity",
    "spherical_mean",
    "random_sphere",
    "slerp",
    # Hyperbolic
    "project_poincare",
    "mobius_add",
    "hyperbolic_distance",
    "exp_map",
    "log_map",
    "hyperbolic_mean",
]

