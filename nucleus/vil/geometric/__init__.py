"""
VIL Geometric Module
Geometric learning for vision-metalearning integration.

Version: 1.0
Date: 2026-01-25
"""

from .metalearning import (
    ManifoldType,
    GeometricState,
    GeometricUpdate,
    SphericalManifold,
    HyperbolicManifold,
    GeometricMetalearning,
    create_geometric_metalearning,
)

__all__ = [
    "ManifoldType",
    "GeometricState",
    "GeometricUpdate",
    "SphericalManifold",
    "HyperbolicManifold",
    "GeometricMetalearning",
    "create_geometric_metalearning",
]
