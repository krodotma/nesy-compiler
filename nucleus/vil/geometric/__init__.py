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
from .attractor import (
    BasinType,
    AttractorBasin,
    ConvergenceResult,
    AttractorDynamics,
    create_attractor_dynamics,
)
from .fiber_bundle import (
    ConnectionType,
    FiberBundle,
    ParallelTransportResult,
    FiberBundleGeometry,
    create_fiber_bundle_geometry,
)

__all__ = [
    # Metalearning
    "ManifoldType",
    "GeometricState",
    "GeometricUpdate",
    "SphericalManifold",
    "HyperbolicManifold",
    "GeometricMetalearning",
    "create_geometric_metalearning",

    # Attractor
    "BasinType",
    "AttractorBasin",
    "ConvergenceResult",
    "AttractorDynamics",
    "create_attractor_dynamics",

    # Fiber Bundle
    "ConnectionType",
    "FiberBundle",
    "ParallelTransportResult",
    "FiberBundleGeometry",
    "create_fiber_bundle_geometry",
]
