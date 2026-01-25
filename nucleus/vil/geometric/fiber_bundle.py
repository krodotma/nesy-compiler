"""
VIL Fiber Bundle Geometry
Parallel transport and curvature on fiber bundles.

Based on Learning Tower v1.0 L4: Fiber Bundle Geometry:
- Base manifold (vision embedding space)
- Fiber (additional feature dimensions)
- Connection (parallel transport rule)
- Curvature (deviation from flatness)

Applications:
- Parallel transport of gradients
- Curvature-aware updates
- Geometric deep learning
- Gauge equivariant neural networks

Version: 1.0
Date: 1.0
Date: 2026-01-25
"""

import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ConnectionType(str, Enum):
    """Types of connections on fiber bundles."""

    LEVI_CIVITA = "levi_civita"  # Metric-compatible, torsion-free
    WEYL = "weyl"  # Conformal connection
    GAUGE = "gauge"  # Yang-Mills gauge connection


@dataclass
class FiberBundle:
    """
    Fiber bundle structure.

    Total space E = B × F (locally)
    - Base manifold B (vision embeddings)
    - Fiber F (feature dimensions)
    - Projection π: E → B
    """

    base_dim: int  # Dimension of base manifold
    fiber_dim: int  # Dimension of fiber
    total_dim: int = 0

    # Connection (Christoffel symbols)
    connection: Optional[np.ndarray] = None  # Γ^i_jk

    # Curvature tensor
    curvature: Optional[np.ndarray] = None  # R^i_jkl

    def __post_init__(self):
        self.total_dim = self.base_dim + self.fiber_dim

    def get_split(self, vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split vector into base and fiber components."""
        base = vector[:self.base_dim]
        fiber = vector[self.base_dim:]
        return base, fiber

    def combine(self, base: np.ndarray, fiber: np.ndarray) -> np.ndarray:
        """Combine base and fiber into total vector."""
        return np.concatenate([base, fiber])


@dataclass
class ParallelTransportResult:
    """
    Result of parallel transport.

    Contains:
    - Transported vector
    - Holonomy (rotation due to curvature)
    - Path length
    - Curvature integrated
    """

    transported: np.ndarray
    original: np.ndarray
    holonomy: float  # Total rotation angle
    path_length: float
    curvature_integral: float = 0.0


class FiberBundleGeometry:
    """
    Fiber bundle geometry operations.

    Features:
    1. Parallel transport on bundles
    2. Curvature computation
    3. Holonomy calculation
    4. Connection forms
    """

    def __init__(
        self,
        base_dim: int = 32,
        fiber_dim: int = 32,
        connection_type: ConnectionType = ConnectionType.LEVI_CIVITA,
    ):
        self.base_dim = base_dim
        self.fiber_dim = fiber_dim
        self.connection_type = connection_type

        self.bundle = FiberBundle(base_dim=base_dim, fiber_dim=fiber_dim)

        # Initialize connection (identity for Euclidean base)
        self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Initialize connection coefficients."""
        dim = self.bundle.base_dim + self.bundle.fiber_dim
        # Initialize with zero connection (Euclidean)
        self.bundle.connection = np.zeros((dim, dim, dim))

    def compute_metric(
        self,
        points: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compute Riemannian metric from point cloud.

        Uses empirical Fisher information metric:
        g_ij = (1/N) Σ_k ∂_i log p(x_k) * ∂_j log p(x_k)

        Simplified: g = X^T X (covariance)
        """
        if not points:
            return np.eye(self.bundle.total_dim)

        X = np.array(points)
        cov = np.cov(X.T)
        return cov + np.eye(self.bundle.total_dim) * 1e-8  # Regularize

    def compute_christoffel_symbols(
        self,
        metric: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Christoffel symbols (Levi-Civita connection).

        Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)

        Args:
            metric: Riemannian metric tensor g_ij

        Returns:
            Christoffel symbols Γ^k_ij
        """
        dim = len(metric)
        inverse_metric = np.linalg.inv(metric)
        gamma = np.zeros((dim, dim, dim))

        # Numerical derivatives
        epsilon = 1e-6

        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    # Compute derivatives of metric
                    dg_il = np.zeros(dim)
                    dg_jl = np.zeros(dim)
                    dg_ij = np.zeros(dim)

                    for l in range(dim):
                        # ∂_i g_jl
                        g_plus = metric.copy()
                        g_plus[j, l] += epsilon
                        g_minus = metric.copy()
                        g_minus[j, l] -= epsilon
                        dg_il[l] = (g_plus[i, l] - g_minus[i, l]) / (2 * epsilon)

                        # ∂_j g_il
                        g_plus = metric.copy()
                        g_plus[i, l] += epsilon
                        g_minus = metric.copy()
                        g_minus[i, l] -= epsilon
                        dg_jl[l] = (g_plus[j, l] - g_minus[j, l]) / (2 * epsilon)

                        # ∂_l g_ij
                        g_plus = metric.copy()
                        g_plus[i, j] += epsilon
                        g_minus = metric.copy()
                        g_minus[i, j] -= epsilon
                        dg_ij[l] = (g_plus[i, j] - g_minus[i, j]) / (2 * epsilon)

                    # Levi-Civita formula
                    term1 = dg_il
                    term2 = dg_jl
                    term3 = -dg_ij

                    for l in range(dim):
                        gamma[k, i, j] += (
                            0.5 * inverse_metric[k, l] *
                            (term1[l] + term2[l] + term3[l])
                        )

        self.bundle.connection = gamma
        return gamma

    def parallel_transport(
        self,
        vector: np.ndarray,
        from_point: np.ndarray,
        to_point: np.ndarray,
        path: Optional[List[np.ndarray]] = None,
    ) -> ParallelTransportResult:
        """
        Parallel transport vector between points on fiber bundle.

        Args:
            vector: Vector to transport
            from_point: Starting point on base manifold
            to_point: Ending point on base manifold
            path: Optional path points (default: geodesic)

        Returns:
            ParallelTransportResult
        """
        if path is None:
            # Straight line path (geodesic in Euclidean)
            num_steps = 10
            path = [
                from_point + (to_point - from_point) * (i / num_steps)
                for i in range(num_steps + 1)
            ]

        transported = vector.copy()
        holonomy = 0.0

        # Transport along path
        for i in range(len(path) - 1):
            x_i = path[i]
            x_j = path[i + 1]

            # Update using connection
            dx = x_j - x_i

            # Parallel transport equation: ∇_v V = 0
            # dV^k/dt + Γ^k_ij V^i dx^j/dt = 0

            if self.bundle.connection is not None:
                for k in range(len(transported)):
                    for i in range(len(dx)):
                        for j in range(len(transported)):
                            transported[k] -= (
                                self.bundle.connection[k, i, j] *
                                transported[j] *
                                dx[i]
                            )

            # Accumulate holonomy (rotation)
            if i > 0:
                cos_angle = np.dot(
                    transported / (np.linalg.norm(transported) + 1e-8),
                    vector / (np.linalg.norm(vector) + 1e-8),
                )
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                holonomy += np.arccos(cos_angle)

        # Path length
        path_length = sum(
            np.linalg.norm(path[i+1] - path[i])
            for i in range(len(path) - 1)
        )

        return ParallelTransportResult(
            transported=transported,
            original=vector,
            holonomy=holonomy,
            path_length=path_length,
        )

    def compute_curvature(
        self,
        point: np.ndarray,
    ) -> float:
        """
        Compute scalar curvature at point.

        R = R^i_ij^i (Ricci scalar)

        Simplified: K = det(R)^(1/dim)
        """
        if self.bundle.connection is None:
            return 0.0

        dim = self.bundle.total_dim
        gamma = self.bundle.connection

        # Riemann curvature tensor: R^i_jkl = ∂_k Γ^i_lj - ∂_l Γ^i_kj + Γ^i_mk Γ^m_lj - Γ^i_ml Γ^m_kj

        R = np.zeros((dim, dim, dim, dim))

        epsilon = 1e-6

        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        # Numerical derivatives of connection
                        dgamma_dk = np.zeros(dim)
                        dgamma_dl = np.zeros(dim)

                        for m in range(dim):
                            g_plus = gamma.copy()
                            g_plus[i, m, j] += epsilon
                            g_minus = gamma.copy()
                            g_minus[i, m, j] -= epsilon
                            dgamma_dk[m] = (g_plus[i, m, j] - g_minus[i, m, j]) / (2 * epsilon)

                            g_plus = gamma.copy()
                            g_plus[i, m, j] += epsilon
                            g_minus = gamma.copy()
                            g_minus[i, m, j] -= epsilon
                            dgamma_dl[m] = (g_plus[i, m, j] - g_minus[i, m, j]) / (2 * epsilon)

                        # Riemann formula
                        term1 = dgamma_dl  # ∂_k Γ^i_lj
                        term2 = -dgamma_dk  # -∂_l Γ^i_kj

                        term3 = 0.0
                        term4 = 0.0
                        for m in range(dim):
                            term3 += gamma[i, m, k] * gamma[m, l, j]
                            term4 -= gamma[i, m, l] * gamma[m, k, j]

                        R[i, j, k, l] = term1 + term2 + term3 + term4

        # Ricci scalar: R = g^ij R_ijk^k
        # Simplified: scalar curvature as norm of R
        return float(np.linalg.norm(R))

    def get_stats(self) -> Dict[str, Any]:
        """Get fiber bundle statistics."""
        return {
            "base_dim": self.base_dim,
            "fiber_dim": self.fiber_dim,
            "total_dim": self.bundle.total_dim,
            "connection_type": self.connection_type.value,
            "has_connection": self.bundle.connection is not None,
        }


def create_fiber_bundle_geometry(
    base_dim: int = 32,
    fiber_dim: int = 32,
    connection_type: ConnectionType = ConnectionType.LEVI_CIVITA,
) -> FiberBundleGeometry:
    """Create fiber bundle geometry with default config."""
    return FiberBundleGeometry(
        base_dim=base_dim,
        fiber_dim=fiber_dim,
        connection_type=connection_type,
    )


__all__ = [
    "ConnectionType",
    "FiberBundle",
    "ParallelTransportResult",
    "FiberBundleGeometry",
    "create_fiber_bundle_geometry",
]
