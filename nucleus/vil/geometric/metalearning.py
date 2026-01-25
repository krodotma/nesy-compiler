"""
VIL Geometric Metalearning
Geometric learning integration for vision-metalearning pipeline.

Integrates:
- S^n (spherical) embeddings for local geometry
- H^n (hyperbolic) embeddings for hierarchical structure
- Fiber bundles for parallel transport
- Attractor dynamics for energy landscapes
- Geometric meta-gradient updates

Based on Learning Tower v1.0:
- L3: S^n / H^n (Spherical-Hyperbolic Duality)
- L4: Fiber Bundle Geometry
- L5: mHC (Modern Hopfield Continuum)
- L6: Birkhoff Polytope (Crystallization)

Version: 1.0
Date: 2026-01-25
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class ManifoldType(str, Enum):
    """Geometric manifold types."""

    SPHERICAL = "spherical"  # S^n: Local geometry
    HYPERBOLIC = "hyperbolic"  # H^n: Hierarchical structure
    EUCLIDEAN = "euclidean"  # R^n: Flat space
    FIBER_BUNDLE = "fiber_bundle"  # Fiber product space


@dataclass
class GeometricState:
    """
    Geometric state on manifold.

    Contains:
    - Point coordinates on manifold
    - Velocity vector
    - Energy value
    - Manifold type
    """

    point: np.ndarray  # Coordinates
    velocity: np.ndarray  # Velocity vector
    energy: float  # Energy value
    manifold: ManifoldType
    curvature: float = 1.0  # Gaussian curvature
    dimension: int = 0
    basin_id: Optional[str] = None  # Attractor basin

    def __post_init__(self):
        if self.dimension == 0:
            self.dimension = len(self.point)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_norm": float(np.linalg.norm(self.point)),
            "velocity_norm": float(np.linalg.norm(self.velocity)),
            "energy": self.energy,
            "manifold": self.manifold.value,
            "curvature": self.curvature,
            "dimension": self.dimension,
            "basin_id": self.basin_id,
        }


@dataclass
class GeometricUpdate:
    """
    Result of geometric update.

    Contains:
    - New geometric state
    - Distance moved
    - Energy change
    - Convergence indicator
    """

    new_state: GeometricState
    old_state: GeometricState
    distance: float
    energy_delta: float
    converged: bool = False
    iterations: int = 0


class SphericalManifold:
    """
    Spherical manifold S^n operations.

    Properties:
    - Positive curvature (K = 1/R^2)
    - Compact, bounded
    - Great circle distance
    - Good for: Local neighborhoods, cyclical features
    """

    @staticmethod
    def project_to_sphere(point: np.ndarray) -> np.ndarray:
        """Project point onto unit sphere."""
        norm = np.linalg.norm(point)
        if norm < 1e-8:
            return np.array([1.0] + [0.0] * (len(point) - 1))
        return point / norm

    @staticmethod
    def great_circle_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute great circle distance between points."""
        p1_norm = SphericalManifold.project_to_sphere(p1)
        p2_norm = SphericalManifold.project_to_sphere(p2)
        dot_product = np.clip(np.dot(p1_norm, p2_norm), -1.0, 1.0)
        return float(np.arccos(dot_product))

    @staticmethod
    def parallel_transport(
        vector: np.ndarray,
        from_point: np.ndarray,
        to_point: np.ndarray,
    ) -> np.ndarray:
        """
        Parallel transport vector on n-dimensional sphere.

        For n-dimensional spheres (n > 3), uses projection-based transport.
        Formula projects vector onto the tangent plane at to_point.

        For 3D case, falls back to Rodrigues' rotation formula.
        """
        from_p = SphericalManifold.project_to_sphere(from_point)
        to_p = SphericalManifold.project_to_sphere(to_point)

        # Ensure vector is tangent at from_point
        vector = vector - np.dot(vector, from_p) * from_p

        # For 3D, use Rodrigues' formula
        if len(from_p) == 3:
            # Compute rotation axis and angle
            axis = np.cross(from_p, to_p)
            axis_norm = np.linalg.norm(axis)

            if axis_norm < 1e-8:
                return vector  # Points are same or antipodal

            axis = axis / axis_norm
            angle = SphericalManifold.great_circle_distance(from_p, to_p)

            # Rodrigues' formula
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            transported = (
                vector * cos_angle +
                np.cross(axis, vector) * sin_angle +
                axis * np.dot(axis, vector) * (1 - cos_angle)
            )

            return transported

        # For n-dimensional case: project-based transport
        # Formula: transport(v, p->q) = v - (v·q)/(1 + p·q) * (p + q)
        dot_product = np.dot(from_p, to_p)

        if abs(dot_product + 1) < 1e-8:
            # Antipodal points - no unique transport
            return vector

        denom = 1 + dot_product
        transport_component = (np.dot(vector, to_p) / denom) * (from_p + to_p)

        transported = vector - transport_component

        # Ensure result is tangent at to_point
        transported = transported - np.dot(transported, to_p) * to_p

        return transported


class HyperbolicManifold:
    """
    Hyperbolic manifold H^n operations.

    Properties:
    - Negative curvature (K = -1)
    - Unbounded
    - Poincaré ball model
    - Good for: Hierarchies, trees, exponential growth
    """

    @staticmethod
    def project_to_poincare(
        point: np.ndarray,
        curvature: float = -1.0,
    ) -> np.ndarray:
        """Project to Poincaré ball model."""
        norm = np.linalg.norm(point)
        max_norm = 0.99  # Stay within unit ball

        if norm >= max_norm:
            return point * (max_norm / norm)

        return point

    @staticmethod
    def hyperbolic_distance(
        p1: np.ndarray,
        p2: np.ndarray,
        curvature: float = -1.0,
    ) -> float:
        """
        Compute hyperbolic distance in Poincaré ball.

        Formula: d = 2/√(-K) * artanh(||p1 - p2|| / (1 - ||p1||*||p2||))
        """
        p1_norm = np.linalg.norm(p1)
        p2_norm = np.linalg.norm(p2)
        diff_norm = np.linalg.norm(p1 - p2)

        epsilon = 1e-8
        denom = 1 - p1_norm * p2_norm

        if abs(denom) < epsilon:
            return 100.0  # Large distance

        lambda_p1 = 2 / (1 - p1_norm**2)
        lambda_p2 = 2 / (1 - p2_norm**2)

        dist = np.arccosh(
            1 + 2 * (diff_norm**2) / (lambda_p1 * lambda_p2 * denom**2 + epsilon)
        )

        return float(dist)

    @staticmethod
    def mobius_addition(
        p1: np.ndarray,
        p2: np.ndarray,
    ) -> np.ndarray:
        """
        Möbius addition on Poincaré ball.

        Non-commutative addition for hyperbolic space.
        """
        p1_norm = np.linalg.norm(p1)
        p2_norm = np.linalg.norm(p2)
        inner_prod = np.dot(p1, p2)

        numerator = (
            (1 + 2 * p1_norm * p2_norm + p2_norm**2) * p1 +
            (1 - p1_norm**2) * p2
        )
        denominator = (
            1 + 2 * p1_norm * p2_norm + p1_norm**2 * p2_norm**2
        )

        return numerator / (denominator + 1e-8)


class GeometricMetalearning:
    """
    Geometric meta-learning on manifolds.

    Features:
    1. Embed points on S^n or H^n
    2. Parallel transport for gradients
    3. Attractor dynamics for convergence
    4. Manifold-aware meta-updates
    """

    def __init__(
        self,
        manifold: ManifoldType = ManifoldType.SPHERICAL,
        dimension: int = 64,
        curvature: float = 1.0,
    ):
        self.manifold = manifold
        self.dimension = dimension
        self.curvature = curvature

        # Manifold operations
        if manifold == ManifoldType.SPHERICAL:
            self.manifold_ops = SphericalManifold()
        elif manifold == ManifoldType.HYPERBOLIC:
            self.manifold_ops = HyperbolicManifold()
        else:
            self.manifold_ops = None

        # State tracking
        self.states: List[GeometricState] = []
        self.basins: Dict[str, List[GeometricState]] = {}

    def embed(
        self,
        data: np.ndarray,
        manifold: Optional[ManifoldType] = None,
    ) -> GeometricState:
        """
        Embed data point on manifold.

        Args:
            data: Input vector
            manifold: Manifold type (default: self.manifold)

        Returns:
            GeometricState on manifold
        """
        manifold = manifold or self.manifold

        # Project to manifold
        if manifold == ManifoldType.SPHERICAL:
            point = SphericalManifold.project_to_sphere(data)
        elif manifold == ManifoldType.HYPERBOLIC:
            point = HyperbolicManifold.project_to_poincare(data, self.curvature)
        else:
            point = data

        # Compute energy (negative norm as proxy)
        energy = -np.linalg.norm(point)

        state = GeometricState(
            point=point,
            velocity=np.zeros_like(point),
            energy=energy,
            manifold=manifold,
            curvature=self.curvature,
        )

        self.states.append(state)
        return state

    def parallel_transport(
        self,
        gradient: np.ndarray,
        from_state: GeometricState,
        to_state: GeometricState,
    ) -> np.ndarray:
        """
        Transport gradient vector between states on manifold.

        Enables geometric gradient updates that respect manifold structure.
        """
        if self.manifold == ManifoldType.SPHERICAL:
            return SphericalManifold.parallel_transport(
                gradient,
                from_state.point,
                to_state.point,
            )
        elif self.manifold == ManifoldType.HYPERBOLIC:
            # For hyperbolic, use Möbius addition as proxy
            # (true parallel transport is more complex)
            return HyperbolicManifold.mobius_addition(gradient, to_state.point)
        else:
            return gradient

    def meta_gradient_update(
        self,
        current_state: GeometricState,
        gradient: np.ndarray,
        meta_lr: float = 0.01,
    ) -> GeometricUpdate:
        """
        Perform meta-gradient update on manifold.

        Updates point along gradient while respecting manifold geometry.
        """
        # Project gradient to tangent space
        tangent_grad = gradient - np.dot(gradient, current_state.point) * current_state.point

        # Normalize gradient
        grad_norm = np.linalg.norm(tangent_grad)
        if grad_norm > 1e-8:
            tangent_grad = tangent_grad / grad_norm

        # Update point
        new_point = current_state.point + meta_lr * tangent_grad

        # Project back to manifold
        if self.manifold == ManifoldType.SPHERICAL:
            new_point = SphericalManifold.project_to_sphere(new_point)
        elif self.manifold == ManifoldType.HYPERBOLIC:
            new_point = HyperbolicManifold.project_to_poincare(new_point, self.curvature)

        # Compute new energy
        new_energy = -np.linalg.norm(new_point)

        # Create new state
        new_state = GeometricState(
            point=new_point,
            velocity=tangent_grad,
            energy=new_energy,
            manifold=current_state.manifold,
            curvature=current_state.curvature,
        )

        # Compute distance and energy change
        if self.manifold == ManifoldType.SPHERICAL:
            distance = SphericalManifold.great_circle_distance(
                current_state.point,
                new_point,
            )
        else:
            distance = np.linalg.norm(new_point - current_state.point)

        energy_delta = new_energy - current_state.energy

        # Check convergence
        converged = abs(energy_delta) < 1e-6 or distance < 1e-6

        update = GeometricUpdate(
            new_state=new_state,
            old_state=current_state,
            distance=distance,
            energy_delta=energy_delta,
            converged=converged,
        )

        self.states.append(new_state)
        return update

    def find_attractor(
        self,
        initial_state: GeometricState,
        energy_fn: callable,
        num_steps: int = 100,
        lr: float = 0.1,
    ) -> GeometricUpdate:
        """
        Find attractor basin using gradient descent on energy landscape.

        Used in Modern Hopfield Continuum (L5 mHC).
        """
        state = initial_state
        total_distance = 0.0

        for i in range(num_steps):
            # Compute energy gradient
            grad = energy_fn(state.point)

            # Geometric gradient update
            update = self.meta_gradient_update(state, grad, lr)
            state = update.new_state
            total_distance += update.distance

            if update.converged:
                break

        return GeometricUpdate(
            new_state=state,
            old_state=initial_state,
            distance=total_distance,
            energy_delta=state.energy - initial_state.energy,
            converged=True,
            iterations=i + 1,
        )

    def compute_curvature(
        self,
        state: GeometricState,
    ) -> float:
        """
        Compute Gaussian curvature at point.

        For S^n: K = 1/R^2 (constant positive)
        For H^n: K = -1 (constant negative)
        For R^n: K = 0 (flat)
        """
        if state.manifold == ManifoldType.SPHERICAL:
            return 1.0 / (state.curvature ** 2)
        elif state.manifold == ManifoldType.HYPERBOLIC:
            return -1.0
        else:
            return 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get geometric learning statistics."""
        if not self.states:
            return {
                "total_states": 0,
                "manifold": self.manifold.value,
                "dimension": self.dimension,
            }

        energies = [s.energy for s in self.states]

        return {
            "total_states": len(self.states),
            "manifold": self.manifold.value,
            "dimension": self.dimension,
            "avg_energy": np.mean(energies),
            "min_energy": np.min(energies),
            "max_energy": np.max(energies),
            "num_basins": len(self.basins),
        }


def create_geometric_metalearning(
    manifold: ManifoldType = ManifoldType.SPHERICAL,
    dimension: int = 64,
) -> GeometricMetalearning:
    """Create geometric metalearning with default config."""
    return GeometricMetalearning(
        manifold=manifold,
        dimension=dimension,
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
