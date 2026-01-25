"""
Theia Fiber Bundle — Geometric structure over base manifold.

A fiber bundle E → M consists of:
    - Total space E (where embeddings live)
    - Base space M (spherical-hyperbolic hybrid S^n ∐ H^n)
    - Projection π: E → M
    - Fiber F_p over each point p ∈ M (local symbolic structure)
    - Connection ∇ (parallel transport of structure)
    - Curvature R (learned geometry, obstruction to global triviality)

This provides richer structure than flat vector spaces:
    - Concepts = sections σ: M → E
    - Relationships = connection ∇
    - Analogies = parallel transport
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod

from theia.geometric.spherical import project_sphere, geodesic_distance
from theia.geometric.hyperbolic import project_poincare, hyperbolic_distance


@dataclass
class FiberPoint:
    """
    Point in fiber bundle.
    
    Attributes:
        base: Point in base manifold M
        fiber: Point in fiber F over base
        manifold: Which base manifold ('sphere' or 'hyperbolic')
    """
    base: np.ndarray       # Point in M
    fiber: np.ndarray      # Point in fiber F
    manifold: str = "sphere"
    
    @property
    def total(self) -> np.ndarray:
        """Full point in total space E."""
        return np.concatenate([self.base, self.fiber])
    
    @classmethod
    def from_total(cls, point: np.ndarray, base_dim: int, manifold: str = "sphere") -> "FiberPoint":
        """Decompose total space point into base + fiber."""
        return cls(
            base=point[:base_dim],
            fiber=point[base_dim:],
            manifold=manifold,
        )


class Connection:
    """
    Connection on fiber bundle — enables parallel transport.
    
    A connection specifies how to transport fiber data along
    paths in the base manifold.
    
    Learned connections capture how concepts relate across
    different regions of the representation space.
    """
    
    def __init__(self, base_dim: int, fiber_dim: int):
        self.base_dim = base_dim
        self.fiber_dim = fiber_dim
        
        # Connection coefficients (Christoffel symbols analog)
        # Shape: [base_dim, fiber_dim, fiber_dim]
        self.christoffel = np.zeros((base_dim, fiber_dim, fiber_dim))
    
    def parallel_transport(
        self,
        fiber_vector: np.ndarray,
        start_base: np.ndarray,
        end_base: np.ndarray,
        steps: int = 10,
    ) -> np.ndarray:
        """
        Parallel transport fiber vector along geodesic.
        
        Transports a vector in the fiber from start_base to end_base
        along the shortest path in the base manifold.
        
        Args:
            fiber_vector: Vector in fiber at start_base
            start_base: Starting point in base
            end_base: Ending point in base
            steps: Number of integration steps
            
        Returns:
            Transported vector in fiber at end_base
        """
        # Linear interpolation (simplified; should use geodesic)
        v = fiber_vector.copy()
        
        for i in range(steps):
            t = i / steps
            base_point = (1 - t) * start_base + t * end_base
            
            # Apply connection (infinitesimal transport)
            delta_base = (end_base - start_base) / steps
            
            # dv = -Γ^k_ij v^j dx^i
            for k in range(self.fiber_dim):
                for i in range(self.base_dim):
                    for j in range(self.fiber_dim):
                        v[k] -= self.christoffel[i, k, j] * v[j] * delta_base[i]
        
        return v
    
    def curvature(self, base_point: np.ndarray) -> np.ndarray:
        """
        Compute curvature tensor at base point.
        
        Curvature measures the obstruction to global triviality —
        non-zero curvature means parallel transport depends on path.
        
        Returns:
            Curvature tensor R of shape [base_dim, base_dim, fiber_dim, fiber_dim]
        """
        # Simplified: return trace of Christoffel variations
        # Full implementation would compute R^k_lij = ∂_i Γ^k_lj - ∂_j Γ^k_li + ...
        return np.einsum('ikl,jlm->ijkm', self.christoffel, self.christoffel)
    
    def set_christoffel(self, values: np.ndarray) -> None:
        """Set connection coefficients (for learning)."""
        assert values.shape == self.christoffel.shape
        self.christoffel = values.copy()


class FiberBundle:
    """
    Fiber bundle over spherical-hyperbolic base.
    
    Provides the geometric substrate for neurosymbolic computation:
        - Base M = S^n ∐ H^n (spherical-hyperbolic hybrid)
        - Fiber F carries local symbolic affordances
        - Connection enables relating symbols across regions
    
    This is the L1 layer of Theia architecture.
    """
    
    def __init__(
        self,
        base_dim: int,
        fiber_dim: int,
        use_hyperbolic: bool = True,
    ):
        self.base_dim = base_dim
        self.fiber_dim = fiber_dim
        self.use_hyperbolic = use_hyperbolic
        
        # Connection for parallel transport
        self.connection = Connection(base_dim, fiber_dim)
        
        # Cached sections (learned concept mappings)
        self._sections: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    
    @property
    def total_dim(self) -> int:
        return self.base_dim + self.fiber_dim
    
    def project(self, point: np.ndarray, manifold: str = "sphere") -> np.ndarray:
        """
        Project point to base manifold.
        
        Args:
            point: Point in total space
            manifold: 'sphere' or 'hyperbolic'
            
        Returns:
            Projected point in base
        """
        base = point[:self.base_dim]
        
        if manifold == "sphere":
            return project_sphere(base)
        elif manifold == "hyperbolic":
            return project_poincare(base)
        else:
            raise ValueError(f"Unknown manifold: {manifold}")
    
    def lift(
        self,
        base_point: np.ndarray,
        fiber_value: np.ndarray,
        manifold: str = "sphere",
    ) -> FiberPoint:
        """
        Lift base point to total space with fiber value.
        
        Args:
            base_point: Point in base M
            fiber_value: Value in fiber F
            manifold: Base manifold type
            
        Returns:
            Point in total space E
        """
        # Project base to correct manifold
        if manifold == "sphere":
            base = project_sphere(base_point)
        else:
            base = project_poincare(base_point)
        
        return FiberPoint(base=base, fiber=fiber_value, manifold=manifold)
    
    def section(
        self,
        base_point: np.ndarray,
        section_name: str = "default",
    ) -> FiberPoint:
        """
        Evaluate section at base point.
        
        A section σ: M → E assigns a fiber value to each point.
        Represents a "concept" distributed over the space.
        
        Args:
            base_point: Point in base M
            section_name: Named section to evaluate
            
        Returns:
            Point in total space
        """
        if section_name not in self._sections:
            # Default section: zero fiber
            fiber = np.zeros(self.fiber_dim)
        else:
            fiber = self._sections[section_name](base_point)
        
        return self.lift(base_point, fiber)
    
    def add_section(
        self,
        name: str,
        mapping: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """Add named section (concept mapping)."""
        self._sections[name] = mapping
    
    def analogy(
        self,
        a: FiberPoint,
        b: FiberPoint,
        c: FiberPoint,
    ) -> FiberPoint:
        """
        Compute analogy: a is to b as c is to ?
        
        Uses parallel transport:
            1. Compute difference in fiber at a vs b
            2. Transport difference to c
            3. Apply to get result
        
        This is the geometric interpretation of analogical reasoning.
        """
        # Fiber difference: what changed from a to b?
        diff = b.fiber - a.fiber
        
        # Transport this difference to c's location
        transported = self.connection.parallel_transport(
            diff,
            a.base,
            c.base,
        )
        
        # Apply to c
        result_fiber = c.fiber + transported
        
        return FiberPoint(
            base=c.base.copy(),
            fiber=result_fiber,
            manifold=c.manifold,
        )
    
    def distance(self, p1: FiberPoint, p2: FiberPoint) -> float:
        """
        Compute distance in total space.
        
        Combines base distance with fiber distance.
        """
        # Base distance (geodesic on manifold)
        if p1.manifold == "sphere":
            base_dist = float(geodesic_distance(p1.base, p2.base))
        else:
            base_dist = float(hyperbolic_distance(p1.base, p2.base))
        
        # Fiber distance (Euclidean)
        fiber_dist = float(np.linalg.norm(p1.fiber - p2.fiber))
        
        # Combined (could weight differently)
        return np.sqrt(base_dist**2 + fiber_dist**2)


__all__ = [
    "FiberPoint",
    "Connection",
    "FiberBundle",
]
