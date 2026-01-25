"""
Theia Birkhoff Polytope — Face lattice and dynamics.

The Birkhoff polytope B_n is the set of n×n doubly stochastic matrices.

Properties:
    - Vertices: n! permutation matrices
    - Dimension: (n-1)²
    - n² - 2n + 2 facets
    - Face lattice captures hierarchical structure

Integration:
    - L2 mHC attractors crystallize toward vertices
    - Sinkhorn operator projects onto B_n
    - Lyapunov coupling: L(x) = E(x) + λ·d(x, B_n)²
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from itertools import permutations


@dataclass
class Face:
    """
    A face of the Birkhoff polytope.
    
    Attributes:
        dimension: Dimension of the face (0=vertex, 1=edge, ...)
        vertices: Indices of vertices in this face
        pattern: Zero pattern matrix (1 where entries can be nonzero)
    """
    dimension: int
    vertices: List[int] = field(default_factory=list)
    pattern: Optional[np.ndarray] = None
    
    def contains(self, other: "Face") -> bool:
        """Check if other face is contained in this face."""
        return set(other.vertices).issubset(set(self.vertices))


class BirkhoffPolytope:
    """
    The Birkhoff polytope B_n of doubly stochastic matrices.
    
    Provides:
        - Vertex enumeration (permutation matrices)
        - Face lattice navigation
        - Distance to polytope
        - Crystallization dynamics
    """
    
    def __init__(self, n: int):
        self.n = n
        self._vertices: Optional[np.ndarray] = None
        self._faces: Dict[int, List[Face]] = {}  # dim -> faces
        
    @property
    def vertices(self) -> np.ndarray:
        """Get all vertices (permutation matrices) as flattened arrays."""
        if self._vertices is None:
            self._vertices = self._enumerate_vertices()
        return self._vertices
    
    @property
    def n_vertices(self) -> int:
        """Number of vertices = n!"""
        from math import factorial
        return factorial(self.n)
    
    @property
    def dimension(self) -> int:
        """Dimension of polytope = (n-1)²"""
        return (self.n - 1) ** 2
    
    def _enumerate_vertices(self) -> np.ndarray:
        """Enumerate all vertices (permutation matrices)."""
        vertices = []
        for perm in permutations(range(self.n)):
            P = np.zeros((self.n, self.n))
            for i, j in enumerate(perm):
                P[i, j] = 1.0
            vertices.append(P.flatten())
        return np.array(vertices)
    
    def is_doubly_stochastic(self, X: np.ndarray, tol: float = 1e-6) -> bool:
        """Check if matrix is doubly stochastic."""
        if X.shape != (self.n, self.n):
            return False
        
        row_sums = np.sum(X, axis=1)
        col_sums = np.sum(X, axis=0)
        
        return (
            np.all(X >= -tol) and
            np.allclose(row_sums, 1.0, atol=tol) and
            np.allclose(col_sums, 1.0, atol=tol)
        )
    
    def distance_to_vertex(self, X: np.ndarray, vertex_idx: int) -> float:
        """Euclidean distance from X to a specific vertex."""
        x_flat = X.flatten()
        return float(np.linalg.norm(x_flat - self.vertices[vertex_idx]))
    
    def nearest_vertex(self, X: np.ndarray) -> Tuple[int, float]:
        """Find nearest vertex to X."""
        x_flat = X.flatten()
        distances = np.linalg.norm(self.vertices - x_flat, axis=1)
        idx = int(np.argmin(distances))
        return idx, float(distances[idx])
    
    def crystallization_pressure(self, X: np.ndarray) -> float:
        """
        Compute crystallization pressure toward nearest vertex.
        
        pressure = d(X, V_nearest)²
        
        This is the term that couples with mHC energy.
        """
        _, dist = self.nearest_vertex(X)
        return dist ** 2
    
    def lyapunov_energy(
        self,
        X: np.ndarray,
        mhc_energy: float,
        lambda_coupling: float = 0.5,
    ) -> float:
        """
        Compute coupled Lyapunov energy.
        
        L(X) = E_mhc(X) + λ·crystallization_pressure(X)
        
        This couples the Hopfield energy with Birkhoff crystallization.
        """
        pressure = self.crystallization_pressure(X)
        return mhc_energy + lambda_coupling * pressure
    
    def project_to_polytope(
        self,
        X: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> np.ndarray:
        """
        Project matrix onto B_n via Sinkhorn iteration.
        """
        # Ensure positive entries
        A = np.maximum(X, 1e-10)
        
        for _ in range(max_iter):
            # Row normalization
            row_sums = A.sum(axis=1, keepdims=True)
            A = A / row_sums
            
            # Column normalization
            col_sums = A.sum(axis=0, keepdims=True)
            A = A / col_sums
            
            # Check convergence
            if (np.allclose(A.sum(axis=1), 1.0, atol=tol) and
                np.allclose(A.sum(axis=0), 1.0, atol=tol)):
                break
        
        return A
    
    def crystallize_step(
        self,
        X: np.ndarray,
        step_size: float = 0.1,
    ) -> np.ndarray:
        """
        Take one gradient step toward nearest vertex.
        
        Moves X toward crystallization.
        """
        idx, _ = self.nearest_vertex(X)
        target = self.vertices[idx].reshape(self.n, self.n)
        
        # Gradient step toward target
        X_new = X + step_size * (target - X)
        
        # Project back onto polytope
        return self.project_to_polytope(X_new)


# =============================================================================
# FACE LATTICE
# =============================================================================

class FaceLattice:
    """
    Face lattice of the Birkhoff polytope.
    
    The face lattice captures the hierarchical structure:
        - Level 0: Vertices (permutation matrices)
        - Level 1: Edges
        - ...
        - Top: The full polytope
    """
    
    def __init__(self, polytope: BirkhoffPolytope):
        self.polytope = polytope
        self._lattice: Dict[int, List[Face]] = {}
        self._edges: List[Tuple[int, int]] = []
        
    def compute_vertices(self) -> List[Face]:
        """Compute vertex faces (dimension 0)."""
        vertices = []
        for i in range(self.polytope.n_vertices):
            vertices.append(Face(dimension=0, vertices=[i]))
        self._lattice[0] = vertices
        return vertices
    
    def compute_edges(self) -> List[Face]:
        """
        Compute edge faces (dimension 1).
        
        Two permutation matrices are connected by an edge
        iff they differ by a single transposition.
        """
        if 0 not in self._lattice:
            self.compute_vertices()
        
        edges = []
        n = self.polytope.n
        vertices = self.polytope.vertices
        
        # For each pair of vertices
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                # Check if they differ by exactly 2 positions in each row/col
                diff = np.abs(vertices[i] - vertices[j])
                if np.sum(diff) == 4:  # Single transposition
                    edges.append(Face(dimension=1, vertices=[i, j]))
                    self._edges.append((i, j))
        
        self._lattice[1] = edges
        return edges
    
    def get_neighbors(self, vertex_idx: int) -> List[int]:
        """Get indices of vertices adjacent to given vertex."""
        if 1 not in self._lattice:
            self.compute_edges()
        
        neighbors = []
        for (i, j) in self._edges:
            if i == vertex_idx:
                neighbors.append(j)
            elif j == vertex_idx:
                neighbors.append(i)
        return neighbors
    
    def navigate_to_nearest(
        self,
        X: np.ndarray,
        target_idx: int,
        max_steps: int = 100,
    ) -> List[int]:
        """
        Navigate through face lattice from nearest vertex to target.
        
        Returns path as list of vertex indices.
        """
        if 1 not in self._lattice:
            self.compute_edges()
        
        current_idx, _ = self.polytope.nearest_vertex(X)
        path = [current_idx]
        
        for _ in range(max_steps):
            if current_idx == target_idx:
                break
            
            # BFS-style: find neighbor closest to target
            neighbors = self.get_neighbors(current_idx)
            if not neighbors:
                break
            
            # Use Euclidean distance in flattened space
            target_flat = self.polytope.vertices[target_idx]
            distances = [
                np.linalg.norm(self.polytope.vertices[n] - target_flat)
                for n in neighbors
            ]
            
            best_neighbor = neighbors[int(np.argmin(distances))]
            path.append(best_neighbor)
            current_idx = best_neighbor
        
        return path


__all__ = [
    "Face",
    "BirkhoffPolytope",
    "FaceLattice",
]
