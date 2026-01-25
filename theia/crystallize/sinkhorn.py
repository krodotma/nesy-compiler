"""
Theia Sinkhorn Operator — Projection to Birkhoff polytope.

The Sinkhorn algorithm iteratively normalizes rows and columns
to produce a doubly stochastic matrix (element of Birkhoff polytope B_n).

As temperature β → ∞, solutions crystallize toward vertices (permutations).
"""

import numpy as np
from typing import Tuple


def sinkhorn(
    M: np.ndarray,
    beta: float = 1.0,
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int]:
    """
    Sinkhorn-Knopp algorithm for projecting to Birkhoff polytope.
    
    Given a non-negative matrix M, computes:
        P = D_1 @ exp(β * M) @ D_2
    
    where D_1, D_2 are diagonal matrices such that P is doubly stochastic.
    
    Args:
        M: Input matrix [n, n] (typically a cost or similarity matrix)
        beta: Temperature parameter (higher = sharper)
        iterations: Maximum iterations
        tolerance: Convergence threshold
        
    Returns:
        (doubly_stochastic_matrix, iterations_used)
    """
    # Apply temperature scaling
    K = np.exp(beta * M)
    
    n = K.shape[0]
    u = np.ones(n)
    v = np.ones(n)
    
    for i in range(iterations):
        u_prev = u.copy()
        
        # Row normalization
        u = 1.0 / (K @ v + 1e-8)
        
        # Column normalization
        v = 1.0 / (K.T @ u + 1e-8)
        
        # Check convergence
        if np.max(np.abs(u - u_prev)) < tolerance:
            break
    
    # Compute doubly stochastic matrix
    P = np.diag(u) @ K @ np.diag(v)
    
    return P, i + 1


def sinkhorn_log_stable(
    M: np.ndarray,
    beta: float = 1.0,
    iterations: int = 100,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int]:
    """
    Log-stabilized Sinkhorn algorithm.
    
    More numerically stable for high temperature values.
    Works in log-space to avoid overflow.
    
    Args:
        M: Input matrix [n, n]
        beta: Temperature parameter
        iterations: Maximum iterations
        tolerance: Convergence threshold
        
    Returns:
        (doubly_stochastic_matrix, iterations_used)
    """
    n = M.shape[0]
    
    # Log of row/column scaling factors
    f = np.zeros(n)
    g = np.zeros(n)
    
    log_K = beta * M
    
    for i in range(iterations):
        f_prev = f.copy()
        
        # Log-space row normalization: f = -logsumexp(log_K + g)
        f = -np.log(np.sum(np.exp(log_K + g[np.newaxis, :]), axis=1) + 1e-8)
        
        # Log-space column normalization: g = -logsumexp(log_K.T + f)
        g = -np.log(np.sum(np.exp(log_K.T + f[np.newaxis, :]), axis=1) + 1e-8)
        
        # Check convergence
        if np.max(np.abs(f - f_prev)) < tolerance:
            break
    
    # Compute doubly stochastic matrix
    P = np.exp(f[:, np.newaxis] + log_K + g[np.newaxis, :])
    
    return P, i + 1


def hungarian_from_sinkhorn(P: np.ndarray) -> np.ndarray:
    """
    Extract hard permutation from soft doubly stochastic matrix.
    
    Uses row-wise argmax (greedy) for simple extraction.
    For optimal extraction, use scipy.optimize.linear_sum_assignment.
    
    Args:
        P: Doubly stochastic matrix
        
    Returns:
        Permutation matrix (binary)
    """
    n = P.shape[0]
    perm = np.zeros_like(P)
    
    # Greedy assignment by row
    used_cols = set()
    for i in range(n):
        row = P[i].copy()
        # Mask used columns
        for j in used_cols:
            row[j] = -np.inf
        j = np.argmax(row)
        perm[i, j] = 1.0
        used_cols.add(j)
    
    return perm


def distance_to_birkhoff(P: np.ndarray) -> float:
    """
    Compute distance from P to nearest doubly stochastic matrix.
    
    Measures how far P is from the Birkhoff polytope.
    Uses L2 distance from row/column sum constraints.
    
    Args:
        P: Any matrix
        
    Returns:
        Distance to Birkhoff polytope (0 if already doubly stochastic)
    """
    row_sums = np.sum(P, axis=1)
    col_sums = np.sum(P, axis=0)
    
    row_error = np.sum((row_sums - 1) ** 2)
    col_error = np.sum((col_sums - 1) ** 2)
    
    return np.sqrt(row_error + col_error)


def distance_to_vertex(P: np.ndarray) -> float:
    """
    Compute distance from P to nearest vertex of Birkhoff polytope.
    
    Vertices are permutation matrices (binary).
    
    Args:
        P: Doubly stochastic matrix
        
    Returns:
        Frobenius distance to nearest permutation matrix
    """
    # Get nearest permutation via greedy extraction
    perm = hungarian_from_sinkhorn(P)
    return np.linalg.norm(P - perm, 'fro')


def crystallization_pressure(P: np.ndarray, lambda_: float = 1.0) -> float:
    """
    Compute crystallization pressure toward discrete structure.
    
    Part of Lyapunov coupling: L(x) = E(x) + λ·d(x, B_n)²
    
    Args:
        P: Current matrix (may be soft doubly stochastic)
        lambda_: Pressure coefficient
        
    Returns:
        Pressure value (higher = more incentive to crystallize)
    """
    vertex_dist = distance_to_vertex(P)
    return lambda_ * (vertex_dist ** 2)


__all__ = [
    "sinkhorn",
    "sinkhorn_log_stable",
    "hungarian_from_sinkhorn",
    "distance_to_birkhoff",
    "distance_to_vertex",
    "crystallization_pressure",
]
