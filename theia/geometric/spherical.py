"""
Theia Spherical Embeddings — S^n projections and operations.

S^n captures local angular similarity (what transformers do with cosine).
Projects vectors to unit sphere, computes geodesic distances.
"""

import math
from typing import List, Tuple, Union
import numpy as np


def project_sphere(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Project vector(s) to unit sphere via L2 normalization.
    
    Args:
        x: Vector or batch of vectors [..., d]
        eps: Small constant for numerical stability
        
    Returns:
        Normalized vector(s) on S^{d-1}
    """
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)


def geodesic_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute geodesic (angular) distance on sphere.
    
    d(a, b) = arccos(⟨a, b⟩)
    
    Args:
        a, b: Unit vectors on sphere
        
    Returns:
        Angular distance in radians [0, π]
    """
    # Clamp for numerical stability
    dot = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.arccos(dot)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity (equivalent to dot product on unit sphere).
    
    Args:
        a, b: Vectors (not necessarily normalized)
        
    Returns:
        Cosine similarity in [-1, 1]
    """
    a_norm = project_sphere(a)
    b_norm = project_sphere(b)
    return np.sum(a_norm * b_norm, axis=-1)


def spherical_mean(vectors: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Compute spherical mean (Fréchet mean on sphere).
    
    For unit vectors, this is approximately the normalized weighted sum.
    
    Args:
        vectors: Array of unit vectors [n, d]
        weights: Optional weights [n]
        
    Returns:
        Mean vector on sphere
    """
    if weights is None:
        weights = np.ones(len(vectors))
    weights = weights / weights.sum()
    
    weighted_sum = np.sum(vectors * weights[:, np.newaxis], axis=0)
    return project_sphere(weighted_sum)


def random_sphere(n: int, d: int) -> np.ndarray:
    """
    Generate n random points uniformly on S^{d-1}.
    
    Uses Gaussian projection method.
    """
    x = np.random.randn(n, d)
    return project_sphere(x)


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation (SLERP).
    
    Args:
        a, b: Unit vectors
        t: Interpolation parameter [0, 1]
        
    Returns:
        Interpolated point on geodesic from a to b
    """
    dot = np.clip(np.sum(a * b), -1.0, 1.0)
    theta = np.arccos(dot)
    
    if theta < 1e-6:
        return a  # Points are essentially the same
    
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / sin_theta


__all__ = [
    "project_sphere",
    "geodesic_distance", 
    "cosine_similarity",
    "spherical_mean",
    "random_sphere",
    "slerp",
]
