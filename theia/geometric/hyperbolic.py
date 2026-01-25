"""
Theia Hyperbolic Embeddings — Poincaré ball model for H^n.

H^n captures global hierarchical structure (tree-like).
The Poincaré ball model embeds hyperbolic space in a unit ball.

Key insight: Hyperbolic space has exponentially more room at edges,
making it ideal for embedding hierarchies.
"""

import math
import numpy as np


def project_poincare(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Project vector(s) into Poincaré ball (||x|| < 1).
    
    Uses retraction for points outside ball.
    
    Args:
        x: Vector(s) to project
        eps: Margin from boundary for numerical stability
        
    Returns:
        Vector(s) inside Poincaré ball
    """
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    max_norm = 1 - eps
    
    # Only project if outside ball
    scale = np.where(norm > max_norm, max_norm / (norm + 1e-8), 1.0)
    return x * scale


def mobius_add(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Möbius addition in Poincaré ball.
    
    x ⊕ y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / 
            (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
    
    Args:
        x, y: Points in Poincaré ball
        c: Negative curvature parameter (default 1.0)
        
    Returns:
        Möbius sum in Poincaré ball
    """
    x_sq = np.sum(x * x, axis=-1, keepdims=True)
    y_sq = np.sum(y * y, axis=-1, keepdims=True)
    xy = np.sum(x * y, axis=-1, keepdims=True)
    
    num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
    denom = 1 + 2 * c * xy + c * c * x_sq * y_sq
    
    return project_poincare(num / (denom + 1e-8))


def hyperbolic_distance(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Compute hyperbolic distance in Poincaré ball.
    
    d(x, y) = (2/√c) * arctanh(√c * ||−x ⊕ y||)
    
    Args:
        x, y: Points in Poincaré ball
        c: Negative curvature parameter
        
    Returns:
        Hyperbolic distance (always positive)
    """
    diff = mobius_add(-x, y, c)
    norm = np.linalg.norm(diff, axis=-1)
    
    # Clamp for numerical stability
    norm = np.clip(norm, 0, 1 - 1e-5)
    
    return (2 / math.sqrt(c)) * np.arctanh(math.sqrt(c) * norm)


def exp_map(origin: np.ndarray, direction: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Exponential map from tangent space to Poincaré ball.
    
    exp_x(v) = x ⊕ (tanh(√c * λ_x * ||v|| / 2) * v / (√c * ||v||))
    
    where λ_x = 2 / (1 - c||x||²) is the conformal factor.
    
    Args:
        origin: Base point in Poincaré ball
        direction: Tangent vector at origin
        c: Curvature parameter
        
    Returns:
        Point in Poincaré ball along geodesic
    """
    origin_sq = np.sum(origin * origin, axis=-1, keepdims=True)
    lambda_x = 2 / (1 - c * origin_sq + 1e-8)
    
    direction_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction_norm = np.clip(direction_norm, 1e-8, None)
    
    sqrt_c = math.sqrt(c)
    coef = np.tanh(sqrt_c * lambda_x * direction_norm / 2)
    normalized = direction / (sqrt_c * direction_norm)
    
    return mobius_add(origin, coef * normalized, c)


def log_map(origin: np.ndarray, point: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Logarithmic map from Poincaré ball to tangent space.
    
    log_x(y) = (2 / (√c * λ_x)) * arctanh(√c * ||−x ⊕ y||) * (−x ⊕ y) / ||−x ⊕ y||
    
    Args:
        origin: Base point
        point: Target point
        c: Curvature parameter
        
    Returns:
        Tangent vector at origin pointing toward point
    """
    origin_sq = np.sum(origin * origin, axis=-1, keepdims=True)
    lambda_x = 2 / (1 - c * origin_sq + 1e-8)
    
    diff = mobius_add(-origin, point, c)
    diff_norm = np.linalg.norm(diff, axis=-1, keepdims=True)
    diff_norm = np.clip(diff_norm, 1e-8, 1 - 1e-5)
    
    sqrt_c = math.sqrt(c)
    coef = (2 / (sqrt_c * lambda_x)) * np.arctanh(sqrt_c * diff_norm)
    
    return coef * diff / diff_norm


def hyperbolic_mean(
    points: np.ndarray, 
    weights: np.ndarray = None,
    c: float = 1.0,
    iterations: int = 10,
) -> np.ndarray:
    """
    Compute hyperbolic mean (Fréchet mean in Poincaré ball).
    
    Uses iterative algorithm: start at centroid, 
    repeatedly move toward weighted mean in tangent space.
    
    Args:
        points: Array of points [n, d]
        weights: Optional weights [n]
        c: Curvature parameter
        iterations: Number of iterations
        
    Returns:
        Mean point in Poincaré ball
    """
    if weights is None:
        weights = np.ones(len(points)) / len(points)
    else:
        weights = weights / weights.sum()
    
    # Initialize at origin (can also use Euclidean mean projected)
    mean = np.zeros_like(points[0])
    
    for _ in range(iterations):
        # Map all points to tangent space at current mean
        tangent_vectors = np.array([log_map(mean, p, c) for p in points])
        
        # Weighted average in tangent space
        weighted_tangent = np.sum(tangent_vectors * weights[:, np.newaxis], axis=0)
        
        # Map back to ball
        mean = exp_map(mean, weighted_tangent, c)
    
    return mean


__all__ = [
    "project_poincare",
    "mobius_add",
    "hyperbolic_distance",
    "exp_map",
    "log_map",
    "hyperbolic_mean",
]
