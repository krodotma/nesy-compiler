"""
Theia Modern Hopfield Network — Dense associative memory with exponential capacity.

Based on Ramsauer et al. "Hopfield Networks is All You Need" (2021):
- Energy: E(x) = -Σ log(Σ exp(β⟨x,ξ⟩))
- Dynamics: ẋ = -∇E(x) = softmax retrieval
- Capacity: Exponential in dimension (not linear like classic Hopfield)

Key insight: Transformer attention IS Hopfield update.
Attractors ≅ stored patterns (memories).
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class HopfieldMemory:
    """
    Modern Hopfield Network with dense associative memory.
    
    Stores patterns as attractors in energy landscape.
    Retrieval via energy minimization (softmax attention).
    
    Attributes:
        patterns: Stored patterns [n_patterns, d]
        beta: Inverse temperature (higher = sharper)
        max_iter: Maximum retrieval iterations
        tolerance: Convergence tolerance
    """
    patterns: np.ndarray  # [n_patterns, d]
    beta: float = 1.0
    max_iter: int = 100
    tolerance: float = 1e-6
    
    def energy(self, x: np.ndarray) -> float:
        """
        Compute energy of state x.
        
        E(x) = -log(Σ_μ exp(β⟨x, ξ_μ⟩)) + const
        
        Lower energy = closer to stored pattern.
        """
        # Compute similarities
        similarities = self.beta * (self.patterns @ x)  # [n_patterns]
        
        # Log-sum-exp for numerical stability
        max_sim = np.max(similarities)
        lse = max_sim + np.log(np.sum(np.exp(similarities - max_sim)))
        
        return -lse
    
    def energy_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute gradient of energy.
        
        ∇E(x) = -β Σ_μ softmax(β⟨x,ξ⟩)_μ · ξ_μ
        
        This is the update direction for dynamics.
        """
        similarities = self.beta * (self.patterns @ x)
        
        # Softmax attention weights
        max_sim = np.max(similarities)
        exp_sim = np.exp(similarities - max_sim)
        weights = exp_sim / np.sum(exp_sim)  # [n_patterns]
        
        # Weighted sum of patterns (attention output)
        gradient = -self.beta * (weights @ self.patterns)
        
        return gradient
    
    def retrieve(self, query: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """
        Retrieve pattern from memory via energy minimization.
        
        Starting from query, follow gradient flow until convergence.
        This is equivalent to iterative attention.
        
        Args:
            query: Initial state [d]
            
        Returns:
            (retrieved_pattern, iterations, final_energy)
        """
        x = query.copy()
        
        if self.patterns.size == 0:
            return x, 0, 0.0
            
        for i in range(self.max_iter):
            # Compute attention weights
            similarities = self.beta * (self.patterns @ x)
            max_sim = np.max(similarities)
            exp_sim = np.exp(similarities - max_sim)
            weights = exp_sim / np.sum(exp_sim)
            
            # Update state (attention output)
            x_new = weights @ self.patterns
            
            # Check convergence
            if np.linalg.norm(x_new - x) < self.tolerance:
                return x_new, i + 1, self.energy(x_new)
            
            x = x_new
        
        return x, self.max_iter, self.energy(x)
    
    def retrieve_soft(self, query: np.ndarray) -> np.ndarray:
        """
        Single-step soft retrieval (attention).
        
        This is what transformers do in one attention layer.
        
        Args:
            query: Query vector [d]
            
        Returns:
            Soft retrieval result [d]
        """
        similarities = self.beta * (self.patterns @ query)
        weights = np.exp(similarities - np.max(similarities))
        weights = weights / np.sum(weights)
        return weights @ self.patterns
    
    def add_pattern(self, pattern: np.ndarray) -> None:
        """Add pattern to memory."""
        pattern = pattern.reshape(1, -1)
        self.patterns = np.vstack([self.patterns, pattern])
    
    def clear(self) -> None:
        """Clear all stored patterns."""
        d = self.patterns.shape[1]
        self.patterns = np.zeros((0, d))


def create_memory(dimension: int, beta: float = 1.0) -> HopfieldMemory:
    """Create empty Hopfield memory."""
    return HopfieldMemory(
        patterns=np.zeros((0, dimension)),
        beta=beta,
    )


def store_patterns(patterns: List[np.ndarray], beta: float = 1.0) -> HopfieldMemory:
    """Create Hopfield memory with initial patterns."""
    return HopfieldMemory(
        patterns=np.array(patterns),
        beta=beta,
    )


def capacity_estimate(dimension: int, epsilon: float = 0.01) -> int:
    """
    Estimate memory capacity for given dimension.
    
    Modern Hopfield nets have capacity ~ exp(d) / sqrt(d).
    This is exponentially larger than classic Hopfield's 0.14 * d.
    
    Args:
        dimension: Pattern dimension
        epsilon: Error tolerance
        
    Returns:
        Estimated number of patterns that can be reliably stored
    """
    # Rough estimate: exp(d) / (d * log(1/epsilon))
    from math import exp, log
    return int(exp(dimension * 0.1) / (dimension * log(1/epsilon)))


__all__ = [
    "HopfieldMemory",
    "create_memory",
    "store_patterns",
    "capacity_estimate",
]
