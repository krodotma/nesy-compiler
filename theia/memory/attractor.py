"""
Theia Attractor Dynamics — Basin detection and ICL pattern retrieval.

Attractors are fixed points of Hopfield dynamics.
ICL (in-context learning) is modeled as pattern retrieval:
    - In-context examples → store as patterns
    - Query → retrieve via energy minimization
    - Result → interpolated response

This explains why transformers do ICL: attention IS Hopfield retrieval.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from theia.memory.hopfield import HopfieldMemory, store_patterns


@dataclass
class Attractor:
    """
    Identified attractor (fixed point) in energy landscape.
    
    Attributes:
        center: The attractor location
        energy: Energy at attractor (lower = stronger)
        basin_radius: Estimated radius of attraction basin
        pattern_idx: Index of associated stored pattern (if known)
    """
    center: np.ndarray
    energy: float
    basin_radius: float = 0.0
    pattern_idx: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AttractorLandscape:
    """
    Tools for analyzing attractor structure of Hopfield memory.
    
    Provides:
        - Basin of attraction estimation
        - Attractor detection
        - ICL as pattern retrieval
    """
    
    def __init__(self, memory: HopfieldMemory):
        self.memory = memory
        self._attractors: List[Attractor] = []
        self._cached = False
    
    def find_attractors(self, n_samples: int = 100) -> List[Attractor]:
        """
        Find attractors by sampling random initial points.
        
        Uses flow dynamics to converge to attractors,
        then clusters similar endpoints.
        """
        dim = self.memory.patterns.shape[1]
        attractors = []
        
        for i in range(n_samples):
            # Random starting point
            x0 = np.random.randn(dim)
            x0 = x0 / np.linalg.norm(x0)
            
            # Flow to attractor
            x_final, iters, energy = self.memory.retrieve(x0)
            
            # Check if this is a new attractor
            is_new = True
            for att in attractors:
                if np.linalg.norm(x_final - att.center) < 0.1:
                    is_new = False
                    break
            
            if is_new:
                # Identify which pattern this corresponds to
                pattern_idx = self._nearest_pattern(x_final)
                
                attractors.append(Attractor(
                    center=x_final,
                    energy=energy,
                    pattern_idx=pattern_idx,
                ))
        
        self._attractors = attractors
        self._cached = True
        
        return attractors
    
    def _nearest_pattern(self, point: np.ndarray) -> Optional[int]:
        """Find index of nearest stored pattern."""
        if len(self.memory.patterns) == 0:
            return None
        
        distances = np.linalg.norm(self.memory.patterns - point, axis=1)
        return int(np.argmin(distances))
    
    def estimate_basin_radius(
        self, 
        attractor: Attractor,
        n_samples: int = 50,
    ) -> float:
        """
        Estimate radius of basin of attraction.
        
        Samples points at increasing distances from attractor,
        checks if they still converge to same attractor.
        """
        center = attractor.center
        dim = len(center)
        
        # Binary search for basin edge
        low, high = 0.0, 2.0
        
        for _ in range(10):  # ~10 bits of precision
            mid = (low + high) / 2
            
            # Sample points at radius mid
            converge_count = 0
            for _ in range(n_samples):
                direction = np.random.randn(dim)
                direction = direction / np.linalg.norm(direction)
                
                test_point = center + mid * direction
                result, _, _ = self.memory.retrieve(test_point)
                
                if np.linalg.norm(result - center) < 0.1:
                    converge_count += 1
            
            # If most points converge, basin extends further
            if converge_count / n_samples > 0.5:
                low = mid
            else:
                high = mid
        
        radius = (low + high) / 2
        attractor.basin_radius = radius
        return radius
    
    def in_context_learn(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        query: np.ndarray,
        beta_boost: float = 2.0,
    ) -> np.ndarray:
        """
        In-context learning as pattern retrieval.
        
        Given (input, output) examples and a query input,
        retrieves the appropriate output via Hopfield dynamics.
        
        This models how transformers do ICL:
        1. In-context examples stored as patterns
        2. Query triggers retrieval
        3. Softmax attention = Hopfield update
        
        Args:
            examples: List of (input, output) pairs
            query: Query input
            beta_boost: Temperature increase for sharper retrieval
            
        Returns:
            Retrieved output
        """
        if not examples:
            return query  # No context, return query unchanged
        
        # Create temporary memory with example outputs
        outputs = np.array([ex[1] for ex in examples])
        inputs = np.array([ex[0] for ex in examples])
        
        # Compute attention weights based on input similarity
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        inputs_norm = inputs / (np.linalg.norm(inputs, axis=1, keepdims=True) + 1e-8)
        
        # Similarity with temperature
        sims = (inputs_norm @ query_norm) * (self.memory.beta * beta_boost)
        
        # Softmax attention
        weights = np.exp(sims - np.max(sims))
        weights = weights / (np.sum(weights) + 1e-8)
        
        # Weighted combination of outputs (attention output)
        result = weights @ outputs
        
        return result
    
    def get_energy_surface(
        self,
        grid_size: int = 50,
        dims: Tuple[int, int] = (0, 1),
        range_: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 2D slice of energy surface for visualization.
        
        Returns:
            (X, Y, E) grids for plotting
        """
        dim = self.memory.patterns.shape[1]
        
        x = np.linspace(-range_, range_, grid_size)
        y = np.linspace(-range_, range_, grid_size)
        X, Y = np.meshgrid(x, y)
        E = np.zeros_like(X)
        
        for i in range(grid_size):
            for j in range(grid_size):
                point = np.zeros(dim)
                point[dims[0]] = X[i, j]
                point[dims[1]] = Y[i, j]
                E[i, j] = self.memory.energy(point)
        
        return X, Y, E


def create_icl_memory(
    examples: List[Tuple[np.ndarray, np.ndarray]],
    beta: float = 5.0,
) -> Tuple[HopfieldMemory, AttractorLandscape]:
    """
    Create memory and landscape from in-context examples.
    
    Convenience function for ICL pattern retrieval.
    """
    if not examples:
        raise ValueError("Need at least one example")
    
    # Concatenate input-output pairs as patterns
    patterns = [np.concatenate([inp, out]) for inp, out in examples]
    
    memory = store_patterns(patterns, beta=beta)
    landscape = AttractorLandscape(memory)
    
    return memory, landscape


__all__ = [
    "Attractor",
    "AttractorLandscape",
    "create_icl_memory",
]
