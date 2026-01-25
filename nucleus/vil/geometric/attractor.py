"""
VIL Attractor Dynamics
Energy landscape and attractor basin computation for vision-metalearning.

Based on Modern Hopfield Continuum (L5 mHC) from Learning Tower v1.0:
- Energy function minimization
- Attractor basin identification
- Convergence rate calculation
- Hopfield network dynamics

Integrates with:
- Vision embeddings → Energy landscape
- Metalearning gradients → Attractor updates
- CMP tracking → Basin quality metrics

Version: 1.0
Date: 2026-01-25
"""

import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class BasinType(str, Enum):
    """Types of attractor basins."""

    STABLE = "stable"  # Deep, wide basin
    METASTABLE = "metastable"  # Shallow basin
    SADDLE = "saddle"  # Saddle point
    UNSTABLE = "unstable"  # Repeller


@dataclass
class AttractorBasin:
    """
    Attractor basin in energy landscape.

    Contains:
    - Basin center (attractor point)
    - Basin ID
    - Basin type
    - Depth (energy value)
    - Width (radius of attraction)
    - States in basin
    """

    basin_id: str
    center: np.ndarray
    energy: float
    basin_type: BasinType
    width: float = 0.0
    states: List[np.ndarray] = field(default_factory=list)
    convergence_rate: float = 0.0
    quality_score: float = 0.0

    def add_state(self, state: np.ndarray) -> None:
        """Add state to basin."""
        self.states.append(state)

    def compute_width(self) -> float:
        """Compute basin width as average distance to center."""
        if not self.states:
            return 0.0

        distances = [
            np.linalg.norm(state - self.center)
            for state in self.states
        ]
        self.width = float(np.mean(distances))
        return self.width

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "basin_id": self.basin_id,
            "energy": self.energy,
            "basin_type": self.basin_type.value,
            "width": self.width,
            "num_states": len(self.states),
            "convergence_rate": self.convergence_rate,
            "quality_score": self.quality_score,
        }


@dataclass
class ConvergenceResult:
    """
    Result of attractor convergence.

    Contains:
    - Final state (attractor)
    - Initial state
    - Energy trajectory
    - Iterations to converge
    - Converged flag
    """

    initial_state: np.ndarray
    final_state: np.ndarray
    basin_id: Optional[str]
    energy_trajectory: List[float]
    iterations: int
    converged: bool
    convergence_rate: float = 0.0
    distance_traveled: float = 0.0


class AttractorDynamics:
    """
    Attractor dynamics on energy landscapes.

    Features:
    1. Energy function minimization
    2. Basin identification and tracking
    3. Convergence rate calculation
    4. Hopfield-style update rules
    """

    def __init__(
        self,
        dimension: int = 64,
        beta: float = 5.0,  # Inverse temperature for Hopfield
        convergence_threshold: float = 1e-6,
        max_iterations: int = 1000,
    ):
        self.dimension = dimension
        self.beta = beta
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations

        # Basin storage
        self.basins: Dict[str, AttractorBasin] = {}
        self._basin_counter = 0

        # Statistics
        self.stats = {
            "total_convergences": 0,
            "successful_convergences": 0,
            "failed_convergences": 0,
            "avg_iterations": 0.0,
            "avg_convergence_rate": 0.0,
            "num_basins": 0,
        }

    def create_energy_function(
        self,
        patterns: List[np.ndarray],
    ) -> Callable[[np.ndarray], float]:
        """
        Create Hopfield energy function from patterns.

        Energy: E = -0.5 * Σ_i Σ_j w_ij * s_i * s_j
        where w = Σ_k ξ_k^T ξ_k (pattern correlations)

        Args:
            patterns: Stored patterns for Hopfield network

        Returns:
            Energy function E(x)
        """
        if not patterns:
            return lambda x: -np.linalg.norm(x)

        # Create weight matrix
        d = len(patterns[0])
        W = np.zeros((d, d))

        for pattern in patterns:
            pattern = pattern / (np.linalg.norm(pattern) + 1e-8)
            W += np.outer(pattern, pattern)

        W /= len(patterns)

        def energy(x: np.ndarray) -> float:
            x_norm = x / (np.linalg.norm(x) + 1e-8)
            return float(-0.5 * x_norm @ W @ x_norm)

        return energy

    def find_attractor(
        self,
        initial_state: np.ndarray,
        energy_fn: Callable[[np.ndarray], float],
        learning_rate: float = 0.1,
    ) -> ConvergenceResult:
        """
        Find attractor using gradient descent on energy.

        Args:
            initial_state: Starting point
            energy_fn: Energy function to minimize
            learning_rate: Step size for gradient descent

        Returns:
            ConvergenceResult with trajectory
        """
        state = initial_state.copy()
        trajectory = [energy_fn(state)]
        iterations = 0
        converged = False
        convergence_rate = 0.0

        for i in range(self.max_iterations):
            # Compute gradient (numerical)
            grad = self._compute_gradient(state, energy_fn)

            # Update state
            new_state = state - learning_rate * grad

            # Normalize to prevent drift
            norm = np.linalg.norm(new_state)
            if norm > 0:
                new_state = new_state / norm

            # Check convergence
            delta = np.linalg.norm(new_state - state)
            energy_delta = abs(trajectory[-1] - energy_fn(new_state))

            state = new_state
            trajectory.append(energy_fn(state))
            iterations += 1

            if delta < self.convergence_threshold or energy_delta < self.convergence_threshold:
                converged = True
                convergence_rate = 1.0 / (iterations + 1)
                break

        # Distance traveled
        distance_traveled = np.linalg.norm(state - initial_state)

        # Find or create basin
        basin_id = self._find_or_create_basin(state, energy_fn(state))

        # Update stats
        self.stats["total_convergences"] += 1
        if converged:
            self.stats["successful_convergences"] += 1
        else:
            self.stats["failed_convergences"] += 1

        self.stats["avg_iterations"] = (
            (self.stats["avg_iterations"] * (self.stats["total_convergences"] - 1) + iterations) /
            self.stats["total_convergences"]
        )

        return ConvergenceResult(
            initial_state=initial_state,
            final_state=state,
            basin_id=basin_id,
            energy_trajectory=trajectory,
            iterations=iterations,
            converged=converged,
            convergence_rate=convergence_rate,
            distance_traveled=distance_traveled,
        )

    def hopfield_update(
        self,
        state: np.ndarray,
        patterns: List[np.ndarray],
        num_steps: int = 10,
    ) -> np.ndarray:
        """
        Hopfield network update rule.

        s_i(t+1) = sign(Σ_j w_ij * s_j(t))
        where w = Σ_k ξ_k^T ξ_k

        Args:
            state: Current state
            patterns: Stored patterns
            num_steps: Number of update steps

        Returns:
            Updated state
        """
        # Create weight matrix
        d = len(state)
        W = np.zeros((d, d))

        for pattern in patterns:
            pattern = pattern / (np.linalg.norm(pattern) + 1e-8)
            W += np.outer(pattern, pattern)

        W /= len(patterns)

        # Apply Hopfield update
        current = state.copy()
        for _ in range(num_steps):
            # Compute activation
            activation = W @ current
            # Binary threshold
            current = np.sign(activation)
            # Handle zero case
            current[current == 0] = 1

        return current

    def classify_basin(
        self,
        basin: AttractorBasin,
    ) -> BasinType:
        """
        Classify basin type based on properties.

        Criteria:
        - Stable: wide (> threshold), deep energy
        - Metastable: narrow, moderate depth
        - Saddle: moderate width, energy near average
        - Unstable: very narrow, high energy
        """
        if basin.width > 0.5 and basin.energy < -0.5:
            return BasinType.STABLE
        elif basin.width < 0.2:
            return BasinType.UNSTABLE
        elif abs(basin.energy) < 0.3:
            return BasinType.SADDLE
        else:
            return BasinType.METASTABLE

    def _find_or_create_basin(
        self,
        state: np.ndarray,
        energy: float,
    ) -> str:
        """Find existing basin or create new one."""
        # Check for nearby basins
        for basin_id, basin in self.basins.items():
            distance = np.linalg.norm(state - basin.center)
            if distance < basin.width * 0.5:
                # Add to existing basin
                basin.add_state(state)
                basin.compute_width()
                return basin_id

        # Create new basin
        basin_id = f"basin_{self._basin_counter}"
        self._basin_counter += 1

        basin_type = self._classify_basin(
            AttractorBasin(
                basin_id=basin_id,
                center=state,
                energy=energy,
                basin_type=BasinType.METASTABLE,
            )
        )

        basin = AttractorBasin(
            basin_id=basin_id,
            center=state,
            energy=energy,
            basin_type=basin_type,
        )
        basin.add_state(state)
        basin.compute_width()

        self.basins[basin_id] = basin
        self.stats["num_basins"] = len(self.basins)

        return basin_id

    def _compute_gradient(
        self,
        state: np.ndarray,
        energy_fn: Callable[[np.ndarray], float],
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """Compute numerical gradient of energy function."""
        grad = np.zeros_like(state)
        base_energy = energy_fn(state)

        for i in range(len(state)):
            delta = np.zeros_like(state)
            delta[i] = epsilon

            energy_plus = energy_fn(state + delta)
            grad[i] = (energy_plus - base_energy) / epsilon

        return grad

    def get_basin_info(self, basin_id: str) -> Optional[Dict[str, Any]]:
        """Get information about specific basin."""
        if basin_id not in self.basins:
            return None
        return self.basins[basin_id].to_dict()

    def get_stats(self) -> Dict[str, Any]:
        """Get attractor dynamics statistics."""
        return {
            **self.stats,
            "basin_breakdown": {
                basin_type.value: sum(
                    1 for b in self.basins.values()
                    if b.basin_type == basin_type
                )
                for basin_type in BasinType
            },
        }


def create_attractor_dynamics(
    dimension: int = 64,
    beta: float = 5.0,
) -> AttractorDynamics:
    """Create attractor dynamics with default config."""
    return AttractorDynamics(
        dimension=dimension,
        beta=beta,
    )


__all__ = [
    "BasinType",
    "AttractorBasin",
    "ConvergenceResult",
    "AttractorDynamics",
    "create_attractor_dynamics",
]
