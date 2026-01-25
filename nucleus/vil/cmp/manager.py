"""
VIL CMP Integration
Integration with Clade Manager Protocol for evolutionary fitness tracking.

Based on nucleus/tools/clade_manager.py:
- CMP (Code Maturity Potential) tracking with golden ratio weighting
- Clade speciation, evaluation, and merging
- Evolutionary lineage provenance
- Fitness calculation: task_completion * PHI + test_coverage + ...

Integration points:
- Vision CMP → Clade fitness
- Metalearning CMP → Adaptive rates
- Lineage tracking for vision evolution
- CMP-driven ICL buffer management

Version: 1.0
Date: 2026-01-25
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Constants
PHI = 1.618033988749895  # Golden ratio
CMP_DISCOUNT = 1 / PHI  # ~0.618 per generation
GLOBAL_CMP_FLOOR = 0.236  # 1/PHI^3


class CladeState(str, Enum):
    """Clade lifecycle states."""

    ACTIVE = "active"
    CONVERGING = "converging"  # fitness ≥ 0.618
    MERGED = "merged"
    DORMANT = "dormant"
    EXTINCT = "extinct"  # fitness < 0.236


@dataclass
class VisionCMPMetrics:
    """
    Vision-specific CMP metrics.

    Extends base CMP with vision quality tracking:
    - capture_quality: H* based frame quality
    - analysis_confidence: VLM inference confidence
    - icl_diversity: ICL buffer diversity score
    - geometric_stability: Manifold convergence rate
    """

    capture_quality: float = 0.0
    analysis_confidence: float = 0.0
    icl_diversity: float = 0.0
    geometric_stability: float = 0.0

    # Base CMP fields
    task_completion: float = 0.0
    test_coverage: float = 0.0
    bug_rate: float = 0.0
    review_velocity: float = 0.0
    divergence_ratio: float = 0.0

    def calculate_fitness(
        self,
        phi_weighted: bool = True,
    ) -> float:
        """
        Calculate phi-weighted CMP fitness.

        Formula:
        fitness = task_completion * PHI +
                  test_coverage * 1.0 +
                  bug_rate * (1/PHI) +
                  review_velocity * (1/PHI^2) +
                  divergence_ratio * (1/PHI^3)

        Vision extension:
        += capture_quality * PHI +
           analysis_confidence * 1.0 +
           icl_diversity * (1/PHI) +
           geometric_stability * (1/PHI^2)
        """
        base_fitness = (
            self.task_completion * PHI +
            self.test_coverage * 1.0 +
            self.bug_rate * (1 / PHI) +
            self.review_velocity * (1 / PHI ** 2) +
            self.divergence_ratio * (1 / PHI ** 3)
        )

        vision_extension = (
            self.capture_quality * PHI +
            self.analysis_confidence * 1.0 +
            self.icl_diversity * (1 / PHI) +
            self.geometric_stability * (1 / PHI ** 2)
        )

        if phi_weighted:
            return base_fitness + vision_extension
        else:
            return (base_fitness + vision_extension) / (PHI + 3)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "task_completion": self.task_completion,
            "test_coverage": self.test_coverage,
            "bug_rate": self.bug_rate,
            "review_velocity": self.review_velocity,
            "divergence_ratio": self.divergence_ratio,
            "capture_quality": self.capture_quality,
            "analysis_confidence": self.analysis_confidence,
            "icl_diversity": self.icl_diversity,
            "geometric_stability": self.geometric_stability,
            "fitness": self.calculate_fitness(),
        }


@dataclass
class CladeCMP:
    """
    Clade CMP tracking.

    Manages:
    - Clade state and lifecycle
    - CMP metrics over time
    - Lineage tracking
    - Fitness history
    """

    clade_id: str
    parent_id: Optional[str] = None
    generation: int = 0
    state: CladeState = CladeState.ACTIVE

    # CMP metrics
    metrics: VisionCMPMetrics = field(default_factory=VisionCMPMetrics)
    fitness_history: List[float] = field(default_factory=list)

    # Evolutionary parameters
    pressure: float = 1.0  # Selection pressure
    mutation_rate: float = 0.1  # Mutation probability
    phi_weighted: bool = True

    # Lineage
    children: List[str] = field(default_factory=list)
    ancestry: List[str] = field(default_factory=list)

    def calculate_fitness(self) -> float:
        """Calculate current fitness."""
        fitness = self.metrics.calculate_fitness(self.phi_weighted)
        self.fitness_history.append(fitness)
        return fitness

    def update_state(self) -> CladeState:
        """Update clade state based on fitness."""
        fitness = self.calculate_fitness()

        if fitness >= PHI:
            self.state = CladeState.CONVERGING
        elif fitness < GLOBAL_CMP_FLOOR:
            self.state = CladeState.EXTINCT
        elif self.state == CladeState.CONVERGING and fitness < PHI * 0.8:
            self.state = CladeState.ACTIVE

        return self.state

    def add_child(self, child_id: str) -> None:
        """Add child clade."""
        self.children.append(child_id)

    def get_ancestry_string(self) -> str:
        """Get ancestry as string."""
        if not self.ancestry:
            return f"gen{self.generation}"
        return " → ".join([f"gen{i}" for i in self.ancestry[-3:]] + [f"gen{self.generation}"])


class VILCMPManager:
    """
    VIL-specific CMP manager.

    Integrates vision CMP with Clade Manager functionality.

    Features:
    1. Vision CMP tracking
    2. Clade lifecycle management
    3. Evolutionary operations (speciate, merge, extinct)
    4. Lineage tracking
    5. Phi-weighted fitness
    """

    def __init__(
        self,
        phi_weighted: bool = True,
        merge_threshold: float = PHI,  # Merge when fitness difference < PHI
        speciate_threshold: float = 0.1,  # Speciate when fitness delta > threshold
        bus_emitter: Optional[callable] = None,
    ):
        self.phi_weighted = phi_weighted
        self.merge_threshold = merge_threshold
        self.speciate_threshold = speciate_threshold
        self.bus_emitter = bus_emitter

        # Clade storage
        self.clades: Dict[str, CladeCMP] = {}
        self._clade_counter = 0

        # Statistics
        self.stats = {
            "total_clades": 0,
            "active_clades": 0,
            "converging_clades": 0,
            "merged_clades": 0,
            "extinct_clades": 0,
            "avg_fitness": 0.0,
            "max_generation": 0,
        }

    def create_clade(
        self,
        parent_id: Optional[str] = None,
        metrics: Optional[VisionCMPMetrics] = None,
    ) -> CladeCMP:
        """Create new clade with CMP tracking."""
        clade_id = f"clade_{self._clade_counter}"
        self._clade_counter += 1

        # Determine generation
        generation = 0
        ancestry = []
        if parent_id and parent_id in self.clades:
            parent = self.clades[parent_id]
            generation = parent.generation + 1
            ancestry = parent.ancestry + [parent.generation]
            parent.add_child(clade_id)

        clade = CladeCMP(
            clade_id=clade_id,
            parent_id=parent_id,
            generation=generation,
            ancestry=ancestry,
            metrics=metrics or VisionCMPMetrics(),
            phi_weighted=self.phi_weighted,
        )

        self.clades[clade_id] = clade
        self.stats["total_clades"] += 1
        self.stats["active_clades"] += 1

        return clade_id, clade

    def update_clade_cmp(
        self,
        clade_id: str,
        metrics_update: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        """
        Update CMP metrics for clade.

        Args:
            clade_id: Clade to update
            metrics_update: Optional metric updates

        Returns:
            New fitness value
        """
        if clade_id not in self.clades:
            return None

        clade = self.clades[clade_id]

        # Update metrics
        if metrics_update:
            for key, value in metrics_update.items():
                if hasattr(clade.metrics, key):
                    setattr(clade.metrics, key, value)

        # Update state and fitness
        old_state = clade.state
        new_state = clade.update_state()

        # Update stats
        if old_state != new_state:
            self.stats[f"{old_state.value}_clades"] = (
                self.stats.get(f"{old_state.value}_clades", 1) - 1
            )
            self.stats[f"{new_state.value}_clades"] = (
                self.stats.get(f"{new_state.value}_clades", 0) + 1
            )

        # Emit event
        self._emit_cmp_event(clade)

        return clade.calculate_fitness()

    def recommend_merge(
        self,
        clade_ids: Optional[List[str]] = None,
    ) -> Optional[Tuple[str, str]]:
        """
        Recommend clades for merging.

        Merge when:
        1. Both clades are converging (fitness ≥ PHI)
        2. Fitness difference < merge_threshold
        3. Not parent-child relationship

        Args:
            clade_ids: Specific clades to check, or all active

        Returns:
            Tuple of (clade_a, clade_b) to merge, or None
        """
        if clade_ids is None:
            clade_ids = [
                cid for cid, clade in self.clades.items()
                if clade.state in [CladeState.ACTIVE, CladeState.CONVERGING]
            ]

        # Check pairs
        for i, id_a in enumerate(clade_ids):
            for id_b in clade_ids[i+1:]:
                clade_a = self.clades.get(id_a)
                clade_b = self.clades.get(id_b)

                if not clade_a or not clade_b:
                    continue

                # Check relationship
                if id_a in clade_b.ancestry or id_b in clade_a.ancestry:
                    continue  # Don't merge parent-child

                # Check fitness similarity
                fitness_a = clade_a.calculate_fitness()
                fitness_b = clade_b.calculate_fitness()

                if fitness_a >= PHI and fitness_b >= PHI:
                    if abs(fitness_a - fitness_b) < self.merge_threshold:
                        return (id_a, id_b)

        return None

    def merge_clades(
        self,
        clade_a_id: str,
        clade_b_id: str,
    ) -> str:
        """
        Merge two clades into new clade.

        Args:
            clade_a_id: First clade
            clade_b_id: Second clade

        Returns:
            New merged clade ID
        """
        clade_a = self.clades[clade_a_id]
        clade_b = self.clades[clade_b_id]

        # Calculate merged fitness (weighted average)
        fitness_a = clade_a.calculate_fitness()
        fitness_b = clade_b.calculate_fitness()
        merged_fitness = (fitness_a + fitness_b) / 2

        # Create merged metrics
        merged_metrics = VisionCMPMetrics(
            task_completion=(
                clade_a.metrics.task_completion +
                clade_b.metrics.task_completion
            ) / 2,
            capture_quality=(
                clade_a.metrics.capture_quality +
                clade_b.metrics.capture_quality
            ) / 2,
            analysis_confidence=(
                clade_a.metrics.analysis_confidence +
                clade_b.metrics.analysis_confidence
            ) / 2,
        )

        # Create new clade
        new_id, new_clade = self.create_clade(metrics=merged_metrics)

        # Mark old clades as merged
        clade_a.state = CladeState.MERGED
        clade_b.state = CladeState.MERGED

        self.stats["merged_clades"] += 2
        self.stats["active_clades"] -= 1  # Net: -2 +1 = -1

        # Emit merge event
        self._emit_merge_event(clade_a_id, clade_b_id, new_id, merged_fitness)

        return new_id

    def speciate(
        self,
        parent_id: str,
        pressure: float = 1.0,
        mutation_rate: float = 0.1,
    ) -> str:
        """
        Speciate new clade from parent.

        Args:
            parent_id: Parent clade ID
            pressure: Selection pressure
            mutation_rate: Mutation probability

        Returns:
            New clade ID
        """
        if parent_id not in self.clades:
            return None

        parent = self.clades[parent_id]

        # Create mutated metrics
        import random
        mutated_metrics = VisionCMPMetrics(
            task_completion=max(0, min(1, parent.metrics.task_completion + np.random.randn() * mutation_rate)),
            capture_quality=max(0, min(1, parent.metrics.capture_quality + np.random.randn() * mutation_rate)),
            analysis_confidence=max(0, min(1, parent.metrics.analysis_confidence + np.random.randn() * mutation_rate)),
        )

        # Create new clade
        new_id, new_clade = self.create_clade(
            parent_id=parent_id,
            metrics=mutated_metrics,
        )
        new_clade.pressure = pressure
        new_clade.mutation_rate = mutation_rate

        # Emit speciation event
        self._emit_speciate_event(parent_id, new_id)

        return new_id

    def get_clade_info(self, clade_id: str) -> Optional[Dict[str, Any]]:
        """Get information about clade."""
        if clade_id not in self.clades:
            return None

        clade = self.clades[clade_id]
        return {
            "clade_id": clade.clade_id,
            "parent_id": clade.parent_id,
            "generation": clade.generation,
            "state": clade.state.value,
            "ancestry_string": clade.get_ancestry_string(),
            "num_children": len(clade.children),
            "metrics": clade.metrics.to_dict(),
            "fitness": clade.calculate_fitness(),
            "pressure": clade.pressure,
            "mutation_rate": clade.mutation_rate,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get CMP statistics."""
        if not self.clades:
            return self.stats

        fitnesses = [c.calculate_fitness() for c in self.clades.values()]

        return {
            **self.stats,
            "avg_fitness": np.mean(fitnesses) if fitnesses else 0.0,
            "max_fitness": max(fitnesses) if fitnesses else 0.0,
            "min_fitness": min(fitnesses) if fitnesses else 0.0,
        }

    def _emit_cmp_event(self, clade: CladeCMP) -> None:
        """Emit CMP update event."""
        if not self.bus_emitter:
            return

        event = {
            "topic": "vil.cmp.update",
            "data": {
                "clade_id": clade.clade_id,
                "state": clade.state.value,
                "fitness": clade.calculate_fitness(),
                "generation": clade.generation,
                "metrics": clade.metrics.to_dict(),
            },
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[VILCMPManager] Bus emission error: {e}")

    def _emit_merge_event(
        self,
        clade_a: str,
        clade_b: str,
        new_clade: str,
        fitness: float,
    ) -> None:
        """Emit merge event."""
        if not self.bus_emitter:
            return

        event = {
            "topic": "vil.cmp.merge",
            "data": {
                "clade_a": clade_a,
                "clade_b": clade_b,
                "new_clade": new_clade,
                "merged_fitness": fitness,
            },
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[VILCMPManager] Bus emission error: {e}")

    def _emit_speciate_event(
        self,
        parent: str,
        new_clade: str,
    ) -> None:
        """Emit speciation event."""
        if not self.bus_emitter:
            return

        event = {
            "topic": "vil.cmp.speciate",
            "data": {
                "parent": parent,
                "new_clade": new_clade,
            },
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[VILCMPManager] Bus emission error: {e}")


def create_vil_cmp_manager(
    phi_weighted: bool = True,
    bus_emitter: Optional[callable] = None,
) -> VILCMPManager:
    """Create VIL CMP manager with default config."""
    return VILCMPManager(
        phi_weighted=phi_weighted,
        bus_emitter=bus_emitter,
    )


__all__ = [
    "CladeState",
    "VisionCMPMetrics",
    "CladeCMP",
    "VILCMPManager",
    "create_vil_cmp_manager",
    "PHI",
    "CMP_DISCOUNT",
    "GLOBAL_CMP_FLOOR",
]
