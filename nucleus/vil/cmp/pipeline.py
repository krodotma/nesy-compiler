"""
VIL CMP Pipeline Integration
Connects CMP tracking with vision and metalearning pipelines.

Features:
1. Vision CMP extraction from vision pipeline results
2. Metalearning CMP from training results
3. CMP-driven ICL buffer management
4. Evolutionary operations (speciate, merge)
5. Lineage tracking for vision evolution

Version: 1.0
Date: 2026-01-25
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from nucleus.vil.cmp.manager import (
    VILCMPManager,
    VisionCMPMetrics,
    CladeCMP,
    CladeState,
    create_vil_cmp_manager,
    PHI,
)
from nucleus.vil.pipeline.vision_pipeline import PipelineResult
from nucleus.vil.pipeline.metalearning_adapter import MetalearningResult


@dataclass
class CMPIntegrationResult:
    """
    Result from CMP integration with vision/metalearning.

    Contains:
    - Clade ID
    - CMP metrics
    - Fitness score
    - State transition
    - Operations performed
    """

    clade_id: Optional[str] = None
    metrics: Optional[VisionCMPMetrics] = None
    fitness: float = 0.0
    old_state: Optional[CladeState] = None
    new_state: Optional[CladeState] = None
    operations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clade_id": self.clade_id,
            "fitness": self.fitness,
            "old_state": self.old_state.value if self.old_state else None,
            "new_state": self.new_state.value if self.new_state else None,
            "operations": self.operations,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


class CMPPipelineIntegrator:
    """
    Integrates CMP tracking with vision and metalearning pipelines.

    Flow:
    1. Extract CMP from vision pipeline result
    2. Extract CMP from metalearning result
    3. Update/create clade
    4. Perform evolutionary operations
    5. Drive ICL buffer management
    """

    def __init__(
        self,
        bus_emitter: Optional[callable] = None,
        auto_evolve: bool = True,  # Auto-speciate/merge
    ):
        self.bus_emitter = bus_emitter
        self.auto_evolve = auto_evolve

        # CMP manager
        self.cmp_manager = create_vil_cmp_manager(bus_emitter=bus_emitter)

        # Vision Clade mapping (frame_id → clade_id)
        self.vision_clades: Dict[str, str] = {}

        # Statistics
        self.stats = {
            "vision_processed": 0,
            "metalearning_processed": 0,
            "clades_created": 0,
            "speciations": 0,
            "merges": 0,
            "avg_fitness": 0.0,
        }

    def extract_vision_cmp(
        self,
        vision_result: PipelineResult,
    ) -> VisionCMPMetrics:
        """
        Extract CMP metrics from vision pipeline result.

        Args:
            vision_result: Result from VisionPipeline

        Returns:
            VisionCMPMetrics
        """
        # Extract vision-specific metrics
        capture_quality = (
            vision_result.quality_score.quality if vision_result.quality_score else 0.5
        )
        analysis_confidence = vision_result.vlm_confidence
        icl_diversity = (
            vision_result.icl_buffer_size / 5.0  # Normalize to [0, 1]
            if vision_result.icl_buffer_size else 0.0
        )
        geometric_stability = 1.0 - (
            vision_result.entropy.novelty_score if vision_result.entropy else 0.0
        )

        # Base metrics (simulated for vision-only)
        task_completion = float(analysis_confidence > 0.7)
        test_coverage = 0.8  # Assume good coverage
        bug_rate = 0.0  # No bugs in vision
        review_velocity = min(1.0, capture_quality)
        divergence_ratio = 0.1

        return VisionCMPMetrics(
            capture_quality=capture_quality,
            analysis_confidence=analysis_confidence,
            icl_diversity=icl_diversity,
            geometric_stability=geometric_stability,
            task_completion=task_completion,
            test_coverage=test_coverage,
            bug_rate=bug_rate,
            review_velocity=review_velocity,
            divergence_ratio=divergence_ratio,
        )

    def extract_metalearning_cmp(
        self,
        ml_result: MetalearningResult,
    ) -> VisionCMPMetrics:
        """
        Extract CMP metrics from metalearning result.

        Args:
            ml_result: Result from MetalearningAdapter

        Returns:
            VisionCMPMetrics
        """
        # Extract metalearning-specific metrics
        task_completion = ml_result.accuracy
        test_coverage = ml_result.accuracy  # Use accuracy as proxy
        bug_rate = 1.0 - ml_result.accuracy  # Error rate
        review_velocity = min(1.0, ml_result.iterations / 100)
        divergence_ratio = abs(ml_result.inner_loss - ml_result.outer_loss) if ml_result.outer_loss > 0 else 0.1

        # Vision extensions (simulated)
        capture_quality = 0.8  # Assume good quality
        analysis_confidence = ml_result.accuracy
        icl_diversity = 0.7  # Assume good diversity
        geometric_stability = 1.0 - ml_result.inner_loss

        return VisionCMPMetrics(
            capture_quality=capture_quality,
            analysis_confidence=analysis_confidence,
            icl_diversity=icl_diversity,
            geometric_stability=geometric_stability,
            task_completion=task_completion,
            test_coverage=test_coverage,
            bug_rate=bug_rate,
            review_velocity=review_velocity,
            divergence_ratio=divergence_ratio,
        )

    async def process_vision_result(
        self,
        vision_result: PipelineResult,
        create_new_clade: bool = True,
    ) -> CMPIntegrationResult:
        """
        Process vision pipeline result with CMP tracking.

        Args:
            vision_result: Result from VisionPipeline
            create_new_clade: Create new clade if none exists

        Returns:
            CMPIntegrationResult
        """
        result = CMPIntegrationResult()

        # Extract CMP
        metrics = self.extract_vision_cmp(vision_result)
        result.metrics = metrics

        # Get or create clade
        frame_id = vision_result.frame_id
        clade_id = self.vision_clades.get(frame_id)

        if clade_id is None and create_new_clade:
            # Create new clade for this vision lineage
            clade_id, clade = self.cmp_manager.create_clade(metrics=metrics)
            self.vision_clades[frame_id] = clade_id
            result.clade_id = clade_id
            result.operations.append("create_clade")
            self.stats["clades_created"] += 1
        elif clade_id:
            # Update existing clade
            fitness = self.cmp_manager.update_clade_cmp(
                clade_id,
                metrics_update={
                    "capture_quality": metrics.capture_quality,
                    "analysis_confidence": metrics.analysis_confidence,
                    "icl_diversity": metrics.icl_diversity,
                    "geometric_stability": metrics.geometric_stability,
                },
            )
            result.clade_id = clade_id
            result.operations.append("update_clade")

        # Get fitness and state
        if result.clade_id:
            clade = self.cmp_manager.clades.get(result.clade_id)
            if clade:
                result.fitness = clade.calculate_fitness()
                result.new_state = clade.state

        # Evolutionary operations
        if self.auto_evolve and result.clade_id:
            await self._evolve_clade(result.clade_id, result.operations)

        # Update stats
        self.stats["vision_processed"] += 1
        self.stats["avg_fitness"] = (
            (self.stats["avg_fitness"] * (self.stats["vision_processed"] - 1) + result.fitness) /
            self.stats["vision_processed"]
        )

        return result

    async def process_metalearning_result(
        self,
        ml_result: MetalearningResult,
    ) -> CMPIntegrationResult:
        """
        Process metalearning result with CMP tracking.

        Args:
            ml_result: Result from MetalearningAdapter

        Returns:
            CMPIntegrationResult
        """
        result = CMPIntegrationResult()

        # Extract CMP
        metrics = self.extract_metalearning_cmp(ml_result)
        result.metrics = metrics

        # Update or create clade for this task
        clade_id, clade = self.cmp_manager.create_clade(metrics=metrics)
        result.clade_id = clade_id
        result.operations.append("create_clade")
        self.stats["clades_created"] += 1

        # Get fitness and state
        result.fitness = clade.calculate_fitness()
        result.new_state = clade.state

        # Evolutionary operations
        if self.auto_evolve:
            await self._evolve_clade(clade_id, result.operations)

        # Update stats
        self.stats["metalearning_processed"] += 1

        return result

    async def _evolve_clade(
        self,
        clade_id: str,
        operations: List[str],
    ) -> None:
        """Perform evolutionary operations on clade."""
        clade = self.cmp_manager.clades.get(clade_id)
        if not clade:
            return

        # Check for merge
        merge_pair = self.cmp_manager.recommend_merge()
        if merge_pair:
            merged_id = self.cmp_manager.merge_clades(*merge_pair)
            operations.append("merge")
            self.stats["merges"] += 1
            return

        # Check for speciation (if high fitness)
        if clade.calculate_fitness() > PHI:
            new_id = self.cmp_manager.speciate(clade_id)
            operations.append("speciate")
            self.stats["speciations"] += 1
            return

    def get_icl_strategy_from_cmp(
        self,
        clade_id: str,
    ) -> str:
        """
        Get ICL selection strategy based on clade CMP.

        Mapping:
        - High fitness (> PHI): Use "novel" (explore)
        - Medium fitness: Use "diverse" (balanced)
        - Low fitness: Use "recent" (exploit known good)

        Args:
            clade_id: Clade ID

        Returns:
            Strategy name
        """
        clade = self.cmp_manager.clades.get(clade_id)
        if not clade:
            return "recent"

        fitness = clade.calculate_fitness()

        if fitness > PHI:
            return "novel"  # Explore
        elif fitness > 0.8:
            return "diverse"  # Balanced
        else:
            return "recent"  # Exploit

    def get_stats(self) -> Dict[str, Any]:
        """Get CMP integration statistics."""
        return {
            **self.stats,
            "cmp_stats": self.cmp_manager.get_stats(),
            "vision_clades": len(self.vision_clades),
        }


def create_cmp_pipeline_integrator(
    bus_emitter: Optional[callable] = None,
    auto_evolve: bool = True,
) -> CMPPipelineIntegrator:
    """Create CMP pipeline integrator with default config."""
    return CMPPipelineIntegrator(
        bus_emitter=bus_emitter,
        auto_evolve=auto_evolve,
    )


__all__ = [
    "CMPIntegrationResult",
    "CMPPipelineIntegrator",
    "create_cmp_pipeline_integrator",
]
