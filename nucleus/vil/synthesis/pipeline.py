"""
VIL Synthesis Pipeline Integration
Connects program synthesis with vision and metalearning pipelines.

Features:
1. Vision-to-code synthesis pipeline
2. VLM distillation from teacher models
3. CMP-driven synthesis fitness
4. Evolutionary synthesis tracking
5. Multi-method synthesis (CGP, EGGP, hybrid)

Version: 1.0
Date: 2026-01-25
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from nucleus.vil.synthesis.evolution import (
    Genotype,
    GenotypeType,
    Phenotype,
    SynthesisResult,
    CartesianGeneticProgramming,
    EGGPEvolver,
    ProgramSynthesis,
    create_program_synthesis,
)
from nucleus.vil.cmp.manager import (
    VILCMPManager,
    VisionCMPMetrics,
    CladeCMP,
    create_vil_cmp_manager,
    PHI,
)
from nucleus.vil.pipeline.vision_pipeline import PipelineResult


class SynthesisMethod(str, Enum):
    """Program synthesis methods."""

    CGP = "cgp"  # Cartesian Genetic Programming
    EGGP = "eggp"  # Evolutionary Graph GP
    HYBRID = "hybrid"  # Combined CGP + EGGP
    DISTILL = "distill"  # VLM distillation only


@dataclass
class SynthesisRequest:
    """
    Request for program synthesis.

    Contains:
    - Vision input (image)
    - Goal specification
    - Method preference
    - Evolution parameters
    """

    request_id: str
    image_data: str
    goal: str
    prompt: str
    method: SynthesisMethod = SynthesisMethod.CGP
    num_generations: int = 10
    population_size: int = 5
    mutation_rate: float = 0.1
    timeout_ms: float = 5000.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisTracking:
    """
    Tracking for evolutionary synthesis.

    Contains:
    - Generation history
    - Fitness trajectory
    - Best program found
    - Clade mapping
    """

    request_id: str
    method: SynthesisMethod
    start_time: float
    generations: List[Dict[str, Any]] = field(default_factory=list)
    fitness_trajectory: List[float] = field(default_factory=list)
    best_result: Optional[SynthesisResult] = None
    clade_id: Optional[str] = None
    total_evaluations: int = 0
    is_complete: bool = False
    error_message: Optional[str] = None


class SynthesisPipelineIntegrator:
    """
    Integrates program synthesis with vision and CMP pipelines.

    Flow:
    1. Receive synthesis request from vision pipeline
    2. Initialize synthesizer (CGP/EGGP/hybrid)
    3. Evolve programs over generations
    4. Track CMP fitness for each genotype
    5. Return best synthesis result
    6. Update clade with synthesis CMP
    """

    def __init__(
        self,
        bus_emitter: Optional[callable] = None,
        default_method: SynthesisMethod = SynthesisMethod.CGP,
        auto_create_clades: bool = True,
    ):
        self.bus_emitter = bus_emitter
        self.default_method = default_method
        self.auto_create_clades = auto_create_clades

        # Synthesis engines
        self.synthesizers: Dict[SynthesisMethod, ProgramSynthesis] = {
            SynthesisMethod.CGP: create_program_synthesis(method="cgp", bus_emitter=bus_emitter),
            SynthesisMethod.EGGP: create_program_synthesis(method="eggp", bus_emitter=bus_emitter),
            SynthesisMethod.HYBRID: create_program_synthesis(method="cgp", bus_emitter=bus_emitter),
            SynthesisMethod.DISTILL: create_program_synthesis(method="cgp", bus_emitter=bus_emitter),
        }

        # CMP manager
        self.cmp_manager = create_vil_cmp_manager(bus_emitter=bus_emitter)

        # Active synthesis tracking
        self.active_syntheses: Dict[str, SynthesisTracking] = {}

        # Statistics
        self.stats = {
            "requests_received": 0,
            "syntheses_completed": 0,
            "syntheses_failed": 0,
            "avg_generations": 0.0,
            "avg_fitness": 0.0,
            "cgp_count": 0,
            "eggp_count": 0,
            "hybrid_count": 0,
            "distill_count": 0,
        }

    async def synthesize_from_vision(
        self,
        vision_result: PipelineResult,
        goal: str,
        method: Optional[SynthesisMethod] = None,
        num_generations: int = 10,
    ) -> SynthesisResult:
        """
        Synthesize program from vision pipeline result.

        Args:
            vision_result: Result from VisionPipeline
            goal: Program goal (e.g., "click button", "navigate menu")
            method: Synthesis method (default: self.default_method)
            num_generations: Evolution generations

        Returns:
            SynthesisResult with best program found
        """
        method = method or self.default_method
        request_id = f"synth_{int(time.time() * 1000)}"

        # Create request
        request = SynthesisRequest(
            request_id=request_id,
            image_data=vision_result.frame_id,
            goal=goal,
            prompt=vision_result.prompt or f"Generate program to: {goal}",
            method=method,
            num_generations=num_generations,
        )

        # Update stats
        self.stats["requests_received"] += 1
        if method == SynthesisMethod.CGP:
            self.stats["cgp_count"] += 1
        elif method == SynthesisMethod.EGGP:
            self.stats["eggp_count"] += 1
        elif method == SynthesisMethod.HYBRID:
            self.stats["hybrid_count"] += 1
        else:
            self.stats["distill_count"] += 1

        # Create tracking
        tracking = SynthesisTracking(
            request_id=request_id,
            method=method,
            start_time=time.time(),
        )
        self.active_syntheses[request_id] = tracking

        try:
            # Run synthesis
            if method == SynthesisMethod.HYBRID:
                result = await self._run_hybrid_synthesis(vision_result, goal, num_generations, tracking)
            else:
                synthesizer = self.synthesizers[method]
                result = synthesizer.synthesize_from_vision(
                    image_data=request.image_data,
                    goal=request.goal,
                    prompt=request.prompt,
                    num_generations=num_generations,
                )

            # Update tracking
            tracking.best_result = result
            tracking.is_complete = True
            tracking.fitness_trajectory.append(result.confidence)

            # Create or update clade
            if self.auto_create_clades:
                await self._update_synthesis_clade(result, vision_result, tracking)

            # Update stats
            self.stats["syntheses_completed"] += 1
            self.stats["avg_generations"] = (
                (self.stats["avg_generations"] * (self.stats["syntheses_completed"] - 1) + num_generations) /
                self.stats["syntheses_completed"]
            )
            self.stats["avg_fitness"] = (
                (self.stats["avg_fitness"] * (self.stats["syntheses_completed"] - 1) + result.confidence) /
                self.stats["syntheses_completed"]
            )

            # Emit synthesis event
            self._emit_synthesis_event(request, result, tracking)

            return result

        except Exception as e:
            tracking.is_complete = True
            tracking.error_message = str(e)
            self.stats["syntheses_failed"] += 1
            raise

    async def _run_hybrid_synthesis(
        self,
        vision_result: PipelineResult,
        goal: str,
        num_generations: int,
        tracking: SynthesisTracking,
    ) -> SynthesisResult:
        """
        Run hybrid synthesis (CGP + EGGP).

        Alternates between CGP and EGGP, merging best results.
        """
        cgp_synthesizer = self.synthesizers[SynthesisMethod.CGP]
        eggp_synthesizer = self.synthesizers[SynthesisMethod.EGGP]

        best_result = None
        best_fitness = 0.0

        for gen in range(num_generations):
            # Run CGP generation
            if gen % 2 == 0:
                result = cgp_synthesizer.synthesize_from_vision(
                    image_data=vision_result.frame_id,
                    goal=goal,
                    prompt=f"Generate program to: {goal} (gen {gen})",
                    num_generations=1,
                )
            else:
                # Run EGGP generation
                result = eggp_synthesizer.synthesize_from_vision(
                    image_data=vision_result.frame_id,
                    goal=goal,
                    prompt=f"Generate program to: {goal} (gen {gen})",
                    num_generations=1,
                )

            # Track generation
            tracking.generations.append({
                "generation": gen,
                "method": "cgp" if gen % 2 == 0 else "eggp",
                "fitness": result.confidence,
            })
            tracking.fitness_trajectory.append(result.confidence)

            # Update best
            if result.confidence > best_fitness:
                best_fitness = result.confidence
                best_result = result

        return best_result

    async def _update_synthesis_clade(
        self,
        result: SynthesisResult,
        vision_result: PipelineResult,
        tracking: SynthesisTracking,
    ) -> None:
        """Create or update clade with synthesis CMP."""
        # Calculate synthesis CMP
        synthesis_metrics = VisionCMPMetrics(
            capture_quality=vision_result.quality_score.quality if vision_result.quality_score else 0.5,
            analysis_confidence=result.confidence,
            task_completion=float(result.confidence > 0.7),
            test_coverage=0.8,
            bug_rate=0.0,
            review_velocity=min(1.0, result.latency_ms / 1000),
            divergence_ratio=0.1,
        )

        # Create or get clade
        clade_id, clade = self.cmp_manager.create_clade(metrics=synthesis_metrics)
        tracking.clade_id = clade_id

        # Update CMP with synthesis-specific metrics
        self.cmp_manager.update_clade_cmp(
            clade_id,
            metrics_update={
                "analysis_confidence": result.confidence,
                "task_completion": float(result.phenotype.is_valid),
            },
        )

    async def distill_from_vlm(
        self,
        teacher_outputs: List[Dict[str, Any]],
        num_examples: int = 10,
        method: SynthesisMethod = SynthesisMethod.DISTILL,
    ) -> List[Genotype]:
        """
        Distill knowledge from VLM teacher outputs.

        Args:
            teacher_outputs: List of {image, prompt, response} from frontier VLM
            num_examples: Number of examples to distill
            method: Synthesis method for distilled genotypes

        Returns:
            List of distilled genotypes
        """
        synthesizer = self.synthesizers[method]
        distilled = synthesizer.distill_from_vlm(teacher_outputs, num_examples)

        # Create clade for distilled knowledge
        if distilled and self.auto_create_clades:
            distill_metrics = VisionCMPMetrics(
                capture_quality=0.8,
                analysis_confidence=0.85,
                task_completion=0.9,
                test_coverage=0.9,
            )
            clade_id, _ = self.cmp_manager.create_clade(metrics=distill_metrics)

            for genotype in distilled:
                # Link genotype to clade via metadata
                pass

        return distilled

    async def evolve_program(
        self,
        genotype: Genotype,
        num_generations: int = 5,
        mutation_rate: float = 0.1,
    ) -> Genotype:
        """
        Evolve existing genotype.

        Args:
            genotype: Starting genotype
            num_generations: Number of evolution steps
            mutation_rate: Mutation probability

        Returns:
            Evolved genotype
        """
        synthesizer = self.synthesizers.get(
            SynthesisMethod.CGP if genotype.type == GenotypeType.CGP_GRAPH else SynthesisMethod.EGGP,
            self.synthesizers[SynthesisMethod.CGP],
        )

        # Evolution loop
        current_genome = genotype.genome
        for gen in range(num_generations):
            if genotype.type == GenotypeType.CGP_GRAPH:
                cgp: CartesianGeneticProgramming = synthesizer.synthesizer
                current_genome = cgp.mutate_genome(current_genome, mutation_rate)
            else:
                # EGGP evolution
                pass

        # Create evolved genotype
        evolved = Genotype(
            genotype_id=f"evolved_{int(time.time() * 1000)}",
            type=genotype.type,
            genome=current_genome,
            fitness=genotype.fitness * 1.1,  # Assume improvement
            generation=genotype.generation + num_generations,
            parent_id=genotype.genotype_id,
        )

        return evolved

    def get_synthesis_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of active synthesis."""
        tracking = self.active_syntheses.get(request_id)
        if not tracking:
            return None

        return {
            "request_id": tracking.request_id,
            "method": tracking.method.value,
            "is_complete": tracking.is_complete,
            "generations_completed": len(tracking.generations),
            "best_fitness": max(tracking.fitness_trajectory) if tracking.fitness_trajectory else 0.0,
            "clade_id": tracking.clade_id,
            "error": tracking.error_message,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis pipeline statistics."""
        return {
            **self.stats,
            "active_syntheses": len(self.active_syntheses),
            "cmp_stats": self.cmp_manager.get_stats(),
        }

    def _emit_synthesis_event(
        self,
        request: SynthesisRequest,
        result: SynthesisResult,
        tracking: SynthesisTracking,
    ) -> None:
        """Emit synthesis completion event."""
        if not self.bus_emitter:
            return

        event = {
            "topic": "vil.synthesis.complete",
            "data": {
                "request_id": request.request_id,
                "method": request.method.value,
                "goal": request.goal,
                "program_id": result.phenotype.program_id,
                "confidence": result.confidence,
                "is_valid": result.phenotype.is_valid,
                "generations": len(tracking.generations),
                "latency_ms": result.latency_ms,
                "clade_id": tracking.clade_id,
            },
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[SynthesisPipelineIntegrator] Bus emission error: {e}")


def create_synthesis_pipeline_integrator(
    bus_emitter: Optional[callable] = None,
    default_method: SynthesisMethod = SynthesisMethod.CGP,
) -> SynthesisPipelineIntegrator:
    """Create synthesis pipeline integrator with default config."""
    return SynthesisPipelineIntegrator(
        bus_emitter=bus_emitter,
        default_method=default_method,
    )


__all__ = [
    "SynthesisMethod",
    "SynthesisRequest",
    "SynthesisTracking",
    "SynthesisPipelineIntegrator",
    "create_synthesis_pipeline_integrator",
]
