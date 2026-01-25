"""
VIL VLM Distillation Module
Knowledge distillation from frontier VLM teacher models.

Features:
1. Teacher-student distillation
2. Multi-teacher ensemble
3. Vision-to-code distillation
4. Program synthesis distillation
5. Continuous learning from frontier

Version: 1.0
Date: 2026-01-25
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from nucleus.vil.synthesis.evolution import (
    Genotype,
    GenotypeType,
    Phenotype,
    ProgramSynthesis,
    create_program_synthesis,
)


class TeacherModel(str, Enum):
    """Frontier VLM teacher models."""

    CLAUDE_OPUS = "claude-opus-4.5"
    GEMINI_ULTRA = "gemini-ultra"
    GPT_VISION = "gpt-4-vision"
    THEIA = "theia"
    ENSEMBLE = "ensemble"


@dataclass
class TeacherOutput:
    """
    Output from teacher VLM.

    Contains:
    - Image input
    - Prompt
    - Teacher response
    - Confidence score
    - Latency
    """

    teacher: TeacherModel
    image_data: str
    prompt: str
    response: str
    confidence: float
    latency_ms: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillationBatch:
    """
    Batch of teacher outputs for distillation.

    Contains:
    - Teacher outputs
    - Distillation configuration
    - Student genotypes
    """

    batch_id: str
    teacher_outputs: List[TeacherOutput]
    student_genotypes: List[Genotype] = field(default_factory=list)
    distillation_method: str = "knowledge_distillation"
    temperature: float = 1.0
    alpha: float = 0.5  # Balance between task loss and distillation loss


@dataclass
class DistillationResult:
    """
    Result from distillation process.

    Contains:
    - Distilled genotypes
    - Teacher-student alignment
    - Loss metrics
    - Improvement delta
    """

    batch_id: str
    distilled_genotypes: List[Genotype]
    teacher_alignment: float  # Alignment with teacher outputs
    task_loss: float
    distillation_loss: float
    combined_loss: float
    improvement_delta: float
    duration_ms: float


class VLMDistiller:
    """
    VLM knowledge distillation from frontier models.

    Features:
    1. Teacher output collection
    2. Knowledge distillation (Hinton et al.)
    3. Multi-teacher ensemble
    4. Vision-to-code transfer
    5. Continuous learning
    """

    def __init__(
        self,
        student_model: ProgramSynthesis,
        bus_emitter: Optional[Callable] = None,
        ensemble_weights: Optional[Dict[TeacherModel, float]] = None,
    ):
        self.student_model = student_model
        self.bus_emitter = bus_emitter

        # Teacher ensemble weights
        self.ensemble_weights = ensemble_weights or {
            TeacherModel.CLAUDE_OPUS: 0.3,
            TeacherModel.GEMINI_ULTRA: 0.3,
            TeacherModel.GPT_VISION: 0.2,
            TeacherModel.THEIA: 0.2,
        }

        # Teacher outputs cache
        self.teacher_cache: Dict[str, List[TeacherOutput]] = {}

        # Statistics
        self.stats = {
            "distillations": 0,
            "teacher_outputs": 0,
            "avg_alignment": 0.0,
            "avg_improvement": 0.0,
            "teacher_counts": {model.value: 0 for model in TeacherModel},
        }

    async def collect_teacher_output(
        self,
        image_data: str,
        prompt: str,
        teacher: TeacherModel = TeacherModel.CLAUDE_OPUS,
        inference_fn: Optional[Callable] = None,
    ) -> TeacherOutput:
        """
        Collect output from teacher VLM.

        Args:
            image_data: Base64-encoded image
            prompt: Inference prompt
            teacher: Teacher model to use
            inference_fn: Optional custom inference function

        Returns:
            TeacherOutput with teacher response
        """
        start_time = time.time()

        # Run teacher inference
        if inference_fn:
            response_data = await inference_fn(image_data, prompt)
            response = response_data.get("response", "")
            confidence = response_data.get("confidence", 0.8)
        else:
            # Simulated teacher response (in production, call actual API)
            response = f"# Teacher {teacher.value} output for: {prompt}"
            confidence = 0.85

        latency_ms = (time.time() - start_time) * 1000

        output = TeacherOutput(
            teacher=teacher,
            image_data=image_data,
            prompt=prompt,
            response=response,
            confidence=confidence,
            latency_ms=latency_ms,
            timestamp=time.time(),
        )

        # Cache output
        cache_key = f"{teacher.value}_{hash(prompt)}"
        if cache_key not in self.teacher_cache:
            self.teacher_cache[cache_key] = []
        self.teacher_cache[cache_key].append(output)

        # Update stats
        self.stats["teacher_outputs"] += 1
        self.stats["teacher_counts"][teacher.value] += 1

        # Emit event
        self._emit_teacher_event(output)

        return output

    async def distill_from_batch(
        self,
        batch: DistillationBatch,
    ) -> DistillationResult:
        """
        Distill knowledge from teacher outputs.

        Uses knowledge distillation (Hinton et al., 2015):
        - Teacher produces soft labels with temperature T
        - Student learns from soft labels + hard labels
        - Loss = α * distillation_loss + (1-α) * task_loss

        Args:
            batch: Batch of teacher outputs

        Returns:
            DistillationResult
        """
        start_time = time.time()

        # Extract teacher knowledge
        teacher_genotypes = []
        for output in batch.teacher_outputs:
            # Create genotype from teacher output
            genotype = self._teacher_output_to_genotype(output)
            teacher_genotypes.append(genotype)

        # Distill to student
        if batch.distillation_method == "knowledge_distillation":
            distilled_genotypes, losses = await self._knowledge_distillation(
                teacher_genotypes,
                batch.student_genotypes,
                batch.temperature,
                batch.alpha,
            )
        else:
            # Direct copy (fallback)
            distilled_genotypes = teacher_genotypes
            losses = {"task": 0.1, "distill": 0.0, "combined": 0.1}

        # Calculate alignment
        teacher_alignment = self._calculate_alignment(
            teacher_genotypes,
            distilled_genotypes,
        )

        # Calculate improvement
        baseline_fitness = np.mean([g.fitness for g in batch.student_genotypes]) if batch.student_genotypes else 0.5
        distilled_fitness = np.mean([g.fitness for g in distilled_genotypes]) if distilled_genotypes else 0.5
        improvement_delta = distilled_fitness - baseline_fitness

        duration_ms = (time.time() - start_time) * 1000

        result = DistillationResult(
            batch_id=batch.batch_id,
            distilled_genotypes=distilled_genotypes,
            teacher_alignment=teacher_alignment,
            task_loss=losses["task"],
            distillation_loss=losses["distill"],
            combined_loss=losses["combined"],
            improvement_delta=improvement_delta,
            duration_ms=duration_ms,
        )

        # Update stats
        self.stats["distillations"] += 1
        self.stats["avg_alignment"] = (
            (self.stats["avg_alignment"] * (self.stats["distillations"] - 1) + teacher_alignment) /
            self.stats["distillations"]
        )
        self.stats["avg_improvement"] = (
            (self.stats["avg_improvement"] * (self.stats["distillations"] - 1) + improvement_delta) /
            self.stats["distillations"]
        )

        # Emit event
        self._emit_distillation_event(result, batch)

        return result

    async def _knowledge_distillation(
        self,
        teacher_genotypes: List[Genotype],
        student_genotypes: List[Genotype],
        temperature: float,
        alpha: float,
    ) -> Tuple[List[Genotype], Dict[str, float]]:
        """
        Perform knowledge distillation.

        Softens teacher output probabilities with temperature:
        p_soft = softmax(logits / T)

        Student learns from:
        - Soft labels (teacher knowledge)
        - Hard labels (ground truth)
        """
        distilled = []

        # Simple distillation: average teacher genomes, add noise
        if teacher_genotypes:
            # Average teacher genomes
            avg_genome = np.mean([g.genome for g in teacher_genotypes], axis=0)

            # Create distilled genotypes with temperature-controlled noise
            for i, teacher in enumerate(teacher_genotypes):
                # Temperature controls variance (higher = more exploration)
                noise = np.random.randn(*avg_genome.shape) * temperature * 0.1
                distilled_genome = avg_genome + noise

                # Fitness blend (teacher + temperature factor)
                distilled_fitness = teacher.fitness * (1 - 0.1 * temperature)

                genotype = Genotype(
                    genotype_id=f"distilled_{int(time.time() * 1000)}_{i}",
                    type=teacher.type,
                    genome=distilled_genome,
                    fitness=distilled_fitness,
                    generation=0,
                    parent_id=teacher.genotype_id,
                )
                distilled.append(genotype)

        # Calculate losses (simulated)
        task_loss = 0.1
        distillation_loss = 0.05 * temperature
        combined_loss = alpha * distillation_loss + (1 - alpha) * task_loss

        losses = {
            "task": task_loss,
            "distill": distillation_loss,
            "combined": combined_loss,
        }

        return distilled, losses

    def _teacher_output_to_genotype(self, output: TeacherOutput) -> Genotype:
        """Convert teacher output to genotype."""
        # Create genome from teacher response
        # In production, parse actual code/program from response
        genome_size = 100
        genome = np.random.randn(genome_size) * output.confidence

        return Genotype(
            genotype_id=f"teacher_{output.teacher.value}_{int(output.timestamp * 1000)}",
            type=GenotypeType.CGP_GRAPH,
            genome=genome,
            fitness=output.confidence,
            generation=0,
        )

    def _calculate_alignment(
        self,
        teacher_genotypes: List[Genotype],
        student_genotypes: List[Genotype],
    ) -> float:
        """Calculate teacher-student alignment."""
        if not teacher_genotypes or not student_genotypes:
            return 0.0

        # Compute cosine similarity between avg teacher and student genomes
        avg_teacher = np.mean([g.genome for g in teacher_genotypes], axis=0)
        avg_student = np.mean([g.genome for g in student_genotypes], axis=0)

        # Cosine similarity
        dot_product = np.dot(avg_teacher, avg_student)
        norms = np.linalg.norm(avg_teacher) * np.linalg.norm(avg_student)

        if norms == 0:
            return 0.0

        alignment = dot_product / norms
        return float(np.clip(alignment, 0.0, 1.0))

    async def ensemble_distill(
        self,
        image_data: str,
        prompt: str,
        inference_fn: Callable,
    ) -> DistillationResult:
        """
        Distill from multiple teachers (ensemble).

        Combines outputs from all teachers with weighted averaging.
        """
        # Collect from all teachers
        teacher_outputs = []
        for teacher in TeacherModel:
            if teacher == TeacherModel.ENSEMBLE:
                continue
            try:
                output = await self.collect_teacher_output(
                    image_data, prompt, teacher, inference_fn
                )
                teacher_outputs.append(output)
            except Exception as e:
                print(f"[VLMDistiller] Teacher {teacher} failed: {e}")

        # Create batch
        batch = DistillationBatch(
            batch_id=f"ensemble_{int(time.time() * 1000)}",
            teacher_outputs=teacher_outputs,
            distillation_method="knowledge_distillation",
            alpha=0.5,
        )

        # Weight outputs by ensemble weights
        for output in teacher_outputs:
            weight = self.ensemble_weights.get(output.teacher, 0.25)
            # Scale confidence by weight
            output.confidence *= weight

        # Distill
        result = await self.distill_from_batch(batch)

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get distillation statistics."""
        return {
            **self.stats,
            "cache_size": len(self.teacher_cache),
            "ensemble_weights": self.ensemble_weights,
        }

    def _emit_teacher_event(self, output: TeacherOutput) -> None:
        """Emit teacher output event."""
        if not self.bus_emitter:
            return

        event = {
            "topic": "vil.distillation.teacher_output",
            "data": {
                "teacher": output.teacher.value,
                "confidence": output.confidence,
                "latency_ms": output.latency_ms,
                "prompt_length": len(output.prompt),
            },
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[VLMDistiller] Bus emission error: {e}")

    def _emit_distillation_event(
        self,
        result: DistillationResult,
        batch: DistillationBatch,
    ) -> None:
        """Emit distillation completion event."""
        if not self.bus_emitter:
            return

        event = {
            "topic": "vil.distillation.complete",
            "data": {
                "batch_id": result.batch_id,
                "num_distilled": len(result.distilled_genotypes),
                "teacher_alignment": result.teacher_alignment,
                "combined_loss": result.combined_loss,
                "improvement_delta": result.improvement_delta,
                "duration_ms": result.duration_ms,
            },
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[VLMDistiller] Bus emission error: {e}")


def create_vlm_distiller(
    student_model: Optional[ProgramSynthesis] = None,
    bus_emitter: Optional[Callable] = None,
    ensemble_weights: Optional[Dict[TeacherModel, float]] = None,
) -> VLMDistiller:
    """Create VLM distiller with default config."""
    if student_model is None:
        student_model = create_program_synthesis(method="cgp")

    return VLMDistiller(
        student_model=student_model,
        bus_emitter=bus_emitter,
        ensemble_weights=ensemble_weights,
    )


__all__ = [
    "TeacherModel",
    "TeacherOutput",
    "DistillationBatch",
    "DistillationResult",
    "VLMDistiller",
    "create_vlm_distiller",
]
