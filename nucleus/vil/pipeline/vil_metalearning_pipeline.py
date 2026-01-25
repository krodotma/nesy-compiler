"""
VIL-Metalearning Unified Pipeline
Connects vision pipeline with metalearning for end-to-end learning.

Architecture:
Vision Pipeline → Task Extraction → Metalearning → CMP Update → Bus Event

This enables:
1. Vision frames to become metalearning tasks
2. Few-shot adaptation from vision examples
3. CMP tracking for metalearning quality
4. Geometric updates based on meta-loss
5. Reflexive Gödel machine integration

Version: 1.0
Date: 2026-01-25
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from nucleus.vil.events import (
    VILEventType,
    VisionEvent,
    LearningEvent,
    create_trace_id,
)
from nucleus.vil.pipeline.vision_pipeline import VisionPipeline, PipelineResult
from nucleus.vil.pipeline.metalearning_adapter import (
    MetalearningAdapter,
    MetalearningTask,
    MetalearningResult,
    MetalearningMethod,
    create_metalearning_adapter,
)
from nucleus.vil.pipeline.icl_pipeline import VisionToICLPipeline, ICLStrategy


@dataclass
class VILMetalearningResult:
    """
    Combined result from VIL + Metalearning pipeline.

    Contains:
    - Vision pipeline result
    - Metalearning result
    - Combined metrics
    """

    vision_result: PipelineResult
    metalearning_result: Optional[MetalearningResult] = None

    # Combined metrics
    combined_loss: float = 0.0
    combined_accuracy: float = 0.0
    improvement_score: float = 0.0

    # CMP
    cmp_delta: float = 0.0  # Change in CMP from metalearning

    # Geometric
    embedding_delta: float = 0.0  # Change in embedding space

    # Timing
    total_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vision": self.vision_result.to_dict(),
            "metalearning": {
                "task_id": self.metalearning_result.task_id if self.metalearning_result else None,
                "inner_loss": self.metalearning_result.inner_loss if self.metalearning_result else None,
                "outer_loss": self.metalearning_result.outer_loss if self.metalearning_result else None,
                "accuracy": self.metalearning_result.accuracy if self.metalearning_result else None,
            } if self.metalearning_result else None,
            "combined": {
                "loss": self.combined_loss,
                "accuracy": self.combined_accuracy,
                "improvement": self.improvement_score,
            },
            "cmp_delta": self.cmp_delta,
            "total_latency_ms": self.total_latency_ms,
        }


class VILMetalearningPipeline:
    """
    Unified VIL + Metalearning pipeline.

    Flow:
    1. Vision pipeline processes frame
    2. Extract task from vision + VLM result
    3. Add to metalearning adapter
    4. Run inner/outer loops
    5. Update CMP based on results
    6. Emit combined events
    """

    def __init__(
        self,
        bus_emitter: Optional[callable] = None,
        metalearning_method: MetalearningMethod = MetalearningMethod.MAML,
        enable_vision: bool = True,
        enable_metalearning: bool = True,
        enable_vlm: bool = True,  # Enable VLM inference in vision pipeline
        auto_adapt: bool = True,  # Auto-run adaptation on new tasks
    ):
        self.bus_emitter = bus_emitter
        self.metalearning_method = metalearning_method
        self.enable_vision = enable_vision
        self.enable_metalearning = enable_metalearning
        self.enable_vlm = enable_vlm
        self.auto_adapt = auto_adapt

        # Components
        self.vision_pipeline = VisionPipeline(
            bus_emitter=bus_emitter,
            enable_vlm=enable_vlm,
            enable_icl=True,
        )
        self.metalearning_adapter = create_metalearning_adapter(
            method=metalearning_method,
            bus_emitter=bus_emitter,
        )

        # Statistics
        self.stats = {
            "total_processed": 0,
            "vision_success": 0,
            "metalearning_success": 0,
            "combined_success": 0,
            "avg_improvement": 0.0,
            "avg_cmp_delta": 0.0,
            "total_latency_ms": 0.0,
        }

    async def process_frame_with_metalearning(
        self,
        image_data: str,
        prompt: str = "Describe and analyze this image.",
        run_inner_loop: bool = True,
        run_outer_loop: bool = False,  # Expensive, run periodically
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VILMetalearningResult:
        """
        Process frame with full vision + metalearning pipeline.

        Args:
            image_data: Base64-encoded image
            prompt: Prompt for VLM
            run_inner_loop: Run task-specific adaptation
            run_outer_loop: Run meta-learning update
            metadata: Additional metadata

        Returns:
            VILMetalearningResult with combined metrics
        """
        start_time = time.time()
        trace_id = create_trace_id("vil_meta")

        # Step 1: Vision pipeline
        vision_result = await self.vision_pipeline.process_frame(
            image_data=image_data,
            prompt=prompt,
            metadata=metadata,
        )

        # Initialize result
        result = VILMetalearningResult(
            vision_result=vision_result,
            total_latency_ms=(time.time() - start_time) * 1000,
        )

        # If vision failed, return early
        if not vision_result.success:
            self.stats["total_processed"] += 1
            return result

        self.stats["vision_success"] += 1

        # Step 2: Extract task from vision result
        if self.enable_metalearning and vision_result.vlm_response:
            # Create task from vision + VLM
            task_id = f"task_{vision_result.frame_id}"
            embedding = self._create_task_embedding(vision_result)

            # Add task to metalearning adapter
            meta_task = await self.metalearning_adapter.add_task(
                task_id=task_id,
                vision_data=image_data,
                prompt=prompt,
                response=vision_result.vlm_response,
                success=vision_result.vlm_confidence > 0.7,
                embedding=embedding,
                metadata={
                    "vision_trace_id": vision_result.trace_id,
                    "h_star": vision_result.entropy.h_star if vision_result.entropy else None,
                    "cmp_fitness": vision_result.cmp_fitness,
                },
            )

            # Step 3: Run inner loop (task-specific adaptation)
            if run_inner_loop and self.auto_adapt:
                meta_result = await self.metalearning_adapter.inner_loop(
                    task_id=task_id,
                )
                result.metalearning_result = meta_result

                # Calculate combined metrics
                result.combined_loss = meta_result.inner_loss
                result.combined_accuracy = (
                    vision_result.vlm_confidence * 0.5 +
                    meta_result.accuracy * 0.5
                )
                result.improvement_score = (
                    meta_result.accuracy - vision_result.vlm_confidence
                )
                result.cmp_delta = (
                    meta_result.accuracy * 0.618 -  # PHI weighting
                    vision_result.cmp_fitness
                )

                # Update vision tracker with metalearning results
                if hasattr(self.vision_pipeline, 'tracker'):
                    self.vision_pipeline.tracker.update_state(
                        frame_id=vision_result.frame_id,
                        new_state=None,  # Don't change state
                        additional_metrics={
                            "metalearning_accuracy": meta_result.accuracy,
                            "inner_loss": meta_result.inner_loss,
                        },
                    )

                self.stats["metalearning_success"] += 1

            # Step 4: Run outer loop periodically
            if run_outer_loop:
                outer_result = await self.metalearning_adapter.outer_loop(
                    num_tasks=10,
                )
                # Update combined metrics with outer loop results
                if result.metalearning_result:
                    result.metalearning_result.outer_loss = outer_result.outer_loss

        # Step 5: Emit combined event
        await self._emit_combined_event(result, trace_id)

        # Update stats
        self.stats["total_processed"] += 1
        if result.vision_result.success and result.metalearning_result:
            self.stats["combined_success"] += 1

        self.stats["avg_improvement"] = (
            (self.stats["avg_improvement"] * (self.stats["total_processed"] - 1) +
             result.improvement_score) / self.stats["total_processed"]
        )
        self.stats["avg_cmp_delta"] = (
            (self.stats["avg_cmp_delta"] * (self.stats["total_processed"] - 1) +
             result.cmp_delta) / self.stats["total_processed"]
        )
        self.stats["total_latency_ms"] = (
            (self.stats["total_latency_ms"] * (self.stats["total_processed"] - 1) +
             result.total_latency_ms) / self.stats["total_processed"]
        )

        result.total_latency_ms = (time.time() - start_time) * 1000

        return result

    async def adapt_to_new_task(
        self,
        example_images: List[str],
        example_prompts: List[str],
        example_responses: List[str],
        test_image: str,
        test_prompt: str,
    ) -> VILMetalearningResult:
        """
        Few-shot adaptation to new task using examples.

        Args:
            example_images: Training example images
            example_prompts: Training example prompts
            example_responses: Training example responses
            test_image: Test image
            test_prompt: Test prompt

        Returns:
            VILMetalearningResult with few-shot results
        """
        # Add all examples as tasks
        for i, (img, prompt, response) in enumerate(
            zip(example_images, example_prompts, example_responses)
        ):
            await self.metalearning_adapter.add_task(
                task_id=f"fewshot_example_{i}",
                vision_data=img,
                prompt=prompt,
                response=response,
                success=True,
            )

        # Process test image
        result = await self.process_frame_with_metalearning(
            image_data=test_image,
            prompt=test_prompt,
            run_inner_loop=True,
            run_outer_loop=True,
        )

        return result

    async def get_metalearning_state(self) -> Dict[str, Any]:
        """Get current metalearning state."""
        return {
            "vision": self.vision_pipeline.get_stats(),
            "metalearning": self.metalearning_adapter.get_stats(),
            "combined": self.stats,
        }

    def _create_task_embedding(self, vision_result: PipelineResult) -> np.ndarray:
        """Create task embedding from vision result."""
        # Combine multiple features into embedding
        features = []

        # VLM confidence
        features.append(vision_result.vlm_confidence)

        # H* entropy
        if vision_result.entropy:
            features.append(vision_result.entropy.h_star)
            features.append(vision_result.entropy.quality_score)
            features.append(vision_result.entropy.novelty_score)

        # CMP fitness
        features.append(vision_result.cmp_fitness)

        # Pad to 64 dimensions
        while len(features) < 64:
            features.append(0.0)

        return np.array(features[:64], dtype=np.float32)

    async def _emit_combined_event(
        self,
        result: VILMetalearningResult,
        trace_id: str,
    ) -> None:
        """Emit combined VIL+metalearning event."""
        if not self.bus_emitter:
            return

        event = {
            "topic": "vil.metalearning.combined",
            "data": {
                "trace_id": trace_id,
                "frame_id": result.vision_result.frame_id,
                "vision_success": result.vision_result.success,
                "metalearning_success": result.metalearning_result is not None,
                "combined_accuracy": result.combined_accuracy,
                "improvement_score": result.improvement_score,
                "cmp_delta": result.cmp_delta,
                "total_latency_ms": result.total_latency_ms,
                "metalearning_method": self.metalearning_method.value,
            },
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[VILMetalearningPipeline] Bus emission error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with current stats organized by component.
        """
        return {
            "vision": {
                "total_processed": self.stats["total_processed"],
                "success": self.stats["vision_success"],
                "enabled": self.enable_vision,
            },
            "metalearning": {
                "success": self.stats["metalearning_success"],
                "method": self.metalearning_method.value,
                "enabled": self.enable_metalearning,
            },
            "combined": {
                "success": self.stats["combined_success"],
                "avg_improvement": self.stats["avg_improvement"],
                "avg_cmp_delta": self.stats["avg_cmp_delta"],
                "avg_latency_ms": self.stats["total_latency_ms"],
            },
            "auto_adapt_enabled": self.auto_adapt,
            "vlm_enabled": self.enable_vlm,
        }


def create_vil_metalearning_pipeline(
    bus_emitter: Optional[callable] = None,
    metalearning_method: MetalearningMethod = MetalearningMethod.MAML,
) -> VILMetalearningPipeline:
    """Create VIL+metalearning pipeline with default config."""
    return VILMetalearningPipeline(
        bus_emitter=bus_emitter,
        metalearning_method=metalearning_method,
    )


__all__ = [
    "VILMetalearningResult",
    "VILMetalearningPipeline",
    "create_vil_metalearning_pipeline",
]
