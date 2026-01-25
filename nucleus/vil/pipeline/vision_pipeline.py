"""
VIL Vision Pipeline - End-to-End Integration
Combines all vision components into a unified processing pipeline.

Components:
1. TheiaVILAdapter - VLM inference
2. EntropyNormalizer - H* computation
3. VisionToICLPipeline - ICL buffer management
4. VisionTracker - CMP + lineage tracking
5. VisionQualityProcessor - Quality + deduplication

Flow:
Vision Frame → Quality → Dedup → Entropy → CMP → VLM → ICL → Bus Event

Version: 1.0
Date: 2026-01-25
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import traceback

from nucleus.vil.events import (
    VILEventType,
    VisionEvent,
    LearningEvent,
    create_trace_id,
)
from nucleus.vil.pipeline.theia_adapter import TheiaVILAdapter, VLMInference
from nucleus.vil.pipeline.entropy import compute_h_star, EntropyMetrics
from nucleus.vil.pipeline.icl_pipeline import VisionToICLPipeline, ICLStrategy
from nucleus.vil.pipeline.vision_tracking import VisionTracker, VisionState, VisionFrame
from nucleus.vil.pipeline.vision_quality import VisionQualityProcessor, QualityScore


class PipelineError(Exception):
    """Pipeline-specific error."""

    def __init__(self, message: str, frame_id: str, stage: str, recovery_suggested: bool = False):
        self.message = message
        self.frame_id = frame_id
        self.stage = stage
        self.recovery_suggested = recovery_suggested
        super().__init__(f"[{stage}] {message} (frame: {frame_id})")


@dataclass
class PipelineResult:
    """Result from vision pipeline processing."""

    frame_id: str
    success: bool
    stage: str  # Last completed stage
    trace_id: str

    # Quality metrics
    quality_score: Optional[QualityScore] = None
    is_duplicate: bool = False

    # Entropy metrics
    entropy: Optional[EntropyMetrics] = None

    # CMP metrics
    cmp_fitness: float = 0.0
    generation: int = 0
    lineage_id: str = ""

    # VLM inference
    vlm_response: Optional[str] = None
    vlm_confidence: float = 0.0

    # ICL
    icl_added: bool = False
    icl_buffer_size: int = 0

    # Errors
    error: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False

    # Timing
    total_latency_ms: float = 0.0
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_id": self.frame_id,
            "success": self.success,
            "stage": self.stage,
            "trace_id": self.trace_id,
            "quality": self.quality_score.to_dict() if self.quality_score else None,
            "is_duplicate": self.is_duplicate,
            "h_star": self.entropy.h_star if self.entropy else None,
            "cmp_fitness": self.cmp_fitness,
            "generation": self.generation,
            "lineage_id": self.lineage_id,
            "vlm_confidence": self.vlm_confidence,
            "icl_added": self.icl_added,
            "icl_buffer_size": self.icl_buffer_size,
            "error": self.error,
            "total_latency_ms": self.total_latency_ms,
        }


class VisionPipeline:
    """
    End-to-end vision processing pipeline.

    Stages:
    1. Quality assessment
    2. Deduplication check
    3. Entropy calculation
    4. CMP tracking
    5. VLM inference
    6. ICL buffer update
    7. Bus event emission
    """

    def __init__(
        self,
        bus_emitter: Optional[callable] = None,
        quality_threshold: float = 0.6,
        dedup_threshold: float = 0.85,
        enable_icl: bool = True,
        enable_vlm: bool = True,
    ):
        self.bus_emitter = bus_emitter
        self.quality_threshold = quality_threshold
        self.dedup_threshold = dedup_threshold
        self.enable_icl = enable_icl
        self.enable_vlm = enable_vlm

        # Components
        self.quality_processor = VisionQualityProcessor(
            quality_threshold=quality_threshold,
            dedup_threshold=dedup_threshold,
        )
        self.tracker = VisionTracker(bus_emitter=bus_emitter)
        self.icl_pipeline = VisionToICLPipeline(
            buffer_size=5,
            bus_emitter=bus_emitter,
        )
        self.theia_adapter = TheiaVILAdapter(bus_emitter=bus_emitter)

        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "recovered": 0,
            "avg_latency_ms": 0.0,
            "by_stage": {
                "quality": 0,
                "dedup": 0,
                "entropy": 0,
                "cmp": 0,
                "vlm": 0,
                "icl": 0,
            },
        }

    async def process_frame(
        self,
        image_data: str,
        prompt: str = "Describe this image.",
        metadata: Optional[Dict[str, Any]] = None,
        retry_on_error: bool = True,
    ) -> PipelineResult:
        """
        Process single frame through full pipeline.

        Args:
            image_data: Base64-encoded image
            prompt: Prompt for VLM inference
            metadata: Additional metadata
            retry_on_error: Whether to retry on error

        Returns:
            PipelineResult with all metrics
        """
        start_time = time.time()
        trace_id = create_trace_id("vision_pipeline")
        frame_id = f"frame_{int(start_time * 1000)}"
        stage_latencies = {}

        result = PipelineResult(
            frame_id=frame_id,
            success=False,
            stage="initialized",
            trace_id=trace_id,
        )

        try:
            # Stage 1: Quality assessment
            stage_start = time.time()
            quality = self.quality_processor.assess_quality(image_data)
            stage_latencies["quality"] = (time.time() - stage_start) * 1000
            result.quality_score = quality
            result.stage = "quality"

            if quality.overall < self.quality_threshold:
                result.success = False
                result.error = f"Quality below threshold: {quality.overall:.2f} < {self.quality_threshold}"
                return result

            # Stage 2: Deduplication
            stage_start = time.time()
            dedup = self.quality_processor.check_duplicate(image_data, frame_id)
            stage_latencies["dedup"] = (time.time() - stage_start) * 1000
            result.is_duplicate = dedup.is_duplicate
            result.stage = "dedup"

            if dedup.is_duplicate:
                result.success = False
                result.error = f"Duplicate frame: similarity={dedup.similarity_score:.2f}"
                return result

            # Stage 3: Entropy calculation
            stage_start = time.time()
            entropy = compute_h_star(image_data=image_data)
            stage_latencies["entropy"] = (time.time() - stage_start) * 1000
            result.entropy = entropy
            result.stage = "entropy"

            # Stage 4: CMP tracking
            stage_start = time.time()
            vision_frame = self.tracker.create_frame(
                image_data=image_data,
                capture_quality=entropy.quality_score,
                icl_value=entropy.novelty_score,
                metadata=metadata,
            )
            stage_latencies["cmp"] = (time.time() - stage_start) * 1000
            result.cmp_fitness = vision_frame.cmp.phi_weighted_fitness
            result.generation = vision_frame.cmp.generation
            result.lineage_id = vision_frame.lineage.lineage_id
            result.stage = "cmp"

            # Stage 5: VLM inference (optional)
            vlm_result: Optional[VLMInference] = None
            if self.enable_vlm:
                stage_start = time.time()
                try:
                    vlm_result = await self.theia_adapter.infer(
                        image=image_data,
                        prompt=prompt,
                        trace_id=trace_id,
                    )
                    stage_latencies["vlm"] = (time.time() - stage_start) * 1000
                    result.vlm_response = vlm_result.response
                    result.vlm_confidence = vlm_result.confidence
                    result.stage = "vlm"

                    # Update tracker with VLM confidence
                    self.tracker.update_state(
                        frame_id=vision_frame.frame_id,
                        new_state=VisionState.ANALYZED,
                        additional_metrics={"analysis_confidence": vlm_result.confidence},
                    )
                except Exception as e:
                    if not retry_on_error:
                        raise PipelineError(str(e), frame_id, "vlm")
                    # Recovery: continue without VLM
                    result.recovery_attempted = True
                    result.recovery_successful = True
                    stage_latencies["vlm"] = (time.time() - stage_start) * 1000

            # Stage 6: ICL buffer update (optional)
            if self.enable_icl and vlm_result:
                stage_start = time.time()
                icl_example = await self.icl_pipeline.process_frame(
                    image_data=image_data,
                    prompt=prompt,
                    response=vlm_result.response,
                    success=vlm_result.confidence > 0.7,
                )
                stage_latencies["icl"] = (time.time() - stage_start) * 1000
                result.icl_added = icl_example is not None
                result.icl_buffer_size = self.icl_pipeline.stats["icl_examples"]
                result.stage = "icl"

                # Update tracker state
                if icl_example:
                    self.tracker.update_state(
                        frame_id=vision_frame.frame_id,
                        new_state=VisionState.ICL_ADDED,
                    )
                else:
                    self.tracker.update_state(
                        frame_id=vision_frame.frame_id,
                        new_state=VisionState.ACCEPTED,
                    )

            # Stage 7: Emit bus event
            await self._emit_pipeline_event(result, vision_frame)

            # Success
            result.success = True
            result.total_latency_ms = (time.time() - start_time) * 1000
            result.stage_latencies = stage_latencies

            self.stats["total_processed"] += 1
            self.stats["successful"] += 1
            for stage_name in stage_latencies:
                self.stats["by_stage"][stage_name] += 1

            # Update average latency
            self.stats["avg_latency_ms"] = (
                (self.stats["avg_latency_ms"] * (self.stats["total_processed"] - 1) + result.total_latency_ms) /
                self.stats["total_processed"]
            )

        except PipelineError as e:
            result.success = False
            result.error = str(e)
            result.stage = e.stage
            result.recovery_attempted = retry_on_error

            self.stats["total_processed"] += 1
            self.stats["failed"] += 1

            if retry_on_error and not e.recovery_suggested:
                # Attempt recovery
                result.recovery_attempted = True
                try:
                    # Recovery: process without VLM
                    self.enable_vlm = False
                    recovery_result = await self.process_frame(
                        image_data, prompt, metadata, retry_on_error=False
                    )
                    result.recovery_successful = recovery_result.success
                    self.stats["recovered"] += 1
                except:
                    result.recovery_successful = False
                finally:
                    self.enable_vlm = True

        except Exception as e:
            result.success = False
            result.error = f"Unexpected error: {str(e)}"
            result.stage = "error"
            result.total_latency_ms = (time.time() - start_time) * 1000

            self.stats["total_processed"] += 1
            self.stats["failed"] += 1

        return result

    async def process_batch(
        self,
        frames: List[Dict[str, Any]],
        prompt: str = "Describe this image.",
    ) -> List[PipelineResult]:
        """
        Process batch of frames.

        Args:
            frames: List of dicts with image_data and optional metadata
            prompt: Prompt for VLM inference

        Returns:
            List of PipelineResults
        """
        results = []

        for frame_dict in frames:
            image_data = frame_dict["image_data"]
            metadata = frame_dict.get("metadata")

            result = await self.process_frame(image_data, prompt, metadata)
            results.append(result)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful"] / max(1, self.stats["total_processed"])
            ),
            "failure_rate": (
                self.stats["failed"] / max(1, self.stats["total_processed"])
            ),
            "recovery_rate": (
                self.stats["recovered"] / max(1, self.stats["failed"])
            ),
            "quality_processor": self.quality_processor.get_stats(),
            "tracker": self.tracker.get_stats(),
            "icl_pipeline": self.icl_pipeline.get_stats(),
            "theia_adapter": self.theia_adapter.get_stats(),
        }

    async def _emit_pipeline_event(
        self,
        result: PipelineResult,
        frame: VisionFrame,
    ) -> None:
        """Emit pipeline completion event to bus."""
        if not self.bus_emitter:
            return

        event = {
            "topic": "vil.pipeline.complete",
            "data": {
                "trace_id": result.trace_id,
                "frame_id": result.frame_id,
                "success": result.success,
                "stage": result.stage,
                "latency_ms": result.total_latency_ms,
                "cmp_fitness": result.cmp_fitness,
                "generation": result.generation,
                "lineage_id": result.lineage_id,
                "icl_added": result.icl_added,
                "vlm_confidence": result.vlm_confidence,
            },
        }

        try:
            self.bus_emitter(event)
        except Exception as e:
            print(f"[VisionPipeline] Bus emission error: {e}")


def create_vision_pipeline(
    bus_emitter: Optional[callable] = None,
    quality_threshold: float = 0.6,
    dedup_threshold: float = 0.85,
) -> VisionPipeline:
    """Create vision pipeline with default config."""
    return VisionPipeline(
        bus_emitter=bus_emitter,
        quality_threshold=quality_threshold,
        dedup_threshold=dedup_threshold,
    )


__all__ = [
    "PipelineError",
    "PipelineResult",
    "VisionPipeline",
    "create_vision_pipeline",
]
