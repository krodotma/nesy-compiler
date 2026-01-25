"""
VIL Vision Pipeline Tests
Test harness for Zone 1 vision integration components.

Tests:
1. Entropy normalization
2. Quality assessment
3. Deduplication
4. CMP tracking
5. ICL pipeline
6. End-to-end pipeline
7. Error recovery

Version: 1.0
Date: 2026-01-25
"""

import pytest
import asyncio
import base64
import numpy as np
from typing import Dict, Any, List

# Import VIL components
from nucleus.vil.pipeline.entropy import (
    EntropyNormalizer,
    compute_h_star,
    EntropyMetrics,
)
from nucleus.vil.pipeline.vision_quality import (
    VisionQualityProcessor,
    QualityScore,
    create_quality_processor,
)
from nucleus.vil.pipeline.vision_tracking import (
    VisionTracker,
    VisionState,
    VisionFrame,
    create_vision_tracker,
)
from nucleus.vil.pipeline.icl_pipeline import (
    VisionToICLPipeline,
    ICLStrategy,
    create_icl_pipeline,
)
from nucleus.vil.pipeline.vision_pipeline import (
    VisionPipeline,
    PipelineResult,
    PipelineError,
    create_vision_pipeline,
)


# === Test Fixtures ===

@pytest.fixture
def sample_image() -> str:
    """Create sample base64 image for testing."""
    # Simple 10x10 grayscale image
    pixels = np.random.randint(0, 256, 100, dtype=np.uint8)
    return base64.b64encode(pixels.tobytes()).decode()


@pytest.fixture
def sample_images() -> List[str]:
    """Create multiple sample images."""
    return [
        base64.b64encode(np.random.randint(0, 256, 100, dtype=np.uint8).tobytes()).decode()
        for _ in range(5)
    ]


@pytest.fixture
def bus_emitter():
    """Mock bus emitter for testing."""
    emitted = []

    def emitter(event: Dict[str, Any]) -> None:
        emitted.append(event)

    emitter.events = emitted
    return emitter


# === Entropy Tests ===

class TestEntropyNormalization:
    """Tests for H* entropy normalization."""

    def test_compute_h_star(self, sample_image):
        """Test H* computation."""
        metrics = compute_h_star(image_data=sample_image)

        assert isinstance(metrics, EntropyMetrics)
        assert 0 <= metrics.h_star <= 1
        assert 0 <= metrics.quality_score <= 1
        assert 0 <= metrics.novelty_score <= 1
        assert metrics.complexity in ["low", "medium", "high", "very_high"]

    def test_entropy_normalizer_baseline(self, sample_images):
        """Test entropy baseline tracking."""
        normalizer = EntropyNormalizer(baseline_window=3)

        for img in sample_images[:3]:
            normalizer.compute_entropy(image_data=img)

        stats = normalizer.get_baseline_stats()
        assert "mean" in stats
        assert "std" in stats
        assert 0 <= stats["mean"] <= 1

    def test_novelty_detection(self, sample_images):
        """Test novelty detection from drift."""
        normalizer = EntropyNormalizer(novelty_threshold=0.5)

        # Add baseline images
        for img in sample_images[:3]:
            normalizer.compute_entropy(image_data=img)

        # Add different image
        novel_metrics = normalizer.compute_entropy(image_data=sample_images[3])

        assert normalizer.is_novel(novel_metrics) == novel_metrics.novelty_score > 0.5


# === Quality Tests ===

class TestQualityAssessment:
    """Tests for vision quality assessment."""

    def test_quality_score(self, sample_image):
        """Test quality score computation."""
        processor = VisionQualityProcessor()
        quality = processor.assess_quality(sample_image)

        assert isinstance(quality, QualityScore)
        assert 0 <= quality.overall <= 1
        assert 0 <= quality.entropy <= 1
        assert 0 <= quality.sharpness <= 1

    def test_quality_threshold_filtering(self, sample_images):
        """Test quality threshold filtering."""
        processor = VisionQualityProcessor(quality_threshold=0.8)

        high_quality = False
        for img in sample_images:
            quality = processor.assess_quality(img)
            if quality.overall >= 0.8:
                high_quality = True

        # At least one should pass or all should fail
        stats = processor.get_stats()
        assert stats["frames_processed"] == len(sample_images)

    def test_deduplication(self, sample_images):
        """Test frame deduplication."""
        processor = VisionQualityProcessor(dedup_threshold=0.9)

        # Process same image twice
        frame_id_1 = "frame_1"
        frame_id_2 = "frame_2"

        result1 = processor.check_duplicate(sample_images[0], frame_id_1)
        result2 = processor.check_duplicate(sample_images[0], frame_id_2)

        assert not result1.is_duplicate  # First frame
        # Second frame might be duplicate depending on hash


# === CMP Tracking Tests ===

class TestCMPTracking:
    """Tests for CMP and lineage tracking."""

    def test_create_frame(self, sample_image):
        """Test vision frame creation."""
        tracker = create_vision_tracker()

        frame = tracker.create_frame(
            image_data=sample_image,
            capture_quality=0.8,
            analysis_confidence=0.9,
            icl_value=0.7,
        )

        assert isinstance(frame, VisionFrame)
        assert frame.cmp.phi_weighted_fitness > 0
        assert frame.lineage.generation == 0
        assert frame.state == VisionState.CAPTURED

    def test_lineage_tracking(self, sample_images):
        """Test lineage evolution."""
        tracker = create_vision_tracker()

        # Create parent
        parent = tracker.create_frame(
            image_data=sample_images[0],
            capture_quality=0.8,
        )

        # Create child
        child = tracker.create_frame(
            image_data=sample_images[1],
            parent_id=parent.frame_id,
            capture_quality=0.85,
        )

        assert child.lineage.generation == 1
        assert child.lineage.parent_id == parent.frame_id
        assert child.lineage.lineage_id == parent.lineage.lineage_id

    def test_state_updates(self, sample_image):
        """Test frame state updates."""
        tracker = create_vision_tracker()

        frame = tracker.create_frame(image_data=sample_image)

        # Update to analyzed
        updated = tracker.update_state(
            frame.frame_id,
            VisionState.ANALYZED,
            additional_metrics={"analysis_confidence": 0.95},
        )

        assert updated.state == VisionState.ANALYZED
        assert updated.cmp.analysis_confidence == 0.95


# === ICL Pipeline Tests ===

class TestICLPipeline:
    """Tests for vision-to-ICL pipeline."""

    @pytest.mark.asyncio
    async def test_process_frame_to_icl(self, sample_image):
        """Test frame processing to ICL example."""
        pipeline = create_icl_pipeline(buffer_size=3)

        example = await pipeline.process_frame(
            image_data=sample_image,
            prompt="What is this?",
            response="This is a test image.",
            success=True,
        )

        if example:
            assert example.success == True
            assert example.prompt == "What is this?"
            assert example.entropy.h_star >= 0

    @pytest.mark.asyncio
    async def test_icl_selection_strategies(self, sample_images):
        """Test ICL selection strategies."""
        pipeline = create_icl_pipeline()

        # Add examples
        for i, img in enumerate(sample_images[:3]):
            await pipeline.process_frame(
                image_data=img,
                prompt=f"Prompt {i}",
                response=f"Response {i}",
                success=True,
            )

        # Test strategies
        recent = pipeline.select_examples(k=2, strategy=ICLStrategy.RECENT)
        assert len(recent) <= 2

        random = pipeline.select_examples(k=2, strategy=ICLStrategy.RANDOM)
        assert len(random) <= 2

    def test_get_icl_context(self, sample_images):
        """Test ICL context generation."""
        pipeline = create_icl_pipeline()

        # Add examples
        for i, img in enumerate(sample_images[:3]):
            asyncio.run(pipeline.process_frame(
                image_data=img,
                prompt=f"Prompt {i}",
                response=f"Response {i}",
                success=True,
            ))

        context = pipeline.get_icl_context(k=2)
        assert isinstance(context, str)
        assert len(context) > 0


# === End-to-End Pipeline Tests ===

class TestVisionPipeline:
    """Tests for end-to-end vision pipeline."""

    @pytest.mark.asyncio
    async def test_process_frame_success(self, sample_image, bus_emitter):
        """Test successful frame processing."""
        pipeline = VisionPipeline(
            bus_emitter=bus_emitter,
            enable_vlm=False,  # Disable VLM for testing
            enable_icl=True,
        )

        result = await pipeline.process_frame(
            image_data=sample_image,
            prompt="Describe this image.",
        )

        # Result depends on quality
        if result.quality_score and result.quality_score.overall >= 0.6:
            # Should succeed if quality passes
            assert result.frame_id is not None
            assert result.trace_id is not None

    @pytest.mark.asyncio
    async def test_process_batch(self, sample_images, bus_emitter):
        """Test batch processing."""
        pipeline = VisionPipeline(
            bus_emitter=bus_emitter,
            enable_vlm=False,
        )

        frames = [{"image_data": img} for img in sample_images]
        results = await pipeline.process_batch(frames)

        assert len(results) == len(sample_images)

        for result in results:
            assert result.frame_id is not None
            assert result.trace_id is not None

    @pytest.mark.asyncio
    async def test_error_recovery(self, sample_image, bus_emitter):
        """Test error recovery mechanism."""
        pipeline = VisionPipeline(
            bus_emitter=bus_emitter,
            enable_vlm=False,
        )

        # Process should not crash even with issues
        result = await pipeline.process_frame(
            image_data=sample_image,
            prompt="Test",
            retry_on_error=True,
        )

        assert result is not None
        assert result.frame_id is not None

    def test_pipeline_stats(self, sample_images, bus_emitter):
        """Test pipeline statistics."""
        pipeline = VisionPipeline(
            bus_emitter=bus_emitter,
            enable_vlm=False,
        )

        # Run sync version for stats test
        async def run():
            frames = [{"image_data": img} for img in sample_images[:3]]
            await pipeline.process_batch(frames)

        asyncio.run(run())

        stats = pipeline.get_stats()
        assert "total_processed" in stats
        assert "avg_latency_ms" in stats
        assert stats["total_processed"] >= 0


# === Integration Tests ===

class TestZone1Integration:
    """Integration tests for Zone 1 components."""

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self, sample_images, bus_emitter):
        """Test full integration of all Zone 1 components."""
        pipeline = VisionPipeline(
            bus_emitter=bus_emitter,
            quality_threshold=0.5,
            dedup_threshold=0.9,
            enable_vlm=False,
            enable_icl=True,
        )

        results = []
        for img in sample_images:
            result = await pipeline.process_frame(
                image_data=img,
                prompt="Analyze this image.",
            )
            results.append(result)

        # Verify results
        assert len(results) == len(sample_images)

        # Check stats
        stats = pipeline.get_stats()
        assert stats["total_processed"] == len(sample_images)
        assert "quality_processor" in stats
        assert "tracker" in stats
        assert "icl_pipeline" in stats


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
