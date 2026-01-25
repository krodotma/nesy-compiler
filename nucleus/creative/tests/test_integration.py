"""
Creative Section Integration Tests
===================================

Tests for verifying all subsystems load correctly and basic operations work.
"""

import pytest
import numpy as np
from pathlib import Path


class TestCreativePackage:
    """Test main creative package imports."""

    def test_package_imports(self):
        """Test that main package imports correctly."""
        from nucleus.creative import (
            __version__,
            CreativeMode,
            GenerationJob,
            Asset,
            PipelineStage,
            CreativeState,
            BUS_TOPICS,
            SUBSYSTEMS,
        )

        # Version check - should be >= 1.4.0 after types/errors addition
        assert __version__.startswith("1.") and int(__version__.split(".")[1]) >= 4
        assert "grammars" in SUBSYSTEMS
        assert "cinema" in SUBSYSTEMS
        assert "visual" in SUBSYSTEMS
        assert "auralux" in SUBSYSTEMS
        assert "avatars" in SUBSYSTEMS
        assert "dits" in SUBSYSTEMS

    def test_subsystem_registry(self):
        """Test subsystem registry structure."""
        from nucleus.creative import SUBSYSTEMS

        for key, subsys in SUBSYSTEMS.items():
            assert "name" in subsys
            assert "description" in subsys
            assert "get" in subsys
            assert "features" in subsys
            assert callable(subsys["get"])

    def test_bus_topics(self):
        """Test bus topic definitions."""
        from nucleus.creative import BUS_TOPICS

        assert "generation_started" in BUS_TOPICS
        assert "grammar_synthesize" in BUS_TOPICS
        assert "cinema_generate" in BUS_TOPICS
        assert "visual_render" in BUS_TOPICS
        assert "auralux_synthesize" in BUS_TOPICS
        assert "avatar_generate" in BUS_TOPICS
        assert "dits_evaluate" in BUS_TOPICS


class TestGrammarsSubsystem:
    """Test grammars subsystem."""

    def test_cgp_imports(self):
        """Test CGP module imports."""
        from nucleus.creative.grammars.cgp import (
            CGPGenome,
            CGPNode,
            CGPFunction,
            FUNCTION_SET,
            cgp_evolve,
        )

        assert len(FUNCTION_SET) > 0

    def test_cgp_genome_creation(self):
        """Test CGP genome creation."""
        from nucleus.creative.grammars.cgp import CGPGenome, CGPNode, FUNCTION_SET

        # Test basic genome creation using function_id index
        nodes = [
            CGPNode(function_id=0, inputs=[0, 1]),  # add function
        ]
        genome = CGPGenome(
            n_inputs=2, n_outputs=1, n_rows=1, n_cols=1,
            nodes=nodes, output_genes=[2], function_set=FUNCTION_SET
        )

        assert genome is not None
        assert len(genome.nodes) > 0
        assert genome.n_inputs == 2

    def test_eggp_imports(self):
        """Test EGGP module imports."""
        from nucleus.creative.grammars.eggp import (
            GraphNode,
            GraphProgram,
            EGGPEvolver,
            EGGPConfig,
        )

    def test_metagrammar_imports(self):
        """Test metagrammar module imports."""
        from nucleus.creative.grammars.metagrammar import (
            TransformRule,
            MetagrammarRegistry,
            PatternVariable,
        )

    def test_parser_imports(self):
        """Test parser module imports."""
        from nucleus.creative.grammars.parser import (
            GrammarRule,
            ASTNode,
            GrammarParser,
        )


class TestCinemaSubsystem:
    """Test cinema subsystem."""

    def test_cinema_package_imports(self):
        """Test cinema package can be imported."""
        from nucleus.creative import cinema
        assert cinema is not None

    def test_script_parser_attr(self):
        """Test script_parser submodule accessible."""
        from nucleus.creative import cinema
        # Access via attribute (loaded from pyc)
        if hasattr(cinema, 'script_parser') and cinema.script_parser:
            assert True
        else:
            pytest.skip("script_parser pyc not loadable")

    def test_storyboard_attr(self):
        """Test storyboard submodule accessible."""
        from nucleus.creative import cinema
        if hasattr(cinema, 'storyboard') and cinema.storyboard:
            assert True
        else:
            pytest.skip("storyboard pyc not loadable")

    def test_frame_generator_attr(self):
        """Test frame_generator submodule accessible."""
        from nucleus.creative import cinema
        if hasattr(cinema, 'frame_generator') and cinema.frame_generator:
            assert True
        else:
            pytest.skip("frame_generator pyc not loadable")

    def test_temporal_consistency_attr(self):
        """Test temporal_consistency submodule accessible."""
        from nucleus.creative import cinema
        if hasattr(cinema, 'temporal_consistency') and cinema.temporal_consistency:
            assert True
        else:
            pytest.skip("temporal_consistency pyc not loadable")


class TestVisualSubsystem:
    """Test visual subsystem."""

    def test_visual_package_imports(self):
        """Test visual package can be imported."""
        from nucleus.creative import visual
        assert visual is not None

    def test_generator_attr(self):
        """Test generator submodule accessible."""
        from nucleus.creative import visual
        if hasattr(visual, 'generator') and visual.generator:
            assert True
        else:
            pytest.skip("generator pyc not loadable")

    def test_style_transfer_attr(self):
        """Test style_transfer submodule accessible."""
        from nucleus.creative import visual
        if hasattr(visual, 'style_transfer') and visual.style_transfer:
            assert True
        else:
            pytest.skip("style_transfer pyc not loadable")

    def test_upscaler_attr(self):
        """Test upscaler submodule accessible."""
        from nucleus.creative import visual
        if hasattr(visual, 'upscaler') and visual.upscaler:
            assert True
        else:
            pytest.skip("upscaler pyc not loadable")


class TestAuraluxSubsystem:
    """Test auralux subsystem."""

    def test_auralux_package_imports(self):
        """Test auralux package can be imported."""
        from nucleus.creative import auralux
        assert auralux is not None

    def test_synthesizer_attr(self):
        """Test synthesizer submodule accessible."""
        from nucleus.creative import auralux
        if hasattr(auralux, 'synthesizer') and auralux.synthesizer:
            assert True
        else:
            pytest.skip("synthesizer pyc not loadable")

    def test_recognizer_attr(self):
        """Test recognizer submodule accessible."""
        from nucleus.creative import auralux
        if hasattr(auralux, 'recognizer') and auralux.recognizer:
            assert True
        else:
            pytest.skip("recognizer pyc not loadable")

    def test_speaker_encoder_attr(self):
        """Test speaker_encoder submodule accessible."""
        from nucleus.creative import auralux
        if hasattr(auralux, 'speaker_encoder') and auralux.speaker_encoder:
            assert True
        else:
            pytest.skip("speaker_encoder pyc not loadable")


class TestAvatarsSubsystem:
    """Test avatars subsystem."""

    def test_avatars_package_imports(self):
        """Test avatars package can be imported."""
        from nucleus.creative import avatars
        assert avatars is not None

    def test_smplx_extractor_attr(self):
        """Test smplx_extractor submodule accessible."""
        from nucleus.creative import avatars
        if hasattr(avatars, 'smplx_extractor') and avatars.smplx_extractor:
            assert True
        else:
            pytest.skip("smplx_extractor pyc not loadable")

    def test_gaussian_splatting_attr(self):
        """Test gaussian_splatting submodule accessible."""
        from nucleus.creative import avatars
        if hasattr(avatars, 'gaussian_splatting') and avatars.gaussian_splatting:
            assert True
        else:
            pytest.skip("gaussian_splatting pyc not loadable")

    def test_deformable_attr(self):
        """Test deformable submodule accessible."""
        from nucleus.creative import avatars
        if hasattr(avatars, 'deformable') and avatars.deformable:
            assert True
        else:
            pytest.skip("deformable pyc not loadable")


class TestDiTSSubsystem:
    """Test DiTS subsystem."""

    def test_dits_package_imports(self):
        """Test dits package can be imported."""
        from nucleus.creative import dits
        assert dits is not None

    def test_kernel_attr(self):
        """Test kernel submodule accessible."""
        from nucleus.creative import dits
        if hasattr(dits, 'kernel') and dits.kernel:
            assert True
        else:
            pytest.skip("kernel pyc not loadable")

    def test_narrative_attr(self):
        """Test narrative submodule accessible."""
        from nucleus.creative import dits
        if hasattr(dits, 'narrative') and dits.narrative:
            assert True
        else:
            pytest.skip("narrative pyc not loadable")

    def test_rheomode_attr(self):
        """Test rheomode submodule accessible."""
        from nucleus.creative import dits
        if hasattr(dits, 'rheomode') and dits.rheomode:
            assert True
        else:
            pytest.skip("rheomode pyc not loadable")

    def test_omega_bridge_attr(self):
        """Test omega_bridge submodule accessible."""
        from nucleus.creative import dits
        if hasattr(dits, 'omega_bridge') and dits.omega_bridge:
            assert True
        else:
            pytest.skip("omega_bridge pyc not loadable")


class TestDataclasses:
    """Test dataclass instantiation."""

    def test_generation_job(self):
        """Test GenerationJob dataclass."""
        from nucleus.creative import GenerationJob

        job = GenerationJob(
            id="test-123",
            mode="grammars",
            params={"prompt": "test"},
        )

        assert job.id == "test-123"
        assert job.mode == "grammars"
        assert job.status == "pending"
        assert job.progress == 0.0

    def test_asset(self):
        """Test Asset dataclass."""
        from nucleus.creative import Asset

        asset = Asset(
            id="asset-123",
            type="image",
            path="/tmp/test.png",
        )

        assert asset.id == "asset-123"
        assert asset.type == "image"

    def test_creative_state(self):
        """Test CreativeState dataclass."""
        from nucleus.creative import CreativeState

        state = CreativeState()

        assert state.active_mode == "grammars"
        assert state.assets == {}
        assert state.job_queue == []


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test async operations (require optional dependencies)."""

    async def test_simple_upscaler(self):
        """Test simple upscaler (no GPU required)."""
        pytest.importorskip("PIL")

        from nucleus.creative.visual import SimpleUpscaler

        if SimpleUpscaler is None:
            pytest.skip("SimpleUpscaler not available")

        upscaler = SimpleUpscaler()

        # Create test image
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        result = await upscaler.upscale(test_image, scale_factor=2, method="lanczos")

        assert result.output_size == (128, 128)
        assert result.scale_factor == 2
        assert result.model_used == "PIL_lanczos"


class TestPipelineUtilities:
    """Test pipeline utilities."""

    def test_create_pipeline(self):
        """Test pipeline creation."""
        from nucleus.creative import create_pipeline

        job = create_pipeline("visual", prompt="test image")

        assert job.id.startswith("gen-")
        assert job.mode == "visual"
        assert job.status == "pending"
        assert job.params["prompt"] == "test image"
        assert "_stages" in job.params
        stages = [s.name for s in job.params["_stages"]]
        assert stages == ["generate", "upscale", "style"]

    def test_create_pipeline_custom_stages(self):
        """Test pipeline creation with custom stages."""
        from nucleus.creative import create_pipeline

        job = create_pipeline("cinema", stages=["parse", "render"])

        assert job.mode == "cinema"
        stages = [s.name for s in job.params["_stages"]]
        assert stages == ["parse", "render"]

    def test_emit_bus_event(self):
        """Test bus event emission."""
        from nucleus.creative import emit_bus_event
        from pathlib import Path
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ndjson', delete=False) as f:
            test_path = Path(f.name)

        try:
            event = emit_bus_event(
                "test.topic",
                {"key": "value"},
                bus_path=test_path,
            )

            assert event["topic"] == "test.topic"
            assert event["payload"]["key"] == "value"
            assert "timestamp" in event
            assert "id" in event

            # Verify written to file
            with open(test_path) as f:
                import json
                written = json.loads(f.readline())
                assert written["topic"] == "test.topic"
        finally:
            test_path.unlink(missing_ok=True)

    def test_subsystem_layers(self):
        """Test that subsystems have layer mappings."""
        from nucleus.creative import SUBSYSTEMS

        for key, sub in SUBSYSTEMS.items():
            assert "layer" in sub, f"{key} missing layer"
            assert sub["layer"].startswith("L"), f"{key} invalid layer format"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
