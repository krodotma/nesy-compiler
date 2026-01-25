#!/usr/bin/env python3
"""
Cross-Subsystem Pipeline Example
================================

Demonstrates multi-subsystem pipelines that chain operations
across grammars, visual, cinema, auralux, avatars, and dits.
"""

import asyncio
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# =============================================================================
# MOCK SUBSYSTEM HANDLERS
# =============================================================================

class MockSubsystemHandlers:
    """Mock handlers for all subsystems."""

    @staticmethod
    async def handle_grammars_parse(inputs: dict) -> dict:
        """Parse grammar specification."""
        print("   [grammars.parse] Parsing grammar specification...")
        await asyncio.sleep(0.1)
        return {
            "ast": {"type": "grammar", "rules": 5},
            "node_count": 15,
        }

    @staticmethod
    async def handle_grammars_synthesize(inputs: dict) -> dict:
        """Synthesize program."""
        print("   [grammars.synthesize] Running CGP synthesis...")
        await asyncio.sleep(0.2)
        return {
            "program": "f(x) = x * 2 + 1",
            "fitness": 0.95,
        }

    @staticmethod
    async def handle_visual_generate(inputs: dict) -> dict:
        """Generate image."""
        prompt = inputs.get("prompt", "default")
        print(f"   [visual.generate] Generating image: '{prompt[:30]}...'")
        await asyncio.sleep(0.2)
        return {
            "image": np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8),
            "seed": 42,
        }

    @staticmethod
    async def handle_visual_upscale(inputs: dict) -> dict:
        """Upscale image."""
        print("   [visual.upscale] Upscaling image 2x...")
        await asyncio.sleep(0.1)
        image = inputs.get("image")
        if image is not None:
            h, w = image.shape[:2]
            return {"image": np.zeros((h * 2, w * 2, 3), dtype=np.uint8)}
        return {"image": np.zeros((1024, 1024, 3), dtype=np.uint8)}

    @staticmethod
    async def handle_avatars_extract(inputs: dict) -> dict:
        """Extract SMPL-X parameters."""
        print("   [avatars.extract] Extracting SMPL-X from image...")
        await asyncio.sleep(0.3)
        return {
            "params": {"shape": np.zeros(10), "pose": np.zeros(63)},
            "vertices": np.random.randn(10475, 3),
            "confidence": 0.92,
        }

    @staticmethod
    async def handle_avatars_gaussian_convert(inputs: dict) -> dict:
        """Convert mesh to 3DGS."""
        print("   [avatars.gaussian_convert] Converting to Gaussians...")
        await asyncio.sleep(0.2)
        return {
            "gaussian_count": 50000,
            "bounds": {"min": [-1, -1, -1], "max": [1, 1, 1]},
        }

    @staticmethod
    async def handle_avatars_deform(inputs: dict) -> dict:
        """Deform with pose."""
        print("   [avatars.deform] Deforming Gaussians with pose...")
        await asyncio.sleep(0.15)
        return {
            "frames": 30,
            "deformed_count": 50000,
        }

    @staticmethod
    async def handle_avatars_pbr(inputs: dict) -> dict:
        """Predict PBR materials."""
        print("   [avatars.pbr] Predicting PBR materials...")
        await asyncio.sleep(0.2)
        return {
            "albedo": np.zeros((1024, 1024, 3)),
            "normal": np.zeros((1024, 1024, 3)),
            "roughness": np.zeros((1024, 1024)),
        }

    @staticmethod
    async def handle_cinema_storyboard(inputs: dict) -> dict:
        """Generate storyboard."""
        print("   [cinema.storyboard] Generating storyboard...")
        await asyncio.sleep(0.2)
        return {
            "panels": 12,
            "total_duration": 45.0,
        }

    @staticmethod
    async def handle_cinema_frame_generate(inputs: dict) -> dict:
        """Generate video frames."""
        print("   [cinema.frame_generate] Generating frames...")
        await asyncio.sleep(0.3)
        return {
            "frames": [np.zeros((720, 1280, 3)) for _ in range(30)],
            "fps": 24,
        }

    @staticmethod
    async def handle_cinema_assemble(inputs: dict) -> dict:
        """Assemble video."""
        print("   [cinema.assemble] Assembling video...")
        await asyncio.sleep(0.2)
        return {
            "output_path": "/tmp/output.mp4",
            "duration_s": 30.0,
        }

    @staticmethod
    async def handle_auralux_synthesize(inputs: dict) -> dict:
        """Synthesize speech."""
        text = inputs.get("text", "")
        print(f"   [auralux.synthesize] Synthesizing: '{text[:30]}...'")
        await asyncio.sleep(0.2)
        return {
            "audio": np.random.randn(24000 * 5),
            "duration_s": 5.0,
        }

    @staticmethod
    async def handle_dits_evaluate(inputs: dict) -> dict:
        """Evaluate DiTS state."""
        print("   [dits.evaluate] Evaluating DiTS state...")
        await asyncio.sleep(0.1)
        return {
            "state": "exploring",
            "rank": 2,
            "modifications": ["mhc_beta_update"],
        }

    @staticmethod
    async def handle_dits_narrative(inputs: dict) -> dict:
        """Construct narrative."""
        print("   [dits.narrative] Constructing narrative...")
        await asyncio.sleep(0.15)
        return {
            "episodes": 5,
            "coherence_score": 0.85,
            "text": "The hero embarks on a journey of discovery...",
        }


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

@dataclass
class StageResult:
    """Result from a pipeline stage."""
    name: str
    status: str
    output: dict
    elapsed_ms: float


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    status: str
    stages: list
    final_output: dict
    total_ms: float


class SimplePipelineOrchestrator:
    """Simple pipeline orchestrator for demonstration."""

    def __init__(self):
        self.handlers = MockSubsystemHandlers()

    async def execute_stage(
        self,
        stage_name: str,
        subsystem: str,
        operation: str,
        inputs: dict,
    ) -> StageResult:
        """Execute a single pipeline stage."""
        import time
        start = time.time()

        # Get handler
        handler_name = f"handle_{subsystem}_{operation}"
        handler = getattr(self.handlers, handler_name, None)

        if handler is None:
            return StageResult(
                name=stage_name,
                status="skipped",
                output={},
                elapsed_ms=0,
            )

        try:
            output = await handler(inputs)
            status = "completed"
        except Exception as e:
            output = {"error": str(e)}
            status = "failed"

        elapsed = (time.time() - start) * 1000

        return StageResult(
            name=stage_name,
            status=status,
            output=output,
            elapsed_ms=elapsed,
        )

    async def run_pipeline(
        self,
        stages: list[dict],
        initial_inputs: dict,
    ) -> PipelineResult:
        """Run a multi-stage pipeline."""
        import time
        start = time.time()

        stage_results = []
        context = dict(initial_inputs)

        for stage in stages:
            # Merge context into inputs
            inputs = {**context, **stage.get("params", {})}

            result = await self.execute_stage(
                stage_name=stage["name"],
                subsystem=stage["subsystem"],
                operation=stage["operation"],
                inputs=inputs,
            )

            stage_results.append(result)

            # Update context with outputs
            if result.status == "completed":
                context.update(result.output)

            print(f"      -> {result.status} ({result.elapsed_ms:.1f}ms)")

        total_ms = (time.time() - start) * 1000
        final_status = "completed" if all(r.status == "completed" for r in stage_results) else "partial"

        return PipelineResult(
            status=final_status,
            stages=stage_results,
            final_output=context,
            total_ms=total_ms,
        )


# =============================================================================
# MAIN EXAMPLE
# =============================================================================

async def main():
    """Run cross-subsystem pipeline examples."""

    from nucleus.creative import emit_bus_event
    from nucleus.creative.bus.topics import PipelineTopics

    print("=" * 60)
    print("Cross-Subsystem Pipeline Examples")
    print("=" * 60)

    orchestrator = SimplePipelineOrchestrator()

    # =================================================================
    # Pipeline 1: Avatar Video Pipeline
    # =================================================================
    print("\n" + "=" * 60)
    print("Pipeline 1: Avatar Video Generation")
    print("=" * 60)

    avatar_video_stages = [
        {"name": "extract", "subsystem": "avatars", "operation": "extract", "params": {}},
        {"name": "gaussian_convert", "subsystem": "avatars", "operation": "gaussian_convert", "params": {}},
        {"name": "pbr_predict", "subsystem": "avatars", "operation": "pbr", "params": {}},
        {"name": "deform", "subsystem": "avatars", "operation": "deform", "params": {}},
        {"name": "frame_generate", "subsystem": "cinema", "operation": "frame_generate", "params": {}},
        {"name": "assemble", "subsystem": "cinema", "operation": "assemble", "params": {}},
    ]

    print("\n   Running avatar video pipeline...")
    result = await orchestrator.run_pipeline(
        stages=avatar_video_stages,
        initial_inputs={"image": np.random.randint(0, 255, (512, 512, 3))},
    )

    print(f"\n   Pipeline status: {result.status}")
    print(f"   Total time: {result.total_ms:.1f}ms")
    print(f"   Stages completed: {sum(1 for s in result.stages if s.status == 'completed')}/{len(result.stages)}")

    emit_bus_event(
        topic=PipelineTopics.GENERATION_COMPLETE,
        payload={
            "pipeline": "avatar_video",
            "status": result.status,
            "duration_ms": result.total_ms,
        },
    )

    # =================================================================
    # Pipeline 2: Voice Narrative Pipeline
    # =================================================================
    print("\n" + "=" * 60)
    print("Pipeline 2: Voice Narrative Generation")
    print("=" * 60)

    voice_narrative_stages = [
        {"name": "evaluate", "subsystem": "dits", "operation": "evaluate", "params": {}},
        {"name": "narrative", "subsystem": "dits", "operation": "narrative", "params": {}},
        {"name": "synthesize", "subsystem": "auralux", "operation": "synthesize",
         "params": {"text": "The hero embarks on a journey..."}},
    ]

    print("\n   Running voice narrative pipeline...")
    result = await orchestrator.run_pipeline(
        stages=voice_narrative_stages,
        initial_inputs={"environment": {"state": "initial"}},
    )

    print(f"\n   Pipeline status: {result.status}")
    print(f"   Total time: {result.total_ms:.1f}ms")
    if "text" in result.final_output:
        print(f"   Narrative: '{result.final_output['text'][:50]}...'")
    if "duration_s" in result.final_output:
        print(f"   Audio duration: {result.final_output['duration_s']:.1f}s")

    # =================================================================
    # Pipeline 3: Grammar Visualization Pipeline
    # =================================================================
    print("\n" + "=" * 60)
    print("Pipeline 3: Grammar Visualization")
    print("=" * 60)

    grammar_viz_stages = [
        {"name": "parse", "subsystem": "grammars", "operation": "parse", "params": {}},
        {"name": "synthesize", "subsystem": "grammars", "operation": "synthesize", "params": {}},
        {"name": "generate", "subsystem": "visual", "operation": "generate",
         "params": {"prompt": "Technical diagram of program structure"}},
        {"name": "upscale", "subsystem": "visual", "operation": "upscale", "params": {}},
    ]

    print("\n   Running grammar visualization pipeline...")
    result = await orchestrator.run_pipeline(
        stages=grammar_viz_stages,
        initial_inputs={"grammar_spec": "S -> A B | C"},
    )

    print(f"\n   Pipeline status: {result.status}")
    print(f"   Total time: {result.total_ms:.1f}ms")
    if "fitness" in result.final_output:
        print(f"   Synthesis fitness: {result.final_output['fitness']:.2f}")

    # =================================================================
    # Pipeline 4: Full Production Pipeline
    # =================================================================
    print("\n" + "=" * 60)
    print("Pipeline 4: Full Production (Avatar + Voice + Video)")
    print("=" * 60)

    # This pipeline has parallel branches that merge
    print("\n   Phase 1: Parallel extraction")

    # Run parallel stages
    avatar_task = orchestrator.execute_stage(
        "avatar_extract", "avatars", "extract",
        {"image": np.zeros((512, 512, 3))},
    )
    dits_task = orchestrator.execute_stage(
        "dits_evaluate", "dits", "evaluate",
        {"environment": {}},
    )

    avatar_result, dits_result = await asyncio.gather(avatar_task, dits_task)

    print(f"   Avatar extraction: {avatar_result.status}")
    print(f"   DiTS evaluation: {dits_result.status}")

    print("\n   Phase 2: Processing")

    # Sequential stages after parallel
    full_production_stages = [
        {"name": "gaussian_convert", "subsystem": "avatars", "operation": "gaussian_convert", "params": {}},
        {"name": "narrative", "subsystem": "dits", "operation": "narrative", "params": {}},
        {"name": "deform", "subsystem": "avatars", "operation": "deform", "params": {}},
        {"name": "synthesize", "subsystem": "auralux", "operation": "synthesize",
         "params": {"text": "Welcome to the full production demo"}},
        {"name": "frame_generate", "subsystem": "cinema", "operation": "frame_generate", "params": {}},
        {"name": "assemble", "subsystem": "cinema", "operation": "assemble", "params": {}},
    ]

    result = await orchestrator.run_pipeline(
        stages=full_production_stages,
        initial_inputs={
            **avatar_result.output,
            **dits_result.output,
        },
    )

    print(f"\n   Pipeline status: {result.status}")
    print(f"   Total time: {result.total_ms:.1f}ms")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)

    pipelines = [
        ("Avatar Video", ["avatars", "cinema"]),
        ("Voice Narrative", ["dits", "auralux"]),
        ("Grammar Visualization", ["grammars", "visual"]),
        ("Full Production", ["avatars", "dits", "auralux", "cinema"]),
    ]

    print("\n   Available pipelines:")
    for name, subsystems in pipelines:
        print(f"   - {name}: {' -> '.join(subsystems)}")

    print("\n   Subsystem interactions:")
    print("   - grammars -> visual: AST visualization")
    print("   - dits -> auralux: Narrative to speech")
    print("   - avatars -> cinema: 3D avatar to video")
    print("   - all: Full production pipeline")

    print("\n" + "=" * 60)
    print("Cross-subsystem pipeline examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
