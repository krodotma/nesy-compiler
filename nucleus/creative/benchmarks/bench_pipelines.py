"""
Pipeline Benchmarks
===================

Benchmarks for cross-subsystem pipelines combining multiple creative operations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Dict, List
from enum import Enum

from .bench_runner import BenchmarkSuite


class PipelineStage(Enum):
    """Pipeline processing stages."""
    INPUT = "input"
    PREPROCESS = "preprocess"
    GENERATE = "generate"
    POSTPROCESS = "postprocess"
    OUTPUT = "output"


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    success: bool
    stages_completed: List[str]
    total_time_ms: float
    output_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for a creative pipeline."""
    name: str
    stages: List[PipelineStage]
    params: Dict[str, Any]
    parallel: bool = False
    timeout_ms: float = 10000.0


class MockPipelineStage:
    """Mock pipeline stage for benchmarking."""

    def __init__(self, stage_type: PipelineStage, complexity: float = 1.0):
        self.stage_type = stage_type
        self.complexity = complexity

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through this stage."""
        # Simulate processing based on complexity
        iterations = int(1000 * self.complexity)
        result = 0.0

        for i in range(iterations):
            result += random.random()
            if i % 100 == 0:
                result = result / (i + 1)

        return {
            **data,
            f"{self.stage_type.value}_output": result,
            f"{self.stage_type.value}_complete": True,
        }


class CreativePipeline:
    """
    Cross-subsystem creative pipeline.

    Combines operations from multiple subsystems:
    - Grammars for structural generation
    - Visual for image synthesis
    - Auralux for audio synthesis
    - Avatars for 3D generation
    - DiTS for narrative control
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._stages: List[MockPipelineStage] = [
            MockPipelineStage(stage, complexity=1.0)
            for stage in config.stages
        ]

    def run(self, input_data: Dict[str, Any]) -> PipelineResult:
        """Execute the pipeline."""
        data = input_data.copy()
        completed: List[str] = []
        total_time = 0.0

        for stage in self._stages:
            start = random.random() * 10  # Simulated timing
            data = stage.process(data)
            elapsed = random.random() * 10

            completed.append(stage.stage_type.value)
            total_time += elapsed

        return PipelineResult(
            success=True,
            stages_completed=completed,
            total_time_ms=total_time,
            output_size=len(str(data)),
            metadata={"config": self.config.name},
        )


class TextToVideoWithNarration:
    """
    Pipeline: Text -> Visual -> Auralux -> Cinema

    Generates video with narration from text prompt.
    """

    def __init__(self, visual_steps: int = 30, audio_quality: str = "medium"):
        self.visual_steps = visual_steps
        self.audio_quality = audio_quality

    def generate(self, prompt: str) -> PipelineResult:
        """Generate video with narration from prompt."""
        stages_completed = []

        # Stage 1: Parse and structure prompt
        structure = self._parse_prompt(prompt)
        stages_completed.append("parse")

        # Stage 2: Generate visual frames
        frames = self._generate_frames(structure, self.visual_steps)
        stages_completed.append("visual")

        # Stage 3: Generate narration audio
        audio = self._generate_audio(prompt)
        stages_completed.append("audio")

        # Stage 4: Composite video
        video = self._composite_video(frames, audio)
        stages_completed.append("composite")

        return PipelineResult(
            success=True,
            stages_completed=stages_completed,
            total_time_ms=random.uniform(100, 500),
            output_size=len(frames) * 1000 + len(audio),
            metadata={
                "n_frames": len(frames),
                "audio_samples": len(audio),
                "prompt_length": len(prompt),
            },
        )

    def _parse_prompt(self, prompt: str) -> Dict[str, Any]:
        """Parse and structure the prompt."""
        words = prompt.lower().split()
        return {
            "tokens": words,
            "length": len(words),
            "sentiment": sum(1 if w in ["happy", "bright", "good"] else -1 if w in ["sad", "dark", "bad"] else 0 for w in words),
        }

    def _generate_frames(self, structure: Dict[str, Any], n_steps: int) -> List[List[float]]:
        """Generate visual frames."""
        frames = []
        for i in range(n_steps):
            # Simulate frame generation
            frame = [random.random() for _ in range(64 * 64 * 3)]
            frames.append(frame)
        return frames

    def _generate_audio(self, text: str) -> List[float]:
        """Generate narration audio."""
        # Simulate TTS
        samples_per_char = 500
        return [random.gauss(0, 0.1) for _ in range(len(text) * samples_per_char)]

    def _composite_video(self, frames: List[List[float]], audio: List[float]) -> Dict[str, Any]:
        """Composite frames and audio into video."""
        return {
            "n_frames": len(frames),
            "frame_rate": 24,
            "duration_sec": len(frames) / 24,
            "audio_synced": True,
        }


class GrammarGuidedImageGeneration:
    """
    Pipeline: Grammar -> DiTS -> Visual

    Uses grammar evolution to guide image generation through narrative structure.
    """

    def __init__(self, grammar_complexity: int = 50, dits_depth: int = 10):
        self.grammar_complexity = grammar_complexity
        self.dits_depth = dits_depth

    def generate(self, seed_grammar: str, visual_prompt: str) -> PipelineResult:
        """Generate image guided by evolved grammar."""
        stages_completed = []

        # Stage 1: Evolve grammar
        evolved = self._evolve_grammar(seed_grammar)
        stages_completed.append("grammar_evolve")

        # Stage 2: Generate narrative structure via DiTS
        narrative = self._generate_narrative(evolved)
        stages_completed.append("dits_narrative")

        # Stage 3: Generate image from narrative
        image = self._generate_image(visual_prompt, narrative)
        stages_completed.append("visual_generate")

        return PipelineResult(
            success=True,
            stages_completed=stages_completed,
            total_time_ms=random.uniform(50, 200),
            output_size=len(image),
            metadata={
                "grammar_nodes": self.grammar_complexity,
                "narrative_episodes": len(narrative),
            },
        )

    def _evolve_grammar(self, seed: str) -> List[str]:
        """Evolve grammar from seed."""
        rules = [seed]
        for _ in range(self.grammar_complexity):
            # Simulate grammar evolution
            new_rule = random.choice(rules) + f"_v{random.randint(0, 100)}"
            rules.append(new_rule)
        return rules

    def _generate_narrative(self, grammar: List[str]) -> List[Dict[str, Any]]:
        """Generate narrative from grammar."""
        episodes = []
        for i in range(self.dits_depth):
            episodes.append({
                "id": i,
                "content": grammar[i % len(grammar)],
                "transitions": [j for j in range(max(0, i-2), min(len(grammar), i+3))],
            })
        return episodes

    def _generate_image(self, prompt: str, narrative: List[Dict[str, Any]]) -> List[float]:
        """Generate image from prompt and narrative context."""
        # Simulate image generation influenced by narrative
        context_weight = len(narrative) / 10
        return [random.gauss(0.5, 0.2) * context_weight for _ in range(256 * 256 * 3)]


class AvatarAnimation:
    """
    Pipeline: Auralux (STT) -> DiTS -> Avatars

    Animates avatar based on speech input and narrative context.
    """

    def __init__(self, use_lip_sync: bool = True, n_gaussians: int = 10000):
        self.use_lip_sync = use_lip_sync
        self.n_gaussians = n_gaussians

    def animate(self, audio_samples: List[float], duration_sec: float) -> PipelineResult:
        """Animate avatar from audio input."""
        stages_completed = []

        # Stage 1: Transcribe audio
        transcript = self._transcribe(audio_samples)
        stages_completed.append("stt")

        # Stage 2: Generate animation narrative
        narrative = self._generate_animation_narrative(transcript)
        stages_completed.append("dits_animation")

        # Stage 3: Generate SMPLX poses
        poses = self._generate_poses(narrative, duration_sec)
        stages_completed.append("smplx_poses")

        # Stage 4: Deform Gaussians
        if self.use_lip_sync:
            gaussians = self._apply_lip_sync(poses, transcript)
            stages_completed.append("lip_sync")
        else:
            gaussians = self._deform_gaussians(poses)

        stages_completed.append("gaussian_deform")

        return PipelineResult(
            success=True,
            stages_completed=stages_completed,
            total_time_ms=random.uniform(200, 800),
            output_size=self.n_gaussians * 50,  # Approx bytes per Gaussian
            metadata={
                "n_poses": len(poses),
                "n_gaussians": self.n_gaussians,
                "duration_sec": duration_sec,
            },
        )

    def _transcribe(self, audio: List[float]) -> List[Dict[str, Any]]:
        """Transcribe audio to words with timings."""
        # Simulate transcription
        words = []
        n_words = len(audio) // 5000  # ~5000 samples per word
        for i in range(n_words):
            words.append({
                "word": f"word_{i}",
                "start": i * 0.5,
                "end": (i + 1) * 0.5 - 0.1,
                "phonemes": ["p", "h", "o", "n"],
            })
        return words

    def _generate_animation_narrative(self, transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate animation sequence from transcript."""
        return [
            {
                "frame": i * 24,  # 24fps
                "word": transcript[i % len(transcript)]["word"] if transcript else "silence",
                "emotion": random.choice(["neutral", "happy", "speaking"]),
            }
            for i in range(len(transcript) * 12)  # 12 frames per word
        ]

    def _generate_poses(self, narrative: List[Dict[str, Any]], duration: float) -> List[List[float]]:
        """Generate SMPLX poses from narrative."""
        n_frames = int(duration * 24)
        poses = []
        for i in range(n_frames):
            # 63 body pose params + 90 hand + 50 face = 203 total
            pose = [random.gauss(0, 0.1) for _ in range(203)]
            poses.append(pose)
        return poses

    def _apply_lip_sync(self, poses: List[List[float]], transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply lip sync to Gaussians."""
        result = []
        for i, pose in enumerate(poses):
            # Find corresponding phoneme
            word_idx = i // 12  # 12 frames per word
            if word_idx < len(transcript) and transcript[word_idx].get("phonemes"):
                phoneme = transcript[word_idx]["phonemes"][i % len(transcript[word_idx]["phonemes"])]
            else:
                phoneme = "silence"

            result.append({
                "frame": i,
                "pose": pose[:10],  # Truncated for efficiency
                "lip_shape": phoneme,
                "n_deformed": self.n_gaussians,
            })
        return result

    def _deform_gaussians(self, poses: List[List[float]]) -> List[Dict[str, Any]]:
        """Deform Gaussians without lip sync."""
        return [
            {
                "frame": i,
                "pose": pose[:10],
                "n_deformed": self.n_gaussians,
            }
            for i, pose in enumerate(poses)
        ]


class MultiModalSynthesis:
    """
    Pipeline: All subsystems

    Full multimodal synthesis combining all creative subsystems.
    """

    def __init__(self):
        self.text_to_video = TextToVideoWithNarration()
        self.grammar_guided = GrammarGuidedImageGeneration()
        self.avatar_anim = AvatarAnimation()

    def synthesize(self, prompt: str) -> PipelineResult:
        """Run full multimodal synthesis."""
        stages_completed = []

        # Grammar-guided scene generation
        scene_result = self.grammar_guided.generate("scene", prompt)
        stages_completed.extend(["grammar:" + s for s in scene_result.stages_completed])

        # Text-to-video for background
        video_result = self.text_to_video.generate(prompt)
        stages_completed.extend(["video:" + s for s in video_result.stages_completed])

        # Avatar animation
        audio = [random.gauss(0, 0.1) for _ in range(44100 * 5)]  # 5 seconds
        avatar_result = self.avatar_anim.animate(audio, 5.0)
        stages_completed.extend(["avatar:" + s for s in avatar_result.stages_completed])

        # Final composition
        stages_completed.append("final_composite")

        total_time = (
            scene_result.total_time_ms +
            video_result.total_time_ms +
            avatar_result.total_time_ms +
            random.uniform(50, 100)  # Composition time
        )

        return PipelineResult(
            success=True,
            stages_completed=stages_completed,
            total_time_ms=total_time,
            output_size=scene_result.output_size + video_result.output_size + avatar_result.output_size,
            metadata={
                "prompt": prompt,
                "scene_stages": len(scene_result.stages_completed),
                "video_stages": len(video_result.stages_completed),
                "avatar_stages": len(avatar_result.stages_completed),
            },
        )


class PipelinesBenchmark(BenchmarkSuite):
    """Benchmark suite for cross-subsystem pipelines."""

    @property
    def name(self) -> str:
        return "pipelines"

    @property
    def description(self) -> str:
        return "Cross-subsystem pipeline benchmarks"

    def __init__(self):
        self._text_to_video: Optional[TextToVideoWithNarration] = None
        self._grammar_guided: Optional[GrammarGuidedImageGeneration] = None
        self._avatar_anim: Optional[AvatarAnimation] = None
        self._multimodal: Optional[MultiModalSynthesis] = None
        self._test_audio: List[float] = []
        self._test_prompts: List[str] = []

    def setup(self) -> None:
        """Setup pipelines and test data."""
        self._text_to_video = TextToVideoWithNarration(visual_steps=30)
        self._grammar_guided = GrammarGuidedImageGeneration(grammar_complexity=50)
        self._avatar_anim = AvatarAnimation(n_gaussians=10000)
        self._multimodal = MultiModalSynthesis()

        # Test data
        self._test_audio = [random.gauss(0, 0.1) for _ in range(44100 * 3)]  # 3 seconds
        self._test_prompts = [
            "A beautiful sunset over mountains",
            "A futuristic city at night",
            "An ancient forest with magical creatures",
            "A serene lake reflecting the sky",
            "A bustling marketplace in a fantasy world",
        ]

    def get_benchmarks(self) -> list[tuple[str, Callable[[], Any]]]:
        """Get all pipeline benchmarks."""
        return [
            # Basic pipeline operations
            ("pipeline_create", self._pipeline_create),
            ("pipeline_run_simple", self._pipeline_run_simple),
            ("pipeline_run_complex", self._pipeline_run_complex),

            # Text to video pipeline
            ("t2v_short", self._t2v_short),
            ("t2v_medium", self._t2v_medium),
            ("t2v_long", self._t2v_long),

            # Grammar-guided generation
            ("grammar_visual_simple", self._grammar_visual_simple),
            ("grammar_visual_complex", self._grammar_visual_complex),

            # Avatar animation pipeline
            ("avatar_short", self._avatar_short),
            ("avatar_with_lipsync", self._avatar_with_lipsync),

            # Multi-modal synthesis
            ("multimodal_simple", self._multimodal_simple),
            ("multimodal_full", self._multimodal_full),

            # Pipeline combinations
            ("parallel_pipelines", self._parallel_pipelines),
            ("sequential_pipelines", self._sequential_pipelines),
        ]

    def _pipeline_create(self) -> CreativePipeline:
        """Create basic pipeline."""
        config = PipelineConfig(
            name="basic",
            stages=[PipelineStage.INPUT, PipelineStage.GENERATE, PipelineStage.OUTPUT],
            params={"quality": "medium"},
        )
        return CreativePipeline(config)

    def _pipeline_run_simple(self) -> PipelineResult:
        """Run simple pipeline."""
        pipeline = self._pipeline_create()
        return pipeline.run({"prompt": "test"})

    def _pipeline_run_complex(self) -> PipelineResult:
        """Run complex pipeline."""
        config = PipelineConfig(
            name="complex",
            stages=list(PipelineStage),
            params={"quality": "high", "iterations": 100},
        )
        pipeline = CreativePipeline(config)
        return pipeline.run({"prompt": self._test_prompts[0], "seed": 42})

    def _t2v_short(self) -> PipelineResult:
        """Short text-to-video (5 word prompt)."""
        return self._text_to_video.generate("sunset over mountains")

    def _t2v_medium(self) -> PipelineResult:
        """Medium text-to-video."""
        return self._text_to_video.generate(self._test_prompts[0])

    def _t2v_long(self) -> PipelineResult:
        """Long text-to-video (50+ words)."""
        long_prompt = " ".join(self._test_prompts)
        return self._text_to_video.generate(long_prompt)

    def _grammar_visual_simple(self) -> PipelineResult:
        """Simple grammar-guided generation."""
        return self._grammar_guided.generate("S -> NP VP", "forest scene")

    def _grammar_visual_complex(self) -> PipelineResult:
        """Complex grammar-guided generation."""
        self._grammar_guided.grammar_complexity = 100
        self._grammar_guided.dits_depth = 20
        result = self._grammar_guided.generate(
            "S -> NP VP | S CONJ S",
            self._test_prompts[2],
        )
        self._grammar_guided.grammar_complexity = 50
        self._grammar_guided.dits_depth = 10
        return result

    def _avatar_short(self) -> PipelineResult:
        """Short avatar animation (1 second)."""
        short_audio = self._test_audio[:44100]
        return self._avatar_anim.animate(short_audio, 1.0)

    def _avatar_with_lipsync(self) -> PipelineResult:
        """Avatar animation with lip sync."""
        return self._avatar_anim.animate(self._test_audio, 3.0)

    def _multimodal_simple(self) -> PipelineResult:
        """Simple multimodal synthesis."""
        return self._multimodal.synthesize("sunset")

    def _multimodal_full(self) -> PipelineResult:
        """Full multimodal synthesis."""
        return self._multimodal.synthesize(self._test_prompts[0])

    def _parallel_pipelines(self) -> List[PipelineResult]:
        """Run multiple pipelines (simulated parallel)."""
        results = []
        for prompt in self._test_prompts[:3]:
            result = self._text_to_video.generate(prompt)
            results.append(result)
        return results

    def _sequential_pipelines(self) -> PipelineResult:
        """Run sequential dependent pipelines."""
        # First: grammar generation
        grammar_result = self._grammar_guided.generate("S -> A B", "input")

        # Second: use output in text-to-video
        video_result = self._text_to_video.generate(
            f"scene with {grammar_result.output_size} elements"
        )

        # Third: avatar based on video
        avatar_result = self._avatar_anim.animate(
            self._test_audio[:int(video_result.total_time_ms * 44.1)],
            video_result.total_time_ms / 1000,
        )

        return PipelineResult(
            success=True,
            stages_completed=(
                grammar_result.stages_completed +
                video_result.stages_completed +
                avatar_result.stages_completed
            ),
            total_time_ms=(
                grammar_result.total_time_ms +
                video_result.total_time_ms +
                avatar_result.total_time_ms
            ),
            output_size=avatar_result.output_size,
            metadata={
                "pipeline_type": "sequential",
                "n_stages": (
                    len(grammar_result.stages_completed) +
                    len(video_result.stages_completed) +
                    len(avatar_result.stages_completed)
                ),
            },
        )
