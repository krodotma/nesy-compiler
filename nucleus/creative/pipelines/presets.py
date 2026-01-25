"""
Pipeline Presets
=================

Pre-configured pipelines for common creative workflows.

Presets:
- visual_generation: Image generation with optional upscaling and style
- cinema_generation: Video generation from script to final output
- avatar_creation: 3D avatar generation and animation
- audio_synthesis: Voice and audio generation
- full_production: Complete multi-subsystem production pipeline

Example:
    >>> from nucleus.creative.pipelines import get_preset, PIPELINE_PRESETS
    >>> preset = get_preset("visual_generation")
    >>> orchestrator = PipelineOrchestrator()
    >>> result = await orchestrator.execute(preset.to_config())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .orchestrator import (
    PipelineConfig,
    StageConfig,
    RecoveryConfig,
    RecoveryStrategy,
)


# =============================================================================
# PRESET TYPES
# =============================================================================


@dataclass
class StagePreset:
    """Preset configuration for a pipeline stage."""

    name: str
    subsystem: str
    operation: str
    description: str = ""
    params: dict = field(default_factory=dict)
    timeout_s: float = 300.0
    required: bool = True
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    max_retries: int = 3
    depends_on: List[str] = field(default_factory=list)

    def to_stage_config(self) -> StageConfig:
        """Convert to StageConfig."""
        return StageConfig(
            name=self.name,
            subsystem=self.subsystem,
            operation=self.operation,
            params=dict(self.params),
            timeout_s=self.timeout_s,
            required=self.required,
            recovery=RecoveryConfig(
                strategy=self.recovery_strategy,
                max_retries=self.max_retries,
            ),
            depends_on=list(self.depends_on),
        )


@dataclass
class PipelinePreset:
    """Preset configuration for a complete pipeline."""

    name: str
    description: str
    stages: List[StagePreset]
    version: str = "1.0.0"
    timeout_s: float = 1800.0
    parallel_stages: bool = False
    checkpointing: bool = True
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_config(
        self,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> PipelineConfig:
        """
        Convert to PipelineConfig with optional parameter overrides.

        Args:
            overrides: Dict mapping stage names to param overrides

        Returns:
            PipelineConfig ready for execution
        """
        overrides = overrides or {}
        stages = []

        for stage_preset in self.stages:
            stage_config = stage_preset.to_stage_config()

            # Apply overrides if present
            if stage_config.name in overrides:
                stage_config.params.update(overrides[stage_config.name])

            stages.append(stage_config)

        return PipelineConfig(
            name=self.name,
            description=self.description,
            version=self.version,
            stages=stages,
            timeout_s=self.timeout_s,
            parallel_stages=self.parallel_stages,
            checkpointing=self.checkpointing,
            metadata={
                **self.metadata,
                "category": self.category,
                "tags": self.tags,
            },
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "tags": self.tags,
            "stages": [
                {
                    "name": s.name,
                    "subsystem": s.subsystem,
                    "operation": s.operation,
                    "description": s.description,
                }
                for s in self.stages
            ],
            "timeout_s": self.timeout_s,
            "parallel_stages": self.parallel_stages,
        }


# =============================================================================
# VISUAL GENERATION PRESET
# =============================================================================

_VISUAL_GENERATION = PipelinePreset(
    name="visual_generation",
    description="Image generation pipeline with optional upscaling and style transfer",
    category="visual",
    tags=["image", "generation", "diffusion"],
    stages=[
        StagePreset(
            name="generate",
            subsystem="visual",
            operation="generate",
            description="Generate base image from prompt",
            params={
                "width": 1024,
                "height": 1024,
                "guidance_scale": 7.5,
                "num_inference_steps": 30,
            },
            timeout_s=120.0,
        ),
        StagePreset(
            name="upscale",
            subsystem="visual",
            operation="upscale",
            description="Upscale image to higher resolution",
            params={
                "scale": 2,
                "model": "real-esrgan",
            },
            timeout_s=60.0,
            required=False,
            depends_on=["generate"],
        ),
        StagePreset(
            name="style",
            subsystem="visual",
            operation="style_transfer",
            description="Apply optional style transfer",
            params={
                "strength": 0.5,
            },
            timeout_s=60.0,
            required=False,
            depends_on=["upscale"],
        ),
    ],
)

# =============================================================================
# CINEMA GENERATION PRESET
# =============================================================================

_CINEMA_GENERATION = PipelinePreset(
    name="cinema_generation",
    description="Video generation pipeline from script to final video",
    category="cinema",
    tags=["video", "generation", "narrative"],
    timeout_s=3600.0,  # 1 hour for video
    stages=[
        StagePreset(
            name="parse_script",
            subsystem="cinema",
            operation="parse_script",
            description="Parse screenplay into structured scenes",
            timeout_s=30.0,
        ),
        StagePreset(
            name="storyboard",
            subsystem="cinema",
            operation="storyboard",
            description="Generate storyboard panels from script",
            params={
                "shots_per_scene": 3,
                "include_camera_directions": True,
            },
            timeout_s=120.0,
            depends_on=["parse_script"],
        ),
        StagePreset(
            name="frame_generate",
            subsystem="cinema",
            operation="frame_generate",
            description="Generate video frames for each shot",
            params={
                "fps": 24,
                "width": 1920,
                "height": 1080,
            },
            timeout_s=1800.0,  # 30 min for frame generation
            depends_on=["storyboard"],
        ),
        StagePreset(
            name="temporal_consistency",
            subsystem="cinema",
            operation="temporal_consistency",
            description="Apply temporal consistency across frames",
            params={
                "strength": 0.8,
                "method": "optical_flow",
            },
            timeout_s=600.0,
            depends_on=["frame_generate"],
        ),
        StagePreset(
            name="assemble",
            subsystem="cinema",
            operation="assemble",
            description="Assemble final video with transitions",
            params={
                "preset": "web_hd",
                "transition_type": "dissolve",
                "transition_duration": 0.5,
            },
            timeout_s=300.0,
            depends_on=["temporal_consistency"],
        ),
    ],
)

# =============================================================================
# AVATAR CREATION PRESET
# =============================================================================

_AVATAR_CREATION = PipelinePreset(
    name="avatar_creation",
    description="3D avatar generation from image with animation support",
    category="avatars",
    tags=["3d", "avatar", "smpl-x", "gaussian-splatting"],
    timeout_s=1800.0,
    stages=[
        StagePreset(
            name="extract",
            subsystem="avatars",
            operation="extract",
            description="Extract SMPL-X parameters from input image",
            params={
                "model": "smpl-x",
                "include_hands": True,
                "include_face": True,
            },
            timeout_s=120.0,
        ),
        StagePreset(
            name="gaussian_convert",
            subsystem="avatars",
            operation="gaussian_convert",
            description="Convert mesh to 3D Gaussian splatting representation",
            params={
                "gaussian_count": 50000,
                "sh_degree": 3,
            },
            timeout_s=300.0,
            depends_on=["extract"],
        ),
        StagePreset(
            name="pbr_predict",
            subsystem="avatars",
            operation="pbr_predict",
            description="Predict PBR materials for avatar",
            params={
                "resolution": 1024,
                "include_subsurface": True,
            },
            timeout_s=180.0,
            depends_on=["extract"],
        ),
        StagePreset(
            name="deform",
            subsystem="avatars",
            operation="deform",
            description="Apply pose deformation to gaussians",
            params={
                "animation_frames": 30,
                "blend_shapes": True,
            },
            timeout_s=300.0,
            depends_on=["gaussian_convert", "pbr_predict"],
        ),
        StagePreset(
            name="render",
            subsystem="avatars",
            operation="render",
            description="Render final avatar frames",
            params={
                "width": 1920,
                "height": 1080,
                "samples": 64,
            },
            timeout_s=600.0,
            depends_on=["deform"],
        ),
    ],
)

# =============================================================================
# AUDIO SYNTHESIS PRESET
# =============================================================================

_AUDIO_SYNTHESIS = PipelinePreset(
    name="audio_synthesis",
    description="Voice and audio synthesis pipeline",
    category="auralux",
    tags=["audio", "voice", "tts", "synthesis"],
    stages=[
        StagePreset(
            name="text_process",
            subsystem="auralux",
            operation="text_process",
            description="Process and normalize input text",
            params={
                "normalize_numbers": True,
                "expand_abbreviations": True,
            },
            timeout_s=30.0,
        ),
        StagePreset(
            name="synthesize",
            subsystem="auralux",
            operation="synthesize",
            description="Synthesize speech from text",
            params={
                "model": "default",
                "sample_rate": 24000,
            },
            timeout_s=120.0,
            depends_on=["text_process"],
        ),
        StagePreset(
            name="enhance",
            subsystem="auralux",
            operation="enhance",
            description="Enhance audio quality",
            params={
                "denoise": True,
                "normalize": True,
                "target_lufs": -16.0,
            },
            timeout_s=60.0,
            depends_on=["synthesize"],
        ),
        StagePreset(
            name="export",
            subsystem="auralux",
            operation="export",
            description="Export final audio file",
            params={
                "format": "wav",
                "bit_depth": 16,
            },
            timeout_s=30.0,
            depends_on=["enhance"],
        ),
    ],
)

# =============================================================================
# VOICE CLONE PRESET
# =============================================================================

_VOICE_CLONE = PipelinePreset(
    name="voice_clone",
    description="Voice cloning and synthesis pipeline",
    category="auralux",
    tags=["audio", "voice", "cloning", "tts"],
    stages=[
        StagePreset(
            name="extract_embedding",
            subsystem="auralux",
            operation="extract_embedding",
            description="Extract speaker embedding from reference audio",
            params={
                "model": "speaker_encoder",
            },
            timeout_s=60.0,
        ),
        StagePreset(
            name="synthesize",
            subsystem="auralux",
            operation="synthesize_with_embedding",
            description="Synthesize speech using cloned voice",
            params={
                "model": "voice_clone",
            },
            timeout_s=180.0,
            depends_on=["extract_embedding"],
        ),
        StagePreset(
            name="enhance",
            subsystem="auralux",
            operation="enhance",
            description="Enhance synthesized audio",
            timeout_s=60.0,
            depends_on=["synthesize"],
        ),
    ],
)

# =============================================================================
# GRAMMAR EVOLUTION PRESET
# =============================================================================

_GRAMMAR_EVOLUTION = PipelinePreset(
    name="grammar_evolution",
    description="Grammar synthesis and evolution pipeline",
    category="grammars",
    tags=["grammar", "cgp", "evolution", "synthesis"],
    stages=[
        StagePreset(
            name="parse",
            subsystem="grammars",
            operation="parse",
            description="Parse grammar specification",
            timeout_s=30.0,
        ),
        StagePreset(
            name="synthesize",
            subsystem="grammars",
            operation="synthesize",
            description="Synthesize program from grammar",
            params={
                "method": "cgp",
                "population_size": 100,
            },
            timeout_s=300.0,
            depends_on=["parse"],
        ),
        StagePreset(
            name="evolve",
            subsystem="grammars",
            operation="evolve",
            description="Evolve synthesized program",
            params={
                "generations": 50,
                "mutation_rate": 0.1,
            },
            timeout_s=600.0,
            depends_on=["synthesize"],
        ),
        StagePreset(
            name="visualize",
            subsystem="grammars",
            operation="visualize",
            description="Visualize evolved grammar as AST",
            timeout_s=30.0,
            required=False,
            depends_on=["evolve"],
        ),
    ],
)

# =============================================================================
# DITS NARRATIVE PRESET
# =============================================================================

_DITS_NARRATIVE = PipelinePreset(
    name="dits_narrative",
    description="DiTS narrative construction pipeline",
    category="dits",
    tags=["dits", "narrative", "mu-calculus", "story"],
    stages=[
        StagePreset(
            name="evaluate",
            subsystem="dits",
            operation="evaluate",
            description="Evaluate current DiTS state",
            timeout_s=30.0,
        ),
        StagePreset(
            name="descend",
            subsystem="dits",
            operation="descend",
            description="Descend through mu-calculus ranks",
            params={
                "max_depth": 5,
            },
            timeout_s=120.0,
            depends_on=["evaluate"],
        ),
        StagePreset(
            name="construct",
            subsystem="dits",
            operation="construct",
            description="Construct narrative from descent",
            timeout_s=180.0,
            depends_on=["descend"],
        ),
    ],
)

# =============================================================================
# FULL PRODUCTION PRESET
# =============================================================================

_FULL_PRODUCTION = PipelinePreset(
    name="full_production",
    description="Complete multi-subsystem production pipeline",
    category="production",
    tags=["full", "production", "avatar", "video", "audio"],
    timeout_s=7200.0,  # 2 hours
    parallel_stages=False,
    stages=[
        # Avatar extraction phase
        StagePreset(
            name="avatar_extract",
            subsystem="avatars",
            operation="extract",
            description="Extract avatar from input image",
            timeout_s=120.0,
        ),
        # DiTS narrative phase (can run parallel with avatar)
        StagePreset(
            name="dits_evaluate",
            subsystem="dits",
            operation="evaluate",
            description="Evaluate narrative state",
            timeout_s=60.0,
        ),
        StagePreset(
            name="dits_narrative",
            subsystem="dits",
            operation="construct",
            description="Construct narrative",
            timeout_s=180.0,
            depends_on=["dits_evaluate"],
        ),
        # Voice synthesis
        StagePreset(
            name="voice_synthesize",
            subsystem="auralux",
            operation="synthesize",
            description="Synthesize narrative voice",
            timeout_s=180.0,
            depends_on=["dits_narrative"],
        ),
        # Avatar animation
        StagePreset(
            name="avatar_gaussian",
            subsystem="avatars",
            operation="gaussian_convert",
            description="Convert to gaussians",
            timeout_s=300.0,
            depends_on=["avatar_extract"],
        ),
        StagePreset(
            name="avatar_deform",
            subsystem="avatars",
            operation="deform",
            description="Deform with animation",
            timeout_s=300.0,
            depends_on=["avatar_gaussian"],
        ),
        # Video generation
        StagePreset(
            name="frame_generate",
            subsystem="cinema",
            operation="frame_generate",
            description="Generate video frames",
            timeout_s=1800.0,
            depends_on=["avatar_deform"],
        ),
        # Final assembly
        StagePreset(
            name="assemble",
            subsystem="cinema",
            operation="assemble",
            description="Assemble final video with audio",
            timeout_s=600.0,
            depends_on=["frame_generate", "voice_synthesize"],
        ),
    ],
)

# =============================================================================
# PRESETS REGISTRY
# =============================================================================

PIPELINE_PRESETS: Dict[str, PipelinePreset] = {
    "visual_generation": _VISUAL_GENERATION,
    "cinema_generation": _CINEMA_GENERATION,
    "avatar_creation": _AVATAR_CREATION,
    "audio_synthesis": _AUDIO_SYNTHESIS,
    "voice_clone": _VOICE_CLONE,
    "grammar_evolution": _GRAMMAR_EVOLUTION,
    "dits_narrative": _DITS_NARRATIVE,
    "full_production": _FULL_PRODUCTION,
}


# =============================================================================
# PRESET UTILITIES
# =============================================================================


def get_preset(name: str) -> Optional[PipelinePreset]:
    """
    Get a pipeline preset by name.

    Args:
        name: Preset name

    Returns:
        PipelinePreset if found, None otherwise
    """
    return PIPELINE_PRESETS.get(name)


def list_presets(category: Optional[str] = None) -> List[PipelinePreset]:
    """
    List available presets.

    Args:
        category: Optional category filter

    Returns:
        List of matching presets
    """
    presets = list(PIPELINE_PRESETS.values())
    if category:
        presets = [p for p in presets if p.category == category]
    return presets


def get_preset_names() -> List[str]:
    """Get list of all preset names."""
    return list(PIPELINE_PRESETS.keys())


def get_categories() -> List[str]:
    """Get list of all preset categories."""
    return list(set(p.category for p in PIPELINE_PRESETS.values()))


def create_custom_preset(
    name: str,
    base_preset: str,
    stage_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs,
) -> PipelinePreset:
    """
    Create a custom preset based on an existing one.

    Args:
        name: Name for the new preset
        base_preset: Name of the base preset
        stage_overrides: Stage parameter overrides
        **kwargs: Additional PipelinePreset attributes to override

    Returns:
        New PipelinePreset

    Raises:
        ValueError: If base preset not found
    """
    base = get_preset(base_preset)
    if base is None:
        raise ValueError(f"Base preset '{base_preset}' not found")

    stage_overrides = stage_overrides or {}

    # Copy and modify stages
    new_stages = []
    for stage in base.stages:
        new_stage = StagePreset(
            name=stage.name,
            subsystem=stage.subsystem,
            operation=stage.operation,
            description=stage.description,
            params={**stage.params, **stage_overrides.get(stage.name, {})},
            timeout_s=stage.timeout_s,
            required=stage.required,
            recovery_strategy=stage.recovery_strategy,
            max_retries=stage.max_retries,
            depends_on=list(stage.depends_on),
        )
        new_stages.append(new_stage)

    return PipelinePreset(
        name=name,
        description=kwargs.get("description", base.description),
        stages=new_stages,
        version=kwargs.get("version", base.version),
        timeout_s=kwargs.get("timeout_s", base.timeout_s),
        parallel_stages=kwargs.get("parallel_stages", base.parallel_stages),
        checkpointing=kwargs.get("checkpointing", base.checkpointing),
        category=kwargs.get("category", base.category),
        tags=kwargs.get("tags", list(base.tags)),
        metadata=kwargs.get("metadata", dict(base.metadata)),
    )
