"""
Pluribus Creative Section
=========================

Unified multimodal AI/ML subsystem for generative content creation.

Subsystems:
- grammars: CGP/EGGP synthesis, metagrammars, AST visualization
- cinema: Video generation, temporal consistency, multi-shot narrative
- visual: Image generation, style transfer, diffusion models
- auralux: Voice synthesis, TTS/STT, speaker embeddings
- avatars: 3DGS, Neural PBR, SMPL-X, procedural parameters
- dits: Diegetic Transition System, narrative constructivity, μ/ν calculus

Architecture layers (from Theia):
- L0: Vision Capture + Generation
- L1: Geometric Substrate (S^n ⊣ H^n)
- L2: Modern Hopfield Continuum (mHC)
- L3: Birkhoff Polytope Dynamics
- L4: DNA (Dual Neurosymbolic Automata)
- L5: Ω Reflexive Domain
"""

__version__ = "1.4.0"
__all__ = [
    # Subsystems
    "grammars",
    "cinema",
    "visual",
    "auralux",
    "avatars",
    "dits",
    # Core Types
    "CreativeMode",
    "PipelineStage",
    "Asset",
    "GenerationJob",
    "CreativeState",
    # Shared Types (from types module)
    "types",
    "ImageArray",
    "AudioArray",
    "MeshVertices",
    "ProgressCallback",
    "StatusType",
    "QualityLevel",
    "ProcessingMode",
    "Success",
    "Failure",
    "Result",
    "ValidationResult",
    "BoundingBox",
    "TimeRange",
    "OperationMetrics",
    # Errors (from errors module)
    "errors",
    "ErrorCode",
    "CreativeError",
    "ValidationError",
    "ProcessingError",
    "ResourceError",
    "ConfigurationError",
    "ProviderError",
    "GrammarsError",
    "CinemaError",
    "VisualError",
    "AuraluxError",
    "AvatarsError",
    "DiTSError",
    # Resilience (from resilience module)
    "resilience",
    "retry",
    "circuit_breaker",
    "timeout",
    "fallback",
    "resilient",
    "ResilienceError",
    "RetryExhaustedError",
    "CircuitBreakerOpenError",
    "TimeoutExceededError",
    "FallbackError",
    # Health (from health module)
    "health",
    "HealthStatus",
    "SubsystemHealth",
    "HealthCheckResult",
    "HealthChecker",
    "check_health",
    "is_healthy",
    # Observability (from observability package)
    "observability",
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "timed",
    "get_metrics",
    "Span",
    "Tracer",
    "traced",
    "get_current_trace_id",
    "StructuredLogger",
    "LogLevel",
    "get_logger",
    # Utilities
    "BUS_TOPICS",
    "SUBSYSTEMS",
    "emit_bus_event",
    "create_pipeline",
    "run_pipeline",
]

from typing import Literal, TypedDict, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

# Import shared types module
from . import types
from .types import (
    ImageArray,
    AudioArray,
    MeshVertices,
    ProgressCallback,
    StatusType,
    QualityLevel,
    ProcessingMode,
    Success,
    Failure,
    Result,
    ValidationResult,
    BoundingBox,
    TimeRange,
    OperationMetrics,
)

# Import errors module
from . import errors
from .errors import (
    ErrorCode,
    CreativeError,
    ValidationError,
    ProcessingError,
    ResourceError,
    ConfigurationError,
    ProviderError,
    GrammarsError,
    CinemaError,
    VisualError,
    AuraluxError,
    AvatarsError,
    DiTSError,
)

# Import resilience module
from . import resilience
from .resilience import (
    retry,
    circuit_breaker,
    timeout,
    fallback,
    resilient,
    ResilienceError,
    RetryExhaustedError,
    CircuitBreakerOpenError,
    TimeoutExceededError,
    FallbackError,
)

# Import health module
from . import health
from .health import (
    HealthStatus,
    SubsystemHealth,
    HealthCheckResult,
    HealthChecker,
    check_health,
    is_healthy,
)

# Import observability package
from . import observability
from .observability import (
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    timed,
    get_metrics,
    Span,
    Tracer,
    traced,
    get_current_trace_id,
    StructuredLogger,
    LogLevel,
    get_logger,
)


# Shared type definitions

CreativeMode = Literal["grammars", "cinema", "visual", "auralux", "avatars", "dits"]


@dataclass
class PipelineStage:
    """A single stage in a creative pipeline."""
    name: str
    status: Literal["pending", "running", "completed", "failed"]
    progress: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class Asset:
    """A creative asset (image, video, audio, model, etc.)."""
    id: str
    type: Literal["grammar", "video", "image", "audio", "avatar", "narrative"]
    path: str
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GenerationJob:
    """A queued or active generation job."""
    id: str
    mode: CreativeMode
    params: dict
    priority: int = 50
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output: Optional[Asset] = None
    error: Optional[str] = None


@dataclass
class CreativeState:
    """Global state for the Creative section."""
    active_mode: CreativeMode = "grammars"
    pipeline_stages: list[PipelineStage] = field(default_factory=list)
    assets: dict[str, Asset] = field(default_factory=dict)
    job_queue: list[GenerationJob] = field(default_factory=list)
    active_job: Optional[GenerationJob] = None


# Bus event topics
BUS_TOPICS = {
    "generation_started": "creative.generation.started",
    "generation_progress": "creative.generation.progress",
    "generation_completed": "creative.generation.completed",
    "generation_failed": "creative.generation.failed",
    # Grammar events
    "grammar_synthesize": "creative.grammar.synthesize",
    "grammar_evolved": "creative.grammar.evolved",
    # Cinema events
    "cinema_generate": "creative.cinema.generate",
    "cinema_frame_ready": "creative.cinema.frame_ready",
    # Visual events
    "visual_render": "creative.visual.render",
    "visual_style_applied": "creative.visual.style_applied",
    # Auralux events
    "auralux_synthesize": "creative.auralux.synthesize",
    "auralux_speaker_ready": "creative.auralux.speaker_ready",
    # Avatar events
    "avatar_generate": "creative.avatar.generate",
    "avatar_mesh_ready": "creative.avatar.mesh_ready",
    # DiTS events
    "dits_evaluate": "creative.dits.evaluate",
    "dits_transition": "creative.dits.transition",
}


# Lazy imports for subsystems
def get_grammars():
    """Get grammars subsystem."""
    from . import grammars
    return grammars


def get_cinema():
    """Get cinema subsystem."""
    from . import cinema
    return cinema


def get_visual():
    """Get visual subsystem."""
    from . import visual
    return visual


def get_auralux():
    """Get auralux subsystem."""
    from . import auralux
    return auralux


def get_avatars():
    """Get avatars subsystem."""
    from . import avatars
    return avatars


def get_dits():
    """Get dits subsystem."""
    from . import dits
    return dits


# Registry of all subsystems
SUBSYSTEMS = {
    "grammars": {
        "name": "Grammars",
        "description": "CGP/EGGP synthesis, metagrammars, AST visualization",
        "get": get_grammars,
        "features": ["BNF/EBNF Editor", "AST Visualizer", "Synthesis Playground", "Evolution Engine"],
        "layer": "L4",  # DNA layer
    },
    "cinema": {
        "name": "Cinema",
        "description": "Video generation, temporal consistency, multi-shot narrative",
        "get": get_cinema,
        "features": ["Storyboard Editor", "Scene Compositor", "Temporal Consistency", "Multi-Shot Narrative"],
        "layer": "L0",  # Vision layer
    },
    "visual": {
        "name": "Visual",
        "description": "Image generation, style transfer, diffusion models",
        "get": get_visual,
        "features": ["Text-to-Image", "Style Transfer", "Img2Img", "AI Upscaling"],
        "layer": "L0",  # Vision layer
    },
    "auralux": {
        "name": "Auralux",
        "description": "Voice synthesis, TTS/STT, speaker embeddings",
        "get": get_auralux,
        "features": ["Text-to-Speech", "Speech-to-Text", "Speaker Profiles", "Voice Cloning"],
        "layer": "L2",  # mHC layer
    },
    "avatars": {
        "name": "Avatars",
        "description": "3DGS, Neural PBR, SMPL-X, procedural parameters",
        "get": get_avatars,
        "features": ["3D Gaussian Splatting", "SMPL-X Body Model", "Neural PBR Materials", "Deformable Avatars"],
        "layer": "L1",  # Geometric layer
    },
    "dits": {
        "name": "DiTS",
        "description": "Diegetic Transition System, narrative constructivity, μ/ν calculus",
        "get": get_dits,
        "features": ["μ/ν Calculus Engine", "Narrative Construction", "Omega Loop", "Rheomode"],
        "layer": "L5",  # Omega layer
    },
}


# =============================================================================
# BUS EVENT UTILITIES
# =============================================================================

def emit_bus_event(
    topic: str,
    payload: dict,
    agent: str = "creative",
    bus_path: Optional[Path] = None,
) -> dict:
    """
    Emit an event to the Pluribus bus.

    Args:
        topic: Event topic (e.g., "creative.visual.render")
        payload: Event payload data
        agent: Agent identifier
        bus_path: Optional custom bus path

    Returns:
        The emitted event dict

    Example:
        >>> emit_bus_event("creative.visual.render", {"prompt": "sunset"})
    """
    if bus_path is None:
        bus_path = Path("/pluribus/.pluribus/bus/events.ndjson")

    event = {
        "id": str(uuid.uuid4()),
        "topic": topic,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "payload": payload,
    }

    bus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(bus_path, "a") as f:
        f.write(json.dumps(event) + "\n")

    return event


# =============================================================================
# PIPELINE UTILITIES
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for a creative pipeline."""
    name: str
    stages: list[str]
    mode: CreativeMode
    params: dict = field(default_factory=dict)
    on_progress: Optional[Callable[[str, float], None]] = None
    on_error: Optional[Callable[[str, Exception], None]] = None


def create_pipeline(
    mode: CreativeMode,
    stages: Optional[list[str]] = None,
    **params,
) -> GenerationJob:
    """
    Create a new generation pipeline job.

    Args:
        mode: Which creative subsystem to use
        stages: Optional list of stage names
        **params: Parameters for the pipeline

    Returns:
        A new GenerationJob instance

    Example:
        >>> job = create_pipeline("visual", prompt="sunset over mountains")
        >>> job.id
        'gen-abc123...'
    """
    job_id = f"gen-{uuid.uuid4().hex[:12]}"

    # Default stages per mode
    default_stages = {
        "grammars": ["parse", "synthesize", "evolve"],
        "cinema": ["script", "storyboard", "generate", "assemble"],
        "visual": ["generate", "upscale", "style"],
        "auralux": ["synthesize", "enhance", "export"],
        "avatars": ["extract", "generate", "animate", "render"],
        "dits": ["evaluate", "descend", "construct"],
    }

    if stages is None:
        stages = default_stages.get(mode, ["process"])

    job = GenerationJob(
        id=job_id,
        mode=mode,
        params=params,
        status="pending",
    )

    # Create pipeline stages
    job.params["_stages"] = [
        PipelineStage(name=s, status="pending")
        for s in stages
    ]

    # Emit creation event
    emit_bus_event(
        BUS_TOPICS["generation_started"],
        {
            "job_id": job_id,
            "mode": mode,
            "stages": stages,
            "params": {k: v for k, v in params.items() if not k.startswith("_")},
        },
    )

    return job


async def run_pipeline(
    job: GenerationJob,
    handlers: Optional[dict[str, Callable]] = None,
) -> Asset:
    """
    Execute a generation pipeline.

    Args:
        job: The GenerationJob to execute
        handlers: Optional dict mapping stage names to handler functions

    Returns:
        The generated Asset

    Example:
        >>> job = create_pipeline("visual", prompt="sunset")
        >>> asset = await run_pipeline(job)
    """
    import asyncio

    job.status = "running"
    job.started_at = datetime.now(timezone.utc)

    stages = job.params.get("_stages", [])
    handlers = handlers or {}

    try:
        for i, stage in enumerate(stages):
            stage.status = "running"
            stage.started_at = datetime.now(timezone.utc)

            # Emit progress
            progress = (i / len(stages)) * 100
            job.progress = progress
            emit_bus_event(
                BUS_TOPICS["generation_progress"],
                {"job_id": job.id, "stage": stage.name, "progress": progress},
            )

            # Run handler if available
            handler = handlers.get(stage.name)
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    await handler(job, stage)
                else:
                    handler(job, stage)

            stage.status = "completed"
            stage.completed_at = datetime.now(timezone.utc)
            stage.progress = 100.0

        # Create output asset
        asset = Asset(
            id=f"asset-{uuid.uuid4().hex[:12]}",
            type=_mode_to_asset_type(job.mode),
            path=f"/tmp/creative/{job.id}/output",
            metadata={"job_id": job.id, "mode": job.mode},
        )

        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        job.progress = 100.0
        job.output = asset

        emit_bus_event(
            BUS_TOPICS["generation_completed"],
            {"job_id": job.id, "asset_id": asset.id},
        )

        return asset

    except Exception as e:
        job.status = "failed"
        job.error = str(e)

        emit_bus_event(
            BUS_TOPICS["generation_failed"],
            {"job_id": job.id, "error": str(e)},
        )

        raise


def _mode_to_asset_type(mode: CreativeMode) -> str:
    """Map creative mode to asset type."""
    mapping = {
        "grammars": "grammar",
        "cinema": "video",
        "visual": "image",
        "auralux": "audio",
        "avatars": "avatar",
        "dits": "narrative",
    }
    return mapping.get(mode, "unknown")
