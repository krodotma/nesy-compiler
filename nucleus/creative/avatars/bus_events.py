"""
Bus Events for Avatars Subsystem
================================

Defines bus event topics and payloads for the avatars subsystem.
These events enable communication with other Creative subsystems
and the Pluribus bus infrastructure.

Event Categories:
- Extraction: SMPL-X parameter extraction events
- Gaussian: 3D Gaussian splatting events
- Deformation: Avatar deformation events
- Material: Neural PBR material events
- Procedural: Procedural generation events
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from enum import Enum, auto
from pathlib import Path
from datetime import datetime, timezone
import uuid
import json


class AvatarEventType(Enum):
    """Types of avatar events."""
    # Extraction events
    EXTRACTION_STARTED = auto()
    EXTRACTION_PROGRESS = auto()
    EXTRACTION_COMPLETED = auto()
    EXTRACTION_FAILED = auto()

    # Gaussian splatting events
    GAUSSIAN_CLOUD_CREATED = auto()
    GAUSSIAN_CLOUD_UPDATED = auto()
    GAUSSIAN_CLOUD_OPTIMIZED = auto()
    GAUSSIAN_RENDER_STARTED = auto()
    GAUSSIAN_RENDER_COMPLETED = auto()

    # Deformation events
    DEFORMATION_STARTED = auto()
    DEFORMATION_PROGRESS = auto()
    DEFORMATION_COMPLETED = auto()
    DEFORMATION_FAILED = auto()

    # Material events
    MATERIAL_CREATED = auto()
    MATERIAL_UPDATED = auto()
    MATERIAL_APPLIED = auto()
    TEXTURE_GENERATED = auto()

    # Procedural events
    PROCEDURAL_GENERATION_STARTED = auto()
    PROCEDURAL_GENERATION_PROGRESS = auto()
    PROCEDURAL_GENERATION_COMPLETED = auto()
    PROCEDURAL_BATCH_STARTED = auto()
    PROCEDURAL_BATCH_COMPLETED = auto()

    # General avatar events
    AVATAR_CREATED = auto()
    AVATAR_UPDATED = auto()
    AVATAR_EXPORTED = auto()
    AVATAR_IMPORTED = auto()


# Topic namespace
AVATAR_TOPIC_PREFIX = "creative.avatar"


# Topic definitions
AVATAR_TOPICS = {
    # Extraction
    "extraction_started": f"{AVATAR_TOPIC_PREFIX}.extraction.started",
    "extraction_progress": f"{AVATAR_TOPIC_PREFIX}.extraction.progress",
    "extraction_completed": f"{AVATAR_TOPIC_PREFIX}.extraction.completed",
    "extraction_failed": f"{AVATAR_TOPIC_PREFIX}.extraction.failed",

    # Gaussian
    "gaussian_created": f"{AVATAR_TOPIC_PREFIX}.gaussian.created",
    "gaussian_updated": f"{AVATAR_TOPIC_PREFIX}.gaussian.updated",
    "gaussian_optimized": f"{AVATAR_TOPIC_PREFIX}.gaussian.optimized",
    "render_started": f"{AVATAR_TOPIC_PREFIX}.render.started",
    "render_completed": f"{AVATAR_TOPIC_PREFIX}.render.completed",

    # Deformation
    "deformation_started": f"{AVATAR_TOPIC_PREFIX}.deformation.started",
    "deformation_progress": f"{AVATAR_TOPIC_PREFIX}.deformation.progress",
    "deformation_completed": f"{AVATAR_TOPIC_PREFIX}.deformation.completed",
    "deformation_failed": f"{AVATAR_TOPIC_PREFIX}.deformation.failed",

    # Material
    "material_created": f"{AVATAR_TOPIC_PREFIX}.material.created",
    "material_updated": f"{AVATAR_TOPIC_PREFIX}.material.updated",
    "material_applied": f"{AVATAR_TOPIC_PREFIX}.material.applied",
    "texture_generated": f"{AVATAR_TOPIC_PREFIX}.texture.generated",

    # Procedural
    "procedural_started": f"{AVATAR_TOPIC_PREFIX}.procedural.started",
    "procedural_progress": f"{AVATAR_TOPIC_PREFIX}.procedural.progress",
    "procedural_completed": f"{AVATAR_TOPIC_PREFIX}.procedural.completed",
    "batch_started": f"{AVATAR_TOPIC_PREFIX}.batch.started",
    "batch_completed": f"{AVATAR_TOPIC_PREFIX}.batch.completed",

    # General
    "avatar_created": f"{AVATAR_TOPIC_PREFIX}.created",
    "avatar_updated": f"{AVATAR_TOPIC_PREFIX}.updated",
    "avatar_exported": f"{AVATAR_TOPIC_PREFIX}.exported",
    "avatar_imported": f"{AVATAR_TOPIC_PREFIX}.imported",
}


@dataclass
class AvatarBusEvent:
    """
    Base class for avatar bus events.

    Attributes:
        id: Unique event identifier.
        type: Event type.
        topic: Bus topic.
        timestamp: Event timestamp.
        agent: Agent that emitted the event.
        payload: Event-specific payload.
        trace_id: Optional trace ID for correlation.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: AvatarEventType = AvatarEventType.AVATAR_CREATED
    topic: str = AVATAR_TOPICS["avatar_created"]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str = "avatars"
    payload: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for bus emission."""
        return {
            "id": self.id,
            "type": self.type.name,
            "topic": self.topic,
            "timestamp": self.timestamp.isoformat(),
            "agent": self.agent,
            "payload": self.payload,
            "trace_id": self.trace_id,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AvatarBusEvent":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=AvatarEventType[data.get("type", "AVATAR_CREATED")],
            topic=data.get("topic", AVATAR_TOPICS["avatar_created"]),
            timestamp=timestamp or datetime.now(timezone.utc),
            agent=data.get("agent", "avatars"),
            payload=data.get("payload", {}),
            trace_id=data.get("trace_id"),
        )


# Specific event payloads

@dataclass
class ExtractionStartedPayload:
    """Payload for extraction started event."""
    source_type: str  # "image" or "video"
    source_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionProgressPayload:
    """Payload for extraction progress event."""
    progress: float  # 0-100
    stage: str  # Current stage name
    frame_index: Optional[int] = None
    total_frames: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionCompletedPayload:
    """Payload for extraction completed event."""
    params_id: str
    num_frames: int = 1
    confidence: float = 0.0
    processing_time_ms: float = 0.0
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionFailedPayload:
    """Payload for extraction failed event."""
    error: str
    error_code: Optional[str] = None
    stage: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GaussianCloudPayload:
    """Payload for Gaussian cloud events."""
    cloud_id: str
    num_gaussians: int
    sh_degree: int = 0
    bounds: Optional[Dict[str, List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeformationPayload:
    """Payload for deformation events."""
    avatar_id: str
    source_pose_id: str
    target_pose_id: str
    deformation_type: str = "LINEAR_BLEND_SKINNING"
    quality_score: float = 1.0
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MaterialPayload:
    """Payload for material events."""
    material_id: str
    material_name: str
    material_type: str = "STANDARD"
    textures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProceduralPayload:
    """Payload for procedural generation events."""
    avatar_id: str
    style_seed: int
    gender: str = "NEUTRAL"
    body_type: str = "AVERAGE"
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RenderPayload:
    """Payload for render events."""
    render_id: str
    width: int
    height: int
    num_gaussians: int
    render_time_ms: float = 0.0
    output_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Event factory functions

def create_extraction_started_event(
    source_type: str,
    source_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> AvatarBusEvent:
    """Create an extraction started event."""
    return AvatarBusEvent(
        type=AvatarEventType.EXTRACTION_STARTED,
        topic=AVATAR_TOPICS["extraction_started"],
        payload=ExtractionStartedPayload(
            source_type=source_type,
            source_path=source_path,
            config=config or {},
        ).to_dict(),
        trace_id=trace_id,
    )


def create_extraction_progress_event(
    progress: float,
    stage: str,
    frame_index: Optional[int] = None,
    total_frames: Optional[int] = None,
    trace_id: Optional[str] = None,
) -> AvatarBusEvent:
    """Create an extraction progress event."""
    return AvatarBusEvent(
        type=AvatarEventType.EXTRACTION_PROGRESS,
        topic=AVATAR_TOPICS["extraction_progress"],
        payload=ExtractionProgressPayload(
            progress=progress,
            stage=stage,
            frame_index=frame_index,
            total_frames=total_frames,
        ).to_dict(),
        trace_id=trace_id,
    )


def create_extraction_completed_event(
    params_id: str,
    num_frames: int = 1,
    confidence: float = 0.0,
    processing_time_ms: float = 0.0,
    output_path: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> AvatarBusEvent:
    """Create an extraction completed event."""
    return AvatarBusEvent(
        type=AvatarEventType.EXTRACTION_COMPLETED,
        topic=AVATAR_TOPICS["extraction_completed"],
        payload=ExtractionCompletedPayload(
            params_id=params_id,
            num_frames=num_frames,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
            output_path=output_path,
        ).to_dict(),
        trace_id=trace_id,
    )


def create_extraction_failed_event(
    error: str,
    error_code: Optional[str] = None,
    stage: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> AvatarBusEvent:
    """Create an extraction failed event."""
    return AvatarBusEvent(
        type=AvatarEventType.EXTRACTION_FAILED,
        topic=AVATAR_TOPICS["extraction_failed"],
        payload=ExtractionFailedPayload(
            error=error,
            error_code=error_code,
            stage=stage,
        ).to_dict(),
        trace_id=trace_id,
    )


def create_gaussian_cloud_event(
    event_type: AvatarEventType,
    cloud_id: str,
    num_gaussians: int,
    sh_degree: int = 0,
    bounds: Optional[Dict[str, List[float]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> AvatarBusEvent:
    """Create a Gaussian cloud event."""
    topic_map = {
        AvatarEventType.GAUSSIAN_CLOUD_CREATED: AVATAR_TOPICS["gaussian_created"],
        AvatarEventType.GAUSSIAN_CLOUD_UPDATED: AVATAR_TOPICS["gaussian_updated"],
        AvatarEventType.GAUSSIAN_CLOUD_OPTIMIZED: AVATAR_TOPICS["gaussian_optimized"],
    }

    return AvatarBusEvent(
        type=event_type,
        topic=topic_map.get(event_type, AVATAR_TOPICS["gaussian_created"]),
        payload=GaussianCloudPayload(
            cloud_id=cloud_id,
            num_gaussians=num_gaussians,
            sh_degree=sh_degree,
            bounds=bounds,
            metadata=metadata or {},
        ).to_dict(),
        trace_id=trace_id,
    )


def create_deformation_started_event(
    avatar_id: str,
    source_pose_id: str,
    target_pose_id: str,
    deformation_type: str = "LINEAR_BLEND_SKINNING",
    trace_id: Optional[str] = None,
) -> AvatarBusEvent:
    """Create a deformation started event."""
    return AvatarBusEvent(
        type=AvatarEventType.DEFORMATION_STARTED,
        topic=AVATAR_TOPICS["deformation_started"],
        payload=DeformationPayload(
            avatar_id=avatar_id,
            source_pose_id=source_pose_id,
            target_pose_id=target_pose_id,
            deformation_type=deformation_type,
        ).to_dict(),
        trace_id=trace_id,
    )


def create_deformation_completed_event(
    avatar_id: str,
    source_pose_id: str,
    target_pose_id: str,
    deformation_type: str = "LINEAR_BLEND_SKINNING",
    quality_score: float = 1.0,
    processing_time_ms: float = 0.0,
    trace_id: Optional[str] = None,
) -> AvatarBusEvent:
    """Create a deformation completed event."""
    return AvatarBusEvent(
        type=AvatarEventType.DEFORMATION_COMPLETED,
        topic=AVATAR_TOPICS["deformation_completed"],
        payload=DeformationPayload(
            avatar_id=avatar_id,
            source_pose_id=source_pose_id,
            target_pose_id=target_pose_id,
            deformation_type=deformation_type,
            quality_score=quality_score,
            processing_time_ms=processing_time_ms,
        ).to_dict(),
        trace_id=trace_id,
    )


def create_material_event(
    event_type: AvatarEventType,
    material_id: str,
    material_name: str,
    material_type: str = "STANDARD",
    textures: Optional[List[str]] = None,
    trace_id: Optional[str] = None,
) -> AvatarBusEvent:
    """Create a material event."""
    topic_map = {
        AvatarEventType.MATERIAL_CREATED: AVATAR_TOPICS["material_created"],
        AvatarEventType.MATERIAL_UPDATED: AVATAR_TOPICS["material_updated"],
        AvatarEventType.MATERIAL_APPLIED: AVATAR_TOPICS["material_applied"],
    }

    return AvatarBusEvent(
        type=event_type,
        topic=topic_map.get(event_type, AVATAR_TOPICS["material_created"]),
        payload=MaterialPayload(
            material_id=material_id,
            material_name=material_name,
            material_type=material_type,
            textures=textures or [],
        ).to_dict(),
        trace_id=trace_id,
    )


def create_procedural_started_event(
    avatar_id: str,
    style_seed: int,
    gender: str = "NEUTRAL",
    body_type: str = "AVERAGE",
    trace_id: Optional[str] = None,
) -> AvatarBusEvent:
    """Create a procedural generation started event."""
    return AvatarBusEvent(
        type=AvatarEventType.PROCEDURAL_GENERATION_STARTED,
        topic=AVATAR_TOPICS["procedural_started"],
        payload=ProceduralPayload(
            avatar_id=avatar_id,
            style_seed=style_seed,
            gender=gender,
            body_type=body_type,
        ).to_dict(),
        trace_id=trace_id,
    )


def create_procedural_completed_event(
    avatar_id: str,
    style_seed: int,
    gender: str = "NEUTRAL",
    body_type: str = "AVERAGE",
    processing_time_ms: float = 0.0,
    trace_id: Optional[str] = None,
) -> AvatarBusEvent:
    """Create a procedural generation completed event."""
    return AvatarBusEvent(
        type=AvatarEventType.PROCEDURAL_GENERATION_COMPLETED,
        topic=AVATAR_TOPICS["procedural_completed"],
        payload=ProceduralPayload(
            avatar_id=avatar_id,
            style_seed=style_seed,
            gender=gender,
            body_type=body_type,
            processing_time_ms=processing_time_ms,
        ).to_dict(),
        trace_id=trace_id,
    )


def create_render_completed_event(
    render_id: str,
    width: int,
    height: int,
    num_gaussians: int,
    render_time_ms: float = 0.0,
    output_path: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> AvatarBusEvent:
    """Create a render completed event."""
    return AvatarBusEvent(
        type=AvatarEventType.GAUSSIAN_RENDER_COMPLETED,
        topic=AVATAR_TOPICS["render_completed"],
        payload=RenderPayload(
            render_id=render_id,
            width=width,
            height=height,
            num_gaussians=num_gaussians,
            render_time_ms=render_time_ms,
            output_path=output_path,
        ).to_dict(),
        trace_id=trace_id,
    )


# Bus emission utility

class AvatarBusEmitter:
    """
    Utility for emitting avatar events to the Pluribus bus.

    Example:
        >>> emitter = AvatarBusEmitter()
        >>> emitter.emit(create_extraction_started_event("image", "/path/to/image.jpg"))
    """

    def __init__(
        self,
        bus_path: Optional[Path] = None,
        agent: str = "avatars",
    ):
        """
        Initialize emitter.

        Args:
            bus_path: Path to bus events file.
            agent: Agent identifier.
        """
        self.bus_path = bus_path or Path("/pluribus/.pluribus/bus/events.ndjson")
        self.agent = agent

    def emit(self, event: AvatarBusEvent) -> None:
        """
        Emit an event to the bus.

        Args:
            event: Event to emit.
        """
        # Override agent
        event.agent = self.agent

        # Ensure bus directory exists
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Append event
        with open(self.bus_path, "a") as f:
            f.write(event.to_json() + "\n")

    def emit_extraction_started(
        self,
        source_type: str,
        source_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ) -> AvatarBusEvent:
        """Emit extraction started event."""
        event = create_extraction_started_event(
            source_type, source_path, config, trace_id
        )
        self.emit(event)
        return event

    def emit_extraction_progress(
        self,
        progress: float,
        stage: str,
        frame_index: Optional[int] = None,
        total_frames: Optional[int] = None,
        trace_id: Optional[str] = None,
    ) -> AvatarBusEvent:
        """Emit extraction progress event."""
        event = create_extraction_progress_event(
            progress, stage, frame_index, total_frames, trace_id
        )
        self.emit(event)
        return event

    def emit_extraction_completed(
        self,
        params_id: str,
        num_frames: int = 1,
        confidence: float = 0.0,
        processing_time_ms: float = 0.0,
        output_path: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> AvatarBusEvent:
        """Emit extraction completed event."""
        event = create_extraction_completed_event(
            params_id, num_frames, confidence, processing_time_ms, output_path, trace_id
        )
        self.emit(event)
        return event

    def emit_extraction_failed(
        self,
        error: str,
        error_code: Optional[str] = None,
        stage: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> AvatarBusEvent:
        """Emit extraction failed event."""
        event = create_extraction_failed_event(error, error_code, stage, trace_id)
        self.emit(event)
        return event

    def emit_deformation_completed(
        self,
        avatar_id: str,
        source_pose_id: str,
        target_pose_id: str,
        quality_score: float = 1.0,
        processing_time_ms: float = 0.0,
        trace_id: Optional[str] = None,
    ) -> AvatarBusEvent:
        """Emit deformation completed event."""
        event = create_deformation_completed_event(
            avatar_id, source_pose_id, target_pose_id,
            quality_score=quality_score,
            processing_time_ms=processing_time_ms,
            trace_id=trace_id
        )
        self.emit(event)
        return event

    def emit_procedural_completed(
        self,
        avatar_id: str,
        style_seed: int,
        gender: str = "NEUTRAL",
        body_type: str = "AVERAGE",
        processing_time_ms: float = 0.0,
        trace_id: Optional[str] = None,
    ) -> AvatarBusEvent:
        """Emit procedural generation completed event."""
        event = create_procedural_completed_event(
            avatar_id, style_seed, gender, body_type, processing_time_ms, trace_id
        )
        self.emit(event)
        return event


# Default emitter instance
_default_emitter: Optional[AvatarBusEmitter] = None


def get_emitter() -> AvatarBusEmitter:
    """Get or create default emitter instance."""
    global _default_emitter
    if _default_emitter is None:
        _default_emitter = AvatarBusEmitter()
    return _default_emitter


def emit_event(event: AvatarBusEvent) -> None:
    """Emit event using default emitter."""
    get_emitter().emit(event)
