"""
Visual Subsystem Bus Events
===========================

Defines bus event topics, payloads, and utilities for the visual subsystem.
Integrates with the Pluribus event bus for async communication.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union
import json
import uuid


class VisualEventTopic(str, Enum):
    """Bus event topics for the visual subsystem."""

    # Generation events
    GENERATION_STARTED = "creative.visual.generation.started"
    GENERATION_PROGRESS = "creative.visual.generation.progress"
    GENERATION_COMPLETED = "creative.visual.generation.completed"
    GENERATION_FAILED = "creative.visual.generation.failed"

    # Rendering events
    RENDER_REQUESTED = "creative.visual.render.requested"
    RENDER_STARTED = "creative.visual.render.started"
    RENDER_COMPLETED = "creative.visual.render.completed"
    RENDER_FAILED = "creative.visual.render.failed"

    # Style transfer events
    STYLE_TRANSFER_STARTED = "creative.visual.style.started"
    STYLE_TRANSFER_PROGRESS = "creative.visual.style.progress"
    STYLE_TRANSFER_COMPLETED = "creative.visual.style.completed"
    STYLE_TRANSFER_FAILED = "creative.visual.style.failed"
    STYLE_APPLIED = "creative.visual.style_applied"

    # Upscale events
    UPSCALE_STARTED = "creative.visual.upscale.started"
    UPSCALE_PROGRESS = "creative.visual.upscale.progress"
    UPSCALE_COMPLETED = "creative.visual.upscale.completed"
    UPSCALE_FAILED = "creative.visual.upscale.failed"

    # Asset events
    ASSET_CREATED = "creative.visual.asset.created"
    ASSET_UPDATED = "creative.visual.asset.updated"
    ASSET_DELETED = "creative.visual.asset.deleted"

    # Provider events
    PROVIDER_CONNECTED = "creative.visual.provider.connected"
    PROVIDER_DISCONNECTED = "creative.visual.provider.disconnected"
    PROVIDER_ERROR = "creative.visual.provider.error"

    # Pipeline events
    PIPELINE_STARTED = "creative.visual.pipeline.started"
    PIPELINE_STAGE_COMPLETED = "creative.visual.pipeline.stage_completed"
    PIPELINE_COMPLETED = "creative.visual.pipeline.completed"
    PIPELINE_FAILED = "creative.visual.pipeline.failed"


class EventPriority(int, Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200


@dataclass
class VisualEventPayload:
    """Base payload for visual events.

    Attributes:
        event_id: Unique event identifier
        timestamp: Event timestamp
        source: Source component/agent
        operation_id: ID of the operation this event relates to
        data: Event-specific data
        metadata: Additional metadata
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "visual_subsystem"
    operation_id: Optional[str] = None
    data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class GenerationEventPayload(VisualEventPayload):
    """Payload for image generation events."""
    prompt: str = ""
    model: str = ""
    width: int = 0
    height: int = 0
    progress: float = 0.0
    num_images: int = 1
    seed: Optional[int] = None


@dataclass
class StyleTransferEventPayload(VisualEventPayload):
    """Payload for style transfer events."""
    style_name: str = ""
    method: str = ""
    content_weight: float = 1.0
    style_weight: float = 100000.0
    progress: float = 0.0
    iterations: int = 0


@dataclass
class UpscaleEventPayload(VisualEventPayload):
    """Payload for upscale events."""
    input_size: tuple[int, int] = (0, 0)
    output_size: tuple[int, int] = (0, 0)
    scale_factor: int = 1
    method: str = ""
    progress: float = 0.0


@dataclass
class AssetEventPayload(VisualEventPayload):
    """Payload for asset events."""
    asset_id: str = ""
    asset_type: str = "image"
    path: str = ""
    size_bytes: int = 0
    dimensions: tuple[int, int] = (0, 0)


@dataclass
class PipelineEventPayload(VisualEventPayload):
    """Payload for pipeline events."""
    pipeline_id: str = ""
    stage: str = ""
    stages_total: int = 0
    stages_completed: int = 0
    progress: float = 0.0


@dataclass
class VisualBusEvent:
    """Complete bus event structure for visual subsystem.

    Attributes:
        id: Unique event ID
        topic: Event topic
        payload: Event payload
        priority: Event priority
        timestamp: Event timestamp
        agent: Originating agent
        trace_id: Trace ID for distributed tracing
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: VisualEventTopic = VisualEventTopic.GENERATION_STARTED
    payload: VisualEventPayload = field(default_factory=VisualEventPayload)
    priority: EventPriority = EventPriority.NORMAL
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent: str = "creative.visual"
    trace_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for bus transmission."""
        return {
            "id": self.id,
            "topic": self.topic.value if isinstance(self.topic, VisualEventTopic) else self.topic,
            "payload": self.payload.to_dict() if hasattr(self.payload, "to_dict") else self.payload,
            "priority": self.priority.value if isinstance(self.priority, EventPriority) else self.priority,
            "timestamp": self.timestamp,
            "agent": self.agent,
            "trace_id": self.trace_id,
        }

    def to_ndjson(self) -> str:
        """Convert to NDJSON format for bus file."""
        return json.dumps(self.to_dict())


class VisualBusEmitter:
    """Emitter for visual subsystem bus events.

    Provides high-level API for emitting events to the Pluribus bus.

    Example:
        >>> emitter = VisualBusEmitter()
        >>> emitter.emit_generation_started("sunset", model="sd-xl", width=1024, height=1024)
        >>> emitter.emit_generation_progress(op_id, 50.0)
        >>> emitter.emit_generation_completed(op_id, asset_id="img-123")
    """

    def __init__(
        self,
        bus_path: Optional[Path] = None,
        agent: str = "creative.visual",
        trace_id: Optional[str] = None,
    ):
        """Initialize the emitter.

        Args:
            bus_path: Path to bus events file
            agent: Agent identifier
            trace_id: Optional trace ID for distributed tracing
        """
        self.bus_path = bus_path or Path("/pluribus/.pluribus/bus/events.ndjson")
        self.agent = agent
        self.trace_id = trace_id
        self._operation_id_counter = 0

    def _generate_operation_id(self) -> str:
        """Generate a unique operation ID."""
        self._operation_id_counter += 1
        return f"vis-op-{uuid.uuid4().hex[:8]}-{self._operation_id_counter}"

    def _emit(self, event: VisualBusEvent) -> VisualBusEvent:
        """Emit an event to the bus.

        Args:
            event: The event to emit

        Returns:
            The emitted event
        """
        # Ensure parent directory exists
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Append to bus file
        with open(self.bus_path, "a") as f:
            f.write(event.to_ndjson() + "\n")

        return event

    def emit(
        self,
        topic: VisualEventTopic,
        payload: Optional[VisualEventPayload] = None,
        priority: EventPriority = EventPriority.NORMAL,
        **kwargs,
    ) -> VisualBusEvent:
        """Emit a custom event.

        Args:
            topic: Event topic
            payload: Event payload
            priority: Event priority
            **kwargs: Additional payload data

        Returns:
            The emitted event
        """
        if payload is None:
            payload = VisualEventPayload(data=kwargs)
        elif kwargs:
            payload.data.update(kwargs)

        event = VisualBusEvent(
            topic=topic,
            payload=payload,
            priority=priority,
            agent=self.agent,
            trace_id=self.trace_id,
        )

        return self._emit(event)

    # Generation event helpers

    def emit_generation_started(
        self,
        prompt: str,
        model: str = "",
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        seed: Optional[int] = None,
        operation_id: Optional[str] = None,
    ) -> tuple[VisualBusEvent, str]:
        """Emit generation started event.

        Args:
            prompt: Generation prompt
            model: Model identifier
            width: Image width
            height: Image height
            num_images: Number of images
            seed: Random seed
            operation_id: Optional operation ID (generated if not provided)

        Returns:
            Tuple of (event, operation_id)
        """
        op_id = operation_id or self._generate_operation_id()

        payload = GenerationEventPayload(
            operation_id=op_id,
            prompt=prompt,
            model=model,
            width=width,
            height=height,
            num_images=num_images,
            seed=seed,
            progress=0.0,
        )

        event = self.emit(VisualEventTopic.GENERATION_STARTED, payload)
        return event, op_id

    def emit_generation_progress(
        self,
        operation_id: str,
        progress: float,
        message: str = "",
    ) -> VisualBusEvent:
        """Emit generation progress event.

        Args:
            operation_id: Operation ID
            progress: Progress percentage (0-100)
            message: Progress message

        Returns:
            The emitted event
        """
        payload = GenerationEventPayload(
            operation_id=operation_id,
            progress=progress,
            data={"message": message} if message else {},
        )

        return self.emit(VisualEventTopic.GENERATION_PROGRESS, payload)

    def emit_generation_completed(
        self,
        operation_id: str,
        asset_id: str = "",
        generation_time: float = 0.0,
        **metadata,
    ) -> VisualBusEvent:
        """Emit generation completed event.

        Args:
            operation_id: Operation ID
            asset_id: Generated asset ID
            generation_time: Time taken in seconds
            **metadata: Additional metadata

        Returns:
            The emitted event
        """
        payload = GenerationEventPayload(
            operation_id=operation_id,
            progress=100.0,
            data={
                "asset_id": asset_id,
                "generation_time": generation_time,
            },
            metadata=metadata,
        )

        return self.emit(VisualEventTopic.GENERATION_COMPLETED, payload)

    def emit_generation_failed(
        self,
        operation_id: str,
        error: str,
        error_code: Optional[str] = None,
    ) -> VisualBusEvent:
        """Emit generation failed event.

        Args:
            operation_id: Operation ID
            error: Error message
            error_code: Optional error code

        Returns:
            The emitted event
        """
        payload = GenerationEventPayload(
            operation_id=operation_id,
            data={
                "error": error,
                "error_code": error_code,
            },
        )

        return self.emit(
            VisualEventTopic.GENERATION_FAILED,
            payload,
            priority=EventPriority.HIGH,
        )

    # Style transfer event helpers

    def emit_style_transfer_started(
        self,
        style_name: str,
        method: str = "neural",
        operation_id: Optional[str] = None,
    ) -> tuple[VisualBusEvent, str]:
        """Emit style transfer started event."""
        op_id = operation_id or self._generate_operation_id()

        payload = StyleTransferEventPayload(
            operation_id=op_id,
            style_name=style_name,
            method=method,
            progress=0.0,
        )

        event = self.emit(VisualEventTopic.STYLE_TRANSFER_STARTED, payload)
        return event, op_id

    def emit_style_transfer_completed(
        self,
        operation_id: str,
        asset_id: str = "",
        transfer_time: float = 0.0,
    ) -> VisualBusEvent:
        """Emit style transfer completed event."""
        payload = StyleTransferEventPayload(
            operation_id=operation_id,
            progress=100.0,
            data={
                "asset_id": asset_id,
                "transfer_time": transfer_time,
            },
        )

        return self.emit(VisualEventTopic.STYLE_TRANSFER_COMPLETED, payload)

    # Upscale event helpers

    def emit_upscale_started(
        self,
        input_size: tuple[int, int],
        scale_factor: int,
        method: str = "lanczos",
        operation_id: Optional[str] = None,
    ) -> tuple[VisualBusEvent, str]:
        """Emit upscale started event."""
        op_id = operation_id or self._generate_operation_id()

        payload = UpscaleEventPayload(
            operation_id=op_id,
            input_size=input_size,
            scale_factor=scale_factor,
            method=method,
            progress=0.0,
        )

        event = self.emit(VisualEventTopic.UPSCALE_STARTED, payload)
        return event, op_id

    def emit_upscale_completed(
        self,
        operation_id: str,
        output_size: tuple[int, int],
        upscale_time: float = 0.0,
    ) -> VisualBusEvent:
        """Emit upscale completed event."""
        payload = UpscaleEventPayload(
            operation_id=operation_id,
            output_size=output_size,
            progress=100.0,
            data={"upscale_time": upscale_time},
        )

        return self.emit(VisualEventTopic.UPSCALE_COMPLETED, payload)

    # Asset event helpers

    def emit_asset_created(
        self,
        asset_id: str,
        asset_type: str,
        path: str,
        dimensions: tuple[int, int] = (0, 0),
        size_bytes: int = 0,
    ) -> VisualBusEvent:
        """Emit asset created event."""
        payload = AssetEventPayload(
            asset_id=asset_id,
            asset_type=asset_type,
            path=path,
            dimensions=dimensions,
            size_bytes=size_bytes,
        )

        return self.emit(VisualEventTopic.ASSET_CREATED, payload)

    # Pipeline event helpers

    def emit_pipeline_started(
        self,
        pipeline_id: str,
        stages: list[str],
    ) -> VisualBusEvent:
        """Emit pipeline started event."""
        payload = PipelineEventPayload(
            operation_id=pipeline_id,
            pipeline_id=pipeline_id,
            stages_total=len(stages),
            stages_completed=0,
            progress=0.0,
            data={"stages": stages},
        )

        return self.emit(VisualEventTopic.PIPELINE_STARTED, payload)

    def emit_pipeline_stage_completed(
        self,
        pipeline_id: str,
        stage: str,
        stage_index: int,
        total_stages: int,
    ) -> VisualBusEvent:
        """Emit pipeline stage completed event."""
        payload = PipelineEventPayload(
            operation_id=pipeline_id,
            pipeline_id=pipeline_id,
            stage=stage,
            stages_total=total_stages,
            stages_completed=stage_index + 1,
            progress=((stage_index + 1) / total_stages) * 100,
        )

        return self.emit(VisualEventTopic.PIPELINE_STAGE_COMPLETED, payload)


# Convenience function for quick event emission
def emit_visual_event(
    topic: Union[str, VisualEventTopic],
    payload: dict,
    bus_path: Optional[Path] = None,
) -> VisualBusEvent:
    """Quick event emission function.

    Args:
        topic: Event topic (string or enum)
        payload: Event payload as dict
        bus_path: Optional custom bus path

    Returns:
        The emitted event
    """
    if isinstance(topic, str):
        # Try to find matching topic
        for t in VisualEventTopic:
            if t.value == topic or t.name == topic:
                topic = t
                break
        else:
            # Use as custom topic string
            topic = VisualEventTopic.RENDER_REQUESTED  # Default

    emitter = VisualBusEmitter(bus_path=bus_path)
    event_payload = VisualEventPayload(data=payload)
    return emitter.emit(topic, event_payload)


# Topic constants for backwards compatibility
VISUAL_BUS_TOPICS = {t.name.lower(): t.value for t in VisualEventTopic}
