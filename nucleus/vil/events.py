"""
VIL (Vision-Integration-Learning) Event System
Unified event schema for vision-metalearning integration.

All events include:
- trace_id: Correlation ID for request tracking
- timestamp: ISO 8601 timestamp
- source: Component that emitted the event
- cmp: Code Maturity Potential metrics

Version: 1.0
Date: 2026-01-25
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Dict, List
from enum import Enum
import uuid
import json


class VILEventType(str, Enum):
    """All VIL event types."""

    # Vision events
    VISION_CAPTURE = "vil.vision.capture"
    VISION_PROCESSED = "vil.vision.processed"
    VISION_ANALYZED = "vil.vision.analyzed"
    VISION_FRAME = "vil.vision.frame"
    VISION_SNAPSHOT = "vil.vision.snapshot"

    # Learning events
    LEARN_META_UPDATE = "vil.learn.meta_update"
    LEARN_ICL_EXAMPLE = "vil.learn.icl_example"
    LEARN_REWARD = "vil.learn.reward"
    LEARN_EPISODE = "vil.learn.episode"
    LEARN_PATTERN = "vil.learn.pattern"

    # Synthesis events
    SYNTH_PROGRAM = "vil.synth.program"
    SYNTH_MUTATION = "vil.synth.mutation"
    SYNTH_DISTILLATION = "vil.synth.distillation"
    SYNTH_CGP_EVOLVE = "vil.synth.cgp_evolve"

    # CMP events
    CMP_FITNESS = "vil.cmp.fitness"
    CMP_LINEAGE = "vil.cmp.lineage"
    CMP_MERGE = "vil.cmp.merge"
    CMP_SPECIATE = "vil.cmp.speciate"

    # Integration events
    INT_PHASE_START = "vil.integration.phase_start"
    INT_PHASE_COMPLETE = "vil.integration.phase_complete"
    INT_STEP_START = "vil.integration.step_start"
    INT_STEP_COMPLETE = "vil.integration.step_complete"
    INT_ERROR = "vil.integration.error"

    # Geometric events
    GEOM_EMBEDDING = "vil.geom.embedding"
    GEOM_ATTRACTION = "vil.geom.attraction"
    GEOM_MANIFOLD = "vil.geom.manifold"


@dataclass
class CmpMetrics:
    """Code Maturity Potential metrics."""

    capture: float = 0.0  # Frame buffer quality
    analysis: float = 0.0  # VLM inference confidence
    lineage: str = ""  # Evolutionary provenance
    generation: int = 0
    fitness: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capture": self.capture,
            "analysis": self.analysis,
            "lineage": self.lineage,
            "generation": self.generation,
            "fitness": self.fitness,
        }


@dataclass
class VILEvent:
    """Base VIL event with trace tracking."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    event_type: VILEventType = VILEventType.VISION_CAPTURE
    source: str = "vil"
    data: Dict[str, Any] = field(default_factory=dict)
    cmp: CmpMetrics = field(default_factory=CmpMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_bus_event(self) -> Dict[str, Any]:
        """Convert to bus event format."""
        return {
            "topic": self.event_type.value,
            "data": {
                "trace_id": self.trace_id,
                "timestamp": self.timestamp,
                "source": self.source,
                **self.data,
                "cmp": self.cmp.to_dict(),
                "metadata": self.metadata,
            },
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_bus_event())


@dataclass
class VisionEvent(VILEvent):
    """Vision-related events."""

    event_type: VILEventType = VILEventType.VISION_CAPTURE

    # Vision-specific fields
    frame_id: str = ""
    image_data: Optional[str] = None  # base64 encoded
    width: int = 0
    height: int = 0
    format: str = "jpeg"
    quality: float = 0.7
    buffer_index: int = 0
    buffer_size: int = 60
    entropy: Optional[float] = None  # Visual entropy measure


@dataclass
class LearningEvent(VILEvent):
    """Learning-related events."""

    event_type: VILEventType = VILEventType.LEARN_META_UPDATE

    # Learning-specific fields
    task_id: str = ""
    loss: float = 0.0
    accuracy: float = 0.0
    reward: float = 0.0
    episode_id: str = ""
    step: int = 0

    # Metalearning fields
    meta_lr: float = 0.0
    inner_loss: float = 0.0
    outer_loss: float = 0.0

    # ICL fields
    icl_examples: int = 0
    icl_buffer_size: int = 5


@dataclass
class SynthesisEvent(VILEvent):
    """Program synthesis events."""

    event_type: VILEventType = VILEventType.SYNTH_PROGRAM

    # Synthesis-specific fields
    program_id: str = ""
    source_image: str = ""
    target_goal: str = ""
    generated_code: str = ""
    confidence: float = 0.0

    # CGP/EGGP fields
    genome_id: str = ""
    mutation_type: str = ""
    fitness_score: float = 0.0
    generation: int = 0


@dataclass
class CMPEvent(VILEvent):
    """Clade Manager Protocol events."""

    event_type: VILEventType = VILEventType.CMP_FITNESS

    # CMP-specific fields
    clade_id: str = ""
    parent_clade: str = ""
    fitness: float = 0.0
    pressure: float = 0.0
    mutation_rate: float = 0.0

    # Golden ratio weighting
    phi_weighted: float = 0.0

    # Lifecycle
    state: str = "active"  # active, converging, merged, dormant, extinct
    lineage_depth: int = 0


@dataclass
class IntegrationEvent(VILEvent):
    """Integration orchestration events."""

    event_type: VILEventType = VILEventType.INT_PHASE_START

    # Integration-specific fields
    zone: int = 0
    phase: str = ""
    step: int = 0
    total_steps: int = 175
    status: str = "in_progress"
    agent: str = ""

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)

    # Progress
    wip_percent: float = 0.0


# Factory functions for creating specific event types

def create_vision_event(
    frame_id: str,
    image_data: Optional[str],
    width: int = 1920,
    height: int = 1080,
    **kwargs
) -> VisionEvent:
    """Create a vision capture event."""
    return VisionEvent(
        event_type=VILEventType.VISION_CAPTURE,
        frame_id=frame_id,
        image_data=image_data,
        width=width,
        height=height,
        **kwargs
    )


def create_learning_event(
    task_id: str,
    loss: float,
    reward: float,
    event_type: VILEventType = VILEventType.LEARN_META_UPDATE,
    **kwargs
) -> LearningEvent:
    """Create a learning event."""
    return LearningEvent(
        event_type=event_type,
        task_id=task_id,
        loss=loss,
        reward=reward,
        **kwargs
    )


def create_synthesis_event(
    program_id: str,
    source_image: str,
    target_goal: str,
    generated_code: str,
    confidence: float,
    **kwargs
) -> SynthesisEvent:
    """Create a synthesis event."""
    return SynthesisEvent(
        event_type=VILEventType.SYNTH_PROGRAM,
        program_id=program_id,
        source_image=source_image,
        target_goal=target_goal,
        generated_code=generated_code,
        confidence=confidence,
        **kwargs
    )


def create_cmp_event(
    clade_id: str,
    fitness: float,
    state: str = "active",
    **kwargs
) -> CMPEvent:
    """Create a CMP event."""
    return CMPEvent(
        event_type=VILEventType.CMP_FITNESS,
        clade_id=clade_id,
        fitness=fitness,
        state=state,
        **kwargs
    )


def create_integration_event(
    zone: int,
    phase: str,
    step: int,
    agent: str,
    status: str = "in_progress",
    **kwargs
) -> IntegrationEvent:
    """Create an integration orchestration event."""
    return IntegrationEvent(
        event_type=VILEventType.INT_PHASE_START,
        zone=zone,
        phase=phase,
        step=step,
        agent=agent,
        status=status,
        **kwargs
    )


def create_vil_event(
    event_type: VILEventType,
    **kwargs
) -> VILEvent:
    """Generic VIL event creator."""
    return VILEvent(event_type=event_type, **kwargs)


# Event validation

def validate_event(event: VILEvent) -> bool:
    """Validate VIL event structure."""
    required_fields = ["trace_id", "timestamp", "event_type", "source"]
    for field in required_fields:
        if not hasattr(event, field) or getattr(event, field) is None:
            return False
    return True


def event_to_bus(event: VILEvent) -> str:
    """Convert event to bus JSON format."""
    if not validate_event(event):
        raise ValueError(f"Invalid VIL event: {event}")
    return event.to_json()


# Trace ID utilities

def create_trace_id(prefix: str = "vil") -> str:
    """Create a new trace ID with prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def parse_trace_id(trace_id: str) -> Dict[str, str]:
    """Parse trace ID into components."""
    parts = trace_id.split("_")
    return {
        "prefix": parts[0] if parts else "unknown",
        "uuid": parts[1] if len(parts) > 1 else "",
    }


__all__ = [
    # Event types
    "VILEventType",

    # Data classes
    "CmpMetrics",
    "VILEvent",
    "VisionEvent",
    "LearningEvent",
    "SynthesisEvent",
    "CMPEvent",
    "IntegrationEvent",

    # Factory functions
    "create_vil_event",
    "create_vision_event",
    "create_learning_event",
    "create_synthesis_event",
    "create_cmp_event",
    "create_integration_event",

    # Utilities
    "validate_event",
    "event_to_bus",
    "create_trace_id",
    "parse_trace_id",
]
