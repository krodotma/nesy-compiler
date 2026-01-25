"""
Bus Events Module
=================

Bus event definitions for the cinema subsystem.
Provides event types and helpers for inter-agent communication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import uuid


class CinemaEventType(Enum):
    """Types of cinema subsystem events."""
    # Script parsing events
    SCRIPT_PARSE_STARTED = "cinema.script.parse.started"
    SCRIPT_PARSE_COMPLETE = "cinema.script.parse.complete"
    SCRIPT_PARSE_FAILED = "cinema.script.parse.failed"

    # Storyboard events
    STORYBOARD_GENERATION_STARTED = "cinema.storyboard.generation.started"
    STORYBOARD_GENERATION_PROGRESS = "cinema.storyboard.generation.progress"
    STORYBOARD_GENERATION_COMPLETE = "cinema.storyboard.generation.complete"
    STORYBOARD_GENERATION_FAILED = "cinema.storyboard.generation.failed"

    # Frame generation events
    FRAME_GENERATION_STARTED = "cinema.frame.generation.started"
    FRAME_GENERATION_PROGRESS = "cinema.frame.generation.progress"
    FRAME_GENERATION_COMPLETE = "cinema.frame.generation.complete"
    FRAME_GENERATION_FAILED = "cinema.frame.generation.failed"
    FRAME_BATCH_COMPLETE = "cinema.frame.batch.complete"

    # Temporal consistency events
    CONSISTENCY_ANALYSIS_STARTED = "cinema.consistency.analysis.started"
    CONSISTENCY_ANALYSIS_COMPLETE = "cinema.consistency.analysis.complete"
    CONSISTENCY_ENFORCEMENT_STARTED = "cinema.consistency.enforcement.started"
    CONSISTENCY_ENFORCEMENT_COMPLETE = "cinema.consistency.enforcement.complete"

    # Video assembly events
    ASSEMBLY_STARTED = "cinema.assembly.started"
    ASSEMBLY_PROGRESS = "cinema.assembly.progress"
    ASSEMBLY_COMPLETE = "cinema.assembly.complete"
    ASSEMBLY_FAILED = "cinema.assembly.failed"

    # Pipeline events
    PIPELINE_STARTED = "cinema.pipeline.started"
    PIPELINE_STAGE_COMPLETE = "cinema.pipeline.stage.complete"
    PIPELINE_COMPLETE = "cinema.pipeline.complete"
    PIPELINE_FAILED = "cinema.pipeline.failed"


class EventLevel(Enum):
    """Event severity/importance levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CinemaEvent:
    """
    A cinema subsystem bus event.

    Attributes:
        event_type: Type of the event
        level: Severity level
        actor: Agent/component that emitted the event
        data: Event-specific data payload
        timestamp: When the event was created
        event_id: Unique event identifier
        correlation_id: ID for correlating related events
        metadata: Additional event metadata
    """
    event_type: CinemaEventType
    level: EventLevel = EventLevel.INFO
    actor: str = "cinema"
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_type': self.event_type.value,
            'level': self.level.value,
            'actor': self.actor,
            'data': self.data,
            'timestamp': self.timestamp,
            'event_id': self.event_id,
            'correlation_id': self.correlation_id,
            'metadata': self.metadata,
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CinemaEvent':
        """Create event from dictionary."""
        return cls(
            event_type=CinemaEventType(data['event_type']),
            level=EventLevel(data.get('level', 'info')),
            actor=data.get('actor', 'cinema'),
            data=data.get('data', {}),
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            event_id=data.get('event_id', str(uuid.uuid4())[:12]),
            correlation_id=data.get('correlation_id'),
            metadata=data.get('metadata', {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'CinemaEvent':
        """Create event from JSON string."""
        return cls.from_dict(json.loads(json_str))


class CinemaEventEmitter:
    """
    Helper class for emitting cinema events.

    Provides convenience methods for common event patterns.
    """

    def __init__(self, actor: str = "cinema", correlation_id: Optional[str] = None):
        """
        Initialize the event emitter.

        Args:
            actor: Default actor name for events
            correlation_id: Default correlation ID for related events
        """
        self.actor = actor
        self.correlation_id = correlation_id or str(uuid.uuid4())[:12]
        self._event_log: List[CinemaEvent] = []

    def emit(self,
             event_type: CinemaEventType,
             data: Optional[Dict[str, Any]] = None,
             level: EventLevel = EventLevel.INFO,
             **kwargs) -> CinemaEvent:
        """
        Emit a cinema event.

        Args:
            event_type: Type of event to emit
            data: Event data payload
            level: Event level
            **kwargs: Additional metadata

        Returns:
            The created CinemaEvent.
        """
        event = CinemaEvent(
            event_type=event_type,
            level=level,
            actor=self.actor,
            data=data or {},
            correlation_id=self.correlation_id,
            metadata=kwargs,
        )
        self._event_log.append(event)
        return event

    def script_parse_started(self, script_path: str, **kwargs) -> CinemaEvent:
        """Emit script parse started event."""
        return self.emit(
            CinemaEventType.SCRIPT_PARSE_STARTED,
            {'script_path': script_path},
            **kwargs
        )

    def script_parse_complete(self, script_path: str, scene_count: int,
                               character_count: int, **kwargs) -> CinemaEvent:
        """Emit script parse complete event."""
        return self.emit(
            CinemaEventType.SCRIPT_PARSE_COMPLETE,
            {
                'script_path': script_path,
                'scene_count': scene_count,
                'character_count': character_count,
            },
            **kwargs
        )

    def script_parse_failed(self, script_path: str, error: str, **kwargs) -> CinemaEvent:
        """Emit script parse failed event."""
        return self.emit(
            CinemaEventType.SCRIPT_PARSE_FAILED,
            {'script_path': script_path, 'error': error},
            level=EventLevel.ERROR,
            **kwargs
        )

    def storyboard_started(self, script_title: str, **kwargs) -> CinemaEvent:
        """Emit storyboard generation started event."""
        return self.emit(
            CinemaEventType.STORYBOARD_GENERATION_STARTED,
            {'script_title': script_title},
            **kwargs
        )

    def storyboard_progress(self, current: int, total: int, **kwargs) -> CinemaEvent:
        """Emit storyboard generation progress event."""
        return self.emit(
            CinemaEventType.STORYBOARD_GENERATION_PROGRESS,
            {'current': current, 'total': total, 'percent': int(100 * current / total)},
            level=EventLevel.DEBUG,
            **kwargs
        )

    def storyboard_complete(self, panel_count: int, duration: float, **kwargs) -> CinemaEvent:
        """Emit storyboard generation complete event."""
        return self.emit(
            CinemaEventType.STORYBOARD_GENERATION_COMPLETE,
            {'panel_count': panel_count, 'duration_seconds': duration},
            **kwargs
        )

    def frame_generation_started(self, panel_count: int, **kwargs) -> CinemaEvent:
        """Emit frame generation started event."""
        return self.emit(
            CinemaEventType.FRAME_GENERATION_STARTED,
            {'panel_count': panel_count},
            **kwargs
        )

    def frame_generation_progress(self, current: int, total: int, **kwargs) -> CinemaEvent:
        """Emit frame generation progress event."""
        return self.emit(
            CinemaEventType.FRAME_GENERATION_PROGRESS,
            {'current': current, 'total': total, 'percent': int(100 * current / total)},
            level=EventLevel.DEBUG,
            **kwargs
        )

    def frame_generation_complete(self, frame_count: int,
                                   total_duration: float, **kwargs) -> CinemaEvent:
        """Emit frame generation complete event."""
        return self.emit(
            CinemaEventType.FRAME_GENERATION_COMPLETE,
            {'frame_count': frame_count, 'total_duration': total_duration},
            **kwargs
        )

    def frame_generation_failed(self, panel_id: str, error: str, **kwargs) -> CinemaEvent:
        """Emit frame generation failed event."""
        return self.emit(
            CinemaEventType.FRAME_GENERATION_FAILED,
            {'panel_id': panel_id, 'error': error},
            level=EventLevel.ERROR,
            **kwargs
        )

    def consistency_analysis_started(self, frame_count: int, **kwargs) -> CinemaEvent:
        """Emit consistency analysis started event."""
        return self.emit(
            CinemaEventType.CONSISTENCY_ANALYSIS_STARTED,
            {'frame_count': frame_count},
            **kwargs
        )

    def consistency_analysis_complete(self,
                                       avg_score: float,
                                       issues_found: int, **kwargs) -> CinemaEvent:
        """Emit consistency analysis complete event."""
        return self.emit(
            CinemaEventType.CONSISTENCY_ANALYSIS_COMPLETE,
            {'average_score': avg_score, 'issues_found': issues_found},
            **kwargs
        )

    def assembly_started(self, frame_count: int, output_path: str, **kwargs) -> CinemaEvent:
        """Emit assembly started event."""
        return self.emit(
            CinemaEventType.ASSEMBLY_STARTED,
            {'frame_count': frame_count, 'output_path': output_path},
            **kwargs
        )

    def assembly_progress(self, percent: int, **kwargs) -> CinemaEvent:
        """Emit assembly progress event."""
        return self.emit(
            CinemaEventType.ASSEMBLY_PROGRESS,
            {'percent': percent},
            level=EventLevel.DEBUG,
            **kwargs
        )

    def assembly_complete(self, output_path: str, duration: float,
                          file_size: int, **kwargs) -> CinemaEvent:
        """Emit assembly complete event."""
        return self.emit(
            CinemaEventType.ASSEMBLY_COMPLETE,
            {
                'output_path': output_path,
                'duration_seconds': duration,
                'file_size_bytes': file_size,
            },
            **kwargs
        )

    def assembly_failed(self, error: str, **kwargs) -> CinemaEvent:
        """Emit assembly failed event."""
        return self.emit(
            CinemaEventType.ASSEMBLY_FAILED,
            {'error': error},
            level=EventLevel.ERROR,
            **kwargs
        )

    def pipeline_started(self, stages: List[str], **kwargs) -> CinemaEvent:
        """Emit pipeline started event."""
        return self.emit(
            CinemaEventType.PIPELINE_STARTED,
            {'stages': stages},
            **kwargs
        )

    def pipeline_stage_complete(self, stage: str, duration: float, **kwargs) -> CinemaEvent:
        """Emit pipeline stage complete event."""
        return self.emit(
            CinemaEventType.PIPELINE_STAGE_COMPLETE,
            {'stage': stage, 'duration_seconds': duration},
            **kwargs
        )

    def pipeline_complete(self, total_duration: float, **kwargs) -> CinemaEvent:
        """Emit pipeline complete event."""
        return self.emit(
            CinemaEventType.PIPELINE_COMPLETE,
            {'total_duration_seconds': total_duration},
            **kwargs
        )

    def pipeline_failed(self, stage: str, error: str, **kwargs) -> CinemaEvent:
        """Emit pipeline failed event."""
        return self.emit(
            CinemaEventType.PIPELINE_FAILED,
            {'failed_stage': stage, 'error': error},
            level=EventLevel.ERROR,
            **kwargs
        )

    def get_event_log(self) -> List[CinemaEvent]:
        """Get all emitted events."""
        return list(self._event_log)

    def clear_event_log(self) -> None:
        """Clear the event log."""
        self._event_log.clear()

    def get_events_by_type(self, event_type: CinemaEventType) -> List[CinemaEvent]:
        """Filter events by type."""
        return [e for e in self._event_log if e.event_type == event_type]

    def get_error_events(self) -> List[CinemaEvent]:
        """Get all error-level events."""
        return [e for e in self._event_log if e.level in (EventLevel.ERROR, EventLevel.CRITICAL)]


# Convenience function for quick event creation
def create_event(event_type: CinemaEventType,
                 data: Optional[Dict[str, Any]] = None,
                 actor: str = "cinema",
                 **kwargs) -> CinemaEvent:
    """
    Create a cinema event without an emitter.

    Args:
        event_type: Type of event
        data: Event data
        actor: Actor name
        **kwargs: Additional metadata

    Returns:
        A new CinemaEvent.
    """
    return CinemaEvent(
        event_type=event_type,
        actor=actor,
        data=data or {},
        metadata=kwargs,
    )


__all__ = [
    'CinemaEventType',
    'EventLevel',
    'CinemaEvent',
    'CinemaEventEmitter',
    'create_event',
]
