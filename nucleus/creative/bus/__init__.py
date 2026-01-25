"""
Pluribus Creative Bus Module
============================

Bus event infrastructure for the Creative subsystem.

This module provides:
- EventSchema: Dataclass for bus event structure
- BusEmitter: Class for emitting events to the NDJSON bus
- Topic definitions and validation for creative subsystem events

Example:
    >>> from nucleus.creative.bus import BusEmitter, TOPICS
    >>> emitter = BusEmitter()
    >>> emitter.emit(TOPICS.VISUAL_RENDER, {"prompt": "sunset"})
"""

__version__ = "1.0.0"
__all__ = [
    # Schemas
    "EventSchema",
    "EventKind",
    "EventLevel",
    "validate_event",
    "event_to_dict",
    "event_from_dict",
    # Emitter
    "BusEmitter",
    "DEFAULT_BUS_PATH",
    # Topics
    "TOPICS",
    "TopicRegistry",
    "validate_topic",
    "is_creative_topic",
    "get_subsystem_from_topic",
    "list_topics_for_subsystem",
]

from .schemas import (
    EventSchema,
    EventKind,
    EventLevel,
    validate_event,
    event_to_dict,
    event_from_dict,
)

from .emitter import (
    BusEmitter,
    DEFAULT_BUS_PATH,
)

from .topics import (
    TOPICS,
    TopicRegistry,
    validate_topic,
    is_creative_topic,
    get_subsystem_from_topic,
    list_topics_for_subsystem,
)
