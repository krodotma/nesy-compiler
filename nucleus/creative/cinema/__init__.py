"""
Cinema Subsystem
================

Video generation, temporal consistency, multi-shot narrative.
"""

from __future__ import annotations

# Import submodules
from . import script_parser
from . import storyboard
from . import frame_generator
from . import temporal_consistency
from . import video_assembler
from . import bus_events

# Export from script_parser
from .script_parser import (
    ElementType,
    SceneElement,
    Script,
    FountainParser,
)

# Export from storyboard
from .storyboard import (
    ShotType,
    CameraMovement,
    ShotAngle,
    StoryboardPanel,
    Storyboard,
    StoryboardGenerator,
)

# Export from frame_generator
from .frame_generator import (
    GenerationModel,
    AspectRatio,
    FrameGenerationConfig,
    GenerationResult,
    FrameGenerator,
)

# Export from temporal_consistency
from .temporal_consistency import (
    FlowMethod,
    ConsistencyMode,
    MotionVector,
    FlowField,
    ConsistencyReport,
    TemporalConsistencyEngine,
)

# Export from video_assembler
from .video_assembler import (
    VideoCodec,
    AudioCodec,
    ContainerFormat,
    QualityPreset,
    AudioTrack,
    AssemblyConfig,
    AssemblyResult,
    VideoAssembler,
)

# Export from bus_events
from .bus_events import (
    CinemaEventType,
    EventLevel,
    CinemaEvent,
    CinemaEventEmitter,
    create_event,
)

__all__ = [
    # Submodules
    "script_parser",
    "storyboard",
    "frame_generator",
    "temporal_consistency",
    "video_assembler",
    "bus_events",
    # script_parser
    "ElementType",
    "SceneElement",
    "Script",
    "FountainParser",
    # storyboard
    "ShotType",
    "CameraMovement",
    "ShotAngle",
    "StoryboardPanel",
    "Storyboard",
    "StoryboardGenerator",
    # frame_generator
    "GenerationModel",
    "AspectRatio",
    "FrameGenerationConfig",
    "GenerationResult",
    "FrameGenerator",
    # temporal_consistency
    "FlowMethod",
    "ConsistencyMode",
    "MotionVector",
    "FlowField",
    "ConsistencyReport",
    "TemporalConsistencyEngine",
    # video_assembler
    "VideoCodec",
    "AudioCodec",
    "ContainerFormat",
    "QualityPreset",
    "AudioTrack",
    "AssemblyConfig",
    "AssemblyResult",
    "VideoAssembler",
    # bus_events
    "CinemaEventType",
    "EventLevel",
    "CinemaEvent",
    "CinemaEventEmitter",
    "create_event",
]
