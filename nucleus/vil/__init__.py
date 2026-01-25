"""
VIL (Vision-Integration-Learning) Package
Unified vision-metalearning integration layer.

Integration of:
- Theia VLM Specialist (vision-language)
- Omega Metalearning (L8 Gödel machines)
- Learning Tower L8 (neurosymbolic architecture)
- Clade Manager CMP (evolutionary fitness)
- Agent Lightning Trainer (RL, experience replay)
- Meta Learner (pattern recognition)
- VisionEye Component (screen capture)

Version: 1.0
Date: 2026-01-25
"""

__version__ = "1.0.0"

from .coordinator import (
    VILCoordinator,
    create_vil_coordinator,
    VILState,
    VILConfig,
    PHI,
    CMP_DISCOUNT,
    GLOBAL_CMP_FLOOR,
)
from .events import (
    VILEventType,
    CmpMetrics,
    VILEvent,
    VisionEvent,
    LearningEvent,
    SynthesisEvent,
    CMPEvent,
    IntegrationEvent,
    create_vil_event,
    create_vision_event,
    create_learning_event,
    create_synthesis_event,
    create_cmp_event,
    create_integration_event,
    validate_event,
    event_to_bus,
    create_trace_id,
    parse_trace_id,
)
from .interfaces import (
    GeometricEmbedding,
    AttractorDynamics,
    VILVisionInterface,
    VILLearningInterface,
    VILSynthesisInterface,
    VILGeometricInterface,
    VILCMPInterface,
    VILICLInterface,
    VILInterface,
)

__all__ = [
    # Coordinator
    "VILCoordinator",
    "create_vil_coordinator",
    "VILState",
    "VILConfig",
    "PHI",
    "CMP_DISCOUNT",
    "GLOBAL_CMP_FLOOR",

    # Events
    "VILEventType",
    "CmpMetrics",
    "VILEvent",
    "VisionEvent",
    "LearningEvent",
    "SynthesisEvent",
    "CMPEvent",
    "IntegrationEvent",
    "create_vil_event",
    "create_vision_event",
    "create_learning_event",
    "create_synthesis_event",
    "create_cmp_event",
    "create_integration_event",
    "validate_event",
    "event_to_bus",
    "create_trace_id",
    "parse_trace_id",

    # Interfaces
    "GeometricEmbedding",
    "AttractorDynamics",
    "VILVisionInterface",
    "VILLearningInterface",
    "VILSynthesisInterface",
    "VILGeometricInterface",
    "VILCMPInterface",
    "VILICLInterface",
    "VILInterface",
]
