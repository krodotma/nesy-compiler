"""
VIL Pipeline Module
Integration adapters for connecting vision and learning components.

Version: 1.0
Date: 2026-01-25
"""

from .theia_adapter import (
    TheiaVILAdapter,
    create_theia_adapter,
    VLMInference,
    THEIA_AVAILABLE,
)
from .entropy import (
    EntropyMetrics,
    EntropyNormalizer,
    compute_h_star,
    get_entropy_normalizer,
)
from .icl_pipeline import (
    ICLStrategy,
    ICLExample,
    VisionToICLPipeline,
    create_icl_pipeline,
)
from .vision_tracking import (
    VisionState,
    VisionLineage,
    VisionCMPMetrics,
    VisionFrame,
    VisionTracker,
    create_vision_tracker,
)
from .vision_quality import (
    QualityScore,
    DeduplicationResult,
    VisionQualityProcessor,
    create_quality_processor,
)
from .vision_pipeline import (
    PipelineError,
    PipelineResult,
    VisionPipeline,
    create_vision_pipeline,
)

__all__ = [
    "TheiaVILAdapter",
    "create_theia_adapter",
    "VLMInference",
    "THEIA_AVAILABLE",
    "EntropyMetrics",
    "EntropyNormalizer",
    "compute_h_star",
    "get_entropy_normalizer",
    "ICLStrategy",
    "ICLExample",
    "VisionToICLPipeline",
    "create_icl_pipeline",
    "VisionState",
    "VisionLineage",
    "VisionCMPMetrics",
    "VisionFrame",
    "VisionTracker",
    "create_vision_tracker",
    "QualityScore",
    "DeduplicationResult",
    "VisionQualityProcessor",
    "create_quality_processor",
    "PipelineError",
    "PipelineResult",
    "VisionPipeline",
    "create_vision_pipeline",
]
