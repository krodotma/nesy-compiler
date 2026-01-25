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
]
