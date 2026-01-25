"""
VIL ICL+ (In-Context Learning Plus) Module
Advanced ICL with geometric embeddings and quality scoring.

Features:
1. Geometric ICL buffer with S^n/H^n embeddings
2. Multi-strategy example selection
3. H* entropy-based quality scoring
4. CMP-driven buffer pruning
5. Deduplication via fingerprinting

Integration points:
- Vision Pipeline → ICL Buffer
- Metalearning → ICL Selection Strategy
- CMP → Buffer Pruning
- Synthesis → ICL Example Generation

Version: 1.0
Date: 2026-01-25
"""

from .buffer import (
    ICLExampleType,
    ICLExample,
    ICLBufferStats,
    GeometricICLBuffer,
    create_geometric_icl_buffer,
)
from .selection import (
    SelectionStrategy,
    SelectionConfig,
    SelectionResult,
    ICLExampleSelector,
    create_icl_example_selector,
)
from .embeddings import (
    ManifoldType,
    ManifoldEmbedding,
    GeometricEmbedder,
    create_geometric_embedder,
)
from .quality import (
    QualityThreshold,
    QualityMetrics,
    QualityReport,
    ICLQualityScorer,
    create_icl_quality_scorer,
)

__all__ = [
    "ICLExampleType",
    "ICLExample",
    "ICLBufferStats",
    "GeometricICLBuffer",
    "create_geometric_icl_buffer",
    "SelectionStrategy",
    "SelectionConfig",
    "SelectionResult",
    "ICLExampleSelector",
    "create_icl_example_selector",
    "ManifoldType",
    "ManifoldEmbedding",
    "GeometricEmbedder",
    "create_geometric_embedder",
    "QualityThreshold",
    "QualityMetrics",
    "QualityReport",
    "ICLQualityScorer",
    "create_icl_quality_scorer",
]
