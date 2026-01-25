"""
Pipelines Module for Creative Section
======================================

Provides pipeline orchestration, presets, and unified pipeline interfaces
for coordinating multi-stage creative generation workflows.

Components:
- PipelineOrchestrator: Execute multi-stage pipelines with error recovery
- PIPELINE_PRESETS: Pre-configured pipelines for common workflows
- UnifiedPipeline: Cross-subsystem pipeline coordination

Example:
    >>> from nucleus.creative.pipelines import (
    ...     PipelineOrchestrator,
    ...     PIPELINE_PRESETS,
    ...     UnifiedPipeline,
    ... )
    >>> orchestrator = PipelineOrchestrator()
    >>> result = await orchestrator.execute(PIPELINE_PRESETS["visual_generation"])
"""

from __future__ import annotations

from .orchestrator import (
    PipelineOrchestrator,
    PipelineConfig,
    StageConfig,
    StageResult,
    PipelineResult,
    PipelineError,
    StageError,
    RecoveryStrategy,
    RecoveryConfig,
)

from .presets import (
    PIPELINE_PRESETS,
    get_preset,
    list_presets,
    PipelinePreset,
    StagePreset,
)

from .unified_pipeline import (
    UnifiedPipeline,
    UnifiedPipelineConfig,
    ResourcePool,
    ResourceAllocation,
    PipelineExecutionContext,
)

__all__ = [
    # Orchestrator
    "PipelineOrchestrator",
    "PipelineConfig",
    "StageConfig",
    "StageResult",
    "PipelineResult",
    "PipelineError",
    "StageError",
    "RecoveryStrategy",
    "RecoveryConfig",
    # Presets
    "PIPELINE_PRESETS",
    "get_preset",
    "list_presets",
    "PipelinePreset",
    "StagePreset",
    # Unified Pipeline
    "UnifiedPipeline",
    "UnifiedPipelineConfig",
    "ResourcePool",
    "ResourceAllocation",
    "PipelineExecutionContext",
]
