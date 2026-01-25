"""
Unified Pipeline Interface
===========================

Provides cross-subsystem pipeline coordination and resource management.

The UnifiedPipeline class enables complex workflows that span multiple
creative subsystems with automatic resource allocation, load balancing,
and execution optimization.

Example:
    >>> config = UnifiedPipelineConfig(
    ...     name="avatar_video",
    ...     subsystems=["avatars", "cinema", "auralux"],
    ... )
    >>> pipeline = UnifiedPipeline(config)
    >>> result = await pipeline.run({"image": input_image})
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import weakref

from .orchestrator import (
    PipelineOrchestrator,
    PipelineConfig,
    PipelineResult,
    StageConfig,
    StageResult,
    PipelineError,
    RecoveryConfig,
    RecoveryStrategy,
)
from .presets import (
    PipelinePreset,
    get_preset,
)

T = TypeVar("T")


# =============================================================================
# RESOURCE TYPES
# =============================================================================


class ResourceType(Enum):
    """Types of resources that can be allocated."""

    GPU = auto()
    CPU = auto()
    MEMORY = auto()
    DISK = auto()
    NETWORK = auto()
    MODEL = auto()


@dataclass
class ResourceAllocation:
    """An allocation of resources for a pipeline."""

    allocation_id: str
    resource_type: ResourceType
    amount: float
    unit: str
    priority: int = 50
    reserved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if allocation has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "allocation_id": self.allocation_id,
            "resource_type": self.resource_type.name,
            "amount": self.amount,
            "unit": self.unit,
            "priority": self.priority,
            "reserved_at": self.reserved_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class ResourceRequirements:
    """Resource requirements for a subsystem or stage."""

    gpu_memory_gb: float = 0.0
    cpu_cores: float = 1.0
    memory_gb: float = 2.0
    disk_gb: float = 1.0
    models: List[str] = field(default_factory=list)
    optional_gpu: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "gpu_memory_gb": self.gpu_memory_gb,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "disk_gb": self.disk_gb,
            "models": self.models,
            "optional_gpu": self.optional_gpu,
        }


# =============================================================================
# RESOURCE POOL
# =============================================================================


class ResourcePool:
    """
    Manages resource allocation across pipelines.

    Features:
    - Thread-safe resource allocation
    - Priority-based scheduling
    - Automatic cleanup of expired allocations
    - Resource utilization tracking
    """

    def __init__(
        self,
        gpu_memory_gb: float = 16.0,
        cpu_cores: float = 8.0,
        memory_gb: float = 32.0,
        disk_gb: float = 100.0,
    ):
        self._capacity = {
            ResourceType.GPU: gpu_memory_gb,
            ResourceType.CPU: cpu_cores,
            ResourceType.MEMORY: memory_gb,
            ResourceType.DISK: disk_gb,
        }
        self._allocated: Dict[str, ResourceAllocation] = {}
        self._lock = asyncio.Lock()
        self._waiters: Dict[ResourceType, List[asyncio.Event]] = {
            rt: [] for rt in ResourceType
        }

    @property
    def utilization(self) -> Dict[str, float]:
        """Get current resource utilization (0-1)."""
        util = {}
        for rt in [ResourceType.GPU, ResourceType.CPU, ResourceType.MEMORY, ResourceType.DISK]:
            allocated = sum(
                a.amount
                for a in self._allocated.values()
                if a.resource_type == rt and not a.is_expired()
            )
            capacity = self._capacity.get(rt, 1.0)
            util[rt.name.lower()] = allocated / capacity if capacity > 0 else 0.0
        return util

    async def allocate(
        self,
        resource_type: ResourceType,
        amount: float,
        priority: int = 50,
        timeout_s: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[ResourceAllocation]:
        """
        Allocate resources.

        Args:
            resource_type: Type of resource to allocate
            amount: Amount to allocate
            priority: Priority (higher = more important)
            timeout_s: Optional timeout for allocation
            metadata: Optional metadata

        Returns:
            ResourceAllocation if successful, None if resources unavailable
        """
        async with self._lock:
            # Clean expired allocations
            self._cleanup_expired()

            # Check availability
            available = self._get_available(resource_type)
            if available >= amount:
                allocation = ResourceAllocation(
                    allocation_id=f"alloc-{uuid.uuid4().hex[:8]}",
                    resource_type=resource_type,
                    amount=amount,
                    unit=self._get_unit(resource_type),
                    priority=priority,
                    metadata=metadata or {},
                )
                self._allocated[allocation.allocation_id] = allocation
                return allocation

        # Wait for resources if timeout specified
        if timeout_s and timeout_s > 0:
            event = asyncio.Event()
            self._waiters[resource_type].append(event)
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout_s)
                # Retry allocation
                return await self.allocate(
                    resource_type, amount, priority, timeout_s=0
                )
            except asyncio.TimeoutError:
                return None
            finally:
                self._waiters[resource_type].remove(event)

        return None

    async def release(self, allocation_id: str) -> bool:
        """
        Release an allocation.

        Args:
            allocation_id: ID of allocation to release

        Returns:
            True if released, False if not found
        """
        async with self._lock:
            if allocation_id in self._allocated:
                allocation = self._allocated.pop(allocation_id)
                # Notify waiters
                for event in self._waiters.get(allocation.resource_type, []):
                    event.set()
                return True
            return False

    async def allocate_requirements(
        self,
        requirements: ResourceRequirements,
        priority: int = 50,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, ResourceAllocation]:
        """
        Allocate all resources specified in requirements.

        Args:
            requirements: Resource requirements
            priority: Priority for allocations
            timeout_s: Optional timeout

        Returns:
            Dict mapping resource type names to allocations
        """
        allocations = {}

        # Allocate each resource type
        resource_map = [
            (ResourceType.GPU, requirements.gpu_memory_gb),
            (ResourceType.CPU, requirements.cpu_cores),
            (ResourceType.MEMORY, requirements.memory_gb),
            (ResourceType.DISK, requirements.disk_gb),
        ]

        for resource_type, amount in resource_map:
            if amount > 0:
                # Skip GPU if optional and unavailable
                if (
                    resource_type == ResourceType.GPU
                    and requirements.optional_gpu
                    and self._get_available(resource_type) < amount
                ):
                    continue

                allocation = await self.allocate(
                    resource_type, amount, priority, timeout_s
                )
                if allocation:
                    allocations[resource_type.name.lower()] = allocation
                elif resource_type != ResourceType.GPU or not requirements.optional_gpu:
                    # Rollback on failure
                    for alloc in allocations.values():
                        await self.release(alloc.allocation_id)
                    raise ResourceError(
                        f"Failed to allocate {amount} {resource_type.name}"
                    )

        return allocations

    async def release_all(self, allocations: Dict[str, ResourceAllocation]) -> None:
        """Release all allocations in a dict."""
        for allocation in allocations.values():
            await self.release(allocation.allocation_id)

    def _get_available(self, resource_type: ResourceType) -> float:
        """Get available amount of a resource type."""
        capacity = self._capacity.get(resource_type, 0.0)
        allocated = sum(
            a.amount
            for a in self._allocated.values()
            if a.resource_type == resource_type and not a.is_expired()
        )
        return max(0.0, capacity - allocated)

    def _get_unit(self, resource_type: ResourceType) -> str:
        """Get unit for a resource type."""
        units = {
            ResourceType.GPU: "GB",
            ResourceType.CPU: "cores",
            ResourceType.MEMORY: "GB",
            ResourceType.DISK: "GB",
            ResourceType.NETWORK: "Mbps",
            ResourceType.MODEL: "instance",
        }
        return units.get(resource_type, "unit")

    def _cleanup_expired(self) -> None:
        """Remove expired allocations."""
        expired = [
            aid
            for aid, alloc in self._allocated.items()
            if alloc.is_expired()
        ]
        for aid in expired:
            del self._allocated[aid]


class ResourceError(Exception):
    """Resource allocation error."""
    pass


# =============================================================================
# EXECUTION CONTEXT
# =============================================================================


@dataclass
class PipelineExecutionContext:
    """
    Execution context for a unified pipeline run.

    Contains all state needed during pipeline execution including
    resource allocations, intermediate results, and progress tracking.
    """

    pipeline_id: str
    pipeline_name: str
    started_at: datetime
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] = field(default_factory=dict)
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    allocations: Dict[str, ResourceAllocation] = field(default_factory=dict)
    checkpoints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _cancelled: bool = field(default=False, init=False)
    _progress_callbacks: List[Callable[[str, float], None]] = field(
        default_factory=list, init=False
    )

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled

    def add_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        """Add a progress callback."""
        self._progress_callbacks.append(callback)

    def emit_progress(self, stage: str, progress: float) -> None:
        """Emit progress update."""
        for callback in self._progress_callbacks:
            try:
                callback(stage, progress)
            except Exception:
                pass

    def checkpoint(self, name: str) -> None:
        """Create a checkpoint of current state."""
        self.checkpoints[name] = {
            "outputs": dict(self.outputs),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def restore_checkpoint(self, name: str) -> bool:
        """Restore from a checkpoint."""
        if name in self.checkpoints:
            self.outputs = dict(self.checkpoints[name]["outputs"])
            return True
        return False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "started_at": self.started_at.isoformat(),
            "stage_results": {k: v.to_dict() for k, v in self.stage_results.items()},
            "checkpoints": list(self.checkpoints.keys()),
            "cancelled": self._cancelled,
        }


# =============================================================================
# UNIFIED PIPELINE CONFIG
# =============================================================================


@dataclass
class UnifiedPipelineConfig:
    """Configuration for a unified pipeline."""

    name: str
    description: str = ""
    subsystems: List[str] = field(default_factory=list)
    preset: Optional[str] = None
    stages: List[StageConfig] = field(default_factory=list)
    resource_requirements: Optional[ResourceRequirements] = None
    timeout_s: float = 3600.0
    max_concurrent_stages: int = 4
    checkpointing: bool = True
    emit_events: bool = True
    retry_failed_stages: bool = True
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Load stages from preset if specified."""
        if self.preset and not self.stages:
            preset_obj = get_preset(self.preset)
            if preset_obj:
                config = preset_obj.to_config()
                self.stages = config.stages
                if not self.description:
                    self.description = config.description

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "subsystems": self.subsystems,
            "preset": self.preset,
            "stages": [s.to_dict() for s in self.stages],
            "timeout_s": self.timeout_s,
            "max_concurrent_stages": self.max_concurrent_stages,
        }


# =============================================================================
# UNIFIED PIPELINE
# =============================================================================


class UnifiedPipeline:
    """
    Unified pipeline for cross-subsystem coordination.

    Provides:
    - Resource management and allocation
    - Cross-subsystem stage coordination
    - Progress tracking and event emission
    - Checkpointing and recovery
    - Concurrent stage execution

    Example:
        >>> config = UnifiedPipelineConfig(
        ...     name="avatar_video",
        ...     preset="avatar_creation",
        ... )
        >>> pipeline = UnifiedPipeline(config)
        >>> async with pipeline.run_context() as ctx:
        ...     result = await pipeline.run({"image": input_image}, ctx)
    """

    def __init__(
        self,
        config: UnifiedPipelineConfig,
        resource_pool: Optional[ResourcePool] = None,
        orchestrator: Optional[PipelineOrchestrator] = None,
    ):
        self.config = config
        self.resource_pool = resource_pool or ResourcePool()
        self.orchestrator = orchestrator or PipelineOrchestrator()
        self._active_contexts: Dict[str, PipelineExecutionContext] = {}

    async def run(
        self,
        inputs: Dict[str, Any],
        context: Optional[PipelineExecutionContext] = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> PipelineResult:
        """
        Run the unified pipeline.

        Args:
            inputs: Initial inputs for the pipeline
            context: Optional execution context
            on_progress: Optional progress callback

        Returns:
            PipelineResult with execution details
        """
        # Create context if not provided
        if context is None:
            context = self._create_context(inputs)

        if on_progress:
            context.add_progress_callback(on_progress)

        self._active_contexts[context.pipeline_id] = context

        try:
            # Allocate resources
            if self.config.resource_requirements:
                context.allocations = await self.resource_pool.allocate_requirements(
                    self.config.resource_requirements,
                    priority=50,
                    timeout_s=60.0,
                )

            # Build pipeline config
            pipeline_config = PipelineConfig(
                name=self.config.name,
                description=self.config.description,
                stages=self.config.stages,
                timeout_s=self.config.timeout_s,
                checkpointing=self.config.checkpointing,
                emit_events=self.config.emit_events,
                on_progress=context.emit_progress,
            )

            # Execute pipeline
            result = await self.orchestrator.execute(
                pipeline_config,
                initial_inputs={**inputs, **context.outputs},
            )

            # Update context with results
            context.outputs.update(result.final_output)
            for stage_result in result.stages:
                context.stage_results[stage_result.name] = stage_result

            return result

        finally:
            # Release resources
            if context.allocations:
                await self.resource_pool.release_all(context.allocations)
            del self._active_contexts[context.pipeline_id]

    @asynccontextmanager
    async def run_context(
        self,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[PipelineExecutionContext]:
        """
        Create an execution context for the pipeline.

        Use as async context manager:
            >>> async with pipeline.run_context() as ctx:
            ...     await pipeline.run(inputs, ctx)
            ...     # ctx contains results

        Args:
            inputs: Optional initial inputs

        Yields:
            PipelineExecutionContext for the run
        """
        context = self._create_context(inputs or {})
        try:
            yield context
        finally:
            # Cleanup on exit
            if context.allocations:
                await self.resource_pool.release_all(context.allocations)

    def _create_context(self, inputs: Dict[str, Any]) -> PipelineExecutionContext:
        """Create a new execution context."""
        return PipelineExecutionContext(
            pipeline_id=f"unified-{uuid.uuid4().hex[:12]}",
            pipeline_name=self.config.name,
            started_at=datetime.now(timezone.utc),
            inputs=inputs,
        )

    async def run_stage(
        self,
        stage_name: str,
        context: PipelineExecutionContext,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> StageResult:
        """
        Run a single stage within a context.

        Args:
            stage_name: Name of stage to run
            context: Execution context
            inputs: Optional additional inputs

        Returns:
            StageResult from stage execution

        Raises:
            ValueError: If stage not found
        """
        # Find stage config
        stage_config = None
        for stage in self.config.stages:
            if stage.name == stage_name:
                stage_config = stage
                break

        if stage_config is None:
            raise ValueError(f"Stage '{stage_name}' not found in pipeline")

        # Merge inputs
        merged_inputs = {**context.outputs, **(inputs or {})}

        # Execute single-stage pipeline
        single_config = PipelineConfig(
            name=f"{self.config.name}:{stage_name}",
            stages=[stage_config],
        )

        result = await self.orchestrator.execute(single_config, merged_inputs)

        # Update context
        if result.stages:
            stage_result = result.stages[0]
            context.stage_results[stage_name] = stage_result
            if stage_result.success:
                context.outputs.update(stage_result.output)
            return stage_result

        return StageResult(name=stage_name, status="failed", error="No result")

    async def run_parallel_stages(
        self,
        stage_names: List[str],
        context: PipelineExecutionContext,
    ) -> Dict[str, StageResult]:
        """
        Run multiple stages in parallel.

        Args:
            stage_names: Names of stages to run
            context: Execution context

        Returns:
            Dict mapping stage names to results
        """
        tasks = [
            self.run_stage(name, context)
            for name in stage_names
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for name, result in zip(stage_names, results):
            if isinstance(result, Exception):
                output[name] = StageResult(
                    name=name,
                    status="failed",
                    error=str(result),
                )
            else:
                output[name] = result

        return output

    def get_active_pipelines(self) -> List[PipelineExecutionContext]:
        """Get list of currently executing pipelines."""
        return list(self._active_contexts.values())

    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """
        Request cancellation of a running pipeline.

        Args:
            pipeline_id: Pipeline ID to cancel

        Returns:
            True if pipeline was found and cancellation requested
        """
        if pipeline_id in self._active_contexts:
            self._active_contexts[pipeline_id].cancel()
            return True
        return False


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_unified_pipeline(
    name: str,
    preset: Optional[str] = None,
    stages: Optional[List[StageConfig]] = None,
    subsystems: Optional[List[str]] = None,
    **kwargs,
) -> UnifiedPipeline:
    """
    Create a unified pipeline.

    Args:
        name: Pipeline name
        preset: Optional preset name to use
        stages: Optional list of stage configs
        subsystems: Optional list of subsystems used
        **kwargs: Additional config options

    Returns:
        UnifiedPipeline instance
    """
    config = UnifiedPipelineConfig(
        name=name,
        preset=preset,
        stages=stages or [],
        subsystems=subsystems or [],
        **kwargs,
    )
    return UnifiedPipeline(config)


def create_pipeline_from_preset(
    preset_name: str,
    overrides: Optional[Dict[str, Any]] = None,
    resource_pool: Optional[ResourcePool] = None,
) -> UnifiedPipeline:
    """
    Create a unified pipeline from a preset.

    Args:
        preset_name: Name of the preset
        overrides: Optional stage parameter overrides
        resource_pool: Optional shared resource pool

    Returns:
        UnifiedPipeline instance

    Raises:
        ValueError: If preset not found
    """
    preset = get_preset(preset_name)
    if preset is None:
        raise ValueError(f"Preset '{preset_name}' not found")

    config = UnifiedPipelineConfig(
        name=preset.name,
        description=preset.description,
        preset=preset_name,
    )

    # Apply overrides
    if overrides:
        for stage in config.stages:
            if stage.name in overrides:
                stage.params.update(overrides[stage.name])

    return UnifiedPipeline(config, resource_pool=resource_pool)


# =============================================================================
# SUBSYSTEM RESOURCE DEFAULTS
# =============================================================================

SUBSYSTEM_RESOURCES: Dict[str, ResourceRequirements] = {
    "visual": ResourceRequirements(
        gpu_memory_gb=8.0,
        cpu_cores=4.0,
        memory_gb=16.0,
        disk_gb=10.0,
        models=["stable-diffusion", "real-esrgan"],
    ),
    "cinema": ResourceRequirements(
        gpu_memory_gb=12.0,
        cpu_cores=8.0,
        memory_gb=32.0,
        disk_gb=50.0,
        models=["video-diffusion", "temporal-consistency"],
    ),
    "avatars": ResourceRequirements(
        gpu_memory_gb=10.0,
        cpu_cores=4.0,
        memory_gb=16.0,
        disk_gb=20.0,
        models=["smpl-x", "3dgs-avatar"],
    ),
    "auralux": ResourceRequirements(
        gpu_memory_gb=4.0,
        cpu_cores=2.0,
        memory_gb=8.0,
        disk_gb=5.0,
        models=["tts-model", "vocoder"],
        optional_gpu=True,
    ),
    "grammars": ResourceRequirements(
        gpu_memory_gb=0.0,
        cpu_cores=4.0,
        memory_gb=8.0,
        disk_gb=1.0,
        optional_gpu=True,
    ),
    "dits": ResourceRequirements(
        gpu_memory_gb=0.0,
        cpu_cores=2.0,
        memory_gb=4.0,
        disk_gb=1.0,
        optional_gpu=True,
    ),
}


def get_subsystem_resources(subsystem: str) -> ResourceRequirements:
    """Get default resource requirements for a subsystem."""
    return SUBSYSTEM_RESOURCES.get(
        subsystem,
        ResourceRequirements(),
    )


def combine_resource_requirements(
    *requirements: ResourceRequirements,
) -> ResourceRequirements:
    """Combine multiple resource requirements (takes max of each)."""
    if not requirements:
        return ResourceRequirements()

    return ResourceRequirements(
        gpu_memory_gb=max(r.gpu_memory_gb for r in requirements),
        cpu_cores=max(r.cpu_cores for r in requirements),
        memory_gb=max(r.memory_gb for r in requirements),
        disk_gb=sum(r.disk_gb for r in requirements),  # Sum disk space
        models=list(set(m for r in requirements for m in r.models)),
        optional_gpu=all(r.optional_gpu for r in requirements),
    )
