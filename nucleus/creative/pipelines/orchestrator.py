"""
Pipeline Orchestrator
======================

Provides stage execution, sequencing, error handling, and recovery
for multi-stage creative pipelines.

Example:
    >>> orchestrator = PipelineOrchestrator()
    >>> config = PipelineConfig(
    ...     name="visual_generation",
    ...     stages=[
    ...         StageConfig(name="generate", subsystem="visual", operation="generate"),
    ...         StageConfig(name="upscale", subsystem="visual", operation="upscale"),
    ...     ],
    ... )
    >>> result = await orchestrator.execute(config)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T")


# =============================================================================
# ERROR TYPES
# =============================================================================


class PipelineError(Exception):
    """Base exception for pipeline errors."""

    def __init__(
        self,
        message: str,
        pipeline_id: Optional[str] = None,
        stage_name: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.pipeline_id = pipeline_id
        self.stage_name = stage_name
        self.cause = cause

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "type": self.__class__.__name__,
            "message": str(self),
            "pipeline_id": self.pipeline_id,
            "stage_name": self.stage_name,
            "cause": str(self.cause) if self.cause else None,
        }


class StageError(PipelineError):
    """Error during stage execution."""

    def __init__(
        self,
        message: str,
        stage_name: str,
        stage_index: int = 0,
        recoverable: bool = True,
        **kwargs,
    ):
        super().__init__(message, stage_name=stage_name, **kwargs)
        self.stage_index = stage_index
        self.recoverable = recoverable


# =============================================================================
# RECOVERY STRATEGIES
# =============================================================================


class RecoveryStrategy(Enum):
    """Recovery strategies for failed stages."""

    NONE = auto()  # No recovery, fail immediately
    RETRY = auto()  # Retry the failed stage
    SKIP = auto()  # Skip the failed stage and continue
    FALLBACK = auto()  # Use a fallback handler
    ROLLBACK = auto()  # Rollback to previous checkpoint
    RESTART = auto()  # Restart the entire pipeline


@dataclass
class RecoveryConfig:
    """Configuration for recovery behavior."""

    strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    max_retries: int = 3
    retry_delay_s: float = 1.0
    retry_backoff: float = 2.0
    fallback_handler: Optional[Callable[..., Any]] = None
    on_recovery_failure: RecoveryStrategy = RecoveryStrategy.NONE


# =============================================================================
# STAGE CONFIGURATION
# =============================================================================


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""

    name: str
    subsystem: str
    operation: str
    params: dict = field(default_factory=dict)
    timeout_s: float = 300.0
    required: bool = True
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    depends_on: List[str] = field(default_factory=list)
    on_success: Optional[Callable[[Dict], Any]] = None
    on_failure: Optional[Callable[[Exception], Any]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "subsystem": self.subsystem,
            "operation": self.operation,
            "params": self.params,
            "timeout_s": self.timeout_s,
            "required": self.required,
            "depends_on": self.depends_on,
        }


# =============================================================================
# STAGE RESULT
# =============================================================================


@dataclass
class StageResult:
    """Result from a pipeline stage execution."""

    name: str
    status: str  # "pending", "running", "completed", "failed", "skipped"
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    elapsed_ms: float = 0.0
    retry_count: int = 0
    checksum: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if stage completed successfully."""
        return self.status == "completed"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "elapsed_ms": self.elapsed_ms,
            "retry_count": self.retry_count,
        }


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================


@dataclass
class PipelineConfig:
    """Configuration for a complete pipeline."""

    name: str
    stages: List[StageConfig]
    description: str = ""
    version: str = "1.0.0"
    timeout_s: float = 1800.0  # 30 minutes default
    parallel_stages: bool = False
    checkpointing: bool = True
    emit_events: bool = True
    on_progress: Optional[Callable[[str, float], None]] = None
    on_error: Optional[Callable[[str, Exception], None]] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "stages": [s.to_dict() for s in self.stages],
            "timeout_s": self.timeout_s,
            "parallel_stages": self.parallel_stages,
            "metadata": self.metadata,
        }


# =============================================================================
# PIPELINE RESULT
# =============================================================================


@dataclass
class PipelineResult:
    """Result from pipeline execution."""

    pipeline_id: str
    pipeline_name: str
    status: str  # "pending", "running", "completed", "failed", "partial"
    stages: List[StageResult]
    final_output: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_ms: float = 0.0
    error: Optional[str] = None
    checkpoints: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if pipeline completed successfully."""
        return self.status == "completed"

    @property
    def completed_count(self) -> int:
        """Number of completed stages."""
        return sum(1 for s in self.stages if s.status == "completed")

    @property
    def failed_count(self) -> int:
        """Number of failed stages."""
        return sum(1 for s in self.stages if s.status == "failed")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "pipeline_name": self.pipeline_name,
            "status": self.status,
            "stages": [s.to_dict() for s in self.stages],
            "final_output": self.final_output,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_ms": self.total_ms,
            "error": self.error,
            "checkpoints": self.checkpoints,
        }


# =============================================================================
# STAGE HANDLER PROTOCOL
# =============================================================================


class StageHandler(Protocol):
    """Protocol for stage handlers."""

    async def __call__(
        self,
        inputs: Dict[str, Any],
        stage_config: StageConfig,
    ) -> Dict[str, Any]:
        """Execute stage and return outputs."""
        ...


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================


class PipelineOrchestrator:
    """
    Orchestrates multi-stage pipeline execution with error recovery.

    Features:
    - Stage sequencing and dependency resolution
    - Retry with exponential backoff
    - Checkpointing and rollback
    - Progress tracking and event emission
    - Parallel stage execution (when enabled)

    Example:
        >>> orchestrator = PipelineOrchestrator()
        >>> orchestrator.register_handler("visual", "generate", visual_generate_handler)
        >>> result = await orchestrator.execute(pipeline_config)
    """

    def __init__(self):
        self._handlers: Dict[str, Dict[str, StageHandler]] = {}
        self._checkpoints: Dict[str, Dict[str, Any]] = {}
        self._active_pipelines: Dict[str, PipelineResult] = {}

    def register_handler(
        self,
        subsystem: str,
        operation: str,
        handler: StageHandler,
    ) -> None:
        """
        Register a handler for a subsystem operation.

        Args:
            subsystem: Subsystem name (e.g., "visual", "cinema")
            operation: Operation name (e.g., "generate", "upscale")
            handler: Async function that executes the operation
        """
        if subsystem not in self._handlers:
            self._handlers[subsystem] = {}
        self._handlers[subsystem][operation] = handler

    def get_handler(
        self,
        subsystem: str,
        operation: str,
    ) -> Optional[StageHandler]:
        """Get a registered handler."""
        return self._handlers.get(subsystem, {}).get(operation)

    async def execute(
        self,
        config: PipelineConfig,
        initial_inputs: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Execute a pipeline.

        Args:
            config: Pipeline configuration
            initial_inputs: Initial context/inputs for the pipeline

        Returns:
            PipelineResult with execution details

        Raises:
            PipelineError: If pipeline execution fails critically
        """
        pipeline_id = f"pipe-{uuid.uuid4().hex[:12]}"
        started_at = datetime.now(timezone.utc)
        context = dict(initial_inputs or {})

        result = PipelineResult(
            pipeline_id=pipeline_id,
            pipeline_name=config.name,
            status="running",
            stages=[],
            final_output={},
            started_at=started_at,
        )

        self._active_pipelines[pipeline_id] = result

        # Emit start event
        if config.emit_events:
            self._emit_event(
                "creative.pipeline.started",
                {
                    "pipeline_id": pipeline_id,
                    "name": config.name,
                    "stage_count": len(config.stages),
                },
            )

        try:
            # Resolve stage execution order
            ordered_stages = self._resolve_stage_order(config.stages)

            # Execute stages
            if config.parallel_stages:
                stage_results = await self._execute_parallel(
                    ordered_stages, context, config, pipeline_id
                )
            else:
                stage_results = await self._execute_sequential(
                    ordered_stages, context, config, pipeline_id
                )

            result.stages = stage_results

            # Determine final status
            failed_required = any(
                s.status == "failed"
                for s, cfg in zip(stage_results, ordered_stages)
                if cfg.required
            )

            if failed_required:
                result.status = "failed"
                result.error = "Required stage(s) failed"
            elif any(s.status == "failed" for s in stage_results):
                result.status = "partial"
            else:
                result.status = "completed"

            result.final_output = context
            result.completed_at = datetime.now(timezone.utc)
            result.total_ms = (result.completed_at - started_at).total_seconds() * 1000

            # Emit completion event
            if config.emit_events:
                self._emit_event(
                    "creative.pipeline.completed",
                    {
                        "pipeline_id": pipeline_id,
                        "status": result.status,
                        "duration_ms": result.total_ms,
                        "stages_completed": result.completed_count,
                        "stages_failed": result.failed_count,
                    },
                )

            return result

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.completed_at = datetime.now(timezone.utc)
            result.total_ms = (result.completed_at - started_at).total_seconds() * 1000

            if config.emit_events:
                self._emit_event(
                    "creative.pipeline.failed",
                    {
                        "pipeline_id": pipeline_id,
                        "error": str(e),
                    },
                )

            raise PipelineError(
                f"Pipeline '{config.name}' failed: {e}",
                pipeline_id=pipeline_id,
                cause=e,
            ) from e

        finally:
            del self._active_pipelines[pipeline_id]

    async def _execute_sequential(
        self,
        stages: List[StageConfig],
        context: Dict[str, Any],
        config: PipelineConfig,
        pipeline_id: str,
    ) -> List[StageResult]:
        """Execute stages sequentially."""
        results: List[StageResult] = []

        for i, stage in enumerate(stages):
            progress = (i / len(stages)) * 100

            if config.on_progress:
                config.on_progress(stage.name, progress)

            result = await self._execute_stage(
                stage, context, config, pipeline_id, i
            )
            results.append(result)

            # Update context with outputs if successful
            if result.success:
                context.update(result.output)

                # Create checkpoint if enabled
                if config.checkpointing:
                    checkpoint_id = f"{pipeline_id}-{stage.name}"
                    self._checkpoints[checkpoint_id] = dict(context)

            # Stop if required stage failed
            elif stage.required:
                break

        return results

    async def _execute_parallel(
        self,
        stages: List[StageConfig],
        context: Dict[str, Any],
        config: PipelineConfig,
        pipeline_id: str,
    ) -> List[StageResult]:
        """Execute independent stages in parallel."""
        # Group stages by dependencies
        groups = self._group_stages_by_dependencies(stages)
        results: List[StageResult] = []

        for group in groups:
            # Execute group stages in parallel
            tasks = [
                self._execute_stage(stage, context, config, pipeline_id, i)
                for i, stage in enumerate(group)
            ]
            group_results = await asyncio.gather(*tasks, return_exceptions=True)

            for stage, result in zip(group, group_results):
                if isinstance(result, Exception):
                    results.append(
                        StageResult(
                            name=stage.name,
                            status="failed",
                            error=str(result),
                        )
                    )
                else:
                    results.append(result)
                    if result.success:
                        context.update(result.output)

        return results

    async def _execute_stage(
        self,
        stage: StageConfig,
        context: Dict[str, Any],
        pipeline_config: PipelineConfig,
        pipeline_id: str,
        stage_index: int,
    ) -> StageResult:
        """Execute a single stage with retry and recovery."""
        started_at = datetime.now(timezone.utc)
        retry_count = 0

        result = StageResult(
            name=stage.name,
            status="running",
            started_at=started_at,
        )

        # Emit stage start event
        if pipeline_config.emit_events:
            self._emit_event(
                "creative.pipeline.stage.started",
                {
                    "pipeline_id": pipeline_id,
                    "stage_name": stage.name,
                    "stage_index": stage_index,
                },
            )

        # Get handler
        handler = self.get_handler(stage.subsystem, stage.operation)
        if handler is None:
            # Use mock handler if not registered
            handler = self._create_mock_handler(stage.subsystem, stage.operation)

        # Merge context with stage params
        inputs = {**context, **stage.params}

        # Retry loop
        delay = stage.recovery.retry_delay_s
        last_error: Optional[Exception] = None

        while retry_count <= stage.recovery.max_retries:
            try:
                # Execute with timeout
                output = await asyncio.wait_for(
                    handler(inputs, stage),
                    timeout=stage.timeout_s,
                )

                result.status = "completed"
                result.output = output or {}
                result.retry_count = retry_count

                # Callback on success
                if stage.on_success:
                    try:
                        stage.on_success(output)
                    except Exception:
                        pass  # Don't fail on callback errors

                break

            except asyncio.TimeoutError as e:
                last_error = StageError(
                    f"Stage '{stage.name}' timed out after {stage.timeout_s}s",
                    stage_name=stage.name,
                    stage_index=stage_index,
                    recoverable=True,
                )
            except Exception as e:
                last_error = e

            # Check if we should retry
            if (
                stage.recovery.strategy == RecoveryStrategy.RETRY
                and retry_count < stage.recovery.max_retries
            ):
                retry_count += 1
                await asyncio.sleep(delay)
                delay *= stage.recovery.retry_backoff
            else:
                break

        # Handle failure
        if result.status != "completed":
            result.status = "failed"
            result.error = str(last_error) if last_error else "Unknown error"

            # Try fallback if configured
            if (
                stage.recovery.strategy == RecoveryStrategy.FALLBACK
                and stage.recovery.fallback_handler
            ):
                try:
                    fallback_output = await stage.recovery.fallback_handler(inputs)
                    result.status = "completed"
                    result.output = fallback_output or {}
                    result.error = None
                except Exception as fallback_error:
                    result.error = f"Fallback failed: {fallback_error}"

            # Skip if configured
            if stage.recovery.strategy == RecoveryStrategy.SKIP:
                result.status = "skipped"
                result.error = None

            # Callback on failure
            if stage.on_failure and last_error:
                try:
                    stage.on_failure(last_error)
                except Exception:
                    pass

            # Notify pipeline error handler
            if pipeline_config.on_error and last_error:
                pipeline_config.on_error(stage.name, last_error)

        result.completed_at = datetime.now(timezone.utc)
        result.elapsed_ms = (result.completed_at - started_at).total_seconds() * 1000

        # Emit stage completion event
        if pipeline_config.emit_events:
            self._emit_event(
                "creative.pipeline.stage.completed",
                {
                    "pipeline_id": pipeline_id,
                    "stage_name": stage.name,
                    "status": result.status,
                    "elapsed_ms": result.elapsed_ms,
                    "retry_count": retry_count,
                },
            )

        return result

    def _resolve_stage_order(
        self,
        stages: List[StageConfig],
    ) -> List[StageConfig]:
        """Resolve stage execution order based on dependencies."""
        # Build dependency graph
        stage_map = {s.name: s for s in stages}
        visited = set()
        order = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)

            stage = stage_map.get(name)
            if stage:
                for dep in stage.depends_on:
                    if dep in stage_map:
                        visit(dep)
                order.append(stage)

        for stage in stages:
            visit(stage.name)

        return order

    def _group_stages_by_dependencies(
        self,
        stages: List[StageConfig],
    ) -> List[List[StageConfig]]:
        """Group stages into parallel execution groups."""
        groups: List[List[StageConfig]] = []
        completed: set = set()

        remaining = list(stages)
        while remaining:
            # Find stages whose dependencies are all met
            ready = [
                s
                for s in remaining
                if all(d in completed for d in s.depends_on)
            ]

            if not ready:
                # Circular dependency or missing dependency - add remaining
                ready = remaining[:1]

            groups.append(ready)
            for s in ready:
                completed.add(s.name)
                remaining.remove(s)

        return groups

    def _create_mock_handler(
        self,
        subsystem: str,
        operation: str,
    ) -> StageHandler:
        """Create a mock handler for unregistered operations."""

        async def mock_handler(
            inputs: Dict[str, Any],
            stage_config: StageConfig,
        ) -> Dict[str, Any]:
            """Mock handler that passes through inputs."""
            await asyncio.sleep(0.1)  # Simulate some work
            return {"mock": True, "subsystem": subsystem, "operation": operation}

        return mock_handler

    def _emit_event(self, topic: str, payload: dict) -> None:
        """Emit a bus event."""
        try:
            from nucleus.creative import emit_bus_event
            emit_bus_event(topic, payload)
        except Exception:
            pass  # Silently ignore event emission errors

    def rollback_to_checkpoint(
        self,
        pipeline_id: str,
        checkpoint_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Rollback to a saved checkpoint.

        Args:
            pipeline_id: Pipeline ID
            checkpoint_name: Name of the checkpoint (stage name)

        Returns:
            Checkpoint context if found, None otherwise
        """
        checkpoint_id = f"{pipeline_id}-{checkpoint_name}"
        return self._checkpoints.get(checkpoint_id)

    def get_active_pipelines(self) -> List[PipelineResult]:
        """Get list of currently executing pipelines."""
        return list(self._active_pipelines.values())

    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """
        Request cancellation of a running pipeline.

        Args:
            pipeline_id: Pipeline ID to cancel

        Returns:
            True if pipeline was found and cancellation requested
        """
        if pipeline_id in self._active_pipelines:
            self._active_pipelines[pipeline_id].status = "cancelled"
            return True
        return False
