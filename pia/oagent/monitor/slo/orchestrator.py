#!/usr/bin/env python3
"""
Monitor Orchestrator v2 - Step 270

Enhanced monitoring coordination and pipeline orchestration.

PBTSO Phase: PLAN, DISTRIBUTE

Bus Topics:
- a2a.monitor.orchestrate (emitted)
- monitor.pipeline.start (emitted)
- monitor.pipeline.complete (emitted)
- monitor.health.status (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional


class PipelineState(Enum):
    """Pipeline execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class StageType(Enum):
    """Pipeline stage types."""
    COLLECT = "collect"       # Data collection
    TRANSFORM = "transform"   # Data transformation
    ANALYZE = "analyze"       # Analysis
    ALERT = "alert"          # Alert evaluation
    REPORT = "report"        # Report generation
    CUSTOM = "custom"        # Custom stage


@dataclass
class PipelineStage:
    """Pipeline stage definition.

    Attributes:
        stage_id: Unique stage ID
        name: Stage name
        stage_type: Type of stage
        handler: Handler function name
        config: Stage configuration
        timeout_s: Stage timeout
        retry_count: Number of retries
        depends_on: Dependencies (stage IDs)
    """
    stage_id: str
    name: str
    stage_type: StageType
    handler: str
    config: Dict[str, Any] = field(default_factory=dict)
    timeout_s: int = 60
    retry_count: int = 3
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "stage_type": self.stage_type.value,
            "handler": self.handler,
            "config": self.config,
            "timeout_s": self.timeout_s,
            "retry_count": self.retry_count,
            "depends_on": self.depends_on,
        }


@dataclass
class StageResult:
    """Stage execution result.

    Attributes:
        stage_id: Stage ID
        success: Whether stage succeeded
        output: Stage output data
        error: Error message if failed
        duration_ms: Execution duration
        retry_count: Number of retries used
        timestamp: Completion timestamp
    """
    stage_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    retry_count: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_id": self.stage_id,
            "success": self.success,
            "output": self.output if isinstance(self.output, (dict, list, str, int, float, bool, type(None))) else str(self.output),
            "error": self.error,
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count,
            "timestamp": self.timestamp,
        }


@dataclass
class MonitoringPipeline:
    """Monitoring pipeline definition.

    Attributes:
        pipeline_id: Unique pipeline ID
        name: Pipeline name
        description: Pipeline description
        stages: Pipeline stages
        schedule: Cron-like schedule
        enabled: Whether pipeline is enabled
        created_at: Creation timestamp
    """
    pipeline_id: str
    name: str
    description: str = ""
    stages: List[PipelineStage] = field(default_factory=list)
    schedule: str = ""  # cron expression or interval
    enabled: bool = True
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "description": self.description,
            "stages": [s.to_dict() for s in self.stages],
            "schedule": self.schedule,
            "enabled": self.enabled,
            "created_at": self.created_at,
        }

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a stage.

        Args:
            stage: Stage to add
        """
        self.stages.append(stage)

    def get_stage(self, stage_id: str) -> Optional[PipelineStage]:
        """Get stage by ID.

        Args:
            stage_id: Stage ID

        Returns:
            Stage or None
        """
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None


@dataclass
class PipelineExecution:
    """Pipeline execution instance.

    Attributes:
        execution_id: Unique execution ID
        pipeline_id: Pipeline ID
        state: Execution state
        stage_results: Results per stage
        started_at: Start timestamp
        completed_at: Completion timestamp
        error: Error if failed
    """
    execution_id: str
    pipeline_id: str
    state: PipelineState = PipelineState.PENDING
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "pipeline_id": self.pipeline_id,
            "state": self.state.value,
            "stage_results": {k: v.to_dict() for k, v in self.stage_results.items()},
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }

    @property
    def duration_ms(self) -> float:
        """Execution duration in milliseconds."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration.

    Attributes:
        max_concurrent_pipelines: Maximum concurrent pipelines
        max_concurrent_stages: Maximum concurrent stages per pipeline
        default_timeout_s: Default stage timeout
        retry_delay_s: Delay between retries
        health_check_interval_s: Health check interval
    """
    max_concurrent_pipelines: int = 10
    max_concurrent_stages: int = 5
    default_timeout_s: int = 60
    retry_delay_s: int = 5
    health_check_interval_s: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MonitorOrchestrator:
    """
    Orchestrate monitoring pipelines and coordinate components.

    The orchestrator:
    - Manages monitoring pipelines
    - Coordinates stage execution
    - Handles dependencies between stages
    - Provides health monitoring
    - Supports scheduled execution

    Example:
        orchestrator = MonitorOrchestrator()
        await orchestrator.start()

        # Create a pipeline
        pipeline = orchestrator.create_pipeline(
            name="Health Check Pipeline",
            stages=[
                PipelineStage(
                    stage_id="collect",
                    name="Collect Metrics",
                    stage_type=StageType.COLLECT,
                    handler="collect_metrics",
                ),
                PipelineStage(
                    stage_id="analyze",
                    name="Analyze",
                    stage_type=StageType.ANALYZE,
                    handler="analyze_metrics",
                    depends_on=["collect"],
                ),
            ]
        )

        # Execute pipeline
        execution = await orchestrator.execute_pipeline(pipeline.pipeline_id)
    """

    BUS_TOPICS = {
        "orchestrate": "a2a.monitor.orchestrate",
        "pipeline_start": "monitor.pipeline.start",
        "pipeline_complete": "monitor.pipeline.complete",
        "stage_complete": "monitor.stage.complete",
        "health": "monitor.health.status",
    }

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        bus_dir: Optional[str] = None,
    ):
        """Initialize orchestrator.

        Args:
            config: Orchestrator configuration
            bus_dir: Bus directory
        """
        self.config = config or OrchestratorConfig()

        # Pipeline registry
        self._pipelines: Dict[str, MonitoringPipeline] = {}
        self._executions: Dict[str, PipelineExecution] = {}
        self._execution_history: List[PipelineExecution] = []

        # Stage handlers
        self._handlers: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}

        # State
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._execution_semaphore = asyncio.Semaphore(self.config.max_concurrent_pipelines)

        # Component references
        self._components: Dict[str, Any] = {}

        # Register default handlers
        self._register_default_handlers()

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    async def start(self) -> bool:
        """Start the orchestrator.

        Returns:
            True if started
        """
        if self._running:
            return False

        self._running = True

        # Start scheduler
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        # Start health monitoring
        self._health_task = asyncio.create_task(self._health_loop())

        self._emit_bus_event(
            self.BUS_TOPICS["orchestrate"],
            {
                "action": "started",
                "config": self.config.to_dict(),
                "pipelines": len(self._pipelines),
            }
        )

        return True

    async def stop(self) -> bool:
        """Stop the orchestrator.

        Returns:
            True if stopped
        """
        if not self._running:
            return False

        self._running = False

        # Cancel tasks
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        self._emit_bus_event(
            self.BUS_TOPICS["orchestrate"],
            {"action": "stopped"}
        )

        return True

    def create_pipeline(
        self,
        name: str,
        stages: List[PipelineStage],
        description: str = "",
        schedule: str = "",
    ) -> MonitoringPipeline:
        """Create a new pipeline.

        Args:
            name: Pipeline name
            stages: Pipeline stages
            description: Description
            schedule: Schedule expression

        Returns:
            New pipeline
        """
        pipeline_id = str(uuid.uuid4())[:8]

        pipeline = MonitoringPipeline(
            pipeline_id=pipeline_id,
            name=name,
            description=description,
            stages=stages,
            schedule=schedule,
        )

        self._pipelines[pipeline_id] = pipeline

        return pipeline

    def get_pipeline(self, pipeline_id: str) -> Optional[MonitoringPipeline]:
        """Get pipeline by ID.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Pipeline or None
        """
        return self._pipelines.get(pipeline_id)

    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipelines.

        Returns:
            Pipeline summaries
        """
        return [
            {
                "pipeline_id": p.pipeline_id,
                "name": p.name,
                "stages": len(p.stages),
                "schedule": p.schedule,
                "enabled": p.enabled,
            }
            for p in self._pipelines.values()
        ]

    def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            True if deleted
        """
        if pipeline_id not in self._pipelines:
            return False

        del self._pipelines[pipeline_id]
        return True

    async def execute_pipeline(
        self,
        pipeline_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PipelineExecution:
        """Execute a pipeline.

        Args:
            pipeline_id: Pipeline ID
            context: Execution context

        Returns:
            Pipeline execution
        """
        pipeline = self.get_pipeline(pipeline_id)
        if not pipeline:
            raise ValueError(f"Pipeline not found: {pipeline_id}")

        execution_id = str(uuid.uuid4())[:8]
        execution = PipelineExecution(
            execution_id=execution_id,
            pipeline_id=pipeline_id,
        )

        self._executions[execution_id] = execution

        # Execute with semaphore
        async with self._execution_semaphore:
            await self._run_pipeline(pipeline, execution, context or {})

        # Store in history
        self._execution_history.append(execution)
        if len(self._execution_history) > 100:
            self._execution_history = self._execution_history[-100:]

        return execution

    def get_execution(self, execution_id: str) -> Optional[PipelineExecution]:
        """Get execution by ID.

        Args:
            execution_id: Execution ID

        Returns:
            Execution or None
        """
        return self._executions.get(execution_id)

    def get_execution_history(
        self,
        pipeline_id: Optional[str] = None,
        limit: int = 10
    ) -> List[PipelineExecution]:
        """Get execution history.

        Args:
            pipeline_id: Filter by pipeline
            limit: Maximum results

        Returns:
            Execution history
        """
        history = self._execution_history
        if pipeline_id:
            history = [e for e in history if e.pipeline_id == pipeline_id]

        return list(reversed(history[-limit:]))

    def register_handler(
        self,
        name: str,
        handler: Callable[..., Coroutine[Any, Any, Any]]
    ) -> None:
        """Register a stage handler.

        Args:
            name: Handler name
            handler: Async handler function
        """
        self._handlers[name] = handler

    def register_component(self, name: str, component: Any) -> None:
        """Register a monitoring component.

        Args:
            name: Component name
            component: Component instance
        """
        self._components[name] = component

    def get_component(self, name: str) -> Optional[Any]:
        """Get a registered component.

        Args:
            name: Component name

        Returns:
            Component or None
        """
        return self._components.get(name)

    def create_standard_pipeline(self) -> MonitoringPipeline:
        """Create the standard monitoring pipeline.

        Returns:
            Standard pipeline
        """
        stages = [
            PipelineStage(
                stage_id="collect_resources",
                name="Collect Resource Metrics",
                stage_type=StageType.COLLECT,
                handler="collect_resources",
            ),
            PipelineStage(
                stage_id="collect_services",
                name="Collect Service Health",
                stage_type=StageType.COLLECT,
                handler="collect_services",
            ),
            PipelineStage(
                stage_id="analyze_trends",
                name="Analyze Trends",
                stage_type=StageType.ANALYZE,
                handler="analyze_trends",
                depends_on=["collect_resources"],
            ),
            PipelineStage(
                stage_id="check_slos",
                name="Check SLO Compliance",
                stage_type=StageType.ANALYZE,
                handler="check_slos",
                depends_on=["collect_services"],
            ),
            PipelineStage(
                stage_id="evaluate_alerts",
                name="Evaluate Alerts",
                stage_type=StageType.ALERT,
                handler="evaluate_alerts",
                depends_on=["analyze_trends", "check_slos"],
            ),
            PipelineStage(
                stage_id="update_dashboard",
                name="Update Dashboard",
                stage_type=StageType.REPORT,
                handler="update_dashboard",
                depends_on=["evaluate_alerts"],
            ),
        ]

        return self.create_pipeline(
            name="Standard Monitoring Pipeline",
            description="Collects metrics, analyzes trends, checks SLOs, and updates dashboards",
            stages=stages,
            schedule="*/5 * * * *",  # Every 5 minutes
        )

    def get_health_status(self) -> Dict[str, Any]:
        """Get orchestrator health status.

        Returns:
            Health status
        """
        active_executions = sum(
            1 for e in self._executions.values()
            if e.state == PipelineState.RUNNING
        )

        recent_failures = sum(
            1 for e in self._execution_history[-20:]
            if e.state == PipelineState.FAILED
        )

        return {
            "running": self._running,
            "pipelines": len(self._pipelines),
            "active_executions": active_executions,
            "handlers": len(self._handlers),
            "components": list(self._components.keys()),
            "recent_failures": recent_failures,
            "health": "healthy" if recent_failures < 5 else "degraded",
        }

    async def _run_pipeline(
        self,
        pipeline: MonitoringPipeline,
        execution: PipelineExecution,
        context: Dict[str, Any]
    ) -> None:
        """Run a pipeline execution.

        Args:
            pipeline: Pipeline to run
            execution: Execution instance
            context: Execution context
        """
        execution.state = PipelineState.RUNNING
        execution.started_at = time.time()

        self._emit_bus_event(
            self.BUS_TOPICS["pipeline_start"],
            {
                "execution_id": execution.execution_id,
                "pipeline_id": pipeline.pipeline_id,
                "pipeline_name": pipeline.name,
            }
        )

        try:
            # Build dependency graph
            completed: set = set()
            stage_outputs: Dict[str, Any] = {}

            # Execute stages respecting dependencies
            while len(completed) < len(pipeline.stages):
                # Find stages ready to execute
                ready = []
                for stage in pipeline.stages:
                    if stage.stage_id in completed:
                        continue
                    if all(dep in completed for dep in stage.depends_on):
                        ready.append(stage)

                if not ready:
                    # No progress possible - cycle or error
                    raise RuntimeError("Pipeline deadlock - check dependencies")

                # Execute ready stages concurrently (up to limit)
                semaphore = asyncio.Semaphore(self.config.max_concurrent_stages)

                async def run_with_semaphore(stage: PipelineStage):
                    async with semaphore:
                        return await self._run_stage(stage, context, stage_outputs)

                results = await asyncio.gather(
                    *[run_with_semaphore(stage) for stage in ready],
                    return_exceptions=True
                )

                # Process results
                for stage, result in zip(ready, results):
                    if isinstance(result, Exception):
                        stage_result = StageResult(
                            stage_id=stage.stage_id,
                            success=False,
                            error=str(result),
                        )
                    else:
                        stage_result = result
                        stage_outputs[stage.stage_id] = result.output

                    execution.stage_results[stage.stage_id] = stage_result
                    completed.add(stage.stage_id)

                    self._emit_bus_event(
                        self.BUS_TOPICS["stage_complete"],
                        {
                            "execution_id": execution.execution_id,
                            "stage_id": stage.stage_id,
                            "success": stage_result.success,
                        }
                    )

                    if not stage_result.success:
                        execution.state = PipelineState.FAILED
                        execution.error = f"Stage {stage.stage_id} failed: {stage_result.error}"
                        break

                if execution.state == PipelineState.FAILED:
                    break

            if execution.state == PipelineState.RUNNING:
                execution.state = PipelineState.COMPLETED

        except Exception as e:
            execution.state = PipelineState.FAILED
            execution.error = str(e)

        finally:
            execution.completed_at = time.time()

            self._emit_bus_event(
                self.BUS_TOPICS["pipeline_complete"],
                {
                    "execution_id": execution.execution_id,
                    "pipeline_id": pipeline.pipeline_id,
                    "state": execution.state.value,
                    "duration_ms": execution.duration_ms,
                    "error": execution.error,
                }
            )

    async def _run_stage(
        self,
        stage: PipelineStage,
        context: Dict[str, Any],
        stage_outputs: Dict[str, Any]
    ) -> StageResult:
        """Run a single stage.

        Args:
            stage: Stage to run
            context: Execution context
            stage_outputs: Previous stage outputs

        Returns:
            Stage result
        """
        handler = self._handlers.get(stage.handler)
        if not handler:
            return StageResult(
                stage_id=stage.stage_id,
                success=False,
                error=f"Handler not found: {stage.handler}",
            )

        start_time = time.time()
        retry = 0

        while retry <= stage.retry_count:
            try:
                # Build handler context
                handler_context = {
                    **context,
                    "config": stage.config,
                    "dependencies": {
                        dep: stage_outputs.get(dep)
                        for dep in stage.depends_on
                    },
                    "components": self._components,
                }

                # Execute with timeout
                output = await asyncio.wait_for(
                    handler(handler_context),
                    timeout=stage.timeout_s
                )

                return StageResult(
                    stage_id=stage.stage_id,
                    success=True,
                    output=output,
                    duration_ms=(time.time() - start_time) * 1000,
                    retry_count=retry,
                )

            except asyncio.TimeoutError:
                error = f"Stage timed out after {stage.timeout_s}s"
            except Exception as e:
                error = str(e)

            retry += 1
            if retry <= stage.retry_count:
                await asyncio.sleep(self.config.retry_delay_s)

        return StageResult(
            stage_id=stage.stage_id,
            success=False,
            error=error,
            duration_ms=(time.time() - start_time) * 1000,
            retry_count=retry - 1,
        )

    async def _scheduler_loop(self) -> None:
        """Scheduler loop for periodic pipelines."""
        while self._running:
            try:
                # Simple interval-based scheduling
                for pipeline in self._pipelines.values():
                    if not pipeline.enabled or not pipeline.schedule:
                        continue

                    # Parse simple schedules like "*/5" (every 5 minutes)
                    if pipeline.schedule.startswith("*/"):
                        try:
                            interval_minutes = int(pipeline.schedule[2:].split()[0])
                            current_minute = datetime.now().minute
                            if current_minute % interval_minutes == 0:
                                asyncio.create_task(
                                    self.execute_pipeline(pipeline.pipeline_id)
                                )
                        except (ValueError, IndexError):
                            pass

            except Exception as e:
                self._emit_bus_event(
                    "monitor.scheduler.error",
                    {"error": str(e)},
                    level="error"
                )

            await asyncio.sleep(60)  # Check every minute

    async def _health_loop(self) -> None:
        """Health monitoring loop."""
        while self._running:
            try:
                status = self.get_health_status()
                self._emit_bus_event(
                    self.BUS_TOPICS["health"],
                    status
                )
            except Exception:
                pass

            await asyncio.sleep(self.config.health_check_interval_s)

    def _register_default_handlers(self) -> None:
        """Register default stage handlers."""

        async def collect_resources(context: Dict[str, Any]) -> Dict[str, Any]:
            """Collect resource metrics."""
            resource_monitor = context.get("components", {}).get("resource_monitor")
            if resource_monitor:
                metrics = resource_monitor.collect_metrics()
                return metrics.to_dict()
            return {"status": "no resource monitor"}

        async def collect_services(context: Dict[str, Any]) -> Dict[str, Any]:
            """Collect service health."""
            service_monitor = context.get("components", {}).get("service_monitor")
            if service_monitor:
                statuses = service_monitor.get_all_status()
                return {name: s.to_dict() for name, s in statuses.items()}
            return {"status": "no service monitor"}

        async def analyze_trends(context: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze trends."""
            trend_analyzer = context.get("components", {}).get("trend_analyzer")
            if trend_analyzer:
                trends = trend_analyzer.analyze_all(period_days=7)
                return {name: t.to_dict() for name, t in trends.items()}
            return {"status": "no trend analyzer"}

        async def check_slos(context: Dict[str, Any]) -> Dict[str, Any]:
            """Check SLO compliance."""
            slo_tracker = context.get("components", {}).get("slo_tracker")
            if slo_tracker:
                compliance = slo_tracker.get_all_compliance()
                return {name: c.to_dict() for name, c in compliance.items()}
            return {"status": "no slo tracker"}

        async def evaluate_alerts(context: Dict[str, Any]) -> Dict[str, Any]:
            """Evaluate alerts."""
            deps = context.get("dependencies", {})
            alerts = []

            # Check for issues in dependencies
            trends = deps.get("analyze_trends", {})
            slos = deps.get("check_slos", {})

            for name, trend in trends.items():
                if isinstance(trend, dict) and trend.get("significance") == "strong":
                    alerts.append({
                        "type": "trend",
                        "metric": name,
                        "message": f"Strong trend detected: {trend.get('change_percent', 0):.1f}%"
                    })

            for name, slo in slos.items():
                if isinstance(slo, dict) and slo.get("state") == "breached":
                    alerts.append({
                        "type": "slo_breach",
                        "slo": name,
                        "message": f"SLO breach: {name}"
                    })

            return {"alerts": alerts, "count": len(alerts)}

        async def update_dashboard(context: Dict[str, Any]) -> Dict[str, Any]:
            """Update dashboard data."""
            dashboard_builder = context.get("components", {}).get("dashboard_builder")
            if dashboard_builder:
                dashboards = dashboard_builder.list_dashboards()
                return {"dashboards_updated": len(dashboards)}
            return {"status": "no dashboard builder"}

        self.register_handler("collect_resources", collect_resources)
        self.register_handler("collect_services", collect_services)
        self.register_handler("analyze_trends", analyze_trends)
        self.register_handler("check_slos", check_slos)
        self.register_handler("evaluate_alerts", evaluate_alerts)
        self.register_handler("update_dashboard", update_dashboard)

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to bus."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

        return event_id


# Singleton
_orchestrator: Optional[MonitorOrchestrator] = None


def get_orchestrator() -> MonitorOrchestrator:
    """Get or create the orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MonitorOrchestrator()
    return _orchestrator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Orchestrator v2 (Step 270)")
    parser.add_argument("--status", action="store_true", help="Show orchestrator status")
    parser.add_argument("--pipelines", action="store_true", help="List pipelines")
    parser.add_argument("--create-standard", action="store_true", help="Create standard pipeline")
    parser.add_argument("--execute", metavar="ID", help="Execute a pipeline")
    parser.add_argument("--history", metavar="ID", help="Show execution history for pipeline")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    orchestrator = get_orchestrator()

    if args.status:
        status = orchestrator.get_health_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("Monitor Orchestrator Status:")
            print(f"  Running: {status['running']}")
            print(f"  Pipelines: {status['pipelines']}")
            print(f"  Active Executions: {status['active_executions']}")
            print(f"  Handlers: {status['handlers']}")
            print(f"  Components: {', '.join(status['components']) or 'none'}")
            print(f"  Health: {status['health']}")

    if args.pipelines:
        pipelines = orchestrator.list_pipelines()
        if args.json:
            print(json.dumps(pipelines, indent=2))
        else:
            print("Pipelines:")
            for p in pipelines:
                enabled = "enabled" if p["enabled"] else "disabled"
                print(f"  [{p['pipeline_id']}] {p['name']} ({p['stages']} stages, {enabled})")

    if args.create_standard:
        pipeline = orchestrator.create_standard_pipeline()
        print(f"Created standard pipeline: {pipeline.pipeline_id}")
        if args.json:
            print(json.dumps(pipeline.to_dict(), indent=2))

    if args.execute:
        async def run():
            await orchestrator.start()
            execution = await orchestrator.execute_pipeline(args.execute)
            await orchestrator.stop()
            return execution

        execution = asyncio.run(run())
        if args.json:
            print(json.dumps(execution.to_dict(), indent=2))
        else:
            print(f"Execution: {execution.execution_id}")
            print(f"  State: {execution.state.value}")
            print(f"  Duration: {execution.duration_ms:.1f}ms")
            if execution.error:
                print(f"  Error: {execution.error}")
            print(f"  Stages:")
            for stage_id, result in execution.stage_results.items():
                status = "OK" if result.success else "FAIL"
                print(f"    [{status}] {stage_id}: {result.duration_ms:.1f}ms")

    if args.history:
        history = orchestrator.get_execution_history(args.history)
        if args.json:
            print(json.dumps([e.to_dict() for e in history], indent=2))
        else:
            print(f"Execution History for {args.history}:")
            for e in history:
                print(f"  [{e.execution_id}] {e.state.value} ({e.duration_ms:.1f}ms)")
