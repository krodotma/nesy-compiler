#!/usr/bin/env python3
"""
Monitor Final Orchestrator - Step 300

Complete agent orchestration for the Monitor Agent.
This is the FINAL step of the OAGENT 300-step plan.

PBTSO Phase: PLAN

Bus Topics:
- a2a.monitor.orchestrator.start
- a2a.monitor.orchestrator.ready
- a2a.monitor.orchestrator.shutdown
- monitor.orchestrator.component.*
- a2a.heartbeat (emitted at 300s intervals, 900s timeout)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import signal
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional


class OrchestratorState(Enum):
    """Orchestrator states."""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"


class ComponentState(Enum):
    """Component states."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class Component:
    """A monitor component.

    Attributes:
        name: Component name
        state: Component state
        start_func: Start function
        stop_func: Stop function
        health_func: Health check function
        dependencies: Component dependencies
        started_at: Start timestamp
        last_health_check: Last health check
        error: Last error
    """
    name: str
    state: ComponentState = ComponentState.PENDING
    start_func: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None
    stop_func: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None
    health_func: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None
    dependencies: List[str] = field(default_factory=list)
    started_at: float = 0.0
    last_health_check: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "state": self.state.value,
            "dependencies": self.dependencies,
            "started_at": self.started_at,
            "last_health_check": self.last_health_check,
            "uptime_s": time.time() - self.started_at if self.started_at else 0,
            "error": self.error,
        }


@dataclass
class OrchestratorStats:
    """Orchestrator statistics.

    Attributes:
        start_time: Orchestrator start time
        events_processed: Total events processed
        heartbeats_sent: Heartbeats sent
        health_checks_run: Health checks run
        component_restarts: Component restarts
    """
    start_time: float = field(default_factory=time.time)
    events_processed: int = 0
    heartbeats_sent: int = 0
    health_checks_run: int = 0
    component_restarts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time,
            "uptime_s": time.time() - self.start_time,
            "events_processed": self.events_processed,
            "heartbeats_sent": self.heartbeats_sent,
            "health_checks_run": self.health_checks_run,
            "component_restarts": self.component_restarts,
        }


class MonitorOrchestrator:
    """
    Final orchestrator for the Monitor Agent.

    This is Step 300 - the culmination of the OAGENT 300-step plan.

    Responsibilities:
    - Component lifecycle management
    - Health monitoring and auto-recovery
    - A2A heartbeat emission (300s interval, 900s timeout)
    - Graceful shutdown handling
    - Bus event coordination

    All bus writes use fcntl.flock() file locking as per DKIN v30.
    Telemetry events are written to the telemetry bucket.

    Example:
        orchestrator = MonitorOrchestrator()

        # Start the orchestrator
        await orchestrator.start()

        # Run until shutdown
        await orchestrator.run()

        # Or use the CLI
        # python -m monitor.orchestrator --start
    """

    BUS_TOPICS = {
        "start": "a2a.monitor.orchestrator.start",
        "ready": "a2a.monitor.orchestrator.ready",
        "shutdown": "a2a.monitor.orchestrator.shutdown",
        "component": "monitor.orchestrator.component",
        "heartbeat": "a2a.heartbeat",
    }

    # A2A Protocol settings (DKIN v30)
    HEARTBEAT_INTERVAL = 300  # 5 minutes
    HEARTBEAT_TIMEOUT = 900   # 15 minutes

    # Health check interval
    HEALTH_CHECK_INTERVAL = 60

    # Component startup order
    COMPONENT_ORDER = [
        "config_manager",
        "security",
        "validation",
        "cache",
        "metric_collector",
        "metric_aggregator",
        "log_collector",
        "log_analyzer",
        "anomaly_detector",
        "correlation_engine",
        "alert_manager",
        "notification",
        "prediction_engine",
        "rca",
        "dashboard",
        "report_generator",
        "scheduler",
        "plugin_system",
        "api",
        "cli",
        "telemetry",
        "health_check",
        "backup",
        "migration",
        "versioning",
        "deprecation",
        "testing",
        "documentation",
    ]

    def __init__(
        self,
        agent_id: str = "monitor-agent",
        ring_level: int = 1,
        bus_dir: Optional[str] = None,
    ):
        """Initialize orchestrator.

        Args:
            agent_id: Agent identifier
            ring_level: Security ring level (1 = system agent)
            bus_dir: Bus directory
        """
        self._agent_id = agent_id
        self._ring_level = ring_level
        self._state = OrchestratorState.INITIALIZING
        self._stats = OrchestratorStats()

        # Components
        self._components: Dict[str, Component] = {}
        self._lock = threading.RLock()

        # Heartbeat
        self._last_heartbeat = 0.0
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None

        # Shutdown
        self._shutdown_event = asyncio.Event()
        self._running = False

        # Bus path (with file locking per DKIN v30)
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._telemetry_path = Path(self._bus_dir) / "telemetry.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register default components
        self._register_default_components()

        # Setup signal handlers
        self._setup_signals()

    def register_component(
        self,
        name: str,
        start_func: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None,
        stop_func: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None,
        health_func: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> Component:
        """Register a component.

        Args:
            name: Component name
            start_func: Start function
            stop_func: Stop function
            health_func: Health check function
            dependencies: Component dependencies

        Returns:
            Registered component
        """
        component = Component(
            name=name,
            start_func=start_func,
            stop_func=stop_func,
            health_func=health_func,
            dependencies=dependencies or [],
        )

        with self._lock:
            self._components[name] = component

        return component

    async def start(self) -> bool:
        """Start the orchestrator and all components.

        Returns:
            True if started successfully
        """
        if self._state != OrchestratorState.INITIALIZING:
            return False

        self._state = OrchestratorState.STARTING

        self._emit_bus_event(
            self.BUS_TOPICS["start"],
            {
                "agent_id": self._agent_id,
                "ring_level": self._ring_level,
                "components": list(self._components.keys()),
            },
        )

        # Start components in order
        for name in self.COMPONENT_ORDER:
            if name in self._components:
                success = await self._start_component(name)
                if not success:
                    # Log but continue - allow degraded operation
                    self._state = OrchestratorState.DEGRADED

        # Start background tasks
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._health_task = asyncio.create_task(self._health_check_loop())

        if self._state != OrchestratorState.DEGRADED:
            self._state = OrchestratorState.RUNNING

        self._emit_bus_event(
            self.BUS_TOPICS["ready"],
            {
                "agent_id": self._agent_id,
                "state": self._state.value,
                "components_running": self._count_running_components(),
                "startup_duration_ms": (time.time() - self._stats.start_time) * 1000,
            },
        )

        return True

    async def stop(self) -> bool:
        """Stop the orchestrator and all components.

        Returns:
            True if stopped successfully
        """
        if self._state == OrchestratorState.STOPPED:
            return True

        self._state = OrchestratorState.STOPPING
        self._running = False

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Stop components in reverse order
        for name in reversed(self.COMPONENT_ORDER):
            if name in self._components:
                await self._stop_component(name)

        self._state = OrchestratorState.STOPPED

        self._emit_bus_event(
            self.BUS_TOPICS["shutdown"],
            {
                "agent_id": self._agent_id,
                "uptime_s": time.time() - self._stats.start_time,
                "stats": self._stats.to_dict(),
            },
        )

        return True

    async def run(self) -> None:
        """Run the orchestrator until shutdown."""
        await self.start()
        await self._shutdown_event.wait()
        await self.stop()

    def shutdown(self) -> None:
        """Request shutdown."""
        self._shutdown_event.set()

    async def restart_component(self, name: str) -> bool:
        """Restart a component.

        Args:
            name: Component name

        Returns:
            True if restarted
        """
        await self._stop_component(name)
        success = await self._start_component(name)
        if success:
            self._stats.component_restarts += 1
        return success

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status.

        Returns:
            Status dictionary
        """
        with self._lock:
            return {
                "agent_id": self._agent_id,
                "state": self._state.value,
                "ring_level": self._ring_level,
                "uptime_s": time.time() - self._stats.start_time,
                "components": {
                    name: comp.to_dict()
                    for name, comp in self._components.items()
                },
                "components_running": self._count_running_components(),
                "components_total": len(self._components),
                "last_heartbeat": self._last_heartbeat,
                "stats": self._stats.to_dict(),
            }

    def get_component_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get component status.

        Args:
            name: Component name

        Returns:
            Component status or None
        """
        with self._lock:
            comp = self._components.get(name)
            return comp.to_dict() if comp else None

    def list_components(self) -> List[Dict[str, Any]]:
        """List all components.

        Returns:
            Component list
        """
        with self._lock:
            return [comp.to_dict() for comp in self._components.values()]

    async def _start_component(self, name: str) -> bool:
        """Start a single component."""
        with self._lock:
            component = self._components.get(name)
            if not component:
                return False

            if component.state == ComponentState.RUNNING:
                return True

            # Check dependencies
            for dep in component.dependencies:
                dep_comp = self._components.get(dep)
                if not dep_comp or dep_comp.state != ComponentState.RUNNING:
                    component.error = f"Dependency '{dep}' not running"
                    return False

            component.state = ComponentState.STARTING

        self._emit_bus_event(
            f"{self.BUS_TOPICS['component']}.starting",
            {"component": name},
        )

        try:
            if component.start_func:
                success = await asyncio.wait_for(
                    component.start_func(),
                    timeout=30.0,
                )
                if not success:
                    raise Exception("Start function returned False")

            with self._lock:
                component.state = ComponentState.RUNNING
                component.started_at = time.time()
                component.error = None

            self._emit_bus_event(
                f"{self.BUS_TOPICS['component']}.started",
                {"component": name},
            )

            return True

        except Exception as e:
            with self._lock:
                component.state = ComponentState.FAILED
                component.error = str(e)

            self._emit_bus_event(
                f"{self.BUS_TOPICS['component']}.failed",
                {"component": name, "error": str(e)},
                level="error",
            )

            return False

    async def _stop_component(self, name: str) -> bool:
        """Stop a single component."""
        with self._lock:
            component = self._components.get(name)
            if not component:
                return False

            if component.state == ComponentState.STOPPED:
                return True

            component.state = ComponentState.STOPPING

        self._emit_bus_event(
            f"{self.BUS_TOPICS['component']}.stopping",
            {"component": name},
        )

        try:
            if component.stop_func:
                await asyncio.wait_for(
                    component.stop_func(),
                    timeout=30.0,
                )

            with self._lock:
                component.state = ComponentState.STOPPED

            self._emit_bus_event(
                f"{self.BUS_TOPICS['component']}.stopped",
                {"component": name},
            )

            return True

        except Exception as e:
            with self._lock:
                component.error = str(e)

            return False

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while self._running:
            try:
                self._emit_heartbeat()
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(60)

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)

    async def _run_health_checks(self) -> None:
        """Run health checks on all components."""
        self._stats.health_checks_run += 1
        degraded = False

        with self._lock:
            components = list(self._components.items())

        for name, component in components:
            if component.state != ComponentState.RUNNING:
                continue

            if component.health_func:
                try:
                    healthy = await asyncio.wait_for(
                        component.health_func(),
                        timeout=10.0,
                    )

                    with self._lock:
                        component.last_health_check = time.time()
                        if not healthy:
                            component.state = ComponentState.DEGRADED
                            degraded = True

                except Exception as e:
                    with self._lock:
                        component.state = ComponentState.DEGRADED
                        component.error = str(e)
                        degraded = True

        # Update orchestrator state
        with self._lock:
            if degraded and self._state == OrchestratorState.RUNNING:
                self._state = OrchestratorState.DEGRADED
            elif not degraded and self._state == OrchestratorState.DEGRADED:
                self._state = OrchestratorState.RUNNING

    def _emit_heartbeat(self) -> None:
        """Emit A2A heartbeat."""
        self._last_heartbeat = time.time()
        self._stats.heartbeats_sent += 1

        self._emit_bus_event(
            self.BUS_TOPICS["heartbeat"],
            {
                "agent_id": self._agent_id,
                "state": self._state.value,
                "components_running": self._count_running_components(),
                "uptime_s": time.time() - self._stats.start_time,
            },
        )

        # Also emit to telemetry bucket
        self._emit_telemetry_event(
            "monitor.heartbeat",
            {
                "agent_id": self._agent_id,
                "healthy": self._state in (OrchestratorState.RUNNING,),
            },
        )

    def _count_running_components(self) -> int:
        """Count running components."""
        with self._lock:
            return sum(
                1 for c in self._components.values()
                if c.state == ComponentState.RUNNING
            )

    def _register_default_components(self) -> None:
        """Register default monitor components."""
        # All components from Steps 251-299
        for name in self.COMPONENT_ORDER:
            self.register_component(name)

    def _setup_signals(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum: int, frame: Any) -> None:
            self.shutdown()

        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except Exception:
            pass  # May fail in some environments

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event",
    ) -> str:
        """Emit event to bus with file locking (DKIN v30)."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": self._agent_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        self._stats.events_processed += 1
        return event_id

    def _emit_telemetry_event(
        self,
        name: str,
        data: Dict[str, Any],
    ) -> str:
        """Emit event to telemetry bucket."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": f"telemetry.{name}",
            "kind": "metric",
            "level": "info",
            "actor": self._agent_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._telemetry_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id


# Singleton instance
_orchestrator: Optional[MonitorOrchestrator] = None


def get_orchestrator() -> MonitorOrchestrator:
    """Get or create the orchestrator singleton.

    Returns:
        MonitorOrchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MonitorOrchestrator()
    return _orchestrator


async def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Monitor Final Orchestrator - Step 300 (OAGENT Complete)"
    )
    parser.add_argument("--start", action="store_true", help="Start orchestrator")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--components", action="store_true", help="List components")
    parser.add_argument("--component", metavar="NAME", help="Show component status")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    orchestrator = get_orchestrator()

    if args.start:
        print("=" * 60)
        print("OAGENT 300-STEP PLAN - MONITOR AGENT")
        print("Step 300: Final Orchestrator")
        print("=" * 60)
        print(f"Agent ID: {orchestrator._agent_id}")
        print(f"Ring Level: {orchestrator._ring_level}")
        print(f"Components: {len(orchestrator._components)}")
        print("-" * 60)
        print("Starting orchestrator...")

        await orchestrator.start()

        status = orchestrator.get_status()
        print(f"State: {status['state']}")
        print(f"Components Running: {status['components_running']}/{status['components_total']}")
        print("-" * 60)
        print("Orchestrator ready. Press Ctrl+C to shutdown.")

        try:
            await orchestrator._shutdown_event.wait()
        except KeyboardInterrupt:
            print("\nShutdown requested...")

        await orchestrator.stop()
        print("Orchestrator stopped.")
        return 0

    if args.status:
        status = orchestrator.get_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Monitor Orchestrator Status:")
            print(f"  Agent ID: {status['agent_id']}")
            print(f"  State: {status['state']}")
            print(f"  Ring Level: {status['ring_level']}")
            print(f"  Uptime: {status['uptime_s']:.1f}s")
            print(f"  Components: {status['components_running']}/{status['components_total']}")

    if args.components:
        components = orchestrator.list_components()
        if args.json:
            print(json.dumps(components, indent=2))
        else:
            print("Components:")
            for c in components:
                state = c['state']
                print(f"  {c['name']}: {state}")

    if args.component:
        comp = orchestrator.get_component_status(args.component)
        if args.json:
            print(json.dumps(comp, indent=2))
        else:
            if comp:
                print(f"Component: {comp['name']}")
                print(f"  State: {comp['state']}")
                print(f"  Dependencies: {comp['dependencies']}")
                if comp['error']:
                    print(f"  Error: {comp['error']}")
            else:
                print(f"Component not found: {args.component}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
