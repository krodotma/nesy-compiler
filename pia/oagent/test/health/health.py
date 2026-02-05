#!/usr/bin/env python3
"""
Step 137: Test Health Check

Health monitoring for the Test Agent.

PBTSO Phase: OBSERVE, VERIFY
Bus Topics:
- test.health.check (emits)
- test.health.status (emits)
- test.health.alert (emits)

Dependencies: Steps 101-136 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import psutil
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# Constants
# ============================================================================

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Component types for health checks."""
    SYSTEM = "system"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    API = "api"
    WORKER = "worker"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class HealthProbe:
    """
    A health probe definition.

    Attributes:
        name: Probe name
        component_type: Type of component
        check_fn: Function to execute check
        interval_s: Check interval
        timeout_s: Check timeout
        failure_threshold: Failures before unhealthy
        success_threshold: Successes before healthy
        enabled: Whether probe is enabled
    """
    name: str
    component_type: ComponentType
    check_fn: Callable[[], tuple]
    interval_s: int = 30
    timeout_s: int = 5
    failure_threshold: int = 3
    success_threshold: int = 1
    enabled: bool = True
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check: Optional[float] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "component_type": self.component_type.value,
            "interval_s": self.interval_s,
            "timeout_s": self.timeout_s,
            "enabled": self.enabled,
            "last_check": self.last_check,
            "last_status": self.last_status.value,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class HealthComponent:
    """
    Health status of a component.

    Attributes:
        name: Component name
        component_type: Type of component
        status: Health status
        message: Status message
        details: Additional details
        last_check: Last check timestamp
        duration_ms: Check duration
    """
    name: str
    component_type: ComponentType
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: float = field(default_factory=time.time)
    duration_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "component_type": self.component_type.value,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "last_check": self.last_check,
            "duration_ms": self.duration_ms,
        }


@dataclass
class HealthReport:
    """
    Overall health report.

    Attributes:
        status: Overall health status
        timestamp: Report timestamp
        components: Component health statuses
        uptime_s: System uptime
        version: Agent version
    """
    status: HealthStatus = HealthStatus.UNKNOWN
    timestamp: float = field(default_factory=time.time)
    components: List[HealthComponent] = field(default_factory=list)
    uptime_s: float = 0
    version: str = "1.0.0"
    agent_id: str = "test-agent"

    @property
    def healthy_components(self) -> int:
        return sum(1 for c in self.components if c.status == HealthStatus.HEALTHY)

    @property
    def unhealthy_components(self) -> int:
        return sum(1 for c in self.components if c.status == HealthStatus.UNHEALTHY)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "uptime_s": self.uptime_s,
            "version": self.version,
            "agent_id": self.agent_id,
            "healthy_components": self.healthy_components,
            "unhealthy_components": self.unhealthy_components,
            "components": [c.to_dict() for c in self.components],
        }


@dataclass
class HealthConfig:
    """
    Configuration for health checks.

    Attributes:
        output_dir: Output directory
        check_interval_s: Default check interval
        enable_system_checks: Enable system checks
        memory_threshold_percent: Memory warning threshold
        disk_threshold_percent: Disk warning threshold
        enable_liveness: Enable liveness probe
        enable_readiness: Enable readiness probe
    """
    output_dir: str = ".pluribus/test-agent/health"
    check_interval_s: int = 30
    enable_system_checks: bool = True
    memory_threshold_percent: float = 90.0
    disk_threshold_percent: float = 90.0
    cpu_threshold_percent: float = 95.0
    enable_liveness: bool = True
    enable_readiness: bool = True
    unhealthy_notification: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_interval_s": self.check_interval_s,
            "enable_system_checks": self.enable_system_checks,
            "memory_threshold_percent": self.memory_threshold_percent,
            "disk_threshold_percent": self.disk_threshold_percent,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class HealthBus:
    """Bus interface for health with file locking."""

    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def heartbeat(self, agent_id: str) -> None:
        """Send A2A heartbeat."""
        now = time.time()
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.emit({
                "topic": "a2a.heartbeat",
                "kind": "heartbeat",
                "actor": agent_id,
                "data": {"status": "alive"},
            })
            self._last_heartbeat = now


# ============================================================================
# Test Health Check
# ============================================================================

class TestHealthCheck:
    """
    Health monitoring for the Test Agent.

    Features:
    - System resource monitoring
    - Component health checks
    - Liveness and readiness probes
    - Health reporting
    - Alert on unhealthy status

    PBTSO Phase: OBSERVE, VERIFY
    Bus Topics: test.health.check, test.health.status, test.health.alert
    """

    BUS_TOPICS = {
        "check": "test.health.check",
        "status": "test.health.status",
        "alert": "test.health.alert",
    }

    def __init__(self, bus=None, config: Optional[HealthConfig] = None):
        """
        Initialize the health checker.

        Args:
            bus: Optional bus instance
            config: Health configuration
        """
        self.bus = bus or HealthBus()
        self.config = config or HealthConfig()
        self._probes: Dict[str, HealthProbe] = {}
        self._components: Dict[str, HealthComponent] = {}
        self._start_time = time.time()
        self._last_report: Optional[HealthReport] = None
        self._running = False

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Register default probes
        self._register_default_probes()

    def _register_default_probes(self) -> None:
        """Register default health probes."""
        if self.config.enable_system_checks:
            # Memory probe
            self.register_probe(HealthProbe(
                name="memory",
                component_type=ComponentType.MEMORY,
                check_fn=self._check_memory,
                interval_s=30,
            ))

            # Disk probe
            self.register_probe(HealthProbe(
                name="disk",
                component_type=ComponentType.DISK,
                check_fn=self._check_disk,
                interval_s=60,
            ))

            # CPU probe
            self.register_probe(HealthProbe(
                name="cpu",
                component_type=ComponentType.SYSTEM,
                check_fn=self._check_cpu,
                interval_s=30,
            ))

    def register_probe(self, probe: HealthProbe) -> None:
        """Register a health probe."""
        self._probes[probe.name] = probe

    def unregister_probe(self, name: str) -> bool:
        """Unregister a health probe."""
        if name in self._probes:
            del self._probes[name]
            return True
        return False

    def check(self, probe_name: Optional[str] = None) -> HealthReport:
        """
        Run health checks.

        Args:
            probe_name: Specific probe to check, or all if None

        Returns:
            HealthReport with check results
        """
        report = HealthReport(
            uptime_s=time.time() - self._start_time,
        )

        probes = [self._probes[probe_name]] if probe_name else list(self._probes.values())

        for probe in probes:
            if not probe.enabled:
                continue

            component = self._run_probe(probe)
            report.components.append(component)
            self._components[probe.name] = component

        # Determine overall status
        report.status = self._calculate_overall_status(report.components)

        # Emit status event
        self._emit_event("status", {
            "status": report.status.value,
            "components": len(report.components),
            "healthy": report.healthy_components,
            "unhealthy": report.unhealthy_components,
        })

        # Alert if unhealthy
        if report.status == HealthStatus.UNHEALTHY and self.config.unhealthy_notification:
            self._emit_alert(report)

        self._last_report = report
        self._save_report(report)

        return report

    def _run_probe(self, probe: HealthProbe) -> HealthComponent:
        """Run a single health probe."""
        start_time = time.time()

        try:
            status, message, details = probe.check_fn()
            duration_ms = (time.time() - start_time) * 1000

            # Update consecutive counts
            if status == HealthStatus.HEALTHY:
                probe.consecutive_successes += 1
                probe.consecutive_failures = 0
            else:
                probe.consecutive_failures += 1
                probe.consecutive_successes = 0

            # Apply thresholds
            if probe.consecutive_failures >= probe.failure_threshold:
                status = HealthStatus.UNHEALTHY
            elif probe.consecutive_successes < probe.success_threshold:
                status = HealthStatus.DEGRADED

            probe.last_check = time.time()
            probe.last_status = status

            self._emit_event("check", {
                "probe": probe.name,
                "status": status.value,
                "duration_ms": duration_ms,
            })

            return HealthComponent(
                name=probe.name,
                component_type=probe.component_type,
                status=status,
                message=message,
                details=details,
                duration_ms=duration_ms,
            )

        except Exception as e:
            probe.consecutive_failures += 1
            probe.last_status = HealthStatus.UNHEALTHY

            return HealthComponent(
                name=probe.name,
                component_type=probe.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000,
            )

    def _calculate_overall_status(self, components: List[HealthComponent]) -> HealthStatus:
        """Calculate overall health status from components."""
        if not components:
            return HealthStatus.UNKNOWN

        unhealthy = sum(1 for c in components if c.status == HealthStatus.UNHEALTHY)
        degraded = sum(1 for c in components if c.status == HealthStatus.DEGRADED)

        if unhealthy > 0:
            return HealthStatus.UNHEALTHY
        elif degraded > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _check_memory(self) -> tuple:
        """Check memory health."""
        try:
            memory = psutil.virtual_memory()
            used_percent = memory.percent

            if used_percent >= self.config.memory_threshold_percent:
                return (
                    HealthStatus.UNHEALTHY,
                    f"Memory usage critical: {used_percent:.1f}%",
                    {"used_percent": used_percent, "available_mb": memory.available / 1024 / 1024},
                )
            elif used_percent >= self.config.memory_threshold_percent * 0.8:
                return (
                    HealthStatus.DEGRADED,
                    f"Memory usage elevated: {used_percent:.1f}%",
                    {"used_percent": used_percent, "available_mb": memory.available / 1024 / 1024},
                )
            else:
                return (
                    HealthStatus.HEALTHY,
                    f"Memory usage normal: {used_percent:.1f}%",
                    {"used_percent": used_percent, "available_mb": memory.available / 1024 / 1024},
                )
        except Exception as e:
            return (HealthStatus.UNKNOWN, f"Unable to check memory: {e}", {})

    def _check_disk(self) -> tuple:
        """Check disk health."""
        try:
            disk = shutil.disk_usage("/")
            used_percent = (disk.used / disk.total) * 100

            if used_percent >= self.config.disk_threshold_percent:
                return (
                    HealthStatus.UNHEALTHY,
                    f"Disk usage critical: {used_percent:.1f}%",
                    {"used_percent": used_percent, "free_gb": disk.free / 1024 / 1024 / 1024},
                )
            elif used_percent >= self.config.disk_threshold_percent * 0.8:
                return (
                    HealthStatus.DEGRADED,
                    f"Disk usage elevated: {used_percent:.1f}%",
                    {"used_percent": used_percent, "free_gb": disk.free / 1024 / 1024 / 1024},
                )
            else:
                return (
                    HealthStatus.HEALTHY,
                    f"Disk usage normal: {used_percent:.1f}%",
                    {"used_percent": used_percent, "free_gb": disk.free / 1024 / 1024 / 1024},
                )
        except Exception as e:
            return (HealthStatus.UNKNOWN, f"Unable to check disk: {e}", {})

    def _check_cpu(self) -> tuple:
        """Check CPU health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)

            if cpu_percent >= self.config.cpu_threshold_percent:
                return (
                    HealthStatus.UNHEALTHY,
                    f"CPU usage critical: {cpu_percent:.1f}%",
                    {"used_percent": cpu_percent},
                )
            elif cpu_percent >= self.config.cpu_threshold_percent * 0.8:
                return (
                    HealthStatus.DEGRADED,
                    f"CPU usage elevated: {cpu_percent:.1f}%",
                    {"used_percent": cpu_percent},
                )
            else:
                return (
                    HealthStatus.HEALTHY,
                    f"CPU usage normal: {cpu_percent:.1f}%",
                    {"used_percent": cpu_percent},
                )
        except Exception as e:
            return (HealthStatus.UNKNOWN, f"Unable to check CPU: {e}", {})

    def liveness(self) -> tuple:
        """
        Liveness probe - is the agent alive?

        Returns:
            (is_alive, message)
        """
        return True, "Agent is alive"

    def readiness(self) -> tuple:
        """
        Readiness probe - is the agent ready to serve?

        Returns:
            (is_ready, message)
        """
        if self._last_report is None:
            return False, "No health check performed yet"

        if self._last_report.status == HealthStatus.UNHEALTHY:
            return False, "Agent is unhealthy"

        return True, "Agent is ready"

    def get_status(self) -> HealthReport:
        """Get current health status."""
        if self._last_report and time.time() - self._last_report.timestamp < self.config.check_interval_s:
            return self._last_report
        return self.check()

    def get_component(self, name: str) -> Optional[HealthComponent]:
        """Get health status of a specific component."""
        return self._components.get(name)

    def _emit_alert(self, report: HealthReport) -> None:
        """Emit health alert."""
        unhealthy = [c for c in report.components if c.status == HealthStatus.UNHEALTHY]

        self._emit_event("alert", {
            "status": report.status.value,
            "unhealthy_components": [c.name for c in unhealthy],
            "message": "; ".join(c.message for c in unhealthy),
        })

    def _save_report(self, report: HealthReport) -> None:
        """Save health report."""
        report_file = Path(self.config.output_dir) / "health_status.json"

        with open(report_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(report.to_dict(), f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    async def check_async(self, probe_name: Optional[str] = None) -> HealthReport:
        """Async version of check."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.check, probe_name)

    async def start_periodic_checks(self, interval_s: Optional[int] = None) -> None:
        """Start periodic health checks."""
        interval = interval_s or self.config.check_interval_s
        self._running = True

        while self._running:
            await self.check_async()
            await asyncio.sleep(interval)

    def stop_periodic_checks(self) -> None:
        """Stop periodic health checks."""
        self._running = False

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.health.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "health",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Health Check."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Health Check")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Check command
    check_parser = subparsers.add_parser("check", help="Run health check")
    check_parser.add_argument("--probe", help="Specific probe to check")

    # Status command
    status_parser = subparsers.add_parser("status", help="Get current health status")

    # Liveness command
    liveness_parser = subparsers.add_parser("liveness", help="Liveness probe")

    # Readiness command
    readiness_parser = subparsers.add_parser("readiness", help="Readiness probe")

    # Probes command
    probes_parser = subparsers.add_parser("probes", help="List registered probes")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/health")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = HealthConfig(output_dir=args.output)
    health = TestHealthCheck(config=config)

    if args.command == "check":
        report = health.check(args.probe)

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            status_icons = {
                HealthStatus.HEALTHY: "[OK]",
                HealthStatus.DEGRADED: "[WARN]",
                HealthStatus.UNHEALTHY: "[FAIL]",
                HealthStatus.UNKNOWN: "[?]",
            }

            print(f"\nHealth Status: {status_icons[report.status]} {report.status.value}")
            print(f"Uptime: {report.uptime_s:.0f}s")
            print(f"\nComponents ({len(report.components)}):")
            for comp in report.components:
                icon = status_icons[comp.status]
                print(f"  {icon} {comp.name}: {comp.message}")

    elif args.command == "status":
        report = health.get_status()

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Status: {report.status.value}")
            print(f"Healthy: {report.healthy_components}/{len(report.components)}")

    elif args.command == "liveness":
        alive, message = health.liveness()
        if args.json:
            print(json.dumps({"alive": alive, "message": message}))
        else:
            print(f"Alive: {alive}")
            print(f"Message: {message}")
        exit(0 if alive else 1)

    elif args.command == "readiness":
        ready, message = health.readiness()
        if args.json:
            print(json.dumps({"ready": ready, "message": message}))
        else:
            print(f"Ready: {ready}")
            print(f"Message: {message}")
        exit(0 if ready else 1)

    elif args.command == "probes":
        probes = [p.to_dict() for p in health._probes.values()]

        if args.json:
            print(json.dumps(probes, indent=2))
        else:
            print(f"\nRegistered Probes ({len(probes)}):")
            for probe in health._probes.values():
                enabled = "[ON]" if probe.enabled else "[OFF]"
                print(f"  {enabled} {probe.name} ({probe.component_type.value})")
                print(f"      Interval: {probe.interval_s}s, Status: {probe.last_status.value}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
