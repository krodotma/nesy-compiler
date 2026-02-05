#!/usr/bin/env python3
"""
Monitor Health Check - Step 287

Health monitoring for the Monitor Agent.

PBTSO Phase: SKILL

Bus Topics:
- monitor.health.check (emitted)
- monitor.health.status (emitted)
- monitor.health.degraded (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional


class HealthStatus(Enum):
    """Health status values."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """Types of health checks."""
    LIVENESS = "liveness"    # Is the service alive?
    READINESS = "readiness"  # Is the service ready to serve?
    STARTUP = "startup"      # Has the service started successfully?
    CUSTOM = "custom"        # Custom health check


@dataclass
class CheckResult:
    """Result of a health check.

    Attributes:
        name: Check name
        status: Health status
        message: Status message
        duration_ms: Check duration
        timestamp: Check timestamp
        details: Additional details
    """
    name: str
    status: HealthStatus
    message: str = ""
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "details": self.details,
        }


@dataclass
class HealthCheck:
    """A health check definition.

    Attributes:
        name: Check name
        check_type: Type of check
        check_func: Check function
        interval_s: Check interval
        timeout_s: Check timeout
        critical: Whether failure is critical
        enabled: Whether check is enabled
    """
    name: str
    check_type: CheckType
    check_func: Callable[[], Coroutine[Any, Any, CheckResult]]
    interval_s: int = 30
    timeout_s: int = 10
    critical: bool = False
    enabled: bool = True


@dataclass
class DependencyHealth:
    """Health of a dependency.

    Attributes:
        name: Dependency name
        status: Health status
        last_check: Last check timestamp
        consecutive_failures: Number of consecutive failures
        latency_ms: Last check latency
    """
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_check: float = 0.0
    consecutive_failures: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check,
            "consecutive_failures": self.consecutive_failures,
            "latency_ms": self.latency_ms,
        }


class MonitorHealthCheck:
    """
    Health monitoring for the Monitor Agent.

    Provides:
    - Liveness/readiness probes
    - Dependency health tracking
    - Health aggregation
    - Alerting on degradation

    Example:
        health = MonitorHealthCheck()

        # Register a check
        health.register_check(HealthCheck(
            name="database",
            check_type=CheckType.READINESS,
            check_func=check_database,
            critical=True,
        ))

        # Run health check
        result = await health.check_all()

        # Get overall status
        status = health.get_status()
    """

    BUS_TOPICS = {
        "check": "monitor.health.check",
        "status": "monitor.health.status",
        "degraded": "monitor.health.degraded",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        check_interval_s: int = 30,
        failure_threshold: int = 3,
        bus_dir: Optional[str] = None,
    ):
        """Initialize health check.

        Args:
            check_interval_s: Default check interval
            failure_threshold: Failures before unhealthy
            bus_dir: Bus directory
        """
        self._check_interval = check_interval_s
        self._failure_threshold = failure_threshold
        self._last_heartbeat = time.time()
        self._start_time = time.time()

        # Health checks
        self._checks: Dict[str, HealthCheck] = {}
        self._results: Dict[str, CheckResult] = {}
        self._dependencies: Dict[str, DependencyHealth] = {}
        self._lock = threading.RLock()

        # Background task
        self._running = False
        self._check_task: Optional[asyncio.Task] = None

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register default checks
        self._register_default_checks()

    def register_check(self, check: HealthCheck) -> None:
        """Register a health check.

        Args:
            check: Health check definition
        """
        with self._lock:
            self._checks[check.name] = check

    def unregister_check(self, name: str) -> bool:
        """Unregister a health check.

        Args:
            name: Check name

        Returns:
            True if removed
        """
        with self._lock:
            if name in self._checks:
                del self._checks[name]
                return True
            return False

    def register_dependency(self, name: str) -> None:
        """Register a dependency.

        Args:
            name: Dependency name
        """
        with self._lock:
            if name not in self._dependencies:
                self._dependencies[name] = DependencyHealth(name=name)

    def update_dependency(
        self,
        name: str,
        status: HealthStatus,
        latency_ms: float = 0.0,
    ) -> None:
        """Update dependency health.

        Args:
            name: Dependency name
            status: Health status
            latency_ms: Check latency
        """
        with self._lock:
            if name not in self._dependencies:
                self._dependencies[name] = DependencyHealth(name=name)

            dep = self._dependencies[name]
            dep.last_check = time.time()
            dep.latency_ms = latency_ms

            if status == HealthStatus.HEALTHY:
                dep.status = status
                dep.consecutive_failures = 0
            else:
                dep.consecutive_failures += 1
                if dep.consecutive_failures >= self._failure_threshold:
                    dep.status = HealthStatus.UNHEALTHY
                else:
                    dep.status = HealthStatus.DEGRADED

    async def run_check(self, name: str) -> Optional[CheckResult]:
        """Run a specific health check.

        Args:
            name: Check name

        Returns:
            Check result or None
        """
        with self._lock:
            check = self._checks.get(name)
            if not check or not check.enabled:
                return None

        start_time = time.time()
        try:
            # Run with timeout
            result = await asyncio.wait_for(
                check.check_func(),
                timeout=check.timeout_s,
            )
            result.duration_ms = (time.time() - start_time) * 1000
        except asyncio.TimeoutError:
            result = CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Check timed out",
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            result = CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

        with self._lock:
            self._results[name] = result

        # Emit check event
        self._emit_bus_event(
            self.BUS_TOPICS["check"],
            result.to_dict(),
        )

        # Check for degradation
        if result.status != HealthStatus.HEALTHY:
            self._emit_bus_event(
                self.BUS_TOPICS["degraded"],
                {
                    "check": name,
                    "status": result.status.value,
                    "message": result.message,
                },
                level="warning",
            )

        return result

    async def check_all(self) -> Dict[str, CheckResult]:
        """Run all health checks.

        Returns:
            Dictionary of check results
        """
        tasks = []
        check_names = []

        with self._lock:
            for name, check in self._checks.items():
                if check.enabled:
                    tasks.append(self.run_check(name))
                    check_names.append(name)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        result_dict = {}
        for name, result in zip(check_names, results):
            if isinstance(result, CheckResult):
                result_dict[name] = result
            elif isinstance(result, Exception):
                result_dict[name] = CheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(result),
                )

        return result_dict

    async def check_liveness(self) -> CheckResult:
        """Run liveness check.

        Returns:
            Liveness result
        """
        with self._lock:
            for name, check in self._checks.items():
                if check.check_type == CheckType.LIVENESS and check.enabled:
                    result = await self.run_check(name)
                    if result:
                        return result

        # Default liveness - process is alive
        return CheckResult(
            name="liveness",
            status=HealthStatus.HEALTHY,
            message="Process is alive",
        )

    async def check_readiness(self) -> CheckResult:
        """Run readiness check.

        Returns:
            Readiness result
        """
        unhealthy = []

        with self._lock:
            for name, check in self._checks.items():
                if check.check_type == CheckType.READINESS and check.enabled:
                    result = await self.run_check(name)
                    if result and result.status != HealthStatus.HEALTHY:
                        unhealthy.append(name)

        if unhealthy:
            return CheckResult(
                name="readiness",
                status=HealthStatus.UNHEALTHY,
                message=f"Checks failed: {', '.join(unhealthy)}",
            )

        return CheckResult(
            name="readiness",
            status=HealthStatus.HEALTHY,
            message="Service is ready",
        )

    def get_status(self) -> Dict[str, Any]:
        """Get overall health status.

        Returns:
            Health status
        """
        with self._lock:
            # Determine overall status
            all_healthy = True
            critical_unhealthy = False

            for name, check in self._checks.items():
                result = self._results.get(name)
                if result:
                    if result.status != HealthStatus.HEALTHY:
                        all_healthy = False
                        if check.critical:
                            critical_unhealthy = True

            # Check dependencies
            for dep in self._dependencies.values():
                if dep.status != HealthStatus.HEALTHY:
                    all_healthy = False

            if critical_unhealthy:
                overall = HealthStatus.UNHEALTHY
            elif not all_healthy:
                overall = HealthStatus.DEGRADED
            else:
                overall = HealthStatus.HEALTHY

            return {
                "status": overall.value,
                "uptime_s": time.time() - self._start_time,
                "checks": {
                    name: result.to_dict()
                    for name, result in self._results.items()
                },
                "dependencies": {
                    name: dep.to_dict()
                    for name, dep in self._dependencies.items()
                },
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }

    def get_result(self, name: str) -> Optional[CheckResult]:
        """Get a check result.

        Args:
            name: Check name

        Returns:
            Check result or None
        """
        with self._lock:
            return self._results.get(name)

    def list_checks(self) -> List[Dict[str, Any]]:
        """List all registered checks.

        Returns:
            List of check info
        """
        with self._lock:
            return [
                {
                    "name": check.name,
                    "type": check.check_type.value,
                    "interval_s": check.interval_s,
                    "timeout_s": check.timeout_s,
                    "critical": check.critical,
                    "enabled": check.enabled,
                    "last_result": self._results.get(check.name, {}).to_dict()
                    if check.name in self._results else None,
                }
                for check in self._checks.values()
            ]

    async def start_background_checks(self) -> None:
        """Start background health checks."""
        if self._running:
            return

        self._running = True
        self._check_task = asyncio.create_task(self._background_loop())

    async def stop_background_checks(self) -> None:
        """Stop background health checks."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None

    async def _background_loop(self) -> None:
        """Background check loop."""
        while self._running:
            try:
                await self.check_all()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self._check_interval)

    def _register_default_checks(self) -> None:
        """Register default health checks."""

        async def check_memory() -> CheckResult:
            """Check memory usage."""
            try:
                import psutil
                process = psutil.Process()
                memory_percent = process.memory_percent()

                if memory_percent > 90:
                    return CheckResult(
                        name="memory",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Memory usage critical: {memory_percent:.1f}%",
                        details={"memory_percent": memory_percent},
                    )
                elif memory_percent > 75:
                    return CheckResult(
                        name="memory",
                        status=HealthStatus.DEGRADED,
                        message=f"Memory usage high: {memory_percent:.1f}%",
                        details={"memory_percent": memory_percent},
                    )
                else:
                    return CheckResult(
                        name="memory",
                        status=HealthStatus.HEALTHY,
                        message=f"Memory usage normal: {memory_percent:.1f}%",
                        details={"memory_percent": memory_percent},
                    )
            except Exception as e:
                return CheckResult(
                    name="memory",
                    status=HealthStatus.UNKNOWN,
                    message=str(e),
                )

        async def check_disk() -> CheckResult:
            """Check disk usage."""
            try:
                import psutil
                disk = psutil.disk_usage("/")
                disk_percent = disk.percent

                if disk_percent > 95:
                    return CheckResult(
                        name="disk",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Disk usage critical: {disk_percent}%",
                        details={"disk_percent": disk_percent},
                    )
                elif disk_percent > 85:
                    return CheckResult(
                        name="disk",
                        status=HealthStatus.DEGRADED,
                        message=f"Disk usage high: {disk_percent}%",
                        details={"disk_percent": disk_percent},
                    )
                else:
                    return CheckResult(
                        name="disk",
                        status=HealthStatus.HEALTHY,
                        message=f"Disk usage normal: {disk_percent}%",
                        details={"disk_percent": disk_percent},
                    )
            except Exception as e:
                return CheckResult(
                    name="disk",
                    status=HealthStatus.UNKNOWN,
                    message=str(e),
                )

        async def check_bus() -> CheckResult:
            """Check bus connectivity."""
            try:
                if self._bus_path.parent.exists():
                    return CheckResult(
                        name="bus",
                        status=HealthStatus.HEALTHY,
                        message="Bus directory accessible",
                    )
                else:
                    return CheckResult(
                        name="bus",
                        status=HealthStatus.UNHEALTHY,
                        message="Bus directory not accessible",
                    )
            except Exception as e:
                return CheckResult(
                    name="bus",
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                )

        self.register_check(HealthCheck(
            name="memory",
            check_type=CheckType.LIVENESS,
            check_func=check_memory,
            interval_s=60,
            critical=True,
        ))

        self.register_check(HealthCheck(
            name="disk",
            check_type=CheckType.READINESS,
            check_func=check_disk,
            interval_s=300,
            critical=False,
        ))

        self.register_check(HealthCheck(
            name="bus",
            check_type=CheckType.READINESS,
            check_func=check_bus,
            interval_s=30,
            critical=True,
        ))

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now
        status = self.get_status()

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "monitor_health_check",
                "status": status["status"],
                "uptime_s": status["uptime_s"],
            },
        )

        return True

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event",
    ) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-health-check",
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

        return event_id


# Singleton instance
_health_check: Optional[MonitorHealthCheck] = None


def get_health_check() -> MonitorHealthCheck:
    """Get or create the health check singleton.

    Returns:
        MonitorHealthCheck instance
    """
    global _health_check
    if _health_check is None:
        _health_check = MonitorHealthCheck()
    return _health_check


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Health Check (Step 287)")
    parser.add_argument("--status", action="store_true", help="Show health status")
    parser.add_argument("--check", action="store_true", help="Run all checks")
    parser.add_argument("--liveness", action="store_true", help="Run liveness check")
    parser.add_argument("--readiness", action="store_true", help="Run readiness check")
    parser.add_argument("--list", action="store_true", help="List all checks")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    health = get_health_check()

    if args.status:
        status = health.get_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Health Status: {status['status']}")
            print(f"  Uptime: {status['uptime_s']:.1f}s")
            print("  Checks:")
            for name, result in status["checks"].items():
                print(f"    {name}: {result['status']}")

    if args.check:
        async def run_all():
            return await health.check_all()
        results = asyncio.run(run_all())
        if args.json:
            print(json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2))
        else:
            print("Health Checks:")
            for name, result in results.items():
                icon = "V" if result.status == HealthStatus.HEALTHY else "X"
                print(f"  {icon} {name}: {result.status.value} ({result.duration_ms:.1f}ms)")

    if args.liveness:
        async def run_liveness():
            return await health.check_liveness()
        result = asyncio.run(run_liveness())
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            icon = "V" if result.status == HealthStatus.HEALTHY else "X"
            print(f"{icon} Liveness: {result.status.value}")
            print(f"  {result.message}")

    if args.readiness:
        async def run_readiness():
            return await health.check_readiness()
        result = asyncio.run(run_readiness())
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            icon = "V" if result.status == HealthStatus.HEALTHY else "X"
            print(f"{icon} Readiness: {result.status.value}")
            print(f"  {result.message}")

    if args.list:
        checks = health.list_checks()
        if args.json:
            print(json.dumps(checks, indent=2))
        else:
            print("Registered Checks:")
            for check in checks:
                enabled = "enabled" if check["enabled"] else "disabled"
                critical = " (critical)" if check["critical"] else ""
                print(f"  {check['name']}: {check['type']} [{enabled}]{critical}")
