#!/usr/bin/env python3
"""
health_check.py - Health Monitoring (Step 87)

PBTSO Phase: VERIFY, TEST

Provides:
- Component health checks
- Dependency monitoring
- Resource usage tracking
- Heartbeat management
- Health reporting

Bus Topics:
- code.health.check
- code.health.status
- code.health.heartbeat
- code.health.alert

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import json
import os
import psutil
import socket
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckConfig:
    """Configuration for health checking."""
    check_interval_s: int = 30
    timeout_s: float = 10.0
    failure_threshold: int = 3
    success_threshold: int = 1
    enable_heartbeat: bool = True
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900
    enable_resource_monitoring: bool = True
    memory_threshold_percent: float = 90.0
    disk_threshold_percent: float = 90.0
    cpu_threshold_percent: float = 90.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_interval_s": self.check_interval_s,
            "timeout_s": self.timeout_s,
            "failure_threshold": self.failure_threshold,
            "heartbeat_interval_s": self.heartbeat_interval_s,
            "heartbeat_timeout_s": self.heartbeat_timeout_s,
        }


# =============================================================================
# Agent Bus with File Locking
# =============================================================================

class LockedAgentBus:
    """Agent bus with file locking for safe concurrent writes."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        line = json.dumps(full_event, ensure_ascii=False, separators=(",", ":")) + "\n"

        fd = os.open(str(self.bus_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        return event_id


# =============================================================================
# Health Check Types
# =============================================================================

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class HealthReport:
    """Aggregated health report."""
    overall_status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: float = field(default_factory=time.time)
    uptime_s: float = 0.0
    version: str = "0.1.0"
    resources: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp,
            "iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "uptime_s": self.uptime_s,
            "version": self.version,
            "resources": self.resources,
        }


# =============================================================================
# Health Check Interface
# =============================================================================

class HealthCheck(ABC):
    """Abstract base class for health checks."""

    def __init__(self, name: str, critical: bool = False):
        self.name = name
        self.critical = critical

    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform health check."""
        pass


class DependencyCheck(HealthCheck):
    """Check external dependency health."""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], bool],
        critical: bool = True,
    ):
        super().__init__(name, critical)
        self.check_func = check_func

    async def check(self) -> HealthCheckResult:
        start = time.time()
        try:
            if asyncio.iscoroutinefunction(self.check_func):
                healthy = await self.check_func()
            else:
                healthy = self.check_func()

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.HEALTHY if healthy else HealthStatus.UNHEALTHY,
                message="" if healthy else "Check failed",
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# Built-in Checks
# =============================================================================

class MemoryCheck(HealthCheck):
    """Check memory usage."""

    def __init__(self, threshold_percent: float = 90.0):
        super().__init__("memory", critical=True)
        self.threshold = threshold_percent

    async def check(self) -> HealthCheckResult:
        start = time.time()
        memory = psutil.virtual_memory()
        percent = memory.percent

        if percent >= self.threshold:
            status = HealthStatus.UNHEALTHY
            message = f"Memory usage at {percent:.1f}% (threshold: {self.threshold}%)"
        elif percent >= self.threshold * 0.8:
            status = HealthStatus.DEGRADED
            message = f"Memory usage at {percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage at {percent:.1f}%"

        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            duration_ms=(time.time() - start) * 1000,
            metadata={
                "total_bytes": memory.total,
                "available_bytes": memory.available,
                "percent": percent,
            },
        )


class DiskCheck(HealthCheck):
    """Check disk usage."""

    def __init__(self, path: str = "/", threshold_percent: float = 90.0):
        super().__init__("disk", critical=True)
        self.path = path
        self.threshold = threshold_percent

    async def check(self) -> HealthCheckResult:
        start = time.time()
        disk = psutil.disk_usage(self.path)
        percent = disk.percent

        if percent >= self.threshold:
            status = HealthStatus.UNHEALTHY
            message = f"Disk usage at {percent:.1f}% (threshold: {self.threshold}%)"
        elif percent >= self.threshold * 0.8:
            status = HealthStatus.DEGRADED
            message = f"Disk usage at {percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage at {percent:.1f}%"

        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            duration_ms=(time.time() - start) * 1000,
            metadata={
                "total_bytes": disk.total,
                "free_bytes": disk.free,
                "percent": percent,
            },
        )


class CPUCheck(HealthCheck):
    """Check CPU usage."""

    def __init__(self, threshold_percent: float = 90.0):
        super().__init__("cpu", critical=False)
        self.threshold = threshold_percent

    async def check(self) -> HealthCheckResult:
        start = time.time()
        percent = psutil.cpu_percent(interval=0.1)

        if percent >= self.threshold:
            status = HealthStatus.DEGRADED
            message = f"CPU usage at {percent:.1f}% (threshold: {self.threshold}%)"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU usage at {percent:.1f}%"

        return HealthCheckResult(
            name=self.name,
            status=status,
            message=message,
            duration_ms=(time.time() - start) * 1000,
            metadata={
                "percent": percent,
                "count": psutil.cpu_count(),
            },
        )


class BusCheck(HealthCheck):
    """Check agent bus health."""

    def __init__(self, bus_path: Optional[Path] = None):
        super().__init__("bus", critical=True)
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        self.bus_path = bus_path or (pluribus_root / ".pluribus" / "bus" / "events.ndjson")

    async def check(self) -> HealthCheckResult:
        start = time.time()

        # Check if bus directory exists and is writable
        bus_dir = self.bus_path.parent

        if not bus_dir.exists():
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message="Bus directory does not exist",
                duration_ms=(time.time() - start) * 1000,
            )

        if not os.access(bus_dir, os.W_OK):
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message="Bus directory not writable",
                duration_ms=(time.time() - start) * 1000,
            )

        # Check bus file size
        if self.bus_path.exists():
            size = self.bus_path.stat().st_size
            if size > 100 * 1024 * 1024:  # 100MB
                return HealthCheckResult(
                    name=self.name,
                    status=HealthStatus.DEGRADED,
                    message=f"Bus file is large: {size / 1024 / 1024:.1f}MB",
                    duration_ms=(time.time() - start) * 1000,
                )

        return HealthCheckResult(
            name=self.name,
            status=HealthStatus.HEALTHY,
            message="Bus is healthy",
            duration_ms=(time.time() - start) * 1000,
        )


# =============================================================================
# Health Checker
# =============================================================================

class HealthChecker:
    """
    Health monitoring system.

    PBTSO Phase: VERIFY, TEST

    Features:
    - Extensible health checks
    - Resource monitoring
    - Heartbeat management
    - Alert notification

    Usage:
        checker = HealthChecker(config)
        checker.add_check(DependencyCheck("database", check_db))
        report = await checker.check_all()
    """

    BUS_TOPICS = {
        "check": "code.health.check",
        "status": "code.health.status",
        "heartbeat": "code.health.heartbeat",
        "alert": "code.health.alert",
    }

    def __init__(
        self,
        config: Optional[HealthCheckConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or HealthCheckConfig()
        self.bus = bus or LockedAgentBus()
        self._checks: List[HealthCheck] = []
        self._check_history: Dict[str, List[HealthCheckResult]] = {}
        self._start_time = time.time()
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._lock = Lock()
        self._last_status: HealthStatus = HealthStatus.UNKNOWN

        # Add built-in checks
        if self.config.enable_resource_monitoring:
            self._add_builtin_checks()

    def _add_builtin_checks(self) -> None:
        """Add built-in health checks."""
        self._checks.append(MemoryCheck(self.config.memory_threshold_percent))
        self._checks.append(DiskCheck(threshold_percent=self.config.disk_threshold_percent))
        self._checks.append(CPUCheck(self.config.cpu_threshold_percent))
        self._checks.append(BusCheck())

    # =========================================================================
    # Check Management
    # =========================================================================

    def add_check(self, check: HealthCheck) -> None:
        """Add a health check."""
        self._checks.append(check)

    def remove_check(self, name: str) -> bool:
        """Remove a health check by name."""
        for check in self._checks:
            if check.name == name:
                self._checks.remove(check)
                return True
        return False

    async def check_one(self, name: str) -> Optional[HealthCheckResult]:
        """Run a single health check."""
        for check in self._checks:
            if check.name == name:
                try:
                    result = await asyncio.wait_for(
                        check.check(),
                        timeout=self.config.timeout_s,
                    )
                except asyncio.TimeoutError:
                    result = HealthCheckResult(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message="Check timed out",
                    )
                except Exception as e:
                    result = HealthCheckResult(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=str(e),
                    )

                # Store in history
                with self._lock:
                    if name not in self._check_history:
                        self._check_history[name] = []
                    self._check_history[name].append(result)
                    # Keep last 100
                    self._check_history[name] = self._check_history[name][-100:]

                return result

        return None

    async def check_all(self) -> HealthReport:
        """Run all health checks and generate report."""
        results: List[HealthCheckResult] = []

        for check in self._checks:
            result = await self.check_one(check.name)
            if result:
                results.append(result)

        # Determine overall status
        overall = self._calculate_overall_status(results)

        # Get resource info
        resources = self._get_resource_info()

        report = HealthReport(
            overall_status=overall,
            checks=results,
            uptime_s=time.time() - self._start_time,
            resources=resources,
        )

        # Emit status event
        self.bus.emit({
            "topic": self.BUS_TOPICS["status"],
            "kind": "health",
            "actor": "health-checker",
            "data": report.to_dict(),
        })

        # Check for status change
        if overall != self._last_status:
            self._emit_alert(overall, self._last_status)
            self._last_status = overall

        return report

    def _calculate_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Calculate overall status from check results."""
        if not results:
            return HealthStatus.UNKNOWN

        critical_unhealthy = any(
            r.status == HealthStatus.UNHEALTHY
            for r in results
            for c in self._checks
            if c.name == r.name and c.critical
        )

        if critical_unhealthy:
            return HealthStatus.UNHEALTHY

        any_unhealthy = any(r.status == HealthStatus.UNHEALTHY for r in results)
        any_degraded = any(r.status == HealthStatus.DEGRADED for r in results)

        if any_unhealthy:
            return HealthStatus.DEGRADED
        if any_degraded:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def _get_resource_info(self) -> Dict[str, Any]:
        """Get current resource information."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "memory": {
                "total_mb": memory.total / 1024 / 1024,
                "available_mb": memory.available / 1024 / 1024,
                "percent": memory.percent,
            },
            "disk": {
                "total_gb": disk.total / 1024 / 1024 / 1024,
                "free_gb": disk.free / 1024 / 1024 / 1024,
                "percent": disk.percent,
            },
            "cpu": {
                "percent": psutil.cpu_percent(interval=0.1),
                "count": psutil.cpu_count(),
            },
            "process": {
                "pid": os.getpid(),
                "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            },
        }

    def _emit_alert(self, new_status: HealthStatus, old_status: HealthStatus) -> None:
        """Emit health alert on status change."""
        self.bus.emit({
            "topic": self.BUS_TOPICS["alert"],
            "kind": "alert",
            "level": "warning" if new_status == HealthStatus.DEGRADED else "error",
            "actor": "health-checker",
            "data": {
                "old_status": old_status.value,
                "new_status": new_status.value,
            },
        })

    # =========================================================================
    # Background Tasks
    # =========================================================================

    async def start(self) -> None:
        """Start background health checking and heartbeat."""
        if self._running:
            return

        self._running = True

        # Start check loop
        self._check_task = asyncio.create_task(self._check_loop())

        # Start heartbeat
        if self.config.enable_heartbeat:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop background tasks."""
        self._running = False

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

    async def _check_loop(self) -> None:
        """Background check loop."""
        while self._running:
            try:
                await self.check_all()
                await asyncio.sleep(self.config.check_interval_s)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self.config.check_interval_s)

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval_s)
                self._emit_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def _emit_heartbeat(self) -> None:
        """Emit heartbeat event."""
        self.bus.emit({
            "topic": self.BUS_TOPICS["heartbeat"],
            "kind": "heartbeat",
            "actor": "health-checker",
            "data": {
                "status": self._last_status.value,
                "uptime_s": time.time() - self._start_time,
                "checks": len(self._checks),
            },
        })

    # =========================================================================
    # Queries
    # =========================================================================

    def get_history(self, name: str, limit: int = 10) -> List[HealthCheckResult]:
        """Get check history for a specific check."""
        with self._lock:
            history = self._check_history.get(name, [])
            return history[-limit:]

    def get_status(self) -> HealthStatus:
        """Get current overall status."""
        return self._last_status

    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self._last_status == HealthStatus.HEALTHY


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Health Check."""
    import argparse

    parser = argparse.ArgumentParser(description="Health Check (Step 87)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check command
    check_parser = subparsers.add_parser("check", help="Run health checks")
    check_parser.add_argument("--name", "-n", help="Check specific component")
    check_parser.add_argument("--json", action="store_true", help="JSON output")

    # status command
    subparsers.add_parser("status", help="Get health status")

    # resources command
    subparsers.add_parser("resources", help="Show resource usage")

    args = parser.parse_args()
    checker = HealthChecker()

    async def run() -> int:
        if args.command == "check":
            if args.name:
                result = await checker.check_one(args.name)
                if result:
                    if args.json:
                        print(json.dumps(result.to_dict(), indent=2))
                    else:
                        print(f"{result.name}: {result.status.value}")
                        print(f"  Message: {result.message}")
                        print(f"  Duration: {result.duration_ms:.2f}ms")
                    return 0 if result.status == HealthStatus.HEALTHY else 1
                else:
                    print(f"Unknown check: {args.name}")
                    return 1
            else:
                report = await checker.check_all()
                if args.json:
                    print(json.dumps(report.to_dict(), indent=2))
                else:
                    print(f"Overall: {report.overall_status.value}")
                    print(f"Uptime: {report.uptime_s:.0f}s")
                    print("\nChecks:")
                    for check in report.checks:
                        symbol = "[OK]" if check.status == HealthStatus.HEALTHY else "[!]"
                        print(f"  {symbol} {check.name}: {check.status.value}")
                return 0 if report.overall_status == HealthStatus.HEALTHY else 1

        elif args.command == "status":
            report = await checker.check_all()
            print(report.overall_status.value)
            return 0 if report.overall_status == HealthStatus.HEALTHY else 1

        elif args.command == "resources":
            resources = checker._get_resource_info()
            print(json.dumps(resources, indent=2))
            return 0

        return 1

    return asyncio.run(run())


if __name__ == "__main__":
    import sys
    sys.exit(main())
