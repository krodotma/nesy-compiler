#!/usr/bin/env python3
"""
Review Health Check System (Step 187)

Health monitoring system for the Review Agent with component checks,
dependency validation, and A2A heartbeat management.

PBTSO Phase: OBSERVE, VERIFY
Bus Topics: review.health.check, review.health.status, a2a.heartbeat

Health Features:
- Component health checks
- Dependency status
- Resource monitoring
- A2A heartbeat protocol
- Health aggregation

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import psutil
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Awaitable

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300  # 5 minutes
A2A_HEARTBEAT_TIMEOUT = 900   # 15 minutes


# ============================================================================
# Types
# ============================================================================

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"       # All checks passing
    DEGRADED = "degraded"     # Some checks failing, still operational
    UNHEALTHY = "unhealthy"   # Critical checks failing
    UNKNOWN = "unknown"       # Unable to determine health


class CheckType(Enum):
    """Types of health checks."""
    LIVENESS = "liveness"     # Is the service alive?
    READINESS = "readiness"   # Is the service ready to accept requests?
    STARTUP = "startup"       # Has the service finished starting?
    DEPENDENCY = "dependency" # Is a dependency available?
    RESOURCE = "resource"     # Are resources within limits?


class CheckPriority(Enum):
    """Priority of health checks."""
    CRITICAL = "critical"  # Failure means unhealthy
    HIGH = "high"          # Failure means degraded
    LOW = "low"            # Informational


@dataclass
class CheckResult:
    """
    Result of a health check.

    Attributes:
        name: Check name
        status: Health status
        message: Status message
        duration_ms: Check duration
        timestamp: Check timestamp
        details: Additional details
        check_type: Type of check
        priority: Check priority
    """
    name: str
    status: HealthStatus
    message: str = ""
    duration_ms: float = 0
    timestamp: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    check_type: CheckType = CheckType.LIVENESS
    priority: CheckPriority = CheckPriority.HIGH

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": round(self.duration_ms, 2),
            "timestamp": self.timestamp,
            "details": self.details,
            "check_type": self.check_type.value,
            "priority": self.priority.value,
        }


@dataclass
class HealthCheck:
    """
    Definition of a health check.

    Attributes:
        name: Check name
        check_fn: Async function that performs the check
        check_type: Type of check
        priority: Check priority
        timeout_seconds: Check timeout
        interval_seconds: How often to run
        description: Check description
    """
    name: str
    check_fn: Callable[[], Awaitable[CheckResult]]
    check_type: CheckType = CheckType.LIVENESS
    priority: CheckPriority = CheckPriority.HIGH
    timeout_seconds: float = 5.0
    interval_seconds: float = 30.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without check_fn)."""
        return {
            "name": self.name,
            "check_type": self.check_type.value,
            "priority": self.priority.value,
            "timeout_seconds": self.timeout_seconds,
            "interval_seconds": self.interval_seconds,
            "description": self.description,
        }


@dataclass
class HealthConfig:
    """
    Configuration for health checking.

    Attributes:
        enabled: Enable health checking
        check_interval_seconds: Default check interval
        timeout_seconds: Default check timeout
        heartbeat_interval: A2A heartbeat interval
        heartbeat_timeout: A2A heartbeat timeout
        enable_resource_checks: Enable resource monitoring
        memory_threshold_percent: Memory warning threshold
        cpu_threshold_percent: CPU warning threshold
        disk_threshold_percent: Disk warning threshold
    """
    enabled: bool = True
    check_interval_seconds: float = 30.0
    timeout_seconds: float = 5.0
    heartbeat_interval: int = A2A_HEARTBEAT_INTERVAL
    heartbeat_timeout: int = A2A_HEARTBEAT_TIMEOUT
    enable_resource_checks: bool = True
    memory_threshold_percent: float = 90.0
    cpu_threshold_percent: float = 90.0
    disk_threshold_percent: float = 90.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HealthReport:
    """
    Aggregated health report.

    Attributes:
        report_id: Unique report ID
        overall_status: Overall health status
        checks: Individual check results
        timestamp: Report timestamp
        duration_ms: Total check duration
        component: Component name
        version: Component version
    """
    report_id: str
    overall_status: HealthStatus
    checks: List[CheckResult] = field(default_factory=list)
    timestamp: str = ""
    duration_ms: float = 0
    component: str = "review-agent"
    version: str = "1.0.0"

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "overall_status": self.overall_status.value,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 2),
            "component": self.component,
            "version": self.version,
            "summary": {
                "total": len(self.checks),
                "healthy": sum(1 for c in self.checks if c.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for c in self.checks if c.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for c in self.checks if c.status == HealthStatus.UNHEALTHY),
            },
        }


# ============================================================================
# Built-in Health Checks
# ============================================================================

async def check_memory(threshold: float = 90.0) -> CheckResult:
    """Check memory usage."""
    try:
        memory = psutil.virtual_memory()
        used_percent = memory.percent

        if used_percent >= threshold:
            return CheckResult(
                name="memory",
                status=HealthStatus.DEGRADED,
                message=f"Memory usage high: {used_percent:.1f}%",
                check_type=CheckType.RESOURCE,
                priority=CheckPriority.HIGH,
                details={
                    "used_percent": used_percent,
                    "available_mb": memory.available / 1024 / 1024,
                    "total_mb": memory.total / 1024 / 1024,
                },
            )

        return CheckResult(
            name="memory",
            status=HealthStatus.HEALTHY,
            message=f"Memory OK: {used_percent:.1f}%",
            check_type=CheckType.RESOURCE,
            priority=CheckPriority.HIGH,
            details={
                "used_percent": used_percent,
                "available_mb": memory.available / 1024 / 1024,
            },
        )
    except Exception as e:
        return CheckResult(
            name="memory",
            status=HealthStatus.UNKNOWN,
            message=f"Failed to check memory: {e}",
            check_type=CheckType.RESOURCE,
            priority=CheckPriority.HIGH,
        )


async def check_cpu(threshold: float = 90.0) -> CheckResult:
    """Check CPU usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)

        if cpu_percent >= threshold:
            return CheckResult(
                name="cpu",
                status=HealthStatus.DEGRADED,
                message=f"CPU usage high: {cpu_percent:.1f}%",
                check_type=CheckType.RESOURCE,
                priority=CheckPriority.HIGH,
                details={"cpu_percent": cpu_percent},
            )

        return CheckResult(
            name="cpu",
            status=HealthStatus.HEALTHY,
            message=f"CPU OK: {cpu_percent:.1f}%",
            check_type=CheckType.RESOURCE,
            priority=CheckPriority.HIGH,
            details={"cpu_percent": cpu_percent},
        )
    except Exception as e:
        return CheckResult(
            name="cpu",
            status=HealthStatus.UNKNOWN,
            message=f"Failed to check CPU: {e}",
            check_type=CheckType.RESOURCE,
            priority=CheckPriority.HIGH,
        )


async def check_disk(path: str = "/", threshold: float = 90.0) -> CheckResult:
    """Check disk usage."""
    try:
        disk = psutil.disk_usage(path)
        used_percent = disk.percent

        if used_percent >= threshold:
            return CheckResult(
                name="disk",
                status=HealthStatus.DEGRADED,
                message=f"Disk usage high: {used_percent:.1f}%",
                check_type=CheckType.RESOURCE,
                priority=CheckPriority.HIGH,
                details={
                    "path": path,
                    "used_percent": used_percent,
                    "free_gb": disk.free / 1024 / 1024 / 1024,
                },
            )

        return CheckResult(
            name="disk",
            status=HealthStatus.HEALTHY,
            message=f"Disk OK: {used_percent:.1f}%",
            check_type=CheckType.RESOURCE,
            priority=CheckPriority.HIGH,
            details={
                "path": path,
                "used_percent": used_percent,
                "free_gb": disk.free / 1024 / 1024 / 1024,
            },
        )
    except Exception as e:
        return CheckResult(
            name="disk",
            status=HealthStatus.UNKNOWN,
            message=f"Failed to check disk: {e}",
            check_type=CheckType.RESOURCE,
            priority=CheckPriority.HIGH,
        )


async def check_bus_path(bus_path: Path) -> CheckResult:
    """Check event bus accessibility."""
    try:
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Try to write a test event
        test_file = bus_path.parent / ".health_check"
        test_file.write_text("check")
        test_file.unlink()

        return CheckResult(
            name="event_bus",
            status=HealthStatus.HEALTHY,
            message="Event bus accessible",
            check_type=CheckType.DEPENDENCY,
            priority=CheckPriority.CRITICAL,
            details={"path": str(bus_path)},
        )
    except Exception as e:
        return CheckResult(
            name="event_bus",
            status=HealthStatus.UNHEALTHY,
            message=f"Event bus not accessible: {e}",
            check_type=CheckType.DEPENDENCY,
            priority=CheckPriority.CRITICAL,
            details={"path": str(bus_path)},
        )


# ============================================================================
# Health Checker
# ============================================================================

class HealthChecker:
    """
    Health checking system for the Review Agent.

    Example:
        checker = HealthChecker()

        # Register a custom check
        async def check_database():
            # ... check database connection ...
            return CheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connected",
            )

        checker.register(HealthCheck(
            name="database",
            check_fn=check_database,
            check_type=CheckType.DEPENDENCY,
            priority=CheckPriority.CRITICAL,
        ))

        # Run all checks
        report = await checker.check_all()
        print(report.overall_status)

        # Start background checking
        await checker.start()
    """

    BUS_TOPICS = {
        "check": "review.health.check",
        "status": "review.health.status",
        "heartbeat": "a2a.heartbeat",
    }

    def __init__(
        self,
        config: Optional[HealthConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the health checker.

        Args:
            config: Health check configuration
            bus_path: Path to event bus file
        """
        self.config = config or HealthConfig()
        self.bus_path = bus_path or self._get_bus_path()

        # Registered checks
        self._checks: Dict[str, HealthCheck] = {}

        # Last results
        self._last_results: Dict[str, CheckResult] = {}
        self._last_report: Optional[HealthReport] = None

        # Background task
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_heartbeat = time.time()

        # Register default checks
        self._register_default_checks()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "health") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "health-checker",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        if self.config.enable_resource_checks:
            # Memory check
            self.register(HealthCheck(
                name="memory",
                check_fn=lambda: check_memory(self.config.memory_threshold_percent),
                check_type=CheckType.RESOURCE,
                priority=CheckPriority.HIGH,
                description="Memory usage check",
            ))

            # CPU check
            self.register(HealthCheck(
                name="cpu",
                check_fn=lambda: check_cpu(self.config.cpu_threshold_percent),
                check_type=CheckType.RESOURCE,
                priority=CheckPriority.LOW,
                description="CPU usage check",
            ))

            # Disk check
            self.register(HealthCheck(
                name="disk",
                check_fn=lambda: check_disk("/", self.config.disk_threshold_percent),
                check_type=CheckType.RESOURCE,
                priority=CheckPriority.HIGH,
                description="Disk usage check",
            ))

        # Event bus check
        self.register(HealthCheck(
            name="event_bus",
            check_fn=lambda: check_bus_path(self.bus_path),
            check_type=CheckType.DEPENDENCY,
            priority=CheckPriority.CRITICAL,
            description="Event bus accessibility",
        ))

    def register(self, check: HealthCheck) -> None:
        """
        Register a health check.

        Args:
            check: Health check to register
        """
        self._checks[check.name] = check

    def unregister(self, name: str) -> bool:
        """
        Unregister a health check.

        Args:
            name: Check name

        Returns:
            True if unregistered
        """
        if name in self._checks:
            del self._checks[name]
            return True
        return False

    async def run_check(self, name: str) -> CheckResult:
        """
        Run a single health check.

        Args:
            name: Check name

        Returns:
            Check result
        """
        check = self._checks.get(name)
        if not check:
            return CheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check '{name}' not found",
            )

        start_time = time.time()

        try:
            result = await asyncio.wait_for(
                check.check_fn(),
                timeout=check.timeout_seconds,
            )
            result.duration_ms = (time.time() - start_time) * 1000
            result.check_type = check.check_type
            result.priority = check.priority
        except asyncio.TimeoutError:
            result = CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {check.timeout_seconds}s",
                duration_ms=(time.time() - start_time) * 1000,
                check_type=check.check_type,
                priority=check.priority,
            )
        except Exception as e:
            result = CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {e}",
                duration_ms=(time.time() - start_time) * 1000,
                check_type=check.check_type,
                priority=check.priority,
            )

        self._last_results[name] = result
        return result

    async def check_all(
        self,
        check_type: Optional[CheckType] = None,
    ) -> HealthReport:
        """
        Run all health checks.

        Args:
            check_type: Filter by check type

        Returns:
            Health report

        Emits:
            review.health.check
            review.health.status
        """
        start_time = time.time()
        report_id = str(uuid.uuid4())[:8]

        # Filter checks
        checks_to_run = self._checks.values()
        if check_type:
            checks_to_run = [c for c in checks_to_run if c.check_type == check_type]

        # Run checks concurrently
        results = await asyncio.gather(*[
            self.run_check(check.name)
            for check in checks_to_run
        ])

        # Determine overall status
        overall_status = self._calculate_overall_status(results)

        report = HealthReport(
            report_id=report_id,
            overall_status=overall_status,
            checks=list(results),
            duration_ms=(time.time() - start_time) * 1000,
        )

        self._last_report = report

        # Emit events
        self._emit_event(self.BUS_TOPICS["check"], {
            "report_id": report_id,
            "checks_run": len(results),
            "duration_ms": report.duration_ms,
        })

        self._emit_event(self.BUS_TOPICS["status"], {
            "report_id": report_id,
            "status": overall_status.value,
            "summary": report.to_dict()["summary"],
        })

        return report

    def _calculate_overall_status(self, results: List[CheckResult]) -> HealthStatus:
        """Calculate overall health status from individual results."""
        critical_unhealthy = any(
            r.status == HealthStatus.UNHEALTHY and r.priority == CheckPriority.CRITICAL
            for r in results
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

    async def liveness(self) -> CheckResult:
        """
        Kubernetes-style liveness probe.

        Returns:
            Liveness check result
        """
        return CheckResult(
            name="liveness",
            status=HealthStatus.HEALTHY,
            message="Service is alive",
            check_type=CheckType.LIVENESS,
            priority=CheckPriority.CRITICAL,
        )

    async def readiness(self) -> CheckResult:
        """
        Kubernetes-style readiness probe.

        Returns:
            Readiness check result
        """
        # Run critical checks only
        critical_checks = [
            c for c in self._checks.values()
            if c.priority == CheckPriority.CRITICAL
        ]

        if not critical_checks:
            return CheckResult(
                name="readiness",
                status=HealthStatus.HEALTHY,
                message="Service is ready",
                check_type=CheckType.READINESS,
                priority=CheckPriority.CRITICAL,
            )

        results = await asyncio.gather(*[
            self.run_check(c.name) for c in critical_checks
        ])

        all_healthy = all(r.status == HealthStatus.HEALTHY for r in results)

        return CheckResult(
            name="readiness",
            status=HealthStatus.HEALTHY if all_healthy else HealthStatus.UNHEALTHY,
            message="Service is ready" if all_healthy else "Service not ready",
            check_type=CheckType.READINESS,
            priority=CheckPriority.CRITICAL,
            details={
                "checks_passed": sum(1 for r in results if r.status == HealthStatus.HEALTHY),
                "checks_total": len(results),
            },
        )

    def heartbeat(self) -> Dict[str, Any]:
        """
        Send A2A heartbeat.

        Returns:
            Heartbeat status
        """
        now = time.time()
        status = {
            "agent": "health-checker",
            "healthy": self._last_report.overall_status == HealthStatus.HEALTHY if self._last_report else True,
            "overall_status": self._last_report.overall_status.value if self._last_report else "unknown",
            "checks_registered": len(self._checks),
            "last_heartbeat": self._last_heartbeat,
            "interval": self.config.heartbeat_interval,
            "timeout": self.config.heartbeat_timeout,
        }
        self._last_heartbeat = now

        self._emit_event(self.BUS_TOPICS["heartbeat"], status, kind="heartbeat")
        return status

    async def start(self) -> None:
        """Start background health checking."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._background_loop())

    async def stop(self) -> None:
        """Stop background health checking."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _background_loop(self) -> None:
        """Background check loop."""
        while self._running:
            try:
                await self.check_all()

                # Send heartbeat if interval passed
                if time.time() - self._last_heartbeat >= self.config.heartbeat_interval:
                    self.heartbeat()

                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(5)  # Brief delay on error

    def get_last_result(self, name: str) -> Optional[CheckResult]:
        """Get last result for a check."""
        return self._last_results.get(name)

    def get_last_report(self) -> Optional[HealthReport]:
        """Get last health report."""
        return self._last_report

    def get_checks(self) -> List[Dict[str, Any]]:
        """Get all registered checks."""
        return [c.to_dict() for c in self._checks.values()]


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Health Check."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Health Check (Step 187)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Check command
    check_parser = subparsers.add_parser("check", help="Run health checks")
    check_parser.add_argument("--name", help="Specific check to run")
    check_parser.add_argument("--type", choices=["liveness", "readiness", "resource", "dependency"],
                              help="Check type filter")

    # List command
    subparsers.add_parser("list", help="List registered checks")

    # Status command
    subparsers.add_parser("status", help="Show current status")

    # Liveness command
    subparsers.add_parser("liveness", help="Kubernetes liveness probe")

    # Readiness command
    subparsers.add_parser("readiness", help="Kubernetes readiness probe")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    checker = HealthChecker()

    if args.command == "check":
        if args.name:
            result = asyncio.run(checker.run_check(args.name))
            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                status_symbol = {"healthy": "+", "degraded": "~", "unhealthy": "-", "unknown": "?"}
                print(f"[{status_symbol.get(result.status.value, '?')}] {result.name}: {result.message}")
        else:
            check_type = None
            if args.type:
                check_type = CheckType[args.type.upper()]
            report = asyncio.run(checker.check_all(check_type))
            if args.json:
                print(json.dumps(report.to_dict(), indent=2))
            else:
                print(f"Health Report: {report.report_id}")
                print(f"  Overall: {report.overall_status.value.upper()}")
                print(f"  Duration: {report.duration_ms:.1f}ms")
                print("\nChecks:")
                for check in report.checks:
                    status_symbol = {"healthy": "+", "degraded": "~", "unhealthy": "-", "unknown": "?"}
                    print(f"  [{status_symbol.get(check.status.value, '?')}] {check.name}: {check.message}")

    elif args.command == "list":
        checks = checker.get_checks()
        if args.json:
            print(json.dumps(checks, indent=2))
        else:
            print(f"Registered Checks: {len(checks)}")
            for check in checks:
                print(f"  {check['name']} ({check['check_type']}, {check['priority']})")

    elif args.command == "status":
        report = checker.get_last_report()
        if report:
            if args.json:
                print(json.dumps(report.to_dict(), indent=2))
            else:
                print(f"Status: {report.overall_status.value.upper()}")
                print(f"  Last Check: {report.timestamp}")
        else:
            print("No health check run yet")
            return 1

    elif args.command == "liveness":
        result = asyncio.run(checker.liveness())
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Liveness: {result.status.value.upper()}")
        return 0 if result.status == HealthStatus.HEALTHY else 1

    elif args.command == "readiness":
        result = asyncio.run(checker.readiness())
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Readiness: {result.status.value.upper()}")
        return 0 if result.status == HealthStatus.HEALTHY else 1

    else:
        # Default: run all checks
        report = asyncio.run(checker.check_all())
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Health: {report.overall_status.value.upper()}")
            summary = report.to_dict()["summary"]
            print(f"  Checks: {summary['healthy']}/{summary['total']} healthy")

    return 0


if __name__ == "__main__":
    sys.exit(main())
