#!/usr/bin/env python3
"""
health_check.py - Health Monitoring (Step 37)

Health check system for Research Agent components.
Monitors service health, dependencies, and resources.

PBTSO Phase: MONITOR

Bus Topics:
- a2a.research.health.check
- a2a.research.health.status
- research.health.alert

Protocol: DKIN v30, PAIP v16, CITIZEN v2
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
from typing import Any, Callable, Dict, List, Optional, Union

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"       # All systems operational
    DEGRADED = "degraded"     # Some issues, but operational
    UNHEALTHY = "unhealthy"   # Critical issues
    UNKNOWN = "unknown"       # Cannot determine status


class CheckType(Enum):
    """Types of health checks."""
    LIVENESS = "liveness"     # Is the service alive?
    READINESS = "readiness"   # Is the service ready to serve?
    STARTUP = "startup"       # Has the service started?


@dataclass
class HealthConfig:
    """Configuration for health checker."""

    check_interval_seconds: int = 30
    timeout_seconds: int = 10
    failure_threshold: int = 3
    success_threshold: int = 1
    enable_metrics: bool = True
    emit_to_bus: bool = True
    bus_path: Optional[str] = None

    # A2A heartbeat configuration per protocol
    heartbeat_interval: int = 300  # 300s interval
    heartbeat_timeout: int = 900   # 900s timeout

    def __post_init__(self):
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class HealthCheck:
    """A health check definition."""

    name: str
    check_type: CheckType
    check_fn: Callable[[], bool]
    timeout_seconds: int = 10
    interval_seconds: int = 30
    failure_threshold: int = 3
    critical: bool = False  # If true, failure means UNHEALTHY

    # State
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check: Optional[float] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_error: Optional[str] = None
    last_duration_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.check_type.value,
            "status": self.last_status.value,
            "last_check": self.last_check,
            "last_error": self.last_error,
            "last_duration_ms": self.last_duration_ms,
            "consecutive_failures": self.consecutive_failures,
            "critical": self.critical,
        }


@dataclass
class ComponentHealth:
    """Health status of a component."""

    name: str
    status: HealthStatus
    checks: List[HealthCheck] = field(default_factory=list)
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp,
        }


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    components: List[ComponentHealth] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    uptime_seconds: float = 0
    version: str = "0.2.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "components": [c.to_dict() for c in self.components],
            "timestamp": self.timestamp,
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
        }


# ============================================================================
# Health Checker
# ============================================================================


class HealthChecker:
    """
    Health monitoring for Research Agent.

    Features:
    - Multiple health check types (liveness, readiness, startup)
    - Component-level health aggregation
    - Configurable thresholds
    - Async and sync check support
    - A2A heartbeat integration

    PBTSO Phase: MONITOR

    Example:
        checker = HealthChecker()

        # Register checks
        checker.register_check(
            "database",
            CheckType.READINESS,
            lambda: db.ping(),
        )

        # Run checks
        health = checker.check_health()
        print(health.status)

        # Start background monitoring
        checker.start()
    """

    def __init__(
        self,
        config: Optional[HealthConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the health checker.

        Args:
            config: Health check configuration
            bus: AgentBus for event emission
        """
        self.config = config or HealthConfig()
        self.bus = bus or AgentBus()

        # Check registry
        self._checks: Dict[str, HealthCheck] = {}
        self._components: Dict[str, List[str]] = {}  # component -> check names
        self._lock = threading.Lock()

        # State
        self._start_time = time.time()
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Statistics
        self._stats = {
            "total_checks": 0,
            "successful_checks": 0,
            "failed_checks": 0,
        }

        # Register default checks
        self._register_default_checks()

    def register_check(
        self,
        name: str,
        check_type: CheckType,
        check_fn: Callable[[], bool],
        component: str = "default",
        timeout_seconds: Optional[int] = None,
        interval_seconds: Optional[int] = None,
        failure_threshold: Optional[int] = None,
        critical: bool = False,
    ) -> None:
        """
        Register a health check.

        Args:
            name: Check name
            check_type: Type of check
            check_fn: Check function (returns True if healthy)
            component: Component this check belongs to
            timeout_seconds: Check timeout
            interval_seconds: Check interval
            failure_threshold: Failures before unhealthy
            critical: If true, failure is critical
        """
        check = HealthCheck(
            name=name,
            check_type=check_type,
            check_fn=check_fn,
            timeout_seconds=timeout_seconds or self.config.timeout_seconds,
            interval_seconds=interval_seconds or self.config.check_interval_seconds,
            failure_threshold=failure_threshold or self.config.failure_threshold,
            critical=critical,
        )

        with self._lock:
            self._checks[name] = check

            if component not in self._components:
                self._components[component] = []
            if name not in self._components[component]:
                self._components[component].append(name)

    def unregister_check(self, name: str) -> bool:
        """Unregister a health check."""
        with self._lock:
            if name in self._checks:
                del self._checks[name]

                # Remove from components
                for check_list in self._components.values():
                    if name in check_list:
                        check_list.remove(name)

                return True
            return False

    def check_health(self, check_type: Optional[CheckType] = None) -> SystemHealth:
        """
        Run all health checks and return system health.

        Args:
            check_type: Filter by check type

        Returns:
            SystemHealth with status
        """
        component_healths = []

        with self._lock:
            for component, check_names in self._components.items():
                checks = []
                component_status = HealthStatus.HEALTHY

                for name in check_names:
                    check = self._checks.get(name)
                    if check is None:
                        continue

                    if check_type and check.check_type != check_type:
                        continue

                    # Run check
                    self._run_check(check)
                    checks.append(check)

                    # Update component status
                    if check.last_status == HealthStatus.UNHEALTHY:
                        if check.critical:
                            component_status = HealthStatus.UNHEALTHY
                        elif component_status != HealthStatus.UNHEALTHY:
                            component_status = HealthStatus.DEGRADED

                component_healths.append(ComponentHealth(
                    name=component,
                    status=component_status,
                    checks=checks,
                ))

        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        for ch in component_healths:
            if ch.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
                break
            elif ch.status == HealthStatus.DEGRADED:
                overall_status = HealthStatus.DEGRADED

        system_health = SystemHealth(
            status=overall_status,
            components=component_healths,
            uptime_seconds=time.time() - self._start_time,
        )

        # Emit to bus
        if self.config.emit_to_bus:
            self._emit_health(system_health)

        return system_health

    def check_liveness(self) -> bool:
        """Quick liveness check."""
        health = self.check_health(CheckType.LIVENESS)
        return health.status != HealthStatus.UNHEALTHY

    def check_readiness(self) -> bool:
        """Quick readiness check."""
        health = self.check_health(CheckType.READINESS)
        return health.status == HealthStatus.HEALTHY

    def get_check(self, name: str) -> Optional[HealthCheck]:
        """Get a specific health check."""
        return self._checks.get(name)

    def start(self) -> None:
        """Start background health monitoring."""
        if self._running:
            return

        self._stop_event.clear()
        self._running = True

        # Start check thread
        self._check_thread = threading.Thread(target=self._check_loop, daemon=True)
        self._check_thread.start()

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def stop(self) -> None:
        """Stop background health monitoring."""
        self._stop_event.set()
        self._running = False

        if self._check_thread:
            self._check_thread.join(timeout=5)
            self._check_thread = None

        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None

    def get_stats(self) -> Dict[str, Any]:
        """Get health checker statistics."""
        return {
            **self._stats,
            "registered_checks": len(self._checks),
            "components": len(self._components),
            "uptime_seconds": time.time() - self._start_time,
            "running": self._running,
        }

    def _run_check(self, check: HealthCheck) -> None:
        """Run a single health check."""
        start_time = time.time()
        self._stats["total_checks"] += 1

        try:
            # Run with timeout
            result = self._run_with_timeout(check.check_fn, check.timeout_seconds)

            check.last_duration_ms = (time.time() - start_time) * 1000
            check.last_check = time.time()

            if result:
                check.consecutive_successes += 1
                check.consecutive_failures = 0
                check.last_status = HealthStatus.HEALTHY
                check.last_error = None
                self._stats["successful_checks"] += 1
            else:
                check.consecutive_failures += 1
                check.consecutive_successes = 0
                check.last_error = "Check returned False"
                self._stats["failed_checks"] += 1

                if check.consecutive_failures >= check.failure_threshold:
                    check.last_status = HealthStatus.UNHEALTHY
                else:
                    check.last_status = HealthStatus.DEGRADED

        except Exception as e:
            check.last_duration_ms = (time.time() - start_time) * 1000
            check.last_check = time.time()
            check.consecutive_failures += 1
            check.consecutive_successes = 0
            check.last_error = str(e)
            self._stats["failed_checks"] += 1

            if check.consecutive_failures >= check.failure_threshold:
                check.last_status = HealthStatus.UNHEALTHY
            else:
                check.last_status = HealthStatus.DEGRADED

    def _run_with_timeout(self, func: Callable, timeout: int) -> bool:
        """Run function with timeout."""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Check timed out after {timeout}s")

    def _check_loop(self) -> None:
        """Background check loop."""
        while not self._stop_event.is_set():
            self.check_health()
            self._stop_event.wait(self.config.check_interval_seconds)

    def _heartbeat_loop(self) -> None:
        """A2A heartbeat loop (per DKIN protocol)."""
        while not self._stop_event.is_set():
            self._emit_heartbeat()
            self._stop_event.wait(self.config.heartbeat_interval)

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        # Process check
        self.register_check(
            "process",
            CheckType.LIVENESS,
            lambda: True,  # If we're running, we're alive
            component="core",
        )

        # Memory check
        def check_memory():
            try:
                import psutil
                mem = psutil.virtual_memory()
                return mem.percent < 90
            except ImportError:
                return True

        self.register_check(
            "memory",
            CheckType.READINESS,
            check_memory,
            component="resources",
        )

        # Disk check
        def check_disk():
            try:
                import shutil
                usage = shutil.disk_usage("/")
                return (usage.used / usage.total) < 0.95
            except Exception:
                return True

        self.register_check(
            "disk",
            CheckType.READINESS,
            check_disk,
            component="resources",
        )

        # Bus check
        def check_bus():
            bus_path = Path(self.config.bus_path)
            return bus_path.parent.exists()

        self.register_check(
            "bus",
            CheckType.READINESS,
            check_bus,
            component="core",
            critical=True,
        )

    def _emit_health(self, health: SystemHealth) -> None:
        """Emit health status to bus."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": health.timestamp,
            "iso": datetime.fromtimestamp(health.timestamp, tz=timezone.utc).isoformat() + "Z",
            "topic": "a2a.research.health.status",
            "kind": "health",
            "level": "warning" if health.status != HealthStatus.HEALTHY else "info",
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": health.to_dict(),
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _emit_heartbeat(self) -> None:
        """Emit A2A heartbeat."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": "a2a.research.heartbeat",
            "kind": "heartbeat",
            "level": "info",
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": {
                "interval": self.config.heartbeat_interval,
                "timeout": self.config.heartbeat_timeout,
                "uptime_seconds": time.time() - self._start_time,
            },
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Health Check."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Health Check (Step 37)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Check command
    check_parser = subparsers.add_parser("check", help="Run health checks")
    check_parser.add_argument("--type", choices=["liveness", "readiness", "startup"])
    check_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show current status")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Start background monitoring")

    args = parser.parse_args()

    checker = HealthChecker()

    if args.command == "check":
        check_type = None
        if args.type:
            check_type = CheckType(args.type)

        health = checker.check_health(check_type)

        if args.json:
            print(json.dumps(health.to_dict(), indent=2))
        else:
            status_color = {
                HealthStatus.HEALTHY: "\033[32m",
                HealthStatus.DEGRADED: "\033[33m",
                HealthStatus.UNHEALTHY: "\033[31m",
                HealthStatus.UNKNOWN: "\033[37m",
            }
            reset = "\033[0m"

            print(f"System Health: {status_color[health.status]}{health.status.value.upper()}{reset}")
            print(f"Uptime: {health.uptime_seconds:.0f}s")
            print()

            for component in health.components:
                print(f"  {component.name}: {status_color[component.status]}{component.status.value}{reset}")
                for check in component.checks:
                    check_status = status_color[check.last_status]
                    print(f"    - {check.name}: {check_status}{check.last_status.value}{reset} ({check.last_duration_ms:.1f}ms)")
                    if check.last_error:
                        print(f"      Error: {check.last_error}")

        return 0 if health.status == HealthStatus.HEALTHY else 1

    elif args.command == "status":
        health = checker.check_health()

        if args.json:
            print(json.dumps({"status": health.status.value}, indent=2))
        else:
            print(f"Status: {health.status.value}")

        return 0 if health.status == HealthStatus.HEALTHY else 1

    elif args.command == "stats":
        stats = checker.get_stats()

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Health Check Statistics:")
            print(f"  Total Checks: {stats['total_checks']}")
            print(f"  Successful: {stats['successful_checks']}")
            print(f"  Failed: {stats['failed_checks']}")
            print(f"  Registered Checks: {stats['registered_checks']}")
            print(f"  Components: {stats['components']}")

    elif args.command == "monitor":
        print("Starting health monitoring (Ctrl+C to stop)...")
        checker.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            checker.stop()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
