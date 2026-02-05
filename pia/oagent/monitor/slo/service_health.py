#!/usr/bin/env python3
"""
Service Health Monitor - Step 263

Tracks service availability and health across the system.

PBTSO Phase: VERIFY

Bus Topics:
- monitor.service.health (emitted)
- monitor.service.status (emitted)
- monitor.service.alert (emitted)

Protocol: DKIN v30, CITIZEN v2, HOLON v2
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
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import urllib.request
import urllib.error


class HealthState(Enum):
    """Service health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """Types of health checks."""
    HTTP = "http"
    TCP = "tcp"
    PROCESS = "process"
    CUSTOM = "custom"


@dataclass
class ServiceDependency:
    """Service dependency definition.

    Attributes:
        name: Dependency name
        service_name: Dependent service
        dependency_type: Type of dependency (hard/soft)
        required: Whether dependency is required
    """
    name: str
    service_name: str
    dependency_type: str = "hard"  # hard or soft
    required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HealthCheckConfig:
    """Health check configuration.

    Attributes:
        check_type: Type of check
        endpoint: Endpoint to check (URL, host:port, etc.)
        method: HTTP method (for HTTP checks)
        expected_status: Expected HTTP status code
        timeout_s: Check timeout
        interval_s: Check interval
        healthy_threshold: Consecutive successes for healthy
        unhealthy_threshold: Consecutive failures for unhealthy
    """
    check_type: CheckType
    endpoint: str
    method: str = "GET"
    expected_status: int = 200
    timeout_s: float = 5.0
    interval_s: int = 30
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_type": self.check_type.value,
            "endpoint": self.endpoint,
            "method": self.method,
            "expected_status": self.expected_status,
            "timeout_s": self.timeout_s,
            "interval_s": self.interval_s,
            "healthy_threshold": self.healthy_threshold,
            "unhealthy_threshold": self.unhealthy_threshold,
        }


@dataclass
class HealthCheckResult:
    """Health check result.

    Attributes:
        success: Whether check succeeded
        latency_ms: Check latency
        status_code: HTTP status code (for HTTP checks)
        response_body: Response body (if captured)
        error_message: Error message if failed
        timestamp: Check timestamp
    """
    success: bool
    latency_ms: float
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ServiceStatus:
    """Service status.

    Attributes:
        service_name: Name of the service
        state: Current health state
        last_check: Last check result
        consecutive_successes: Consecutive successful checks
        consecutive_failures: Consecutive failed checks
        uptime_percent: Uptime percentage (24h window)
        last_state_change: Timestamp of last state change
        dependencies: Service dependencies
        metadata: Additional metadata
    """
    service_name: str
    state: HealthState
    last_check: Optional[HealthCheckResult] = None
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    uptime_percent: float = 100.0
    last_state_change: float = field(default_factory=time.time)
    dependencies: List[ServiceDependency] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "state": self.state.value,
            "last_check": self.last_check.to_dict() if self.last_check else None,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
            "uptime_percent": self.uptime_percent,
            "last_state_change": self.last_state_change,
            "dependencies": [d.to_dict() for d in self.dependencies],
            "metadata": self.metadata,
        }


@dataclass
class ServiceDefinition:
    """Service definition for monitoring.

    Attributes:
        name: Service name
        health_check: Health check configuration
        dependencies: Service dependencies
        critical: Whether service is critical
        owner: Service owner
        description: Service description
    """
    name: str
    health_check: HealthCheckConfig
    dependencies: List[ServiceDependency] = field(default_factory=list)
    critical: bool = False
    owner: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "health_check": self.health_check.to_dict(),
            "dependencies": [d.to_dict() for d in self.dependencies],
            "critical": self.critical,
            "owner": self.owner,
            "description": self.description,
        }


@dataclass
class ServiceAlert:
    """Service health alert.

    Attributes:
        service_name: Service name
        state: Current state
        previous_state: Previous state
        message: Alert message
        timestamp: Alert timestamp
        alert_id: Unique alert ID
    """
    service_name: str
    state: HealthState
    previous_state: HealthState
    message: str
    timestamp: float = field(default_factory=time.time)
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "service_name": self.service_name,
            "state": self.state.value,
            "previous_state": self.previous_state.value,
            "message": self.message,
            "timestamp": self.timestamp,
        }


class ServiceHealthMonitor:
    """
    Monitor service health and availability.

    The monitor:
    - Performs health checks on configured services
    - Tracks service state transitions
    - Monitors dependencies
    - Calculates uptime metrics
    - Generates alerts on state changes

    Example:
        monitor = ServiceHealthMonitor()
        monitor.register_service(ServiceDefinition(
            name="api-server",
            health_check=HealthCheckConfig(
                check_type=CheckType.HTTP,
                endpoint="http://localhost:8080/health",
            )
        ))
        await monitor.start()

        status = monitor.get_service_status("api-server")
        print(f"API Server: {status.state.value}")
    """

    BUS_TOPICS = {
        "health": "monitor.service.health",
        "status": "monitor.service.status",
        "alert": "monitor.service.alert",
    }

    def __init__(
        self,
        history_size: int = 1000,
        bus_dir: Optional[str] = None,
    ):
        """Initialize service health monitor.

        Args:
            history_size: Check history size per service
            bus_dir: Bus directory
        """
        self.history_size = history_size

        # Service registry
        self._services: Dict[str, ServiceDefinition] = {}
        self._status: Dict[str, ServiceStatus] = {}
        self._check_history: Dict[str, List[HealthCheckResult]] = {}

        # State
        self._running = False
        self._check_tasks: Dict[str, asyncio.Task] = {}
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._alert_callbacks: List[Callable[[ServiceAlert], None]] = []

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def register_service(self, service: ServiceDefinition) -> None:
        """Register a service for monitoring.

        Args:
            service: Service definition
        """
        self._services[service.name] = service
        self._status[service.name] = ServiceStatus(
            service_name=service.name,
            state=HealthState.UNKNOWN,
            dependencies=service.dependencies,
        )
        self._check_history[service.name] = []

        # Start check task if running
        if self._running:
            self._start_service_check(service.name)

    def unregister_service(self, name: str) -> bool:
        """Unregister a service.

        Args:
            name: Service name

        Returns:
            True if unregistered
        """
        if name not in self._services:
            return False

        # Stop check task
        if name in self._check_tasks:
            self._check_tasks[name].cancel()
            del self._check_tasks[name]

        del self._services[name]
        del self._status[name]
        del self._check_history[name]
        return True

    async def start(self) -> bool:
        """Start the health monitor.

        Returns:
            True if started
        """
        if self._running:
            return False

        self._running = True

        # Start check tasks for all services
        for name in self._services:
            self._start_service_check(name)

        self._emit_bus_event(
            "monitor.service.started",
            {"services": list(self._services.keys())}
        )

        return True

    async def stop(self) -> bool:
        """Stop the health monitor.

        Returns:
            True if stopped
        """
        if not self._running:
            return False

        self._running = False

        # Cancel all check tasks
        for task in self._check_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._check_tasks.clear()
        self._executor.shutdown(wait=False)
        return True

    def get_service_status(self, name: str) -> Optional[ServiceStatus]:
        """Get service status.

        Args:
            name: Service name

        Returns:
            Service status or None
        """
        return self._status.get(name)

    def get_all_status(self) -> Dict[str, ServiceStatus]:
        """Get status of all services.

        Returns:
            Dictionary of service statuses
        """
        return dict(self._status)

    def get_healthy_services(self) -> List[str]:
        """Get list of healthy services.

        Returns:
            List of healthy service names
        """
        return [
            name for name, status in self._status.items()
            if status.state == HealthState.HEALTHY
        ]

    def get_unhealthy_services(self) -> List[str]:
        """Get list of unhealthy services.

        Returns:
            List of unhealthy service names
        """
        return [
            name for name, status in self._status.items()
            if status.state == HealthState.UNHEALTHY
        ]

    def get_service_uptime(
        self,
        name: str,
        window_s: int = 86400
    ) -> float:
        """Calculate service uptime over window.

        Args:
            name: Service name
            window_s: Time window in seconds

        Returns:
            Uptime percentage (0-100)
        """
        if name not in self._check_history:
            return 0.0

        cutoff = time.time() - window_s
        history = [
            h for h in self._check_history[name]
            if h.timestamp >= cutoff
        ]

        if not history:
            return 0.0

        successful = sum(1 for h in history if h.success)
        return 100.0 * successful / len(history)

    def get_check_history(
        self,
        name: str,
        window_s: int = 3600
    ) -> List[HealthCheckResult]:
        """Get check history for a service.

        Args:
            name: Service name
            window_s: Time window

        Returns:
            Check history
        """
        if name not in self._check_history:
            return []

        cutoff = time.time() - window_s
        return [
            h for h in self._check_history[name]
            if h.timestamp >= cutoff
        ]

    async def check_service(self, name: str) -> HealthCheckResult:
        """Perform health check for a service.

        Args:
            name: Service name

        Returns:
            Check result
        """
        if name not in self._services:
            return HealthCheckResult(
                success=False,
                latency_ms=0.0,
                error_message=f"Service not found: {name}",
            )

        service = self._services[name]
        config = service.health_check

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._perform_check,
            config
        )

        # Update status
        self._update_status(name, result)

        # Store in history
        self._check_history[name].append(result)
        if len(self._check_history[name]) > self.history_size:
            self._check_history[name] = self._check_history[name][-self.history_size:]

        return result

    def register_alert_callback(
        self,
        callback: Callable[[ServiceAlert], None]
    ) -> None:
        """Register alert callback.

        Args:
            callback: Callback function
        """
        self._alert_callbacks.append(callback)

    def get_dependency_tree(
        self,
        name: str
    ) -> Dict[str, Any]:
        """Get dependency tree for a service.

        Args:
            name: Service name

        Returns:
            Dependency tree
        """
        if name not in self._services:
            return {}

        def build_tree(svc_name: str, visited: set) -> Dict[str, Any]:
            if svc_name in visited:
                return {"name": svc_name, "circular": True}

            visited.add(svc_name)
            service = self._services.get(svc_name)
            if not service:
                return {
                    "name": svc_name,
                    "state": "unknown",
                    "dependencies": [],
                }

            status = self._status.get(svc_name)
            return {
                "name": svc_name,
                "state": status.state.value if status else "unknown",
                "dependencies": [
                    build_tree(dep.service_name, visited.copy())
                    for dep in service.dependencies
                    if dep.service_name in self._services
                ],
            }

        return build_tree(name, set())

    def get_status_summary(self) -> Dict[str, Any]:
        """Get overall status summary.

        Returns:
            Status summary
        """
        total = len(self._status)
        by_state = {}
        for status in self._status.values():
            state = status.state.value
            by_state[state] = by_state.get(state, 0) + 1

        critical_unhealthy = [
            name for name, status in self._status.items()
            if status.state == HealthState.UNHEALTHY
            and self._services[name].critical
        ]

        return {
            "total_services": total,
            "by_state": by_state,
            "healthy_percent": 100.0 * by_state.get("healthy", 0) / total if total > 0 else 0.0,
            "critical_unhealthy": critical_unhealthy,
        }

    def _start_service_check(self, name: str) -> None:
        """Start check task for a service.

        Args:
            name: Service name
        """
        if name in self._check_tasks:
            return

        service = self._services[name]
        task = asyncio.create_task(
            self._check_loop(name, service.health_check.interval_s)
        )
        self._check_tasks[name] = task

    async def _check_loop(self, name: str, interval_s: int) -> None:
        """Check loop for a service.

        Args:
            name: Service name
            interval_s: Check interval
        """
        while self._running and name in self._services:
            try:
                result = await self.check_service(name)

                # Emit event
                status = self._status.get(name)
                if status:
                    self._emit_bus_event(
                        self.BUS_TOPICS["status"],
                        status.to_dict()
                    )
            except Exception as e:
                self._emit_bus_event(
                    "monitor.service.error",
                    {"service": name, "error": str(e)},
                    level="error"
                )

            await asyncio.sleep(interval_s)

    def _perform_check(self, config: HealthCheckConfig) -> HealthCheckResult:
        """Perform health check.

        Args:
            config: Check configuration

        Returns:
            Check result
        """
        start = time.time()

        if config.check_type == CheckType.HTTP:
            return self._perform_http_check(config, start)
        elif config.check_type == CheckType.TCP:
            return self._perform_tcp_check(config, start)
        elif config.check_type == CheckType.PROCESS:
            return self._perform_process_check(config, start)
        else:
            return HealthCheckResult(
                success=False,
                latency_ms=0.0,
                error_message=f"Unknown check type: {config.check_type}",
            )

    def _perform_http_check(
        self,
        config: HealthCheckConfig,
        start: float
    ) -> HealthCheckResult:
        """Perform HTTP health check.

        Args:
            config: Check configuration
            start: Start timestamp

        Returns:
            Check result
        """
        try:
            request = urllib.request.Request(
                config.endpoint,
                method=config.method,
            )
            request.add_header("User-Agent", "PluribusMonitor/1.0")

            with urllib.request.urlopen(
                request,
                timeout=config.timeout_s
            ) as response:
                latency_ms = (time.time() - start) * 1000
                status_code = response.getcode()
                body = response.read().decode("utf-8", errors="replace")[:1000]

                success = status_code == config.expected_status

                return HealthCheckResult(
                    success=success,
                    latency_ms=latency_ms,
                    status_code=status_code,
                    response_body=body if not success else None,
                    error_message=f"Expected {config.expected_status}, got {status_code}" if not success else None,
                )

        except urllib.error.HTTPError as e:
            latency_ms = (time.time() - start) * 1000
            return HealthCheckResult(
                success=False,
                latency_ms=latency_ms,
                status_code=e.code,
                error_message=str(e),
            )
        except urllib.error.URLError as e:
            latency_ms = (time.time() - start) * 1000
            return HealthCheckResult(
                success=False,
                latency_ms=latency_ms,
                error_message=f"URL error: {e.reason}",
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return HealthCheckResult(
                success=False,
                latency_ms=latency_ms,
                error_message=str(e),
            )

    def _perform_tcp_check(
        self,
        config: HealthCheckConfig,
        start: float
    ) -> HealthCheckResult:
        """Perform TCP health check.

        Args:
            config: Check configuration
            start: Start timestamp

        Returns:
            Check result
        """
        try:
            # Parse host:port
            if ":" in config.endpoint:
                host, port_str = config.endpoint.rsplit(":", 1)
                port = int(port_str)
            else:
                host = config.endpoint
                port = 80

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(config.timeout_s)

            try:
                result = sock.connect_ex((host, port))
                latency_ms = (time.time() - start) * 1000

                if result == 0:
                    return HealthCheckResult(
                        success=True,
                        latency_ms=latency_ms,
                    )
                else:
                    return HealthCheckResult(
                        success=False,
                        latency_ms=latency_ms,
                        error_message=f"Connection failed: error code {result}",
                    )
            finally:
                sock.close()

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return HealthCheckResult(
                success=False,
                latency_ms=latency_ms,
                error_message=str(e),
            )

    def _perform_process_check(
        self,
        config: HealthCheckConfig,
        start: float
    ) -> HealthCheckResult:
        """Perform process health check.

        Args:
            config: Check configuration
            start: Start timestamp

        Returns:
            Check result
        """
        try:
            # Check if process is running by name
            process_name = config.endpoint

            import subprocess
            result = subprocess.run(
                ["pgrep", "-f", process_name],
                capture_output=True,
                timeout=config.timeout_s
            )

            latency_ms = (time.time() - start) * 1000

            if result.returncode == 0:
                pids = result.stdout.decode().strip().split("\n")
                return HealthCheckResult(
                    success=True,
                    latency_ms=latency_ms,
                    response_body=f"PIDs: {', '.join(pids)}",
                )
            else:
                return HealthCheckResult(
                    success=False,
                    latency_ms=latency_ms,
                    error_message=f"Process not found: {process_name}",
                )

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return HealthCheckResult(
                success=False,
                latency_ms=latency_ms,
                error_message=str(e),
            )

    def _update_status(
        self,
        name: str,
        result: HealthCheckResult
    ) -> None:
        """Update service status based on check result.

        Args:
            name: Service name
            result: Check result
        """
        if name not in self._status:
            return

        status = self._status[name]
        previous_state = status.state

        if result.success:
            status.consecutive_successes += 1
            status.consecutive_failures = 0
        else:
            status.consecutive_failures += 1
            status.consecutive_successes = 0

        status.last_check = result

        # Determine new state
        config = self._services[name].health_check
        new_state = status.state

        if status.consecutive_successes >= config.healthy_threshold:
            new_state = HealthState.HEALTHY
        elif status.consecutive_failures >= config.unhealthy_threshold:
            new_state = HealthState.UNHEALTHY
        elif status.consecutive_failures > 0:
            new_state = HealthState.DEGRADED

        # Check if state changed
        if new_state != previous_state:
            status.state = new_state
            status.last_state_change = time.time()

            # Generate alert
            alert = ServiceAlert(
                service_name=name,
                state=new_state,
                previous_state=previous_state,
                message=f"Service {name} changed from {previous_state.value} to {new_state.value}",
            )

            self._emit_bus_event(
                self.BUS_TOPICS["alert"],
                alert.to_dict(),
                level="warning" if new_state != HealthState.HEALTHY else "info"
            )

            for callback in self._alert_callbacks:
                callback(alert)

        # Update uptime
        status.uptime_percent = self.get_service_uptime(name, 86400)

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
_monitor: Optional[ServiceHealthMonitor] = None


def get_service_health_monitor() -> ServiceHealthMonitor:
    """Get or create the service health monitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = ServiceHealthMonitor()
    return _monitor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Service Health Monitor (Step 263)")
    parser.add_argument("--status", action="store_true", help="Show status summary")
    parser.add_argument("--check", metavar="URL", help="Check a URL once")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    monitor = get_service_health_monitor()

    if args.check:
        # Quick one-off check
        service = ServiceDefinition(
            name="test",
            health_check=HealthCheckConfig(
                check_type=CheckType.HTTP,
                endpoint=args.check,
            )
        )
        monitor.register_service(service)

        async def main():
            result = await monitor.check_service("test")
            if args.json:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                status = "OK" if result.success else "FAIL"
                print(f"[{status}] {args.check}: {result.latency_ms:.1f}ms")
                if result.error_message:
                    print(f"  Error: {result.error_message}")

        asyncio.run(main())

    if args.status:
        summary = monitor.get_status_summary()
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("Service Health Summary:")
            print(f"  Total services: {summary['total_services']}")
            print(f"  Healthy: {summary['healthy_percent']:.1f}%")
            print(f"  By state: {summary['by_state']}")
            if summary['critical_unhealthy']:
                print(f"  Critical unhealthy: {', '.join(summary['critical_unhealthy'])}")
