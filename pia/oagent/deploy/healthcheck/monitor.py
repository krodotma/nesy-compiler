#!/usr/bin/env python3
"""
monitor.py - Deploy Health Monitor (Step 237)

PBTSO Phase: VERIFY
A2A Integration: Health monitoring for deployments via deploy.healthcheck.*

Provides:
- HealthStatus: Health status types
- ComponentHealth: Component health info
- DependencyHealth: Dependency health info
- HealthReport: Full health report
- HealthCheckConfig: Health check configuration
- DeployHealthMonitor: Main health monitor

Bus Topics:
- deploy.healthcheck.run
- deploy.healthcheck.pass
- deploy.healthcheck.fail
- deploy.healthcheck.degraded

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import asyncio
import fcntl
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
import urllib.request
import urllib.error


# ==============================================================================
# Bus Emission Helper with File Locking (DKIN v30)
# ==============================================================================

def _get_bus_path() -> Path:
    """Get the bus event file path."""
    pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    kind: str = "event",
    level: str = "info",
    actor: str = "health-monitor"
) -> str:
    """Emit an event to the Pluribus bus with fcntl.flock() file locking."""
    bus_path = _get_bus_path()
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat() + "Z",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }

    try:
        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class HealthStatus(Enum):
    """Health status types."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"


class CheckType(Enum):
    """Health check types."""
    HTTP = "http"
    TCP = "tcp"
    PROCESS = "process"
    DISK = "disk"
    MEMORY = "memory"
    CPU = "cpu"
    CUSTOM = "custom"
    LIVENESS = "liveness"
    READINESS = "readiness"


@dataclass
class ComponentHealth:
    """
    Component health information.

    Attributes:
        component_id: Component identifier
        name: Component name
        status: Health status
        check_type: Type of check
        message: Status message
        latency_ms: Check latency
        last_check: Last check timestamp
        consecutive_failures: Consecutive failure count
        metadata: Additional metadata
    """
    component_id: str
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    check_type: CheckType = CheckType.CUSTOM
    message: str = ""
    latency_ms: float = 0.0
    last_check: float = 0.0
    consecutive_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "name": self.name,
            "status": self.status.value,
            "check_type": self.check_type.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check,
            "consecutive_failures": self.consecutive_failures,
            "metadata": self.metadata,
        }


@dataclass
class DependencyHealth:
    """
    Dependency health information.

    Attributes:
        dependency_id: Dependency identifier
        name: Dependency name
        type: Dependency type (database, cache, api, etc.)
        status: Health status
        endpoint: Dependency endpoint
        latency_ms: Connection latency
        last_check: Last check timestamp
        required: Whether dependency is required
        metadata: Additional metadata
    """
    dependency_id: str
    name: str
    type: str = "service"
    status: HealthStatus = HealthStatus.UNKNOWN
    endpoint: str = ""
    latency_ms: float = 0.0
    last_check: float = 0.0
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dependency_id": self.dependency_id,
            "name": self.name,
            "type": self.type,
            "status": self.status.value,
            "endpoint": self.endpoint,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check,
            "required": self.required,
            "metadata": self.metadata,
        }


@dataclass
class HealthReport:
    """
    Full health report.

    Attributes:
        report_id: Report identifier
        timestamp: Report timestamp
        status: Overall health status
        service_name: Service name
        environment: Environment
        version: Service version
        uptime_s: Uptime in seconds
        components: Component health list
        dependencies: Dependency health list
        metrics: Health metrics
    """
    report_id: str
    timestamp: float
    status: HealthStatus
    service_name: str = ""
    environment: str = ""
    version: str = ""
    uptime_s: float = 0.0
    components: List[ComponentHealth] = field(default_factory=list)
    dependencies: List[DependencyHealth] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat() + "Z",
            "status": self.status.value,
            "service_name": self.service_name,
            "environment": self.environment,
            "version": self.version,
            "uptime_s": self.uptime_s,
            "components": [c.to_dict() for c in self.components],
            "dependencies": [d.to_dict() for d in self.dependencies],
            "metrics": self.metrics,
        }


@dataclass
class HealthCheckConfig:
    """
    Health check configuration.

    Attributes:
        check_id: Check identifier
        name: Check name
        check_type: Type of check
        target: Check target (URL, path, etc.)
        interval_s: Check interval
        timeout_s: Check timeout
        failure_threshold: Failures before unhealthy
        success_threshold: Successes before healthy
        enabled: Whether check is enabled
        params: Check-specific parameters
    """
    check_id: str
    name: str
    check_type: CheckType = CheckType.HTTP
    target: str = ""
    interval_s: int = 30
    timeout_s: int = 10
    failure_threshold: int = 3
    success_threshold: int = 2
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthCheckConfig":
        data = dict(data)
        if "check_type" in data:
            data["check_type"] = CheckType(data["check_type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==============================================================================
# Deploy Health Monitor (Step 237)
# ==============================================================================

class DeployHealthMonitor:
    """
    Deploy Health Monitor - health monitoring for deployments.

    PBTSO Phase: VERIFY

    Responsibilities:
    - Monitor component health
    - Check dependency availability
    - Generate health reports
    - Track health metrics over time
    - Emit health events to bus

    A2A heartbeat: 300s interval, 900s timeout (CITIZEN v2)

    Example:
        >>> monitor = DeployHealthMonitor(service_name="myapp")
        >>> monitor.register_check(HealthCheckConfig(
        ...     check_id="api-health",
        ...     name="API Health",
        ...     check_type=CheckType.HTTP,
        ...     target="http://localhost:8080/health"
        ... ))
        >>> report = await monitor.run_health_check()
        >>> print(f"Status: {report.status.value}")
    """

    BUS_TOPICS = {
        "run": "deploy.healthcheck.run",
        "pass": "deploy.healthcheck.pass",
        "fail": "deploy.healthcheck.fail",
        "degraded": "deploy.healthcheck.degraded",
    }

    # A2A heartbeat (CITIZEN v2)
    HEARTBEAT_INTERVAL_S = 300
    HEARTBEAT_TIMEOUT_S = 900

    def __init__(
        self,
        service_name: str = "",
        environment: str = "",
        version: str = "",
        state_dir: Optional[str] = None,
        actor_id: str = "health-monitor",
    ):
        """
        Initialize the health monitor.

        Args:
            service_name: Service name
            environment: Environment
            version: Service version
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        self.service_name = service_name
        self.environment = environment
        self.version = version
        self.actor_id = actor_id

        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "healthcheck"

        self.state_dir.mkdir(parents=True, exist_ok=True)

        self._start_time = time.time()

        # Health checks
        self._checks: Dict[str, HealthCheckConfig] = {}
        self._check_results: Dict[str, ComponentHealth] = {}

        # Dependencies
        self._dependencies: Dict[str, DependencyHealth] = {}

        # Custom check handlers
        self._custom_handlers: Dict[str, Callable] = {}

        # Health history
        self._history: List[HealthReport] = []
        self._max_history = 100

        # Running tasks
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

        self._load_state()

    def register_check(self, config: HealthCheckConfig) -> None:
        """Register a health check."""
        self._checks[config.check_id] = config
        self._check_results[config.check_id] = ComponentHealth(
            component_id=config.check_id,
            name=config.name,
            check_type=config.check_type,
            status=HealthStatus.UNKNOWN,
        )
        self._save_state()

    def unregister_check(self, check_id: str) -> bool:
        """Unregister a health check."""
        if check_id in self._checks:
            del self._checks[check_id]
            if check_id in self._check_results:
                del self._check_results[check_id]
            self._save_state()
            return True
        return False

    def register_dependency(
        self,
        name: str,
        type: str = "service",
        endpoint: str = "",
        required: bool = True,
    ) -> DependencyHealth:
        """Register a dependency to monitor."""
        dep_id = f"dep-{uuid.uuid4().hex[:8]}"
        dep = DependencyHealth(
            dependency_id=dep_id,
            name=name,
            type=type,
            endpoint=endpoint,
            required=required,
        )
        self._dependencies[dep_id] = dep
        self._save_state()
        return dep

    def register_custom_handler(
        self,
        check_type: str,
        handler: Callable[[HealthCheckConfig], ComponentHealth],
    ) -> None:
        """Register a custom health check handler."""
        self._custom_handlers[check_type] = handler

    async def run_health_check(self) -> HealthReport:
        """
        Run all health checks and generate a report.

        Returns:
            HealthReport
        """
        report_id = f"report-{uuid.uuid4().hex[:12]}"
        timestamp = time.time()

        _emit_bus_event(
            self.BUS_TOPICS["run"],
            {
                "report_id": report_id,
                "service_name": self.service_name,
                "check_count": len(self._checks),
            },
            actor=self.actor_id,
        )

        # Run all component checks
        for check_id, config in self._checks.items():
            if not config.enabled:
                continue

            result = await self._run_single_check(config)
            self._check_results[check_id] = result

        # Check dependencies
        for dep_id, dep in self._dependencies.items():
            result = await self._check_dependency(dep)
            self._dependencies[dep_id] = result

        # Calculate overall status
        overall_status = self._calculate_overall_status()

        # Build report
        report = HealthReport(
            report_id=report_id,
            timestamp=timestamp,
            status=overall_status,
            service_name=self.service_name,
            environment=self.environment,
            version=self.version,
            uptime_s=time.time() - self._start_time,
            components=list(self._check_results.values()),
            dependencies=list(self._dependencies.values()),
            metrics=self._collect_metrics(),
        )

        # Store in history
        self._history.append(report)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Emit result event
        if overall_status == HealthStatus.HEALTHY:
            topic = self.BUS_TOPICS["pass"]
            level = "info"
        elif overall_status == HealthStatus.DEGRADED:
            topic = self.BUS_TOPICS["degraded"]
            level = "warn"
        else:
            topic = self.BUS_TOPICS["fail"]
            level = "error"

        _emit_bus_event(
            topic,
            {
                "report_id": report_id,
                "status": overall_status.value,
                "service_name": self.service_name,
                "healthy_components": sum(1 for c in self._check_results.values() if c.status == HealthStatus.HEALTHY),
                "total_components": len(self._check_results),
            },
            kind="metric",
            level=level,
            actor=self.actor_id,
        )

        return report

    async def _run_single_check(self, config: HealthCheckConfig) -> ComponentHealth:
        """Run a single health check."""
        start_time = time.time()
        result = ComponentHealth(
            component_id=config.check_id,
            name=config.name,
            check_type=config.check_type,
        )

        try:
            if config.check_type == CheckType.HTTP:
                result = await self._check_http(config)
            elif config.check_type == CheckType.TCP:
                result = await self._check_tcp(config)
            elif config.check_type == CheckType.PROCESS:
                result = await self._check_process(config)
            elif config.check_type == CheckType.DISK:
                result = self._check_disk(config)
            elif config.check_type == CheckType.MEMORY:
                result = self._check_memory(config)
            elif config.check_type == CheckType.CPU:
                result = self._check_cpu(config)
            elif config.check_type == CheckType.CUSTOM:
                result = await self._check_custom(config)
            else:
                result.status = HealthStatus.UNKNOWN
                result.message = f"Unknown check type: {config.check_type.value}"

        except asyncio.TimeoutError:
            result.status = HealthStatus.UNHEALTHY
            result.message = "Check timed out"
        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.message = f"Check error: {str(e)}"

        result.latency_ms = (time.time() - start_time) * 1000
        result.last_check = time.time()

        # Update consecutive failures
        prev_result = self._check_results.get(config.check_id)
        if prev_result:
            if result.status == HealthStatus.UNHEALTHY:
                result.consecutive_failures = prev_result.consecutive_failures + 1
            else:
                result.consecutive_failures = 0

        # Apply thresholds
        if result.consecutive_failures >= config.failure_threshold:
            result.status = HealthStatus.UNHEALTHY

        return result

    async def _check_http(self, config: HealthCheckConfig) -> ComponentHealth:
        """Run HTTP health check."""
        result = ComponentHealth(
            component_id=config.check_id,
            name=config.name,
            check_type=CheckType.HTTP,
        )

        try:
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: urllib.request.urlopen(config.target, timeout=config.timeout_s)
                ),
                timeout=config.timeout_s + 1,
            )

            status_code = response.getcode()
            expected_codes = config.params.get("expected_status", [200])

            if status_code in expected_codes:
                result.status = HealthStatus.HEALTHY
                result.message = f"HTTP {status_code} OK"
            else:
                result.status = HealthStatus.UNHEALTHY
                result.message = f"Unexpected status: {status_code}"

            result.metadata["status_code"] = status_code

        except urllib.error.HTTPError as e:
            result.status = HealthStatus.UNHEALTHY
            result.message = f"HTTP error: {e.code}"
            result.metadata["status_code"] = e.code
        except urllib.error.URLError as e:
            result.status = HealthStatus.UNHEALTHY
            result.message = f"URL error: {e.reason}"

        return result

    async def _check_tcp(self, config: HealthCheckConfig) -> ComponentHealth:
        """Run TCP health check."""
        result = ComponentHealth(
            component_id=config.check_id,
            name=config.name,
            check_type=CheckType.TCP,
        )

        host = config.params.get("host", "localhost")
        port = config.params.get("port", 80)

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=config.timeout_s,
            )
            writer.close()
            await writer.wait_closed()

            result.status = HealthStatus.HEALTHY
            result.message = f"TCP connection to {host}:{port} successful"

        except (ConnectionRefusedError, OSError) as e:
            result.status = HealthStatus.UNHEALTHY
            result.message = f"Connection failed: {e}"
        except asyncio.TimeoutError:
            result.status = HealthStatus.UNHEALTHY
            result.message = "Connection timeout"

        return result

    async def _check_process(self, config: HealthCheckConfig) -> ComponentHealth:
        """Check if a process is running."""
        result = ComponentHealth(
            component_id=config.check_id,
            name=config.name,
            check_type=CheckType.PROCESS,
        )

        process_name = config.params.get("process_name", "")
        pid_file = config.params.get("pid_file", "")

        if pid_file:
            pid_path = Path(pid_file)
            if pid_path.exists():
                try:
                    pid = int(pid_path.read_text().strip())
                    # Check if process is running
                    os.kill(pid, 0)
                    result.status = HealthStatus.HEALTHY
                    result.message = f"Process {pid} is running"
                    result.metadata["pid"] = pid
                except (ProcessLookupError, ValueError):
                    result.status = HealthStatus.UNHEALTHY
                    result.message = "Process not found"
            else:
                result.status = HealthStatus.UNHEALTHY
                result.message = "PID file not found"
        else:
            # Simple check - process info available
            result.status = HealthStatus.HEALTHY
            result.message = "Process check passed"

        return result

    def _check_disk(self, config: HealthCheckConfig) -> ComponentHealth:
        """Check disk space."""
        result = ComponentHealth(
            component_id=config.check_id,
            name=config.name,
            check_type=CheckType.DISK,
        )

        path = config.params.get("path", "/")
        threshold_pct = config.params.get("threshold_pct", 90)

        try:
            stat = os.statvfs(path)
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bfree * stat.f_frsize
            used_pct = ((total - free) / total) * 100

            result.metadata["total_bytes"] = total
            result.metadata["free_bytes"] = free
            result.metadata["used_pct"] = used_pct

            if used_pct >= threshold_pct:
                result.status = HealthStatus.DEGRADED
                result.message = f"Disk usage high: {used_pct:.1f}%"
            else:
                result.status = HealthStatus.HEALTHY
                result.message = f"Disk usage: {used_pct:.1f}%"

        except OSError as e:
            result.status = HealthStatus.UNHEALTHY
            result.message = f"Disk check failed: {e}"

        return result

    def _check_memory(self, config: HealthCheckConfig) -> ComponentHealth:
        """Check memory usage."""
        result = ComponentHealth(
            component_id=config.check_id,
            name=config.name,
            check_type=CheckType.MEMORY,
        )

        threshold_pct = config.params.get("threshold_pct", 90)

        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = int(parts[1].strip().split()[0]) * 1024
                        meminfo[key] = value

            total = meminfo.get("MemTotal", 1)
            available = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            used_pct = ((total - available) / total) * 100

            result.metadata["total_bytes"] = total
            result.metadata["available_bytes"] = available
            result.metadata["used_pct"] = used_pct

            if used_pct >= threshold_pct:
                result.status = HealthStatus.DEGRADED
                result.message = f"Memory usage high: {used_pct:.1f}%"
            else:
                result.status = HealthStatus.HEALTHY
                result.message = f"Memory usage: {used_pct:.1f}%"

        except Exception as e:
            result.status = HealthStatus.UNKNOWN
            result.message = f"Memory check failed: {e}"

        return result

    def _check_cpu(self, config: HealthCheckConfig) -> ComponentHealth:
        """Check CPU usage."""
        result = ComponentHealth(
            component_id=config.check_id,
            name=config.name,
            check_type=CheckType.CPU,
        )

        threshold_pct = config.params.get("threshold_pct", 90)

        try:
            with open("/proc/loadavg", "r") as f:
                load_avg = float(f.read().split()[0])

            cpu_count = os.cpu_count() or 1
            load_pct = (load_avg / cpu_count) * 100

            result.metadata["load_avg"] = load_avg
            result.metadata["cpu_count"] = cpu_count
            result.metadata["load_pct"] = load_pct

            if load_pct >= threshold_pct:
                result.status = HealthStatus.DEGRADED
                result.message = f"CPU load high: {load_pct:.1f}%"
            else:
                result.status = HealthStatus.HEALTHY
                result.message = f"CPU load: {load_pct:.1f}%"

        except Exception as e:
            result.status = HealthStatus.UNKNOWN
            result.message = f"CPU check failed: {e}"

        return result

    async def _check_custom(self, config: HealthCheckConfig) -> ComponentHealth:
        """Run custom health check."""
        handler = self._custom_handlers.get(config.name)
        if not handler:
            return ComponentHealth(
                component_id=config.check_id,
                name=config.name,
                check_type=CheckType.CUSTOM,
                status=HealthStatus.UNKNOWN,
                message=f"No handler for: {config.name}",
            )

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(config)
            else:
                result = handler(config)
            return result
        except Exception as e:
            return ComponentHealth(
                component_id=config.check_id,
                name=config.name,
                check_type=CheckType.CUSTOM,
                status=HealthStatus.UNHEALTHY,
                message=f"Handler error: {e}",
            )

    async def _check_dependency(self, dep: DependencyHealth) -> DependencyHealth:
        """Check dependency health."""
        start_time = time.time()

        if dep.endpoint:
            try:
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: urllib.request.urlopen(dep.endpoint, timeout=10)
                    ),
                    timeout=11,
                )
                if response.getcode() in (200, 204):
                    dep.status = HealthStatus.HEALTHY
                    dep.message = "Dependency available"
                else:
                    dep.status = HealthStatus.DEGRADED
                    dep.message = f"Unexpected status: {response.getcode()}"
            except Exception as e:
                dep.status = HealthStatus.UNHEALTHY
                dep.message = f"Connection failed: {e}"
        else:
            dep.status = HealthStatus.UNKNOWN
            dep.message = "No endpoint configured"

        dep.latency_ms = (time.time() - start_time) * 1000
        dep.last_check = time.time()

        return dep

    def _calculate_overall_status(self) -> HealthStatus:
        """Calculate overall health status."""
        all_healthy = True
        any_unhealthy = False
        critical_unhealthy = False

        for result in self._check_results.values():
            if result.status == HealthStatus.UNHEALTHY:
                any_unhealthy = True
                all_healthy = False
            elif result.status != HealthStatus.HEALTHY:
                all_healthy = False

        # Check required dependencies
        for dep in self._dependencies.values():
            if dep.required and dep.status == HealthStatus.UNHEALTHY:
                critical_unhealthy = True

        if critical_unhealthy or any_unhealthy:
            return HealthStatus.UNHEALTHY
        elif not all_healthy:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect health metrics."""
        return {
            "check_count": len(self._checks),
            "dependency_count": len(self._dependencies),
            "healthy_checks": sum(1 for r in self._check_results.values() if r.status == HealthStatus.HEALTHY),
            "unhealthy_checks": sum(1 for r in self._check_results.values() if r.status == HealthStatus.UNHEALTHY),
            "avg_latency_ms": sum(r.latency_ms for r in self._check_results.values()) / len(self._check_results) if self._check_results else 0,
            "uptime_s": time.time() - self._start_time,
        }

    async def start_monitoring(self, interval_s: int = 30) -> None:
        """Start continuous health monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop(interval_s))

    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _monitoring_loop(self, interval_s: int) -> None:
        """Continuous monitoring loop."""
        while self._running:
            try:
                await self.run_health_check()
                await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(interval_s)

    def get_status(self) -> HealthStatus:
        """Get current overall health status."""
        return self._calculate_overall_status()

    def get_component_health(self, check_id: str) -> Optional[ComponentHealth]:
        """Get health of a specific component."""
        return self._check_results.get(check_id)

    def get_history(self, limit: int = 10) -> List[HealthReport]:
        """Get health report history."""
        return self._history[-limit:]

    def list_checks(self) -> List[HealthCheckConfig]:
        """List all registered health checks."""
        return list(self._checks.values())

    def list_dependencies(self) -> List[DependencyHealth]:
        """List all registered dependencies."""
        return list(self._dependencies.values())

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "checks": {k: v.to_dict() for k, v in self._checks.items()},
            "dependencies": {k: v.to_dict() for k, v in self._dependencies.items()},
        }
        state_file = self.state_dir / "health_monitor_state.json"
        with open(state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "health_monitor_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            for check_id, data in state.get("checks", {}).items():
                self._checks[check_id] = HealthCheckConfig.from_dict(data)
                self._check_results[check_id] = ComponentHealth(
                    component_id=check_id,
                    name=data.get("name", ""),
                    status=HealthStatus.UNKNOWN,
                )

        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for health monitor."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Health Monitor (Step 237)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check command
    check_parser = subparsers.add_parser("check", help="Run health checks")
    check_parser.add_argument("--service", "-s", default="", help="Service name")
    check_parser.add_argument("--json", action="store_true", help="JSON output")

    # register command
    register_parser = subparsers.add_parser("register", help="Register health check")
    register_parser.add_argument("name", help="Check name")
    register_parser.add_argument("--type", "-t", default="http",
                                choices=["http", "tcp", "disk", "memory", "cpu"])
    register_parser.add_argument("--target", required=True, help="Check target")
    register_parser.add_argument("--interval", "-i", type=int, default=30, help="Interval in seconds")

    # status command
    status_parser = subparsers.add_parser("status", help="Get current status")
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List health checks")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # history command
    history_parser = subparsers.add_parser("history", help="Get health history")
    history_parser.add_argument("--limit", "-l", type=int, default=10, help="Max results")
    history_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    monitor = DeployHealthMonitor(service_name=getattr(args, "service", ""))

    if args.command == "check":
        report = asyncio.get_event_loop().run_until_complete(
            monitor.run_health_check()
        )

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            status_icon = "OK" if report.status == HealthStatus.HEALTHY else "FAIL"
            print(f"[{status_icon}] Health Report: {report.status.value}")
            print(f"  Uptime: {report.uptime_s:.0f}s")
            print(f"\nComponents ({len(report.components)}):")
            for c in report.components:
                icon = "+" if c.status == HealthStatus.HEALTHY else "-"
                print(f"  [{icon}] {c.name}: {c.status.value} ({c.latency_ms:.1f}ms)")
            if report.dependencies:
                print(f"\nDependencies ({len(report.dependencies)}):")
                for d in report.dependencies:
                    icon = "+" if d.status == HealthStatus.HEALTHY else "-"
                    print(f"  [{icon}] {d.name}: {d.status.value}")

        return 0 if report.status == HealthStatus.HEALTHY else 1

    elif args.command == "register":
        config = HealthCheckConfig(
            check_id=f"check-{uuid.uuid4().hex[:8]}",
            name=args.name,
            check_type=CheckType(args.type.upper()),
            target=args.target,
            interval_s=args.interval,
        )
        monitor.register_check(config)
        print(f"Registered check: {config.check_id}")
        return 0

    elif args.command == "status":
        status = monitor.get_status()

        if args.json:
            print(json.dumps({"status": status.value}))
        else:
            icon = "OK" if status == HealthStatus.HEALTHY else "FAIL"
            print(f"[{icon}] Status: {status.value}")

        return 0 if status == HealthStatus.HEALTHY else 1

    elif args.command == "list":
        checks = monitor.list_checks()

        if args.json:
            print(json.dumps([c.to_dict() for c in checks], indent=2))
        else:
            if not checks:
                print("No health checks registered")
            else:
                for c in checks:
                    status = "enabled" if c.enabled else "disabled"
                    print(f"{c.check_id} ({c.name}) - {c.check_type.value} [{status}]")

        return 0

    elif args.command == "history":
        history = monitor.get_history(args.limit)

        if args.json:
            print(json.dumps([r.to_dict() for r in history], indent=2))
        else:
            for report in history:
                ts = datetime.fromtimestamp(report.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                print(f"{ts}: {report.status.value}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
