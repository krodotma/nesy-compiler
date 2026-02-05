#!/usr/bin/env python3
"""
checker.py - Health Checker (Step 213)

PBTSO Phase: VERIFY
A2A Integration: Validates deployment health via deploy.health.check

Provides:
- HealthCheckType: Types of health checks
- HealthStatus: Health status enum
- HealthCheckResult: Result of a health check
- HealthCheckConfig: Health check configuration
- HealthChecker: Deployment health validation

Bus Topics:
- deploy.health.check
- deploy.health.passed
- deploy.health.failed
- deploy.health.degraded

Protocol: DKIN v30, CITIZEN v2
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
import urllib.request
import urllib.error


# ==============================================================================
# Bus Emission Helper
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
    actor: str = "health-checker"
) -> str:
    """Emit an event to the Pluribus bus."""
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
            f.write(json.dumps(event) + "\n")
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class HealthCheckType(Enum):
    """Types of health checks."""
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    GRPC = "grpc"
    EXEC = "exec"
    DNS = "dns"
    CUSTOM = "custom"


class HealthStatus(Enum):
    """Health status enum."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    PENDING = "pending"


@dataclass
class HealthCheckConfig:
    """
    Configuration for a health check.

    Attributes:
        check_id: Unique check identifier
        name: Human-readable check name
        check_type: Type of health check
        target: Target endpoint/address
        interval_s: Check interval in seconds
        timeout_s: Check timeout in seconds
        success_threshold: Consecutive successes to be healthy
        failure_threshold: Consecutive failures to be unhealthy
        method: HTTP method for HTTP checks
        path: HTTP path for HTTP checks
        expected_status: Expected HTTP status codes
        expected_body: Expected body content (substring match)
        headers: HTTP headers
        port: Port for TCP/gRPC checks
        command: Command for exec checks
    """
    check_id: str
    name: str
    check_type: HealthCheckType = HealthCheckType.HTTP
    target: str = ""
    interval_s: int = 30
    timeout_s: int = 10
    success_threshold: int = 2
    failure_threshold: int = 3
    method: str = "GET"
    path: str = "/health"
    expected_status: List[int] = field(default_factory=lambda: [200])
    expected_body: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    port: int = 80
    command: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "check_type": self.check_type.value,
            "target": self.target,
            "interval_s": self.interval_s,
            "timeout_s": self.timeout_s,
            "success_threshold": self.success_threshold,
            "failure_threshold": self.failure_threshold,
            "method": self.method,
            "path": self.path,
            "expected_status": self.expected_status,
            "expected_body": self.expected_body,
            "headers": self.headers,
            "port": self.port,
            "command": self.command,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthCheckConfig":
        data = dict(data)
        if "check_type" in data:
            data["check_type"] = HealthCheckType(data["check_type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class HealthCheckResult:
    """
    Result of a health check.

    Attributes:
        check_id: Check identifier
        status: Health status
        latency_ms: Check latency in milliseconds
        message: Status message
        details: Additional details
        timestamp: Check timestamp
        consecutive_successes: Consecutive successful checks
        consecutive_failures: Consecutive failed checks
    """
    check_id: str
    status: HealthStatus = HealthStatus.UNKNOWN
    latency_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    consecutive_successes: int = 0
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "consecutive_successes": self.consecutive_successes,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class ServiceHealth:
    """
    Aggregated health status for a service.

    Attributes:
        service_name: Service name
        status: Overall health status
        checks: Individual check results
        last_check_ts: Last check timestamp
        uptime_pct: Uptime percentage (24h rolling)
        healthy_count: Number of healthy checks
        unhealthy_count: Number of unhealthy checks
    """
    service_name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    checks: List[HealthCheckResult] = field(default_factory=list)
    last_check_ts: float = 0.0
    uptime_pct: float = 100.0
    healthy_count: int = 0
    unhealthy_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "status": self.status.value,
            "checks": [c.to_dict() for c in self.checks],
            "last_check_ts": self.last_check_ts,
            "uptime_pct": self.uptime_pct,
            "healthy_count": self.healthy_count,
            "unhealthy_count": self.unhealthy_count,
        }


# ==============================================================================
# Health Checker (Step 213)
# ==============================================================================

class HealthChecker:
    """
    Health Checker - validates deployment health.

    PBTSO Phase: VERIFY

    Responsibilities:
    - Configure health checks for services
    - Execute health checks periodically
    - Track health status history
    - Calculate uptime metrics
    - Emit health events to bus

    Example:
        >>> checker = HealthChecker()
        >>> config = HealthCheckConfig(
        ...     check_id="api-health",
        ...     name="API Health",
        ...     check_type=HealthCheckType.HTTP,
        ...     target="http://localhost:8080",
        ...     path="/health"
        ... )
        >>> checker.register_check("api-service", config)
        >>> result = await checker.run_check("api-health")
        >>> print(f"Status: {result.status.value}")
    """

    BUS_TOPICS = {
        "check": "deploy.health.check",
        "passed": "deploy.health.passed",
        "failed": "deploy.health.failed",
        "degraded": "deploy.health.degraded",
        "registered": "deploy.health.registered",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "health-checker",
    ):
        """
        Initialize the health checker.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "health"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Check configurations by check_id
        self._checks: Dict[str, HealthCheckConfig] = {}
        # Map of service_name -> [check_ids]
        self._service_checks: Dict[str, List[str]] = {}
        # Current results by check_id
        self._results: Dict[str, HealthCheckResult] = {}
        # History by check_id
        self._history: Dict[str, List[HealthCheckResult]] = {}
        # Custom check handlers
        self._custom_handlers: Dict[str, Callable] = {}
        # Running check loops
        self._running_loops: Dict[str, asyncio.Task] = {}

        self._load_state()

    def register_check(
        self,
        service_name: str,
        config: HealthCheckConfig,
    ) -> None:
        """
        Register a health check for a service.

        Args:
            service_name: Service name
            config: Health check configuration
        """
        self._checks[config.check_id] = config

        if service_name not in self._service_checks:
            self._service_checks[service_name] = []
        if config.check_id not in self._service_checks[service_name]:
            self._service_checks[service_name].append(config.check_id)

        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["registered"],
            {
                "check_id": config.check_id,
                "service_name": service_name,
                "check_type": config.check_type.value,
                "target": config.target,
            },
            actor=self.actor_id,
        )

    def unregister_check(self, check_id: str) -> bool:
        """Unregister a health check."""
        if check_id not in self._checks:
            return False

        # Stop running loop if any
        if check_id in self._running_loops:
            self._running_loops[check_id].cancel()
            del self._running_loops[check_id]

        del self._checks[check_id]

        # Remove from service mappings
        for service, checks in self._service_checks.items():
            if check_id in checks:
                checks.remove(check_id)

        self._save_state()
        return True

    def register_custom_handler(
        self,
        check_type: str,
        handler: Callable[[HealthCheckConfig], HealthCheckResult],
    ) -> None:
        """Register a custom health check handler."""
        self._custom_handlers[check_type] = handler

    async def run_check(self, check_id: str) -> HealthCheckResult:
        """
        Run a single health check.

        Args:
            check_id: Check ID to run

        Returns:
            HealthCheckResult
        """
        config = self._checks.get(check_id)
        if not config:
            return HealthCheckResult(
                check_id=check_id,
                status=HealthStatus.UNKNOWN,
                message="Check not found",
            )

        start_time = time.time()

        try:
            if config.check_type == HealthCheckType.HTTP:
                result = await self._run_http_check(config)
            elif config.check_type == HealthCheckType.HTTPS:
                result = await self._run_http_check(config, https=True)
            elif config.check_type == HealthCheckType.TCP:
                result = await self._run_tcp_check(config)
            elif config.check_type == HealthCheckType.DNS:
                result = await self._run_dns_check(config)
            elif config.check_type == HealthCheckType.EXEC:
                result = await self._run_exec_check(config)
            elif config.check_type == HealthCheckType.CUSTOM:
                result = await self._run_custom_check(config)
            else:
                result = HealthCheckResult(
                    check_id=check_id,
                    status=HealthStatus.UNKNOWN,
                    message=f"Unsupported check type: {config.check_type.value}",
                )
        except Exception as e:
            result = HealthCheckResult(
                check_id=check_id,
                status=HealthStatus.UNHEALTHY,
                message=f"Check error: {str(e)}",
            )

        result.latency_ms = (time.time() - start_time) * 1000
        result.timestamp = time.time()

        # Update consecutive counters
        prev_result = self._results.get(check_id)
        if prev_result:
            if result.status == HealthStatus.HEALTHY:
                result.consecutive_successes = prev_result.consecutive_successes + 1
                result.consecutive_failures = 0
            else:
                result.consecutive_failures = prev_result.consecutive_failures + 1
                result.consecutive_successes = 0
        else:
            result.consecutive_successes = 1 if result.status == HealthStatus.HEALTHY else 0
            result.consecutive_failures = 0 if result.status == HealthStatus.HEALTHY else 1

        # Apply thresholds
        if result.consecutive_successes >= config.success_threshold:
            result.status = HealthStatus.HEALTHY
        elif result.consecutive_failures >= config.failure_threshold:
            result.status = HealthStatus.UNHEALTHY

        # Store result
        self._results[check_id] = result
        if check_id not in self._history:
            self._history[check_id] = []
        self._history[check_id].append(result)
        # Keep last 1000 results
        self._history[check_id] = self._history[check_id][-1000:]

        # Emit events
        self._emit_check_event(config, result)

        return result

    async def _run_http_check(
        self,
        config: HealthCheckConfig,
        https: bool = False,
    ) -> HealthCheckResult:
        """Run an HTTP/HTTPS health check."""
        protocol = "https" if https else "http"
        url = f"{protocol}://{config.target}{config.path}"
        if not config.target.startswith(("http://", "https://")):
            url = f"{protocol}://{config.target}{config.path}"
        else:
            url = f"{config.target}{config.path}"

        try:
            request = urllib.request.Request(url, method=config.method)
            for key, value in config.headers.items():
                request.add_header(key, value)

            # Use asyncio to avoid blocking
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: urllib.request.urlopen(request, timeout=config.timeout_s)
                ),
                timeout=config.timeout_s + 1
            )

            status_code = response.getcode()
            body = response.read().decode("utf-8", errors="ignore")

            if status_code not in config.expected_status:
                return HealthCheckResult(
                    check_id=config.check_id,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Unexpected status: {status_code}",
                    details={"status_code": status_code},
                )

            if config.expected_body and config.expected_body not in body:
                return HealthCheckResult(
                    check_id=config.check_id,
                    status=HealthStatus.UNHEALTHY,
                    message="Response body mismatch",
                    details={"status_code": status_code},
                )

            return HealthCheckResult(
                check_id=config.check_id,
                status=HealthStatus.HEALTHY,
                message="OK",
                details={"status_code": status_code},
            )

        except urllib.error.HTTPError as e:
            return HealthCheckResult(
                check_id=config.check_id,
                status=HealthStatus.UNHEALTHY,
                message=f"HTTP error: {e.code}",
                details={"status_code": e.code},
            )
        except urllib.error.URLError as e:
            return HealthCheckResult(
                check_id=config.check_id,
                status=HealthStatus.UNHEALTHY,
                message=f"URL error: {e.reason}",
            )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                check_id=config.check_id,
                status=HealthStatus.UNHEALTHY,
                message="Timeout",
            )

    async def _run_tcp_check(self, config: HealthCheckConfig) -> HealthCheckResult:
        """Run a TCP health check."""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(config.target, config.port),
                timeout=config.timeout_s
            )
            writer.close()
            await writer.wait_closed()

            return HealthCheckResult(
                check_id=config.check_id,
                status=HealthStatus.HEALTHY,
                message="TCP connection successful",
                details={"port": config.port},
            )
        except (ConnectionRefusedError, OSError) as e:
            return HealthCheckResult(
                check_id=config.check_id,
                status=HealthStatus.UNHEALTHY,
                message=f"Connection failed: {e}",
            )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                check_id=config.check_id,
                status=HealthStatus.UNHEALTHY,
                message="Timeout",
            )

    async def _run_dns_check(self, config: HealthCheckConfig) -> HealthCheckResult:
        """Run a DNS health check."""
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, socket.gethostbyname, config.target),
                timeout=config.timeout_s
            )

            return HealthCheckResult(
                check_id=config.check_id,
                status=HealthStatus.HEALTHY,
                message="DNS resolution successful",
                details={"resolved_ip": result},
            )
        except socket.gaierror as e:
            return HealthCheckResult(
                check_id=config.check_id,
                status=HealthStatus.UNHEALTHY,
                message=f"DNS resolution failed: {e}",
            )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                check_id=config.check_id,
                status=HealthStatus.UNHEALTHY,
                message="DNS timeout",
            )

    async def _run_exec_check(self, config: HealthCheckConfig) -> HealthCheckResult:
        """Run an exec health check."""
        if not config.command:
            return HealthCheckResult(
                check_id=config.check_id,
                status=HealthStatus.UNKNOWN,
                message="No command specified",
            )

        try:
            process = await asyncio.create_subprocess_exec(
                *config.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout_s
            )

            if process.returncode == 0:
                return HealthCheckResult(
                    check_id=config.check_id,
                    status=HealthStatus.HEALTHY,
                    message="Command succeeded",
                    details={
                        "exit_code": process.returncode,
                        "stdout": stdout.decode()[:500],
                    },
                )
            else:
                return HealthCheckResult(
                    check_id=config.check_id,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Command failed with exit code {process.returncode}",
                    details={
                        "exit_code": process.returncode,
                        "stderr": stderr.decode()[:500],
                    },
                )
        except asyncio.TimeoutError:
            return HealthCheckResult(
                check_id=config.check_id,
                status=HealthStatus.UNHEALTHY,
                message="Command timeout",
            )

    async def _run_custom_check(self, config: HealthCheckConfig) -> HealthCheckResult:
        """Run a custom health check."""
        handler = self._custom_handlers.get(config.name)
        if not handler:
            return HealthCheckResult(
                check_id=config.check_id,
                status=HealthStatus.UNKNOWN,
                message=f"No handler registered for: {config.name}",
            )

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(config)
            else:
                result = handler(config)
            return result
        except Exception as e:
            return HealthCheckResult(
                check_id=config.check_id,
                status=HealthStatus.UNHEALTHY,
                message=f"Custom check error: {e}",
            )

    def _emit_check_event(self, config: HealthCheckConfig, result: HealthCheckResult) -> None:
        """Emit health check event."""
        topic = self.BUS_TOPICS["check"]
        level = "info"

        if result.status == HealthStatus.HEALTHY:
            topic = self.BUS_TOPICS["passed"]
        elif result.status == HealthStatus.UNHEALTHY:
            topic = self.BUS_TOPICS["failed"]
            level = "error"
        elif result.status == HealthStatus.DEGRADED:
            topic = self.BUS_TOPICS["degraded"]
            level = "warn"

        _emit_bus_event(
            topic,
            {
                "check_id": config.check_id,
                "name": config.name,
                "status": result.status.value,
                "latency_ms": result.latency_ms,
                "message": result.message,
            },
            kind="metric",
            level=level,
            actor=self.actor_id,
        )

    async def run_all_checks(self, service_name: Optional[str] = None) -> List[HealthCheckResult]:
        """Run all registered health checks."""
        check_ids = []

        if service_name:
            check_ids = self._service_checks.get(service_name, [])
        else:
            check_ids = list(self._checks.keys())

        results = await asyncio.gather(*[self.run_check(cid) for cid in check_ids])
        return list(results)

    def get_service_health(self, service_name: str) -> ServiceHealth:
        """Get aggregated health status for a service."""
        check_ids = self._service_checks.get(service_name, [])
        checks = [self._results.get(cid) for cid in check_ids if cid in self._results]
        checks = [c for c in checks if c is not None]

        healthy_count = sum(1 for c in checks if c.status == HealthStatus.HEALTHY)
        unhealthy_count = sum(1 for c in checks if c.status == HealthStatus.UNHEALTHY)

        # Determine overall status
        if not checks:
            status = HealthStatus.UNKNOWN
        elif unhealthy_count > 0:
            if healthy_count > 0:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY
        else:
            status = HealthStatus.HEALTHY

        # Calculate uptime from history
        uptime_pct = self._calculate_uptime(service_name)

        return ServiceHealth(
            service_name=service_name,
            status=status,
            checks=checks,
            last_check_ts=max((c.timestamp for c in checks), default=0),
            uptime_pct=uptime_pct,
            healthy_count=healthy_count,
            unhealthy_count=unhealthy_count,
        )

    def _calculate_uptime(self, service_name: str) -> float:
        """Calculate uptime percentage for a service (24h rolling)."""
        check_ids = self._service_checks.get(service_name, [])
        if not check_ids:
            return 100.0

        day_ago = time.time() - 86400
        total = 0
        healthy = 0

        for check_id in check_ids:
            for result in self._history.get(check_id, []):
                if result.timestamp >= day_ago:
                    total += 1
                    if result.status == HealthStatus.HEALTHY:
                        healthy += 1

        if total == 0:
            return 100.0

        return (healthy / total) * 100

    async def start_continuous_checks(self, service_name: Optional[str] = None) -> None:
        """Start continuous health checking loops."""
        check_ids = []

        if service_name:
            check_ids = self._service_checks.get(service_name, [])
        else:
            check_ids = list(self._checks.keys())

        for check_id in check_ids:
            if check_id not in self._running_loops:
                task = asyncio.create_task(self._check_loop(check_id))
                self._running_loops[check_id] = task

    async def stop_continuous_checks(self, service_name: Optional[str] = None) -> None:
        """Stop continuous health checking loops."""
        check_ids = []

        if service_name:
            check_ids = self._service_checks.get(service_name, [])
        else:
            check_ids = list(self._running_loops.keys())

        for check_id in check_ids:
            if check_id in self._running_loops:
                self._running_loops[check_id].cancel()
                del self._running_loops[check_id]

    async def _check_loop(self, check_id: str) -> None:
        """Continuous check loop for a single check."""
        config = self._checks.get(check_id)
        if not config:
            return

        while True:
            try:
                await self.run_check(check_id)
                await asyncio.sleep(config.interval_s)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(config.interval_s)

    def get_check(self, check_id: str) -> Optional[HealthCheckConfig]:
        """Get a check configuration."""
        return self._checks.get(check_id)

    def get_result(self, check_id: str) -> Optional[HealthCheckResult]:
        """Get the latest result for a check."""
        return self._results.get(check_id)

    def get_history(self, check_id: str, limit: int = 100) -> List[HealthCheckResult]:
        """Get check history."""
        return self._history.get(check_id, [])[-limit:]

    def list_checks(self, service_name: Optional[str] = None) -> List[HealthCheckConfig]:
        """List all registered checks."""
        if service_name:
            check_ids = self._service_checks.get(service_name, [])
            return [self._checks[cid] for cid in check_ids if cid in self._checks]
        return list(self._checks.values())

    def list_services(self) -> List[str]:
        """List all services with registered checks."""
        return list(self._service_checks.keys())

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "checks": {cid: c.to_dict() for cid, c in self._checks.items()},
            "service_checks": self._service_checks,
        }
        state_file = self.state_dir / "health_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "health_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            for cid, data in state.get("checks", {}).items():
                self._checks[cid] = HealthCheckConfig.from_dict(data)

            self._service_checks = state.get("service_checks", {})
        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for health checker."""
    import argparse

    parser = argparse.ArgumentParser(description="Health Checker (Step 213)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # register command
    register_parser = subparsers.add_parser("register", help="Register a health check")
    register_parser.add_argument("service_name", help="Service name")
    register_parser.add_argument("--name", "-n", required=True, help="Check name")
    register_parser.add_argument("--type", "-t", default="http",
                                choices=["http", "https", "tcp", "dns", "exec"])
    register_parser.add_argument("--target", required=True, help="Target address")
    register_parser.add_argument("--path", "-p", default="/health", help="HTTP path")
    register_parser.add_argument("--port", type=int, default=80, help="Port for TCP")
    register_parser.add_argument("--interval", "-i", type=int, default=30, help="Interval in seconds")
    register_parser.add_argument("--json", action="store_true", help="JSON output")

    # check command
    check_parser = subparsers.add_parser("check", help="Run a health check")
    check_parser.add_argument("check_id", help="Check ID")
    check_parser.add_argument("--json", action="store_true", help="JSON output")

    # status command
    status_parser = subparsers.add_parser("status", help="Get service health status")
    status_parser.add_argument("service_name", help="Service name")
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List health checks")
    list_parser.add_argument("--service", "-s", help="Filter by service")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # run-all command
    run_all_parser = subparsers.add_parser("run-all", help="Run all health checks")
    run_all_parser.add_argument("--service", "-s", help="Filter by service")
    run_all_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    checker = HealthChecker()

    if args.command == "register":
        config = HealthCheckConfig(
            check_id=f"check-{uuid.uuid4().hex[:12]}",
            name=args.name,
            check_type=HealthCheckType(args.type),
            target=args.target,
            path=args.path,
            port=args.port,
            interval_s=args.interval,
        )

        checker.register_check(args.service_name, config)

        if args.json:
            print(json.dumps(config.to_dict(), indent=2))
        else:
            print(f"Registered check: {config.check_id}")
            print(f"  Name: {config.name}")
            print(f"  Type: {config.check_type.value}")
            print(f"  Target: {config.target}")

        return 0

    elif args.command == "check":
        result = asyncio.get_event_loop().run_until_complete(
            checker.run_check(args.check_id)
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status_icon = "OK" if result.status == HealthStatus.HEALTHY else "FAIL"
            print(f"[{status_icon}] {args.check_id}")
            print(f"  Status: {result.status.value}")
            print(f"  Latency: {result.latency_ms:.2f}ms")
            print(f"  Message: {result.message}")

        return 0 if result.status == HealthStatus.HEALTHY else 1

    elif args.command == "status":
        # Run all checks for service first
        asyncio.get_event_loop().run_until_complete(
            checker.run_all_checks(args.service_name)
        )

        health = checker.get_service_health(args.service_name)

        if args.json:
            print(json.dumps(health.to_dict(), indent=2))
        else:
            status_icon = "OK" if health.status == HealthStatus.HEALTHY else "FAIL"
            print(f"[{status_icon}] {health.service_name}")
            print(f"  Status: {health.status.value}")
            print(f"  Uptime: {health.uptime_pct:.2f}%")
            print(f"  Healthy: {health.healthy_count}, Unhealthy: {health.unhealthy_count}")

        return 0 if health.status == HealthStatus.HEALTHY else 1

    elif args.command == "list":
        checks = checker.list_checks(args.service)

        if args.json:
            print(json.dumps([c.to_dict() for c in checks], indent=2))
        else:
            if not checks:
                print("No health checks registered")
            else:
                for c in checks:
                    result = checker.get_result(c.check_id)
                    status = result.status.value if result else "pending"
                    print(f"{c.check_id} ({c.name}) - {status}")

        return 0

    elif args.command == "run-all":
        results = asyncio.get_event_loop().run_until_complete(
            checker.run_all_checks(args.service)
        )

        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            healthy = sum(1 for r in results if r.status == HealthStatus.HEALTHY)
            print(f"Ran {len(results)} checks: {healthy} healthy")
            for r in results:
                status_icon = "OK" if r.status == HealthStatus.HEALTHY else "FAIL"
                print(f"  [{status_icon}] {r.check_id}: {r.message}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
