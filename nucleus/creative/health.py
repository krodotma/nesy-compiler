"""
Health Check System for Creative Section
=========================================

Provides health checks and monitoring for all Creative subsystems.

Components:
- SubsystemHealth: Health status dataclass
- HealthChecker: Probes all subsystems
- HealthRegistry: Global health state
- Bus event emission for health status
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional
import uuid

# Import emit_bus_event lazily to avoid circular import
def _get_emit_event():
    """Lazy import of emit_bus_event."""
    from nucleus.creative import emit_bus_event
    return emit_bus_event


def emit_event(topic: str, payload: dict) -> dict:
    """Emit a bus event (lazy wrapper to avoid circular imports)."""
    return _get_emit_event()(topic, payload)


# =============================================================================
# HEALTH STATUS TYPES
# =============================================================================


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class SubsystemHealth:
    """Health status of a subsystem."""

    name: str
    status: HealthStatus
    latency_ms: float
    error_rate: float  # 0-1, errors in last window
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: Optional[str] = None
    details: dict = field(default_factory=dict)
    consecutive_failures: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "error_rate": self.error_rate,
            "last_check": self.last_check.isoformat(),
            "message": self.message,
            "details": self.details,
            "consecutive_failures": self.consecutive_failures,
        }

    def is_healthy(self) -> bool:
        """Check if subsystem is healthy."""
        return self.status == HealthStatus.HEALTHY


@dataclass
class HealthCheckResult:
    """Result of a complete health check across all subsystems."""

    overall_status: HealthStatus
    subsystems: Dict[str, SubsystemHealth]
    check_duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    check_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "overall_status": self.overall_status.value,
            "subsystems": {k: v.to_dict() for k, v in self.subsystems.items()},
            "check_duration_ms": self.check_duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "check_id": self.check_id,
        }

    @property
    def healthy_count(self) -> int:
        """Number of healthy subsystems."""
        return sum(1 for s in self.subsystems.values() if s.status == HealthStatus.HEALTHY)

    @property
    def unhealthy_count(self) -> int:
        """Number of unhealthy subsystems."""
        return sum(1 for s in self.subsystems.values() if s.status == HealthStatus.UNHEALTHY)

    @property
    def degraded_count(self) -> int:
        """Number of degraded subsystems."""
        return sum(1 for s in self.subsystems.values() if s.status == HealthStatus.DEGRADED)


# =============================================================================
# HEALTH PROBE
# =============================================================================


@dataclass
class HealthProbe:
    """A health check probe for a subsystem."""

    name: str
    check_fn: Callable[[], Coroutine[Any, Any, bool]]
    timeout_s: float = 10.0
    critical: bool = True  # If critical, failure affects overall status

    _last_check_time: float = field(default=0.0, init=False)
    _last_latency_ms: float = field(default=0.0, init=False)
    _success_count: int = field(default=0, init=False)
    _error_count: int = field(default=0, init=False)
    _consecutive_failures: int = field(default=0, init=False)

    async def run(self) -> SubsystemHealth:
        """Run the health check."""
        start_time = time.time()
        self._last_check_time = start_time

        try:
            healthy = await asyncio.wait_for(
                self.check_fn(),
                timeout=self.timeout_s,
            )

            latency_ms = (time.time() - start_time) * 1000
            self._last_latency_ms = latency_ms

            if healthy:
                self._success_count += 1
                self._consecutive_failures = 0

                # Determine status based on latency
                if latency_ms > self.timeout_s * 1000 * 0.8:
                    status = HealthStatus.DEGRADED
                    message = "High latency"
                else:
                    status = HealthStatus.HEALTHY
                    message = "OK"
            else:
                self._error_count += 1
                self._consecutive_failures += 1
                status = HealthStatus.UNHEALTHY
                message = "Health check returned false"

        except asyncio.TimeoutError:
            latency_ms = self.timeout_s * 1000
            self._error_count += 1
            self._consecutive_failures += 1
            status = HealthStatus.UNHEALTHY
            message = f"Health check timed out after {self.timeout_s}s"

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._error_count += 1
            self._consecutive_failures += 1
            status = HealthStatus.UNHEALTHY
            message = str(e)

        total = self._success_count + self._error_count
        error_rate = self._error_count / total if total > 0 else 0.0

        return SubsystemHealth(
            name=self.name,
            status=status,
            latency_ms=latency_ms,
            error_rate=error_rate,
            message=message,
            consecutive_failures=self._consecutive_failures,
        )


# =============================================================================
# HEALTH CHECKER
# =============================================================================


class HealthChecker:
    """Health checker for all Creative subsystems."""

    def __init__(self):
        self._probes: Dict[str, HealthProbe] = {}
        self._last_result: Optional[HealthCheckResult] = None
        self._register_default_probes()

    def register_probe(self, probe: HealthProbe) -> None:
        """Register a health probe."""
        self._probes[probe.name] = probe

    def _register_default_probes(self) -> None:
        """Register default subsystem probes."""

        # Grammars subsystem probe
        async def check_grammars() -> bool:
            """Check grammars subsystem health."""
            try:
                from nucleus.creative import grammars
                return hasattr(grammars, "cgp") or hasattr(grammars, "CGPGenome")
            except Exception:
                return False

        self.register_probe(
            HealthProbe("grammars", check_grammars, critical=False)
        )

        # Visual subsystem probe
        async def check_visual() -> bool:
            """Check visual subsystem health."""
            try:
                from nucleus.creative import visual
                return hasattr(visual, "generator") or hasattr(visual, "ImageGenerator")
            except Exception:
                return False

        self.register_probe(
            HealthProbe("visual", check_visual, critical=False)
        )

        # Auralux subsystem probe
        async def check_auralux() -> bool:
            """Check auralux subsystem health."""
            try:
                from nucleus.creative import auralux
                return hasattr(auralux, "synthesizer") or hasattr(auralux, "VoiceSynthesizer")
            except Exception:
                return False

        self.register_probe(
            HealthProbe("auralux", check_auralux, critical=False)
        )

        # Cinema subsystem probe
        async def check_cinema() -> bool:
            """Check cinema subsystem health."""
            try:
                from nucleus.creative import cinema
                return hasattr(cinema, "storyboard") or hasattr(cinema, "frame_generator")
            except Exception:
                return False

        self.register_probe(
            HealthProbe("cinema", check_cinema, critical=False)
        )

        # Avatars subsystem probe
        async def check_avatars() -> bool:
            """Check avatars subsystem health."""
            try:
                from nucleus.creative import avatars
                # Check for exported classes from the subsystem
                return hasattr(avatars, "GaussianSplatCloud") or hasattr(avatars, "SMPLXParams")
            except Exception:
                return False

        self.register_probe(
            HealthProbe("avatars", check_avatars, critical=False)
        )

        # DiTS subsystem probe
        async def check_dits() -> bool:
            """Check dits subsystem health."""
            try:
                from nucleus.creative import dits
                return hasattr(dits, "kernel") or hasattr(dits, "narrative")
            except Exception:
                return False

        self.register_probe(
            HealthProbe("dits", check_dits, critical=False)
        )

        # Bus connectivity probe
        async def check_bus() -> bool:
            """Check bus connectivity."""
            try:
                from pathlib import Path
                bus_path = Path("/pluribus/.pluribus/bus/events.ndjson")
                return bus_path.parent.exists()
            except Exception:
                return False

        self.register_probe(
            HealthProbe("bus", check_bus, critical=True, timeout_s=2.0)
        )

    async def check(self) -> HealthCheckResult:
        """Run all health checks."""
        start_time = time.time()

        # Run all probes concurrently
        results = await asyncio.gather(
            *[probe.run() for probe in self._probes.values()],
            return_exceptions=True,
        )

        subsystems: Dict[str, SubsystemHealth] = {}
        for probe, result in zip(self._probes.values(), results):
            if isinstance(result, Exception):
                subsystems[probe.name] = SubsystemHealth(
                    name=probe.name,
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=0.0,
                    error_rate=1.0,
                    message=str(result),
                )
            else:
                subsystems[probe.name] = result

        # Determine overall status
        overall_status = self._compute_overall_status(subsystems)

        check_duration_ms = (time.time() - start_time) * 1000

        result = HealthCheckResult(
            overall_status=overall_status,
            subsystems=subsystems,
            check_duration_ms=check_duration_ms,
        )

        self._last_result = result
        return result

    def _compute_overall_status(self, subsystems: Dict[str, SubsystemHealth]) -> HealthStatus:
        """Compute overall health status."""
        critical_unhealthy = False
        any_degraded = False
        any_unhealthy = False

        for name, health in subsystems.items():
            probe = self._probes.get(name)
            if health.status == HealthStatus.UNHEALTHY:
                any_unhealthy = True
                if probe and probe.critical:
                    critical_unhealthy = True
            elif health.status == HealthStatus.DEGRADED:
                any_degraded = True

        if critical_unhealthy:
            return HealthStatus.UNHEALTHY
        if any_unhealthy or any_degraded:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


async def check_health() -> HealthCheckResult:
    """Run a health check using the global checker."""
    return await get_health_checker().check()


def is_healthy() -> bool:
    """Quick synchronous check if system was healthy at last check."""
    checker = get_health_checker()
    if checker._last_result is None:
        return True  # Assume healthy if never checked
    return checker._last_result.overall_status == HealthStatus.HEALTHY
