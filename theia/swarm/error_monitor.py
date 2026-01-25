"""
Theia Error Rate Monitor — Circuit breaker for high error scenarios.

Implements Phase 3 of Prescient Enhancement Plan:
- Telemetry error isolation
- Proactive provider health monitoring
- Circuit breaker pattern
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import deque
from enum import Enum, auto


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()   # Normal operation
    OPEN = auto()     # Blocking requests (high error rate)
    HALF_OPEN = auto() # Testing if recovery is possible


@dataclass
class ErrorWindow:
    """Sliding window for error rate calculation."""
    window_size: int = 100
    errors: deque = field(default_factory=lambda: deque(maxlen=100))
    total: int = 0
    
    def record(self, is_error: bool) -> None:
        """Record a request result."""
        self.errors.append(1 if is_error else 0)
        self.total += 1
    
    def error_rate(self) -> float:
        """Calculate current error rate."""
        if len(self.errors) == 0:
            return 0.0
        return sum(self.errors) / len(self.errors)


class CircuitBreaker:
    """
    Circuit breaker for error rate management.
    
    States:
    - CLOSED: Normal operation, errors are tracked
    - OPEN: High error rate detected, requests blocked
    - HALF_OPEN: Testing recovery, limited requests allowed
    """
    
    def __init__(
        self,
        name: str,
        error_threshold: float = 0.1,  # 10%
        recovery_timeout: float = 30.0,
        half_open_max: int = 3,
    ):
        self.name = name
        self.error_threshold = error_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        
        self.state = CircuitState.CLOSED
        self.window = ErrorWindow()
        self.opened_at: Optional[float] = None
        self.half_open_count = 0
    
    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self.opened_at > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_count = 0
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_count < self.half_open_max:
                self.half_open_count += 1
                return True
            return False
        
        return True
    
    def record_success(self) -> None:
        """Record successful request."""
        self.window.record(is_error=False)
        
        if self.state == CircuitState.HALF_OPEN:
            # Successful half-open request, close circuit
            self.state = CircuitState.CLOSED
    
    def record_failure(self) -> None:
        """Record failed request."""
        self.window.record(is_error=True)
        
        if self.state == CircuitState.CLOSED:
            if self.window.error_rate() > self.error_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = time.time()
        
        elif self.state == CircuitState.HALF_OPEN:
            # Failure during half-open, reopen circuit
            self.state = CircuitState.OPEN
            self.opened_at = time.time()
    
    def status(self) -> Dict[str, Any]:
        """Get circuit status."""
        return {
            "name": self.name,
            "state": self.state.name,
            "error_rate": self.window.error_rate(),
            "threshold": self.error_threshold,
            "total_requests": self.window.total,
        }


class ErrorRateMonitor:
    """
    System-wide error rate monitoring.
    
    Features:
    - Per-provider circuit breakers
    - Telemetry error isolation
    - Proactive health alerts
    """
    
    def __init__(self, bus_path: str = "/pluribus/.pluribus/bus/events.ndjson"):
        self.bus_path = Path(bus_path)
        self.breakers: Dict[str, CircuitBreaker] = {}
        
        # Isolated metrics (don't count toward provider health)
        self.isolated_topics = {
            "telemetry.client.fetch_error",
            "telemetry.client.performance",
            "qa.anomaly.detected",
        }
    
    def get_or_create_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create circuit breaker for provider."""
        if provider not in self.breakers:
            self.breakers[provider] = CircuitBreaker(name=provider)
        return self.breakers[provider]
    
    def should_allow(self, provider: str) -> bool:
        """Check if request to provider should be allowed."""
        breaker = self.get_or_create_breaker(provider)
        return breaker.allow_request()
    
    def record_result(self, provider: str, success: bool, topic: str = "") -> None:
        """Record request result."""
        # Isolate telemetry errors
        if topic in self.isolated_topics:
            return
        
        breaker = self.get_or_create_breaker(provider)
        if success:
            breaker.record_success()
        else:
            breaker.record_failure()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all providers."""
        summary = {}
        for name, breaker in self.breakers.items():
            summary[name] = breaker.status()
        return summary
    
    def emit_health_alert(self, provider: str) -> None:
        """Emit health alert for provider."""
        breaker = self.breakers.get(provider)
        if not breaker:
            return
        
        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "topic": "theia.health.alert",
            "kind": "alert",
            "level": "warn" if breaker.state == CircuitState.HALF_OPEN else "error",
            "actor": "theia-error-monitor",
            "data": breaker.status()
        }
        
        try:
            with open(self.bus_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"[ErrorMonitor] Failed to emit alert: {e}")


__all__ = [
    "CircuitState",
    "ErrorWindow", 
    "CircuitBreaker",
    "ErrorRateMonitor",
]
