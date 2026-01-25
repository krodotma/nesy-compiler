#!/usr/bin/env python3
"""
circuit.py - Circuit Breaker Pattern

P2-094: Create circuit breaker pattern

Implements circuit breaker for protecting against cascading failures.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Callable
from enum import Enum

logger = logging.getLogger("ARK.Perf.Circuit")


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""
    failures: int = 0
    successes: int = 0
    rejections: int = 0
    last_failure: Optional[float] = None
    last_success: Optional[float] = None


class CircuitBreaker:
    """
    Circuit breaker for gate protection.
    
    P2-094: Create circuit breaker pattern
    
    States:
    - CLOSED: Normal operation, count failures
    - OPEN: Rejecting requests, wait for timeout
    - HALF_OPEN: Allow one request to test recovery
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self.opened_at: Optional[float] = None
        self.half_open_attempts = 0
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout passed
            if time.time() - self.opened_at >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
                logger.info("Circuit %s: OPEN -> HALF_OPEN", self.name)
                return True
            self.stats.rejections += 1
            return False
        
        # HALF_OPEN: allow limited attempts
        if self.half_open_attempts < self.half_open_max:
            return True
        self.stats.rejections += 1
        return False
    
    def record_success(self) -> None:
        """Record successful execution."""
        self.stats.successes += 1
        self.stats.last_success = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_attempts += 1
            if self.half_open_attempts >= self.half_open_max:
                self.state = CircuitState.CLOSED
                self.stats.failures = 0
                logger.info("Circuit %s: HALF_OPEN -> CLOSED", self.name)
    
    def record_failure(self) -> None:
        """Record failed execution."""
        self.stats.failures += 1
        self.stats.last_failure = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.stats.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.opened_at = time.time()
                logger.warning("Circuit %s: CLOSED -> OPEN", self.name)
        
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.opened_at = time.time()
            logger.warning("Circuit %s: HALF_OPEN -> OPEN", self.name)
    
    def execute(self, fn: Callable, *args, **kwargs) -> Any:
        """Execute with circuit breaker protection."""
        if not self.can_execute():
            raise CircuitOpenError(f"Circuit {self.name} is OPEN")
        
        try:
            result = fn(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise
    
    def reset(self) -> None:
        """Reset circuit to closed state."""
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self.opened_at = None
        self.half_open_attempts = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self.stats.failures,
            "successes": self.stats.successes,
            "rejections": self.stats.rejections
        }


class CircuitOpenError(Exception):
    """Raised when circuit is open and request rejected."""
    pass
