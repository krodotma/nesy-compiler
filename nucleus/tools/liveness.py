#!/usr/bin/env python3
from __future__ import annotations
from abc import ABC, abstractmethod
import time
from typing import Any

class LivenessMonitor(ABC):
    """
    Abstract base class for liveness/progress monitoring (ω-checks).
    
    This enforces the 'Spec-First Verifier' hypothesis by providing a 
    standard contract for agents to report progress and for the system
    to kill/reset non-converging lineages.
    """

    @abstractmethod
    def heartbeat(self, state: dict[str, Any]) -> None:
        """Record a heartbeat with current state snapshot."""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Return True if the process is making progress and within safety bounds."""
        pass

    @abstractmethod
    def diagnostics(self) -> dict[str, Any]:
        """Return debugging info for failure analysis."""
        pass

class TimeBoundMonitor(LivenessMonitor):
    """Simple wall-clock deadline monitor."""
    
    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        self.start_time = time.time()
        self.last_beat = self.start_time

    def heartbeat(self, state: dict[str, Any]) -> None:
        self.last_beat = time.time()

    def is_healthy(self) -> bool:
        return (time.time() - self.start_time) < self.max_seconds

    def diagnostics(self) -> dict[str, Any]:
        now = time.time()
        return {
            "elapsed": now - self.start_time,
            "since_last_beat": now - self.last_beat,
            "limit": self.max_seconds
        }
