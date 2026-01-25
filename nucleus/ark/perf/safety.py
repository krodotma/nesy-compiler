#!/usr/bin/env python3
"""
safety.py - Kill Switch and Safety Monitors

P2-091: Integrate kill_switch.py
P2-092: Implement spectral stability monitor
P2-093: Add anomaly detection
P2-095: Implement rate limiting for mutations
P2-096: Add resource consumption guards

GLM-4.7 R&D insights: Rate-limit mutations, spectral stability checks.
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from enum import Enum

logger = logging.getLogger("ARK.Perf.Safety")


class SafetyLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SafetyStatus:
    """Current safety status."""
    level: SafetyLevel
    active_alerts: List[str] = field(default_factory=list)
    kill_switch_active: bool = False
    rate_limited: bool = False


class KillSwitch:
    """
    Emergency kill switch for stopping ARK operations.
    
    P2-091: Integrate kill_switch.py
    GLM-4.7 insight: Critical for preventing runaway mutations.
    """
    
    def __init__(self, cool_down_seconds: float = 60.0):
        self.cool_down = cool_down_seconds
        self.active = False
        self.triggered_at: Optional[float] = None
        self.trigger_reason: Optional[str] = None
        self._lock = threading.Lock()
    
    def trigger(self, reason: str = "Manual trigger") -> None:
        """Activate kill switch."""
        with self._lock:
            self.active = True
            self.triggered_at = time.time()
            self.trigger_reason = reason
            logger.critical("KILL SWITCH ACTIVATED: %s", reason)
    
    def reset(self) -> bool:
        """Reset kill switch if cool down passed."""
        with self._lock:
            if not self.active:
                return True
            
            if self.triggered_at and (time.time() - self.triggered_at) >= self.cool_down:
                self.active = False
                self.trigger_reason = None
                logger.info("Kill switch reset")
                return True
            return False
    
    def force_reset(self) -> None:
        """Force reset (bypass cool down)."""
        with self._lock:
            self.active = False
            self.trigger_reason = None
            logger.warning("Kill switch force reset")
    
    def check(self) -> bool:
        """Check if operations are allowed."""
        return not self.active
    
    def guard(self, fn: Callable) -> Callable:
        """Decorator to guard function with kill switch."""
        def wrapper(*args, **kwargs):
            if not self.check():
                raise KillSwitchError(f"Kill switch active: {self.trigger_reason}")
            return fn(*args, **kwargs)
        return wrapper


class KillSwitchError(Exception):
    """Raised when operation blocked by kill switch."""
    pass


class SpectralMonitor:
    """
    Spectral stability monitor for RER ranking.
    
    P2-092: Implement spectral stability monitor
    GLM-4.7 insight: Spectral stability checks are critical for preventing ranking manipulation.
    """
    
    def __init__(self, stability_threshold: float = 0.1, window_size: int = 100):
        self.threshold = stability_threshold
        self.window_size = window_size
        self.eigenvalues: deque = deque(maxlen=window_size)
        self.alerts: List[str] = []
    
    def record_eigenvalue(self, value: float) -> None:
        """Record principal eigenvalue from RER."""
        self.eigenvalues.append(value)
    
    def check_stability(self) -> bool:
        """Check if eigenvalue is stable (not manipulated)."""
        if len(self.eigenvalues) < 2:
            return True
        
        recent = list(self.eigenvalues)[-10:]
        if len(recent) < 2:
            return True
        
        # Check variance
        mean_val = sum(recent) / len(recent)
        variance = sum((v - mean_val) ** 2 for v in recent) / len(recent)
        
        if variance > self.threshold:
            self.alerts.append(f"Spectral instability: variance={variance:.4f}")
            return False
        return True
    
    def get_trend(self) -> str:
        """Get eigenvalue trend."""
        if len(self.eigenvalues) < 10:
            return "insufficient_data"
        
        recent = list(self.eigenvalues)[-10:]
        older = list(self.eigenvalues)[-20:-10] if len(self.eigenvalues) >= 20 else recent
        
        recent_mean = sum(recent) / len(recent)
        older_mean = sum(older) / len(older)
        
        if recent_mean > older_mean * 1.1:
            return "increasing"
        elif recent_mean < older_mean * 0.9:
            return "decreasing"
        return "stable"


class AnomalyDetector:
    """
    Anomaly detection for commit patterns.
    
    P2-093: Add anomaly detection
    """
    
    def __init__(self, z_threshold: float = 3.0, window_size: int = 50):
        self.z_threshold = z_threshold
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
    
    def record(self, metric_name: str, value: float) -> bool:
        """Record metric and check for anomaly. Returns True if anomalous."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque(maxlen=self.window_size)
        
        history = self.metrics[metric_name]
        is_anomaly = False
        
        if len(history) >= 10:
            mean = sum(history) / len(history)
            std = (sum((v - mean) ** 2 for v in history) / len(history)) ** 0.5
            
            if std > 0:
                z_score = abs(value - mean) / std
                if z_score > self.z_threshold:
                    is_anomaly = True
                    logger.warning("Anomaly: %s=%.2f (z=%.2f)", metric_name, value, z_score)
        
        history.append(value)
        return is_anomaly
    
    def get_stats(self, metric_name: str) -> Optional[Dict]:
        """Get stats for a metric."""
        if metric_name not in self.metrics:
            return None
        
        history = list(self.metrics[metric_name])
        if not history:
            return None
        
        return {
            "count": len(history),
            "mean": sum(history) / len(history),
            "min": min(history),
            "max": max(history)
        }


class RateLimiter:
    """
    Rate limiter for mutations.
    
    P2-095: Implement rate limiting for mutations
    GLM-4.7 insight: Rate-limit mutation proposals.
    """
    
    def __init__(self, max_per_minute: int = 60, max_per_hour: int = 500):
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        self.minute_window: deque = deque()
        self.hour_window: deque = deque()
        self._lock = threading.Lock()
    
    def allow(self) -> bool:
        """Check if request is allowed."""
        now = time.time()
        
        with self._lock:
            # Clean old entries
            minute_ago = now - 60
            hour_ago = now - 3600
            
            while self.minute_window and self.minute_window[0] < minute_ago:
                self.minute_window.popleft()
            while self.hour_window and self.hour_window[0] < hour_ago:
                self.hour_window.popleft()
            
            # Check limits
            if len(self.minute_window) >= self.max_per_minute:
                return False
            if len(self.hour_window) >= self.max_per_hour:
                return False
            
            # Record
            self.minute_window.append(now)
            self.hour_window.append(now)
            return True
    
    def get_usage(self) -> Dict[str, int]:
        """Get current usage."""
        return {
            "minute": len(self.minute_window),
            "hour": len(self.hour_window),
            "minute_limit": self.max_per_minute,
            "hour_limit": self.max_per_hour
        }


class ResourceGuard:
    """
    Resource consumption guard.
    
    P2-096: Add resource consumption guards
    """
    
    def __init__(
        self, 
        max_memory_mb: int = 1024,
        max_cpu_percent: float = 80.0,
        max_file_handles: int = 1000
    ):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.max_file_handles = max_file_handles
    
    def check(self) -> Dict[str, Any]:
        """Check resource usage."""
        try:
            import psutil
            process = psutil.Process()
            
            memory_mb = process.memory_info().rss / (1024 * 1024)
            cpu_percent = process.cpu_percent(interval=0.1)
            file_handles = len(process.open_files())
            
            return {
                "memory_mb": memory_mb,
                "memory_ok": memory_mb < self.max_memory_mb,
                "cpu_percent": cpu_percent,
                "cpu_ok": cpu_percent < self.max_cpu_percent,
                "file_handles": file_handles,
                "files_ok": file_handles < self.max_file_handles,
                "all_ok": all([
                    memory_mb < self.max_memory_mb,
                    cpu_percent < self.max_cpu_percent,
                    file_handles < self.max_file_handles
                ])
            }
        except ImportError:
            return {"all_ok": True, "psutil_available": False}
