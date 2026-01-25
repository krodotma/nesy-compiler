#!/usr/bin/env python3
"""
profiler.py - Gate Execution Profiling

P2-081: Profile gate execution time
P2-090: Create `ark benchmark` command

Implements performance profiling for gates and commit pipeline.
"""

import time
import statistics
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from functools import wraps

logger = logging.getLogger("ARK.Perf.Profiler")


@dataclass
class ProfileResult:
    """Result of a single profile measurement."""
    name: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    name: str
    count: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    total_ms: float
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name, "count": self.count,
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2)
        }


class GateProfiler:
    """
    Profiler for gate execution.
    
    P2-081: Profile gate execution time
    """
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.samples: Dict[str, List[float]] = {}
        self.active_timers: Dict[str, float] = {}
    
    def start(self, name: str) -> None:
        """Start timing a named operation."""
        self.active_timers[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """Stop timing and record result."""
        if name not in self.active_timers:
            return 0.0
        
        elapsed = (time.perf_counter() - self.active_timers[name]) * 1000
        del self.active_timers[name]
        
        if name not in self.samples:
            self.samples[name] = []
        
        self.samples[name].append(elapsed)
        
        # Trim to max samples
        if len(self.samples[name]) > self.max_samples:
            self.samples[name] = self.samples[name][-self.max_samples:]
        
        return elapsed
    
    def profile(self, name: str):
        """Decorator for profiling functions."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.start(name)
                try:
                    return func(*args, **kwargs)
                finally:
                    self.stop(name)
            return wrapper
        return decorator
    
    def get_metrics(self, name: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a named operation."""
        if name not in self.samples or not self.samples[name]:
            return None
        
        data = self.samples[name]
        sorted_data = sorted(data)
        n = len(data)
        
        return PerformanceMetrics(
            name=name,
            count=n,
            mean_ms=statistics.mean(data),
            median_ms=statistics.median(data),
            min_ms=min(data),
            max_ms=max(data),
            p95_ms=sorted_data[int(0.95 * n)] if n > 0 else 0,
            p99_ms=sorted_data[int(0.99 * n)] if n > 0 else 0,
            std_ms=statistics.stdev(data) if n > 1 else 0,
            total_ms=sum(data)
        )
    
    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get metrics for all profiled operations."""
        return {name: self.get_metrics(name) for name in self.samples}
    
    def report(self) -> str:
        """Generate performance report."""
        lines = ["Performance Report", "=" * 50, ""]
        
        for name in sorted(self.samples.keys()):
            metrics = self.get_metrics(name)
            if metrics:
                lines.append(f"{name}:")
                lines.append(f"  Count: {metrics.count}")
                lines.append(f"  Mean: {metrics.mean_ms:.2f}ms")
                lines.append(f"  P95: {metrics.p95_ms:.2f}ms")
                lines.append(f"  P99: {metrics.p99_ms:.2f}ms")
                lines.append("")
        
        return "\n".join(lines)
    
    def reset(self) -> None:
        """Clear all samples."""
        self.samples.clear()
        self.active_timers.clear()


# Global profiler instance
_global_profiler = GateProfiler()


def profile(name: str):
    """Global profile decorator."""
    return _global_profiler.profile(name)


def get_profiler() -> GateProfiler:
    """Get global profiler instance."""
    return _global_profiler
