#!/usr/bin/env python3
"""
telemetry.py - Production Telemetry

P2-099: Add telemetry for production monitoring

Implements metrics collection and export for observability.
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("ARK.Perf.Telemetry")


@dataclass
class Metric:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {"name": self.name, "value": self.value, 
                "timestamp": self.timestamp, "labels": self.labels}


class MetricCollector:
    """
    Collects metrics for telemetry.
    
    P2-099: Production monitoring
    """
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timeseries: List[Metric] = []
    
    def inc(self, name: str, value: float = 1.0, labels: Dict = None) -> None:
        """Increment counter."""
        key = f"{name}:{labels}" if labels else name
        self.counters[key] += value
        self._record(name, self.counters[key], labels)
    
    def set(self, name: str, value: float, labels: Dict = None) -> None:
        """Set gauge value."""
        key = f"{name}:{labels}" if labels else name
        self.gauges[key] = value
        self._record(name, value, labels)
    
    def observe(self, name: str, value: float, labels: Dict = None) -> None:
        """Record histogram observation."""
        key = f"{name}:{labels}" if labels else name
        self.histograms[key].append(value)
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
    
    def _record(self, name: str, value: float, labels: Dict = None) -> None:
        """Record to timeseries."""
        self.timeseries.append(Metric(name, value, labels=labels or {}))
        if len(self.timeseries) > self.max_points:
            self.timeseries = self.timeseries[-self.max_points:]
    
    def get_all(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histogram_counts": {k: len(v) for k, v in self.histograms.items()}
        }


class Telemetry:
    """
    Production telemetry system.
    
    P2-099: Add telemetry for production monitoring
    """
    
    def __init__(self, service_name: str = "ark", export_path: Optional[str] = None):
        self.service_name = service_name
        self.export_path = Path(export_path) if export_path else None
        self.collector = MetricCollector()
        self.started_at = time.time()
    
    # Gate metrics
    def gate_executed(self, gate_name: str, passed: bool, duration_ms: float) -> None:
        """Record gate execution."""
        self.collector.inc(f"gate.{gate_name}.total")
        if passed:
            self.collector.inc(f"gate.{gate_name}.passed")
        else:
            self.collector.inc(f"gate.{gate_name}.failed")
        self.collector.observe(f"gate.{gate_name}.duration_ms", duration_ms)
    
    # Commit metrics
    def commit_started(self) -> None:
        self.collector.inc("commit.started")
    
    def commit_completed(self, success: bool, duration_ms: float) -> None:
        self.collector.inc("commit.completed")
        if success:
            self.collector.inc("commit.success")
        else:
            self.collector.inc("commit.failed")
        self.collector.observe("commit.duration_ms", duration_ms)
    
    # CMP metrics
    def cmp_recorded(self, value: float) -> None:
        self.collector.set("cmp.current", value)
        self.collector.observe("cmp.history", value)
    
    # Safety metrics
    def safety_alert(self, level: str, message: str) -> None:
        self.collector.inc(f"safety.alert.{level}")
        logger.warning("Safety alert [%s]: %s", level, message)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get telemetry summary."""
        uptime = time.time() - self.started_at
        return {
            "service": self.service_name,
            "uptime_seconds": uptime,
            "metrics": self.collector.get_all()
        }
    
    def export(self) -> Optional[str]:
        """Export metrics to file."""
        if not self.export_path:
            return None
        
        try:
            data = self.get_summary()
            data["exported_at"] = time.time()
            
            self.export_path.parent.mkdir(parents=True, exist_ok=True)
            self.export_path.write_text(json.dumps(data, indent=2))
            return str(self.export_path)
        except Exception as e:
            logger.error("Failed to export: %s", e)
            return None


# Global telemetry instance
_telemetry = Telemetry()


def get_telemetry() -> Telemetry:
    """Get global telemetry instance."""
    return _telemetry
