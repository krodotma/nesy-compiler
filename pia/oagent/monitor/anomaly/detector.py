#!/usr/bin/env python3
"""
Anomaly Detector - Step 257

Detects anomalies in metrics using statistical methods (z-score, IQR, etc.).

PBTSO Phase: VERIFY

Bus Topics:
- monitor.anomaly.detect (subscribed)
- monitor.anomaly.detected (emitted)

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from ..metrics.collector import MetricCollector, MetricPoint, get_collector as get_metric_collector


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DetectionMethod(Enum):
    """Anomaly detection methods."""
    Z_SCORE = "z_score"           # Standard deviation based
    IQR = "iqr"                   # Interquartile range
    MAD = "mad"                   # Median absolute deviation
    RATE_CHANGE = "rate_change"  # Sudden rate changes
    THRESHOLD = "threshold"       # Fixed thresholds


@dataclass
class Anomaly:
    """A detected anomaly.

    Attributes:
        anomaly_id: Unique identifier
        metric_name: Name of the anomalous metric
        timestamp: When detected
        expected_value: Expected/baseline value
        actual_value: Observed value
        deviation_sigma: Deviation in standard deviations (for z-score)
        severity: Anomaly severity
        method: Detection method used
        details: Additional details
        labels: Metric labels
    """
    anomaly_id: str
    metric_name: str
    timestamp: float
    expected_value: float
    actual_value: float
    deviation_sigma: float
    severity: AnomalySeverity
    method: DetectionMethod
    details: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_id": self.anomaly_id,
            "metric_name": self.metric_name,
            "timestamp": self.timestamp,
            "expected_value": self.expected_value,
            "actual_value": self.actual_value,
            "deviation_sigma": self.deviation_sigma,
            "severity": self.severity.value,
            "method": self.method.value,
            "details": self.details,
            "labels": self.labels,
        }


@dataclass
class AnomalyThreshold:
    """Threshold configuration for a metric.

    Attributes:
        metric_name: Metric name pattern
        min_value: Minimum acceptable value
        max_value: Maximum acceptable value
        z_score_threshold: Z-score threshold for anomaly
        rate_change_threshold: Maximum rate change per second
    """
    metric_name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    z_score_threshold: float = 3.0
    rate_change_threshold: Optional[float] = None


class AnomalyDetector:
    """
    Detect anomalies in metrics using statistical methods.

    The detector:
    - Maintains sliding window baselines
    - Supports multiple detection methods
    - Calculates z-scores, IQR, MAD
    - Detects rate changes

    Example:
        detector = AnomalyDetector(window_size=100)

        # Detect anomaly in single value
        anomaly = detector.detect("cpu.usage", 95.5)
        if anomaly:
            print(f"Anomaly: {anomaly.severity.value}")

        # Run detection on all metrics
        anomalies = detector.detect_all()
    """

    BUS_TOPICS = {
        "detect": "monitor.anomaly.detect",
        "detected": "monitor.anomaly.detected",
    }

    def __init__(
        self,
        window_size: int = 100,
        z_score_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        collector: Optional[MetricCollector] = None,
        bus_dir: Optional[str] = None
    ):
        """Initialize anomaly detector.

        Args:
            window_size: Size of sliding window for baseline
            z_score_threshold: Default z-score threshold
            iqr_multiplier: IQR multiplier for outlier detection
            collector: Metric collector to monitor
            bus_dir: Directory for bus events
        """
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self._collector = collector or get_metric_collector()

        # Sliding window baselines per metric
        self._baselines: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._thresholds: Dict[str, AnomalyThreshold] = {}
        self._anomaly_count: int = 0

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def set_threshold(self, threshold: AnomalyThreshold) -> None:
        """Set custom threshold for a metric.

        Args:
            threshold: Threshold configuration
        """
        self._thresholds[threshold.metric_name] = threshold

    def detect(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        method: DetectionMethod = DetectionMethod.Z_SCORE
    ) -> Optional[Anomaly]:
        """Detect anomaly for a single metric value.

        Args:
            metric_name: Metric name
            value: Current value
            labels: Metric labels
            method: Detection method

        Returns:
            Anomaly if detected, None otherwise
        """
        key = self._get_baseline_key(metric_name, labels)
        baseline = self._baselines[key]

        # Add to baseline
        baseline.append(value)

        # Need minimum samples for detection
        if len(baseline) < self.window_size // 2:
            return None

        # Get threshold config
        threshold = self._thresholds.get(metric_name, AnomalyThreshold(metric_name))

        # Check fixed thresholds first
        if threshold.min_value is not None and value < threshold.min_value:
            return self._create_anomaly(
                metric_name, value, threshold.min_value,
                (threshold.min_value - value) / max(1, abs(threshold.min_value)),
                AnomalySeverity.WARNING, DetectionMethod.THRESHOLD, labels
            )

        if threshold.max_value is not None and value > threshold.max_value:
            return self._create_anomaly(
                metric_name, value, threshold.max_value,
                (value - threshold.max_value) / max(1, abs(threshold.max_value)),
                AnomalySeverity.WARNING, DetectionMethod.THRESHOLD, labels
            )

        # Apply selected method
        if method == DetectionMethod.Z_SCORE:
            return self._detect_z_score(metric_name, value, baseline, threshold, labels)
        elif method == DetectionMethod.IQR:
            return self._detect_iqr(metric_name, value, baseline, labels)
        elif method == DetectionMethod.MAD:
            return self._detect_mad(metric_name, value, baseline, labels)
        elif method == DetectionMethod.RATE_CHANGE:
            return self._detect_rate_change(metric_name, value, baseline, threshold, labels)

        return None

    def detect_multi(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> List[Anomaly]:
        """Detect anomalies using multiple methods.

        Args:
            metric_name: Metric name
            value: Current value
            labels: Metric labels

        Returns:
            List of detected anomalies
        """
        anomalies = []
        for method in [DetectionMethod.Z_SCORE, DetectionMethod.IQR, DetectionMethod.MAD]:
            anomaly = self.detect(metric_name, value, labels, method)
            if anomaly:
                anomalies.append(anomaly)
        return anomalies

    def detect_all(
        self,
        window_s: int = 300,
        method: DetectionMethod = DetectionMethod.Z_SCORE
    ) -> List[Anomaly]:
        """Detect anomalies across all metrics.

        Args:
            window_s: Time window to analyze
            method: Detection method

        Returns:
            List of anomalies
        """
        anomalies = []

        for metric_name in self._collector.list_metrics():
            points = self._collector.query_series(metric_name, window_s)
            if not points:
                continue

            # Get label sets
            label_sets = self._collector.list_labels(metric_name)
            for labels in label_sets:
                label_points = [
                    p for p in points
                    if all(p.labels.get(k) == v for k, v in labels.items())
                ]

                if not label_points:
                    continue

                # Check latest value
                latest = label_points[-1]
                for p in label_points:
                    anomaly = self.detect(metric_name, p.value, labels, method)
                    if anomaly:
                        anomalies.append(anomaly)

        return anomalies

    def get_baseline_stats(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Get baseline statistics for a metric.

        Args:
            metric_name: Metric name
            labels: Metric labels

        Returns:
            Statistics dictionary
        """
        key = self._get_baseline_key(metric_name, labels)
        baseline = list(self._baselines[key])

        if not baseline:
            return {}

        n = len(baseline)
        mean = sum(baseline) / n
        variance = sum((x - mean) ** 2 for x in baseline) / n if n > 1 else 0
        stddev = math.sqrt(variance)

        sorted_values = sorted(baseline)
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        q1 = sorted_values[q1_idx] if q1_idx < n else sorted_values[0]
        q3 = sorted_values[q3_idx] if q3_idx < n else sorted_values[-1]
        iqr = q3 - q1

        return {
            "count": n,
            "mean": mean,
            "stddev": stddev,
            "variance": variance,
            "min": min(baseline),
            "max": max(baseline),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "median": sorted_values[n // 2],
        }

    def handle_detect_request(self, event: Dict[str, Any]) -> List[Anomaly]:
        """Handle detection request from bus.

        Args:
            event: Bus event

        Returns:
            List of anomalies
        """
        data = event.get("data", {})
        metric_name = data.get("metric", data.get("name"))
        value = data.get("value")
        labels = data.get("labels", {})
        method = DetectionMethod(data.get("method", "z_score"))

        if metric_name and value is not None:
            anomaly = self.detect(metric_name, float(value), labels, method)
            anomalies = [anomaly] if anomaly else []
        else:
            window_s = data.get("window_s", 300)
            anomalies = self.detect_all(window_s, method)

        # Emit results
        for anomaly in anomalies:
            self.emit_detected_event(anomaly)

        return anomalies

    def emit_detected_event(self, anomaly: Anomaly) -> str:
        """Emit anomaly detected event.

        Args:
            anomaly: Detected anomaly

        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        level = "warn" if anomaly.severity == AnomalySeverity.WARNING else "error"

        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": self.BUS_TOPICS["detected"],
            "kind": "anomaly",
            "level": level,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": anomaly.to_dict(),
        }

        with open(self._bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        self._anomaly_count += 1
        return event_id

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "window_size": self.window_size,
            "z_score_threshold": self.z_score_threshold,
            "tracked_metrics": len(self._baselines),
            "configured_thresholds": len(self._thresholds),
            "anomalies_detected": self._anomaly_count,
        }

    def _detect_z_score(
        self,
        metric_name: str,
        value: float,
        baseline: Deque[float],
        threshold: AnomalyThreshold,
        labels: Optional[Dict[str, str]]
    ) -> Optional[Anomaly]:
        """Detect using z-score method.

        Z-score = (value - mean) / stddev
        """
        values = list(baseline)[:-1]  # Exclude current value
        if not values:
            return None

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        stddev = math.sqrt(variance)

        if stddev == 0:
            return None

        z_score = abs((value - mean) / stddev)
        z_threshold = threshold.z_score_threshold or self.z_score_threshold

        if z_score > z_threshold:
            severity = (
                AnomalySeverity.CRITICAL if z_score > z_threshold * 1.5
                else AnomalySeverity.WARNING
            )
            return self._create_anomaly(
                metric_name, value, mean, z_score,
                severity, DetectionMethod.Z_SCORE, labels,
                {"stddev": stddev, "threshold": z_threshold}
            )

        return None

    def _detect_iqr(
        self,
        metric_name: str,
        value: float,
        baseline: Deque[float],
        labels: Optional[Dict[str, str]]
    ) -> Optional[Anomaly]:
        """Detect using Interquartile Range method.

        Outlier if value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR
        """
        values = sorted(baseline)
        n = len(values)
        if n < 4:
            return None

        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        q1 = values[q1_idx]
        q3 = values[q3_idx]
        iqr = q3 - q1

        if iqr == 0:
            return None

        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr

        if value < lower_bound or value > upper_bound:
            expected = (q1 + q3) / 2
            deviation = abs(value - expected) / iqr
            severity = (
                AnomalySeverity.CRITICAL if deviation > 3
                else AnomalySeverity.WARNING
            )
            return self._create_anomaly(
                metric_name, value, expected, deviation,
                severity, DetectionMethod.IQR, labels,
                {"q1": q1, "q3": q3, "iqr": iqr}
            )

        return None

    def _detect_mad(
        self,
        metric_name: str,
        value: float,
        baseline: Deque[float],
        labels: Optional[Dict[str, str]]
    ) -> Optional[Anomaly]:
        """Detect using Median Absolute Deviation.

        MAD = median(|Xi - median(X)|)
        Modified Z-score = 0.6745 * (Xi - median) / MAD
        """
        values = sorted(baseline)
        n = len(values)
        if n < 5:
            return None

        median = values[n // 2]
        abs_deviations = sorted(abs(x - median) for x in values)
        mad = abs_deviations[n // 2]

        if mad == 0:
            return None

        # Modified z-score
        modified_z = 0.6745 * abs(value - median) / mad

        if modified_z > self.z_score_threshold:
            severity = (
                AnomalySeverity.CRITICAL if modified_z > self.z_score_threshold * 1.5
                else AnomalySeverity.WARNING
            )
            return self._create_anomaly(
                metric_name, value, median, modified_z,
                severity, DetectionMethod.MAD, labels,
                {"median": median, "mad": mad}
            )

        return None

    def _detect_rate_change(
        self,
        metric_name: str,
        value: float,
        baseline: Deque[float],
        threshold: AnomalyThreshold,
        labels: Optional[Dict[str, str]]
    ) -> Optional[Anomaly]:
        """Detect sudden rate changes."""
        if len(baseline) < 2 or not threshold.rate_change_threshold:
            return None

        prev_value = list(baseline)[-2]
        rate_change = abs(value - prev_value)

        if rate_change > threshold.rate_change_threshold:
            severity = (
                AnomalySeverity.CRITICAL if rate_change > threshold.rate_change_threshold * 2
                else AnomalySeverity.WARNING
            )
            return self._create_anomaly(
                metric_name, value, prev_value, rate_change / threshold.rate_change_threshold,
                severity, DetectionMethod.RATE_CHANGE, labels,
                {"rate_change": rate_change, "threshold": threshold.rate_change_threshold}
            )

        return None

    def _create_anomaly(
        self,
        metric_name: str,
        actual: float,
        expected: float,
        deviation: float,
        severity: AnomalySeverity,
        method: DetectionMethod,
        labels: Optional[Dict[str, str]],
        details: Optional[Dict[str, Any]] = None
    ) -> Anomaly:
        """Create an anomaly object."""
        return Anomaly(
            anomaly_id=f"anomaly-{uuid.uuid4().hex[:8]}",
            metric_name=metric_name,
            timestamp=time.time(),
            expected_value=expected,
            actual_value=actual,
            deviation_sigma=deviation,
            severity=severity,
            method=method,
            details=details or {},
            labels=labels or {},
        )

    def _get_baseline_key(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]]
    ) -> str:
        """Generate unique key for baseline storage."""
        if labels:
            label_str = json.dumps(sorted(labels.items()))
            return f"{metric_name}:{label_str}"
        return metric_name


# Singleton instance
_detector: Optional[AnomalyDetector] = None


def get_detector() -> AnomalyDetector:
    """Get or create the anomaly detector singleton.

    Returns:
        AnomalyDetector instance
    """
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description="Anomaly Detector (Step 257)")
    parser.add_argument("--detect", metavar="NAME=VALUE", help="Detect anomaly in value")
    parser.add_argument("--method", default="z_score", help="Detection method")
    parser.add_argument("--stats", metavar="NAME", help="Show baseline stats for metric")
    parser.add_argument("--status", action="store_true", help="Show detector status")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    detector = get_detector()

    if args.detect:
        name, value = args.detect.split("=")
        method = DetectionMethod(args.method)
        anomaly = detector.detect(name, float(value), None, method)
        if anomaly:
            if args.json:
                print(json.dumps(anomaly.to_dict(), indent=2))
            else:
                print(f"Anomaly detected: {anomaly.severity.value}")
                print(f"  Metric: {anomaly.metric_name}")
                print(f"  Expected: {anomaly.expected_value:.2f}")
                print(f"  Actual: {anomaly.actual_value:.2f}")
                print(f"  Deviation: {anomaly.deviation_sigma:.2f} sigma")
        else:
            print("No anomaly detected")

    if args.stats:
        stats = detector.get_baseline_stats(args.stats)
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Baseline stats for {args.stats}:")
            for k, v in stats.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.status:
        status = detector.get_statistics()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("Anomaly Detector Status:")
            for k, v in status.items():
                print(f"  {k}: {v}")

    if args.demo:
        print("Running anomaly detection demo...")
        # Generate normal data
        for i in range(100):
            value = random.gauss(50, 5)  # Mean=50, Stddev=5
            detector.detect("demo.metric", value)

        # Inject anomaly
        anomaly = detector.detect("demo.metric", 80)  # 6 sigma away
        if anomaly:
            print(f"Anomaly detected at value=80: {anomaly.severity.value}")
            print(f"  Deviation: {anomaly.deviation_sigma:.2f} sigma")
        else:
            print("No anomaly detected (unexpected)")

        stats = detector.get_baseline_stats("demo.metric")
        print(f"Baseline: mean={stats['mean']:.2f}, stddev={stats['stddev']:.2f}")
