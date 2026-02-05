#!/usr/bin/env python3
"""
Monitor Prediction Engine - Step 276

Predictive alerting and anomaly forecasting.

PBTSO Phase: RESEARCH

Bus Topics:
- monitor.prediction.forecast (subscribed)
- monitor.prediction.alert (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import math
import os
import socket
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class PredictionType(Enum):
    """Types of predictions."""
    THRESHOLD_BREACH = "threshold_breach"
    TREND_CONTINUATION = "trend_continuation"
    SEASONALITY = "seasonality"
    CAPACITY_EXHAUSTION = "capacity_exhaustion"
    FAILURE_PROBABILITY = "failure_probability"


class AlertLevel(Enum):
    """Predictive alert levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MetricDataPoint:
    """A metric data point for prediction.

    Attributes:
        timestamp: Data point timestamp
        value: Metric value
        labels: Additional labels
    """
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Prediction:
    """A prediction result.

    Attributes:
        prediction_id: Unique prediction ID
        metric_name: Metric being predicted
        prediction_type: Type of prediction
        current_value: Current metric value
        predicted_value: Predicted future value
        prediction_time: When prediction is for
        confidence: Prediction confidence (0-1)
        threshold: Relevant threshold
        time_to_threshold_s: Time until threshold breach
        trend: Trend direction
        model_used: Model name used
        metadata: Additional metadata
    """
    prediction_id: str
    metric_name: str
    prediction_type: PredictionType
    current_value: float
    predicted_value: float
    prediction_time: float
    confidence: float = 0.0
    threshold: Optional[float] = None
    time_to_threshold_s: Optional[float] = None
    trend: str = "stable"
    model_used: str = "linear"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction_id": self.prediction_id,
            "metric_name": self.metric_name,
            "prediction_type": self.prediction_type.value,
            "current_value": self.current_value,
            "predicted_value": self.predicted_value,
            "prediction_time": self.prediction_time,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "time_to_threshold_s": self.time_to_threshold_s,
            "trend": self.trend,
            "model_used": self.model_used,
            "metadata": self.metadata,
        }


@dataclass
class PredictiveAlert:
    """A predictive alert.

    Attributes:
        alert_id: Unique alert ID
        prediction: Source prediction
        level: Alert level
        message: Alert message
        recommended_action: Recommended action
        created_at: Creation timestamp
        acknowledged: Whether acknowledged
    """
    alert_id: str
    prediction: Prediction
    level: AlertLevel
    message: str
    recommended_action: str = ""
    created_at: float = field(default_factory=time.time)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "prediction": self.prediction.to_dict(),
            "level": self.level.value,
            "message": self.message,
            "recommended_action": self.recommended_action,
            "created_at": self.created_at,
            "acknowledged": self.acknowledged,
        }


@dataclass
class ThresholdConfig:
    """Threshold configuration for a metric.

    Attributes:
        metric_name: Metric name
        warning_threshold: Warning level
        critical_threshold: Critical level
        direction: Threshold direction (above/below)
    """
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    direction: str = "above"  # "above" or "below"


class PredictionEngine:
    """
    Predictive alerting engine.

    The engine:
    - Forecasts metric values
    - Predicts threshold breaches
    - Generates predictive alerts
    - Supports multiple prediction models

    Example:
        engine = PredictionEngine()

        # Add metric data
        for point in data_points:
            engine.add_data_point("cpu.usage", point)

        # Configure threshold
        engine.set_threshold("cpu.usage", warning=70, critical=90)

        # Get predictions
        predictions = engine.predict("cpu.usage", horizon_s=3600)

        # Check for alerts
        alerts = engine.get_predictive_alerts()
    """

    BUS_TOPICS = {
        "forecast": "monitor.prediction.forecast",
        "alert": "monitor.prediction.alert",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        data_retention_hours: int = 24,
        bus_dir: Optional[str] = None,
    ):
        """Initialize prediction engine.

        Args:
            data_retention_hours: Hours of data to retain
            bus_dir: Bus directory
        """
        self._data_retention_hours = data_retention_hours
        self._metric_data: Dict[str, List[MetricDataPoint]] = defaultdict(list)
        self._thresholds: Dict[str, ThresholdConfig] = {}
        self._predictions: Dict[str, Prediction] = {}
        self._alerts: Dict[str, PredictiveAlert] = {}
        self._prediction_history: List[Prediction] = []
        self._last_heartbeat = time.time()

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def add_data_point(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[float] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Add a metric data point.

        Args:
            metric_name: Metric name
            value: Metric value
            timestamp: Data point timestamp
            labels: Additional labels
        """
        point = MetricDataPoint(
            timestamp=timestamp or time.time(),
            value=value,
            labels=labels or {},
        )

        self._metric_data[metric_name].append(point)

        # Prune old data
        cutoff = time.time() - (self._data_retention_hours * 3600)
        self._metric_data[metric_name] = [
            p for p in self._metric_data[metric_name]
            if p.timestamp >= cutoff
        ]

    def add_data_points(
        self,
        metric_name: str,
        points: List[MetricDataPoint],
    ) -> int:
        """Add multiple data points.

        Args:
            metric_name: Metric name
            points: Data points

        Returns:
            Number of points added
        """
        for point in points:
            self.add_data_point(
                metric_name,
                point.value,
                point.timestamp,
                point.labels,
            )
        return len(points)

    def set_threshold(
        self,
        metric_name: str,
        warning: float,
        critical: float,
        direction: str = "above",
    ) -> None:
        """Set threshold configuration for a metric.

        Args:
            metric_name: Metric name
            warning: Warning threshold
            critical: Critical threshold
            direction: Threshold direction
        """
        self._thresholds[metric_name] = ThresholdConfig(
            metric_name=metric_name,
            warning_threshold=warning,
            critical_threshold=critical,
            direction=direction,
        )

    def get_threshold(self, metric_name: str) -> Optional[ThresholdConfig]:
        """Get threshold configuration.

        Args:
            metric_name: Metric name

        Returns:
            Threshold config or None
        """
        return self._thresholds.get(metric_name)

    def predict(
        self,
        metric_name: str,
        horizon_s: int = 3600,
        model: str = "auto",
    ) -> Optional[Prediction]:
        """Predict future metric value.

        Args:
            metric_name: Metric to predict
            horizon_s: Prediction horizon in seconds
            model: Prediction model to use

        Returns:
            Prediction or None
        """
        data = self._metric_data.get(metric_name)
        if not data or len(data) < 3:
            return None

        # Sort by timestamp
        sorted_data = sorted(data, key=lambda p: p.timestamp)
        values = [p.value for p in sorted_data]
        timestamps = [p.timestamp for p in sorted_data]

        current_value = values[-1]
        current_time = timestamps[-1]
        prediction_time = current_time + horizon_s

        # Select model
        if model == "auto":
            model = self._select_model(values)

        # Predict based on model
        if model == "linear":
            predicted_value = self._predict_linear(timestamps, values, prediction_time)
        elif model == "exponential":
            predicted_value = self._predict_exponential(timestamps, values, prediction_time)
        elif model == "average":
            predicted_value = self._predict_moving_average(values)
        else:
            predicted_value = self._predict_linear(timestamps, values, prediction_time)

        # Calculate confidence
        confidence = self._calculate_confidence(values, model)

        # Determine trend
        trend = self._calculate_trend(values)

        # Check threshold
        threshold_config = self._thresholds.get(metric_name)
        time_to_threshold = None
        threshold = None

        if threshold_config:
            threshold = threshold_config.critical_threshold
            time_to_threshold = self._time_to_threshold(
                timestamps, values, threshold, threshold_config.direction
            )

        prediction = Prediction(
            prediction_id=f"pred-{uuid.uuid4().hex[:8]}",
            metric_name=metric_name,
            prediction_type=PredictionType.THRESHOLD_BREACH if time_to_threshold else PredictionType.TREND_CONTINUATION,
            current_value=current_value,
            predicted_value=predicted_value,
            prediction_time=prediction_time,
            confidence=confidence,
            threshold=threshold,
            time_to_threshold_s=time_to_threshold,
            trend=trend,
            model_used=model,
            metadata={
                "data_points": len(values),
                "horizon_s": horizon_s,
            },
        )

        self._predictions[prediction.prediction_id] = prediction
        self._prediction_history.append(prediction)

        if len(self._prediction_history) > 1000:
            self._prediction_history = self._prediction_history[-1000:]

        # Check for alerts
        self._check_for_alert(prediction)

        return prediction

    def predict_all(
        self,
        horizon_s: int = 3600,
    ) -> List[Prediction]:
        """Predict all configured metrics.

        Args:
            horizon_s: Prediction horizon

        Returns:
            List of predictions
        """
        predictions = []

        for metric_name in self._thresholds.keys():
            prediction = self.predict(metric_name, horizon_s)
            if prediction:
                predictions.append(prediction)

        return predictions

    def get_predictive_alerts(
        self,
        level: Optional[AlertLevel] = None,
        unacknowledged_only: bool = False,
    ) -> List[PredictiveAlert]:
        """Get predictive alerts.

        Args:
            level: Filter by level
            unacknowledged_only: Only unacknowledged alerts

        Returns:
            Alerts
        """
        alerts = list(self._alerts.values())

        if level:
            alerts = [a for a in alerts if a.level == level]

        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        alerts.sort(key=lambda a: a.created_at, reverse=True)
        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert ID

        Returns:
            True if acknowledged
        """
        alert = self._alerts.get(alert_id)
        if alert:
            alert.acknowledged = True
            return True
        return False

    def get_metric_stats(self, metric_name: str) -> Dict[str, Any]:
        """Get statistics for a metric.

        Args:
            metric_name: Metric name

        Returns:
            Statistics
        """
        data = self._metric_data.get(metric_name)
        if not data:
            return {}

        values = [p.value for p in data]

        return {
            "metric_name": metric_name,
            "data_points": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "current": values[-1] if values else 0,
            "trend": self._calculate_trend(values),
            "has_threshold": metric_name in self._thresholds,
        }

    def list_metrics(self) -> List[str]:
        """List metrics with data.

        Returns:
            Metric names
        """
        return list(self._metric_data.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Statistics
        """
        total_alerts = len(self._alerts)
        unacked = sum(1 for a in self._alerts.values() if not a.acknowledged)

        return {
            "metrics_tracked": len(self._metric_data),
            "thresholds_configured": len(self._thresholds),
            "total_predictions": len(self._prediction_history),
            "active_alerts": total_alerts,
            "unacknowledged_alerts": unacked,
            "total_data_points": sum(len(d) for d in self._metric_data.values()),
        }

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "prediction_engine",
                "status": "healthy",
                "metrics": len(self._metric_data),
            }
        )

        return True

    def _select_model(self, values: List[float]) -> str:
        """Select best model for data.

        Args:
            values: Data values

        Returns:
            Model name
        """
        if len(values) < 10:
            return "average"

        # Check for strong trend
        trend = self._calculate_trend(values)
        if trend in ("increasing", "decreasing"):
            return "linear"

        return "average"

    def _predict_linear(
        self,
        timestamps: List[float],
        values: List[float],
        target_time: float,
    ) -> float:
        """Linear regression prediction.

        Args:
            timestamps: Data timestamps
            values: Data values
            target_time: Target timestamp

        Returns:
            Predicted value
        """
        n = len(values)
        if n < 2:
            return values[-1] if values else 0

        # Simple linear regression
        x_mean = sum(timestamps) / n
        y_mean = sum(values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(timestamps, values))
        denominator = sum((x - x_mean) ** 2 for x in timestamps)

        if denominator == 0:
            return y_mean

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        return slope * target_time + intercept

    def _predict_exponential(
        self,
        timestamps: List[float],
        values: List[float],
        target_time: float,
    ) -> float:
        """Exponential smoothing prediction.

        Args:
            timestamps: Data timestamps
            values: Data values
            target_time: Target timestamp

        Returns:
            Predicted value
        """
        if not values:
            return 0

        alpha = 0.3
        smoothed = values[0]

        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed

        # Extend trend
        if len(values) >= 2:
            trend = values[-1] - values[-2]
            time_diff = target_time - timestamps[-1]
            avg_interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1) if len(timestamps) > 1 else 1
            steps = time_diff / avg_interval if avg_interval > 0 else 1

            return smoothed + trend * steps

        return smoothed

    def _predict_moving_average(
        self,
        values: List[float],
        window: int = 10,
    ) -> float:
        """Moving average prediction.

        Args:
            values: Data values
            window: Window size

        Returns:
            Predicted value
        """
        if not values:
            return 0

        recent = values[-window:]
        return sum(recent) / len(recent)

    def _calculate_confidence(
        self,
        values: List[float],
        model: str,
    ) -> float:
        """Calculate prediction confidence.

        Args:
            values: Data values
            model: Model used

        Returns:
            Confidence score
        """
        if len(values) < 5:
            return 0.3

        # Based on data stability and quantity
        n = len(values)
        quantity_factor = min(1.0, n / 100)

        # Calculate coefficient of variation
        mean = sum(values) / n
        if mean == 0:
            return 0.5

        std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
        cv = std / abs(mean)

        stability_factor = max(0.2, 1.0 - cv)

        return quantity_factor * 0.4 + stability_factor * 0.6

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction.

        Args:
            values: Data values

        Returns:
            Trend description
        """
        if len(values) < 3:
            return "stable"

        # Compare recent values to older values
        recent = values[-len(values)//3:]
        older = values[:len(values)//3]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        if older_avg == 0:
            return "stable"

        change = (recent_avg - older_avg) / abs(older_avg)

        if change > 0.1:
            return "increasing"
        elif change < -0.1:
            return "decreasing"
        else:
            return "stable"

    def _time_to_threshold(
        self,
        timestamps: List[float],
        values: List[float],
        threshold: float,
        direction: str,
    ) -> Optional[float]:
        """Calculate time until threshold is breached.

        Args:
            timestamps: Data timestamps
            values: Data values
            threshold: Threshold value
            direction: Threshold direction

        Returns:
            Time in seconds or None
        """
        if len(values) < 2:
            return None

        current = values[-1]

        # Check if already breached
        if direction == "above" and current >= threshold:
            return 0
        if direction == "below" and current <= threshold:
            return 0

        # Calculate trend
        recent = values[-min(10, len(values)):]
        recent_ts = timestamps[-min(10, len(timestamps)):]

        if len(recent) < 2:
            return None

        # Simple linear projection
        time_span = recent_ts[-1] - recent_ts[0]
        if time_span == 0:
            return None

        value_change = recent[-1] - recent[0]
        rate = value_change / time_span

        if rate == 0:
            return None

        # Calculate time to threshold
        gap = threshold - current
        if direction == "above" and rate > 0:
            return gap / rate
        elif direction == "below" and rate < 0:
            return gap / rate

        return None

    def _check_for_alert(self, prediction: Prediction) -> Optional[PredictiveAlert]:
        """Check if prediction warrants an alert.

        Args:
            prediction: Prediction to check

        Returns:
            Alert if warranted
        """
        threshold_config = self._thresholds.get(prediction.metric_name)
        if not threshold_config:
            return None

        # Check time to threshold
        if prediction.time_to_threshold_s is None:
            return None

        # Determine alert level based on time to breach
        if prediction.time_to_threshold_s <= 300:  # 5 minutes
            level = AlertLevel.CRITICAL
        elif prediction.time_to_threshold_s <= 1800:  # 30 minutes
            level = AlertLevel.WARNING
        elif prediction.time_to_threshold_s <= 7200:  # 2 hours
            level = AlertLevel.INFO
        else:
            return None

        # Create alert
        alert = PredictiveAlert(
            alert_id=f"palert-{uuid.uuid4().hex[:8]}",
            prediction=prediction,
            level=level,
            message=f"Predicted {prediction.metric_name} will breach {prediction.threshold} in {prediction.time_to_threshold_s:.0f}s",
            recommended_action=self._get_recommended_action(prediction, threshold_config),
        )

        self._alerts[alert.alert_id] = alert

        self._emit_bus_event(
            self.BUS_TOPICS["alert"],
            alert.to_dict(),
            level=level.value
        )

        return alert

    def _get_recommended_action(
        self,
        prediction: Prediction,
        config: ThresholdConfig,
    ) -> str:
        """Get recommended action for prediction.

        Args:
            prediction: Prediction
            config: Threshold config

        Returns:
            Recommended action
        """
        metric = prediction.metric_name

        if "cpu" in metric.lower():
            return "Consider scaling compute resources or optimizing CPU-intensive processes"
        elif "memory" in metric.lower():
            return "Review memory usage patterns and consider scaling or optimizing memory-intensive operations"
        elif "disk" in metric.lower():
            return "Free up disk space or expand storage capacity"
        elif "latency" in metric.lower():
            return "Investigate latency sources and optimize performance"
        else:
            return f"Take action to prevent {metric} from breaching threshold"

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id


# Singleton instance
_engine: Optional[PredictionEngine] = None


def get_engine() -> PredictionEngine:
    """Get or create the prediction engine singleton.

    Returns:
        PredictionEngine instance
    """
    global _engine
    if _engine is None:
        _engine = PredictionEngine()
    return _engine


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Prediction Engine (Step 276)")
    parser.add_argument("--predict", metavar="METRIC", help="Predict metric")
    parser.add_argument("--horizon", type=int, default=3600, help="Prediction horizon in seconds")
    parser.add_argument("--set-threshold", metavar="METRIC", help="Set threshold for metric")
    parser.add_argument("--warning", type=float, help="Warning threshold")
    parser.add_argument("--critical", type=float, help="Critical threshold")
    parser.add_argument("--alerts", action="store_true", help="Show predictive alerts")
    parser.add_argument("--metrics", action="store_true", help="List metrics")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    engine = get_engine()

    if args.set_threshold:
        engine.set_threshold(
            args.set_threshold,
            warning=args.warning or 70,
            critical=args.critical or 90,
        )
        print(f"Set threshold for {args.set_threshold}")

    if args.predict:
        prediction = engine.predict(args.predict, horizon_s=args.horizon)
        if prediction:
            if args.json:
                print(json.dumps(prediction.to_dict(), indent=2))
            else:
                print(f"Prediction: {prediction.prediction_id}")
                print(f"  Current: {prediction.current_value:.2f}")
                print(f"  Predicted: {prediction.predicted_value:.2f}")
                print(f"  Confidence: {prediction.confidence:.2f}")
                print(f"  Trend: {prediction.trend}")
                if prediction.time_to_threshold_s:
                    print(f"  Time to threshold: {prediction.time_to_threshold_s:.0f}s")
        else:
            print(f"No prediction available for {args.predict}")

    if args.alerts:
        alerts = engine.get_predictive_alerts()
        if args.json:
            print(json.dumps([a.to_dict() for a in alerts], indent=2))
        else:
            print(f"Predictive Alerts ({len(alerts)}):")
            for a in alerts:
                acked = "ACK" if a.acknowledged else "NEW"
                print(f"  [{a.level.value}] [{acked}] {a.message}")

    if args.metrics:
        metrics = engine.list_metrics()
        if args.json:
            print(json.dumps(metrics))
        else:
            print("Metrics:")
            for m in metrics:
                stats = engine.get_metric_stats(m)
                print(f"  {m}: {stats.get('data_points', 0)} points, trend={stats.get('trend', 'unknown')}")

    if args.stats:
        stats = engine.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Prediction Engine Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
