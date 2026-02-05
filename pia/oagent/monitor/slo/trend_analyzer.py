#!/usr/bin/env python3
"""
Trend Analyzer - Step 267

Performs long-term trend analysis on metrics and resource usage.

PBTSO Phase: VERIFY

Bus Topics:
- monitor.trends.analyze (emitted)
- monitor.trends.detected (emitted)
- monitor.trends.alert (emitted)

Protocol: DKIN v30, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import json
import math
import os
import socket
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class TrendType(Enum):
    """Types of trends."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    CYCLICAL = "cyclical"
    ANOMALOUS = "anomalous"


class TrendPeriod(Enum):
    """Trend analysis periods."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class TrendSignificance(Enum):
    """Trend significance levels."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


@dataclass
class DataPoint:
    """Data point for trend analysis.

    Attributes:
        metric: Metric name
        value: Metric value
        timestamp: Data point timestamp
        labels: Optional labels
    """
    metric: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SeasonalPattern:
    """Seasonal pattern detection result.

    Attributes:
        metric: Metric name
        period: Pattern period
        amplitude: Pattern amplitude
        phase: Pattern phase offset
        strength: Pattern strength (0-1)
        peak_time: Time of peak within period
        trough_time: Time of trough within period
    """
    metric: str
    period: TrendPeriod
    amplitude: float
    phase: float
    strength: float
    peak_time: str
    trough_time: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric,
            "period": self.period.value,
            "amplitude": self.amplitude,
            "phase": self.phase,
            "strength": self.strength,
            "peak_time": self.peak_time,
            "trough_time": self.trough_time,
        }


@dataclass
class Trend:
    """Detected trend.

    Attributes:
        metric: Metric name
        trend_type: Type of trend
        significance: Trend significance
        slope: Trend slope (rate of change)
        r_squared: Goodness of fit (0-1)
        start_value: Value at start of period
        end_value: Value at end of period
        change_percent: Percentage change
        period_days: Analysis period in days
        confidence_interval: 95% confidence interval
        timestamp: Analysis timestamp
    """
    metric: str
    trend_type: TrendType
    significance: TrendSignificance
    slope: float
    r_squared: float
    start_value: float
    end_value: float
    change_percent: float
    period_days: int
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric,
            "trend_type": self.trend_type.value,
            "significance": self.significance.value,
            "slope": self.slope,
            "r_squared": self.r_squared,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "change_percent": self.change_percent,
            "period_days": self.period_days,
            "confidence_interval": list(self.confidence_interval),
            "timestamp": self.timestamp,
        }


@dataclass
class TrendAlert:
    """Trend-based alert.

    Attributes:
        metric: Metric name
        trend: Detected trend
        message: Alert message
        severity: Alert severity
        timestamp: Alert timestamp
        alert_id: Unique alert ID
    """
    metric: str
    trend: Trend
    message: str
    severity: str
    timestamp: float = field(default_factory=time.time)
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "metric": self.metric,
            "trend": self.trend.to_dict(),
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp,
        }


@dataclass
class TrendSummary:
    """Trend analysis summary.

    Attributes:
        metric: Metric name
        trends: Trends by period
        seasonal_patterns: Detected seasonal patterns
        forecast: Short-term forecast
        anomalies: Detected anomalies
    """
    metric: str
    trends: Dict[str, Trend] = field(default_factory=dict)
    seasonal_patterns: List[SeasonalPattern] = field(default_factory=list)
    forecast: List[DataPoint] = field(default_factory=list)
    anomalies: List[DataPoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric,
            "trends": {k: v.to_dict() for k, v in self.trends.items()},
            "seasonal_patterns": [p.to_dict() for p in self.seasonal_patterns],
            "forecast": [f.to_dict() for f in self.forecast],
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


class TrendAnalyzer:
    """
    Analyze long-term trends in metrics.

    The analyzer:
    - Detects increasing/decreasing trends
    - Identifies seasonal patterns
    - Calculates statistical significance
    - Generates forecasts
    - Alerts on significant trends

    Example:
        analyzer = TrendAnalyzer()

        # Record data points
        for point in data:
            analyzer.record(point)

        # Analyze trends
        trends = analyzer.analyze("cpu.usage", period_days=30)
        print(f"Trend: {trends.trend_type.value}")

        # Detect seasonal patterns
        patterns = analyzer.detect_seasonality("requests.count")
        for pattern in patterns:
            print(f"Pattern: {pattern.period.value}, strength: {pattern.strength}")
    """

    BUS_TOPICS = {
        "analyze": "monitor.trends.analyze",
        "detected": "monitor.trends.detected",
        "alert": "monitor.trends.alert",
    }

    def __init__(
        self,
        history_days: int = 90,
        min_data_points: int = 10,
        bus_dir: Optional[str] = None,
    ):
        """Initialize trend analyzer.

        Args:
            history_days: Days of history to retain
            min_data_points: Minimum points for analysis
            bus_dir: Bus directory
        """
        self.history_days = history_days
        self.min_data_points = min_data_points

        # Data storage
        self._data: Dict[str, List[DataPoint]] = defaultdict(list)
        self._trend_cache: Dict[str, Trend] = {}

        # Callbacks
        self._alert_callbacks: List[Callable[[TrendAlert], None]] = []

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, point: DataPoint) -> bool:
        """Record a data point.

        Args:
            point: Data point

        Returns:
            True if recorded
        """
        self._data[point.metric].append(point)

        # Prune old data
        cutoff = time.time() - (self.history_days * 86400)
        self._data[point.metric] = [
            p for p in self._data[point.metric]
            if p.timestamp >= cutoff
        ]

        return True

    def record_batch(self, points: List[DataPoint]) -> int:
        """Record multiple data points.

        Args:
            points: Data points

        Returns:
            Number recorded
        """
        count = 0
        for point in points:
            if self.record(point):
                count += 1
        return count

    def analyze(
        self,
        metric: str,
        period_days: int = 30
    ) -> Optional[Trend]:
        """Analyze trend for a metric.

        Args:
            metric: Metric name
            period_days: Analysis period in days

        Returns:
            Detected trend or None
        """
        data = self._get_data(metric, period_days)

        if len(data) < self.min_data_points:
            return None

        # Calculate linear regression
        slope, intercept, r_squared = self._linear_regression(data)

        # Determine trend type
        if r_squared < 0.1:
            trend_type = TrendType.STABLE
            significance = TrendSignificance.NONE
        else:
            if slope > 0:
                trend_type = TrendType.INCREASING
            elif slope < 0:
                trend_type = TrendType.DECREASING
            else:
                trend_type = TrendType.STABLE

            # Determine significance
            if r_squared > 0.7:
                significance = TrendSignificance.STRONG
            elif r_squared > 0.4:
                significance = TrendSignificance.MODERATE
            elif r_squared > 0.1:
                significance = TrendSignificance.WEAK
            else:
                significance = TrendSignificance.NONE

        # Calculate change
        start_value = data[0].value
        end_value = data[-1].value
        change_percent = 100.0 * (end_value - start_value) / start_value if start_value != 0 else 0.0

        # Calculate confidence interval
        ci = self._calculate_confidence_interval(data, slope, intercept)

        trend = Trend(
            metric=metric,
            trend_type=trend_type,
            significance=significance,
            slope=slope,
            r_squared=r_squared,
            start_value=start_value,
            end_value=end_value,
            change_percent=change_percent,
            period_days=period_days,
            confidence_interval=ci,
        )

        self._trend_cache[metric] = trend

        # Emit event
        self._emit_bus_event(
            self.BUS_TOPICS["detected"],
            trend.to_dict()
        )

        # Check for alerts
        self._check_trend_alerts(trend)

        return trend

    def analyze_all(self, period_days: int = 30) -> Dict[str, Trend]:
        """Analyze trends for all metrics.

        Args:
            period_days: Analysis period

        Returns:
            Dictionary of trends
        """
        results = {}
        for metric in self._data:
            trend = self.analyze(metric, period_days)
            if trend:
                results[metric] = trend
        return results

    def detect_seasonality(
        self,
        metric: str,
        periods: Optional[List[TrendPeriod]] = None
    ) -> List[SeasonalPattern]:
        """Detect seasonal patterns.

        Args:
            metric: Metric name
            periods: Periods to analyze

        Returns:
            Detected patterns
        """
        data = self._get_data(metric, self.history_days)

        if len(data) < self.min_data_points:
            return []

        periods = periods or [TrendPeriod.DAILY, TrendPeriod.WEEKLY]
        patterns = []

        for period in periods:
            pattern = self._detect_period_pattern(metric, data, period)
            if pattern and pattern.strength > 0.3:
                patterns.append(pattern)

        return patterns

    def get_summary(self, metric: str) -> Optional[TrendSummary]:
        """Get complete trend summary for a metric.

        Args:
            metric: Metric name

        Returns:
            Trend summary or None
        """
        if metric not in self._data:
            return None

        # Analyze at different periods
        trends = {}
        for period in [7, 14, 30, 60]:
            trend = self.analyze(metric, period)
            if trend:
                trends[f"{period}d"] = trend

        # Detect seasonality
        patterns = self.detect_seasonality(metric)

        # Generate forecast
        forecast = self._generate_forecast(metric, days_ahead=7)

        # Find anomalies
        anomalies = self._detect_anomalies(metric)

        return TrendSummary(
            metric=metric,
            trends=trends,
            seasonal_patterns=patterns,
            forecast=forecast,
            anomalies=anomalies,
        )

    def get_all_metrics(self) -> List[str]:
        """Get list of tracked metrics.

        Returns:
            Metric names
        """
        return list(self._data.keys())

    def get_data(
        self,
        metric: str,
        window_days: int = 30
    ) -> List[DataPoint]:
        """Get data points for a metric.

        Args:
            metric: Metric name
            window_days: Time window

        Returns:
            Data points
        """
        return self._get_data(metric, window_days)

    def get_aggregated_data(
        self,
        metric: str,
        period: TrendPeriod,
        window_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get aggregated data by period.

        Args:
            metric: Metric name
            period: Aggregation period
            window_days: Time window

        Returns:
            Aggregated data
        """
        data = self._get_data(metric, window_days)

        # Group by period
        grouped: Dict[str, List[float]] = defaultdict(list)

        for point in data:
            dt = datetime.fromtimestamp(point.timestamp)
            if period == TrendPeriod.HOURLY:
                key = dt.strftime("%Y-%m-%d %H:00")
            elif period == TrendPeriod.DAILY:
                key = dt.strftime("%Y-%m-%d")
            elif period == TrendPeriod.WEEKLY:
                key = dt.strftime("%Y-W%W")
            else:
                key = dt.strftime("%Y-%m")

            grouped[key].append(point.value)

        # Calculate aggregates
        result = []
        for key, values in sorted(grouped.items()):
            result.append({
                "period": key,
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            })

        return result

    def compare_periods(
        self,
        metric: str,
        period1_days: int,
        period2_days: int
    ) -> Optional[Dict[str, Any]]:
        """Compare two time periods.

        Args:
            metric: Metric name
            period1_days: First period (most recent)
            period2_days: Second period (comparison)

        Returns:
            Comparison result or None
        """
        now = time.time()

        # Get data for period 1 (most recent)
        cutoff1 = now - (period1_days * 86400)
        period1_data = [
            p.value for p in self._data.get(metric, [])
            if p.timestamp >= cutoff1
        ]

        # Get data for period 2 (previous period)
        cutoff2_start = cutoff1 - (period2_days * 86400)
        cutoff2_end = cutoff1
        period2_data = [
            p.value for p in self._data.get(metric, [])
            if cutoff2_start <= p.timestamp < cutoff2_end
        ]

        if not period1_data or not period2_data:
            return None

        avg1 = sum(period1_data) / len(period1_data)
        avg2 = sum(period2_data) / len(period2_data)

        change = avg1 - avg2
        change_percent = 100.0 * change / avg2 if avg2 != 0 else 0.0

        return {
            "metric": metric,
            "period1_days": period1_days,
            "period2_days": period2_days,
            "period1_avg": avg1,
            "period2_avg": avg2,
            "change": change,
            "change_percent": change_percent,
            "improved": change > 0,  # Assuming higher is better
        }

    def register_alert_callback(
        self,
        callback: Callable[[TrendAlert], None]
    ) -> None:
        """Register alert callback.

        Args:
            callback: Callback function
        """
        self._alert_callbacks.append(callback)

    def _get_data(self, metric: str, window_days: int) -> List[DataPoint]:
        """Get data within window.

        Args:
            metric: Metric name
            window_days: Time window

        Returns:
            Data points sorted by timestamp
        """
        cutoff = time.time() - (window_days * 86400)
        data = [
            p for p in self._data.get(metric, [])
            if p.timestamp >= cutoff
        ]
        return sorted(data, key=lambda p: p.timestamp)

    def _linear_regression(
        self,
        data: List[DataPoint]
    ) -> Tuple[float, float, float]:
        """Calculate linear regression.

        Args:
            data: Data points

        Returns:
            (slope, intercept, r_squared)
        """
        n = len(data)
        if n < 2:
            return 0.0, 0.0, 0.0

        x_values = [(p.timestamp - data[0].timestamp) / 86400 for p in data]  # Days
        y_values = [p.value for p in data]

        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0, y_mean, 0.0

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        # Calculate R-squared
        y_pred = [intercept + slope * x for x in x_values]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(y_values, y_pred))
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return slope, intercept, max(0, r_squared)

    def _calculate_confidence_interval(
        self,
        data: List[DataPoint],
        slope: float,
        intercept: float
    ) -> Tuple[float, float]:
        """Calculate 95% confidence interval for slope.

        Args:
            data: Data points
            slope: Regression slope
            intercept: Regression intercept

        Returns:
            (lower, upper) bounds
        """
        n = len(data)
        if n < 3:
            return (slope, slope)

        x_values = [(p.timestamp - data[0].timestamp) / 86400 for p in data]
        y_values = [p.value for p in data]
        y_pred = [intercept + slope * x for x in x_values]

        # Calculate standard error
        mse = sum((y - yp) ** 2 for y, yp in zip(y_values, y_pred)) / (n - 2)
        x_mean = sum(x_values) / n
        ss_x = sum((x - x_mean) ** 2 for x in x_values)

        if ss_x == 0:
            return (slope, slope)

        se = math.sqrt(mse / ss_x)

        # 95% CI (approximately 1.96 for large n)
        t_value = 1.96 if n > 30 else 2.0
        margin = t_value * se

        return (slope - margin, slope + margin)

    def _detect_period_pattern(
        self,
        metric: str,
        data: List[DataPoint],
        period: TrendPeriod
    ) -> Optional[SeasonalPattern]:
        """Detect pattern for a specific period.

        Args:
            metric: Metric name
            data: Data points
            period: Period to detect

        Returns:
            Seasonal pattern or None
        """
        if period == TrendPeriod.HOURLY:
            bucket_size = 3600
            num_buckets = 24
        elif period == TrendPeriod.DAILY:
            bucket_size = 86400
            num_buckets = 7
        elif period == TrendPeriod.WEEKLY:
            bucket_size = 86400 * 7
            num_buckets = 4
        else:
            bucket_size = 86400 * 30
            num_buckets = 12

        # Group data by bucket
        buckets: Dict[int, List[float]] = defaultdict(list)
        for point in data:
            if period == TrendPeriod.HOURLY:
                bucket = int(datetime.fromtimestamp(point.timestamp).hour)
            elif period == TrendPeriod.DAILY:
                bucket = int(datetime.fromtimestamp(point.timestamp).weekday())
            elif period == TrendPeriod.WEEKLY:
                bucket = int((point.timestamp % (bucket_size * num_buckets)) // bucket_size)
            else:
                bucket = int(datetime.fromtimestamp(point.timestamp).month) - 1

            buckets[bucket].append(point.value)

        if len(buckets) < 2:
            return None

        # Calculate bucket averages
        bucket_avgs = {k: sum(v) / len(v) for k, v in buckets.items() if v}
        if not bucket_avgs:
            return None

        overall_avg = sum(bucket_avgs.values()) / len(bucket_avgs)

        # Calculate amplitude and strength
        max_bucket = max(bucket_avgs.items(), key=lambda x: x[1])
        min_bucket = min(bucket_avgs.items(), key=lambda x: x[1])

        amplitude = max_bucket[1] - min_bucket[1]

        # Calculate strength (variance explained by pattern)
        total_variance = sum((v - overall_avg) ** 2 for v in bucket_avgs.values())
        if total_variance == 0:
            return None

        strength = amplitude / (2 * (total_variance ** 0.5)) if total_variance > 0 else 0

        # Format peak/trough times
        if period == TrendPeriod.HOURLY:
            peak_time = f"{max_bucket[0]:02d}:00"
            trough_time = f"{min_bucket[0]:02d}:00"
        elif period == TrendPeriod.DAILY:
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            peak_time = days[max_bucket[0]]
            trough_time = days[min_bucket[0]]
        else:
            peak_time = f"Week {max_bucket[0] + 1}"
            trough_time = f"Week {min_bucket[0] + 1}"

        return SeasonalPattern(
            metric=metric,
            period=period,
            amplitude=amplitude,
            phase=max_bucket[0],
            strength=min(1.0, strength),
            peak_time=peak_time,
            trough_time=trough_time,
        )

    def _generate_forecast(
        self,
        metric: str,
        days_ahead: int = 7
    ) -> List[DataPoint]:
        """Generate simple forecast.

        Args:
            metric: Metric name
            days_ahead: Days to forecast

        Returns:
            Forecast data points
        """
        data = self._get_data(metric, 30)
        if len(data) < self.min_data_points:
            return []

        slope, intercept, _ = self._linear_regression(data)
        base_time = data[0].timestamp
        last_x = (data[-1].timestamp - base_time) / 86400

        forecast = []
        for i in range(1, days_ahead + 1):
            future_x = last_x + i
            future_value = intercept + slope * future_x
            forecast.append(DataPoint(
                metric=metric,
                value=future_value,
                timestamp=data[-1].timestamp + (i * 86400),
            ))

        return forecast

    def _detect_anomalies(self, metric: str) -> List[DataPoint]:
        """Detect anomalous data points.

        Args:
            metric: Metric name

        Returns:
            Anomalous points
        """
        data = self._get_data(metric, 30)
        if len(data) < 10:
            return []

        values = [p.value for p in data]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5

        if std_dev == 0:
            return []

        # Find points > 3 standard deviations from mean
        anomalies = [
            p for p in data
            if abs(p.value - mean) > 3 * std_dev
        ]

        return anomalies

    def _check_trend_alerts(self, trend: Trend) -> None:
        """Check trend for alerts.

        Args:
            trend: Detected trend
        """
        if trend.significance not in (TrendSignificance.STRONG, TrendSignificance.MODERATE):
            return

        if abs(trend.change_percent) > 20:
            direction = "increasing" if trend.trend_type == TrendType.INCREASING else "decreasing"
            severity = "warning" if abs(trend.change_percent) < 50 else "error"

            alert = TrendAlert(
                metric=trend.metric,
                trend=trend,
                message=f"Significant {direction} trend detected for {trend.metric}: {trend.change_percent:+.1f}% over {trend.period_days} days",
                severity=severity,
            )

            self._emit_bus_event(
                self.BUS_TOPICS["alert"],
                alert.to_dict(),
                level=severity
            )

            for callback in self._alert_callbacks:
                callback(alert)

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to bus."""
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
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

        return event_id


# Singleton
_analyzer: Optional[TrendAnalyzer] = None


def get_trend_analyzer() -> TrendAnalyzer:
    """Get or create the trend analyzer singleton."""
    global _analyzer
    if _analyzer is None:
        _analyzer = TrendAnalyzer()
    return _analyzer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trend Analyzer (Step 267)")
    parser.add_argument("--metrics", action="store_true", help="List tracked metrics")
    parser.add_argument("--analyze", metavar="METRIC", help="Analyze trend for metric")
    parser.add_argument("--summary", metavar="METRIC", help="Get full summary for metric")
    parser.add_argument("--period", type=int, default=30, help="Analysis period in days")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    analyzer = get_trend_analyzer()

    if args.metrics:
        metrics = analyzer.get_all_metrics()
        if args.json:
            print(json.dumps(metrics))
        else:
            print("Tracked Metrics:")
            for m in metrics:
                print(f"  - {m}")

    if args.analyze:
        trend = analyzer.analyze(args.analyze, args.period)
        if trend:
            if args.json:
                print(json.dumps(trend.to_dict(), indent=2))
            else:
                print(f"Trend Analysis: {args.analyze}")
                print(f"  Type: {trend.trend_type.value}")
                print(f"  Significance: {trend.significance.value}")
                print(f"  Change: {trend.change_percent:+.1f}%")
                print(f"  R-squared: {trend.r_squared:.3f}")
                print(f"  Slope: {trend.slope:.4f}/day")
        else:
            print(f"No trend data for: {args.analyze}")

    if args.summary:
        summary = analyzer.get_summary(args.summary)
        if summary:
            if args.json:
                print(json.dumps(summary.to_dict(), indent=2))
            else:
                print(f"Summary: {args.summary}")
                print("  Trends:")
                for period, trend in summary.trends.items():
                    print(f"    {period}: {trend.trend_type.value} ({trend.change_percent:+.1f}%)")
                if summary.seasonal_patterns:
                    print("  Seasonal Patterns:")
                    for pattern in summary.seasonal_patterns:
                        print(f"    {pattern.period.value}: strength={pattern.strength:.2f}")
                if summary.anomalies:
                    print(f"  Anomalies: {len(summary.anomalies)} detected")
        else:
            print(f"No data for: {args.summary}")
