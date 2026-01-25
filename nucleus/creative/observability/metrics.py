"""
Metrics Module for Observability
================================

Provides metrics collection primitives: Counter, Gauge, Histogram,
and a MetricsCollector for aggregating all metrics.
"""

from __future__ import annotations

import time
import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class Counter:
    """
    A monotonically increasing counter metric.

    Counters track cumulative values that only increase, such as
    request counts, error counts, or bytes processed.

    Attributes:
        name: The metric name.
        labels: Dictionary of label key-value pairs for metric dimensions.

    Example:
        >>> counter = Counter("http_requests_total", {"method": "GET"})
        >>> counter.inc()
        >>> counter.inc(5)
        >>> print(counter.value)
        6.0
    """
    name: str
    labels: Dict[str, str] = field(default_factory=dict)
    _value: float = field(default=0.0, init=False)

    def inc(self, amount: float = 1.0) -> None:
        """
        Increment the counter by the given amount.

        Args:
            amount: The value to add (default 1.0). Must be non-negative.

        Raises:
            ValueError: If amount is negative.
        """
        if amount < 0:
            raise ValueError("Counter increment must be non-negative")
        self._value += amount

    @property
    def value(self) -> float:
        """Return the current counter value."""
        return self._value

    def reset(self) -> None:
        """Reset the counter to zero."""
        self._value = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Export counter as a dictionary."""
        return {
            "name": self.name,
            "type": "counter",
            "labels": self.labels,
            "value": self._value,
        }


@dataclass
class Gauge:
    """
    A gauge metric that can increase and decrease.

    Gauges represent a single numerical value that can go up or down,
    such as temperature, memory usage, or active connections.

    Attributes:
        name: The metric name.
        labels: Dictionary of label key-value pairs for metric dimensions.

    Example:
        >>> gauge = Gauge("active_connections", {"service": "api"})
        >>> gauge.set(10)
        >>> gauge.inc(5)
        >>> gauge.dec(3)
        >>> print(gauge.value)
        12.0
    """
    name: str
    labels: Dict[str, str] = field(default_factory=dict)
    _value: float = field(default=0.0, init=False)

    def set(self, value: float) -> None:
        """
        Set the gauge to a specific value.

        Args:
            value: The new gauge value.
        """
        self._value = value

    def inc(self, amount: float = 1.0) -> None:
        """
        Increment the gauge by the given amount.

        Args:
            amount: The value to add (default 1.0).
        """
        self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """
        Decrement the gauge by the given amount.

        Args:
            amount: The value to subtract (default 1.0).
        """
        self._value -= amount

    @property
    def value(self) -> float:
        """Return the current gauge value."""
        return self._value

    def to_dict(self) -> Dict[str, Any]:
        """Export gauge as a dictionary."""
        return {
            "name": self.name,
            "type": "gauge",
            "labels": self.labels,
            "value": self._value,
        }


@dataclass
class Histogram:
    """
    A histogram for tracking value distributions.

    Histograms track the distribution of values, providing counts
    in configurable buckets along with sum and count aggregates.
    Useful for latency measurements, request sizes, etc.

    Attributes:
        name: The metric name.
        buckets: List of bucket boundaries (upper bounds).
        labels: Dictionary of label key-value pairs for metric dimensions.

    Example:
        >>> hist = Histogram("request_duration_seconds")
        >>> hist.observe(0.1)
        >>> hist.observe(0.5)
        >>> hist.observe(1.2)
        >>> print(hist.mean)
        0.6
    """
    name: str
    buckets: List[float] = field(
        default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    labels: Dict[str, str] = field(default_factory=dict)
    _values: List[float] = field(default_factory=list, init=False)

    def observe(self, value: float) -> None:
        """
        Record an observation in the histogram.

        Args:
            value: The observed value to record.
        """
        self._values.append(value)

    @property
    def count(self) -> int:
        """Return the number of observations."""
        return len(self._values)

    @property
    def sum(self) -> float:
        """Return the sum of all observations."""
        return sum(self._values) if self._values else 0.0

    @property
    def mean(self) -> float:
        """Return the mean of all observations."""
        return self.sum / self.count if self.count > 0 else 0.0

    @property
    def min(self) -> Optional[float]:
        """Return the minimum observed value."""
        return min(self._values) if self._values else None

    @property
    def max(self) -> Optional[float]:
        """Return the maximum observed value."""
        return max(self._values) if self._values else None

    def get_bucket_counts(self) -> Dict[str, int]:
        """
        Get counts for each bucket boundary.

        Returns:
            Dictionary mapping bucket boundary string to count of values <= boundary.
        """
        counts = {}
        for bucket in self.buckets:
            counts[str(bucket)] = sum(1 for v in self._values if v <= bucket)
        counts["+Inf"] = len(self._values)
        return counts

    def percentile(self, p: float) -> Optional[float]:
        """
        Calculate the p-th percentile of observations.

        Args:
            p: Percentile value between 0 and 100.

        Returns:
            The p-th percentile value, or None if no observations.
        """
        if not self._values or p < 0 or p > 100:
            return None
        sorted_values = sorted(self._values)
        k = (len(sorted_values) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_values) else f
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])

    def reset(self) -> None:
        """Reset all observations."""
        self._values.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export histogram as a dictionary."""
        return {
            "name": self.name,
            "type": "histogram",
            "labels": self.labels,
            "count": self.count,
            "sum": self.sum,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "buckets": self.get_bucket_counts(),
        }


class MetricsCollector:
    """
    Central collector for all metrics types.

    The MetricsCollector manages the lifecycle of metrics and provides
    factory methods for creating counters, gauges, and histograms.

    Example:
        >>> collector = MetricsCollector()
        >>> requests = collector.counter("requests_total", method="GET")
        >>> requests.inc()
        >>> latency = collector.histogram("request_latency")
        >>> latency.observe(0.1)
        >>> print(collector.get_all())
    """

    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}

    def counter(self, name: str, **labels: str) -> Counter:
        """
        Get or create a counter with the given name and labels.

        Args:
            name: The counter name.
            **labels: Label key-value pairs.

        Returns:
            The counter instance.
        """
        key = self._make_key(name, labels)
        if key not in self._counters:
            self._counters[key] = Counter(name, labels)
        return self._counters[key]

    def gauge(self, name: str, **labels: str) -> Gauge:
        """
        Get or create a gauge with the given name and labels.

        Args:
            name: The gauge name.
            **labels: Label key-value pairs.

        Returns:
            The gauge instance.
        """
        key = self._make_key(name, labels)
        if key not in self._gauges:
            self._gauges[key] = Gauge(name, labels)
        return self._gauges[key]

    def histogram(self, name: str, buckets: Optional[List[float]] = None, **labels: str) -> Histogram:
        """
        Get or create a histogram with the given name and labels.

        Args:
            name: The histogram name.
            buckets: Optional custom bucket boundaries.
            **labels: Label key-value pairs.

        Returns:
            The histogram instance.
        """
        key = self._make_key(name, labels)
        if key not in self._histograms:
            if buckets:
                self._histograms[key] = Histogram(name, buckets=buckets, labels=labels)
            else:
                self._histograms[key] = Histogram(name, labels=labels)
        return self._histograms[key]

    def _make_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key from name and labels."""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}" if label_str else name

    def get_all(self) -> Dict[str, Any]:
        """
        Get all metrics as a dictionary.

        Returns:
            Dictionary with counters, gauges, and histograms.
        """
        return {
            "counters": {k: v.value for k, v in self._counters.items()},
            "gauges": {k: v.value for k, v in self._gauges.items()},
            "histograms": {
                k: {"count": v.count, "sum": v.sum, "mean": v.mean}
                for k, v in self._histograms.items()
            },
        }

    def export_all(self) -> List[Dict[str, Any]]:
        """
        Export all metrics in a list format suitable for serialization.

        Returns:
            List of metric dictionaries.
        """
        metrics = []
        for counter in self._counters.values():
            metrics.append(counter.to_dict())
        for gauge in self._gauges.values():
            metrics.append(gauge.to_dict())
        for histogram in self._histograms.values():
            metrics.append(histogram.to_dict())
        return metrics

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            String in Prometheus exposition format.
        """
        lines = []

        for counter in self._counters.values():
            labels = ",".join(f'{k}="{v}"' for k, v in counter.labels.items())
            label_str = f"{{{labels}}}" if labels else ""
            lines.append(f"# TYPE {counter.name} counter")
            lines.append(f"{counter.name}{label_str} {counter.value}")

        for gauge in self._gauges.values():
            labels = ",".join(f'{k}="{v}"' for k, v in gauge.labels.items())
            label_str = f"{{{labels}}}" if labels else ""
            lines.append(f"# TYPE {gauge.name} gauge")
            lines.append(f"{gauge.name}{label_str} {gauge.value}")

        for hist in self._histograms.values():
            labels = ",".join(f'{k}="{v}"' for k, v in hist.labels.items())
            base_label = f"{{{labels}}}" if labels else ""
            lines.append(f"# TYPE {hist.name} histogram")
            bucket_counts = hist.get_bucket_counts()
            for bucket, count in bucket_counts.items():
                le_label = f'le="{bucket}"'
                if labels:
                    bucket_label = f"{{{labels},{le_label}}}"
                else:
                    bucket_label = f"{{{le_label}}}"
                lines.append(f"{hist.name}_bucket{bucket_label} {count}")
            lines.append(f"{hist.name}_sum{base_label} {hist.sum}")
            lines.append(f"{hist.name}_count{base_label} {hist.count}")

        return "\n".join(lines)

    def reset_all(self) -> None:
        """Reset all metrics to their initial state."""
        for counter in self._counters.values():
            counter.reset()
        for gauge in self._gauges.values():
            gauge.set(0.0)
        for histogram in self._histograms.values():
            histogram.reset()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """
    Get the global metrics collector singleton.

    Returns:
        The global MetricsCollector instance.
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def reset_metrics() -> None:
    """Reset the global metrics collector."""
    global _metrics_collector
    _metrics_collector = None


def timed(name: str, **labels: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to time a function and record duration in a histogram.

    Args:
        name: The histogram metric name.
        **labels: Label key-value pairs for the metric.

    Returns:
        Decorator function.

    Example:
        >>> @timed("function_duration_seconds", function="my_func")
        ... def my_func():
        ...     time.sleep(0.1)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                get_metrics().histogram(name, **labels).observe(duration)

        return wrapper

    return decorator


def counted(name: str, **labels: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to count function invocations.

    Args:
        name: The counter metric name.
        **labels: Label key-value pairs for the metric.

    Returns:
        Decorator function.

    Example:
        >>> @counted("function_calls_total", function="my_func")
        ... def my_func():
        ...     pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            get_metrics().counter(name, **labels).inc()
            return func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "Counter",
    "Gauge",
    "Histogram",
    "MetricsCollector",
    "get_metrics",
    "reset_metrics",
    "timed",
    "counted",
]
