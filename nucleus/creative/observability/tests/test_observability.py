"""
Tests for the Observability Subsystem
=====================================

Tests metrics collection, tracing, and structured logging.
"""

import pytest
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# Ensure nucleus is importable
sys.path.insert(0, str(Path(__file__).parents[4]))


# -----------------------------------------------------------------------------
# Import Classes from Observability Module
# -----------------------------------------------------------------------------

try:
    from nucleus.creative.observability import (
        # Metrics
        Counter,
        Gauge,
        Histogram,
        MetricsCollector,
        get_metrics,
        timed,
        # Tracing
        Span,
        Tracer,
        get_tracer,
        get_current_trace_id,
        traced,
        # Logging
        StructuredLogger,
        LogLevel,
        get_logger,
    )
    HAS_OBSERVABILITY = True
except ImportError as e:
    HAS_OBSERVABILITY = False
    Counter = None
    Gauge = None
    Histogram = None
    MetricsCollector = None
    get_metrics = None
    timed = None
    Span = None
    Tracer = None
    get_tracer = None
    get_current_trace_id = None
    traced = None
    StructuredLogger = None
    LogLevel = None
    get_logger = None


# -----------------------------------------------------------------------------
# Skip decorator for when observability not available
# -----------------------------------------------------------------------------

requires_observability = pytest.mark.skipif(
    not HAS_OBSERVABILITY,
    reason="Observability module not available"
)


# -----------------------------------------------------------------------------
# Smoke Tests
# -----------------------------------------------------------------------------


class TestObservabilitySmoke:
    """Smoke tests verifying imports work."""

    def test_observability_module_importable(self):
        """Test that observability module can be imported."""
        from nucleus.creative import observability
        assert observability is not None

    @requires_observability
    def test_metrics_classes_exist(self):
        """Test metrics classes are defined."""
        assert Counter is not None
        assert Gauge is not None
        assert Histogram is not None
        assert MetricsCollector is not None

    @requires_observability
    def test_tracing_classes_exist(self):
        """Test tracing classes are defined."""
        assert Span is not None
        assert Tracer is not None

    @requires_observability
    def test_logging_classes_exist(self):
        """Test logging classes are defined."""
        assert StructuredLogger is not None
        assert LogLevel is not None


# -----------------------------------------------------------------------------
# Counter Tests
# -----------------------------------------------------------------------------


@requires_observability
class TestCounter:
    """Tests for Counter metric."""

    def test_counter_creation(self):
        """Test creating a Counter."""
        counter = Counter(name="test_counter")
        assert counter.name == "test_counter"
        assert counter.value == 0.0

    def test_counter_increment(self):
        """Test incrementing a counter."""
        counter = Counter(name="requests")
        counter.inc()
        assert counter.value == 1.0

    def test_counter_increment_by_amount(self):
        """Test incrementing counter by specific amount."""
        counter = Counter(name="bytes_sent")
        counter.inc(100.0)
        counter.inc(50.0)
        assert counter.value == 150.0

    def test_counter_with_labels(self):
        """Test counter with labels."""
        counter = Counter(
            name="http_requests",
            labels={"method": "GET", "path": "/api"},
        )
        counter.inc()
        assert counter.labels["method"] == "GET"
        assert counter.value == 1.0

    def test_counter_multiple_increments(self):
        """Test multiple counter increments."""
        counter = Counter(name="events")
        for _ in range(100):
            counter.inc()
        assert counter.value == 100.0


# -----------------------------------------------------------------------------
# Gauge Tests
# -----------------------------------------------------------------------------


@requires_observability
class TestGauge:
    """Tests for Gauge metric."""

    def test_gauge_creation(self):
        """Test creating a Gauge."""
        gauge = Gauge(name="temperature")
        assert gauge.name == "temperature"
        assert gauge.value == 0.0

    def test_gauge_set(self):
        """Test setting gauge value."""
        gauge = Gauge(name="cpu_usage")
        gauge.set(75.5)
        assert gauge.value == 75.5

    def test_gauge_increment(self):
        """Test incrementing gauge."""
        gauge = Gauge(name="active_connections")
        gauge.set(10)
        gauge.inc(5)
        assert gauge.value == 15

    def test_gauge_decrement(self):
        """Test decrementing gauge."""
        gauge = Gauge(name="queue_size")
        gauge.set(100)
        gauge.dec(25)
        assert gauge.value == 75

    def test_gauge_negative_values(self):
        """Test gauge with negative values."""
        gauge = Gauge(name="delta")
        gauge.set(-10)
        assert gauge.value == -10

    def test_gauge_with_labels(self):
        """Test gauge with labels."""
        gauge = Gauge(
            name="memory_usage",
            labels={"process": "worker", "instance": "1"},
        )
        gauge.set(1024)
        assert gauge.labels["process"] == "worker"


# -----------------------------------------------------------------------------
# Histogram Tests
# -----------------------------------------------------------------------------


@requires_observability
class TestHistogram:
    """Tests for Histogram metric."""

    def test_histogram_creation(self):
        """Test creating a Histogram."""
        hist = Histogram(name="request_duration")
        assert hist.name == "request_duration"
        assert hist.count == 0

    def test_histogram_observe(self):
        """Test observing values."""
        hist = Histogram(name="latency")
        hist.observe(0.1)
        hist.observe(0.2)
        hist.observe(0.3)
        assert hist.count == 3

    def test_histogram_sum(self):
        """Test histogram sum."""
        hist = Histogram(name="processing_time")
        hist.observe(1.0)
        hist.observe(2.0)
        hist.observe(3.0)
        assert hist.sum == 6.0

    def test_histogram_mean(self):
        """Test histogram mean."""
        hist = Histogram(name="response_time")
        hist.observe(10.0)
        hist.observe(20.0)
        assert hist.mean == 15.0

    def test_histogram_custom_buckets(self):
        """Test histogram with custom buckets."""
        buckets = [0.1, 0.5, 1.0, 5.0, 10.0]
        hist = Histogram(name="custom", buckets=buckets)
        assert hist.buckets == buckets

    def test_histogram_empty_mean(self):
        """Test histogram mean when empty."""
        hist = Histogram(name="empty")
        assert hist.mean == 0.0


# -----------------------------------------------------------------------------
# MetricsCollector Tests
# -----------------------------------------------------------------------------


@requires_observability
class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_collector_creation(self):
        """Test creating a MetricsCollector."""
        collector = MetricsCollector()
        assert collector is not None

    def test_collector_counter(self):
        """Test getting counter from collector."""
        collector = MetricsCollector()
        counter = collector.counter("test_counter")
        counter.inc()
        assert counter.value == 1.0

    def test_collector_gauge(self):
        """Test getting gauge from collector."""
        collector = MetricsCollector()
        gauge = collector.gauge("test_gauge")
        gauge.set(42)
        assert gauge.value == 42

    def test_collector_histogram(self):
        """Test getting histogram from collector."""
        collector = MetricsCollector()
        hist = collector.histogram("test_hist")
        hist.observe(1.5)
        assert hist.count == 1

    def test_collector_get_all(self):
        """Test getting all metrics."""
        collector = MetricsCollector()
        collector.counter("c1").inc(10)
        collector.gauge("g1").set(20)
        collector.histogram("h1").observe(30)

        all_metrics = collector.get_all()
        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histograms" in all_metrics

    def test_collector_same_metric_reuse(self):
        """Test that same metric name returns same instance."""
        collector = MetricsCollector()
        c1 = collector.counter("same_counter")
        c2 = collector.counter("same_counter")
        c1.inc()
        assert c2.value == 1.0  # Same instance


@requires_observability
class TestTimedDecorator:
    """Tests for the timed decorator."""

    def test_timed_function(self):
        """Test timed decorator records duration."""
        @timed("test_timing")
        def slow_function():
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"

        # Check histogram was updated
        metrics = get_metrics()
        hist = metrics.histogram("test_timing")
        assert hist.count >= 1


# -----------------------------------------------------------------------------
# Span Tests
# -----------------------------------------------------------------------------


@requires_observability
class TestSpan:
    """Tests for Span class."""

    def test_span_creation(self):
        """Test creating a Span."""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            name="test_operation",
        )
        assert span.trace_id == "trace-123"
        assert span.span_id == "span-456"
        assert span.name == "test_operation"

    def test_span_add_event(self):
        """Test adding event to span."""
        span = Span(
            trace_id="t1",
            span_id="s1",
            name="op",
        )
        span.add_event("start_processing", key="value")
        assert len(span.events) == 1
        assert span.events[0]["name"] == "start_processing"

    def test_span_set_attribute(self):
        """Test setting span attribute."""
        span = Span(
            trace_id="t1",
            span_id="s1",
            name="op",
        )
        span.set_attribute("http.status_code", 200)
        assert span.attributes["http.status_code"] == 200

    def test_span_end(self):
        """Test ending a span."""
        span = Span(
            trace_id="t1",
            span_id="s1",
            name="op",
        )
        assert span.end_time is None
        span.end()
        assert span.end_time is not None

    def test_span_duration(self):
        """Test span duration calculation."""
        span = Span(
            trace_id="t1",
            span_id="s1",
            name="op",
        )
        time.sleep(0.01)
        span.end()
        assert span.duration_ms > 0


# -----------------------------------------------------------------------------
# Tracer Tests
# -----------------------------------------------------------------------------


@requires_observability
class TestTracer:
    """Tests for Tracer class."""

    def test_tracer_creation(self):
        """Test creating a Tracer."""
        tracer = Tracer(service_name="test_service")
        assert tracer.service_name == "test_service"

    def test_tracer_start_span(self):
        """Test starting a span."""
        tracer = Tracer()
        with tracer.start_span("test_op") as span:
            assert span is not None
            assert span.name == "test_op"

    def test_tracer_nested_spans(self):
        """Test nested spans."""
        tracer = Tracer()
        with tracer.start_span("parent") as parent:
            with tracer.start_span("child") as child:
                assert child.parent_span_id == parent.span_id
                assert child.trace_id == parent.trace_id

    def test_tracer_span_ends_on_exit(self):
        """Test span ends when context exits."""
        tracer = Tracer()
        span_ref = None
        with tracer.start_span("auto_end") as span:
            span_ref = span
            assert span.end_time is None
        assert span_ref.end_time is not None


@requires_observability
class TestTracedDecorator:
    """Tests for the traced decorator."""

    def test_traced_function(self):
        """Test traced decorator creates span."""
        @traced("test_traced_op")
        def traced_function():
            return "traced"

        result = traced_function()
        assert result == "traced"

    def test_traced_function_error(self):
        """Test traced decorator handles errors."""
        @traced("error_op")
        def failing_function():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            failing_function()


# -----------------------------------------------------------------------------
# StructuredLogger Tests
# -----------------------------------------------------------------------------


@requires_observability
class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    def test_logger_creation(self):
        """Test creating a StructuredLogger."""
        logger = StructuredLogger(name="test_logger")
        assert logger.name == "test_logger"

    def test_logger_info(self):
        """Test info level logging."""
        logger = StructuredLogger(name="test")
        # Should not raise
        logger.info("Test message")

    def test_logger_debug(self):
        """Test debug level logging."""
        logger = StructuredLogger(name="test")
        logger.debug("Debug message", extra_field="value")

    def test_logger_warning(self):
        """Test warning level logging."""
        logger = StructuredLogger(name="test")
        logger.warning("Warning message")

    def test_logger_error(self):
        """Test error level logging."""
        logger = StructuredLogger(name="test")
        logger.error("Error message", error_code=500)

    def test_logger_critical(self):
        """Test critical level logging."""
        logger = StructuredLogger(name="test")
        logger.critical("Critical message")

    def test_logger_bind(self):
        """Test binding context to logger."""
        logger = StructuredLogger(name="test")
        bound = logger.bind(request_id="123", user_id="456")
        assert bound._context["request_id"] == "123"
        assert bound._context["user_id"] == "456"

    def test_logger_bind_chain(self):
        """Test chaining bind calls."""
        logger = StructuredLogger(name="test")
        bound1 = logger.bind(a="1")
        bound2 = bound1.bind(b="2")
        assert bound2._context["a"] == "1"
        assert bound2._context["b"] == "2"


@requires_observability
class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_levels_exist(self):
        """Test all log levels exist."""
        assert LogLevel.DEBUG.value == 10
        assert LogLevel.INFO.value == 20
        assert LogLevel.WARNING.value == 30
        assert LogLevel.ERROR.value == 40
        assert LogLevel.CRITICAL.value == 50

    def test_log_level_ordering(self):
        """Test log level ordering."""
        assert LogLevel.DEBUG.value < LogLevel.INFO.value
        assert LogLevel.INFO.value < LogLevel.WARNING.value
        assert LogLevel.WARNING.value < LogLevel.ERROR.value
        assert LogLevel.ERROR.value < LogLevel.CRITICAL.value


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


@requires_observability
class TestObservabilityIntegration:
    """Integration tests for observability."""

    def test_metrics_tracing_together(self):
        """Test using metrics and tracing together."""
        tracer = get_tracer()
        metrics = get_metrics()

        counter = metrics.counter("integration_test_counter")
        hist = metrics.histogram("integration_test_duration")

        with tracer.start_span("integration_test") as span:
            counter.inc()
            time.sleep(0.01)
            hist.observe(0.01)
            span.set_attribute("counter_value", counter.value)

        assert counter.value >= 1.0

    def test_logging_with_trace_context(self):
        """Test logging includes trace context."""
        tracer = get_tracer()
        logger = get_logger("integration")

        with tracer.start_span("logged_operation") as span:
            trace_id = get_current_trace_id()
            logger.info("Inside traced operation", trace_id=trace_id)
            assert trace_id is not None

    def test_full_observability_pipeline(self):
        """Test full observability pipeline."""
        # Get global instances
        tracer = get_tracer()
        metrics = get_metrics()
        logger = get_logger("pipeline_test")

        request_counter = metrics.counter("pipeline_requests")
        duration_hist = metrics.histogram("pipeline_duration")

        with tracer.start_span("pipeline_operation") as span:
            request_counter.inc()
            logger.info("Starting pipeline")

            # Simulate work
            start = time.time()
            time.sleep(0.005)
            duration = time.time() - start

            duration_hist.observe(duration)
            span.set_attribute("duration", duration)
            logger.info("Pipeline complete", duration=duration)

        assert request_counter.value >= 1


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------


@requires_observability
class TestObservabilityEdgeCases:
    """Edge case tests for observability."""

    def test_counter_zero_increment(self):
        """Test incrementing counter by zero."""
        counter = Counter(name="zero_test")
        counter.inc(0.0)
        assert counter.value == 0.0

    def test_gauge_very_large_value(self):
        """Test gauge with very large value."""
        gauge = Gauge(name="large")
        gauge.set(1e18)
        assert gauge.value == 1e18

    def test_gauge_very_small_value(self):
        """Test gauge with very small value."""
        gauge = Gauge(name="small")
        gauge.set(1e-18)
        assert gauge.value == 1e-18

    def test_histogram_negative_observation(self):
        """Test histogram with negative value."""
        hist = Histogram(name="negative")
        hist.observe(-1.0)
        assert hist.count == 1
        assert hist.sum == -1.0

    def test_span_empty_name(self):
        """Test span with empty name."""
        span = Span(trace_id="t", span_id="s", name="")
        assert span.name == ""

    def test_logger_empty_message(self):
        """Test logging empty message."""
        logger = StructuredLogger(name="test")
        logger.info("")  # Should not raise

    def test_logger_unicode_message(self):
        """Test logging unicode message."""
        logger = StructuredLogger(name="test")
        logger.info("Message with unicode: Bonjour!")

    def test_many_span_events(self):
        """Test span with many events."""
        span = Span(trace_id="t", span_id="s", name="many_events")
        for i in range(1000):
            span.add_event(f"event_{i}")
        assert len(span.events) == 1000

    def test_deeply_nested_spans(self):
        """Test deeply nested spans."""
        tracer = Tracer()

        def nest(depth, parent_trace_id=None):
            if depth == 0:
                return
            with tracer.start_span(f"level_{depth}") as span:
                if parent_trace_id:
                    assert span.trace_id == parent_trace_id
                nest(depth - 1, span.trace_id)

        nest(10)  # 10 levels deep
