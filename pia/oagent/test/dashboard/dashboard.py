#!/usr/bin/env python3
"""
Step 122: Test Dashboard

Provides real-time test status dashboard with live updates.

PBTSO Phase: OBSERVE, VERIFY
Bus Topics:
- test.dashboard.update (emits)
- test.dashboard.status (emits)
- test.run.* (subscribes)

Dependencies: Steps 101-121 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque


# ============================================================================
# Constants
# ============================================================================

class WidgetType(Enum):
    """Types of dashboard widgets."""
    STATUS_SUMMARY = "status_summary"
    TEST_PROGRESS = "test_progress"
    COVERAGE_GAUGE = "coverage_gauge"
    FAILURE_LIST = "failure_list"
    FLAKY_TESTS = "flaky_tests"
    TIMELINE = "timeline"
    TREND_CHART = "trend_chart"
    METRICS_TABLE = "metrics_table"


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class DashboardMetrics:
    """Metrics displayed on the dashboard."""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    running: int = 0
    pending: int = 0
    coverage_percent: float = 0.0
    mutation_score: float = 0.0
    flaky_count: int = 0
    regression_count: int = 0
    avg_duration_ms: float = 0.0
    throughput_tests_per_sec: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        completed = self.passed + self.failed
        if completed == 0:
            return 0.0
        return (self.passed / completed) * 100

    @property
    def completion_percent(self) -> float:
        """Calculate completion percentage."""
        if self.total_tests == 0:
            return 0.0
        completed = self.passed + self.failed + self.skipped
        return (completed / self.total_tests) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "running": self.running,
            "pending": self.pending,
            "coverage_percent": self.coverage_percent,
            "mutation_score": self.mutation_score,
            "flaky_count": self.flaky_count,
            "regression_count": self.regression_count,
            "success_rate": self.success_rate,
            "completion_percent": self.completion_percent,
            "avg_duration_ms": self.avg_duration_ms,
            "throughput_tests_per_sec": self.throughput_tests_per_sec,
        }


@dataclass
class TestEvent:
    """An event in the test timeline."""
    timestamp: float
    event_type: str
    test_name: str
    status: Optional[str] = None
    duration_ms: Optional[float] = None
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "test_name": self.test_name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "message": self.message,
        }


@dataclass
class DashboardConfig:
    """
    Configuration for the test dashboard.

    Attributes:
        update_interval_ms: How often to push updates
        max_timeline_events: Maximum events in timeline
        widgets: Enabled widget types
        output_dir: Directory for dashboard state
        enable_websocket: Enable WebSocket updates
        websocket_port: Port for WebSocket server
        retention_hours: Hours to retain history
    """
    update_interval_ms: int = 500
    max_timeline_events: int = 100
    widgets: List[WidgetType] = field(default_factory=lambda: [
        WidgetType.STATUS_SUMMARY,
        WidgetType.TEST_PROGRESS,
        WidgetType.FAILURE_LIST,
        WidgetType.TIMELINE,
    ])
    output_dir: str = ".pluribus/test-agent/dashboard"
    enable_websocket: bool = False
    websocket_port: int = 8765
    retention_hours: int = 24

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "update_interval_ms": self.update_interval_ms,
            "max_timeline_events": self.max_timeline_events,
            "widgets": [w.value for w in self.widgets],
            "enable_websocket": self.enable_websocket,
            "websocket_port": self.websocket_port,
        }


@dataclass
class DashboardState:
    """Current state of the dashboard."""
    run_id: Optional[str] = None
    status: str = "idle"
    started_at: Optional[float] = None
    metrics: DashboardMetrics = field(default_factory=DashboardMetrics)
    timeline: List[TestEvent] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    flaky_tests: List[str] = field(default_factory=list)
    running_tests: Set[str] = field(default_factory=set)
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "elapsed_s": time.time() - self.started_at if self.started_at else 0,
            "metrics": self.metrics.to_dict(),
            "timeline": [e.to_dict() for e in self.timeline[-20:]],  # Last 20
            "failures": self.failures[:10],  # Top 10 failures
            "flaky_tests": self.flaky_tests,
            "running_tests": list(self.running_tests),
            "last_update": self.last_update,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class DashboardBus:
    """Bus interface for dashboard with file locking."""

    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()
        self._subscribers: Dict[str, List[Callable]] = {}

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to a topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(handler)

    def poll_events(self, since_ts: Optional[str] = None) -> List[Dict[str, Any]]:
        """Poll for new events since timestamp."""
        events = []
        try:
            with open(self.bus_path, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    for line in f:
                        try:
                            event = json.loads(line)
                            if since_ts is None or event.get("ts", "") > since_ts:
                                events.append(event)
                        except json.JSONDecodeError:
                            continue
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass
        return events

    def heartbeat(self, agent_id: str) -> None:
        """Send A2A heartbeat."""
        now = time.time()
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.emit({
                "topic": "a2a.heartbeat",
                "kind": "heartbeat",
                "actor": agent_id,
                "data": {"status": "alive"},
            })
            self._last_heartbeat = now


# ============================================================================
# Circuit Breaker
# ============================================================================

class CircuitBreaker:
    """Circuit breaker for external service calls."""

    def __init__(self, failure_threshold: int = 5, reset_timeout_s: float = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open

    def can_proceed(self) -> bool:
        """Check if circuit allows proceeding."""
        if self.state == "closed":
            return True
        if self.state == "open":
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.reset_timeout_s:
                self.state = "half-open"
                return True
            return False
        # half-open
        return True

    def record_success(self) -> None:
        """Record successful call."""
        if self.state == "half-open":
            self.state = "closed"
        self.failures = 0

    def record_failure(self) -> None:
        """Record failed call."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"


# ============================================================================
# Test Dashboard
# ============================================================================

class TestDashboard:
    """
    Real-time test status dashboard.

    Features:
    - Live test progress tracking
    - Metrics visualization
    - Timeline of events
    - Failure highlights
    - Flaky test monitoring
    - WebSocket updates (optional)

    PBTSO Phase: OBSERVE, VERIFY
    Bus Topics: test.dashboard.update, test.dashboard.status
    """

    BUS_TOPICS = {
        "update": "test.dashboard.update",
        "status": "test.dashboard.status",
    }

    SUBSCRIBED_TOPICS = [
        "test.run.start",
        "test.run.complete",
        "test.run.progress",
        "test.run.result",
        "test.flaky.detected",
        "test.regression.found",
    ]

    def __init__(self, bus=None, config: Optional[DashboardConfig] = None):
        """
        Initialize the test dashboard.

        Args:
            bus: Optional bus instance
            config: Dashboard configuration
        """
        self.bus = bus or DashboardBus()
        self.config = config or DashboardConfig()
        self.state = DashboardState()
        self._circuit_breaker = CircuitBreaker()
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        self._websocket_clients: Set[Any] = set()
        self._last_poll_ts: Optional[str] = None
        self._duration_samples: deque = deque(maxlen=100)

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def start_run(self, run_id: str, total_tests: int) -> None:
        """Start tracking a new test run."""
        self.state = DashboardState(
            run_id=run_id,
            status="running",
            started_at=time.time(),
            metrics=DashboardMetrics(
                total_tests=total_tests,
                pending=total_tests,
            ),
        )

        self._add_event(TestEvent(
            timestamp=time.time(),
            event_type="run_started",
            test_name="",
            message=f"Test run {run_id} started with {total_tests} tests",
        ))

        self._emit_status()
        self._save_state()

    def end_run(self, run_id: str) -> None:
        """End the current test run."""
        if self.state.run_id == run_id:
            self.state.status = "completed"

            self._add_event(TestEvent(
                timestamp=time.time(),
                event_type="run_completed",
                test_name="",
                message=f"Test run {run_id} completed",
            ))

            self._emit_status()
            self._save_state()

    def record_test_start(self, test_name: str) -> None:
        """Record a test starting."""
        self.state.running_tests.add(test_name)
        self.state.metrics.running += 1
        self.state.metrics.pending -= 1

        self._add_event(TestEvent(
            timestamp=time.time(),
            event_type="test_started",
            test_name=test_name,
        ))

        self._update_dashboard()

    def record_test_result(
        self,
        test_name: str,
        status: str,
        duration_ms: float,
        error_message: Optional[str] = None,
    ) -> None:
        """Record a test result."""
        self.state.running_tests.discard(test_name)
        self.state.metrics.running = max(0, self.state.metrics.running - 1)

        # Update counts
        if status == "passed":
            self.state.metrics.passed += 1
        elif status == "failed":
            self.state.metrics.failed += 1
            self.state.failures.append({
                "test_name": test_name,
                "duration_ms": duration_ms,
                "error_message": error_message,
                "timestamp": time.time(),
            })
        elif status == "skipped":
            self.state.metrics.skipped += 1

        # Track duration
        self._duration_samples.append(duration_ms)
        self.state.metrics.avg_duration_ms = sum(self._duration_samples) / len(self._duration_samples)

        # Calculate throughput
        if self.state.started_at:
            elapsed = time.time() - self.state.started_at
            completed = self.state.metrics.passed + self.state.metrics.failed + self.state.metrics.skipped
            if elapsed > 0:
                self.state.metrics.throughput_tests_per_sec = completed / elapsed

        self._add_event(TestEvent(
            timestamp=time.time(),
            event_type="test_completed",
            test_name=test_name,
            status=status,
            duration_ms=duration_ms,
            message=error_message,
        ))

        self._update_dashboard()

    def update_coverage(self, coverage_percent: float) -> None:
        """Update coverage metric."""
        self.state.metrics.coverage_percent = coverage_percent
        self._update_dashboard()

    def update_mutation_score(self, score: float) -> None:
        """Update mutation score metric."""
        self.state.metrics.mutation_score = score
        self._update_dashboard()

    def add_flaky_test(self, test_name: str) -> None:
        """Add a flaky test to the list."""
        if test_name not in self.state.flaky_tests:
            self.state.flaky_tests.append(test_name)
            self.state.metrics.flaky_count = len(self.state.flaky_tests)

            self._add_event(TestEvent(
                timestamp=time.time(),
                event_type="flaky_detected",
                test_name=test_name,
                message=f"Flaky test detected: {test_name}",
            ))

            self._update_dashboard()

    def add_regression(self, test_name: str, regression_type: str) -> None:
        """Add a regression to the list."""
        self.state.metrics.regression_count += 1

        self._add_event(TestEvent(
            timestamp=time.time(),
            event_type="regression_detected",
            test_name=test_name,
            message=f"Regression detected: {regression_type}",
        ))

        self._update_dashboard()

    def _add_event(self, event: TestEvent) -> None:
        """Add an event to the timeline."""
        self.state.timeline.append(event)

        # Trim to max events
        if len(self.state.timeline) > self.config.max_timeline_events:
            self.state.timeline = self.state.timeline[-self.config.max_timeline_events:]

    def _update_dashboard(self) -> None:
        """Update dashboard state."""
        self.state.last_update = time.time()
        self._save_state()

        # Emit update if interval passed
        self._emit_update()

    def _emit_status(self) -> None:
        """Emit dashboard status event."""
        self.bus.emit({
            "topic": self.BUS_TOPICS["status"],
            "kind": "dashboard",
            "actor": "test-agent",
            "data": {
                "run_id": self.state.run_id,
                "status": self.state.status,
                "metrics": self.state.metrics.to_dict(),
            },
        })

    def _emit_update(self) -> None:
        """Emit dashboard update event."""
        self.bus.emit({
            "topic": self.BUS_TOPICS["update"],
            "kind": "dashboard",
            "actor": "test-agent",
            "data": self.state.to_dict(),
        })

    def _save_state(self) -> None:
        """Save dashboard state to file."""
        state_file = Path(self.config.output_dir) / "dashboard_state.json"

        try:
            with open(state_file, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(self.state.to_dict(), f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def get_state(self) -> Dict[str, Any]:
        """Get current dashboard state."""
        return self.state.to_dict()

    def get_metrics(self) -> DashboardMetrics:
        """Get current metrics."""
        return self.state.metrics

    def render_console(self) -> str:
        """Render dashboard for console output."""
        metrics = self.state.metrics

        # Build progress bar
        completion = int(metrics.completion_percent / 5)  # 20 chars = 100%
        progress_bar = "[" + "#" * completion + "-" * (20 - completion) + "]"

        # Status indicator
        if self.state.status == "running":
            status_indicator = "[RUNNING]"
        elif metrics.failed > 0:
            status_indicator = "[FAIL]"
        else:
            status_indicator = "[PASS]"

        lines = [
            "",
            f"{'='*60}",
            f"Test Dashboard {status_indicator}",
            f"{'='*60}",
            "",
            f"Run ID: {self.state.run_id or 'None'}",
            f"Status: {self.state.status}",
            "",
            f"Progress: {progress_bar} {metrics.completion_percent:.1f}%",
            "",
            f"Tests: {metrics.passed} passed | {metrics.failed} failed | "
            f"{metrics.skipped} skipped | {metrics.running} running",
            f"Success Rate: {metrics.success_rate:.1f}%",
            "",
            f"Coverage: {metrics.coverage_percent:.1f}%",
            f"Avg Duration: {metrics.avg_duration_ms:.0f}ms",
            f"Throughput: {metrics.throughput_tests_per_sec:.2f} tests/sec",
            "",
        ]

        # Recent failures
        if self.state.failures:
            lines.append("Recent Failures:")
            for failure in self.state.failures[:5]:
                lines.append(f"  - {failure['test_name']}")

        # Running tests
        if self.state.running_tests:
            lines.append("")
            lines.append("Currently Running:")
            for test in list(self.state.running_tests)[:5]:
                lines.append(f"  - {test}")

        lines.append(f"{'='*60}")

        return "\n".join(lines)

    async def start_polling(self) -> None:
        """Start polling for bus events."""
        self._running = True

        while self._running:
            # Poll for new events
            events = self.bus.poll_events(self._last_poll_ts)

            for event in events:
                self._last_poll_ts = event.get("ts")
                topic = event.get("topic", "")

                # Process relevant events
                if topic == "test.run.start":
                    data = event.get("data", {})
                    self.start_run(
                        run_id=data.get("run_id", ""),
                        total_tests=data.get("total_tests", 0),
                    )
                elif topic == "test.run.result":
                    data = event.get("data", {})
                    self.record_test_result(
                        test_name=data.get("test_name", ""),
                        status=data.get("status", "unknown"),
                        duration_ms=data.get("duration_ms", 0),
                        error_message=data.get("error_message"),
                    )
                elif topic == "test.run.complete":
                    data = event.get("data", {})
                    self.end_run(data.get("run_id", ""))
                elif topic == "test.flaky.detected":
                    data = event.get("data", {})
                    self.add_flaky_test(data.get("test_name", ""))

            # Send heartbeat
            self.bus.heartbeat("test-agent-dashboard")

            await asyncio.sleep(self.config.update_interval_ms / 1000)

    def stop_polling(self) -> None:
        """Stop polling for events."""
        self._running = False

    async def run_async(self) -> None:
        """Run the dashboard asynchronously."""
        await self.start_polling()


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Dashboard")
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/dashboard")
    parser.add_argument("--watch", action="store_true", help="Watch mode")
    parser.add_argument("--interval", type=int, default=500, help="Update interval (ms)")
    parser.add_argument("--state", action="store_true", help="Show current state")

    args = parser.parse_args()

    config = DashboardConfig(
        output_dir=args.output,
        update_interval_ms=args.interval,
    )

    dashboard = TestDashboard(config=config)

    if args.state:
        # Load and display current state
        state_file = Path(args.output) / "dashboard_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                print(json.dumps(state, indent=2))
        else:
            print("No dashboard state found")
        return

    if args.watch:
        # Watch mode - continuous console updates
        try:
            import curses

            def run_curses(stdscr):
                stdscr.clear()
                curses.curs_set(0)

                while True:
                    state_file = Path(args.output) / "dashboard_state.json"
                    if state_file.exists():
                        try:
                            with open(state_file) as f:
                                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                                try:
                                    state = json.load(f)
                                finally:
                                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                            # Update dashboard state
                            dashboard.state = DashboardState(
                                run_id=state.get("run_id"),
                                status=state.get("status", "idle"),
                                started_at=state.get("started_at"),
                                metrics=DashboardMetrics(**state.get("metrics", {})),
                            )
                        except (json.JSONDecodeError, IOError):
                            pass

                    stdscr.clear()
                    output = dashboard.render_console()
                    for i, line in enumerate(output.split("\n")):
                        try:
                            stdscr.addstr(i, 0, line[:curses.COLS - 1])
                        except curses.error:
                            pass
                    stdscr.refresh()
                    time.sleep(args.interval / 1000)

            curses.wrapper(run_curses)

        except ImportError:
            # Fallback to simple console output
            while True:
                print("\033[2J\033[H")  # Clear screen
                print(dashboard.render_console())
                time.sleep(args.interval / 1000)

    else:
        # Single render
        print(dashboard.render_console())


if __name__ == "__main__":
    main()
