#!/usr/bin/env python3
"""
Step 114: Load Tester

Provides concurrent load testing capabilities for stress testing.

PBTSO Phase: TEST, VERIFY
Bus Topics:
- test.load.run (subscribes)
- test.load.result (emits)
- telemetry.load (emits)

Dependencies: Step 106 (Test Runner Orchestrator)
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# Constants
# ============================================================================

DEFAULT_CONCURRENT_USERS = 10
DEFAULT_DURATION_S = 60
DEFAULT_RAMP_UP_S = 10


class LoadPattern(Enum):
    """Load patterns for testing."""
    CONSTANT = "constant"  # Fixed number of concurrent users
    RAMP_UP = "ramp_up"  # Gradually increase users
    STEP = "step"  # Step-wise increase
    SPIKE = "spike"  # Sudden spike in load
    WAVE = "wave"  # Sinusoidal load pattern


class RequestStatus(Enum):
    """Status of a load test request."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class LoadScenario:
    """
    A load test scenario.

    Attributes:
        name: Scenario name
        target: Target function or URL
        weight: Weight for scenario selection (higher = more frequent)
        setup: Optional setup function
        validate: Optional response validation function
    """
    name: str
    target: Callable
    weight: int = 1
    setup: Optional[Callable] = None
    validate: Optional[Callable[[Any], bool]] = None
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadConfig:
    """
    Configuration for load testing.

    Attributes:
        scenarios: List of load scenarios
        concurrent_users: Number of concurrent users
        duration_s: Test duration in seconds
        ramp_up_s: Time to ramp up to full load
        pattern: Load pattern
        requests_per_second: Target RPS (None = unlimited)
        timeout_s: Request timeout
        output_dir: Directory for reports
    """
    scenarios: List[LoadScenario] = field(default_factory=list)
    concurrent_users: int = DEFAULT_CONCURRENT_USERS
    duration_s: int = DEFAULT_DURATION_S
    ramp_up_s: int = DEFAULT_RAMP_UP_S
    pattern: LoadPattern = LoadPattern.RAMP_UP
    requests_per_second: Optional[float] = None
    timeout_s: float = 30.0
    output_dir: str = ".pluribus/test-agent/load"
    think_time_ms: int = 0  # Delay between requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "concurrent_users": self.concurrent_users,
            "duration_s": self.duration_s,
            "ramp_up_s": self.ramp_up_s,
            "pattern": self.pattern.value,
            "requests_per_second": self.requests_per_second,
            "timeout_s": self.timeout_s,
            "think_time_ms": self.think_time_ms,
        }


@dataclass
class RequestResult:
    """Result of a single load test request."""
    request_id: str
    scenario_name: str
    status: RequestStatus
    latency_ms: float
    timestamp: float
    response_size: int = 0
    error_message: Optional[str] = None
    user_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "scenario_name": self.scenario_name,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
            "response_size": self.response_size,
            "error_message": self.error_message,
            "user_id": self.user_id,
        }


@dataclass
class LoadStats:
    """Statistics for load test results."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    timeout_requests: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    error_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "timeout_requests": self.timeout_requests,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "requests_per_second": self.requests_per_second,
            "error_rate": self.error_rate,
        }

    @classmethod
    def from_results(
        cls,
        results: List[RequestResult],
        duration_s: float,
    ) -> "LoadStats":
        """Calculate stats from request results."""
        if not results:
            return cls(
                total_requests=0, successful_requests=0, failed_requests=0,
                timeout_requests=0, avg_latency_ms=0, min_latency_ms=0,
                max_latency_ms=0, p50_latency_ms=0, p95_latency_ms=0,
                p99_latency_ms=0, requests_per_second=0, error_rate=0
            )

        successful = [r for r in results if r.status == RequestStatus.SUCCESS]
        failed = [r for r in results if r.status == RequestStatus.FAILURE]
        timeouts = [r for r in results if r.status == RequestStatus.TIMEOUT]

        latencies = sorted([r.latency_ms for r in results])
        n = len(latencies)

        return cls(
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            timeout_requests=len(timeouts),
            avg_latency_ms=statistics.mean(latencies),
            min_latency_ms=latencies[0],
            max_latency_ms=latencies[-1],
            p50_latency_ms=latencies[int(n * 0.50)],
            p95_latency_ms=latencies[int(n * 0.95)] if n > 20 else latencies[-1],
            p99_latency_ms=latencies[int(n * 0.99)] if n > 100 else latencies[-1],
            requests_per_second=len(results) / duration_s if duration_s > 0 else 0,
            error_rate=(len(failed) + len(timeouts)) / len(results) * 100,
        )


@dataclass
class LoadResult:
    """Complete result of a load test."""
    run_id: str
    config: LoadConfig
    started_at: float
    completed_at: Optional[float] = None
    stats: Optional[LoadStats] = None
    results: List[RequestResult] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    scenario_stats: Dict[str, LoadStats] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        """Get total duration."""
        if self.completed_at:
            return self.completed_at - self.started_at
        return time.time() - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "stats": self.stats.to_dict() if self.stats else None,
            "scenario_stats": {k: v.to_dict() for k, v in self.scenario_stats.items()},
            "timeline": self.timeline,
        }


# ============================================================================
# Bus Interface
# ============================================================================

class LoadBus:
    """Bus interface for load testing."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }
        try:
            with open(self.bus_path, "a") as f:
                f.write(json.dumps(event_with_meta) + "\n")
        except IOError:
            pass


# ============================================================================
# Load Tester
# ============================================================================

class LoadTester:
    """
    Orchestrates concurrent load testing.

    Features:
    - Multiple load patterns (constant, ramp-up, step, spike, wave)
    - Concurrent user simulation
    - Latency percentile tracking
    - Scenario-based testing
    - Rate limiting support

    PBTSO Phase: TEST, VERIFY
    Bus Topics: test.load.run, test.load.result
    """

    BUS_TOPICS = {
        "run": "test.load.run",
        "result": "test.load.result",
        "telemetry": "telemetry.load",
        "progress": "test.load.progress",
    }

    def __init__(self, bus=None):
        """
        Initialize the load tester.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus or LoadBus()
        self._stop_flag = False

    def run_load_test(self, config: LoadConfig) -> LoadResult:
        """
        Execute a load test.

        Args:
            config: Load test configuration

        Returns:
            LoadResult with complete results
        """
        run_id = str(uuid.uuid4())
        result = LoadResult(
            run_id=run_id,
            config=config,
            started_at=time.time(),
        )

        self._stop_flag = False

        # Emit start event
        self._emit_event("run", {
            "run_id": run_id,
            "status": "started",
            "concurrent_users": config.concurrent_users,
            "duration_s": config.duration_s,
        })

        # Build scenario weights
        scenarios_weighted = self._build_weighted_scenarios(config.scenarios)

        # Run load test
        all_results = []
        timeline = []

        with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            # Submit virtual users
            futures = []
            start_time = time.time()

            for user_id in range(config.concurrent_users):
                # Calculate user start delay based on ramp-up
                if config.ramp_up_s > 0:
                    delay = (user_id / config.concurrent_users) * config.ramp_up_s
                else:
                    delay = 0

                future = executor.submit(
                    self._run_virtual_user,
                    user_id,
                    scenarios_weighted,
                    config,
                    start_time,
                    delay,
                )
                futures.append(future)

            # Collect results periodically
            interval_s = 5
            last_report = start_time

            while any(not f.done() for f in futures):
                time.sleep(0.5)

                current_time = time.time()
                elapsed = current_time - start_time

                # Check if test duration exceeded
                if elapsed >= config.duration_s:
                    self._stop_flag = True

                # Periodic progress report
                if current_time - last_report >= interval_s:
                    # Collect completed results so far
                    completed_count = sum(1 for f in futures if f.done())
                    self._emit_event("progress", {
                        "run_id": run_id,
                        "elapsed_s": elapsed,
                        "users_completed": completed_count,
                    })
                    last_report = current_time

            # Collect all results
            for future in futures:
                try:
                    user_results = future.result()
                    all_results.extend(user_results)
                except Exception as e:
                    pass  # User thread failed

        result.completed_at = time.time()
        result.results = all_results

        # Calculate statistics
        result.stats = LoadStats.from_results(all_results, result.duration_s)

        # Calculate per-scenario stats
        for scenario in config.scenarios:
            scenario_results = [r for r in all_results if r.scenario_name == scenario.name]
            if scenario_results:
                result.scenario_stats[scenario.name] = LoadStats.from_results(
                    scenario_results, result.duration_s
                )

        # Build timeline
        result.timeline = self._build_timeline(all_results, result.duration_s)

        # Emit completion event
        self._emit_event("result", {
            "run_id": run_id,
            "status": "completed",
            "total_requests": result.stats.total_requests,
            "requests_per_second": result.stats.requests_per_second,
            "avg_latency_ms": result.stats.avg_latency_ms,
            "error_rate": result.stats.error_rate,
        })

        # Save report
        self._save_report(result, config.output_dir)

        return result

    def _run_virtual_user(
        self,
        user_id: int,
        scenarios: List[Tuple[LoadScenario, float]],
        config: LoadConfig,
        start_time: float,
        initial_delay: float,
    ) -> List[RequestResult]:
        """Run a virtual user's workload."""
        results = []

        # Initial delay for ramp-up
        if initial_delay > 0:
            time.sleep(initial_delay)

        end_time = start_time + config.duration_s

        while not self._stop_flag and time.time() < end_time:
            # Select scenario based on weights
            scenario = self._select_scenario(scenarios)

            # Execute request
            result = self._execute_request(scenario, user_id, config)
            results.append(result)

            # Think time
            if config.think_time_ms > 0:
                time.sleep(config.think_time_ms / 1000)

            # Rate limiting
            if config.requests_per_second:
                # Simple rate limiting per user
                target_interval = 1.0 / (config.requests_per_second / config.concurrent_users)
                actual_interval = result.latency_ms / 1000
                if actual_interval < target_interval:
                    time.sleep(target_interval - actual_interval)

        return results

    def _execute_request(
        self,
        scenario: LoadScenario,
        user_id: int,
        config: LoadConfig,
    ) -> RequestResult:
        """Execute a single request."""
        request_id = str(uuid.uuid4())
        timestamp = time.time()

        try:
            # Setup if needed
            if scenario.setup:
                scenario.setup()

            # Execute with timeout
            start = time.perf_counter()

            import threading

            result_holder = [None]
            exception_holder = [None]

            def target():
                try:
                    result_holder[0] = scenario.target(**scenario.params)
                except Exception as e:
                    exception_holder[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(config.timeout_s)

            end = time.perf_counter()
            latency_ms = (end - start) * 1000

            if thread.is_alive():
                return RequestResult(
                    request_id=request_id,
                    scenario_name=scenario.name,
                    status=RequestStatus.TIMEOUT,
                    latency_ms=config.timeout_s * 1000,
                    timestamp=timestamp,
                    user_id=user_id,
                )

            if exception_holder[0]:
                raise exception_holder[0]

            response = result_holder[0]

            # Validate response if validator provided
            if scenario.validate:
                if not scenario.validate(response):
                    return RequestResult(
                        request_id=request_id,
                        scenario_name=scenario.name,
                        status=RequestStatus.FAILURE,
                        latency_ms=latency_ms,
                        timestamp=timestamp,
                        user_id=user_id,
                        error_message="Validation failed",
                    )

            # Calculate response size
            response_size = 0
            if response is not None:
                try:
                    response_size = len(str(response))
                except:
                    pass

            return RequestResult(
                request_id=request_id,
                scenario_name=scenario.name,
                status=RequestStatus.SUCCESS,
                latency_ms=latency_ms,
                timestamp=timestamp,
                response_size=response_size,
                user_id=user_id,
            )

        except Exception as e:
            return RequestResult(
                request_id=request_id,
                scenario_name=scenario.name,
                status=RequestStatus.ERROR,
                latency_ms=0,
                timestamp=timestamp,
                user_id=user_id,
                error_message=str(e),
            )

    def _build_weighted_scenarios(
        self,
        scenarios: List[LoadScenario],
    ) -> List[Tuple[LoadScenario, float]]:
        """Build weighted scenario list for selection."""
        total_weight = sum(s.weight for s in scenarios)
        result = []
        cumulative = 0

        for scenario in scenarios:
            cumulative += scenario.weight / total_weight
            result.append((scenario, cumulative))

        return result

    def _select_scenario(
        self,
        weighted_scenarios: List[Tuple[LoadScenario, float]],
    ) -> LoadScenario:
        """Select a scenario based on weights."""
        import random
        r = random.random()

        for scenario, threshold in weighted_scenarios:
            if r <= threshold:
                return scenario

        return weighted_scenarios[-1][0]

    def _build_timeline(
        self,
        results: List[RequestResult],
        duration_s: float,
    ) -> List[Dict[str, Any]]:
        """Build timeline of stats over time."""
        if not results:
            return []

        # Group by time buckets (1 second each)
        buckets: Dict[int, List[RequestResult]] = {}
        start_time = min(r.timestamp for r in results)

        for r in results:
            bucket = int(r.timestamp - start_time)
            if bucket not in buckets:
                buckets[bucket] = []
            buckets[bucket].append(r)

        # Calculate stats per bucket
        timeline = []
        for second in range(int(duration_s) + 1):
            bucket_results = buckets.get(second, [])

            if bucket_results:
                latencies = [r.latency_ms for r in bucket_results]
                successful = sum(1 for r in bucket_results if r.status == RequestStatus.SUCCESS)

                timeline.append({
                    "second": second,
                    "requests": len(bucket_results),
                    "successful": successful,
                    "avg_latency_ms": statistics.mean(latencies),
                    "error_rate": (len(bucket_results) - successful) / len(bucket_results) * 100,
                })
            else:
                timeline.append({
                    "second": second,
                    "requests": 0,
                    "successful": 0,
                    "avg_latency_ms": 0,
                    "error_rate": 0,
                })

        return timeline

    def _save_report(self, result: LoadResult, output_dir: str) -> None:
        """Save load test report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        results_file = output_path / f"load_report_{result.run_id}.json"
        with open(results_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Generate markdown report
        report_file = output_path / f"load_report_{result.run_id}.md"
        with open(report_file, "w") as f:
            f.write(self._generate_report(result))

    def _generate_report(self, result: LoadResult) -> str:
        """Generate markdown load test report."""
        stats = result.stats
        lines = [
            "# Load Test Report",
            "",
            f"**Run ID**: {result.run_id}",
            f"**Date**: {datetime.fromtimestamp(result.started_at).isoformat()}",
            f"**Duration**: {result.duration_s:.2f}s",
            f"**Concurrent Users**: {result.config.concurrent_users}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Requests | {stats.total_requests} |",
            f"| Successful | {stats.successful_requests} |",
            f"| Failed | {stats.failed_requests} |",
            f"| Timeouts | {stats.timeout_requests} |",
            f"| Requests/sec | {stats.requests_per_second:.2f} |",
            f"| Error Rate | {stats.error_rate:.2f}% |",
            "",
            "## Latency",
            "",
            f"| Percentile | Latency |",
            f"|------------|---------|",
            f"| Min | {stats.min_latency_ms:.2f}ms |",
            f"| Avg | {stats.avg_latency_ms:.2f}ms |",
            f"| p50 | {stats.p50_latency_ms:.2f}ms |",
            f"| p95 | {stats.p95_latency_ms:.2f}ms |",
            f"| p99 | {stats.p99_latency_ms:.2f}ms |",
            f"| Max | {stats.max_latency_ms:.2f}ms |",
        ]

        # Per-scenario stats
        if result.scenario_stats:
            lines.extend([
                "",
                "## Scenario Breakdown",
                "",
                "| Scenario | Requests | RPS | Avg Latency | Error Rate |",
                "|----------|----------|-----|-------------|------------|",
            ])

            for name, s in result.scenario_stats.items():
                lines.append(
                    f"| {name} | {s.total_requests} | "
                    f"{s.requests_per_second:.2f} | "
                    f"{s.avg_latency_ms:.2f}ms | "
                    f"{s.error_rate:.2f}% |"
                )

        return "\n".join(lines)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.load.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "load_test",
            "actor": "test-agent",
            "data": data,
        })

    def stop(self) -> None:
        """Stop the running load test."""
        self._stop_flag = True

    async def run_load_test_async(self, config: LoadConfig) -> LoadResult:
        """Async version of load testing."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run_load_test, config)


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Load Tester."""
    import argparse

    parser = argparse.ArgumentParser(description="Load Tester")
    parser.add_argument("module", help="Module containing load scenarios")
    parser.add_argument("--users", "-u", type=int, default=10, help="Concurrent users")
    parser.add_argument("--duration", "-d", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--ramp-up", type=int, default=10, help="Ramp-up time in seconds")
    parser.add_argument("--rps", type=float, help="Target requests per second")
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/load")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Import module and find scenarios
    import importlib
    try:
        module = importlib.import_module(args.module)
    except ImportError as e:
        print(f"Error importing module: {e}")
        exit(1)

    scenarios = []

    # Find scenario functions (decorated or prefixed with 'scenario_')
    for name in dir(module):
        if name.startswith("scenario_"):
            obj = getattr(module, name)
            if callable(obj):
                scenarios.append(LoadScenario(
                    name=name.replace("scenario_", ""),
                    target=obj,
                ))

    if not scenarios:
        print("No scenarios found in module")
        exit(1)

    config = LoadConfig(
        scenarios=scenarios,
        concurrent_users=args.users,
        duration_s=args.duration,
        ramp_up_s=args.ramp_up,
        requests_per_second=args.rps,
        output_dir=args.output,
    )

    tester = LoadTester()
    result = tester.run_load_test(config)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        stats = result.stats
        print(f"\n{'='*60}")
        print(f"Load Test Complete")
        print(f"{'='*60}")
        print(f"Run ID: {result.run_id}")
        print(f"Duration: {result.duration_s:.2f}s")
        print(f"Total Requests: {stats.total_requests}")
        print(f"Requests/sec: {stats.requests_per_second:.2f}")
        print(f"Avg Latency: {stats.avg_latency_ms:.2f}ms")
        print(f"p95 Latency: {stats.p95_latency_ms:.2f}ms")
        print(f"Error Rate: {stats.error_rate:.2f}%")
        print(f"{'='*60}")

        if stats.error_rate > 5:
            print(f"\nWARNING: Error rate exceeds 5%")
            exit(1)


if __name__ == "__main__":
    main()
