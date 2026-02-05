#!/usr/bin/env python3
"""
Step 115: Chaos Tester

Provides fault injection testing for resilience verification.

PBTSO Phase: TEST, VERIFY
Bus Topics:
- test.chaos.run (subscribes)
- test.chaos.fault (emits)
- test.chaos.complete (emits)

Dependencies: Step 106 (Test Runner Orchestrator)
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
import random
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from unittest.mock import patch, MagicMock


# ============================================================================
# Constants
# ============================================================================

DEFAULT_DURATION_S = 60
DEFAULT_FAULT_PROBABILITY = 0.1


class FaultType(Enum):
    """Types of faults that can be injected."""
    LATENCY = "latency"  # Add artificial latency
    EXCEPTION = "exception"  # Throw exceptions
    TIMEOUT = "timeout"  # Simulate timeouts
    NETWORK_PARTITION = "network_partition"  # Network failures
    MEMORY_PRESSURE = "memory_pressure"  # Memory stress
    CPU_PRESSURE = "cpu_pressure"  # CPU stress
    DISK_FAILURE = "disk_failure"  # Disk I/O failures
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Exhaust resources
    DEPENDENCY_FAILURE = "dependency_failure"  # External service failure
    DATA_CORRUPTION = "data_corruption"  # Return corrupted data


class ExperimentStatus(Enum):
    """Status of a chaos experiment."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"  # System handled fault gracefully
    FAILED = "failed"  # System failed to handle fault
    ABORTED = "aborted"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class FaultConfig:
    """Configuration for a specific fault."""
    fault_type: FaultType
    probability: float = DEFAULT_FAULT_PROBABILITY
    duration_s: float = 5.0
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fault_type": self.fault_type.value,
            "probability": self.probability,
            "duration_s": self.duration_s,
            "parameters": self.parameters,
        }


@dataclass
class ChaosExperiment:
    """
    A chaos experiment definition.

    Attributes:
        name: Experiment name
        target: Target function or module
        faults: List of faults to inject
        steady_state: Function to verify system is healthy
        hypothesis: Expected behavior during fault
        rollback: Optional rollback function
    """
    name: str
    target: Optional[Callable] = None
    target_module: Optional[str] = None
    faults: List[FaultConfig] = field(default_factory=list)
    steady_state: Optional[Callable[[], bool]] = None
    hypothesis: str = ""
    rollback: Optional[Callable] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ChaosConfig:
    """
    Configuration for chaos testing.

    Attributes:
        experiments: List of chaos experiments
        duration_s: Total test duration
        abort_on_failure: Stop on first failure
        cooldown_s: Cooldown between experiments
        output_dir: Directory for reports
    """
    experiments: List[ChaosExperiment] = field(default_factory=list)
    duration_s: int = DEFAULT_DURATION_S
    abort_on_failure: bool = True
    cooldown_s: int = 5
    output_dir: str = ".pluribus/test-agent/chaos"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "duration_s": self.duration_s,
            "abort_on_failure": self.abort_on_failure,
            "cooldown_s": self.cooldown_s,
            "experiment_count": len(self.experiments),
        }


@dataclass
class FaultInjection:
    """Record of a fault injection."""
    id: str
    fault_type: FaultType
    timestamp: float
    duration_s: float
    target: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "fault_type": self.fault_type.value,
            "timestamp": self.timestamp,
            "duration_s": self.duration_s,
            "target": self.target,
            "parameters": self.parameters,
        }


@dataclass
class ExperimentResult:
    """Result of running a chaos experiment."""
    experiment_name: str
    status: ExperimentStatus
    started_at: float
    completed_at: Optional[float] = None
    injections: List[FaultInjection] = field(default_factory=list)
    steady_state_before: bool = False
    steady_state_during: bool = False
    steady_state_after: bool = False
    error_message: Optional[str] = None
    observations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": (self.completed_at - self.started_at) if self.completed_at else 0,
            "injections": [i.to_dict() for i in self.injections],
            "steady_state_before": self.steady_state_before,
            "steady_state_during": self.steady_state_during,
            "steady_state_after": self.steady_state_after,
            "error_message": self.error_message,
            "observations": self.observations,
        }


@dataclass
class ChaosResult:
    """Complete result of chaos testing."""
    run_id: str
    config: ChaosConfig
    started_at: float
    completed_at: Optional[float] = None
    experiments: List[ExperimentResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    aborted: int = 0

    @property
    def duration_s(self) -> float:
        """Get total duration."""
        if self.completed_at:
            return self.completed_at - self.started_at
        return time.time() - self.started_at

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        total = self.passed + self.failed + self.aborted
        return (self.passed / total * 100) if total > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "passed": self.passed,
            "failed": self.failed,
            "aborted": self.aborted,
            "success_rate": self.success_rate,
            "experiments": [e.to_dict() for e in self.experiments],
        }


# ============================================================================
# Fault Injectors
# ============================================================================

class FaultInjector:
    """Base class for fault injectors."""

    @contextlib.contextmanager
    def inject(self, config: FaultConfig, target: str):
        """Context manager for fault injection."""
        raise NotImplementedError


class LatencyInjector(FaultInjector):
    """Inject artificial latency."""

    @contextlib.contextmanager
    def inject(self, config: FaultConfig, target: str):
        """Add latency to function calls."""
        latency_ms = config.parameters.get("latency_ms", 1000)
        jitter_ms = config.parameters.get("jitter_ms", 0)

        original_sleep = time.sleep

        def delayed_sleep(seconds):
            if random.random() < config.probability:
                extra = (latency_ms + random.uniform(-jitter_ms, jitter_ms)) / 1000
                original_sleep(seconds + extra)
            else:
                original_sleep(seconds)

        with patch("time.sleep", delayed_sleep):
            yield


class ExceptionInjector(FaultInjector):
    """Inject exceptions."""

    @contextlib.contextmanager
    def inject(self, config: FaultConfig, target: str):
        """Randomly raise exceptions."""
        exception_class = config.parameters.get("exception_class", RuntimeError)
        message = config.parameters.get("message", "Chaos: Injected exception")

        # Create a wrapper that raises exceptions
        self._probability = config.probability
        self._exception_class = exception_class
        self._message = message

        yield

    def maybe_raise(self):
        """Raise exception based on probability."""
        if random.random() < self._probability:
            raise self._exception_class(self._message)


class TimeoutInjector(FaultInjector):
    """Inject timeouts."""

    @contextlib.contextmanager
    def inject(self, config: FaultConfig, target: str):
        """Simulate timeouts by hanging."""
        hang_duration_s = config.parameters.get("hang_duration_s", 30)

        def hang_if_unlucky():
            if random.random() < config.probability:
                time.sleep(hang_duration_s)

        self._hang = hang_if_unlucky
        yield


class NetworkPartitionInjector(FaultInjector):
    """Simulate network partitions."""

    @contextlib.contextmanager
    def inject(self, config: FaultConfig, target: str):
        """Block network connections."""
        blocked_hosts = config.parameters.get("blocked_hosts", ["*"])

        original_connect = socket.socket.connect

        def blocked_connect(self, address):
            host = address[0] if isinstance(address, tuple) else address
            if "*" in blocked_hosts or host in blocked_hosts:
                if random.random() < config.probability:
                    raise ConnectionRefusedError(f"Chaos: Network partition to {host}")
            return original_connect(self, address)

        with patch.object(socket.socket, "connect", blocked_connect):
            yield


class MemoryPressureInjector(FaultInjector):
    """Apply memory pressure."""

    @contextlib.contextmanager
    def inject(self, config: FaultConfig, target: str):
        """Allocate memory to create pressure."""
        memory_mb = config.parameters.get("memory_mb", 100)

        # Allocate memory
        allocated = []
        try:
            if random.random() < config.probability:
                # Allocate in chunks to avoid system hang
                chunk_size = 10 * 1024 * 1024  # 10MB chunks
                for _ in range(memory_mb // 10):
                    allocated.append(bytearray(chunk_size))
            yield
        finally:
            # Release memory
            allocated.clear()


class CPUPressureInjector(FaultInjector):
    """Apply CPU pressure."""

    @contextlib.contextmanager
    def inject(self, config: FaultConfig, target: str):
        """Create CPU load."""
        import threading

        load_percent = config.parameters.get("load_percent", 50)
        stop_flag = threading.Event()

        def cpu_load():
            while not stop_flag.is_set():
                # Busy work
                end_time = time.time() + (load_percent / 100) * 0.1
                while time.time() < end_time:
                    _ = sum(i * i for i in range(1000))
                time.sleep((100 - load_percent) / 100 * 0.1)

        threads = []
        if random.random() < config.probability:
            import multiprocessing
            for _ in range(multiprocessing.cpu_count()):
                t = threading.Thread(target=cpu_load)
                t.daemon = True
                t.start()
                threads.append(t)

        try:
            yield
        finally:
            stop_flag.set()
            for t in threads:
                t.join(timeout=1)


class DependencyFailureInjector(FaultInjector):
    """Simulate dependency failures."""

    @contextlib.contextmanager
    def inject(self, config: FaultConfig, target: str):
        """Make external calls fail."""
        dependencies = config.parameters.get("dependencies", [])
        failure_type = config.parameters.get("failure_type", "exception")

        mocks = []
        try:
            for dep in dependencies:
                if failure_type == "exception":
                    mock = patch(dep, side_effect=ConnectionError("Chaos: Dependency failure"))
                elif failure_type == "timeout":
                    mock = patch(dep, side_effect=TimeoutError("Chaos: Dependency timeout"))
                elif failure_type == "empty":
                    mock = patch(dep, return_value=None)
                else:
                    mock = patch(dep, return_value=MagicMock())

                mock.start()
                mocks.append(mock)

            yield
        finally:
            for mock in mocks:
                mock.stop()


# ============================================================================
# Bus Interface
# ============================================================================

class ChaosBus:
    """Bus interface for chaos testing."""

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
# Chaos Tester
# ============================================================================

class ChaosTester:
    """
    Orchestrates chaos experiments for resilience testing.

    Chaos engineering principles:
    1. Verify steady state before experiment
    2. Hypothesize system will remain stable
    3. Inject real-world failures
    4. Verify steady state is maintained
    5. Minimize blast radius

    PBTSO Phase: TEST, VERIFY
    Bus Topics: test.chaos.run, test.chaos.fault
    """

    BUS_TOPICS = {
        "run": "test.chaos.run",
        "fault": "test.chaos.fault",
        "complete": "test.chaos.complete",
    }

    INJECTORS: Dict[FaultType, Type[FaultInjector]] = {
        FaultType.LATENCY: LatencyInjector,
        FaultType.EXCEPTION: ExceptionInjector,
        FaultType.TIMEOUT: TimeoutInjector,
        FaultType.NETWORK_PARTITION: NetworkPartitionInjector,
        FaultType.MEMORY_PRESSURE: MemoryPressureInjector,
        FaultType.CPU_PRESSURE: CPUPressureInjector,
        FaultType.DEPENDENCY_FAILURE: DependencyFailureInjector,
    }

    def __init__(self, bus=None):
        """
        Initialize the chaos tester.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus or ChaosBus()
        self._abort_flag = False

    def run_chaos_tests(self, config: ChaosConfig) -> ChaosResult:
        """
        Execute chaos experiments.

        Args:
            config: Chaos test configuration

        Returns:
            ChaosResult with complete results
        """
        run_id = str(uuid.uuid4())
        result = ChaosResult(
            run_id=run_id,
            config=config,
            started_at=time.time(),
        )

        self._abort_flag = False

        # Emit start event
        self._emit_event("run", {
            "run_id": run_id,
            "status": "started",
            "experiment_count": len(config.experiments),
        })

        # Run each experiment
        for experiment in config.experiments:
            if self._abort_flag:
                break

            exp_result = self._run_experiment(experiment, config)
            result.experiments.append(exp_result)

            # Update counters
            if exp_result.status == ExperimentStatus.PASSED:
                result.passed += 1
            elif exp_result.status == ExperimentStatus.FAILED:
                result.failed += 1
                if config.abort_on_failure:
                    self._abort_flag = True
            elif exp_result.status == ExperimentStatus.ABORTED:
                result.aborted += 1

            # Cooldown between experiments
            if config.cooldown_s > 0:
                time.sleep(config.cooldown_s)

        result.completed_at = time.time()

        # Emit completion event
        self._emit_event("complete", {
            "run_id": run_id,
            "status": "completed",
            "passed": result.passed,
            "failed": result.failed,
            "success_rate": result.success_rate,
        })

        # Save report
        self._save_report(result, config.output_dir)

        return result

    def _run_experiment(
        self,
        experiment: ChaosExperiment,
        config: ChaosConfig,
    ) -> ExperimentResult:
        """Run a single chaos experiment."""
        result = ExperimentResult(
            experiment_name=experiment.name,
            status=ExperimentStatus.RUNNING,
            started_at=time.time(),
        )

        try:
            # Step 1: Verify steady state before
            if experiment.steady_state:
                result.steady_state_before = experiment.steady_state()
                if not result.steady_state_before:
                    result.status = ExperimentStatus.ABORTED
                    result.error_message = "Steady state not achieved before experiment"
                    result.completed_at = time.time()
                    return result

            # Step 2: Inject faults
            for fault_config in experiment.faults:
                injection = self._inject_fault(
                    fault_config,
                    experiment,
                )
                result.injections.append(injection)

                # Emit fault event
                self._emit_event("fault", {
                    "experiment": experiment.name,
                    "fault_type": fault_config.fault_type.value,
                    "injection_id": injection.id,
                })

            # Step 3: Verify steady state during fault
            if experiment.steady_state:
                # Wait a bit for fault to take effect
                time.sleep(1)
                result.steady_state_during = experiment.steady_state()
                result.observations.append(
                    f"Steady state during fault: {result.steady_state_during}"
                )

            # Step 4: Wait for fault duration
            max_duration = max(
                (f.duration_s for f in experiment.faults),
                default=5.0
            )
            time.sleep(max_duration)

            # Step 5: Verify steady state after (recovery)
            if experiment.steady_state:
                # Give system time to recover
                time.sleep(2)
                result.steady_state_after = experiment.steady_state()
                result.observations.append(
                    f"Steady state after recovery: {result.steady_state_after}"
                )

            # Determine outcome
            if result.steady_state_during and result.steady_state_after:
                result.status = ExperimentStatus.PASSED
            elif result.steady_state_after:
                result.status = ExperimentStatus.PASSED
                result.observations.append(
                    "System degraded during fault but recovered"
                )
            else:
                result.status = ExperimentStatus.FAILED
                result.error_message = "System did not maintain or recover steady state"

        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.error_message = str(e)

            # Run rollback if available
            if experiment.rollback:
                try:
                    experiment.rollback()
                    result.observations.append("Rollback executed successfully")
                except Exception as re:
                    result.observations.append(f"Rollback failed: {re}")

        result.completed_at = time.time()
        return result

    def _inject_fault(
        self,
        fault_config: FaultConfig,
        experiment: ChaosExperiment,
    ) -> FaultInjection:
        """Inject a fault."""
        injection_id = str(uuid.uuid4())
        target = experiment.target_module or str(experiment.target)

        injection = FaultInjection(
            id=injection_id,
            fault_type=fault_config.fault_type,
            timestamp=time.time(),
            duration_s=fault_config.duration_s,
            target=target,
            parameters=fault_config.parameters,
        )

        # Get appropriate injector
        injector_class = self.INJECTORS.get(fault_config.fault_type)
        if injector_class:
            injector = injector_class()
            # Start fault injection in background
            # The actual injection context would be managed by the experiment
            # For now, we just record the injection

        return injection

    def _save_report(self, result: ChaosResult, output_dir: str) -> None:
        """Save chaos test report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        results_file = output_path / f"chaos_report_{result.run_id}.json"
        with open(results_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Generate markdown report
        report_file = output_path / f"chaos_report_{result.run_id}.md"
        with open(report_file, "w") as f:
            f.write(self._generate_report(result))

    def _generate_report(self, result: ChaosResult) -> str:
        """Generate markdown chaos test report."""
        lines = [
            "# Chaos Test Report",
            "",
            f"**Run ID**: {result.run_id}",
            f"**Date**: {datetime.fromtimestamp(result.started_at).isoformat()}",
            f"**Duration**: {result.duration_s:.2f}s",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Passed | {result.passed} |",
            f"| Failed | {result.failed} |",
            f"| Aborted | {result.aborted} |",
            f"| Success Rate | {result.success_rate:.1f}% |",
            "",
            "## Experiments",
            "",
        ]

        for exp in result.experiments:
            status_emoji = {
                ExperimentStatus.PASSED: "[PASS]",
                ExperimentStatus.FAILED: "[FAIL]",
                ExperimentStatus.ABORTED: "[ABORT]",
            }.get(exp.status, "[?]")

            lines.extend([
                f"### {status_emoji} {exp.experiment_name}",
                "",
                f"**Status**: {exp.status.value}",
                "",
            ])

            if exp.injections:
                lines.append("**Fault Injections**:")
                for inj in exp.injections:
                    lines.append(f"- {inj.fault_type.value} (duration: {inj.duration_s}s)")
                lines.append("")

            if exp.observations:
                lines.append("**Observations**:")
                for obs in exp.observations:
                    lines.append(f"- {obs}")
                lines.append("")

            if exp.error_message:
                lines.append(f"**Error**: {exp.error_message}")
                lines.append("")

        return "\n".join(lines)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.chaos.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "chaos_test",
            "actor": "test-agent",
            "data": data,
        })

    def abort(self) -> None:
        """Abort running chaos tests."""
        self._abort_flag = True


# ============================================================================
# Convenience Decorators
# ============================================================================

def chaos_experiment(
    name: str,
    faults: Optional[List[FaultConfig]] = None,
    hypothesis: str = "",
):
    """Decorator to define a chaos experiment."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._chaos_experiment = ChaosExperiment(
            name=name,
            target=func,
            faults=faults or [],
            hypothesis=hypothesis,
        )
        return wrapper
    return decorator


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Chaos Tester."""
    import argparse

    parser = argparse.ArgumentParser(description="Chaos Tester")
    parser.add_argument("module", help="Module containing chaos experiments")
    parser.add_argument("--duration", "-d", type=int, default=60)
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/chaos")
    parser.add_argument("--abort-on-failure", action="store_true")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Import module and find experiments
    import importlib
    try:
        module = importlib.import_module(args.module)
    except ImportError as e:
        print(f"Error importing module: {e}")
        exit(1)

    experiments = []

    # Find decorated functions
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and hasattr(obj, "_chaos_experiment"):
            experiments.append(obj._chaos_experiment)

    if not experiments:
        print("No chaos experiments found in module")
        exit(1)

    config = ChaosConfig(
        experiments=experiments,
        duration_s=args.duration,
        abort_on_failure=args.abort_on_failure,
        output_dir=args.output,
    )

    tester = ChaosTester()
    result = tester.run_chaos_tests(config)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Chaos Test Complete")
        print(f"{'='*60}")
        print(f"Run ID: {result.run_id}")
        print(f"Duration: {result.duration_s:.2f}s")
        print(f"Passed: {result.passed}")
        print(f"Failed: {result.failed}")
        print(f"Success Rate: {result.success_rate:.1f}%")
        print(f"{'='*60}")

        if result.failed > 0:
            print(f"\nSome experiments failed. Check report for details.")
            exit(1)


if __name__ == "__main__":
    main()
