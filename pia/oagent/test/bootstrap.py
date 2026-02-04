#!/usr/bin/env python3
"""
Step 101: Test Agent Bootstrap Module

Initializes the Test Agent with configuration and A2A bus integration.

PBTSO Phase: SKILL, SEQUESTER
Bus Topics:
- a2a.test.bootstrap.start (emits)
- a2a.test.bootstrap.complete (emits)
- a2a.task.dispatch (subscribes)

Dependencies: Code Agent (Steps 51-100)
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# Constants
# ============================================================================

DEFAULT_COVERAGE_THRESHOLD = 0.8
DEFAULT_PARALLEL_WORKERS = 4
DEFAULT_RING_LEVEL = 2  # Test Agent operates at Ring 2


class PBTSOPhase(Enum):
    """PBTSO execution phases for the Test Agent."""
    PLAN = "PLAN"          # Test planning and strategy
    BUILD = "BUILD"        # Test fixture/mock generation
    TEST = "TEST"          # Test execution
    SKILL = "SKILL"        # Test generation capabilities
    OBSERVE = "OBSERVE"    # Coverage observation
    SEQUESTER = "SEQUESTER"  # Test isolation
    VERIFY = "VERIFY"      # Result verification
    DISTRIBUTE = "DISTRIBUTE"  # Parallel distribution


class TestFramework(Enum):
    """Supported test frameworks."""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    VITEST = "vitest"
    JEST = "jest"
    MOCHA = "mocha"


class TestType(Enum):
    """Types of tests the agent can generate/run."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PROPERTY = "property"
    MUTATION = "mutation"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TestAgentConfig:
    """
    Configuration for the Test Agent.

    Attributes:
        agent_id: Unique identifier for this agent instance
        ring_level: Security ring level (0=core, 1=trusted, 2=standard)
        parallel_workers: Number of parallel test workers
        coverage_threshold: Minimum code coverage threshold (0.0-1.0)
        mutation_testing: Enable mutation testing
        frameworks: Enabled test frameworks
        test_types: Enabled test types
        output_dir: Directory for test outputs
        timeout_s: Default test timeout in seconds
        quarantine_flaky: Auto-quarantine flaky tests
    """
    agent_id: str = "test-agent"
    ring_level: int = DEFAULT_RING_LEVEL
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD
    mutation_testing: bool = True
    frameworks: List[TestFramework] = field(default_factory=lambda: [TestFramework.PYTEST])
    test_types: List[TestType] = field(default_factory=lambda: [
        TestType.UNIT, TestType.INTEGRATION, TestType.PROPERTY
    ])
    output_dir: str = ".pluribus/test-agent"
    timeout_s: int = 300
    quarantine_flaky: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "ring_level": self.ring_level,
            "parallel_workers": self.parallel_workers,
            "coverage_threshold": self.coverage_threshold,
            "mutation_testing": self.mutation_testing,
            "frameworks": [f.value for f in self.frameworks],
            "test_types": [t.value for t in self.test_types],
            "output_dir": self.output_dir,
            "timeout_s": self.timeout_s,
            "quarantine_flaky": self.quarantine_flaky,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestAgentConfig":
        """Create config from dictionary."""
        return cls(
            agent_id=data.get("agent_id", "test-agent"),
            ring_level=data.get("ring_level", DEFAULT_RING_LEVEL),
            parallel_workers=data.get("parallel_workers", DEFAULT_PARALLEL_WORKERS),
            coverage_threshold=data.get("coverage_threshold", DEFAULT_COVERAGE_THRESHOLD),
            mutation_testing=data.get("mutation_testing", True),
            frameworks=[TestFramework(f) for f in data.get("frameworks", ["pytest"])],
            test_types=[TestType(t) for t in data.get("test_types", ["unit", "integration", "property"])],
            output_dir=data.get("output_dir", ".pluribus/test-agent"),
            timeout_s=data.get("timeout_s", 300),
            quarantine_flaky=data.get("quarantine_flaky", True),
        )


# ============================================================================
# Bus Interface (Lightweight)
# ============================================================================

class TestAgentBus:
    """
    Lightweight bus interface for Test Agent.

    Emits events to the A2A bus and handles subscriptions.
    Compatible with the broader PIA bus infrastructure.
    """

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self.handlers: Dict[str, List[Callable]] = {}

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """
        Emit an event to the bus.

        Args:
            event: Event dictionary with topic, kind, actor, data fields
        """
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                f.write(json.dumps(event_with_meta) + "\n")
        except IOError as e:
            # Log but don't fail on bus write errors
            print(f"[TestAgentBus] Warning: Failed to emit event: {e}")

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to a bus topic.

        Args:
            topic: Topic pattern to subscribe to
            handler: Callback function for matching events
        """
        if topic not in self.handlers:
            self.handlers[topic] = []
        self.handlers[topic].append(handler)

    def _topic_matches(self, pattern: str, topic: str) -> bool:
        """Check if a topic matches a pattern (supports wildcards)."""
        if pattern == topic:
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return topic.startswith(prefix + ".")
        return False


# ============================================================================
# Test Agent Bootstrap
# ============================================================================

@dataclass
class BootstrapState:
    """State of the Test Agent bootstrap process."""
    phase: PBTSOPhase = PBTSOPhase.SKILL
    status: str = "initializing"
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    generators_loaded: List[str] = field(default_factory=list)
    runners_loaded: List[str] = field(default_factory=list)
    error: Optional[str] = None


class TestAgentBootstrap:
    """
    Bootstrap manager for the Test Agent.

    Responsibilities:
    - Initialize agent configuration
    - Connect to A2A bus
    - Load test generators and runners
    - Emit bootstrap lifecycle events

    PBTSO Phases: SKILL, SEQUESTER
    """

    BUS_TOPICS = {
        "bootstrap_start": "a2a.test.bootstrap.start",
        "bootstrap_complete": "a2a.test.bootstrap.complete",
        "task_dispatch": "a2a.task.dispatch",
        "test_ready": "a2a.test.ready",
    }

    def __init__(self, config: Optional[TestAgentConfig] = None):
        """
        Initialize the Test Agent Bootstrap.

        Args:
            config: Agent configuration (uses defaults if not provided)
        """
        self.config = config or TestAgentConfig()
        self.bus = TestAgentBus()
        self.state = BootstrapState()
        self._generators: Dict[str, Any] = {}
        self._runners: Dict[str, Any] = {}

        # Subscribe to task dispatch for incoming tasks
        self.bus.subscribe(self.BUS_TOPICS["task_dispatch"], self._handle_task_dispatch)

    def bootstrap(self) -> bool:
        """
        Execute the bootstrap sequence.

        Returns:
            True if bootstrap succeeded, False otherwise
        """
        self._emit_bootstrap_start()

        try:
            # Phase 1: SKILL - Load generators
            self.state.phase = PBTSOPhase.SKILL
            self._load_generators()

            # Phase 2: SEQUESTER - Set up isolation
            self.state.phase = PBTSOPhase.SEQUESTER
            self._setup_isolation()

            # Mark complete
            self.state.status = "ready"
            self.state.completed_at = time.time()

            self._emit_bootstrap_complete(success=True)
            return True

        except Exception as e:
            self.state.status = "failed"
            self.state.error = str(e)
            self._emit_bootstrap_complete(success=False, error=str(e))
            return False

    def _load_generators(self) -> None:
        """Load test generators based on config."""
        # Lazy import to avoid circular dependencies
        from .generators.unit import UnitTestGenerator
        from .generators.integration import IntegrationTestGenerator
        from .generators.e2e import E2ETestGenerator
        from .generators.property import PropertyTestGenerator

        generator_map = {
            TestType.UNIT: UnitTestGenerator,
            TestType.INTEGRATION: IntegrationTestGenerator,
            TestType.E2E: E2ETestGenerator,
            TestType.PROPERTY: PropertyTestGenerator,
        }

        for test_type in self.config.test_types:
            if test_type in generator_map:
                gen_class = generator_map[test_type]
                self._generators[test_type.value] = gen_class(self.bus)
                self.state.generators_loaded.append(test_type.value)

    def _setup_isolation(self) -> None:
        """Set up test isolation environment."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_path / "results").mkdir(exist_ok=True)
        (output_path / "coverage").mkdir(exist_ok=True)
        (output_path / "mutations").mkdir(exist_ok=True)
        (output_path / "quarantine").mkdir(exist_ok=True)

    def _emit_bootstrap_start(self) -> None:
        """Emit bootstrap start event."""
        self.bus.emit({
            "topic": self.BUS_TOPICS["bootstrap_start"],
            "kind": "lifecycle",
            "actor": self.config.agent_id,
            "data": {
                "config": self.config.to_dict(),
                "started_at": self.state.started_at,
            }
        })

    def _emit_bootstrap_complete(self, success: bool, error: Optional[str] = None) -> None:
        """Emit bootstrap complete event."""
        self.bus.emit({
            "topic": self.BUS_TOPICS["bootstrap_complete"],
            "kind": "lifecycle",
            "actor": self.config.agent_id,
            "data": {
                "success": success,
                "generators_loaded": self.state.generators_loaded,
                "runners_loaded": self.state.runners_loaded,
                "duration_s": (self.state.completed_at or time.time()) - self.state.started_at,
                "error": error,
            }
        })

        if success:
            # Also emit ready signal
            self.bus.emit({
                "topic": self.BUS_TOPICS["test_ready"],
                "kind": "lifecycle",
                "actor": self.config.agent_id,
                "data": {
                    "capabilities": self.state.generators_loaded,
                    "frameworks": [f.value for f in self.config.frameworks],
                }
            })

    def _handle_task_dispatch(self, event: Dict[str, Any]) -> None:
        """Handle incoming task dispatch events."""
        data = event.get("data", {})
        target = data.get("target")

        # Only process tasks targeting this agent
        if target != self.config.agent_id:
            return

        topic = data.get("topic", "")
        payload = data.get("payload", {})

        # Route to appropriate handler
        if topic.startswith("test.unit."):
            self._handle_unit_test_request(payload)
        elif topic.startswith("test.integration."):
            self._handle_integration_test_request(payload)
        elif topic.startswith("test.coverage."):
            self._handle_coverage_request(payload)

    def _handle_unit_test_request(self, payload: Dict[str, Any]) -> None:
        """Handle unit test generation request."""
        if "unit" in self._generators:
            self._generators["unit"].generate(payload)

    def _handle_integration_test_request(self, payload: Dict[str, Any]) -> None:
        """Handle integration test generation request."""
        if "integration" in self._generators:
            self._generators["integration"].generate(payload)

    def _handle_coverage_request(self, payload: Dict[str, Any]) -> None:
        """Handle coverage analysis request."""
        # Delegate to coverage analyzer
        pass

    def get_generator(self, test_type: str) -> Optional[Any]:
        """Get a loaded generator by type."""
        return self._generators.get(test_type)

    def get_state(self) -> Dict[str, Any]:
        """Get current bootstrap state."""
        return {
            "phase": self.state.phase.value,
            "status": self.state.status,
            "started_at": self.state.started_at,
            "completed_at": self.state.completed_at,
            "generators_loaded": self.state.generators_loaded,
            "runners_loaded": self.state.runners_loaded,
            "error": self.state.error,
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Agent Bootstrap."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Agent Bootstrap")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--coverage", type=float, default=0.8, help="Coverage threshold")
    parser.add_argument("--status", action="store_true", help="Show bootstrap status")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without executing")

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config) as f:
            config = TestAgentConfig.from_dict(json.load(f))
    else:
        config = TestAgentConfig(
            parallel_workers=args.workers,
            coverage_threshold=args.coverage,
        )

    if args.dry_run:
        print(f"[DRY RUN] Would bootstrap Test Agent with config:")
        print(json.dumps(config.to_dict(), indent=2))
        return

    # Bootstrap
    bootstrap = TestAgentBootstrap(config)

    if args.status:
        print(json.dumps(bootstrap.get_state(), indent=2))
        return

    success = bootstrap.bootstrap()

    if success:
        print(f"[OK] Test Agent bootstrapped successfully")
        print(f"     Generators: {bootstrap.state.generators_loaded}")
    else:
        print(f"[FAIL] Test Agent bootstrap failed: {bootstrap.state.error}")
        exit(1)


if __name__ == "__main__":
    main()
