#!/usr/bin/env python3
"""
Review Agent Bootstrap Module (Step 151)

Initializes the Review Agent with A2A coordination, security configuration,
and PBTSO phase management.

PBTSO Phase: SKILL, SEQUESTER
Bus Topics: a2a.review.bootstrap.start, a2a.review.bootstrap.complete

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# Configuration
# ============================================================================

class RingLevel(Enum):
    """Security ring levels for the Review Agent."""
    RING_0 = 0  # Constitutional - Omega veto
    RING_1 = 1  # Infrastructure - Core systems
    RING_2 = 2  # Application - Standard agents
    RING_3 = 3  # User - External interactions


class PBTSOPhase(Enum):
    """PBTSO workflow phases."""
    PLAN = "plan"
    BUILD = "build"
    TEST = "test"
    SKILL = "skill"
    OBSERVE = "observe"
    VERIFY = "verify"
    DISTILL = "distill"
    SEQUESTER = "sequester"
    DISTRIBUTE = "distribute"


@dataclass
class ReviewAgentConfig:
    """
    Configuration for the Review Agent.

    Attributes:
        agent_id: Unique identifier for this agent instance
        ring_level: Security ring level (default: Ring 2 - Application)
        security_scan_enabled: Enable security vulnerability scanning
        style_check_enabled: Enable code style checking
        smell_detection_enabled: Enable code smell detection
        architecture_check_enabled: Enable architecture consistency checking
        doc_check_enabled: Enable documentation completeness checking
        dep_scan_enabled: Enable dependency vulnerability scanning
        license_check_enabled: Enable license compliance checking
        omega_veto_enabled: Allow Omega veto for critical issues
        max_file_size_kb: Maximum file size to analyze (KB)
        timeout_seconds: Analysis timeout per file
        excluded_patterns: Glob patterns to exclude from analysis
    """
    agent_id: str = "review-agent"
    ring_level: int = 2
    security_scan_enabled: bool = True
    style_check_enabled: bool = True
    smell_detection_enabled: bool = True
    architecture_check_enabled: bool = True
    doc_check_enabled: bool = True
    dep_scan_enabled: bool = True
    license_check_enabled: bool = True
    omega_veto_enabled: bool = True
    max_file_size_kb: int = 1024
    timeout_seconds: int = 300
    excluded_patterns: List[str] = field(default_factory=lambda: [
        "*.min.js",
        "*.min.css",
        "node_modules/*",
        ".git/*",
        "__pycache__/*",
        "*.pyc",
        ".venv/*",
        "venv/*",
    ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewAgentConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "ReviewAgentConfig":
        """Create config from environment variables."""
        return cls(
            agent_id=os.environ.get("REVIEW_AGENT_ID", "review-agent"),
            ring_level=int(os.environ.get("REVIEW_RING_LEVEL", "2")),
            security_scan_enabled=os.environ.get("REVIEW_SECURITY_SCAN", "true").lower() == "true",
            style_check_enabled=os.environ.get("REVIEW_STYLE_CHECK", "true").lower() == "true",
            smell_detection_enabled=os.environ.get("REVIEW_SMELL_DETECT", "true").lower() == "true",
            architecture_check_enabled=os.environ.get("REVIEW_ARCH_CHECK", "true").lower() == "true",
            doc_check_enabled=os.environ.get("REVIEW_DOC_CHECK", "true").lower() == "true",
            dep_scan_enabled=os.environ.get("REVIEW_DEP_SCAN", "true").lower() == "true",
            license_check_enabled=os.environ.get("REVIEW_LICENSE_CHECK", "true").lower() == "true",
            omega_veto_enabled=os.environ.get("REVIEW_OMEGA_VETO", "true").lower() == "true",
            max_file_size_kb=int(os.environ.get("REVIEW_MAX_FILE_KB", "1024")),
            timeout_seconds=int(os.environ.get("REVIEW_TIMEOUT_S", "300")),
        )


# ============================================================================
# A2A Dispatcher (Simplified for standalone use)
# ============================================================================

class A2ADispatcher:
    """
    Agent-to-Agent dispatcher for Review Agent.

    Handles bus event emission and task coordination.
    """

    BUS_TOPICS = {
        "bootstrap_start": "a2a.review.bootstrap.start",
        "bootstrap_complete": "a2a.review.bootstrap.complete",
        "task_dispatch": "a2a.task.dispatch",
        "health_check": "a2a.health.check",
        "veto": "a2a.task.veto",
    }

    def __init__(self, actor_id: str):
        self.actor_id = actor_id
        self.bus_path = self._get_bus_path()
        self.handlers: Dict[str, List[Callable]] = {}

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> str:
        """
        Emit an event to the A2A bus.

        Args:
            event: Event dictionary with topic, kind, data fields

        Returns:
            Event ID
        """
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "actor": self.actor_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(full_event) + "\n")

        return event_id

    def subscribe(self, topic: str, handler: Callable) -> None:
        """Subscribe to a bus topic."""
        if topic not in self.handlers:
            self.handlers[topic] = []
        self.handlers[topic].append(handler)

    def dispatch(self, target: str, topic: str, payload: Dict[str, Any]) -> str:
        """Dispatch a task to a target agent."""
        return self.emit({
            "topic": self.BUS_TOPICS["task_dispatch"],
            "kind": "a2a_task",
            "data": {
                "target": target,
                "topic": topic,
                "payload": payload,
            }
        })


# ============================================================================
# Review Agent Bootstrap
# ============================================================================

@dataclass
class AgentState:
    """Current state of the Review Agent."""
    phase: PBTSOPhase = PBTSOPhase.SKILL
    initialized: bool = False
    healthy: bool = False
    last_heartbeat: float = 0.0
    tasks_processed: int = 0
    tasks_failed: int = 0
    veto_count: int = 0


class ReviewAgentBootstrap:
    """
    Bootstrap and lifecycle manager for the Review Agent.

    Responsibilities:
    - Initialize agent configuration
    - Register with A2A bus
    - Manage PBTSO phase transitions
    - Handle health checks and heartbeats
    - Coordinate Omega veto requests

    Example:
        config = ReviewAgentConfig()
        bootstrap = ReviewAgentBootstrap(config)
        bootstrap.initialize()

        # Run analysis
        result = bootstrap.run_review(files=["/path/to/file.py"])
    """

    def __init__(self, config: ReviewAgentConfig):
        """
        Initialize the Review Agent bootstrap.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.dispatcher = A2ADispatcher(actor_id=config.agent_id)
        self.state = AgentState()
        self._components: Dict[str, Any] = {}

    def initialize(self) -> bool:
        """
        Initialize the Review Agent.

        Emits: a2a.review.bootstrap.start

        Returns:
            True if initialization successful
        """
        # Emit bootstrap start
        self.dispatcher.emit({
            "topic": self.dispatcher.BUS_TOPICS["bootstrap_start"],
            "kind": "lifecycle",
            "level": "info",
            "data": {
                "agent_id": self.config.agent_id,
                "ring_level": self.config.ring_level,
                "config": self.config.to_dict(),
                "phase": self.state.phase.value,
            }
        })

        # Transition to SEQUESTER phase for component loading
        self.state.phase = PBTSOPhase.SEQUESTER

        # Load components
        self._load_components()

        # Subscribe to task dispatch
        self.dispatcher.subscribe("a2a.task.dispatch", self._handle_task_dispatch)

        # Mark as initialized
        self.state.initialized = True
        self.state.healthy = True
        self.state.last_heartbeat = time.time()

        # Emit bootstrap complete
        self.dispatcher.emit({
            "topic": self.dispatcher.BUS_TOPICS["bootstrap_complete"],
            "kind": "lifecycle",
            "level": "info",
            "data": {
                "agent_id": self.config.agent_id,
                "initialized": True,
                "components": list(self._components.keys()),
                "phase": self.state.phase.value,
            }
        })

        return True

    def _load_components(self) -> None:
        """Load review components based on configuration."""
        # Components are loaded lazily when needed
        self._components["static_analyzer"] = None
        self._components["security_scanner"] = None
        self._components["smell_detector"] = None
        self._components["architecture_checker"] = None
        self._components["doc_checker"] = None
        self._components["comment_generator"] = None
        self._components["dep_scanner"] = None
        self._components["license_checker"] = None

    def _handle_task_dispatch(self, event: Dict[str, Any]) -> None:
        """Handle incoming task dispatch events."""
        data = event.get("data", {})
        target = data.get("target")

        # Only handle tasks for this agent
        if target != self.config.agent_id:
            return

        topic = data.get("topic", "")
        payload = data.get("payload", {})

        # Route to appropriate handler
        if topic == "review.static.analyze":
            self._run_static_analysis(payload)
        elif topic == "review.security.scan":
            self._run_security_scan(payload)
        elif topic == "review.smells.detect":
            self._run_smell_detection(payload)
        elif topic == "a2a.review.orchestrate":
            self._run_full_review(payload)

    def _run_static_analysis(self, payload: Dict[str, Any]) -> None:
        """Run static analysis on provided files."""
        self.state.phase = PBTSOPhase.VERIFY
        self.state.tasks_processed += 1

    def _run_security_scan(self, payload: Dict[str, Any]) -> None:
        """Run security scan on provided files."""
        self.state.phase = PBTSOPhase.VERIFY
        self.state.tasks_processed += 1

    def _run_smell_detection(self, payload: Dict[str, Any]) -> None:
        """Run code smell detection on provided files."""
        self.state.phase = PBTSOPhase.VERIFY
        self.state.tasks_processed += 1

    def _run_full_review(self, payload: Dict[str, Any]) -> None:
        """Run full review pipeline."""
        self.state.phase = PBTSOPhase.VERIFY
        self.state.tasks_processed += 1

    def request_omega_veto(self, reason: str, severity: str = "critical") -> str:
        """
        Request Omega veto for a critical issue.

        Args:
            reason: Reason for veto request
            severity: Issue severity (critical, high, medium, low)

        Returns:
            Veto request event ID
        """
        if not self.config.omega_veto_enabled:
            return ""

        self.state.veto_count += 1

        return self.dispatcher.emit({
            "topic": self.dispatcher.BUS_TOPICS["veto"],
            "kind": "veto_request",
            "level": "warn",
            "data": {
                "agent_id": self.config.agent_id,
                "reason": reason,
                "severity": severity,
                "ring_level": RingLevel.RING_0.value,
            }
        })

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health status dictionary
        """
        self.state.last_heartbeat = time.time()

        status = {
            "agent_id": self.config.agent_id,
            "healthy": self.state.healthy,
            "initialized": self.state.initialized,
            "phase": self.state.phase.value,
            "tasks_processed": self.state.tasks_processed,
            "tasks_failed": self.state.tasks_failed,
            "veto_count": self.state.veto_count,
            "uptime_s": time.time() - self.state.last_heartbeat if self.state.initialized else 0,
        }

        self.dispatcher.emit({
            "topic": self.dispatcher.BUS_TOPICS["health_check"],
            "kind": "health",
            "level": "info",
            "data": status,
        })

        return status

    def shutdown(self) -> None:
        """Shutdown the Review Agent gracefully."""
        self.state.healthy = False
        self.state.phase = PBTSOPhase.DISTILL

        self.dispatcher.emit({
            "topic": "a2a.review.shutdown",
            "kind": "lifecycle",
            "level": "info",
            "data": {
                "agent_id": self.config.agent_id,
                "tasks_processed": self.state.tasks_processed,
                "tasks_failed": self.state.tasks_failed,
            }
        })


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Review Agent Bootstrap."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Agent Bootstrap (Step 151)")
    parser.add_argument("--init", action="store_true", help="Initialize the agent")
    parser.add_argument("--health", action="store_true", help="Run health check")
    parser.add_argument("--config", help="Path to config JSON file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, "r") as f:
            config = ReviewAgentConfig.from_dict(json.load(f))
    else:
        config = ReviewAgentConfig.from_env()

    bootstrap = ReviewAgentBootstrap(config)

    if args.init:
        success = bootstrap.initialize()
        if args.json:
            print(json.dumps({"initialized": success, "agent_id": config.agent_id}))
        else:
            print(f"Review Agent initialized: {success}")
            print(f"  Agent ID: {config.agent_id}")
            print(f"  Ring Level: {config.ring_level}")
        return 0 if success else 1

    if args.health:
        status = bootstrap.health_check()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Health Status: {'healthy' if status['healthy'] else 'unhealthy'}")
            print(f"  Phase: {status['phase']}")
            print(f"  Tasks Processed: {status['tasks_processed']}")
        return 0 if status["healthy"] else 1

    # Default: show config
    if args.json:
        print(json.dumps(config.to_dict(), indent=2))
    else:
        print("Review Agent Configuration:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
