#!/usr/bin/env python3
"""
bootstrap.py - Deploy Agent Bootstrap Module (Step 201)

PBTSO Phase: SKILL, SEQUESTER
A2A Integration: Emits a2a.deploy.bootstrap.start, subscribes to a2a.task.dispatch

Provides:
- DeployAgentConfig: Configuration for deploy agent initialization
- DeployAgentBootstrap: Bootstrap class for deploy agent lifecycle

Bus Topics:
- a2a.deploy.bootstrap.start
- a2a.deploy.bootstrap.complete
- a2a.deploy.bootstrap.failed

Protocol: DKIN v30, CITIZEN v2
"""
from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable


# ==============================================================================
# Constants
# ==============================================================================

DEPLOY_AGENT_VERSION = "0.1.0"
DEFAULT_RING_LEVEL = 1  # Higher privilege for deployment operations


# ==============================================================================
# Bus Emission Helper
# ==============================================================================

def _get_bus_path() -> Path:
    """Get the bus event file path from environment or default."""
    pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    kind: str = "event",
    level: str = "info",
    actor: str = "deploy-agent"
) -> str:
    """
    Emit an event to the Pluribus bus.

    Args:
        topic: Event topic (e.g., 'a2a.deploy.bootstrap.start')
        data: Event payload
        kind: Event kind (event, metric, command)
        level: Log level (debug, info, warn, error)
        actor: Actor identifier

    Returns:
        Event ID
    """
    bus_path = _get_bus_path()
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat() + "Z",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }

    try:
        with open(bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")
    except IOError:
        pass  # Best effort emission

    return event_id


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class DeployAgentConfig:
    """
    Configuration for the Deploy Agent.

    Attributes:
        agent_id: Unique identifier for this deploy agent instance
        ring_level: Ring level for privilege (0=highest, 1=deploy, 2=standard)
        environments: List of deployment environments (dev, staging, prod)
        rollback_enabled: Whether automatic rollback is enabled
        blue_green_enabled: Whether blue-green deployments are enabled
        canary_enabled: Whether canary deployments are enabled
        max_concurrent_deploys: Maximum concurrent deployments
        deploy_timeout_s: Timeout for deploy operations in seconds
        health_check_interval_s: Interval for health checks
        artifact_retention_days: Days to retain artifacts
        registry_url: Container registry URL
        state_dir: Directory for state persistence
    """
    agent_id: str = "deploy-agent"
    ring_level: int = DEFAULT_RING_LEVEL
    environments: List[str] = field(default_factory=lambda: ["dev", "staging", "prod"])
    rollback_enabled: bool = True
    blue_green_enabled: bool = True
    canary_enabled: bool = True
    max_concurrent_deploys: int = 3
    deploy_timeout_s: int = 1800  # 30 minutes
    health_check_interval_s: int = 30
    artifact_retention_days: int = 30
    registry_url: str = ""
    state_dir: str = ""

    def __post_init__(self):
        if not self.state_dir:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.state_dir = str(Path(pluribus_root) / ".pluribus" / "deploy")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeployAgentConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "DeployAgentConfig":
        """Create from environment variables."""
        return cls(
            agent_id=os.environ.get("DEPLOY_AGENT_ID", "deploy-agent"),
            ring_level=int(os.environ.get("DEPLOY_RING_LEVEL", str(DEFAULT_RING_LEVEL))),
            environments=os.environ.get("DEPLOY_ENVIRONMENTS", "dev,staging,prod").split(","),
            rollback_enabled=os.environ.get("DEPLOY_ROLLBACK_ENABLED", "true").lower() == "true",
            blue_green_enabled=os.environ.get("DEPLOY_BLUE_GREEN_ENABLED", "true").lower() == "true",
            canary_enabled=os.environ.get("DEPLOY_CANARY_ENABLED", "true").lower() == "true",
            max_concurrent_deploys=int(os.environ.get("DEPLOY_MAX_CONCURRENT", "3")),
            deploy_timeout_s=int(os.environ.get("DEPLOY_TIMEOUT_S", "1800")),
            health_check_interval_s=int(os.environ.get("DEPLOY_HEALTH_CHECK_INTERVAL_S", "30")),
            artifact_retention_days=int(os.environ.get("DEPLOY_ARTIFACT_RETENTION_DAYS", "30")),
            registry_url=os.environ.get("DEPLOY_REGISTRY_URL", ""),
            state_dir=os.environ.get("DEPLOY_STATE_DIR", ""),
        )


# ==============================================================================
# Bootstrap State
# ==============================================================================

@dataclass
class BootstrapState:
    """State of the deploy agent bootstrap process."""
    agent_id: str
    started_at: float
    completed_at: Optional[float] = None
    status: str = "initializing"  # initializing, ready, failed, shutdown
    error: Optional[str] = None
    config_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Deploy Agent Bootstrap (Step 201)
# ==============================================================================

class DeployAgentBootstrap:
    """
    Deploy Agent Bootstrap - initializes and manages deploy agent lifecycle.

    PBTSO Phase: SKILL, SEQUESTER

    Responsibilities:
    - Initialize deploy agent with configuration
    - Set up state directories
    - Register with A2A bus
    - Manage agent lifecycle (start, stop, health)

    Example:
        >>> config = DeployAgentConfig(agent_id="deploy-001")
        >>> bootstrap = DeployAgentBootstrap(config)
        >>> bootstrap.start()
        >>> # ... agent is running ...
        >>> bootstrap.stop()
    """

    BUS_TOPICS = {
        "start": "a2a.deploy.bootstrap.start",
        "complete": "a2a.deploy.bootstrap.complete",
        "failed": "a2a.deploy.bootstrap.failed",
        "health": "a2a.deploy.health.check",
        "dispatch": "a2a.task.dispatch",
    }

    def __init__(self, config: Optional[DeployAgentConfig] = None):
        """
        Initialize the deploy agent bootstrap.

        Args:
            config: Deploy agent configuration (uses defaults if not provided)
        """
        self.config = config or DeployAgentConfig()
        self.state = BootstrapState(
            agent_id=self.config.agent_id,
            started_at=0.0,
        )
        self._handlers: Dict[str, List[Callable]] = {}
        self._running = False

    def start(self) -> bool:
        """
        Start the deploy agent.

        Emits: a2a.deploy.bootstrap.start, a2a.deploy.bootstrap.complete

        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            return True

        self.state.started_at = time.time()
        self.state.status = "initializing"

        # Emit start event
        _emit_bus_event(
            self.BUS_TOPICS["start"],
            {
                "agent_id": self.config.agent_id,
                "ring_level": self.config.ring_level,
                "environments": self.config.environments,
                "config": self.config.to_dict(),
                "version": DEPLOY_AGENT_VERSION,
            },
            actor=self.config.agent_id,
        )

        try:
            # Initialize state directory
            self._init_state_dir()

            # Load persisted state if exists
            self._load_state()

            # Compute config hash for change detection
            self.state.config_hash = self._compute_config_hash()

            self.state.status = "ready"
            self.state.completed_at = time.time()
            self._running = True

            # Emit complete event
            _emit_bus_event(
                self.BUS_TOPICS["complete"],
                {
                    "agent_id": self.config.agent_id,
                    "status": "ready",
                    "bootstrap_time_ms": (self.state.completed_at - self.state.started_at) * 1000,
                    "config_hash": self.state.config_hash,
                },
                actor=self.config.agent_id,
            )

            # Save state
            self._save_state()

            return True

        except Exception as e:
            self.state.status = "failed"
            self.state.error = str(e)

            # Emit failed event
            _emit_bus_event(
                self.BUS_TOPICS["failed"],
                {
                    "agent_id": self.config.agent_id,
                    "error": str(e),
                },
                level="error",
                actor=self.config.agent_id,
            )

            return False

    def stop(self) -> bool:
        """
        Stop the deploy agent gracefully.

        Returns:
            True if stopped successfully
        """
        if not self._running:
            return True

        self.state.status = "shutdown"
        self._running = False

        # Save final state
        self._save_state()

        return True

    def is_running(self) -> bool:
        """Check if the deploy agent is running."""
        return self._running

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the deploy agent.

        Emits: a2a.deploy.health.check

        Returns:
            Health status dictionary
        """
        health = {
            "agent_id": self.config.agent_id,
            "status": self.state.status,
            "running": self._running,
            "uptime_s": time.time() - self.state.started_at if self.state.started_at else 0,
            "config_hash": self.state.config_hash,
            "ts": time.time(),
        }

        _emit_bus_event(
            self.BUS_TOPICS["health"],
            health,
            kind="metric",
            actor=self.config.agent_id,
        )

        return health

    def register_handler(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a handler for A2A bus events.

        Args:
            topic: Event topic to subscribe to
            handler: Callback function for events
        """
        if topic not in self._handlers:
            self._handlers[topic] = []
        self._handlers[topic].append(handler)

    def get_handlers(self, topic: str) -> List[Callable]:
        """Get handlers for a topic."""
        return self._handlers.get(topic, [])

    def _init_state_dir(self) -> None:
        """Initialize the state directory."""
        state_path = Path(self.config.state_dir)
        state_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (state_path / "deployments").mkdir(exist_ok=True)
        (state_path / "artifacts").mkdir(exist_ok=True)
        (state_path / "rollbacks").mkdir(exist_ok=True)

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        state_file = Path(self.config.state_dir) / "bootstrap_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)
                    # Only load non-runtime fields
                    self.state.config_hash = data.get("config_hash", "")
            except (json.JSONDecodeError, IOError):
                pass

    def _save_state(self) -> None:
        """Save state to disk."""
        state_file = Path(self.config.state_dir) / "bootstrap_state.json"
        try:
            with open(state_file, "w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except IOError:
            pass

    def _compute_config_hash(self) -> str:
        """Compute a hash of the configuration for change detection."""
        import hashlib
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for deploy agent bootstrap."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Agent Bootstrap (Step 201)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start command
    start_parser = subparsers.add_parser("start", help="Start the deploy agent")
    start_parser.add_argument("--agent-id", default="deploy-agent", help="Agent ID")
    start_parser.add_argument("--environments", default="dev,staging,prod", help="Comma-separated environments")
    start_parser.add_argument("--json", action="store_true", help="JSON output")

    # status command
    status_parser = subparsers.add_parser("status", help="Show agent status")
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    # health command
    health_parser = subparsers.add_parser("health", help="Health check")
    health_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    if args.command == "start":
        config = DeployAgentConfig(
            agent_id=args.agent_id,
            environments=args.environments.split(","),
        )
        bootstrap = DeployAgentBootstrap(config)
        success = bootstrap.start()

        if args.json:
            print(json.dumps({
                "success": success,
                "agent_id": config.agent_id,
                "status": bootstrap.state.status,
            }))
        else:
            if success:
                print(f"Deploy Agent started: {config.agent_id}")
                print(f"  Status: {bootstrap.state.status}")
                print(f"  Environments: {', '.join(config.environments)}")
            else:
                print(f"Failed to start Deploy Agent: {bootstrap.state.error}")

        return 0 if success else 1

    elif args.command == "status":
        config = DeployAgentConfig.from_env()
        bootstrap = DeployAgentBootstrap(config)
        bootstrap._load_state()

        if args.json:
            print(json.dumps(bootstrap.state.to_dict()))
        else:
            print(f"Deploy Agent: {config.agent_id}")
            print(f"  Status: {bootstrap.state.status}")
            print(f"  Config Hash: {bootstrap.state.config_hash}")

        return 0

    elif args.command == "health":
        config = DeployAgentConfig.from_env()
        bootstrap = DeployAgentBootstrap(config)
        health = bootstrap.health_check()

        if args.json:
            print(json.dumps(health))
        else:
            print(f"Deploy Agent Health: {config.agent_id}")
            print(f"  Status: {health['status']}")
            print(f"  Running: {health['running']}")
            print(f"  Uptime: {health['uptime_s']:.1f}s")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
