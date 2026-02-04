#!/usr/bin/env python3
"""
canary.py - Canary Deployment Manager (Step 207)

PBTSO Phase: ITERATE
A2A Integration: Manages canary via deploy.canary.progress

Provides:
- CanaryConfig: Configuration for canary deployment
- CanaryState: State of canary deployment
- CanaryDeploymentManager: Manages canary deployments

Bus Topics:
- deploy.canary.start
- deploy.canary.progress
- deploy.canary.complete
- deploy.canary.rollback

Protocol: DKIN v30
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ==============================================================================
# Bus Emission Helper
# ==============================================================================

def _get_bus_path() -> Path:
    """Get the bus event file path."""
    pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    kind: str = "event",
    level: str = "info",
    actor: str = "canary-manager"
) -> str:
    """Emit an event to the Pluribus bus."""
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
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class CanaryPhase(Enum):
    """Canary deployment phases."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    ANALYZING = "analyzing"
    PROMOTING = "promoting"
    COMPLETE = "complete"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class CanaryConfig:
    """
    Configuration for canary deployment.

    Attributes:
        steps: Traffic percentage steps (e.g., [5, 10, 25, 50, 100])
        step_duration_s: Duration of each step in seconds
        error_threshold_pct: Error rate threshold for rollback
        latency_threshold_ms: Latency threshold for rollback
        min_sample_size: Minimum requests before analysis
        auto_promote: Automatically promote if metrics are good
        auto_rollback: Automatically rollback if metrics are bad
    """
    steps: List[int] = field(default_factory=lambda: [5, 10, 25, 50, 100])
    step_duration_s: int = 300  # 5 minutes per step
    error_threshold_pct: float = 5.0
    latency_threshold_ms: float = 500.0
    min_sample_size: int = 100
    auto_promote: bool = True
    auto_rollback: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanaryConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CanaryMetrics:
    """
    Metrics for canary analysis.

    Attributes:
        requests_total: Total number of requests
        errors_total: Total number of errors
        error_rate_pct: Error rate percentage
        latency_p50_ms: P50 latency in milliseconds
        latency_p95_ms: P95 latency in milliseconds
        latency_p99_ms: P99 latency in milliseconds
    """
    requests_total: int = 0
    errors_total: int = 0
    error_rate_pct: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanaryState:
    """
    State of a canary deployment.

    Attributes:
        canary_id: Unique canary identifier
        service_name: Service being deployed
        artifact_id: New artifact being canaried
        version: New version being canaried
        baseline_version: Current production version
        config: Canary configuration
        phase: Current phase
        current_step: Current step index
        traffic_pct: Current traffic percentage to canary
        metrics: Current metrics
        created_at: Timestamp when created
        started_at: Timestamp when started
        completed_at: Timestamp when completed
        error: Error message if failed
    """
    canary_id: str
    service_name: str
    artifact_id: str
    version: str
    baseline_version: str = ""
    config: CanaryConfig = field(default_factory=CanaryConfig)
    phase: CanaryPhase = CanaryPhase.PENDING
    current_step: int = 0
    traffic_pct: int = 0
    metrics: CanaryMetrics = field(default_factory=CanaryMetrics)
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canary_id": self.canary_id,
            "service_name": self.service_name,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "baseline_version": self.baseline_version,
            "config": self.config.to_dict(),
            "phase": self.phase.value,
            "current_step": self.current_step,
            "traffic_pct": self.traffic_pct,
            "metrics": self.metrics.to_dict(),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# ==============================================================================
# Canary Deployment Manager (Step 207)
# ==============================================================================

class CanaryDeploymentManager:
    """
    Canary Deployment Manager - manages canary deployment strategy.

    PBTSO Phase: ITERATE

    The canary strategy gradually shifts traffic to the new version:
    1. Deploy new version alongside current
    2. Start with small traffic percentage
    3. Monitor metrics and compare to baseline
    4. Gradually increase traffic if healthy
    5. Promote to 100% or rollback if issues

    Example:
        >>> manager = CanaryDeploymentManager()
        >>> config = CanaryConfig(steps=[5, 25, 50, 100])
        >>> state = await manager.start_canary(
        ...     service_name="myservice",
        ...     artifact_id="artifact-123",
        ...     version="v2.0",
        ...     config=config
        ... )
        >>> while not manager.is_complete(state):
        ...     await manager.advance(state)
    """

    BUS_TOPICS = {
        "start": "deploy.canary.start",
        "progress": "deploy.canary.progress",
        "complete": "deploy.canary.complete",
        "rollback": "deploy.canary.rollback",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "canary-manager",
    ):
        """
        Initialize the canary deployment manager.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "canary"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self._canaries: Dict[str, CanaryState] = {}
        self._load_canaries()

    async def start_canary(
        self,
        service_name: str,
        artifact_id: str,
        version: str,
        baseline_version: str = "",
        config: Optional[CanaryConfig] = None,
    ) -> CanaryState:
        """
        Start a canary deployment.

        Args:
            service_name: Service being deployed
            artifact_id: Artifact ID to deploy
            version: Version string
            baseline_version: Current production version
            config: Canary configuration

        Returns:
            CanaryState for the deployment
        """
        canary_id = f"canary-{uuid.uuid4().hex[:12]}"

        state = CanaryState(
            canary_id=canary_id,
            service_name=service_name,
            artifact_id=artifact_id,
            version=version,
            baseline_version=baseline_version,
            config=config or CanaryConfig(),
        )

        self._canaries[canary_id] = state

        # Emit start event
        _emit_bus_event(
            self.BUS_TOPICS["start"],
            {
                "canary_id": canary_id,
                "service_name": service_name,
                "artifact_id": artifact_id,
                "version": version,
                "baseline_version": baseline_version,
                "steps": state.config.steps,
            },
            actor=self.actor_id,
        )

        # Start the canary
        state.phase = CanaryPhase.DEPLOYING
        state.started_at = time.time()

        # Deploy canary instances (simulated)
        await asyncio.sleep(0.1)

        # Move to first traffic step
        if state.config.steps:
            state.traffic_pct = state.config.steps[0]
            state.current_step = 0
            state.phase = CanaryPhase.ANALYZING

        self._save_canary(state)

        self._emit_progress(state, "Canary deployed, starting analysis")

        return state

    async def advance(self, state: CanaryState) -> bool:
        """
        Advance canary to next step.

        Args:
            state: CanaryState

        Returns:
            True if advanced, False if cannot advance
        """
        if state.phase not in (CanaryPhase.ANALYZING, CanaryPhase.PROMOTING):
            return False

        # Analyze metrics
        analysis_result = await self._analyze_metrics(state)

        if not analysis_result["healthy"]:
            if state.config.auto_rollback:
                await self.rollback(state, analysis_result["reason"])
                return False
            else:
                self._emit_progress(state, f"Warning: {analysis_result['reason']}")
                return False

        # Advance to next step
        next_step = state.current_step + 1

        if next_step >= len(state.config.steps):
            # Promote to 100%
            state.phase = CanaryPhase.PROMOTING
            state.traffic_pct = 100
            await self._promote(state)
            return True

        # Move to next percentage
        state.current_step = next_step
        state.traffic_pct = state.config.steps[next_step]

        # Simulate traffic shift
        await asyncio.sleep(0.05)

        self._save_canary(state)
        self._emit_progress(state, f"Advanced to {state.traffic_pct}% traffic")

        return True

    async def rollback(self, state: CanaryState, reason: str = "") -> bool:
        """
        Rollback canary deployment.

        Args:
            state: CanaryState
            reason: Rollback reason

        Returns:
            True if rolled back
        """
        state.phase = CanaryPhase.ROLLED_BACK
        state.completed_at = time.time()
        state.error = reason or "Manual rollback"
        state.traffic_pct = 0

        self._save_canary(state)

        _emit_bus_event(
            self.BUS_TOPICS["rollback"],
            {
                "canary_id": state.canary_id,
                "service_name": state.service_name,
                "version": state.version,
                "reason": state.error,
                "traffic_pct_at_rollback": state.traffic_pct,
            },
            level="warn",
            actor=self.actor_id,
        )

        return True

    async def _promote(self, state: CanaryState) -> None:
        """Promote canary to full production."""
        state.phase = CanaryPhase.COMPLETE
        state.completed_at = time.time()
        state.traffic_pct = 100

        self._save_canary(state)

        _emit_bus_event(
            self.BUS_TOPICS["complete"],
            {
                "canary_id": state.canary_id,
                "service_name": state.service_name,
                "version": state.version,
                "duration_s": state.completed_at - state.started_at,
                "steps_completed": state.current_step + 1,
            },
            actor=self.actor_id,
        )

    async def _analyze_metrics(self, state: CanaryState) -> Dict[str, Any]:
        """
        Analyze canary metrics.

        Returns:
            Analysis result with healthy flag and reason
        """
        # Simulate metrics collection
        state.metrics.requests_total += 100
        state.metrics.errors_total += 2  # Simulated 2% error rate
        state.metrics.error_rate_pct = (state.metrics.errors_total / state.metrics.requests_total) * 100
        state.metrics.latency_p50_ms = 50.0
        state.metrics.latency_p95_ms = 150.0
        state.metrics.latency_p99_ms = 300.0

        # Check thresholds
        if state.metrics.error_rate_pct > state.config.error_threshold_pct:
            return {
                "healthy": False,
                "reason": f"Error rate {state.metrics.error_rate_pct:.2f}% exceeds threshold {state.config.error_threshold_pct}%",
            }

        if state.metrics.latency_p95_ms > state.config.latency_threshold_ms:
            return {
                "healthy": False,
                "reason": f"P95 latency {state.metrics.latency_p95_ms:.2f}ms exceeds threshold {state.config.latency_threshold_ms}ms",
            }

        return {"healthy": True, "reason": ""}

    def _emit_progress(self, state: CanaryState, message: str) -> None:
        """Emit progress event."""
        _emit_bus_event(
            self.BUS_TOPICS["progress"],
            {
                "canary_id": state.canary_id,
                "service_name": state.service_name,
                "version": state.version,
                "phase": state.phase.value,
                "step": state.current_step,
                "traffic_pct": state.traffic_pct,
                "message": message,
                "metrics": state.metrics.to_dict(),
            },
            kind="metric",
            actor=self.actor_id,
        )

    def is_complete(self, state: CanaryState) -> bool:
        """Check if canary is complete."""
        return state.phase in (CanaryPhase.COMPLETE, CanaryPhase.ROLLED_BACK, CanaryPhase.FAILED)

    def get_canary(self, canary_id: str) -> Optional[CanaryState]:
        """Get a canary by ID."""
        return self._canaries.get(canary_id)

    def list_canaries(self, active_only: bool = False) -> List[CanaryState]:
        """List all canaries."""
        canaries = list(self._canaries.values())
        if active_only:
            canaries = [c for c in canaries if not self.is_complete(c)]
        return canaries

    def _save_canary(self, state: CanaryState) -> None:
        """Save canary state to disk."""
        state_file = self.state_dir / f"{state.canary_id}.json"
        with open(state_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def _load_canaries(self) -> None:
        """Load canaries from disk."""
        for state_file in self.state_dir.glob("*.json"):
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)

                    config = CanaryConfig.from_dict(data.get("config", {}))
                    metrics = CanaryMetrics(**data.get("metrics", {}))

                    state = CanaryState(
                        canary_id=data["canary_id"],
                        service_name=data["service_name"],
                        artifact_id=data["artifact_id"],
                        version=data["version"],
                        baseline_version=data.get("baseline_version", ""),
                        config=config,
                        phase=CanaryPhase(data["phase"]),
                        current_step=data.get("current_step", 0),
                        traffic_pct=data.get("traffic_pct", 0),
                        metrics=metrics,
                        created_at=data.get("created_at", time.time()),
                        started_at=data.get("started_at", 0.0),
                        completed_at=data.get("completed_at", 0.0),
                        error=data.get("error"),
                    )

                    self._canaries[state.canary_id] = state
            except (json.JSONDecodeError, KeyError, IOError):
                continue


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for canary deployment manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Canary Deployment Manager (Step 207)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start command
    start_parser = subparsers.add_parser("start", help="Start a canary deployment")
    start_parser.add_argument("service_name", help="Service name")
    start_parser.add_argument("--artifact", "-a", required=True, help="Artifact ID")
    start_parser.add_argument("--version", "-v", required=True, help="Version")
    start_parser.add_argument("--baseline", "-b", default="", help="Baseline version")
    start_parser.add_argument("--steps", default="5,25,50,100", help="Traffic steps (comma-separated)")
    start_parser.add_argument("--json", action="store_true", help="JSON output")

    # advance command
    advance_parser = subparsers.add_parser("advance", help="Advance canary to next step")
    advance_parser.add_argument("canary_id", help="Canary ID")

    # rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback canary")
    rollback_parser.add_argument("canary_id", help="Canary ID")
    rollback_parser.add_argument("--reason", "-r", default="", help="Rollback reason")

    # status command
    status_parser = subparsers.add_parser("status", help="Get canary status")
    status_parser.add_argument("canary_id", help="Canary ID")
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List canaries")
    list_parser.add_argument("--active", action="store_true", help="Active only")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    manager = CanaryDeploymentManager()

    if args.command == "start":
        steps = [int(s.strip()) for s in args.steps.split(",")]
        config = CanaryConfig(steps=steps)

        state = asyncio.get_event_loop().run_until_complete(
            manager.start_canary(
                service_name=args.service_name,
                artifact_id=args.artifact,
                version=args.version,
                baseline_version=args.baseline,
                config=config,
            )
        )

        if args.json:
            print(json.dumps(state.to_dict(), indent=2))
        else:
            print(f"Started canary: {state.canary_id}")
            print(f"  Service: {state.service_name}")
            print(f"  Version: {state.version}")
            print(f"  Traffic: {state.traffic_pct}%")
            print(f"  Steps: {state.config.steps}")

        return 0

    elif args.command == "advance":
        state = manager.get_canary(args.canary_id)
        if not state:
            print(f"Canary not found: {args.canary_id}")
            return 1

        success = asyncio.get_event_loop().run_until_complete(
            manager.advance(state)
        )

        if success:
            print(f"Advanced to {state.traffic_pct}% traffic (step {state.current_step})")
        else:
            print(f"Cannot advance: {state.phase.value}")

        return 0 if success else 1

    elif args.command == "rollback":
        state = manager.get_canary(args.canary_id)
        if not state:
            print(f"Canary not found: {args.canary_id}")
            return 1

        asyncio.get_event_loop().run_until_complete(
            manager.rollback(state, args.reason)
        )

        print(f"Rolled back canary: {args.canary_id}")
        return 0

    elif args.command == "status":
        state = manager.get_canary(args.canary_id)
        if not state:
            print(f"Canary not found: {args.canary_id}")
            return 1

        if args.json:
            print(json.dumps(state.to_dict(), indent=2))
        else:
            print(f"Canary: {state.canary_id}")
            print(f"  Service: {state.service_name}")
            print(f"  Version: {state.version}")
            print(f"  Phase: {state.phase.value}")
            print(f"  Traffic: {state.traffic_pct}%")
            print(f"  Step: {state.current_step}/{len(state.config.steps)}")
            print(f"  Error Rate: {state.metrics.error_rate_pct:.2f}%")

        return 0

    elif args.command == "list":
        canaries = manager.list_canaries(active_only=args.active)

        if args.json:
            print(json.dumps([c.to_dict() for c in canaries], indent=2))
        else:
            if not canaries:
                print("No canaries found")
            else:
                for c in canaries:
                    print(f"{c.canary_id} ({c.service_name}) - {c.phase.value} @ {c.traffic_pct}%")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
