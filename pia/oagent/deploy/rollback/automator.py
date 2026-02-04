#!/usr/bin/env python3
"""
automator.py - Rollback Automator (Step 208)

PBTSO Phase: ITERATE
A2A Integration: Automates rollback via deploy.rollback.trigger

Provides:
- RollbackTrigger: Trigger types for rollback
- RollbackConfig: Configuration for rollback automation
- RollbackRecord: Record of a rollback event
- RollbackAutomator: Automates rollback decisions and execution

Bus Topics:
- deploy.rollback.trigger
- deploy.rollback.complete
- deploy.rollback.failed
- deploy.rollback.analysis

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
from typing import Any, Callable, Dict, List, Optional


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
    actor: str = "rollback-automator"
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

class RollbackTrigger(Enum):
    """Trigger types for rollback."""
    MANUAL = "manual"
    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    HEALTH_CHECK = "health_check"
    ALERT = "alert"
    CUSTOM = "custom"


class RollbackStatus(Enum):
    """Status of a rollback operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class RollbackConfig:
    """
    Configuration for rollback automation.

    Attributes:
        enabled: Whether auto-rollback is enabled
        error_rate_threshold_pct: Error rate threshold
        latency_threshold_ms: P95 latency threshold
        health_check_failures: Consecutive health check failures
        cooldown_s: Minimum time between rollbacks
        max_rollbacks_per_hour: Maximum rollbacks per hour
        require_approval: Require manual approval
        notify_channels: Notification channels
    """
    enabled: bool = True
    error_rate_threshold_pct: float = 5.0
    latency_threshold_ms: float = 500.0
    health_check_failures: int = 3
    cooldown_s: int = 300  # 5 minutes
    max_rollbacks_per_hour: int = 3
    require_approval: bool = False
    notify_channels: List[str] = field(default_factory=lambda: ["bus"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RollbackConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class RollbackRecord:
    """
    Record of a rollback event.

    Attributes:
        rollback_id: Unique rollback identifier
        service_name: Service being rolled back
        from_version: Version being rolled back from
        to_version: Version being rolled back to
        trigger: Trigger type
        trigger_details: Details about the trigger
        status: Rollback status
        started_at: Timestamp when started
        completed_at: Timestamp when completed
        error: Error message if failed
        metrics_snapshot: Metrics at time of rollback
    """
    rollback_id: str
    service_name: str
    from_version: str
    to_version: str
    trigger: RollbackTrigger = RollbackTrigger.MANUAL
    trigger_details: Dict[str, Any] = field(default_factory=dict)
    status: RollbackStatus = RollbackStatus.PENDING
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    error: Optional[str] = None
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollback_id": self.rollback_id,
            "service_name": self.service_name,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "trigger": self.trigger.value,
            "trigger_details": self.trigger_details,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "metrics_snapshot": self.metrics_snapshot,
        }


# ==============================================================================
# Rollback Automator (Step 208)
# ==============================================================================

class RollbackAutomator:
    """
    Rollback Automator - automates rollback decisions and execution.

    PBTSO Phase: ITERATE

    Responsibilities:
    - Monitor deployment health metrics
    - Trigger automatic rollback on threshold breach
    - Execute rollback operations
    - Track rollback history
    - Emit rollback events to A2A bus

    Example:
        >>> automator = RollbackAutomator()
        >>> config = RollbackConfig(error_rate_threshold_pct=3.0)
        >>> automator.configure("myservice", config)
        >>> # Later, when metrics breach threshold:
        >>> record = await automator.trigger_rollback(
        ...     service_name="myservice",
        ...     from_version="v2.0",
        ...     to_version="v1.9",
        ...     trigger=RollbackTrigger.ERROR_RATE,
        ...     trigger_details={"error_rate": 5.2}
        ... )
    """

    BUS_TOPICS = {
        "trigger": "deploy.rollback.trigger",
        "complete": "deploy.rollback.complete",
        "failed": "deploy.rollback.failed",
        "analysis": "deploy.rollback.analysis",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "rollback-automator",
    ):
        """
        Initialize the rollback automator.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "rollbacks"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        self._configs: Dict[str, RollbackConfig] = {}
        self._rollbacks: Dict[str, RollbackRecord] = {}
        self._last_rollback_ts: Dict[str, float] = {}
        self._rollback_count: Dict[str, List[float]] = {}  # timestamps
        self._rollback_handlers: List[Callable[[RollbackRecord], Any]] = []

        self._load_rollbacks()

    def configure(self, service_name: str, config: RollbackConfig) -> None:
        """
        Configure rollback automation for a service.

        Args:
            service_name: Service name
            config: Rollback configuration
        """
        self._configs[service_name] = config

    def get_config(self, service_name: str) -> RollbackConfig:
        """Get configuration for a service."""
        return self._configs.get(service_name, RollbackConfig())

    async def trigger_rollback(
        self,
        service_name: str,
        from_version: str,
        to_version: str,
        trigger: RollbackTrigger = RollbackTrigger.MANUAL,
        trigger_details: Optional[Dict[str, Any]] = None,
        rollback_executor: Optional[Callable[[str, str], bool]] = None,
    ) -> RollbackRecord:
        """
        Trigger a rollback operation.

        Args:
            service_name: Service to rollback
            from_version: Current version
            to_version: Target version
            trigger: Trigger type
            trigger_details: Details about the trigger
            rollback_executor: Optional callback to execute rollback

        Returns:
            RollbackRecord with result
        """
        rollback_id = f"rollback-{uuid.uuid4().hex[:12]}"
        config = self.get_config(service_name)

        record = RollbackRecord(
            rollback_id=rollback_id,
            service_name=service_name,
            from_version=from_version,
            to_version=to_version,
            trigger=trigger,
            trigger_details=trigger_details or {},
        )

        self._rollbacks[rollback_id] = record

        # Check cooldown
        last_rollback = self._last_rollback_ts.get(service_name, 0)
        if time.time() - last_rollback < config.cooldown_s:
            record.status = RollbackStatus.FAILED
            record.error = f"Cooldown period active (wait {config.cooldown_s - int(time.time() - last_rollback)}s)"
            self._save_rollback(record)
            return record

        # Check max rollbacks per hour
        hour_ago = time.time() - 3600
        recent_rollbacks = [ts for ts in self._rollback_count.get(service_name, []) if ts > hour_ago]
        if len(recent_rollbacks) >= config.max_rollbacks_per_hour:
            record.status = RollbackStatus.FAILED
            record.error = f"Max rollbacks per hour exceeded ({config.max_rollbacks_per_hour})"
            self._save_rollback(record)
            return record

        # Emit trigger event
        _emit_bus_event(
            self.BUS_TOPICS["trigger"],
            {
                "rollback_id": rollback_id,
                "service_name": service_name,
                "from_version": from_version,
                "to_version": to_version,
                "trigger": trigger.value,
                "trigger_details": trigger_details or {},
            },
            actor=self.actor_id,
        )

        try:
            record.status = RollbackStatus.IN_PROGRESS

            # Execute rollback
            if rollback_executor:
                success = rollback_executor(from_version, to_version)
            else:
                # Simulate rollback
                await asyncio.sleep(0.1)
                success = True

            if success:
                record.status = RollbackStatus.COMPLETE
                record.completed_at = time.time()

                # Update tracking
                self._last_rollback_ts[service_name] = time.time()
                if service_name not in self._rollback_count:
                    self._rollback_count[service_name] = []
                self._rollback_count[service_name].append(time.time())

                # Emit complete event
                _emit_bus_event(
                    self.BUS_TOPICS["complete"],
                    {
                        "rollback_id": rollback_id,
                        "service_name": service_name,
                        "from_version": from_version,
                        "to_version": to_version,
                        "duration_ms": (record.completed_at - record.started_at) * 1000,
                    },
                    actor=self.actor_id,
                )

                # Call handlers
                for handler in self._rollback_handlers:
                    try:
                        handler(record)
                    except Exception:
                        pass

            else:
                record.status = RollbackStatus.FAILED
                record.error = "Rollback execution failed"

        except Exception as e:
            record.status = RollbackStatus.FAILED
            record.error = str(e)
            record.completed_at = time.time()

            _emit_bus_event(
                self.BUS_TOPICS["failed"],
                {
                    "rollback_id": rollback_id,
                    "service_name": service_name,
                    "error": str(e),
                },
                level="error",
                actor=self.actor_id,
            )

        self._save_rollback(record)
        return record

    async def analyze_metrics(
        self,
        service_name: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze metrics and determine if rollback is needed.

        Args:
            service_name: Service name
            metrics: Current metrics

        Returns:
            Analysis result with should_rollback flag
        """
        config = self.get_config(service_name)

        if not config.enabled:
            return {"should_rollback": False, "reason": "Auto-rollback disabled"}

        analysis = {
            "should_rollback": False,
            "reason": "",
            "trigger": None,
            "details": {},
        }

        # Check error rate
        error_rate = metrics.get("error_rate_pct", 0.0)
        if error_rate > config.error_rate_threshold_pct:
            analysis["should_rollback"] = True
            analysis["reason"] = f"Error rate {error_rate:.2f}% exceeds threshold {config.error_rate_threshold_pct}%"
            analysis["trigger"] = RollbackTrigger.ERROR_RATE
            analysis["details"]["error_rate"] = error_rate

        # Check latency
        latency_p95 = metrics.get("latency_p95_ms", 0.0)
        if latency_p95 > config.latency_threshold_ms:
            if not analysis["should_rollback"]:
                analysis["should_rollback"] = True
                analysis["reason"] = f"P95 latency {latency_p95:.2f}ms exceeds threshold {config.latency_threshold_ms}ms"
                analysis["trigger"] = RollbackTrigger.LATENCY
            analysis["details"]["latency_p95"] = latency_p95

        # Check health check failures
        health_failures = metrics.get("health_check_failures", 0)
        if health_failures >= config.health_check_failures:
            if not analysis["should_rollback"]:
                analysis["should_rollback"] = True
                analysis["reason"] = f"Health check failures ({health_failures}) exceed threshold ({config.health_check_failures})"
                analysis["trigger"] = RollbackTrigger.HEALTH_CHECK
            analysis["details"]["health_failures"] = health_failures

        # Emit analysis event
        _emit_bus_event(
            self.BUS_TOPICS["analysis"],
            {
                "service_name": service_name,
                "should_rollback": analysis["should_rollback"],
                "reason": analysis["reason"],
                "metrics": metrics,
            },
            kind="metric",
            actor=self.actor_id,
        )

        return analysis

    def register_handler(self, handler: Callable[[RollbackRecord], Any]) -> None:
        """Register a callback for rollback events."""
        self._rollback_handlers.append(handler)

    def get_rollback(self, rollback_id: str) -> Optional[RollbackRecord]:
        """Get a rollback record by ID."""
        return self._rollbacks.get(rollback_id)

    def list_rollbacks(
        self,
        service_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[RollbackRecord]:
        """List rollback records."""
        rollbacks = list(self._rollbacks.values())

        if service_name:
            rollbacks = [r for r in rollbacks if r.service_name == service_name]

        # Sort by started_at descending
        rollbacks.sort(key=lambda r: r.started_at, reverse=True)

        return rollbacks[:limit]

    def get_stats(self, service_name: str) -> Dict[str, Any]:
        """Get rollback statistics for a service."""
        rollbacks = [r for r in self._rollbacks.values() if r.service_name == service_name]

        hour_ago = time.time() - 3600
        day_ago = time.time() - 86400

        return {
            "service_name": service_name,
            "total_rollbacks": len(rollbacks),
            "rollbacks_last_hour": len([r for r in rollbacks if r.started_at > hour_ago]),
            "rollbacks_last_day": len([r for r in rollbacks if r.started_at > day_ago]),
            "successful_rollbacks": len([r for r in rollbacks if r.status == RollbackStatus.COMPLETE]),
            "failed_rollbacks": len([r for r in rollbacks if r.status == RollbackStatus.FAILED]),
            "last_rollback_ts": self._last_rollback_ts.get(service_name, 0),
        }

    def _save_rollback(self, record: RollbackRecord) -> None:
        """Save rollback record to disk."""
        state_file = self.state_dir / f"{record.rollback_id}.json"
        with open(state_file, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

    def _load_rollbacks(self) -> None:
        """Load rollbacks from disk."""
        for state_file in self.state_dir.glob("*.json"):
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)

                    record = RollbackRecord(
                        rollback_id=data["rollback_id"],
                        service_name=data["service_name"],
                        from_version=data["from_version"],
                        to_version=data["to_version"],
                        trigger=RollbackTrigger(data["trigger"]),
                        trigger_details=data.get("trigger_details", {}),
                        status=RollbackStatus(data["status"]),
                        started_at=data.get("started_at", time.time()),
                        completed_at=data.get("completed_at", 0.0),
                        error=data.get("error"),
                        metrics_snapshot=data.get("metrics_snapshot", {}),
                    )

                    self._rollbacks[record.rollback_id] = record
            except (json.JSONDecodeError, KeyError, IOError):
                continue


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for rollback automator."""
    import argparse

    parser = argparse.ArgumentParser(description="Rollback Automator (Step 208)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # trigger command
    trigger_parser = subparsers.add_parser("trigger", help="Trigger a rollback")
    trigger_parser.add_argument("service_name", help="Service name")
    trigger_parser.add_argument("--from", dest="from_version", required=True, help="From version")
    trigger_parser.add_argument("--to", dest="to_version", required=True, help="To version")
    trigger_parser.add_argument("--trigger", "-t", default="manual", choices=["manual", "error_rate", "latency", "health_check", "alert"])
    trigger_parser.add_argument("--json", action="store_true", help="JSON output")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze metrics")
    analyze_parser.add_argument("service_name", help="Service name")
    analyze_parser.add_argument("--error-rate", type=float, default=0.0, help="Error rate %")
    analyze_parser.add_argument("--latency", type=float, default=0.0, help="P95 latency ms")
    analyze_parser.add_argument("--failures", type=int, default=0, help="Health check failures")
    analyze_parser.add_argument("--json", action="store_true", help="JSON output")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Get rollback stats")
    stats_parser.add_argument("service_name", help="Service name")
    stats_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List rollbacks")
    list_parser.add_argument("--service", "-s", help="Filter by service")
    list_parser.add_argument("--limit", "-l", type=int, default=10, help="Limit results")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    automator = RollbackAutomator()

    if args.command == "trigger":
        record = asyncio.get_event_loop().run_until_complete(
            automator.trigger_rollback(
                service_name=args.service_name,
                from_version=args.from_version,
                to_version=args.to_version,
                trigger=RollbackTrigger(args.trigger),
            )
        )

        if args.json:
            print(json.dumps(record.to_dict(), indent=2))
        else:
            status_icon = "OK" if record.status == RollbackStatus.COMPLETE else "FAIL"
            print(f"[{status_icon}] Rollback {record.rollback_id}")
            print(f"  Service: {record.service_name}")
            print(f"  From: {record.from_version} -> To: {record.to_version}")
            print(f"  Status: {record.status.value}")
            if record.error:
                print(f"  Error: {record.error}")

        return 0 if record.status == RollbackStatus.COMPLETE else 1

    elif args.command == "analyze":
        metrics = {
            "error_rate_pct": args.error_rate,
            "latency_p95_ms": args.latency,
            "health_check_failures": args.failures,
        }

        result = asyncio.get_event_loop().run_until_complete(
            automator.analyze_metrics(args.service_name, metrics)
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result["should_rollback"]:
                print(f"ROLLBACK RECOMMENDED: {result['reason']}")
            else:
                print("No rollback needed")

        return 0

    elif args.command == "stats":
        stats = automator.get_stats(args.service_name)

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Rollback Stats: {args.service_name}")
            print(f"  Total: {stats['total_rollbacks']}")
            print(f"  Last hour: {stats['rollbacks_last_hour']}")
            print(f"  Last day: {stats['rollbacks_last_day']}")
            print(f"  Successful: {stats['successful_rollbacks']}")
            print(f"  Failed: {stats['failed_rollbacks']}")

        return 0

    elif args.command == "list":
        rollbacks = automator.list_rollbacks(service_name=args.service, limit=args.limit)

        if args.json:
            print(json.dumps([r.to_dict() for r in rollbacks], indent=2))
        else:
            if not rollbacks:
                print("No rollbacks found")
            else:
                for r in rollbacks:
                    print(f"{r.rollback_id} ({r.service_name}) - {r.status.value}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
