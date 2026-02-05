#!/usr/bin/env python3
"""
manager.py - Traffic Manager (Step 214)

PBTSO Phase: ITERATE
A2A Integration: Manages traffic routing via deploy.traffic.route

Provides:
- RoutingStrategy: Traffic routing strategies
- TrafficRule: Traffic routing rule definition
- TrafficSplit: Traffic split configuration
- TrafficState: Current traffic state
- TrafficManager: Traffic routing and balancing

Bus Topics:
- deploy.traffic.route
- deploy.traffic.split
- deploy.traffic.shift
- deploy.traffic.mirror

Protocol: DKIN v30, CITIZEN v2
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
    actor: str = "traffic-manager"
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

class RoutingStrategy(Enum):
    """Traffic routing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    HEADER_BASED = "header_based"
    COOKIE_BASED = "cookie_based"
    PATH_BASED = "path_based"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"
    STICKY = "sticky"


class TrafficAction(Enum):
    """Traffic rule actions."""
    ROUTE = "route"
    REDIRECT = "redirect"
    MIRROR = "mirror"
    REJECT = "reject"
    RETRY = "retry"
    TIMEOUT = "timeout"


@dataclass
class TrafficTarget:
    """
    Traffic routing target.

    Attributes:
        target_id: Unique target identifier
        name: Target name
        endpoint: Target endpoint URL
        weight: Traffic weight (0-100)
        version: Service version
        metadata: Additional metadata
        healthy: Whether target is healthy
    """
    target_id: str
    name: str
    endpoint: str
    weight: int = 100
    version: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    healthy: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrafficRule:
    """
    Traffic routing rule.

    Attributes:
        rule_id: Unique rule identifier
        name: Rule name
        priority: Rule priority (lower = higher priority)
        service_name: Service this rule applies to
        match_conditions: Conditions for matching traffic
        action: Action to take
        targets: Routing targets
        retry_config: Retry configuration
        timeout_ms: Request timeout
        enabled: Whether rule is enabled
        created_at: Creation timestamp
    """
    rule_id: str
    name: str
    priority: int = 100
    service_name: str = ""
    match_conditions: Dict[str, Any] = field(default_factory=dict)
    action: TrafficAction = TrafficAction.ROUTE
    targets: List[TrafficTarget] = field(default_factory=list)
    retry_config: Dict[str, Any] = field(default_factory=dict)
    timeout_ms: int = 30000
    enabled: bool = True
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "priority": self.priority,
            "service_name": self.service_name,
            "match_conditions": self.match_conditions,
            "action": self.action.value,
            "targets": [t.to_dict() for t in self.targets],
            "retry_config": self.retry_config,
            "timeout_ms": self.timeout_ms,
            "enabled": self.enabled,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrafficRule":
        data = dict(data)
        if "action" in data:
            data["action"] = TrafficAction(data["action"])
        if "targets" in data:
            data["targets"] = [
                TrafficTarget(**t) if isinstance(t, dict) else t
                for t in data["targets"]
            ]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrafficSplit:
    """
    Traffic split configuration.

    Attributes:
        split_id: Unique split identifier
        service_name: Service name
        strategy: Routing strategy
        splits: Target weight mapping
        sticky_config: Sticky session configuration
        updated_at: Last update timestamp
    """
    split_id: str
    service_name: str
    strategy: RoutingStrategy = RoutingStrategy.WEIGHTED
    splits: Dict[str, int] = field(default_factory=dict)  # target_id -> weight
    sticky_config: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_id": self.split_id,
            "service_name": self.service_name,
            "strategy": self.strategy.value,
            "splits": self.splits,
            "sticky_config": self.sticky_config,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrafficSplit":
        data = dict(data)
        if "strategy" in data:
            data["strategy"] = RoutingStrategy(data["strategy"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrafficState:
    """
    Current traffic state for a service.

    Attributes:
        service_name: Service name
        current_split: Current traffic split
        active_rules: Active traffic rules
        targets: Available targets
        metrics: Traffic metrics
        last_updated: Last update timestamp
    """
    service_name: str
    current_split: Optional[TrafficSplit] = None
    active_rules: List[TrafficRule] = field(default_factory=list)
    targets: List[TrafficTarget] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "current_split": self.current_split.to_dict() if self.current_split else None,
            "active_rules": [r.to_dict() for r in self.active_rules],
            "targets": [t.to_dict() for t in self.targets],
            "metrics": self.metrics,
            "last_updated": self.last_updated,
        }


# ==============================================================================
# Traffic Manager (Step 214)
# ==============================================================================

class TrafficManager:
    """
    Traffic Manager - manages traffic routing and balancing.

    PBTSO Phase: ITERATE

    Responsibilities:
    - Configure traffic routing rules
    - Implement traffic splitting for deployments
    - Support canary and blue-green traffic shifts
    - Mirror traffic for testing
    - Track traffic metrics

    Example:
        >>> manager = TrafficManager()
        >>> # Create a weighted traffic split
        >>> split = manager.create_split(
        ...     service_name="api-service",
        ...     strategy=RoutingStrategy.WEIGHTED,
        ...     splits={"v1": 90, "v2": 10}
        ... )
        >>> # Shift traffic gradually
        >>> await manager.shift_traffic("api-service", "v2", 50)
    """

    BUS_TOPICS = {
        "route": "deploy.traffic.route",
        "split": "deploy.traffic.split",
        "shift": "deploy.traffic.shift",
        "mirror": "deploy.traffic.mirror",
        "rule_created": "deploy.traffic.rule.created",
        "rule_updated": "deploy.traffic.rule.updated",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "traffic-manager",
    ):
        """
        Initialize the traffic manager.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "traffic"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Storage
        self._rules: Dict[str, TrafficRule] = {}
        self._splits: Dict[str, TrafficSplit] = {}
        self._targets: Dict[str, TrafficTarget] = {}
        self._service_targets: Dict[str, List[str]] = {}  # service -> target_ids

        self._load_state()

    def register_target(
        self,
        service_name: str,
        name: str,
        endpoint: str,
        version: str = "",
        weight: int = 100,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrafficTarget:
        """
        Register a traffic target.

        Args:
            service_name: Service name
            name: Target name
            endpoint: Target endpoint
            version: Service version
            weight: Initial weight
            metadata: Additional metadata

        Returns:
            Created TrafficTarget
        """
        target_id = f"target-{uuid.uuid4().hex[:12]}"

        target = TrafficTarget(
            target_id=target_id,
            name=name,
            endpoint=endpoint,
            version=version,
            weight=weight,
            metadata=metadata or {},
        )

        self._targets[target_id] = target

        if service_name not in self._service_targets:
            self._service_targets[service_name] = []
        self._service_targets[service_name].append(target_id)

        self._save_state()

        return target

    def create_rule(
        self,
        name: str,
        service_name: str,
        targets: List[TrafficTarget],
        priority: int = 100,
        match_conditions: Optional[Dict[str, Any]] = None,
        action: TrafficAction = TrafficAction.ROUTE,
        timeout_ms: int = 30000,
    ) -> TrafficRule:
        """
        Create a traffic routing rule.

        Args:
            name: Rule name
            service_name: Service this rule applies to
            targets: Routing targets
            priority: Rule priority
            match_conditions: Match conditions
            action: Traffic action
            timeout_ms: Request timeout

        Returns:
            Created TrafficRule
        """
        rule_id = f"rule-{uuid.uuid4().hex[:12]}"

        rule = TrafficRule(
            rule_id=rule_id,
            name=name,
            service_name=service_name,
            priority=priority,
            match_conditions=match_conditions or {},
            action=action,
            targets=targets,
            timeout_ms=timeout_ms,
        )

        self._rules[rule_id] = rule
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["rule_created"],
            {
                "rule_id": rule_id,
                "name": name,
                "service_name": service_name,
                "action": action.value,
                "target_count": len(targets),
            },
            actor=self.actor_id,
        )

        return rule

    def create_split(
        self,
        service_name: str,
        strategy: RoutingStrategy = RoutingStrategy.WEIGHTED,
        splits: Optional[Dict[str, int]] = None,
        sticky_config: Optional[Dict[str, Any]] = None,
    ) -> TrafficSplit:
        """
        Create a traffic split configuration.

        Args:
            service_name: Service name
            strategy: Routing strategy
            splits: Target weight mapping
            sticky_config: Sticky session configuration

        Returns:
            Created TrafficSplit
        """
        split_id = f"split-{uuid.uuid4().hex[:12]}"

        # Validate weights sum to 100
        if splits:
            total = sum(splits.values())
            if total != 100:
                # Normalize weights
                splits = {k: int(v * 100 / total) for k, v in splits.items()}

        split = TrafficSplit(
            split_id=split_id,
            service_name=service_name,
            strategy=strategy,
            splits=splits or {},
            sticky_config=sticky_config or {},
        )

        self._splits[service_name] = split
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["split"],
            {
                "split_id": split_id,
                "service_name": service_name,
                "strategy": strategy.value,
                "splits": splits,
            },
            actor=self.actor_id,
        )

        return split

    async def shift_traffic(
        self,
        service_name: str,
        target: str,
        new_weight: int,
        duration_s: int = 0,
    ) -> TrafficSplit:
        """
        Shift traffic to a target.

        Args:
            service_name: Service name
            target: Target identifier or version
            new_weight: New weight for target
            duration_s: Duration for gradual shift (0 = immediate)

        Returns:
            Updated TrafficSplit
        """
        split = self._splits.get(service_name)
        if not split:
            split = self.create_split(service_name)

        old_splits = dict(split.splits)

        if duration_s > 0:
            # Gradual shift
            steps = 10
            step_duration = duration_s / steps

            current_weight = split.splits.get(target, 0)
            weight_delta = (new_weight - current_weight) / steps

            for i in range(steps):
                intermediate_weight = int(current_weight + weight_delta * (i + 1))
                split.splits[target] = intermediate_weight

                # Redistribute remaining weight
                remaining = 100 - intermediate_weight
                other_targets = [t for t in split.splits if t != target]
                if other_targets:
                    per_target = remaining // len(other_targets)
                    for ot in other_targets:
                        split.splits[ot] = per_target

                split.updated_at = time.time()
                self._save_state()

                await asyncio.sleep(step_duration)
        else:
            # Immediate shift
            split.splits[target] = new_weight

            # Redistribute remaining weight
            remaining = 100 - new_weight
            other_targets = [t for t in split.splits if t != target]
            if other_targets:
                per_target = remaining // len(other_targets)
                for ot in other_targets:
                    split.splits[ot] = per_target

        split.updated_at = time.time()
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["shift"],
            {
                "service_name": service_name,
                "target": target,
                "new_weight": new_weight,
                "old_splits": old_splits,
                "new_splits": split.splits,
                "duration_s": duration_s,
            },
            actor=self.actor_id,
        )

        return split

    async def mirror_traffic(
        self,
        service_name: str,
        primary_target: str,
        mirror_target: str,
        mirror_percentage: int = 100,
    ) -> TrafficRule:
        """
        Set up traffic mirroring.

        Args:
            service_name: Service name
            primary_target: Primary traffic target
            mirror_target: Target to mirror traffic to
            mirror_percentage: Percentage of traffic to mirror

        Returns:
            Created mirroring rule
        """
        # Create mirror rule
        rule = self.create_rule(
            name=f"{service_name}-mirror",
            service_name=service_name,
            targets=[],
            action=TrafficAction.MIRROR,
        )

        rule.match_conditions = {
            "mirror_to": mirror_target,
            "mirror_percentage": mirror_percentage,
            "primary_target": primary_target,
        }

        self._rules[rule.rule_id] = rule
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["mirror"],
            {
                "rule_id": rule.rule_id,
                "service_name": service_name,
                "primary_target": primary_target,
                "mirror_target": mirror_target,
                "mirror_percentage": mirror_percentage,
            },
            actor=self.actor_id,
        )

        return rule

    def route_request(
        self,
        service_name: str,
        headers: Optional[Dict[str, str]] = None,
        path: Optional[str] = None,
        cookies: Optional[Dict[str, str]] = None,
    ) -> Optional[TrafficTarget]:
        """
        Route a request to a target based on rules.

        Args:
            service_name: Service name
            headers: Request headers
            path: Request path
            cookies: Request cookies

        Returns:
            Selected TrafficTarget or None
        """
        # Get applicable rules sorted by priority
        rules = [r for r in self._rules.values()
                if r.service_name == service_name and r.enabled]
        rules.sort(key=lambda r: r.priority)

        # Check header-based routing
        if headers:
            for rule in rules:
                header_match = rule.match_conditions.get("headers", {})
                if all(headers.get(k) == v for k, v in header_match.items()):
                    if rule.targets:
                        return self._select_target(rule.targets)

        # Check path-based routing
        if path:
            for rule in rules:
                path_prefix = rule.match_conditions.get("path_prefix")
                if path_prefix and path.startswith(path_prefix):
                    if rule.targets:
                        return self._select_target(rule.targets)

        # Fall back to traffic split
        split = self._splits.get(service_name)
        if split:
            return self._select_by_weight(split)

        # Default: return first healthy target
        target_ids = self._service_targets.get(service_name, [])
        for tid in target_ids:
            target = self._targets.get(tid)
            if target and target.healthy:
                return target

        return None

    def _select_target(self, targets: List[TrafficTarget]) -> Optional[TrafficTarget]:
        """Select a target using weighted random selection."""
        import random

        healthy_targets = [t for t in targets if t.healthy]
        if not healthy_targets:
            return None

        total_weight = sum(t.weight for t in healthy_targets)
        if total_weight == 0:
            return random.choice(healthy_targets)

        r = random.randint(1, total_weight)
        cumulative = 0
        for target in healthy_targets:
            cumulative += target.weight
            if r <= cumulative:
                return target

        return healthy_targets[-1]

    def _select_by_weight(self, split: TrafficSplit) -> Optional[TrafficTarget]:
        """Select a target based on traffic split weights."""
        import random

        if not split.splits:
            return None

        r = random.randint(1, 100)
        cumulative = 0
        for target_id, weight in split.splits.items():
            cumulative += weight
            if r <= cumulative:
                return self._targets.get(target_id)

        # Return last target
        last_target_id = list(split.splits.keys())[-1]
        return self._targets.get(last_target_id)

    def update_target_health(self, target_id: str, healthy: bool) -> None:
        """Update target health status."""
        target = self._targets.get(target_id)
        if target:
            target.healthy = healthy
            self._save_state()

    def get_state(self, service_name: str) -> TrafficState:
        """Get current traffic state for a service."""
        target_ids = self._service_targets.get(service_name, [])
        targets = [self._targets[tid] for tid in target_ids if tid in self._targets]

        rules = [r for r in self._rules.values() if r.service_name == service_name]
        rules.sort(key=lambda r: r.priority)

        return TrafficState(
            service_name=service_name,
            current_split=self._splits.get(service_name),
            active_rules=rules,
            targets=targets,
            last_updated=time.time(),
        )

    def get_rule(self, rule_id: str) -> Optional[TrafficRule]:
        """Get a traffic rule by ID."""
        return self._rules.get(rule_id)

    def get_split(self, service_name: str) -> Optional[TrafficSplit]:
        """Get traffic split for a service."""
        return self._splits.get(service_name)

    def list_rules(self, service_name: Optional[str] = None) -> List[TrafficRule]:
        """List traffic rules."""
        rules = list(self._rules.values())
        if service_name:
            rules = [r for r in rules if r.service_name == service_name]
        return sorted(rules, key=lambda r: r.priority)

    def list_targets(self, service_name: Optional[str] = None) -> List[TrafficTarget]:
        """List traffic targets."""
        if service_name:
            target_ids = self._service_targets.get(service_name, [])
            return [self._targets[tid] for tid in target_ids if tid in self._targets]
        return list(self._targets.values())

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a traffic rule."""
        if rule_id not in self._rules:
            return False

        del self._rules[rule_id]
        self._save_state()
        return True

    def enable_rule(self, rule_id: str, enabled: bool = True) -> Optional[TrafficRule]:
        """Enable or disable a rule."""
        rule = self._rules.get(rule_id)
        if rule:
            rule.enabled = enabled
            self._save_state()
        return rule

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "rules": {rid: r.to_dict() for rid, r in self._rules.items()},
            "splits": {sid: s.to_dict() for sid, s in self._splits.items()},
            "targets": {tid: t.to_dict() for tid, t in self._targets.items()},
            "service_targets": self._service_targets,
        }
        state_file = self.state_dir / "traffic_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "traffic_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            for rid, data in state.get("rules", {}).items():
                self._rules[rid] = TrafficRule.from_dict(data)

            for sid, data in state.get("splits", {}).items():
                self._splits[sid] = TrafficSplit.from_dict(data)

            for tid, data in state.get("targets", {}).items():
                self._targets[tid] = TrafficTarget(**data)

            self._service_targets = state.get("service_targets", {})
        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for traffic manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Traffic Manager (Step 214)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # target command
    target_parser = subparsers.add_parser("target", help="Register a target")
    target_parser.add_argument("service_name", help="Service name")
    target_parser.add_argument("--name", "-n", required=True, help="Target name")
    target_parser.add_argument("--endpoint", "-e", required=True, help="Target endpoint")
    target_parser.add_argument("--version", "-v", default="", help="Service version")
    target_parser.add_argument("--weight", "-w", type=int, default=100, help="Initial weight")
    target_parser.add_argument("--json", action="store_true", help="JSON output")

    # split command
    split_parser = subparsers.add_parser("split", help="Create traffic split")
    split_parser.add_argument("service_name", help="Service name")
    split_parser.add_argument("--weights", "-w", required=True, help="Weights (target:weight,...)")
    split_parser.add_argument("--strategy", "-s", default="weighted",
                             choices=["weighted", "round_robin", "canary", "blue_green"])
    split_parser.add_argument("--json", action="store_true", help="JSON output")

    # shift command
    shift_parser = subparsers.add_parser("shift", help="Shift traffic")
    shift_parser.add_argument("service_name", help="Service name")
    shift_parser.add_argument("--target", "-t", required=True, help="Target to shift to")
    shift_parser.add_argument("--weight", "-w", type=int, required=True, help="New weight")
    shift_parser.add_argument("--duration", "-d", type=int, default=0, help="Duration in seconds")
    shift_parser.add_argument("--json", action="store_true", help="JSON output")

    # status command
    status_parser = subparsers.add_parser("status", help="Get traffic status")
    status_parser.add_argument("service_name", help="Service name")
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List targets or rules")
    list_parser.add_argument("--service", "-s", help="Filter by service")
    list_parser.add_argument("--type", "-t", default="targets", choices=["targets", "rules"])
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    manager = TrafficManager()

    if args.command == "target":
        target = manager.register_target(
            service_name=args.service_name,
            name=args.name,
            endpoint=args.endpoint,
            version=args.version,
            weight=args.weight,
        )

        if args.json:
            print(json.dumps(target.to_dict(), indent=2))
        else:
            print(f"Registered target: {target.target_id}")
            print(f"  Name: {target.name}")
            print(f"  Endpoint: {target.endpoint}")
            print(f"  Weight: {target.weight}")

        return 0

    elif args.command == "split":
        # Parse weights
        splits = {}
        for item in args.weights.split(","):
            target, weight = item.split(":")
            splits[target.strip()] = int(weight.strip())

        split = manager.create_split(
            service_name=args.service_name,
            strategy=RoutingStrategy(args.strategy),
            splits=splits,
        )

        if args.json:
            print(json.dumps(split.to_dict(), indent=2))
        else:
            print(f"Created split: {split.split_id}")
            print(f"  Service: {split.service_name}")
            print(f"  Strategy: {split.strategy.value}")
            print(f"  Splits: {split.splits}")

        return 0

    elif args.command == "shift":
        split = asyncio.get_event_loop().run_until_complete(
            manager.shift_traffic(
                service_name=args.service_name,
                target=args.target,
                new_weight=args.weight,
                duration_s=args.duration,
            )
        )

        if args.json:
            print(json.dumps(split.to_dict(), indent=2))
        else:
            print(f"Shifted traffic for {args.service_name}")
            print(f"  New splits: {split.splits}")

        return 0

    elif args.command == "status":
        state = manager.get_state(args.service_name)

        if args.json:
            print(json.dumps(state.to_dict(), indent=2))
        else:
            print(f"Traffic State: {state.service_name}")
            if state.current_split:
                print(f"  Split: {state.current_split.splits}")
            print(f"  Targets: {len(state.targets)}")
            print(f"  Rules: {len(state.active_rules)}")

        return 0

    elif args.command == "list":
        if args.type == "targets":
            items = manager.list_targets(args.service)
            if args.json:
                print(json.dumps([t.to_dict() for t in items], indent=2))
            else:
                for t in items:
                    health = "healthy" if t.healthy else "unhealthy"
                    print(f"{t.target_id} ({t.name}) - {t.endpoint} [{health}]")
        else:
            items = manager.list_rules(args.service)
            if args.json:
                print(json.dumps([r.to_dict() for r in items], indent=2))
            else:
                for r in items:
                    status = "enabled" if r.enabled else "disabled"
                    print(f"{r.rule_id} ({r.name}) - {r.action.value} [{status}]")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
