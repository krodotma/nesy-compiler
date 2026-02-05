#!/usr/bin/env python3
"""
Alert Router - Step 258

Routes alerts to appropriate channels based on rules.

PBTSO Phase: ITERATE

Bus Topics:
- monitor.alert.route (subscribed)
- monitor.alert.sent (emitted)

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern

from ..anomaly.detector import Anomaly, AnomalySeverity


class AlertChannel(Enum):
    """Alert delivery channels."""
    BUS = "bus"           # A2A bus
    SLACK = "slack"       # Slack webhook
    EMAIL = "email"       # Email
    PAGERDUTY = "pagerduty"  # PagerDuty
    WEBHOOK = "webhook"   # Generic webhook
    LOG = "log"           # Log file


@dataclass
class RoutingRule:
    """Rule for routing alerts.

    Attributes:
        name: Rule name
        metric_pattern: Regex pattern for metric name
        min_severity: Minimum severity to trigger
        channels: Channels to route to
        labels: Required labels
        suppress_duration_s: Suppress duplicate alerts for this duration
        enabled: Whether rule is active
    """
    name: str
    metric_pattern: str
    min_severity: AnomalySeverity = AnomalySeverity.WARNING
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.BUS])
    labels: Dict[str, str] = field(default_factory=dict)
    suppress_duration_s: int = 300
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "metric_pattern": self.metric_pattern,
            "min_severity": self.min_severity.value,
            "channels": [c.value for c in self.channels],
            "labels": self.labels,
            "suppress_duration_s": self.suppress_duration_s,
            "enabled": self.enabled,
        }


@dataclass
class AlertRoute:
    """Result of routing an alert.

    Attributes:
        alert_id: Unique alert ID
        rule_name: Name of rule that matched
        channels: Channels to deliver to
        anomaly: Source anomaly
        timestamp: When routed
        suppressed: Whether alert was suppressed
        suppress_reason: Reason for suppression
    """
    alert_id: str
    rule_name: str
    channels: List[AlertChannel]
    anomaly: Anomaly
    timestamp: float = field(default_factory=time.time)
    suppressed: bool = False
    suppress_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "rule_name": self.rule_name,
            "channels": [c.value for c in self.channels],
            "anomaly": self.anomaly.to_dict(),
            "timestamp": self.timestamp,
            "suppressed": self.suppressed,
            "suppress_reason": self.suppress_reason,
        }


class AlertRouter:
    """
    Route alerts to appropriate channels.

    The router:
    - Matches anomalies against routing rules
    - Suppresses duplicate alerts
    - Delivers to configured channels
    - Tracks alert delivery

    Example:
        router = AlertRouter()

        # Add routing rule
        router.add_rule(RoutingRule(
            name="critical_errors",
            metric_pattern=".*error.*",
            min_severity=AnomalySeverity.CRITICAL,
            channels=[AlertChannel.SLACK, AlertChannel.PAGERDUTY]
        ))

        # Route an anomaly
        route = router.route(anomaly)
        if route and not route.suppressed:
            router.deliver(route)
    """

    BUS_TOPICS = {
        "route": "monitor.alert.route",
        "sent": "monitor.alert.sent",
    }

    # Channel handlers
    _channel_handlers: Dict[AlertChannel, Callable] = {}

    def __init__(
        self,
        bus_dir: Optional[str] = None,
        default_channels: Optional[List[AlertChannel]] = None
    ):
        """Initialize alert router.

        Args:
            bus_dir: Directory for bus events
            default_channels: Default channels when no rule matches
        """
        self._rules: List[RoutingRule] = []
        self._compiled_patterns: Dict[str, Pattern] = {}
        self._suppression_cache: Dict[str, float] = {}
        self._delivery_count: int = 0
        self._suppression_count: int = 0
        self._default_channels = default_channels or [AlertChannel.BUS, AlertChannel.LOG]

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register default handlers
        self._register_default_handlers()

    def add_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule.

        Args:
            rule: Routing rule
        """
        self._rules.append(rule)
        self._compiled_patterns[rule.metric_pattern] = re.compile(rule.metric_pattern)

    def remove_rule(self, name: str) -> bool:
        """Remove a routing rule by name.

        Args:
            name: Rule name

        Returns:
            True if removed
        """
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                del self._rules[i]
                return True
        return False

    def route(self, anomaly: Anomaly) -> Optional[AlertRoute]:
        """Route an anomaly to appropriate channels.

        Args:
            anomaly: Anomaly to route

        Returns:
            Alert route or None if no matching rules
        """
        matching_rules: List[RoutingRule] = []

        for rule in self._rules:
            if not rule.enabled:
                continue

            # Check severity
            if self._severity_level(anomaly.severity) < self._severity_level(rule.min_severity):
                continue

            # Check metric pattern
            pattern = self._compiled_patterns.get(rule.metric_pattern)
            if pattern and not pattern.match(anomaly.metric_name):
                continue

            # Check required labels
            if rule.labels:
                match = all(
                    anomaly.labels.get(k) == v
                    for k, v in rule.labels.items()
                )
                if not match:
                    continue

            matching_rules.append(rule)

        if not matching_rules:
            # Use default routing for severe anomalies
            if anomaly.severity == AnomalySeverity.CRITICAL:
                return AlertRoute(
                    alert_id=f"alert-{uuid.uuid4().hex[:8]}",
                    rule_name="default",
                    channels=self._default_channels,
                    anomaly=anomaly,
                )
            return None

        # Merge channels from all matching rules
        channels: List[AlertChannel] = []
        rule_names: List[str] = []
        suppress_duration = 0

        for rule in matching_rules:
            for channel in rule.channels:
                if channel not in channels:
                    channels.append(channel)
            rule_names.append(rule.name)
            suppress_duration = max(suppress_duration, rule.suppress_duration_s)

        # Check suppression
        suppression_key = f"{anomaly.metric_name}:{json.dumps(sorted(anomaly.labels.items()))}"
        last_alert = self._suppression_cache.get(suppression_key, 0)

        if time.time() - last_alert < suppress_duration:
            return AlertRoute(
                alert_id=f"alert-{uuid.uuid4().hex[:8]}",
                rule_name=",".join(rule_names),
                channels=channels,
                anomaly=anomaly,
                suppressed=True,
                suppress_reason=f"Suppressed for {suppress_duration}s",
            )

        # Update suppression cache
        self._suppression_cache[suppression_key] = time.time()

        return AlertRoute(
            alert_id=f"alert-{uuid.uuid4().hex[:8]}",
            rule_name=",".join(rule_names),
            channels=channels,
            anomaly=anomaly,
        )

    def deliver(self, route: AlertRoute) -> Dict[AlertChannel, bool]:
        """Deliver an alert via routed channels.

        Args:
            route: Alert route

        Returns:
            Dictionary of channel -> success
        """
        results: Dict[AlertChannel, bool] = {}

        for channel in route.channels:
            handler = self._channel_handlers.get(channel)
            if handler:
                try:
                    success = handler(route)
                    results[channel] = success
                    if success:
                        self._delivery_count += 1
                except Exception as e:
                    results[channel] = False
            else:
                results[channel] = False

        # Emit delivery event
        self._emit_sent_event(route, results)

        return results

    def route_and_deliver(self, anomaly: Anomaly) -> Optional[AlertRoute]:
        """Route and deliver an anomaly in one call.

        Args:
            anomaly: Anomaly to route

        Returns:
            Alert route or None
        """
        route = self.route(anomaly)
        if route and not route.suppressed:
            self.deliver(route)
        elif route and route.suppressed:
            self._suppression_count += 1
        return route

    def register_handler(
        self,
        channel: AlertChannel,
        handler: Callable[[AlertRoute], bool]
    ) -> None:
        """Register a channel handler.

        Args:
            channel: Channel type
            handler: Handler function
        """
        self._channel_handlers[channel] = handler

    def handle_route_request(self, event: Dict[str, Any]) -> Optional[AlertRoute]:
        """Handle routing request from bus.

        Args:
            event: Bus event

        Returns:
            Alert route or None
        """
        data = event.get("data", {})

        # Reconstruct anomaly from event data
        anomaly_data = data.get("anomaly", data)
        if "anomaly_id" not in anomaly_data:
            anomaly_data["anomaly_id"] = f"anomaly-{uuid.uuid4().hex[:8]}"

        anomaly = Anomaly(
            anomaly_id=anomaly_data.get("anomaly_id"),
            metric_name=anomaly_data.get("metric_name", "unknown"),
            timestamp=anomaly_data.get("timestamp", time.time()),
            expected_value=anomaly_data.get("expected_value", 0),
            actual_value=anomaly_data.get("actual_value", 0),
            deviation_sigma=anomaly_data.get("deviation_sigma", 0),
            severity=AnomalySeverity(anomaly_data.get("severity", "warning")),
            method=anomaly_data.get("method", "z_score"),
            labels=anomaly_data.get("labels", {}),
        )

        return self.route_and_deliver(anomaly)

    def get_rules(self) -> List[RoutingRule]:
        """Get all routing rules.

        Returns:
            List of rules
        """
        return self._rules.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "rules_count": len(self._rules),
            "active_rules": sum(1 for r in self._rules if r.enabled),
            "deliveries": self._delivery_count,
            "suppressions": self._suppression_count,
            "suppression_cache_size": len(self._suppression_cache),
        }

    def _register_default_handlers(self) -> None:
        """Register default channel handlers."""
        # Bus handler
        def bus_handler(route: AlertRoute) -> bool:
            self._emit_bus_alert(route)
            return True

        # Log handler
        def log_handler(route: AlertRoute) -> bool:
            log_path = Path(self._bus_dir) / "alerts.log"
            with open(log_path, "a") as f:
                f.write(json.dumps(route.to_dict()) + "\n")
            return True

        self._channel_handlers[AlertChannel.BUS] = bus_handler
        self._channel_handlers[AlertChannel.LOG] = log_handler

    def _emit_bus_alert(self, route: AlertRoute) -> str:
        """Emit alert to bus."""
        event_id = str(uuid.uuid4())
        level = (
            "error" if route.anomaly.severity == AnomalySeverity.CRITICAL
            else "warn"
        )

        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": "monitor.alert",
            "kind": "alert",
            "level": level,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": {
                "alert_id": route.alert_id,
                "metric": route.anomaly.metric_name,
                "severity": route.anomaly.severity.value,
                "value": route.anomaly.actual_value,
                "expected": route.anomaly.expected_value,
                "deviation": route.anomaly.deviation_sigma,
                "rule": route.rule_name,
            },
        }

        with open(self._bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def _emit_sent_event(
        self,
        route: AlertRoute,
        results: Dict[AlertChannel, bool]
    ) -> str:
        """Emit alert sent event."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": self.BUS_TOPICS["sent"],
            "kind": "event",
            "level": "info",
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": {
                "alert_id": route.alert_id,
                "channels": {c.value: s for c, s in results.items()},
                "suppressed": route.suppressed,
            },
        }

        with open(self._bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def _severity_level(self, severity: AnomalySeverity) -> int:
        """Get numeric severity level."""
        levels = {
            AnomalySeverity.INFO: 1,
            AnomalySeverity.WARNING: 2,
            AnomalySeverity.CRITICAL: 3,
        }
        return levels.get(severity, 0)


# Singleton instance
_router: Optional[AlertRouter] = None


def get_router() -> AlertRouter:
    """Get or create the alert router singleton.

    Returns:
        AlertRouter instance
    """
    global _router
    if _router is None:
        _router = AlertRouter()
    return _router


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alert Router (Step 258)")
    parser.add_argument("--add-rule", metavar="NAME", help="Add a routing rule")
    parser.add_argument("--pattern", default=".*", help="Metric pattern for rule")
    parser.add_argument("--severity", default="warning", help="Minimum severity")
    parser.add_argument("--channels", default="bus", help="Comma-separated channels")
    parser.add_argument("--list-rules", action="store_true", help="List all rules")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    router = get_router()

    if args.add_rule:
        channels = [AlertChannel(c.strip()) for c in args.channels.split(",")]
        rule = RoutingRule(
            name=args.add_rule,
            metric_pattern=args.pattern,
            min_severity=AnomalySeverity(args.severity),
            channels=channels,
        )
        router.add_rule(rule)
        print(f"Added rule: {args.add_rule}")

    if args.list_rules:
        rules = router.get_rules()
        if args.json:
            print(json.dumps([r.to_dict() for r in rules], indent=2))
        else:
            print(f"Routing Rules ({len(rules)}):")
            for r in rules:
                status = "enabled" if r.enabled else "disabled"
                channels = ",".join(c.value for c in r.channels)
                print(f"  {r.name}: {r.metric_pattern} -> {channels} ({status})")

    if args.stats:
        stats = router.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Alert Router Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
