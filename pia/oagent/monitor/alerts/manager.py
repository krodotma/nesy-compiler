#!/usr/bin/env python3
"""
Alert Manager - Step 259

Manages alert lifecycle: creation, acknowledgment, resolution.

PBTSO Phase: ITERATE

Bus Topics:
- monitor.alert.create (subscribed)
- monitor.alert.acknowledge (subscribed)
- monitor.alert.resolve (subscribed)

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ..anomaly.detector import Anomaly, AnomalySeverity
from .router import AlertRoute


class AlertState(Enum):
    """Alert lifecycle states."""
    FIRING = "firing"         # Active alert
    ACKNOWLEDGED = "acknowledged"  # Acknowledged but not resolved
    RESOLVED = "resolved"     # Issue resolved
    EXPIRED = "expired"       # Auto-expired


class AlertPriority(Enum):
    """Alert priority levels."""
    P1 = "p1"  # Critical - immediate response
    P2 = "p2"  # High - response within 1 hour
    P3 = "p3"  # Medium - response within 4 hours
    P4 = "p4"  # Low - response within 24 hours


@dataclass
class Alert:
    """A managed alert.

    Attributes:
        alert_id: Unique alert ID
        metric_name: Source metric
        severity: Alert severity
        priority: Alert priority
        state: Current state
        message: Alert message
        labels: Alert labels
        annotations: Additional annotations
        created_at: Creation timestamp
        updated_at: Last update timestamp
        acknowledged_at: Acknowledgment timestamp
        acknowledged_by: Who acknowledged
        resolved_at: Resolution timestamp
        resolved_by: Who resolved
        resolve_note: Resolution note
        firing_count: Number of times fired
        source_anomalies: Source anomaly IDs
    """
    alert_id: str
    metric_name: str
    severity: AnomalySeverity
    priority: AlertPriority = AlertPriority.P3
    state: AlertState = AlertState.FIRING
    message: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None
    resolve_note: Optional[str] = None
    firing_count: int = 1
    source_anomalies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "metric_name": self.metric_name,
            "severity": self.severity.value,
            "priority": self.priority.value,
            "state": self.state.value,
            "message": self.message,
            "labels": self.labels,
            "annotations": self.annotations,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "resolve_note": self.resolve_note,
            "firing_count": self.firing_count,
            "source_anomalies": self.source_anomalies,
            "age_s": time.time() - self.created_at,
        }


class AlertManager:
    """
    Manage alert lifecycle.

    The manager:
    - Creates and tracks alerts
    - Handles acknowledgment and resolution
    - Groups related alerts
    - Manages alert state transitions
    - Auto-resolves stale alerts

    Example:
        manager = AlertManager()

        # Create alert from anomaly
        alert = manager.create_from_anomaly(anomaly)

        # Acknowledge
        manager.acknowledge(alert.alert_id, acknowledged_by="oncall")

        # Resolve
        manager.resolve(alert.alert_id, resolved_by="oncall", note="Fixed")
    """

    BUS_TOPICS = {
        "create": "monitor.alert.create",
        "acknowledge": "monitor.alert.acknowledge",
        "resolve": "monitor.alert.resolve",
        "state_change": "monitor.alert.state",
    }

    def __init__(
        self,
        auto_resolve_hours: int = 24,
        grouping_window_s: int = 300,
        bus_dir: Optional[str] = None
    ):
        """Initialize alert manager.

        Args:
            auto_resolve_hours: Hours before auto-resolving stale alerts
            grouping_window_s: Window for grouping related alerts
            bus_dir: Directory for bus events
        """
        self.auto_resolve_hours = auto_resolve_hours
        self.grouping_window_s = grouping_window_s

        # Alert storage
        self._alerts: Dict[str, Alert] = {}
        self._by_metric: Dict[str, List[str]] = defaultdict(list)
        self._by_state: Dict[AlertState, Set[str]] = {
            state: set() for state in AlertState
        }
        self._lock = threading.RLock()

        # Callbacks
        self._on_state_change: List[Callable[[Alert, AlertState, AlertState], None]] = []

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        metric_name: str,
        severity: AnomalySeverity,
        message: str,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
        priority: Optional[AlertPriority] = None
    ) -> Alert:
        """Create a new alert.

        Args:
            metric_name: Source metric name
            severity: Alert severity
            message: Alert message
            labels: Alert labels
            annotations: Additional annotations
            priority: Alert priority

        Returns:
            Created alert
        """
        with self._lock:
            # Check for existing firing alert for same metric/labels
            existing = self._find_existing_alert(metric_name, labels)
            if existing:
                existing.firing_count += 1
                existing.updated_at = time.time()
                return existing

            # Determine priority from severity if not provided
            if priority is None:
                priority = self._severity_to_priority(severity)

            alert = Alert(
                alert_id=f"alert-{uuid.uuid4().hex[:8]}",
                metric_name=metric_name,
                severity=severity,
                priority=priority,
                message=message,
                labels=labels or {},
                annotations=annotations or {},
            )

            self._alerts[alert.alert_id] = alert
            self._by_metric[metric_name].append(alert.alert_id)
            self._by_state[AlertState.FIRING].add(alert.alert_id)

        # Emit creation event
        self._emit_state_change_event(alert, None, AlertState.FIRING)

        return alert

    def create_from_anomaly(self, anomaly: Anomaly) -> Alert:
        """Create alert from anomaly.

        Args:
            anomaly: Source anomaly

        Returns:
            Created alert
        """
        message = (
            f"Anomaly detected in {anomaly.metric_name}: "
            f"value={anomaly.actual_value:.2f}, expected={anomaly.expected_value:.2f}, "
            f"deviation={anomaly.deviation_sigma:.2f} sigma"
        )

        alert = self.create(
            metric_name=anomaly.metric_name,
            severity=anomaly.severity,
            message=message,
            labels=anomaly.labels,
            annotations={
                "detection_method": anomaly.method.value if hasattr(anomaly.method, 'value') else str(anomaly.method),
                "expected_value": str(anomaly.expected_value),
                "actual_value": str(anomaly.actual_value),
            }
        )

        alert.source_anomalies.append(anomaly.anomaly_id)
        return alert

    def create_from_route(self, route: AlertRoute) -> Alert:
        """Create alert from alert route.

        Args:
            route: Alert route

        Returns:
            Created alert
        """
        return self.create_from_anomaly(route.anomaly)

    def acknowledge(
        self,
        alert_id: str,
        acknowledged_by: str,
        note: Optional[str] = None
    ) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert ID
            acknowledged_by: Who acknowledged
            note: Optional note

        Returns:
            True if acknowledged
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if not alert:
                return False

            if alert.state != AlertState.FIRING:
                return False

            old_state = alert.state
            alert.state = AlertState.ACKNOWLEDGED
            alert.acknowledged_at = time.time()
            alert.acknowledged_by = acknowledged_by
            alert.updated_at = time.time()

            if note:
                alert.annotations["ack_note"] = note

            self._by_state[old_state].discard(alert_id)
            self._by_state[AlertState.ACKNOWLEDGED].add(alert_id)

        # Emit state change
        self._emit_state_change_event(alert, old_state, AlertState.ACKNOWLEDGED)
        self._notify_state_change(alert, old_state, AlertState.ACKNOWLEDGED)

        return True

    def resolve(
        self,
        alert_id: str,
        resolved_by: str,
        note: Optional[str] = None
    ) -> bool:
        """Resolve an alert.

        Args:
            alert_id: Alert ID
            resolved_by: Who resolved
            note: Resolution note

        Returns:
            True if resolved
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if not alert:
                return False

            if alert.state == AlertState.RESOLVED:
                return False

            old_state = alert.state
            alert.state = AlertState.RESOLVED
            alert.resolved_at = time.time()
            alert.resolved_by = resolved_by
            alert.resolve_note = note
            alert.updated_at = time.time()

            self._by_state[old_state].discard(alert_id)
            self._by_state[AlertState.RESOLVED].add(alert_id)

        # Emit state change
        self._emit_state_change_event(alert, old_state, AlertState.RESOLVED)
        self._notify_state_change(alert, old_state, AlertState.RESOLVED)

        return True

    def refire(self, alert_id: str) -> bool:
        """Refire a resolved or acknowledged alert.

        Args:
            alert_id: Alert ID

        Returns:
            True if refired
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if not alert:
                return False

            if alert.state == AlertState.FIRING:
                alert.firing_count += 1
                alert.updated_at = time.time()
                return True

            old_state = alert.state
            alert.state = AlertState.FIRING
            alert.firing_count += 1
            alert.updated_at = time.time()

            self._by_state[old_state].discard(alert_id)
            self._by_state[AlertState.FIRING].add(alert_id)

        # Emit state change
        self._emit_state_change_event(alert, old_state, AlertState.FIRING)
        self._notify_state_change(alert, old_state, AlertState.FIRING)

        return True

    def get(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID.

        Args:
            alert_id: Alert ID

        Returns:
            Alert or None
        """
        return self._alerts.get(alert_id)

    def get_firing(self) -> List[Alert]:
        """Get all firing alerts.

        Returns:
            List of firing alerts
        """
        with self._lock:
            return [
                self._alerts[aid]
                for aid in self._by_state[AlertState.FIRING]
                if aid in self._alerts
            ]

    def get_by_state(self, state: AlertState) -> List[Alert]:
        """Get alerts by state.

        Args:
            state: Alert state

        Returns:
            List of alerts
        """
        with self._lock:
            return [
                self._alerts[aid]
                for aid in self._by_state.get(state, set())
                if aid in self._alerts
            ]

    def get_by_metric(self, metric_name: str) -> List[Alert]:
        """Get alerts by metric name.

        Args:
            metric_name: Metric name

        Returns:
            List of alerts
        """
        with self._lock:
            return [
                self._alerts[aid]
                for aid in self._by_metric.get(metric_name, [])
                if aid in self._alerts
            ]

    def get_active(self) -> List[Alert]:
        """Get all active (firing or acknowledged) alerts.

        Returns:
            List of active alerts
        """
        with self._lock:
            active_ids = (
                self._by_state[AlertState.FIRING] |
                self._by_state[AlertState.ACKNOWLEDGED]
            )
            return [
                self._alerts[aid]
                for aid in active_ids
                if aid in self._alerts
            ]

    def cleanup_old_alerts(self, max_age_hours: int = 168) -> int:
        """Clean up old resolved alerts.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of alerts removed
        """
        cutoff = time.time() - (max_age_hours * 3600)
        removed = 0

        with self._lock:
            for alert_id in list(self._alerts.keys()):
                alert = self._alerts[alert_id]
                if alert.state == AlertState.RESOLVED and alert.resolved_at:
                    if alert.resolved_at < cutoff:
                        del self._alerts[alert_id]
                        self._by_state[AlertState.RESOLVED].discard(alert_id)
                        removed += 1

        return removed

    def auto_resolve_stale(self) -> int:
        """Auto-resolve stale alerts that haven't fired recently.

        Returns:
            Number of alerts auto-resolved
        """
        cutoff = time.time() - (self.auto_resolve_hours * 3600)
        resolved = 0

        with self._lock:
            for alert_id in list(self._by_state[AlertState.FIRING]):
                alert = self._alerts.get(alert_id)
                if alert and alert.updated_at < cutoff:
                    self.resolve(
                        alert_id,
                        resolved_by="auto-resolve",
                        note="Auto-resolved due to inactivity"
                    )
                    resolved += 1

        return resolved

    def on_state_change(
        self,
        callback: Callable[[Alert, AlertState, AlertState], None]
    ) -> None:
        """Register state change callback.

        Args:
            callback: Callback function(alert, old_state, new_state)
        """
        self._on_state_change.append(callback)

    def handle_create_request(self, event: Dict[str, Any]) -> Optional[Alert]:
        """Handle create request from bus.

        Args:
            event: Bus event

        Returns:
            Created alert or None
        """
        data = event.get("data", {})
        return self.create(
            metric_name=data.get("metric_name", "unknown"),
            severity=AnomalySeverity(data.get("severity", "warning")),
            message=data.get("message", ""),
            labels=data.get("labels", {}),
            annotations=data.get("annotations", {}),
        )

    def handle_acknowledge_request(self, event: Dict[str, Any]) -> bool:
        """Handle acknowledge request from bus.

        Args:
            event: Bus event

        Returns:
            Success
        """
        data = event.get("data", {})
        return self.acknowledge(
            alert_id=data.get("alert_id", ""),
            acknowledged_by=data.get("acknowledged_by", event.get("actor", "unknown")),
            note=data.get("note"),
        )

    def handle_resolve_request(self, event: Dict[str, Any]) -> bool:
        """Handle resolve request from bus.

        Args:
            event: Bus event

        Returns:
            Success
        """
        data = event.get("data", {})
        return self.resolve(
            alert_id=data.get("alert_id", ""),
            resolved_by=data.get("resolved_by", event.get("actor", "unknown")),
            note=data.get("note"),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            return {
                "total_alerts": len(self._alerts),
                "firing": len(self._by_state[AlertState.FIRING]),
                "acknowledged": len(self._by_state[AlertState.ACKNOWLEDGED]),
                "resolved": len(self._by_state[AlertState.RESOLVED]),
                "expired": len(self._by_state[AlertState.EXPIRED]),
                "unique_metrics": len(self._by_metric),
            }

    def _find_existing_alert(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]]
    ) -> Optional[Alert]:
        """Find existing firing alert for metric/labels."""
        for alert_id in self._by_state[AlertState.FIRING]:
            alert = self._alerts.get(alert_id)
            if alert and alert.metric_name == metric_name:
                if labels is None or alert.labels == labels:
                    return alert
        return None

    def _severity_to_priority(self, severity: AnomalySeverity) -> AlertPriority:
        """Map severity to priority."""
        mapping = {
            AnomalySeverity.INFO: AlertPriority.P4,
            AnomalySeverity.WARNING: AlertPriority.P3,
            AnomalySeverity.CRITICAL: AlertPriority.P1,
        }
        return mapping.get(severity, AlertPriority.P3)

    def _notify_state_change(
        self,
        alert: Alert,
        old_state: AlertState,
        new_state: AlertState
    ) -> None:
        """Notify state change callbacks."""
        for callback in self._on_state_change:
            try:
                callback(alert, old_state, new_state)
            except Exception:
                pass

    def _emit_state_change_event(
        self,
        alert: Alert,
        old_state: Optional[AlertState],
        new_state: AlertState
    ) -> str:
        """Emit state change event to bus."""
        event_id = str(uuid.uuid4())
        level = (
            "error" if new_state == AlertState.FIRING and alert.severity == AnomalySeverity.CRITICAL
            else "warn" if new_state == AlertState.FIRING
            else "info"
        )

        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": self.BUS_TOPICS["state_change"],
            "kind": "alert",
            "level": level,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": {
                "alert_id": alert.alert_id,
                "metric_name": alert.metric_name,
                "old_state": old_state.value if old_state else None,
                "new_state": new_state.value,
                "severity": alert.severity.value,
                "priority": alert.priority.value,
                "message": alert.message,
            },
        }

        with open(self._bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id


# Singleton instance
_manager: Optional[AlertManager] = None


def get_manager() -> AlertManager:
    """Get or create the alert manager singleton.

    Returns:
        AlertManager instance
    """
    global _manager
    if _manager is None:
        _manager = AlertManager()
    return _manager


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alert Manager (Step 259)")
    parser.add_argument("--create", metavar="METRIC", help="Create alert for metric")
    parser.add_argument("--severity", default="warning", help="Alert severity")
    parser.add_argument("--message", default="Alert", help="Alert message")
    parser.add_argument("--ack", metavar="ID", help="Acknowledge alert")
    parser.add_argument("--resolve", metavar="ID", help="Resolve alert")
    parser.add_argument("--by", default="cli", help="Actor name")
    parser.add_argument("--note", help="Note")
    parser.add_argument("--list", action="store_true", help="List active alerts")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    manager = get_manager()

    if args.create:
        alert = manager.create(
            metric_name=args.create,
            severity=AnomalySeverity(args.severity),
            message=args.message,
        )
        if args.json:
            print(json.dumps(alert.to_dict(), indent=2))
        else:
            print(f"Created alert: {alert.alert_id}")
            print(f"  Metric: {alert.metric_name}")
            print(f"  Severity: {alert.severity.value}")
            print(f"  Priority: {alert.priority.value}")

    if args.ack:
        success = manager.acknowledge(args.ack, args.by, args.note)
        print(f"Acknowledged: {success}")

    if args.resolve:
        success = manager.resolve(args.resolve, args.by, args.note)
        print(f"Resolved: {success}")

    if args.list:
        alerts = manager.get_active()
        if args.json:
            print(json.dumps([a.to_dict() for a in alerts], indent=2))
        else:
            print(f"Active Alerts ({len(alerts)}):")
            for a in alerts:
                print(f"  [{a.state.value}] {a.alert_id}: {a.metric_name} ({a.severity.value})")

    if args.stats:
        stats = manager.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Alert Manager Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
