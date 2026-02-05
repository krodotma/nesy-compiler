#!/usr/bin/env python3
"""
Monitor Notification System - Step 278

Alert routing and notification delivery.

PBTSO Phase: DISTRIBUTE

Bus Topics:
- monitor.notification.send (subscribed)
- monitor.notification.delivered (emitted)
- monitor.notification.failed (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set


class NotificationChannel(Enum):
    """Notification delivery channels."""
    BUS = "bus"
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SMS = "sms"
    TEAMS = "teams"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationStatus(Enum):
    """Notification status."""
    PENDING = "pending"
    SENDING = "sending"
    DELIVERED = "delivered"
    FAILED = "failed"
    SUPPRESSED = "suppressed"


@dataclass
class NotificationRecipient:
    """A notification recipient.

    Attributes:
        recipient_id: Unique recipient ID
        name: Recipient name
        channels: Available channels with configs
        schedule: On-call schedule
        escalation_level: Escalation level
    """
    recipient_id: str
    name: str
    channels: Dict[NotificationChannel, Dict[str, Any]] = field(default_factory=dict)
    schedule: Dict[str, Any] = field(default_factory=dict)
    escalation_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recipient_id": self.recipient_id,
            "name": self.name,
            "channels": {c.value: v for c, v in self.channels.items()},
            "schedule": self.schedule,
            "escalation_level": self.escalation_level,
        }


@dataclass
class RoutingRule:
    """A notification routing rule.

    Attributes:
        rule_id: Unique rule ID
        name: Rule name
        conditions: Matching conditions
        recipients: Target recipients
        channels: Target channels
        priority: Notification priority
        suppress_duplicates: Whether to suppress duplicates
        dedupe_window_s: Deduplication window
        enabled: Whether rule is enabled
    """
    rule_id: str
    name: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    recipients: List[str] = field(default_factory=list)
    channels: List[NotificationChannel] = field(default_factory=list)
    priority: NotificationPriority = NotificationPriority.NORMAL
    suppress_duplicates: bool = True
    dedupe_window_s: int = 300
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "conditions": self.conditions,
            "recipients": self.recipients,
            "channels": [c.value for c in self.channels],
            "priority": self.priority.value,
            "suppress_duplicates": self.suppress_duplicates,
            "dedupe_window_s": self.dedupe_window_s,
            "enabled": self.enabled,
        }


@dataclass
class Notification:
    """A notification to be sent.

    Attributes:
        notification_id: Unique notification ID
        title: Notification title
        message: Notification message
        source: Notification source
        severity: Severity level
        priority: Priority level
        labels: Additional labels
        data: Additional data
        channels: Target channels
        recipients: Target recipients
        status: Notification status
        created_at: Creation timestamp
        sent_at: Sent timestamp
        delivery_results: Results per channel
    """
    notification_id: str
    title: str
    message: str
    source: str = "monitor-agent"
    severity: str = "info"
    priority: NotificationPriority = NotificationPriority.NORMAL
    labels: Dict[str, str] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    channels: List[NotificationChannel] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: float = field(default_factory=time.time)
    sent_at: Optional[float] = None
    delivery_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "notification_id": self.notification_id,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "severity": self.severity,
            "priority": self.priority.value,
            "labels": self.labels,
            "data": self.data,
            "channels": [c.value for c in self.channels],
            "recipients": self.recipients,
            "status": self.status.value,
            "created_at": self.created_at,
            "sent_at": self.sent_at,
            "delivery_results": self.delivery_results,
        }


class NotificationSystem:
    """
    Notification routing and delivery system.

    The system:
    - Routes alerts to appropriate channels
    - Manages notification recipients
    - Handles deduplication and suppression
    - Tracks delivery status

    Example:
        system = NotificationSystem()

        # Add recipient
        system.add_recipient(NotificationRecipient(
            recipient_id="oncall-team",
            name="On-Call Team",
            channels={
                NotificationChannel.SLACK: {"channel": "#alerts"},
                NotificationChannel.PAGERDUTY: {"service_id": "..."},
            }
        ))

        # Add routing rule
        system.add_routing_rule(RoutingRule(
            rule_id="critical-alerts",
            name="Critical Alerts",
            conditions={"severity": "critical"},
            recipients=["oncall-team"],
            channels=[NotificationChannel.PAGERDUTY],
        ))

        # Send notification
        result = await system.notify(
            title="High CPU",
            message="CPU usage exceeds 90%",
            severity="critical",
        )
    """

    BUS_TOPICS = {
        "send": "monitor.notification.send",
        "delivered": "monitor.notification.delivered",
        "failed": "monitor.notification.failed",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        bus_dir: Optional[str] = None,
    ):
        """Initialize notification system.

        Args:
            bus_dir: Bus directory
        """
        self._recipients: Dict[str, NotificationRecipient] = {}
        self._rules: Dict[str, RoutingRule] = {}
        self._notifications: Dict[str, Notification] = {}
        self._notification_history: List[Notification] = []
        self._channel_handlers: Dict[NotificationChannel, Callable] = {}
        self._dedupe_cache: Dict[str, float] = {}
        self._last_heartbeat = time.time()

        # Suppression state
        self._suppressed_sources: Set[str] = set()

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register default handlers
        self._register_default_handlers()

    def add_recipient(self, recipient: NotificationRecipient) -> None:
        """Add a notification recipient.

        Args:
            recipient: Recipient to add
        """
        self._recipients[recipient.recipient_id] = recipient

    def remove_recipient(self, recipient_id: str) -> bool:
        """Remove a recipient.

        Args:
            recipient_id: Recipient ID

        Returns:
            True if removed
        """
        if recipient_id in self._recipients:
            del self._recipients[recipient_id]
            return True
        return False

    def get_recipient(self, recipient_id: str) -> Optional[NotificationRecipient]:
        """Get a recipient by ID.

        Args:
            recipient_id: Recipient ID

        Returns:
            Recipient or None
        """
        return self._recipients.get(recipient_id)

    def list_recipients(self) -> List[Dict[str, Any]]:
        """List all recipients.

        Returns:
            Recipient summaries
        """
        return [r.to_dict() for r in self._recipients.values()]

    def add_routing_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule.

        Args:
            rule: Routing rule
        """
        self._rules[rule.rule_id] = rule

    def remove_routing_rule(self, rule_id: str) -> bool:
        """Remove a routing rule.

        Args:
            rule_id: Rule ID

        Returns:
            True if removed
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def get_routing_rule(self, rule_id: str) -> Optional[RoutingRule]:
        """Get a routing rule by ID.

        Args:
            rule_id: Rule ID

        Returns:
            Rule or None
        """
        return self._rules.get(rule_id)

    def list_routing_rules(self) -> List[Dict[str, Any]]:
        """List all routing rules.

        Returns:
            Rule summaries
        """
        return [r.to_dict() for r in self._rules.values()]

    def register_channel_handler(
        self,
        channel: NotificationChannel,
        handler: Callable[..., Coroutine[Any, Any, bool]],
    ) -> None:
        """Register a channel handler.

        Args:
            channel: Notification channel
            handler: Async handler function
        """
        self._channel_handlers[channel] = handler

    async def notify(
        self,
        title: str,
        message: str,
        severity: str = "info",
        source: str = "monitor-agent",
        labels: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        channels: Optional[List[NotificationChannel]] = None,
        recipients: Optional[List[str]] = None,
    ) -> Notification:
        """Send a notification.

        Args:
            title: Notification title
            message: Notification message
            severity: Severity level
            source: Notification source
            labels: Additional labels
            data: Additional data
            channels: Override channels
            recipients: Override recipients

        Returns:
            Notification
        """
        notification = Notification(
            notification_id=f"notif-{uuid.uuid4().hex[:8]}",
            title=title,
            message=message,
            source=source,
            severity=severity,
            labels=labels or {},
            data=data or {},
        )

        # Check suppression
        if source in self._suppressed_sources:
            notification.status = NotificationStatus.SUPPRESSED
            self._store_notification(notification)
            return notification

        # Route notification
        if channels:
            notification.channels = channels
        if recipients:
            notification.recipients = recipients

        if not notification.channels or not notification.recipients:
            self._route_notification(notification)

        # Check deduplication
        if self._is_duplicate(notification):
            notification.status = NotificationStatus.SUPPRESSED
            self._store_notification(notification)
            return notification

        # Send notification
        await self._send_notification(notification)

        return notification

    async def notify_alert(
        self,
        alert_id: str,
        title: str,
        message: str,
        severity: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Notification:
        """Send alert notification.

        Args:
            alert_id: Alert ID
            title: Alert title
            message: Alert message
            severity: Alert severity
            labels: Alert labels

        Returns:
            Notification
        """
        return await self.notify(
            title=title,
            message=message,
            severity=severity,
            source="alert-manager",
            labels=labels or {},
            data={"alert_id": alert_id},
        )

    def suppress_source(self, source: str, duration_s: int = 3600) -> None:
        """Suppress notifications from a source.

        Args:
            source: Source to suppress
            duration_s: Suppression duration
        """
        self._suppressed_sources.add(source)

        # Schedule unsuppression
        asyncio.get_event_loop().call_later(
            duration_s,
            lambda: self._suppressed_sources.discard(source)
        )

    def unsuppress_source(self, source: str) -> None:
        """Remove suppression for a source.

        Args:
            source: Source to unsuppress
        """
        self._suppressed_sources.discard(source)

    def get_notification(self, notification_id: str) -> Optional[Notification]:
        """Get a notification by ID.

        Args:
            notification_id: Notification ID

        Returns:
            Notification or None
        """
        return self._notifications.get(notification_id)

    def get_notification_history(
        self,
        limit: int = 50,
        status: Optional[NotificationStatus] = None,
    ) -> List[Notification]:
        """Get notification history.

        Args:
            limit: Maximum results
            status: Filter by status

        Returns:
            Notifications
        """
        history = self._notification_history

        if status:
            history = [n for n in history if n.status == status]

        return list(reversed(history[-limit:]))

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics.

        Returns:
            Statistics
        """
        by_status: Dict[str, int] = {}
        by_channel: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for n in self._notification_history:
            status = n.status.value
            by_status[status] = by_status.get(status, 0) + 1

            sev = n.severity
            by_severity[sev] = by_severity.get(sev, 0) + 1

            for channel in n.channels:
                ch = channel.value
                by_channel[ch] = by_channel.get(ch, 0) + 1

        recent = self._notification_history[-100:]
        delivered = sum(1 for n in recent if n.status == NotificationStatus.DELIVERED)
        delivery_rate = delivered / len(recent) if recent else 0.0

        return {
            "total_notifications": len(self._notification_history),
            "recipients": len(self._recipients),
            "routing_rules": len(self._rules),
            "suppressed_sources": len(self._suppressed_sources),
            "by_status": by_status,
            "by_channel": by_channel,
            "by_severity": by_severity,
            "recent_delivery_rate": delivery_rate,
        }

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "notification_system",
                "status": "healthy",
                "notifications": len(self._notification_history),
            }
        )

        return True

    def _route_notification(self, notification: Notification) -> None:
        """Route notification based on rules.

        Args:
            notification: Notification to route
        """
        for rule in self._rules.values():
            if not rule.enabled:
                continue

            if self._matches_conditions(notification, rule.conditions):
                notification.recipients.extend(rule.recipients)
                notification.channels.extend(rule.channels)
                notification.priority = max(
                    notification.priority,
                    rule.priority,
                    key=lambda p: list(NotificationPriority).index(p)
                )

        # Remove duplicates
        notification.recipients = list(set(notification.recipients))
        notification.channels = list(set(notification.channels))

        # Default to bus if no channels
        if not notification.channels:
            notification.channels = [NotificationChannel.BUS]

    def _matches_conditions(
        self,
        notification: Notification,
        conditions: Dict[str, Any],
    ) -> bool:
        """Check if notification matches conditions.

        Args:
            notification: Notification to check
            conditions: Conditions to match

        Returns:
            True if matches
        """
        if not conditions:
            return True

        if "severity" in conditions:
            severities = conditions["severity"]
            if isinstance(severities, str):
                severities = [severities]
            if notification.severity not in severities:
                return False

        if "source" in conditions:
            if notification.source != conditions["source"]:
                return False

        if "labels" in conditions:
            for k, v in conditions["labels"].items():
                if notification.labels.get(k) != v:
                    return False

        return True

    def _is_duplicate(self, notification: Notification) -> bool:
        """Check if notification is a duplicate.

        Args:
            notification: Notification to check

        Returns:
            True if duplicate
        """
        # Create dedupe key
        key = f"{notification.source}:{notification.title}:{notification.severity}"

        now = time.time()

        # Check cache
        if key in self._dedupe_cache:
            last_sent = self._dedupe_cache[key]
            # Find matching rule for dedupe window
            window = 300  # Default 5 minutes
            for rule in self._rules.values():
                if rule.suppress_duplicates and self._matches_conditions(notification, rule.conditions):
                    window = rule.dedupe_window_s
                    break

            if now - last_sent < window:
                return True

        # Update cache
        self._dedupe_cache[key] = now

        # Prune old cache entries
        cutoff = now - 3600
        self._dedupe_cache = {
            k: v for k, v in self._dedupe_cache.items()
            if v >= cutoff
        }

        return False

    async def _send_notification(self, notification: Notification) -> None:
        """Send notification to all channels.

        Args:
            notification: Notification to send
        """
        notification.status = NotificationStatus.SENDING
        notification.sent_at = time.time()

        all_success = True

        for channel in notification.channels:
            handler = self._channel_handlers.get(channel)
            if not handler:
                notification.delivery_results[channel.value] = {
                    "success": False,
                    "error": "No handler registered",
                }
                all_success = False
                continue

            try:
                success = await handler(notification, channel)
                notification.delivery_results[channel.value] = {
                    "success": success,
                }
                if not success:
                    all_success = False
            except Exception as e:
                notification.delivery_results[channel.value] = {
                    "success": False,
                    "error": str(e),
                }
                all_success = False

        notification.status = (
            NotificationStatus.DELIVERED if all_success
            else NotificationStatus.FAILED
        )

        self._store_notification(notification)

        # Emit bus event
        topic = (
            self.BUS_TOPICS["delivered"] if all_success
            else self.BUS_TOPICS["failed"]
        )

        self._emit_bus_event(
            topic,
            {
                "notification_id": notification.notification_id,
                "title": notification.title,
                "channels": [c.value for c in notification.channels],
                "success": all_success,
            }
        )

    def _store_notification(self, notification: Notification) -> None:
        """Store notification in history.

        Args:
            notification: Notification to store
        """
        self._notifications[notification.notification_id] = notification
        self._notification_history.append(notification)

        if len(self._notification_history) > 1000:
            self._notification_history = self._notification_history[-1000:]

    def _register_default_handlers(self) -> None:
        """Register default channel handlers."""

        async def bus_handler(
            notification: Notification,
            channel: NotificationChannel,
        ) -> bool:
            """Handle bus notification."""
            self._emit_bus_event(
                self.BUS_TOPICS["send"],
                {
                    "notification_id": notification.notification_id,
                    "title": notification.title,
                    "message": notification.message,
                    "severity": notification.severity,
                },
                level=notification.severity
            )
            return True

        async def slack_handler(
            notification: Notification,
            channel: NotificationChannel,
        ) -> bool:
            """Handle Slack notification."""
            # Simulate Slack delivery
            return True

        async def email_handler(
            notification: Notification,
            channel: NotificationChannel,
        ) -> bool:
            """Handle email notification."""
            # Simulate email delivery
            return True

        async def webhook_handler(
            notification: Notification,
            channel: NotificationChannel,
        ) -> bool:
            """Handle webhook notification."""
            # Simulate webhook delivery
            return True

        async def pagerduty_handler(
            notification: Notification,
            channel: NotificationChannel,
        ) -> bool:
            """Handle PagerDuty notification."""
            # Simulate PagerDuty delivery
            return True

        self._channel_handlers[NotificationChannel.BUS] = bus_handler
        self._channel_handlers[NotificationChannel.SLACK] = slack_handler
        self._channel_handlers[NotificationChannel.EMAIL] = email_handler
        self._channel_handlers[NotificationChannel.WEBHOOK] = webhook_handler
        self._channel_handlers[NotificationChannel.PAGERDUTY] = pagerduty_handler

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id


# Singleton instance
_system: Optional[NotificationSystem] = None


def get_system() -> NotificationSystem:
    """Get or create the notification system singleton.

    Returns:
        NotificationSystem instance
    """
    global _system
    if _system is None:
        _system = NotificationSystem()
    return _system


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Notification System (Step 278)")
    parser.add_argument("--send", metavar="TITLE", help="Send notification")
    parser.add_argument("--message", default="Test notification", help="Message")
    parser.add_argument("--severity", default="info", help="Severity level")
    parser.add_argument("--recipients", action="store_true", help="List recipients")
    parser.add_argument("--rules", action="store_true", help="List routing rules")
    parser.add_argument("--history", action="store_true", help="Show notification history")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    system = get_system()

    if args.send:
        async def run():
            return await system.notify(
                title=args.send,
                message=args.message,
                severity=args.severity,
            )

        notification = asyncio.run(run())
        if args.json:
            print(json.dumps(notification.to_dict(), indent=2))
        else:
            print(f"Sent notification: {notification.notification_id}")
            print(f"  Status: {notification.status.value}")
            print(f"  Channels: {[c.value for c in notification.channels]}")

    if args.recipients:
        recipients = system.list_recipients()
        if args.json:
            print(json.dumps(recipients, indent=2))
        else:
            print("Recipients:")
            for r in recipients:
                print(f"  [{r['recipient_id']}] {r['name']}")

    if args.rules:
        rules = system.list_routing_rules()
        if args.json:
            print(json.dumps(rules, indent=2))
        else:
            print("Routing Rules:")
            for r in rules:
                enabled = "enabled" if r["enabled"] else "disabled"
                print(f"  [{r['rule_id']}] {r['name']} ({enabled})")

    if args.history:
        history = system.get_notification_history()
        if args.json:
            print(json.dumps([n.to_dict() for n in history], indent=2))
        else:
            print("Notification History:")
            for n in history:
                print(f"  [{n.status.value}] {n.notification_id}: {n.title}")

    if args.stats:
        stats = system.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Notification System Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
