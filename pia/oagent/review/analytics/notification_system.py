#!/usr/bin/env python3
"""
Review Notification System (Step 178)

Alerts and notifications for review events.

PBTSO Phase: DISTRIBUTE
Bus Topics: review.notification.send, review.notification.deliver

Features:
- Multi-channel delivery
- Priority-based routing
- Notification preferences
- Digest/batching support
- Delivery tracking

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


# ============================================================================
# Types
# ============================================================================

class NotificationChannel(Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"


class NotificationPriority(Enum):
    """Notification priority levels."""
    CRITICAL = "critical"   # Immediate delivery
    HIGH = "high"           # Within minutes
    NORMAL = "normal"       # Standard delivery
    LOW = "low"             # Can be batched/delayed


class NotificationType(Enum):
    """Types of notifications."""
    REVIEW_STARTED = "review_started"
    REVIEW_COMPLETED = "review_completed"
    REVIEW_ASSIGNED = "review_assigned"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    ISSUES_FOUND = "issues_found"
    SECURITY_ALERT = "security_alert"
    DEBT_THRESHOLD = "debt_threshold"
    REPORT_READY = "report_ready"
    SYSTEM_ALERT = "system_alert"


class DeliveryStatus(Enum):
    """Notification delivery status."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NotificationConfig:
    """Configuration for the notification system."""
    default_channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.IN_APP]
    )
    batch_interval_seconds: int = 300  # 5 minutes
    max_batch_size: int = 50
    retry_attempts: int = 3
    retry_delay_seconds: int = 30
    digest_enabled: bool = True
    quiet_hours_start: Optional[int] = None  # Hour (0-23)
    quiet_hours_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "default_channels": [c.value for c in self.default_channels],
        }


@dataclass
class NotificationPreference:
    """User notification preferences."""
    user_id: str
    channels: Dict[NotificationType, List[NotificationChannel]] = field(default_factory=dict)
    muted_types: List[NotificationType] = field(default_factory=list)
    digest_enabled: bool = True
    quiet_hours: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "channels": {t.value: [c.value for c in cs] for t, cs in self.channels.items()},
            "muted_types": [t.value for t in self.muted_types],
            "digest_enabled": self.digest_enabled,
            "quiet_hours": self.quiet_hours,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotificationPreference":
        """Create from dictionary."""
        data = data.copy()
        data["channels"] = {
            NotificationType(t): [NotificationChannel(c) for c in cs]
            for t, cs in data.get("channels", {}).items()
        }
        data["muted_types"] = [NotificationType(t) for t in data.get("muted_types", [])]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Notification:
    """
    A notification to be delivered.

    Attributes:
        notification_id: Unique identifier
        notification_type: Type of notification
        priority: Delivery priority
        title: Notification title
        body: Notification body
        data: Additional data
        recipients: Target recipients
        channels: Delivery channels
        created_at: Creation timestamp
        delivered_at: Delivery timestamp
        delivery_status: Status per channel
    """
    notification_id: str
    notification_type: NotificationType
    priority: NotificationPriority
    title: str
    body: str
    data: Dict[str, Any] = field(default_factory=dict)
    recipients: List[str] = field(default_factory=list)
    channels: List[NotificationChannel] = field(default_factory=list)
    created_at: str = ""
    delivered_at: Optional[str] = None
    delivery_status: Dict[str, DeliveryStatus] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "notification_id": self.notification_id,
            "notification_type": self.notification_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "body": self.body,
            "data": self.data,
            "recipients": self.recipients,
            "channels": [c.value for c in self.channels],
            "created_at": self.created_at,
            "delivered_at": self.delivered_at,
            "delivery_status": {k: v.value for k, v in self.delivery_status.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Notification":
        """Create from dictionary."""
        data = data.copy()
        data["notification_type"] = NotificationType(data["notification_type"])
        data["priority"] = NotificationPriority(data["priority"])
        data["channels"] = [NotificationChannel(c) for c in data.get("channels", [])]
        data["delivery_status"] = {
            k: DeliveryStatus(v) for k, v in data.get("delivery_status", {}).items()
        }
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DeliveryResult:
    """Result of notification delivery."""
    notification_id: str
    channel: NotificationChannel
    recipient: str
    success: bool
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "channel": self.channel.value,
        }


@dataclass
class NotificationDigest:
    """A digest of multiple notifications."""
    digest_id: str
    user_id: str
    notifications: List[Notification]
    period_start: str
    period_end: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "digest_id": self.digest_id,
            "user_id": self.user_id,
            "notifications": [n.to_dict() for n in self.notifications],
            "notification_count": len(self.notifications),
            "period_start": self.period_start,
            "period_end": self.period_end,
        }

    def to_summary(self) -> str:
        """Generate digest summary."""
        lines = [
            f"# Notification Digest",
            f"",
            f"**Period:** {self.period_start} to {self.period_end}",
            f"**Notifications:** {len(self.notifications)}",
            "",
        ]

        # Group by type
        by_type: Dict[str, List[Notification]] = {}
        for n in self.notifications:
            t = n.notification_type.value
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(n)

        for ntype, notifs in sorted(by_type.items()):
            lines.append(f"## {ntype.replace('_', ' ').title()} ({len(notifs)})")
            lines.append("")
            for n in notifs[:5]:
                lines.append(f"- **{n.title}**")
                if n.body:
                    lines.append(f"  {n.body[:100]}...")
            if len(notifs) > 5:
                lines.append(f"- _... and {len(notifs) - 5} more_")
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# Notification Store
# ============================================================================

class NotificationStore:
    """Persistent store for notifications."""

    def __init__(self, store_path: Path):
        """Initialize the store."""
        self.store_path = store_path
        self._ensure_store()

    def _ensure_store(self) -> None:
        """Ensure store file exists."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.store_path.exists():
            self._write_store({
                "notifications": [],
                "preferences": [],
                "version": 1,
            })

    def _read_store(self) -> Dict[str, Any]:
        """Read store with file locking."""
        with open(self.store_path, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _write_store(self, data: Dict[str, Any]) -> None:
        """Write store with file locking."""
        with open(self.store_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def add_notification(self, notification: Notification) -> None:
        """Add a notification."""
        data = self._read_store()
        data["notifications"].append(notification.to_dict())
        self._write_store(data)

    def get_notification(self, notification_id: str) -> Optional[Notification]:
        """Get a notification by ID."""
        data = self._read_store()
        for n in data["notifications"]:
            if n["notification_id"] == notification_id:
                return Notification.from_dict(n)
        return None

    def get_pending(self, recipient: Optional[str] = None) -> List[Notification]:
        """Get pending notifications."""
        data = self._read_store()
        notifications = []

        for n_data in data["notifications"]:
            n = Notification.from_dict(n_data)
            # Check if any channel is still pending
            if not n.delivery_status or DeliveryStatus.PENDING in n.delivery_status.values():
                if recipient is None or recipient in n.recipients:
                    notifications.append(n)

        return notifications

    def update_notification(self, notification: Notification) -> bool:
        """Update a notification."""
        data = self._read_store()
        for i, n in enumerate(data["notifications"]):
            if n["notification_id"] == notification.notification_id:
                data["notifications"][i] = notification.to_dict()
                self._write_store(data)
                return True
        return False

    def get_preferences(self, user_id: str) -> Optional[NotificationPreference]:
        """Get user preferences."""
        data = self._read_store()
        for p in data.get("preferences", []):
            if p["user_id"] == user_id:
                return NotificationPreference.from_dict(p)
        return None

    def set_preferences(self, preferences: NotificationPreference) -> None:
        """Set user preferences."""
        data = self._read_store()
        data["preferences"] = [
            p for p in data.get("preferences", [])
            if p["user_id"] != preferences.user_id
        ]
        data["preferences"].append(preferences.to_dict())
        self._write_store(data)


# ============================================================================
# Channel Senders
# ============================================================================

class BaseSender:
    """Base class for notification senders."""

    async def send(
        self,
        notification: Notification,
        recipient: str,
    ) -> DeliveryResult:
        """Send notification to recipient."""
        raise NotImplementedError


class InAppSender(BaseSender):
    """In-app notification sender."""

    async def send(
        self,
        notification: Notification,
        recipient: str,
    ) -> DeliveryResult:
        """Store in-app notification."""
        # In-app notifications are stored in the notification store
        return DeliveryResult(
            notification_id=notification.notification_id,
            channel=NotificationChannel.IN_APP,
            recipient=recipient,
            success=True,
        )


class EmailSender(BaseSender):
    """Email notification sender."""

    async def send(
        self,
        notification: Notification,
        recipient: str,
    ) -> DeliveryResult:
        """Send email notification."""
        # Mock email sending
        return DeliveryResult(
            notification_id=notification.notification_id,
            channel=NotificationChannel.EMAIL,
            recipient=recipient,
            success=True,
        )


class SlackSender(BaseSender):
    """Slack notification sender."""

    async def send(
        self,
        notification: Notification,
        recipient: str,
    ) -> DeliveryResult:
        """Send Slack notification."""
        # Mock Slack sending
        return DeliveryResult(
            notification_id=notification.notification_id,
            channel=NotificationChannel.SLACK,
            recipient=recipient,
            success=True,
        )


class WebhookSender(BaseSender):
    """Webhook notification sender."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send(
        self,
        notification: Notification,
        recipient: str,
    ) -> DeliveryResult:
        """Send webhook notification."""
        # Mock webhook sending
        return DeliveryResult(
            notification_id=notification.notification_id,
            channel=NotificationChannel.WEBHOOK,
            recipient=recipient,
            success=True,
        )


# ============================================================================
# Notification System
# ============================================================================

class NotificationSystem:
    """
    Notification system for review alerts.

    Example:
        system = NotificationSystem()

        # Send notification
        notification = await system.notify(
            notification_type=NotificationType.REVIEW_COMPLETED,
            title="Review Completed",
            body="Your code review is complete.",
            recipients=["user@example.com"],
            priority=NotificationPriority.NORMAL,
        )

        # Check delivery status
        print(notification.delivery_status)
    """

    BUS_TOPICS = {
        "send": "review.notification.send",
        "deliver": "review.notification.deliver",
    }

    SENDERS = {
        NotificationChannel.IN_APP: InAppSender,
        NotificationChannel.EMAIL: EmailSender,
        NotificationChannel.SLACK: SlackSender,
    }

    def __init__(
        self,
        config: Optional[NotificationConfig] = None,
        bus_path: Optional[Path] = None,
        store_path: Optional[Path] = None,
    ):
        """
        Initialize the notification system.

        Args:
            config: System configuration
            bus_path: Path to event bus file
            store_path: Path to notification store
        """
        self.config = config or NotificationConfig()
        self.bus_path = bus_path or self._get_bus_path()
        self.store = NotificationStore(store_path or self._get_store_path())
        self._batch_queue: List[Notification] = []
        self._webhook_sender: Optional[WebhookSender] = None

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _get_store_path(self) -> Path:
        """Get path to notification store."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        data_dir = pluribus_root / ".pluribus" / "review" / "data"
        return data_dir / "notifications.json"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "notification") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "notification-system",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def configure_webhook(self, webhook_url: str) -> None:
        """Configure webhook sender."""
        self._webhook_sender = WebhookSender(webhook_url)
        self.SENDERS[NotificationChannel.WEBHOOK] = lambda: self._webhook_sender

    async def notify(
        self,
        notification_type: NotificationType,
        title: str,
        body: str,
        recipients: List[str],
        priority: NotificationPriority = NotificationPriority.NORMAL,
        data: Optional[Dict[str, Any]] = None,
        channels: Optional[List[NotificationChannel]] = None,
    ) -> Notification:
        """
        Send a notification.

        Args:
            notification_type: Type of notification
            title: Notification title
            body: Notification body
            recipients: Target recipients
            priority: Delivery priority
            data: Additional data
            channels: Delivery channels (uses defaults if not specified)

        Returns:
            Created notification

        Emits:
            review.notification.send
            review.notification.deliver
        """
        notification_id = str(uuid.uuid4())[:8]

        # Determine channels
        if channels is None:
            channels = self.config.default_channels

        notification = Notification(
            notification_id=notification_id,
            notification_type=notification_type,
            priority=priority,
            title=title,
            body=body,
            data=data or {},
            recipients=recipients,
            channels=channels,
        )

        # Initialize delivery status
        for channel in channels:
            notification.delivery_status[channel.value] = DeliveryStatus.PENDING

        # Store notification
        self.store.add_notification(notification)

        self._emit_event(self.BUS_TOPICS["send"], {
            "notification_id": notification_id,
            "notification_type": notification_type.value,
            "priority": priority.value,
            "recipients": recipients,
            "channels": [c.value for c in channels],
        })

        # Check if should batch (low priority, non-critical)
        if (priority == NotificationPriority.LOW and
                self.config.digest_enabled and
                notification_type not in [
                    NotificationType.SECURITY_ALERT,
                    NotificationType.SYSTEM_ALERT,
                ]):
            self._batch_queue.append(notification)
            return notification

        # Deliver immediately
        await self._deliver(notification)

        return notification

    async def _deliver(self, notification: Notification) -> List[DeliveryResult]:
        """Deliver notification to all channels and recipients."""
        results = []

        for channel in notification.channels:
            sender_class = self.SENDERS.get(channel)
            if not sender_class:
                continue

            sender = sender_class() if callable(sender_class) else sender_class

            for recipient in notification.recipients:
                # Check user preferences
                prefs = self.store.get_preferences(recipient)
                if prefs:
                    if notification.notification_type in prefs.muted_types:
                        results.append(DeliveryResult(
                            notification_id=notification.notification_id,
                            channel=channel,
                            recipient=recipient,
                            success=True,
                            error="Muted by user preference",
                        ))
                        notification.delivery_status[channel.value] = DeliveryStatus.SKIPPED
                        continue

                # Deliver
                try:
                    result = await sender.send(notification, recipient)
                    results.append(result)

                    if result.success:
                        notification.delivery_status[channel.value] = DeliveryStatus.DELIVERED
                    else:
                        notification.delivery_status[channel.value] = DeliveryStatus.FAILED

                except Exception as e:
                    results.append(DeliveryResult(
                        notification_id=notification.notification_id,
                        channel=channel,
                        recipient=recipient,
                        success=False,
                        error=str(e),
                    ))
                    notification.delivery_status[channel.value] = DeliveryStatus.FAILED

        notification.delivered_at = datetime.now(timezone.utc).isoformat() + "Z"
        self.store.update_notification(notification)

        self._emit_event(self.BUS_TOPICS["deliver"], {
            "notification_id": notification.notification_id,
            "delivered": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
        })

        return results

    async def send_digest(self, user_id: str) -> Optional[NotificationDigest]:
        """
        Send a digest of batched notifications.

        Args:
            user_id: Target user

        Returns:
            NotificationDigest if notifications were batched
        """
        user_notifications = [
            n for n in self._batch_queue
            if user_id in n.recipients
        ]

        if not user_notifications:
            return None

        # Remove from queue
        self._batch_queue = [
            n for n in self._batch_queue
            if user_id not in n.recipients
        ]

        digest = NotificationDigest(
            digest_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            notifications=user_notifications,
            period_start=user_notifications[0].created_at,
            period_end=datetime.now(timezone.utc).isoformat() + "Z",
        )

        # Send digest as single notification
        await self.notify(
            notification_type=NotificationType.SYSTEM_ALERT,
            title=f"Notification Digest ({len(user_notifications)} items)",
            body=digest.to_summary(),
            recipients=[user_id],
            priority=NotificationPriority.NORMAL,
            data={"digest_id": digest.digest_id},
        )

        return digest

    def get_pending(self, recipient: Optional[str] = None) -> List[Notification]:
        """Get pending notifications."""
        return self.store.get_pending(recipient)

    def set_preferences(
        self,
        user_id: str,
        preferences: NotificationPreference,
    ) -> None:
        """Set user notification preferences."""
        self.store.set_preferences(preferences)

    def get_preferences(self, user_id: str) -> Optional[NotificationPreference]:
        """Get user notification preferences."""
        return self.store.get_preferences(user_id)


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Notification System."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Notification System (Step 178)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Send command
    send_parser = subparsers.add_parser("send", help="Send notification")
    send_parser.add_argument("--type", required=True,
                             choices=[t.value for t in NotificationType])
    send_parser.add_argument("--title", required=True)
    send_parser.add_argument("--body", required=True)
    send_parser.add_argument("--recipient", required=True)
    send_parser.add_argument("--priority", default="normal",
                             choices=[p.value for p in NotificationPriority])
    send_parser.add_argument("--channel", action="append",
                             choices=[c.value for c in NotificationChannel])

    # List command
    list_parser = subparsers.add_parser("list", help="List pending notifications")
    list_parser.add_argument("--recipient")

    # Preferences command
    prefs_parser = subparsers.add_parser("prefs", help="Manage preferences")
    prefs_parser.add_argument("user_id")
    prefs_parser.add_argument("--mute", action="append",
                              choices=[t.value for t in NotificationType])

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    system = NotificationSystem()

    if args.command == "send":
        channels = None
        if args.channel:
            channels = [NotificationChannel(c) for c in args.channel]

        notification = asyncio.run(system.notify(
            notification_type=NotificationType(args.type),
            title=args.title,
            body=args.body,
            recipients=[args.recipient],
            priority=NotificationPriority(args.priority),
            channels=channels,
        ))

        if args.json:
            print(json.dumps(notification.to_dict(), indent=2))
        else:
            print(f"Sent notification: {notification.notification_id}")
            print(f"  Type: {notification.notification_type.value}")
            print(f"  Status: {notification.delivery_status}")

    elif args.command == "list":
        notifications = system.get_pending(args.recipient)

        if args.json:
            print(json.dumps([n.to_dict() for n in notifications], indent=2))
        else:
            print(f"Found {len(notifications)} pending notifications:")
            for n in notifications:
                print(f"  [{n.notification_id}] {n.title}")
                print(f"    Type: {n.notification_type.value}")
                print(f"    Recipients: {', '.join(n.recipients)}")

    elif args.command == "prefs":
        if args.mute:
            prefs = NotificationPreference(
                user_id=args.user_id,
                muted_types=[NotificationType(t) for t in args.mute],
            )
            system.set_preferences(args.user_id, prefs)
            print(f"Updated preferences for {args.user_id}")
        else:
            prefs = system.get_preferences(args.user_id)
            if prefs:
                if args.json:
                    print(json.dumps(prefs.to_dict(), indent=2))
                else:
                    print(f"Preferences for {args.user_id}:")
                    print(f"  Muted: {[t.value for t in prefs.muted_types]}")
                    print(f"  Digest: {prefs.digest_enabled}")
            else:
                print(f"No preferences found for {args.user_id}")

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
