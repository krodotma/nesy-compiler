#!/usr/bin/env python3
"""
notifier.py - Deployment Notification System (Step 225)

PBTSO Phase: ITERATE
A2A Integration: Sends deployment notifications via deploy.notifications.*

Provides:
- NotificationChannel: Notification channels
- NotificationType: Types of notifications
- NotificationPriority: Priority levels
- NotificationTemplate: Notification templates
- Notification: Individual notification
- DeploymentNotificationSystem: Main notification system

Bus Topics:
- deploy.notifications.send
- deploy.notifications.sent
- deploy.notifications.failed
- deploy.notifications.subscribe

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import asyncio
import fcntl
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
import re


# ==============================================================================
# Bus Emission Helper with File Locking
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
    actor: str = "notification-system"
) -> str:
    """Emit an event to the Pluribus bus with file locking."""
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
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class NotificationChannel(Enum):
    """Notification channels."""
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"
    DISCORD = "discord"
    SMS = "sms"
    BUS = "bus"  # Internal bus


class NotificationType(Enum):
    """Types of deployment notifications."""
    DEPLOYMENT_STARTED = "deployment_started"
    DEPLOYMENT_COMPLETED = "deployment_completed"
    DEPLOYMENT_FAILED = "deployment_failed"
    DEPLOYMENT_ROLLED_BACK = "deployment_rolled_back"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"
    HEALTH_CHECK_FAILED = "health_check_failed"
    TRAFFIC_SHIFTED = "traffic_shifted"
    SCHEDULED_DEPLOYMENT = "scheduled_deployment"
    CUSTOM = "custom"


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class NotificationStatus(Enum):
    """Notification status."""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class NotificationTemplate:
    """
    Notification template.

    Attributes:
        template_id: Unique template identifier
        name: Template name
        notification_type: Type of notification
        subject_template: Subject line template
        body_template: Body template
        channels: Target channels
        priority: Default priority
        enabled: Whether template is enabled
    """
    template_id: str
    name: str
    notification_type: NotificationType
    subject_template: str = ""
    body_template: str = ""
    channels: List[NotificationChannel] = field(default_factory=list)
    priority: NotificationPriority = NotificationPriority.NORMAL
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "notification_type": self.notification_type.value,
            "subject_template": self.subject_template,
            "body_template": self.body_template,
            "channels": [c.value for c in self.channels],
            "priority": self.priority.value,
            "enabled": self.enabled,
        }

    def render(self, variables: Dict[str, Any]) -> tuple:
        """Render template with variables."""
        subject = self.subject_template
        body = self.body_template

        for key, value in variables.items():
            placeholder = "{{" + key + "}}"
            subject = subject.replace(placeholder, str(value))
            body = body.replace(placeholder, str(value))

        return subject, body


@dataclass
class Subscriber:
    """
    Notification subscriber.

    Attributes:
        subscriber_id: Unique subscriber identifier
        name: Subscriber name
        channels: Subscribed channels with addresses
        notification_types: Subscribed notification types
        services: Subscribed services (empty = all)
        environments: Subscribed environments
        active: Whether subscriber is active
    """
    subscriber_id: str
    name: str
    channels: Dict[str, str] = field(default_factory=dict)  # channel -> address
    notification_types: List[NotificationType] = field(default_factory=list)
    services: List[str] = field(default_factory=list)
    environments: List[str] = field(default_factory=list)
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subscriber_id": self.subscriber_id,
            "name": self.name,
            "channels": self.channels,
            "notification_types": [t.value for t in self.notification_types],
            "services": self.services,
            "environments": self.environments,
            "active": self.active,
        }


@dataclass
class Notification:
    """
    Individual notification.

    Attributes:
        notification_id: Unique notification identifier
        notification_type: Type of notification
        subject: Notification subject
        body: Notification body
        priority: Priority level
        channels: Target channels
        recipients: Recipients list
        service_name: Associated service
        deployment_id: Associated deployment
        environment: Target environment
        status: Notification status
        created_at: Creation timestamp
        sent_at: Send timestamp
        metadata: Additional metadata
    """
    notification_id: str
    notification_type: NotificationType
    subject: str = ""
    body: str = ""
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: List[NotificationChannel] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    service_name: str = ""
    deployment_id: str = ""
    environment: str = ""
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: float = field(default_factory=time.time)
    sent_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notification_id": self.notification_id,
            "notification_type": self.notification_type.value,
            "subject": self.subject,
            "body": self.body,
            "priority": self.priority.value,
            "channels": [c.value for c in self.channels],
            "recipients": self.recipients,
            "service_name": self.service_name,
            "deployment_id": self.deployment_id,
            "environment": self.environment,
            "status": self.status.value,
            "created_at": self.created_at,
            "sent_at": self.sent_at,
            "metadata": self.metadata,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Notification":
        data = dict(data)
        if "notification_type" in data:
            data["notification_type"] = NotificationType(data["notification_type"])
        if "priority" in data:
            data["priority"] = NotificationPriority(data["priority"])
        if "status" in data:
            data["status"] = NotificationStatus(data["status"])
        if "channels" in data:
            data["channels"] = [NotificationChannel(c) for c in data["channels"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==============================================================================
# Deployment Notification System (Step 225)
# ==============================================================================

class DeploymentNotificationSystem:
    """
    Deployment Notification System - sends deployment alerts and notifications.

    PBTSO Phase: ITERATE

    Responsibilities:
    - Send deployment notifications via multiple channels
    - Manage notification templates
    - Handle subscriber preferences
    - Track notification history
    - Support retry logic for failed notifications

    Example:
        >>> notifier = DeploymentNotificationSystem()
        >>> notifier.notify(
        ...     notification_type=NotificationType.DEPLOYMENT_STARTED,
        ...     service_name="api",
        ...     deployment_id="deploy-123",
        ...     variables={"version": "v2.0.0"},
        ... )
    """

    BUS_TOPICS = {
        "send": "deploy.notifications.send",
        "sent": "deploy.notifications.sent",
        "failed": "deploy.notifications.failed",
        "subscribe": "deploy.notifications.subscribe",
    }

    # Default templates
    DEFAULT_TEMPLATES = {
        NotificationType.DEPLOYMENT_STARTED: {
            "subject": "Deployment Started: {{service_name}} {{version}}",
            "body": "Deployment of {{service_name}} version {{version}} to {{environment}} has started.\n\nDeployment ID: {{deployment_id}}\nStarted by: {{initiated_by}}",
        },
        NotificationType.DEPLOYMENT_COMPLETED: {
            "subject": "Deployment Completed: {{service_name}} {{version}}",
            "body": "Deployment of {{service_name}} version {{version}} to {{environment}} completed successfully.\n\nDeployment ID: {{deployment_id}}\nDuration: {{duration}}",
        },
        NotificationType.DEPLOYMENT_FAILED: {
            "subject": "ALERT: Deployment Failed - {{service_name}} {{version}}",
            "body": "Deployment of {{service_name}} version {{version}} to {{environment}} has FAILED.\n\nDeployment ID: {{deployment_id}}\nError: {{error}}",
        },
        NotificationType.APPROVAL_REQUIRED: {
            "subject": "Approval Required: {{service_name}} {{version}}",
            "body": "Deployment of {{service_name}} version {{version}} to {{environment}} requires approval.\n\nDeployment ID: {{deployment_id}}\nRequestor: {{requestor}}",
        },
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "notification-system",
        max_retries: int = 3,
    ):
        """
        Initialize the notification system.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
            max_retries: Maximum retry attempts for failed notifications
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "notifications"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self.max_retries = max_retries

        self._templates: Dict[str, NotificationTemplate] = {}
        self._subscribers: Dict[str, Subscriber] = {}
        self._notifications: Dict[str, Notification] = {}
        self._channel_handlers: Dict[NotificationChannel, Callable] = {}

        self._load_state()
        self._init_default_templates()

    def _init_default_templates(self) -> None:
        """Initialize default templates."""
        for ntype, template_data in self.DEFAULT_TEMPLATES.items():
            template_id = f"default-{ntype.value}"
            if template_id not in self._templates:
                self._templates[template_id] = NotificationTemplate(
                    template_id=template_id,
                    name=f"Default {ntype.value}",
                    notification_type=ntype,
                    subject_template=template_data["subject"],
                    body_template=template_data["body"],
                    channels=[NotificationChannel.BUS],
                    priority=NotificationPriority.HIGH if "FAILED" in ntype.value else NotificationPriority.NORMAL,
                )

    def register_channel_handler(
        self,
        channel: NotificationChannel,
        handler: Callable[[Notification, str], bool],
    ) -> None:
        """
        Register a handler for a notification channel.

        Args:
            channel: Notification channel
            handler: Handler function (notification, address) -> success
        """
        self._channel_handlers[channel] = handler

    def create_template(
        self,
        name: str,
        notification_type: NotificationType,
        subject_template: str,
        body_template: str,
        channels: Optional[List[NotificationChannel]] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
    ) -> NotificationTemplate:
        """
        Create a notification template.

        Args:
            name: Template name
            notification_type: Type of notification
            subject_template: Subject template with {{placeholders}}
            body_template: Body template with {{placeholders}}
            channels: Target channels
            priority: Default priority

        Returns:
            Created NotificationTemplate
        """
        template_id = f"template-{uuid.uuid4().hex[:12]}"

        template = NotificationTemplate(
            template_id=template_id,
            name=name,
            notification_type=notification_type,
            subject_template=subject_template,
            body_template=body_template,
            channels=channels or [NotificationChannel.BUS],
            priority=priority,
        )

        self._templates[template_id] = template
        self._save_state()

        return template

    def subscribe(
        self,
        name: str,
        channels: Dict[str, str],
        notification_types: Optional[List[NotificationType]] = None,
        services: Optional[List[str]] = None,
        environments: Optional[List[str]] = None,
    ) -> Subscriber:
        """
        Create a notification subscriber.

        Args:
            name: Subscriber name
            channels: Channel addresses (e.g., {"slack": "#deployments"})
            notification_types: Types to subscribe to
            services: Services to subscribe to (empty = all)
            environments: Environments to subscribe to

        Returns:
            Created Subscriber
        """
        subscriber_id = f"subscriber-{uuid.uuid4().hex[:12]}"

        subscriber = Subscriber(
            subscriber_id=subscriber_id,
            name=name,
            channels=channels,
            notification_types=notification_types or list(NotificationType),
            services=services or [],
            environments=environments or [],
        )

        self._subscribers[subscriber_id] = subscriber
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["subscribe"],
            {
                "subscriber_id": subscriber_id,
                "name": name,
                "channels": list(channels.keys()),
            },
            actor=self.actor_id,
        )

        return subscriber

    async def notify(
        self,
        notification_type: NotificationType,
        service_name: str = "",
        deployment_id: str = "",
        environment: str = "",
        variables: Optional[Dict[str, Any]] = None,
        priority: Optional[NotificationPriority] = None,
        channels: Optional[List[NotificationChannel]] = None,
    ) -> Notification:
        """
        Send a deployment notification.

        Args:
            notification_type: Type of notification
            service_name: Associated service
            deployment_id: Associated deployment
            environment: Target environment
            variables: Template variables
            priority: Override priority
            channels: Override channels

        Returns:
            Created Notification
        """
        notification_id = f"notification-{uuid.uuid4().hex[:12]}"

        # Find template
        template = self._find_template(notification_type)
        if template:
            subject, body = template.render(variables or {})
            if priority is None:
                priority = template.priority
            if channels is None:
                channels = template.channels
        else:
            subject = f"{notification_type.value}: {service_name}"
            body = json.dumps(variables or {}, indent=2)
            if priority is None:
                priority = NotificationPriority.NORMAL
            if channels is None:
                channels = [NotificationChannel.BUS]

        # Find recipients
        recipients = self._find_recipients(
            notification_type, service_name, environment
        )

        notification = Notification(
            notification_id=notification_id,
            notification_type=notification_type,
            subject=subject,
            body=body,
            priority=priority,
            channels=channels,
            recipients=recipients,
            service_name=service_name,
            deployment_id=deployment_id,
            environment=environment,
            metadata=variables or {},
        )

        _emit_bus_event(
            self.BUS_TOPICS["send"],
            {
                "notification_id": notification_id,
                "type": notification_type.value,
                "service_name": service_name,
                "deployment_id": deployment_id,
                "priority": priority.value,
            },
            actor=self.actor_id,
        )

        # Send via channels
        await self._send_notification(notification)

        self._notifications[notification_id] = notification
        self._save_state()

        return notification

    def _find_template(
        self,
        notification_type: NotificationType,
    ) -> Optional[NotificationTemplate]:
        """Find template for notification type."""
        # Look for custom template first
        for template in self._templates.values():
            if template.notification_type == notification_type and template.enabled:
                if not template.template_id.startswith("default-"):
                    return template

        # Fall back to default
        default_id = f"default-{notification_type.value}"
        return self._templates.get(default_id)

    def _find_recipients(
        self,
        notification_type: NotificationType,
        service_name: str,
        environment: str,
    ) -> List[str]:
        """Find recipients for notification."""
        recipients = []

        for subscriber in self._subscribers.values():
            if not subscriber.active:
                continue

            # Check notification type
            if notification_type not in subscriber.notification_types:
                continue

            # Check service filter
            if subscriber.services and service_name not in subscriber.services:
                continue

            # Check environment filter
            if subscriber.environments and environment not in subscriber.environments:
                continue

            # Add all channel addresses
            for address in subscriber.channels.values():
                if address not in recipients:
                    recipients.append(address)

        return recipients

    async def _send_notification(self, notification: Notification) -> None:
        """Send notification via all channels."""
        success_count = 0
        error_count = 0

        for channel in notification.channels:
            try:
                handler = self._channel_handlers.get(channel)
                if handler:
                    for recipient in notification.recipients:
                        if asyncio.iscoroutinefunction(handler):
                            result = await handler(notification, recipient)
                        else:
                            result = handler(notification, recipient)

                        if result:
                            success_count += 1
                        else:
                            error_count += 1
                elif channel == NotificationChannel.BUS:
                    # Always succeeds for bus
                    success_count += 1
                else:
                    notification.errors.append(f"No handler for channel: {channel.value}")
                    error_count += 1

            except Exception as e:
                notification.errors.append(f"{channel.value}: {str(e)}")
                error_count += 1

        # Update status
        if error_count == 0:
            notification.status = NotificationStatus.SENT
            notification.sent_at = time.time()

            _emit_bus_event(
                self.BUS_TOPICS["sent"],
                {
                    "notification_id": notification.notification_id,
                    "type": notification.notification_type.value,
                    "channels": [c.value for c in notification.channels],
                },
                actor=self.actor_id,
            )
        else:
            notification.status = NotificationStatus.FAILED

            _emit_bus_event(
                self.BUS_TOPICS["failed"],
                {
                    "notification_id": notification.notification_id,
                    "type": notification.notification_type.value,
                    "errors": notification.errors,
                },
                level="error",
                actor=self.actor_id,
            )

    async def retry_failed(self) -> List[Notification]:
        """Retry failed notifications."""
        retried = []

        for notification in self._notifications.values():
            if notification.status != NotificationStatus.FAILED:
                continue

            retry_count = notification.metadata.get("retry_count", 0)
            if retry_count >= self.max_retries:
                continue

            notification.metadata["retry_count"] = retry_count + 1
            notification.status = NotificationStatus.RETRYING
            notification.errors = []

            await self._send_notification(notification)
            retried.append(notification)

        if retried:
            self._save_state()

        return retried

    def notify_deployment_started(
        self,
        service_name: str,
        version: str,
        deployment_id: str,
        environment: str,
        initiated_by: str = "",
    ) -> Notification:
        """Convenience method for deployment started notification."""
        return asyncio.get_event_loop().run_until_complete(
            self.notify(
                notification_type=NotificationType.DEPLOYMENT_STARTED,
                service_name=service_name,
                deployment_id=deployment_id,
                environment=environment,
                variables={
                    "service_name": service_name,
                    "version": version,
                    "deployment_id": deployment_id,
                    "environment": environment,
                    "initiated_by": initiated_by,
                },
            )
        )

    def notify_deployment_completed(
        self,
        service_name: str,
        version: str,
        deployment_id: str,
        environment: str,
        duration: str = "",
    ) -> Notification:
        """Convenience method for deployment completed notification."""
        return asyncio.get_event_loop().run_until_complete(
            self.notify(
                notification_type=NotificationType.DEPLOYMENT_COMPLETED,
                service_name=service_name,
                deployment_id=deployment_id,
                environment=environment,
                variables={
                    "service_name": service_name,
                    "version": version,
                    "deployment_id": deployment_id,
                    "environment": environment,
                    "duration": duration,
                },
            )
        )

    def notify_deployment_failed(
        self,
        service_name: str,
        version: str,
        deployment_id: str,
        environment: str,
        error: str = "",
    ) -> Notification:
        """Convenience method for deployment failed notification."""
        return asyncio.get_event_loop().run_until_complete(
            self.notify(
                notification_type=NotificationType.DEPLOYMENT_FAILED,
                service_name=service_name,
                deployment_id=deployment_id,
                environment=environment,
                priority=NotificationPriority.CRITICAL,
                variables={
                    "service_name": service_name,
                    "version": version,
                    "deployment_id": deployment_id,
                    "environment": environment,
                    "error": error,
                },
            )
        )

    def get_notification(self, notification_id: str) -> Optional[Notification]:
        """Get a notification by ID."""
        return self._notifications.get(notification_id)

    def list_notifications(
        self,
        status: Optional[NotificationStatus] = None,
        notification_type: Optional[NotificationType] = None,
        limit: int = 100,
    ) -> List[Notification]:
        """List notifications."""
        notifications = list(self._notifications.values())

        if status:
            notifications = [n for n in notifications if n.status == status]
        if notification_type:
            notifications = [n for n in notifications if n.notification_type == notification_type]

        notifications.sort(key=lambda n: n.created_at, reverse=True)
        return notifications[:limit]

    def list_templates(self) -> List[NotificationTemplate]:
        """List all templates."""
        return list(self._templates.values())

    def list_subscribers(self) -> List[Subscriber]:
        """List all subscribers."""
        return list(self._subscribers.values())

    def unsubscribe(self, subscriber_id: str) -> bool:
        """Unsubscribe a subscriber."""
        if subscriber_id not in self._subscribers:
            return False

        del self._subscribers[subscriber_id]
        self._save_state()
        return True

    def delete_template(self, template_id: str) -> bool:
        """Delete a template."""
        if template_id not in self._templates:
            return False

        del self._templates[template_id]
        self._save_state()
        return True

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "templates": {tid: t.to_dict() for tid, t in self._templates.items()},
            "subscribers": {sid: s.to_dict() for sid, s in self._subscribers.items()},
            "notifications": {
                nid: n.to_dict()
                for nid, n in list(self._notifications.items())[-1000:]  # Keep last 1000
            },
        }
        state_file = self.state_dir / "notification_state.json"
        with open(state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "notification_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            for tid, data in state.get("templates", {}).items():
                data["notification_type"] = NotificationType(data["notification_type"])
                data["priority"] = NotificationPriority(data["priority"])
                data["channels"] = [NotificationChannel(c) for c in data.get("channels", [])]
                self._templates[tid] = NotificationTemplate(**{
                    k: v for k, v in data.items() if k in NotificationTemplate.__dataclass_fields__
                })

            for sid, data in state.get("subscribers", {}).items():
                data["notification_types"] = [
                    NotificationType(t) for t in data.get("notification_types", [])
                ]
                self._subscribers[sid] = Subscriber(**{
                    k: v for k, v in data.items() if k in Subscriber.__dataclass_fields__
                })

            for nid, data in state.get("notifications", {}).items():
                self._notifications[nid] = Notification.from_dict(data)
        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for notification system."""
    import argparse

    parser = argparse.ArgumentParser(description="Deployment Notification System (Step 225)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # send command
    send_parser = subparsers.add_parser("send", help="Send a notification")
    send_parser.add_argument("--type", "-t", required=True,
                             choices=[t.value for t in NotificationType])
    send_parser.add_argument("--service", "-s", required=True, help="Service name")
    send_parser.add_argument("--deployment", "-d", default="", help="Deployment ID")
    send_parser.add_argument("--env", "-e", default="prod", help="Environment")
    send_parser.add_argument("--version", "-v", default="", help="Version")
    send_parser.add_argument("--json", action="store_true", help="JSON output")

    # subscribe command
    subscribe_parser = subparsers.add_parser("subscribe", help="Create subscriber")
    subscribe_parser.add_argument("--name", "-n", required=True, help="Subscriber name")
    subscribe_parser.add_argument("--slack", help="Slack channel")
    subscribe_parser.add_argument("--email", help="Email address")
    subscribe_parser.add_argument("--webhook", help="Webhook URL")
    subscribe_parser.add_argument("--json", action="store_true", help="JSON output")

    # template command
    template_parser = subparsers.add_parser("template", help="Create template")
    template_parser.add_argument("--name", "-n", required=True, help="Template name")
    template_parser.add_argument("--type", "-t", required=True,
                                  choices=[t.value for t in NotificationType])
    template_parser.add_argument("--subject", required=True, help="Subject template")
    template_parser.add_argument("--body", required=True, help="Body template")
    template_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List notifications")
    list_parser.add_argument("--status", "-s", help="Filter by status")
    list_parser.add_argument("--type", "-t", help="Filter by type")
    list_parser.add_argument("--limit", "-l", type=int, default=10, help="Limit")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # subscribers command
    subscribers_parser = subparsers.add_parser("subscribers", help="List subscribers")
    subscribers_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    notifier = DeploymentNotificationSystem()

    if args.command == "send":
        notification = asyncio.get_event_loop().run_until_complete(
            notifier.notify(
                notification_type=NotificationType(args.type),
                service_name=args.service,
                deployment_id=args.deployment,
                environment=args.env,
                variables={
                    "service_name": args.service,
                    "version": args.version,
                    "deployment_id": args.deployment,
                    "environment": args.env,
                },
            )
        )

        if args.json:
            print(json.dumps(notification.to_dict(), indent=2))
        else:
            print(f"Sent: {notification.notification_id}")
            print(f"  Type: {notification.notification_type.value}")
            print(f"  Status: {notification.status.value}")

        return 0

    elif args.command == "subscribe":
        channels = {}
        if args.slack:
            channels["slack"] = args.slack
        if args.email:
            channels["email"] = args.email
        if args.webhook:
            channels["webhook"] = args.webhook

        subscriber = notifier.subscribe(
            name=args.name,
            channels=channels,
        )

        if args.json:
            print(json.dumps(subscriber.to_dict(), indent=2))
        else:
            print(f"Subscribed: {subscriber.subscriber_id}")
            print(f"  Name: {subscriber.name}")
            print(f"  Channels: {list(subscriber.channels.keys())}")

        return 0

    elif args.command == "template":
        template = notifier.create_template(
            name=args.name,
            notification_type=NotificationType(args.type),
            subject_template=args.subject,
            body_template=args.body,
        )

        if args.json:
            print(json.dumps(template.to_dict(), indent=2))
        else:
            print(f"Created: {template.template_id}")
            print(f"  Name: {template.name}")

        return 0

    elif args.command == "list":
        status = NotificationStatus(args.status) if args.status else None
        ntype = NotificationType(args.type) if args.type else None
        notifications = notifier.list_notifications(
            status=status,
            notification_type=ntype,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps([n.to_dict() for n in notifications], indent=2))
        else:
            for n in notifications:
                print(f"{n.notification_id} ({n.notification_type.value}) - {n.status.value}")

        return 0

    elif args.command == "subscribers":
        subscribers = notifier.list_subscribers()

        if args.json:
            print(json.dumps([s.to_dict() for s in subscribers], indent=2))
        else:
            for s in subscribers:
                status = "active" if s.active else "inactive"
                print(f"{s.subscriber_id} ({s.name}) - {status}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
