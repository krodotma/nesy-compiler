#!/usr/bin/env python3
"""
Step 125: Test Notification

Sends notifications on test failures and important events.

PBTSO Phase: OBSERVE, VERIFY
Bus Topics:
- test.notify.send (subscribes)
- test.notify.sent (emits)
- test.run.* (subscribes)

Dependencies: Steps 101-124 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import urllib.request
import urllib.parse


# ============================================================================
# Constants
# ============================================================================

class NotificationChannel(Enum):
    """Supported notification channels."""
    CONSOLE = "console"
    FILE = "file"
    WEBHOOK = "webhook"
    SLACK = "slack"
    EMAIL = "email"
    GITHUB = "github"
    BUS = "bus"


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertTrigger(Enum):
    """Events that can trigger alerts."""
    TEST_FAILURE = "test_failure"
    RUN_FAILURE = "run_failure"
    COVERAGE_DROP = "coverage_drop"
    DURATION_SPIKE = "duration_spike"
    FLAKY_DETECTED = "flaky_detected"
    REGRESSION_DETECTED = "regression_detected"
    RUN_COMPLETE = "run_complete"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class AlertRule:
    """
    Rule for when to send alerts.

    Attributes:
        name: Rule name
        trigger: Event that triggers the alert
        severity: Alert severity
        channels: Channels to send to
        conditions: Additional conditions
        enabled: Whether rule is enabled
        cooldown_s: Minimum time between alerts
    """
    name: str
    trigger: AlertTrigger
    severity: AlertSeverity = AlertSeverity.ERROR
    channels: List[NotificationChannel] = field(default_factory=lambda: [NotificationChannel.CONSOLE])
    conditions: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    cooldown_s: int = 60
    last_triggered: Optional[float] = None

    def should_trigger(self, event_data: Dict[str, Any]) -> bool:
        """Check if rule should trigger for event."""
        if not self.enabled:
            return False

        # Check cooldown
        if self.last_triggered:
            if time.time() - self.last_triggered < self.cooldown_s:
                return False

        # Check conditions
        for key, expected in self.conditions.items():
            actual = event_data.get(key)
            if actual != expected:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "trigger": self.trigger.value,
            "severity": self.severity.value,
            "channels": [c.value for c in self.channels],
            "conditions": self.conditions,
            "enabled": self.enabled,
            "cooldown_s": self.cooldown_s,
        }


@dataclass
class Notification:
    """A notification to be sent."""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    channel: NotificationChannel
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sent: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "channel": self.channel.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "sent": self.sent,
            "error": self.error,
        }


@dataclass
class NotifyConfig:
    """
    Configuration for notifications.

    Attributes:
        rules: Alert rules
        channels: Channel configurations
        output_dir: Directory for file notifications
        webhook_url: Default webhook URL
        slack_webhook: Slack webhook URL
        github_token: GitHub token for PR comments
        email_config: Email configuration
        batch_delay_s: Delay for batching notifications
    """
    rules: List[AlertRule] = field(default_factory=list)
    channels: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    output_dir: str = ".pluribus/test-agent/notifications"
    webhook_url: Optional[str] = None
    slack_webhook: Optional[str] = None
    github_token: Optional[str] = None
    email_config: Dict[str, Any] = field(default_factory=dict)
    batch_delay_s: float = 2.0
    max_batch_size: int = 10

    def __post_init__(self):
        """Initialize default rules if none provided."""
        if not self.rules:
            self.rules = [
                AlertRule(
                    name="test_failure",
                    trigger=AlertTrigger.TEST_FAILURE,
                    severity=AlertSeverity.ERROR,
                    channels=[NotificationChannel.CONSOLE, NotificationChannel.BUS],
                ),
                AlertRule(
                    name="run_failure",
                    trigger=AlertTrigger.RUN_FAILURE,
                    severity=AlertSeverity.CRITICAL,
                    channels=[NotificationChannel.CONSOLE, NotificationChannel.BUS],
                ),
                AlertRule(
                    name="regression",
                    trigger=AlertTrigger.REGRESSION_DETECTED,
                    severity=AlertSeverity.ERROR,
                    channels=[NotificationChannel.CONSOLE, NotificationChannel.BUS],
                ),
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rules": [r.to_dict() for r in self.rules],
            "output_dir": self.output_dir,
            "batch_delay_s": self.batch_delay_s,
        }


@dataclass
class NotifyResult:
    """Result of notification sending."""
    total_notifications: int = 0
    sent: int = 0
    failed: int = 0
    notifications: List[Notification] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_notifications": self.total_notifications,
            "sent": self.sent,
            "failed": self.failed,
            "notifications": [n.to_dict() for n in self.notifications],
            "errors": self.errors,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class NotifyBus:
    """Bus interface for notifications with file locking."""

    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def heartbeat(self, agent_id: str) -> None:
        """Send A2A heartbeat."""
        now = time.time()
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.emit({
                "topic": "a2a.heartbeat",
                "kind": "heartbeat",
                "actor": agent_id,
                "data": {"status": "alive"},
            })
            self._last_heartbeat = now


# ============================================================================
# Circuit Breaker for External Services
# ============================================================================

class CircuitBreaker:
    """Circuit breaker for external service calls."""

    def __init__(self, failure_threshold: int = 5, reset_timeout_s: float = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.failures = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"

    def can_proceed(self) -> bool:
        """Check if circuit allows proceeding."""
        if self.state == "closed":
            return True
        if self.state == "open":
            if self.last_failure_time and \
               time.time() - self.last_failure_time > self.reset_timeout_s:
                self.state = "half-open"
                return True
            return False
        return True

    def record_success(self) -> None:
        """Record successful call."""
        if self.state == "half-open":
            self.state = "closed"
        self.failures = 0

    def record_failure(self) -> None:
        """Record failed call."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "open"


# ============================================================================
# Test Notifier
# ============================================================================

class TestNotifier:
    """
    Sends notifications on test events.

    Features:
    - Multiple notification channels
    - Rule-based alerting
    - Batching support
    - Circuit breaker for external services
    - Severity levels

    PBTSO Phase: OBSERVE, VERIFY
    Bus Topics: test.notify.send, test.notify.sent
    """

    BUS_TOPICS = {
        "send": "test.notify.send",
        "sent": "test.notify.sent",
        "error": "test.notify.error",
    }

    def __init__(self, bus=None, config: Optional[NotifyConfig] = None):
        """
        Initialize the test notifier.

        Args:
            bus: Optional bus instance
            config: Notification configuration
        """
        self.bus = bus or NotifyBus()
        self.config = config or NotifyConfig()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._pending_notifications: List[Notification] = []
        self._channel_handlers: Dict[NotificationChannel, Callable] = {
            NotificationChannel.CONSOLE: self._send_console,
            NotificationChannel.FILE: self._send_file,
            NotificationChannel.WEBHOOK: self._send_webhook,
            NotificationChannel.SLACK: self._send_slack,
            NotificationChannel.BUS: self._send_bus,
            NotificationChannel.GITHUB: self._send_github,
        }

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def notify(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        channels: Optional[List[NotificationChannel]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NotifyResult:
        """
        Send a notification.

        Args:
            title: Notification title
            message: Notification message
            severity: Alert severity
            channels: Channels to send to (default: from config)
            metadata: Additional metadata

        Returns:
            NotifyResult with send status
        """
        channels = channels or [NotificationChannel.CONSOLE]
        result = NotifyResult()

        for channel in channels:
            notification = Notification(
                id=str(uuid.uuid4()),
                title=title,
                message=message,
                severity=severity,
                channel=channel,
                metadata=metadata or {},
            )

            try:
                handler = self._channel_handlers.get(channel)
                if handler:
                    # Check circuit breaker
                    breaker = self._get_circuit_breaker(channel.value)
                    if not breaker.can_proceed():
                        notification.error = "Circuit breaker open"
                        result.failed += 1
                    else:
                        handler(notification)
                        notification.sent = True
                        result.sent += 1
                        breaker.record_success()
                else:
                    notification.error = f"Unknown channel: {channel}"
                    result.failed += 1

            except Exception as e:
                notification.error = str(e)
                result.failed += 1
                breaker = self._get_circuit_breaker(channel.value)
                breaker.record_failure()

            result.notifications.append(notification)
            result.total_notifications += 1

        return result

    def process_event(
        self,
        trigger: AlertTrigger,
        event_data: Dict[str, Any],
    ) -> NotifyResult:
        """
        Process an event and send notifications based on rules.

        Args:
            trigger: Event trigger type
            event_data: Event data

        Returns:
            NotifyResult with send status
        """
        result = NotifyResult()

        for rule in self.config.rules:
            if rule.trigger != trigger:
                continue

            if not rule.should_trigger(event_data):
                continue

            # Build notification
            title, message = self._build_notification_content(trigger, event_data, rule)

            # Send to all configured channels
            channel_result = self.notify(
                title=title,
                message=message,
                severity=rule.severity,
                channels=rule.channels,
                metadata=event_data,
            )

            # Mark rule as triggered
            rule.last_triggered = time.time()

            # Merge results
            result.total_notifications += channel_result.total_notifications
            result.sent += channel_result.sent
            result.failed += channel_result.failed
            result.notifications.extend(channel_result.notifications)
            result.errors.extend(channel_result.errors)

        return result

    def _build_notification_content(
        self,
        trigger: AlertTrigger,
        event_data: Dict[str, Any],
        rule: AlertRule,
    ) -> tuple:
        """Build notification title and message."""
        if trigger == AlertTrigger.TEST_FAILURE:
            test_name = event_data.get("test_name", "Unknown")
            title = f"Test Failed: {test_name}"
            message = event_data.get("error_message", "No error message")

        elif trigger == AlertTrigger.RUN_FAILURE:
            run_id = event_data.get("run_id", "Unknown")
            failed = event_data.get("failed", 0)
            title = f"Test Run Failed: {run_id[:8]}..."
            message = f"{failed} tests failed"

        elif trigger == AlertTrigger.COVERAGE_DROP:
            current = event_data.get("current_coverage", 0)
            previous = event_data.get("previous_coverage", 0)
            title = "Coverage Dropped"
            message = f"Coverage dropped from {previous:.1f}% to {current:.1f}%"

        elif trigger == AlertTrigger.DURATION_SPIKE:
            test_name = event_data.get("test_name", "Unknown")
            duration = event_data.get("duration_ms", 0)
            title = f"Duration Spike: {test_name}"
            message = f"Test took {duration:.0f}ms"

        elif trigger == AlertTrigger.FLAKY_DETECTED:
            test_name = event_data.get("test_name", "Unknown")
            title = f"Flaky Test Detected: {test_name}"
            message = f"Test {test_name} is flaky"

        elif trigger == AlertTrigger.REGRESSION_DETECTED:
            test_name = event_data.get("test_name", "Unknown")
            title = f"Regression Detected: {test_name}"
            message = event_data.get("message", "Regression detected")

        elif trigger == AlertTrigger.RUN_COMPLETE:
            run_id = event_data.get("run_id", "Unknown")
            passed = event_data.get("passed", 0)
            failed = event_data.get("failed", 0)
            title = f"Test Run Complete: {run_id[:8]}..."
            message = f"Passed: {passed}, Failed: {failed}"

        else:
            title = f"Test Alert: {trigger.value}"
            message = json.dumps(event_data, indent=2)

        return title, message

    def _get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a channel."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker()
        return self._circuit_breakers[name]

    def _send_console(self, notification: Notification) -> None:
        """Send notification to console."""
        severity_colors = {
            AlertSeverity.INFO: "",
            AlertSeverity.WARNING: "\033[93m",  # Yellow
            AlertSeverity.ERROR: "\033[91m",    # Red
            AlertSeverity.CRITICAL: "\033[91m\033[1m",  # Bold red
        }
        reset = "\033[0m"
        color = severity_colors.get(notification.severity, "")

        print(f"{color}[{notification.severity.value.upper()}] {notification.title}{reset}")
        print(f"  {notification.message}")

    def _send_file(self, notification: Notification) -> None:
        """Send notification to file."""
        output_path = Path(self.config.output_dir)
        output_file = output_path / "notifications.ndjson"

        with open(output_file, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(notification.to_dict()) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _send_webhook(self, notification: Notification) -> None:
        """Send notification to webhook."""
        url = self.config.webhook_url
        if not url:
            raise ValueError("No webhook URL configured")

        payload = json.dumps({
            "title": notification.title,
            "message": notification.message,
            "severity": notification.severity.value,
            "timestamp": notification.timestamp,
            "metadata": notification.metadata,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status >= 400:
                raise Exception(f"Webhook returned {resp.status}")

    def _send_slack(self, notification: Notification) -> None:
        """Send notification to Slack."""
        url = self.config.slack_webhook
        if not url:
            raise ValueError("No Slack webhook configured")

        # Map severity to color
        colors = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9800",
            AlertSeverity.ERROR: "#f44336",
            AlertSeverity.CRITICAL: "#9c27b0",
        }

        payload = json.dumps({
            "attachments": [{
                "color": colors.get(notification.severity, "#36a64f"),
                "title": notification.title,
                "text": notification.message,
                "footer": "Test Agent",
                "ts": int(notification.timestamp),
            }]
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status >= 400:
                raise Exception(f"Slack returned {resp.status}")

    def _send_bus(self, notification: Notification) -> None:
        """Send notification to bus."""
        self.bus.emit({
            "topic": self.BUS_TOPICS["sent"],
            "kind": "notification",
            "actor": "test-agent",
            "data": notification.to_dict(),
        })

    def _send_github(self, notification: Notification) -> None:
        """Send notification as GitHub PR comment."""
        token = self.config.github_token or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise ValueError("No GitHub token configured")

        pr_number = notification.metadata.get("pr_number")
        repo = notification.metadata.get("repo")

        if not pr_number or not repo:
            raise ValueError("PR number and repo required for GitHub notification")

        # Format as markdown comment
        body = f"## {notification.title}\n\n{notification.message}"

        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"

        payload = json.dumps({"body": body}).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            },
        )

        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status >= 400:
                raise Exception(f"GitHub returned {resp.status}")

    async def notify_async(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        channels: Optional[List[NotificationChannel]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> NotifyResult:
        """Async version of notify."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.notify, title, message, severity, channels, metadata
        )

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.config.rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule by name."""
        for i, rule in enumerate(self.config.rules):
            if rule.name == name:
                del self.config.rules[i]
                return True
        return False

    def list_rules(self) -> List[AlertRule]:
        """List all alert rules."""
        return self.config.rules


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Notifier."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Notifier")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Send command
    send_parser = subparsers.add_parser("send", help="Send notification")
    send_parser.add_argument("title", help="Notification title")
    send_parser.add_argument("message", help="Notification message")
    send_parser.add_argument("--severity", choices=["info", "warning", "error", "critical"],
                            default="info")
    send_parser.add_argument("--channel", choices=["console", "file", "webhook", "slack", "bus"],
                            action="append", default=[])
    send_parser.add_argument("--webhook-url", help="Webhook URL")
    send_parser.add_argument("--slack-webhook", help="Slack webhook URL")

    # Test command (simulate an event)
    test_parser = subparsers.add_parser("test", help="Test notification rules")
    test_parser.add_argument("--trigger", choices=["test_failure", "run_failure", "regression"],
                            default="test_failure")
    test_parser.add_argument("--test-name", default="test_example")

    # Rules command
    rules_parser = subparsers.add_parser("rules", help="List alert rules")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/notifications")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = NotifyConfig(output_dir=args.output)

    if hasattr(args, "webhook_url") and args.webhook_url:
        config.webhook_url = args.webhook_url
    if hasattr(args, "slack_webhook") and args.slack_webhook:
        config.slack_webhook = args.slack_webhook

    notifier = TestNotifier(config=config)

    if args.command == "send":
        channels = [NotificationChannel(c) for c in args.channel] or [NotificationChannel.CONSOLE]

        result = notifier.notify(
            title=args.title,
            message=args.message,
            severity=AlertSeverity(args.severity),
            channels=channels,
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"\nSent: {result.sent}/{result.total_notifications}")
            if result.errors:
                for error in result.errors:
                    print(f"  Error: {error}")

    elif args.command == "test":
        trigger = AlertTrigger(args.trigger)
        event_data = {
            "test_name": args.test_name,
            "error_message": "Test error message",
            "run_id": str(uuid.uuid4()),
            "failed": 1,
        }

        result = notifier.process_event(trigger, event_data)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"\nProcessed {trigger.value}")
            print(f"Sent: {result.sent}/{result.total_notifications}")

    elif args.command == "rules":
        rules = notifier.list_rules()
        if args.json:
            print(json.dumps([r.to_dict() for r in rules], indent=2))
        else:
            print("\nAlert Rules:")
            for rule in rules:
                enabled = "[ON]" if rule.enabled else "[OFF]"
                channels = ", ".join(c.value for c in rule.channels)
                print(f"  {enabled} {rule.name} ({rule.trigger.value}) -> {channels}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
