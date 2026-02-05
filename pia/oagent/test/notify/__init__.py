#!/usr/bin/env python3
"""
Test Notification Module - Step 125

Provides test failure alerting capabilities.

Components:
- TestNotifier: Send notifications on test events
- NotificationChannel: Supported notification channels
- AlertRule: Alerting rules

Bus Topics:
- test.notify.send
- test.notify.sent
"""

from .notifier import (
    TestNotifier,
    NotifyConfig,
    NotifyResult,
    NotificationChannel,
    AlertRule,
    AlertSeverity,
)

__all__ = [
    "TestNotifier",
    "NotifyConfig",
    "NotifyResult",
    "NotificationChannel",
    "AlertRule",
    "AlertSeverity",
]
