#!/usr/bin/env python3
"""
manager.py - Deprecation Manager (Step 249)

PBTSO Phase: PLAN
A2A Integration: Deprecation handling via deploy.deprecation.notify

Provides:
- DeprecatedItem: Deprecated item definition
- DeprecationPolicy: Deprecation lifecycle policy
- DeprecationNotice: Deprecation notification
- SunsetSchedule: Sunset scheduling
- MigrationGuide: Migration guidance
- DeprecationManager: Complete deprecation handling

Bus Topics:
- deploy.deprecation.notify
- deploy.deprecation.register
- deploy.deprecation.sunset
- deploy.deprecation.usage

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

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


# ==============================================================================
# Bus Emission Helper with fcntl.flock()
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
    actor: str = "deprecation-manager"
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

class DeprecationStatus(Enum):
    """Deprecation lifecycle status."""
    ACTIVE = "active"           # Still active, deprecation planned
    DEPRECATED = "deprecated"   # Deprecated, still functional
    SUNSET = "sunset"          # End of life, may be removed
    REMOVED = "removed"        # No longer available


class ItemType(Enum):
    """Types of deprecated items."""
    API_ENDPOINT = "api_endpoint"
    API_VERSION = "api_version"
    FEATURE = "feature"
    FIELD = "field"
    PARAMETER = "parameter"
    CONFIGURATION = "configuration"
    SERVICE = "service"


class NotificationChannel(Enum):
    """Notification channels."""
    BUS = "bus"
    WEBHOOK = "webhook"
    EMAIL = "email"
    LOG = "log"
    HEADER = "header"


@dataclass
class DeprecationPolicy:
    """
    Deprecation lifecycle policy.

    Attributes:
        policy_id: Unique policy identifier
        name: Policy name
        deprecation_notice_days: Days before deprecation
        sunset_notice_days: Days before sunset
        grace_period_days: Grace period after sunset
        allow_override: Allow policy override per item
        notification_channels: Enabled notification channels
        auto_sunset: Automatically sunset after period
    """
    policy_id: str
    name: str
    deprecation_notice_days: int = 90
    sunset_notice_days: int = 180
    grace_period_days: int = 30
    allow_override: bool = True
    notification_channels: List[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.BUS, NotificationChannel.LOG]
    )
    auto_sunset: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "deprecation_notice_days": self.deprecation_notice_days,
            "sunset_notice_days": self.sunset_notice_days,
            "grace_period_days": self.grace_period_days,
            "allow_override": self.allow_override,
            "notification_channels": [c.value for c in self.notification_channels],
            "auto_sunset": self.auto_sunset,
        }


@dataclass
class DeprecatedItem:
    """
    Deprecated item definition.

    Attributes:
        item_id: Unique item identifier
        name: Item name
        item_type: Type of item
        status: Deprecation status
        reason: Deprecation reason
        replacement: Replacement item/path
        deprecated_at: Deprecation timestamp
        sunset_at: Sunset timestamp
        removed_at: Removal timestamp
        usage_count: Usage counter
        last_used_at: Last usage timestamp
        metadata: Additional metadata
    """
    item_id: str
    name: str
    item_type: ItemType = ItemType.FEATURE
    status: DeprecationStatus = DeprecationStatus.ACTIVE
    reason: str = ""
    replacement: str = ""
    deprecated_at: float = 0.0
    sunset_at: float = 0.0
    removed_at: float = 0.0
    usage_count: int = 0
    last_used_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "name": self.name,
            "item_type": self.item_type.value,
            "status": self.status.value,
            "reason": self.reason,
            "replacement": self.replacement,
            "deprecated_at": self.deprecated_at,
            "sunset_at": self.sunset_at,
            "removed_at": self.removed_at,
            "usage_count": self.usage_count,
            "last_used_at": self.last_used_at,
            "metadata": self.metadata,
        }


@dataclass
class DeprecationNotice:
    """
    Deprecation notification.

    Attributes:
        notice_id: Unique notice identifier
        item_id: Deprecated item ID
        title: Notice title
        message: Notice message
        severity: Notice severity
        action_required: Action required from users
        deadline: Deadline timestamp
        sent_at: Sent timestamp
        channels: Notification channels used
    """
    notice_id: str
    item_id: str
    title: str
    message: str = ""
    severity: str = "warning"
    action_required: str = ""
    deadline: float = 0.0
    sent_at: float = field(default_factory=time.time)
    channels: List[NotificationChannel] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "notice_id": self.notice_id,
            "item_id": self.item_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity,
            "action_required": self.action_required,
            "deadline": self.deadline,
            "sent_at": self.sent_at,
            "channels": [c.value for c in self.channels],
        }


@dataclass
class SunsetSchedule:
    """
    Sunset scheduling.

    Attributes:
        schedule_id: Unique schedule identifier
        item_id: Item to sunset
        scheduled_at: Scheduled sunset timestamp
        executed: Whether sunset was executed
        executed_at: Execution timestamp
        rollback_until: Rollback available until
    """
    schedule_id: str
    item_id: str
    scheduled_at: float
    executed: bool = False
    executed_at: float = 0.0
    rollback_until: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MigrationGuide:
    """
    Migration guidance for deprecated items.

    Attributes:
        guide_id: Unique guide identifier
        item_id: Deprecated item ID
        title: Guide title
        description: Guide description
        steps: Migration steps
        code_examples: Code examples
        estimated_effort: Estimated effort
        created_at: Creation timestamp
    """
    guide_id: str
    item_id: str
    title: str
    description: str = ""
    steps: List[str] = field(default_factory=list)
    code_examples: Dict[str, str] = field(default_factory=dict)
    estimated_effort: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Deprecation Manager (Step 249)
# ==============================================================================

class DeprecationManager:
    """
    Deprecation Manager - Handle deprecation lifecycle for deployments.

    PBTSO Phase: PLAN

    Responsibilities:
    - Register deprecated items
    - Track deprecation usage
    - Send deprecation notices
    - Schedule sunsets
    - Provide migration guidance

    Example:
        >>> manager = DeprecationManager()
        >>> item = manager.deprecate(
        ...     name="/api/v1/users",
        ...     item_type=ItemType.API_ENDPOINT,
        ...     reason="Replaced by v2",
        ...     replacement="/api/v2/users"
        ... )
        >>> manager.track_usage(item.item_id)
        >>> guide = manager.create_migration_guide(
        ...     item.item_id,
        ...     steps=["Update endpoint URL", "Update response handling"]
        ... )
    """

    BUS_TOPICS = {
        "notify": "deploy.deprecation.notify",
        "register": "deploy.deprecation.register",
        "sunset": "deploy.deprecation.sunset",
        "usage": "deploy.deprecation.usage",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        policy: Optional[DeprecationPolicy] = None,
        actor_id: str = "deprecation-manager",
    ):
        """
        Initialize the deprecation manager.

        Args:
            state_dir: Directory for state persistence
            policy: Default deprecation policy
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "deprecation"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Default policy
        self.policy = policy or DeprecationPolicy(
            policy_id=f"policy-{uuid.uuid4().hex[:8]}",
            name="default",
        )

        # Storage
        self._items: Dict[str, DeprecatedItem] = {}
        self._notices: List[DeprecationNotice] = []
        self._schedules: Dict[str, SunsetSchedule] = {}
        self._guides: Dict[str, MigrationGuide] = {}

        self._load_state()

    def deprecate(
        self,
        name: str,
        item_type: ItemType = ItemType.FEATURE,
        reason: str = "",
        replacement: str = "",
        sunset_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeprecatedItem:
        """
        Register a deprecated item.

        Args:
            name: Item name/identifier
            item_type: Type of item
            reason: Deprecation reason
            replacement: Replacement item
            sunset_days: Days until sunset (uses policy default if None)
            metadata: Additional metadata

        Returns:
            Created DeprecatedItem
        """
        item_id = f"dep-{uuid.uuid4().hex[:12]}"
        now = time.time()

        if sunset_days is None:
            sunset_days = self.policy.sunset_notice_days

        item = DeprecatedItem(
            item_id=item_id,
            name=name,
            item_type=item_type,
            status=DeprecationStatus.DEPRECATED,
            reason=reason,
            replacement=replacement,
            deprecated_at=now,
            sunset_at=now + (sunset_days * 86400),
            metadata=metadata or {},
        )

        self._items[item_id] = item
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["register"],
            {
                "item_id": item_id,
                "name": name,
                "item_type": item_type.value,
                "reason": reason,
                "replacement": replacement,
                "sunset_at": item.sunset_at,
            },
            level="warn",
            actor=self.actor_id,
        )

        # Schedule sunset
        self._schedule_sunset(item)

        # Send initial notice
        self._send_notice(
            item,
            title=f"Deprecation Notice: {name}",
            message=f"{name} has been deprecated. {reason}",
            severity="warning",
            action_required=f"Migrate to {replacement}" if replacement else "Plan migration",
        )

        return item

    def track_usage(self, item_id: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Track usage of a deprecated item.

        Args:
            item_id: Item ID
            context: Usage context

        Returns:
            True if tracked
        """
        item = self._items.get(item_id)
        if not item:
            return False

        item.usage_count += 1
        item.last_used_at = time.time()

        _emit_bus_event(
            self.BUS_TOPICS["usage"],
            {
                "item_id": item_id,
                "name": item.name,
                "usage_count": item.usage_count,
                "context": context,
            },
            kind="metric",
            actor=self.actor_id,
        )

        # Check if usage warning needed
        if item.status == DeprecationStatus.DEPRECATED:
            days_until_sunset = (item.sunset_at - time.time()) / 86400
            if days_until_sunset <= 30 and item.usage_count % 100 == 0:
                self._send_notice(
                    item,
                    title=f"Upcoming Sunset: {item.name}",
                    message=f"{item.name} will be sunset in {int(days_until_sunset)} days. Current usage: {item.usage_count}",
                    severity="error",
                    action_required="Complete migration immediately",
                )

        self._save_state()
        return True

    def _schedule_sunset(self, item: DeprecatedItem) -> SunsetSchedule:
        """Schedule sunset for an item."""
        schedule_id = f"schedule-{uuid.uuid4().hex[:8]}"

        schedule = SunsetSchedule(
            schedule_id=schedule_id,
            item_id=item.item_id,
            scheduled_at=item.sunset_at,
            rollback_until=item.sunset_at + (self.policy.grace_period_days * 86400),
        )

        self._schedules[schedule_id] = schedule
        return schedule

    def sunset(self, item_id: str, force: bool = False) -> Optional[DeprecatedItem]:
        """
        Execute sunset for an item.

        Args:
            item_id: Item ID to sunset
            force: Force sunset even if not scheduled

        Returns:
            Updated DeprecatedItem or None
        """
        item = self._items.get(item_id)
        if not item:
            return None

        now = time.time()

        if not force and now < item.sunset_at:
            return None

        item.status = DeprecationStatus.SUNSET
        self._save_state()

        # Update schedule
        for schedule in self._schedules.values():
            if schedule.item_id == item_id and not schedule.executed:
                schedule.executed = True
                schedule.executed_at = now

        _emit_bus_event(
            self.BUS_TOPICS["sunset"],
            {
                "item_id": item_id,
                "name": item.name,
                "total_usage": item.usage_count,
            },
            level="warn",
            actor=self.actor_id,
        )

        # Send sunset notice
        self._send_notice(
            item,
            title=f"Sunset Complete: {item.name}",
            message=f"{item.name} has been sunset and is no longer available.",
            severity="critical",
            action_required="Ensure all systems have migrated",
        )

        return item

    def remove(self, item_id: str) -> Optional[DeprecatedItem]:
        """
        Mark an item as removed.

        Args:
            item_id: Item ID to remove

        Returns:
            Updated DeprecatedItem or None
        """
        item = self._items.get(item_id)
        if not item:
            return None

        item.status = DeprecationStatus.REMOVED
        item.removed_at = time.time()
        self._save_state()

        return item

    def _send_notice(
        self,
        item: DeprecatedItem,
        title: str,
        message: str,
        severity: str = "warning",
        action_required: str = "",
    ) -> DeprecationNotice:
        """Send a deprecation notice."""
        notice_id = f"notice-{uuid.uuid4().hex[:8]}"

        notice = DeprecationNotice(
            notice_id=notice_id,
            item_id=item.item_id,
            title=title,
            message=message,
            severity=severity,
            action_required=action_required,
            deadline=item.sunset_at,
            channels=self.policy.notification_channels,
        )

        self._notices.append(notice)

        # Emit to bus
        if NotificationChannel.BUS in notice.channels:
            _emit_bus_event(
                self.BUS_TOPICS["notify"],
                {
                    "notice_id": notice_id,
                    "item_id": item.item_id,
                    "title": title,
                    "message": message,
                    "severity": severity,
                    "action_required": action_required,
                },
                level="warn" if severity == "warning" else "error",
                actor=self.actor_id,
            )

        return notice

    def create_migration_guide(
        self,
        item_id: str,
        title: Optional[str] = None,
        description: str = "",
        steps: Optional[List[str]] = None,
        code_examples: Optional[Dict[str, str]] = None,
        estimated_effort: str = "",
    ) -> Optional[MigrationGuide]:
        """
        Create a migration guide for a deprecated item.

        Args:
            item_id: Deprecated item ID
            title: Guide title
            description: Guide description
            steps: Migration steps
            code_examples: Code examples (before/after)
            estimated_effort: Estimated effort

        Returns:
            Created MigrationGuide or None
        """
        item = self._items.get(item_id)
        if not item:
            return None

        guide_id = f"guide-{uuid.uuid4().hex[:8]}"

        guide = MigrationGuide(
            guide_id=guide_id,
            item_id=item_id,
            title=title or f"Migration Guide: {item.name}",
            description=description or f"Guide for migrating from {item.name} to {item.replacement}",
            steps=steps or [],
            code_examples=code_examples or {},
            estimated_effort=estimated_effort,
        )

        self._guides[guide_id] = guide
        self._save_state()

        return guide

    def get_deprecation_header(self, item_id: str) -> Dict[str, str]:
        """
        Get HTTP deprecation headers for an item.

        Returns headers conforming to RFC 8594 (Deprecation header).

        Args:
            item_id: Item ID

        Returns:
            Dict of header name -> value
        """
        item = self._items.get(item_id)
        if not item:
            return {}

        headers = {}

        if item.deprecated_at:
            dep_date = datetime.fromtimestamp(item.deprecated_at, tz=timezone.utc)
            headers["Deprecation"] = dep_date.strftime("%a, %d %b %Y %H:%M:%S GMT")

        if item.sunset_at:
            sunset_date = datetime.fromtimestamp(item.sunset_at, tz=timezone.utc)
            headers["Sunset"] = sunset_date.strftime("%a, %d %b %Y %H:%M:%S GMT")

        if item.replacement:
            headers["Link"] = f'<{item.replacement}>; rel="successor-version"'

        return headers

    def check_pending_sunsets(self) -> List[DeprecatedItem]:
        """Check for items that should be sunset."""
        now = time.time()
        pending = []

        for item in self._items.values():
            if item.status == DeprecationStatus.DEPRECATED and now >= item.sunset_at:
                pending.append(item)

        return pending

    def get_item(self, item_id: str) -> Optional[DeprecatedItem]:
        """Get a deprecated item by ID."""
        return self._items.get(item_id)

    def get_item_by_name(self, name: str) -> Optional[DeprecatedItem]:
        """Get a deprecated item by name."""
        for item in self._items.values():
            if item.name == name:
                return item
        return None

    def list_items(
        self,
        status: Optional[DeprecationStatus] = None,
        item_type: Optional[ItemType] = None,
    ) -> List[DeprecatedItem]:
        """List deprecated items with optional filters."""
        items = list(self._items.values())

        if status:
            items = [i for i in items if i.status == status]

        if item_type:
            items = [i for i in items if i.item_type == item_type]

        return sorted(items, key=lambda i: i.deprecated_at, reverse=True)

    def get_guide(self, guide_id: str) -> Optional[MigrationGuide]:
        """Get a migration guide by ID."""
        return self._guides.get(guide_id)

    def get_guide_for_item(self, item_id: str) -> Optional[MigrationGuide]:
        """Get migration guide for an item."""
        for guide in self._guides.values():
            if guide.item_id == item_id:
                return guide
        return None

    def get_notices(self, item_id: Optional[str] = None, limit: int = 100) -> List[DeprecationNotice]:
        """Get deprecation notices."""
        notices = self._notices
        if item_id:
            notices = [n for n in notices if n.item_id == item_id]
        return notices[-limit:]

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "items": {k: v.to_dict() for k, v in self._items.items()},
            "schedules": {k: v.to_dict() for k, v in self._schedules.items()},
            "guides": {k: v.to_dict() for k, v in self._guides.items()},
            "notices": [n.to_dict() for n in self._notices[-1000:]],
            "policy": self.policy.to_dict(),
        }
        state_file = self.state_dir / "deprecation_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "deprecation_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            for k, v in state.get("items", {}).items():
                v["item_type"] = ItemType(v.get("item_type", "feature"))
                v["status"] = DeprecationStatus(v.get("status", "deprecated"))
                self._items[k] = DeprecatedItem(**{
                    key: val for key, val in v.items()
                    if key in DeprecatedItem.__dataclass_fields__
                })

            for k, v in state.get("schedules", {}).items():
                self._schedules[k] = SunsetSchedule(**v)

            for k, v in state.get("guides", {}).items():
                self._guides[k] = MigrationGuide(**v)

        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for deprecation manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Deprecation Manager (Step 249)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # deprecate command
    dep_parser = subparsers.add_parser("deprecate", help="Register a deprecation")
    dep_parser.add_argument("name", help="Item name")
    dep_parser.add_argument("--type", "-t", default="feature",
                           choices=["api_endpoint", "api_version", "feature", "field", "parameter"])
    dep_parser.add_argument("--reason", "-r", default="", help="Deprecation reason")
    dep_parser.add_argument("--replacement", help="Replacement item")
    dep_parser.add_argument("--sunset-days", "-d", type=int, help="Days until sunset")
    dep_parser.add_argument("--json", action="store_true", help="JSON output")

    # track command
    track_parser = subparsers.add_parser("track", help="Track usage")
    track_parser.add_argument("item_id", help="Item ID")

    # sunset command
    sunset_parser = subparsers.add_parser("sunset", help="Execute sunset")
    sunset_parser.add_argument("item_id", help="Item ID")
    sunset_parser.add_argument("--force", "-f", action="store_true", help="Force sunset")

    # list command
    list_parser = subparsers.add_parser("list", help="List deprecated items")
    list_parser.add_argument("--status", "-s", choices=["deprecated", "sunset", "removed"])
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # info command
    info_parser = subparsers.add_parser("info", help="Get item info")
    info_parser.add_argument("item_id", help="Item ID")
    info_parser.add_argument("--json", action="store_true", help="JSON output")

    # check command
    check_parser = subparsers.add_parser("check", help="Check pending sunsets")
    check_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    manager = DeprecationManager()

    if args.command == "deprecate":
        item = manager.deprecate(
            name=args.name,
            item_type=ItemType(args.type),
            reason=args.reason,
            replacement=args.replacement or "",
            sunset_days=args.sunset_days,
        )

        if args.json:
            print(json.dumps(item.to_dict(), indent=2))
        else:
            sunset_date = datetime.fromtimestamp(item.sunset_at).strftime("%Y-%m-%d")
            print(f"Deprecated: {item.name}")
            print(f"  ID: {item.item_id}")
            print(f"  Type: {item.item_type.value}")
            print(f"  Sunset: {sunset_date}")
            if item.replacement:
                print(f"  Replacement: {item.replacement}")

        return 0

    elif args.command == "track":
        success = manager.track_usage(args.item_id)
        if success:
            item = manager.get_item(args.item_id)
            print(f"Tracked usage: {item.name} (count: {item.usage_count})")
        else:
            print(f"Item not found: {args.item_id}")
            return 1
        return 0

    elif args.command == "sunset":
        item = manager.sunset(args.item_id, force=args.force)
        if item:
            print(f"Sunset complete: {item.name}")
        else:
            print(f"Cannot sunset: {args.item_id}")
            return 1
        return 0

    elif args.command == "list":
        status = DeprecationStatus(args.status) if args.status else None
        items = manager.list_items(status=status)

        if args.json:
            print(json.dumps([i.to_dict() for i in items], indent=2))
        else:
            if not items:
                print("No deprecated items found")
            else:
                for item in items:
                    sunset_date = datetime.fromtimestamp(item.sunset_at).strftime("%Y-%m-%d")
                    print(f"[{item.status.value.upper()}] {item.name}")
                    print(f"  ID: {item.item_id}")
                    print(f"  Sunset: {sunset_date}")
                    print(f"  Usage: {item.usage_count}")

        return 0

    elif args.command == "info":
        item = manager.get_item(args.item_id)
        if not item:
            print(f"Item not found: {args.item_id}")
            return 1

        if args.json:
            print(json.dumps(item.to_dict(), indent=2))
        else:
            print(f"Item: {item.name}")
            print(f"  ID: {item.item_id}")
            print(f"  Type: {item.item_type.value}")
            print(f"  Status: {item.status.value}")
            print(f"  Reason: {item.reason}")
            print(f"  Replacement: {item.replacement or 'None'}")
            print(f"  Usage count: {item.usage_count}")
            if item.last_used_at:
                last_used = datetime.fromtimestamp(item.last_used_at).strftime("%Y-%m-%d %H:%M")
                print(f"  Last used: {last_used}")

        return 0

    elif args.command == "check":
        pending = manager.check_pending_sunsets()

        if args.json:
            print(json.dumps([i.to_dict() for i in pending], indent=2))
        else:
            if not pending:
                print("No pending sunsets")
            else:
                print(f"Pending sunsets ({len(pending)}):")
                for item in pending:
                    print(f"  {item.name} (usage: {item.usage_count})")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
