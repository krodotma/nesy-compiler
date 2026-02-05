#!/usr/bin/env python3
"""
Deprecation Manager (Step 199)

Manages deprecation of features, APIs, and components for the Review Agent
with sunset schedules, migration paths, and notifications.

PBTSO Phase: BUILD, DISTILL
Bus Topics: review.deprecation.announce, review.deprecation.sunset, review.deprecation.usage

Deprecation Features:
- Feature deprecation tracking
- Sunset scheduling
- Usage monitoring
- Migration path documentation
- Warning notifications

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900


# ============================================================================
# Types
# ============================================================================

class DeprecationStatus(Enum):
    """Deprecation status."""
    ACTIVE = "active"             # Feature is current
    DEPRECATED = "deprecated"     # Feature is deprecated but working
    SUNSET = "sunset"             # Feature has been removed
    PLANNED = "planned"           # Deprecation planned


class DeprecationSeverity(Enum):
    """Deprecation severity."""
    LOW = "low"           # Minor feature, minimal impact
    MEDIUM = "medium"     # Moderate impact
    HIGH = "high"         # Significant impact
    CRITICAL = "critical" # Breaking change


@dataclass
class DeprecatedFeature:
    """
    A deprecated feature definition.

    Attributes:
        feature_id: Unique feature identifier
        name: Feature name
        description: Feature description
        status: Deprecation status
        severity: Deprecation severity
        deprecated_at: Deprecation announcement date
        sunset_at: Planned sunset date
        replacement: Replacement feature name
        migration_guide: Migration documentation URL or text
        usage_count: Number of usages detected
        last_used: Last usage timestamp
    """
    feature_id: str
    name: str
    description: str = ""
    status: DeprecationStatus = DeprecationStatus.ACTIVE
    severity: DeprecationSeverity = DeprecationSeverity.MEDIUM
    deprecated_at: Optional[str] = None
    sunset_at: Optional[str] = None
    replacement: Optional[str] = None
    migration_guide: Optional[str] = None
    usage_count: int = 0
    last_used: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_id": self.feature_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "severity": self.severity.value,
            "deprecated_at": self.deprecated_at,
            "sunset_at": self.sunset_at,
            "replacement": self.replacement,
            "migration_guide": self.migration_guide,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
        }

    @property
    def is_deprecated(self) -> bool:
        """Check if feature is deprecated."""
        return self.status in (DeprecationStatus.DEPRECATED, DeprecationStatus.SUNSET)

    @property
    def is_sunset(self) -> bool:
        """Check if feature is sunset."""
        return self.status == DeprecationStatus.SUNSET

    @property
    def days_until_sunset(self) -> Optional[int]:
        """Get days until sunset."""
        if not self.sunset_at:
            return None
        sunset = datetime.fromisoformat(self.sunset_at.rstrip("Z"))
        sunset = sunset.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = sunset - now
        return max(0, delta.days)


@dataclass
class SunsetSchedule:
    """
    Sunset schedule for deprecated features.

    Attributes:
        schedule_id: Unique schedule ID
        feature_ids: Features included
        sunset_date: Planned sunset date
        announcement_date: Public announcement date
        warning_period_days: Days of warning before sunset
        notifications_sent: Notification tracking
    """
    schedule_id: str
    feature_ids: List[str]
    sunset_date: str
    announcement_date: str = ""
    warning_period_days: int = 90
    notifications_sent: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.announcement_date:
            self.announcement_date = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schedule_id": self.schedule_id,
            "feature_ids": self.feature_ids,
            "sunset_date": self.sunset_date,
            "announcement_date": self.announcement_date,
            "warning_period_days": self.warning_period_days,
            "notifications_sent": self.notifications_sent,
        }


@dataclass
class DeprecationPolicy:
    """
    Deprecation policy configuration.

    Attributes:
        minimum_warning_days: Minimum days before sunset
        notification_intervals: Days before sunset to notify
        require_replacement: Require replacement feature
        require_migration_guide: Require migration documentation
    """
    minimum_warning_days: int = 90
    notification_intervals: List[int] = field(default_factory=lambda: [90, 60, 30, 14, 7, 1])
    require_replacement: bool = True
    require_migration_guide: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DeprecationWarning:
    """Warning about deprecated feature usage."""
    feature_id: str
    feature_name: str
    message: str
    severity: DeprecationSeverity
    sunset_at: Optional[str]
    replacement: Optional[str]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_id": self.feature_id,
            "feature_name": self.feature_name,
            "message": self.message,
            "severity": self.severity.value,
            "sunset_at": self.sunset_at,
            "replacement": self.replacement,
            "timestamp": self.timestamp,
        }


# ============================================================================
# Deprecation Manager
# ============================================================================

class DeprecationManager:
    """
    Manages feature deprecation lifecycle.

    Example:
        manager = DeprecationManager()

        # Register feature
        manager.register_feature(DeprecatedFeature(
            feature_id="old-api",
            name="Legacy API v1",
            status=DeprecationStatus.DEPRECATED,
            sunset_at="2025-01-01T00:00:00Z",
            replacement="api-v2",
        ))

        # Track usage
        warning = manager.track_usage("old-api")
        if warning:
            print(f"Warning: {warning.message}")

        # Check sunset status
        sunset_features = manager.check_sunset_schedule()
    """

    BUS_TOPICS = {
        "announce": "review.deprecation.announce",
        "sunset": "review.deprecation.sunset",
        "usage": "review.deprecation.usage",
    }

    def __init__(
        self,
        policy: Optional[DeprecationPolicy] = None,
        bus_path: Optional[Path] = None,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize deprecation manager.

        Args:
            policy: Deprecation policy
            bus_path: Path to event bus file
            storage_path: Path to state storage
        """
        self.policy = policy or DeprecationPolicy()
        self.bus_path = bus_path or self._get_bus_path()
        self.storage_path = storage_path or self._get_storage_path()

        self._features: Dict[str, DeprecatedFeature] = {}
        self._schedules: Dict[str, SunsetSchedule] = {}
        self._warnings: List[DeprecationWarning] = []

        self._last_heartbeat = time.time()

        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _get_storage_path(self) -> Path:
        """Get storage path."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return pluribus_root / ".pluribus" / "deprecation"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "deprecation") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "deprecation-manager",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _load_state(self) -> None:
        """Load state from storage."""
        state_file = self.storage_path / "state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    data = json.load(f)

                for feature_data in data.get("features", {}).values():
                    self._features[feature_data["feature_id"]] = DeprecatedFeature(
                        feature_id=feature_data["feature_id"],
                        name=feature_data["name"],
                        description=feature_data.get("description", ""),
                        status=DeprecationStatus(feature_data.get("status", "active")),
                        severity=DeprecationSeverity(feature_data.get("severity", "medium")),
                        deprecated_at=feature_data.get("deprecated_at"),
                        sunset_at=feature_data.get("sunset_at"),
                        replacement=feature_data.get("replacement"),
                        migration_guide=feature_data.get("migration_guide"),
                        usage_count=feature_data.get("usage_count", 0),
                        last_used=feature_data.get("last_used"),
                    )
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_state(self) -> None:
        """Save state to storage."""
        state_file = self.storage_path / "state.json"

        data = {
            "features": {k: v.to_dict() for k, v in self._features.items()},
            "schedules": {k: v.to_dict() for k, v in self._schedules.items()},
            "updated_at": datetime.now(timezone.utc).isoformat() + "Z",
        }

        with open(state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def register_feature(self, feature: DeprecatedFeature) -> None:
        """
        Register a feature for deprecation tracking.

        Args:
            feature: Feature to register
        """
        self._features[feature.feature_id] = feature
        self._save_state()

        if feature.status == DeprecationStatus.DEPRECATED:
            self._emit_event(self.BUS_TOPICS["announce"], {
                "feature_id": feature.feature_id,
                "name": feature.name,
                "sunset_at": feature.sunset_at,
                "replacement": feature.replacement,
            })

    def deprecate_feature(
        self,
        feature_id: str,
        sunset_date: str,
        replacement: Optional[str] = None,
        migration_guide: Optional[str] = None,
    ) -> Optional[DeprecatedFeature]:
        """
        Deprecate a feature.

        Args:
            feature_id: Feature to deprecate
            sunset_date: Planned sunset date
            replacement: Replacement feature
            migration_guide: Migration documentation

        Returns:
            Updated feature or None if not found
        """
        feature = self._features.get(feature_id)
        if not feature:
            return None

        feature.status = DeprecationStatus.DEPRECATED
        feature.deprecated_at = datetime.now(timezone.utc).isoformat() + "Z"
        feature.sunset_at = sunset_date
        feature.replacement = replacement
        feature.migration_guide = migration_guide

        self._save_state()

        self._emit_event(self.BUS_TOPICS["announce"], {
            "action": "deprecate",
            "feature_id": feature_id,
            "sunset_at": sunset_date,
            "replacement": replacement,
        })

        return feature

    def sunset_feature(self, feature_id: str) -> Optional[DeprecatedFeature]:
        """
        Mark a feature as sunset.

        Args:
            feature_id: Feature to sunset

        Returns:
            Updated feature or None if not found
        """
        feature = self._features.get(feature_id)
        if not feature:
            return None

        feature.status = DeprecationStatus.SUNSET

        self._save_state()

        self._emit_event(self.BUS_TOPICS["sunset"], {
            "feature_id": feature_id,
            "name": feature.name,
            "usage_count": feature.usage_count,
        })

        return feature

    def track_usage(self, feature_id: str) -> Optional[DeprecationWarning]:
        """
        Track usage of a potentially deprecated feature.

        Args:
            feature_id: Feature being used

        Returns:
            DeprecationWarning if feature is deprecated
        """
        feature = self._features.get(feature_id)
        if not feature:
            return None

        # Update usage stats
        feature.usage_count += 1
        feature.last_used = datetime.now(timezone.utc).isoformat() + "Z"
        self._save_state()

        # Emit usage event
        self._emit_event(self.BUS_TOPICS["usage"], {
            "feature_id": feature_id,
            "status": feature.status.value,
            "usage_count": feature.usage_count,
        })

        # Generate warning if deprecated
        if feature.is_deprecated and not feature.is_sunset:
            days = feature.days_until_sunset
            warning = DeprecationWarning(
                feature_id=feature.feature_id,
                feature_name=feature.name,
                message=self._generate_warning_message(feature, days),
                severity=feature.severity,
                sunset_at=feature.sunset_at,
                replacement=feature.replacement,
            )
            self._warnings.append(warning)
            return warning

        if feature.is_sunset:
            warning = DeprecationWarning(
                feature_id=feature.feature_id,
                feature_name=feature.name,
                message=f"Feature '{feature.name}' has been sunset and is no longer available",
                severity=DeprecationSeverity.CRITICAL,
                sunset_at=feature.sunset_at,
                replacement=feature.replacement,
            )
            self._warnings.append(warning)
            return warning

        return None

    def _generate_warning_message(
        self,
        feature: DeprecatedFeature,
        days_until_sunset: Optional[int],
    ) -> str:
        """Generate deprecation warning message."""
        msg = f"Feature '{feature.name}' is deprecated"

        if days_until_sunset is not None:
            if days_until_sunset == 0:
                msg += " and will be sunset TODAY"
            elif days_until_sunset <= 7:
                msg += f" and will be sunset in {days_until_sunset} day(s)"
            elif days_until_sunset <= 30:
                msg += f" and will be sunset in {days_until_sunset} days"
            else:
                msg += f". Sunset scheduled for {feature.sunset_at}"

        if feature.replacement:
            msg += f". Please migrate to '{feature.replacement}'"

        return msg

    def get_feature(self, feature_id: str) -> Optional[DeprecatedFeature]:
        """Get feature by ID."""
        return self._features.get(feature_id)

    def get_deprecated_features(self) -> List[DeprecatedFeature]:
        """Get all deprecated features."""
        return [f for f in self._features.values() if f.is_deprecated]

    def get_upcoming_sunsets(self, days: int = 30) -> List[DeprecatedFeature]:
        """Get features with upcoming sunsets."""
        features = []
        for feature in self._features.values():
            if feature.is_deprecated and not feature.is_sunset:
                days_left = feature.days_until_sunset
                if days_left is not None and days_left <= days:
                    features.append(feature)
        return sorted(features, key=lambda f: f.days_until_sunset or 0)

    def get_warnings(self, limit: int = 100) -> List[DeprecationWarning]:
        """Get recent warnings."""
        return self._warnings[-limit:]

    def check_sunset_schedule(self) -> List[DeprecatedFeature]:
        """
        Check for features that should be sunset.

        Returns:
            List of features past their sunset date
        """
        now = datetime.now(timezone.utc)
        to_sunset = []

        for feature in self._features.values():
            if feature.status == DeprecationStatus.DEPRECATED and feature.sunset_at:
                sunset = datetime.fromisoformat(feature.sunset_at.rstrip("Z"))
                sunset = sunset.replace(tzinfo=timezone.utc)
                if now >= sunset:
                    to_sunset.append(feature)

        return to_sunset

    def create_sunset_schedule(
        self,
        feature_ids: List[str],
        sunset_date: str,
        warning_period_days: int = 90,
    ) -> SunsetSchedule:
        """
        Create a sunset schedule.

        Args:
            feature_ids: Features to include
            sunset_date: Planned sunset date
            warning_period_days: Warning period

        Returns:
            Created schedule
        """
        schedule = SunsetSchedule(
            schedule_id=str(uuid.uuid4())[:8],
            feature_ids=feature_ids,
            sunset_date=sunset_date,
            warning_period_days=warning_period_days,
        )

        self._schedules[schedule.schedule_id] = schedule

        # Update features
        for feature_id in feature_ids:
            feature = self._features.get(feature_id)
            if feature:
                feature.status = DeprecationStatus.DEPRECATED
                feature.sunset_at = sunset_date
                feature.deprecated_at = schedule.announcement_date

        self._save_state()

        self._emit_event(self.BUS_TOPICS["announce"], {
            "schedule_id": schedule.schedule_id,
            "features": feature_ids,
            "sunset_date": sunset_date,
        })

        return schedule

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        deprecated = self.get_deprecated_features()
        upcoming = self.get_upcoming_sunsets(days=30)

        status = {
            "agent": "deprecation-manager",
            "healthy": True,
            "features_tracked": len(self._features),
            "deprecated_features": len(deprecated),
            "upcoming_sunsets_30d": len(upcoming),
            "warnings_generated": len(self._warnings),
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self._emit_event("a2a.heartbeat", status, kind="heartbeat")
        return status


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Deprecation Manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Deprecation Manager (Step 199)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List features")
    list_parser.add_argument("--deprecated", action="store_true", help="Only deprecated")
    list_parser.add_argument("--upcoming", type=int, help="Upcoming sunsets within N days")

    # Register command
    register_parser = subparsers.add_parser("register", help="Register feature")
    register_parser.add_argument("feature_id", help="Feature ID")
    register_parser.add_argument("name", help="Feature name")
    register_parser.add_argument("--status", choices=["active", "deprecated"],
                                 default="active", help="Initial status")

    # Deprecate command
    deprecate_parser = subparsers.add_parser("deprecate", help="Deprecate feature")
    deprecate_parser.add_argument("feature_id", help="Feature ID")
    deprecate_parser.add_argument("--sunset", required=True, help="Sunset date")
    deprecate_parser.add_argument("--replacement", help="Replacement feature")

    # Track command
    track_parser = subparsers.add_parser("track", help="Track feature usage")
    track_parser.add_argument("feature_id", help="Feature ID")

    # Warnings command
    warnings_parser = subparsers.add_parser("warnings", help="Show warnings")
    warnings_parser.add_argument("--limit", type=int, default=20, help="Limit results")

    # Check command
    subparsers.add_parser("check", help="Check sunset schedule")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    manager = DeprecationManager()

    # Register some demo features
    if not manager._features:
        manager.register_feature(DeprecatedFeature(
            feature_id="api-v1",
            name="API v1",
            description="Legacy API version 1",
            status=DeprecationStatus.DEPRECATED,
            sunset_at=(datetime.now(timezone.utc) + timedelta(days=30)).isoformat() + "Z",
            replacement="api-v2",
        ))
        manager.register_feature(DeprecatedFeature(
            feature_id="old-auth",
            name="Legacy Authentication",
            status=DeprecationStatus.ACTIVE,
        ))

    if args.command == "list":
        if args.deprecated:
            features = manager.get_deprecated_features()
        elif args.upcoming:
            features = manager.get_upcoming_sunsets(days=args.upcoming)
        else:
            features = list(manager._features.values())

        if args.json:
            print(json.dumps([f.to_dict() for f in features], indent=2))
        else:
            print(f"Features: {len(features)}")
            for f in features:
                status = f"[{f.status.value}]"
                sunset = f" (sunset: {f.sunset_at})" if f.sunset_at else ""
                print(f"  {f.feature_id}: {f.name} {status}{sunset}")

    elif args.command == "register":
        feature = DeprecatedFeature(
            feature_id=args.feature_id,
            name=args.name,
            status=DeprecationStatus[args.status.upper()],
        )
        manager.register_feature(feature)
        print(f"Registered feature: {args.feature_id}")

    elif args.command == "deprecate":
        feature = manager.deprecate_feature(
            args.feature_id,
            args.sunset,
            replacement=args.replacement,
        )
        if feature:
            print(f"Deprecated: {feature.name}")
            print(f"  Sunset: {feature.sunset_at}")
            if feature.replacement:
                print(f"  Replacement: {feature.replacement}")
        else:
            print(f"Feature not found: {args.feature_id}")
            return 1

    elif args.command == "track":
        warning = manager.track_usage(args.feature_id)
        if warning:
            if args.json:
                print(json.dumps(warning.to_dict(), indent=2))
            else:
                print(f"Warning: {warning.message}")
        else:
            feature = manager.get_feature(args.feature_id)
            if feature:
                print(f"Feature '{feature.name}' is active (usage count: {feature.usage_count})")
            else:
                print(f"Feature not found: {args.feature_id}")

    elif args.command == "warnings":
        warnings = manager.get_warnings(limit=args.limit)
        if args.json:
            print(json.dumps([w.to_dict() for w in warnings], indent=2))
        else:
            print(f"Warnings: {len(warnings)}")
            for w in warnings:
                print(f"  [{w.severity.value}] {w.feature_name}: {w.message}")

    elif args.command == "check":
        to_sunset = manager.check_sunset_schedule()
        if args.json:
            print(json.dumps([f.to_dict() for f in to_sunset], indent=2))
        else:
            if to_sunset:
                print(f"Features past sunset date: {len(to_sunset)}")
                for f in to_sunset:
                    print(f"  {f.feature_id}: {f.name} (was scheduled for {f.sunset_at})")
            else:
                print("No features past sunset date")

    else:
        # Default: show status
        status = manager.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Deprecation Manager: {status['features_tracked']} features, {status['deprecated_features']} deprecated")

    return 0


if __name__ == "__main__":
    sys.exit(main())
