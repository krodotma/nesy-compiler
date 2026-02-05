#!/usr/bin/env python3
"""
Monitor Deprecation Manager - Step 299

Deprecation handling for the Monitor Agent.

PBTSO Phase: PLAN

Bus Topics:
- monitor.deprecation.warning (emitted)
- monitor.deprecation.scheduled (emitted)
- monitor.deprecation.removed (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import fcntl
import functools
import json
import os
import socket
import threading
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast


class DeprecationLevel(Enum):
    """Deprecation severity levels."""
    NOTICE = "notice"       # Soft deprecation, will be removed in future
    WARNING = "warning"     # Should migrate soon
    CRITICAL = "critical"   # Will be removed in next version
    REMOVED = "removed"     # Already removed, error if used


class DeprecationType(Enum):
    """Types of deprecation."""
    ENDPOINT = "endpoint"
    FUNCTION = "function"
    PARAMETER = "parameter"
    FEATURE = "feature"
    BEHAVIOR = "behavior"
    CONFIG = "config"


@dataclass
class Deprecation:
    """A deprecation entry.

    Attributes:
        deprecation_id: Unique deprecation ID
        name: Deprecated item name
        deprecation_type: Type of deprecation
        level: Deprecation level
        message: Deprecation message
        replacement: Suggested replacement
        deprecated_since: When deprecated
        removal_version: Version when removed
        removal_date: Date when removed
        migration_guide: Migration instructions
        usage_count: Number of uses since deprecated
    """
    deprecation_id: str
    name: str
    deprecation_type: DeprecationType
    level: DeprecationLevel
    message: str
    replacement: Optional[str] = None
    deprecated_since: str = ""
    removal_version: Optional[str] = None
    removal_date: Optional[float] = None
    migration_guide: str = ""
    usage_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deprecation_id": self.deprecation_id,
            "name": self.name,
            "type": self.deprecation_type.value,
            "level": self.level.value,
            "message": self.message,
            "replacement": self.replacement,
            "deprecated_since": self.deprecated_since,
            "removal_version": self.removal_version,
            "removal_date": self.removal_date,
            "migration_guide": self.migration_guide,
            "usage_count": self.usage_count,
        }


@dataclass
class DeprecationWarning:
    """A deprecation warning instance.

    Attributes:
        deprecation_id: Deprecation ID
        caller: Caller location
        timestamp: Warning timestamp
        context: Additional context
    """
    deprecation_id: str
    caller: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deprecation_id": self.deprecation_id,
            "caller": self.caller,
            "timestamp": self.timestamp,
            "context": self.context,
        }


F = TypeVar("F", bound=Callable[..., Any])


class MonitorDeprecationManager:
    """
    Deprecation manager for the Monitor Agent.

    Provides:
    - Deprecation registration
    - Usage tracking
    - Warning emission
    - Deprecation decorators
    - Sunset scheduling

    Example:
        deprecation = MonitorDeprecationManager()

        # Register deprecation
        deprecation.register(
            name="old_endpoint",
            deprecation_type=DeprecationType.ENDPOINT,
            message="Use /api/v2/metrics instead",
            replacement="/api/v2/metrics",
            removal_version="2.0.0",
        )

        # Use decorator
        @deprecation.deprecated("old_function", replacement="new_function")
        def old_function():
            pass

        # Check deprecation
        if deprecation.is_deprecated("old_endpoint"):
            warnings = deprecation.get_warnings("old_endpoint")
    """

    BUS_TOPICS = {
        "warning": "monitor.deprecation.warning",
        "scheduled": "monitor.deprecation.scheduled",
        "removed": "monitor.deprecation.removed",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        emit_warnings: bool = True,
        track_usage: bool = True,
        bus_dir: Optional[str] = None,
    ):
        """Initialize deprecation manager.

        Args:
            emit_warnings: Emit warning events
            track_usage: Track deprecated item usage
            bus_dir: Bus directory
        """
        self._emit_warnings = emit_warnings
        self._track_usage = track_usage
        self._last_heartbeat = time.time()

        # Deprecation registry
        self._deprecations: Dict[str, Deprecation] = {}
        self._warnings: List[DeprecationWarning] = []
        self._lock = threading.RLock()

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register known deprecations
        self._register_known_deprecations()

    def register(
        self,
        name: str,
        deprecation_type: DeprecationType,
        message: str,
        replacement: Optional[str] = None,
        level: DeprecationLevel = DeprecationLevel.WARNING,
        deprecated_since: str = "",
        removal_version: Optional[str] = None,
        removal_date: Optional[float] = None,
        migration_guide: str = "",
    ) -> Deprecation:
        """Register a deprecation.

        Args:
            name: Deprecated item name
            deprecation_type: Type of deprecation
            message: Deprecation message
            replacement: Suggested replacement
            level: Deprecation level
            deprecated_since: Version when deprecated
            removal_version: Version when removed
            removal_date: Date when removed
            migration_guide: Migration instructions

        Returns:
            Registered deprecation
        """
        deprecation_id = f"dep-{uuid.uuid4().hex[:12]}"

        deprecation = Deprecation(
            deprecation_id=deprecation_id,
            name=name,
            deprecation_type=deprecation_type,
            level=level,
            message=message,
            replacement=replacement,
            deprecated_since=deprecated_since or "0.3.0",
            removal_version=removal_version,
            removal_date=removal_date,
            migration_guide=migration_guide,
        )

        with self._lock:
            self._deprecations[name] = deprecation

        self._emit_bus_event(
            self.BUS_TOPICS["scheduled"],
            {
                "name": name,
                "type": deprecation_type.value,
                "removal_version": removal_version,
            },
        )

        return deprecation

    def warn(
        self,
        name: str,
        caller: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Deprecation]:
        """Emit deprecation warning.

        Args:
            name: Deprecated item name
            caller: Caller location
            context: Additional context

        Returns:
            Deprecation if found
        """
        with self._lock:
            deprecation = self._deprecations.get(name)
            if not deprecation:
                return None

            # Track usage
            if self._track_usage:
                deprecation.usage_count += 1

            # Record warning
            warning = DeprecationWarning(
                deprecation_id=deprecation.deprecation_id,
                caller=caller,
                context=context or {},
            )
            self._warnings.append(warning)

            # Trim warnings
            if len(self._warnings) > 10000:
                self._warnings = self._warnings[-10000:]

        # Emit warning
        if self._emit_warnings:
            self._emit_bus_event(
                self.BUS_TOPICS["warning"],
                {
                    "name": name,
                    "message": deprecation.message,
                    "replacement": deprecation.replacement,
                    "level": deprecation.level.value,
                    "caller": caller,
                },
                level="warning",
            )

        # Issue Python warning
        msg = deprecation.message
        if deprecation.replacement:
            msg += f" Use {deprecation.replacement} instead."
        warnings.warn(msg, DeprecationWarning, stacklevel=3)

        return deprecation

    def deprecated(
        self,
        name: str,
        replacement: Optional[str] = None,
        message: Optional[str] = None,
        level: DeprecationLevel = DeprecationLevel.WARNING,
    ) -> Callable[[F], F]:
        """Decorator to mark a function as deprecated.

        Args:
            name: Deprecation name
            replacement: Replacement function
            message: Custom message
            level: Deprecation level

        Returns:
            Decorator function
        """
        def decorator(func: F) -> F:
            # Register if not already
            if name not in self._deprecations:
                self.register(
                    name=name,
                    deprecation_type=DeprecationType.FUNCTION,
                    message=message or f"{func.__name__} is deprecated",
                    replacement=replacement,
                    level=level,
                )

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                self.warn(
                    name,
                    caller=f"{func.__module__}.{func.__name__}",
                )
                return func(*args, **kwargs)

            return cast(F, wrapper)

        return decorator

    def deprecated_parameter(
        self,
        param_name: str,
        replacement: Optional[str] = None,
        message: Optional[str] = None,
    ) -> Callable[[F], F]:
        """Decorator to mark a parameter as deprecated.

        Args:
            param_name: Deprecated parameter name
            replacement: Replacement parameter
            message: Custom message

        Returns:
            Decorator function
        """
        dep_name = f"param:{param_name}"

        if dep_name not in self._deprecations:
            self.register(
                name=dep_name,
                deprecation_type=DeprecationType.PARAMETER,
                message=message or f"Parameter '{param_name}' is deprecated",
                replacement=replacement,
            )

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if param_name in kwargs:
                    self.warn(
                        dep_name,
                        caller=f"{func.__module__}.{func.__name__}",
                        context={"parameter": param_name},
                    )
                return func(*args, **kwargs)

            return cast(F, wrapper)

        return decorator

    def is_deprecated(self, name: str) -> bool:
        """Check if an item is deprecated.

        Args:
            name: Item name

        Returns:
            True if deprecated
        """
        return name in self._deprecations

    def is_removed(self, name: str) -> bool:
        """Check if an item is removed.

        Args:
            name: Item name

        Returns:
            True if removed
        """
        with self._lock:
            dep = self._deprecations.get(name)
            return dep is not None and dep.level == DeprecationLevel.REMOVED

    def get_deprecation(self, name: str) -> Optional[Dict[str, Any]]:
        """Get deprecation details.

        Args:
            name: Item name

        Returns:
            Deprecation details or None
        """
        with self._lock:
            dep = self._deprecations.get(name)
            return dep.to_dict() if dep else None

    def list_deprecations(
        self,
        level: Optional[DeprecationLevel] = None,
        deprecation_type: Optional[DeprecationType] = None,
    ) -> List[Dict[str, Any]]:
        """List all deprecations.

        Args:
            level: Filter by level
            deprecation_type: Filter by type

        Returns:
            Deprecation list
        """
        with self._lock:
            deps = list(self._deprecations.values())

            if level:
                deps = [d for d in deps if d.level == level]

            if deprecation_type:
                deps = [d for d in deps if d.deprecation_type == deprecation_type]

            return [d.to_dict() for d in deps]

    def get_warnings_for(
        self,
        name: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get warnings for a deprecated item.

        Args:
            name: Item name
            limit: Maximum results

        Returns:
            Warning list
        """
        with self._lock:
            dep = self._deprecations.get(name)
            if not dep:
                return []

            relevant = [
                w for w in self._warnings
                if w.deprecation_id == dep.deprecation_id
            ]

            return [w.to_dict() for w in reversed(relevant[-limit:])]

    def get_recent_warnings(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent deprecation warnings.

        Args:
            limit: Maximum results

        Returns:
            Warning list
        """
        with self._lock:
            return [w.to_dict() for w in reversed(self._warnings[-limit:])]

    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for deprecated items.

        Returns:
            Usage counts by item
        """
        with self._lock:
            return {
                name: dep.usage_count
                for name, dep in self._deprecations.items()
                if dep.usage_count > 0
            }

    def update_level(
        self,
        name: str,
        level: DeprecationLevel,
    ) -> bool:
        """Update deprecation level.

        Args:
            name: Item name
            level: New level

        Returns:
            True if updated
        """
        with self._lock:
            dep = self._deprecations.get(name)
            if not dep:
                return False

            old_level = dep.level
            dep.level = level

        if level == DeprecationLevel.REMOVED:
            self._emit_bus_event(
                self.BUS_TOPICS["removed"],
                {
                    "name": name,
                    "previous_level": old_level.value,
                },
            )

        return True

    def get_migration_guidance(self, name: str) -> Dict[str, Any]:
        """Get migration guidance for a deprecated item.

        Args:
            name: Item name

        Returns:
            Migration guidance
        """
        with self._lock:
            dep = self._deprecations.get(name)
            if not dep:
                return {"error": "Not found"}

            return {
                "name": name,
                "message": dep.message,
                "replacement": dep.replacement,
                "migration_guide": dep.migration_guide,
                "removal_version": dep.removal_version,
                "removal_date": dep.removal_date,
                "level": dep.level.value,
            }

    def check_for_upcoming_removals(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Check for upcoming removals.

        Args:
            days: Days to look ahead

        Returns:
            Upcoming removals
        """
        cutoff = time.time() + (days * 86400)

        with self._lock:
            upcoming = []
            for dep in self._deprecations.values():
                if dep.removal_date and dep.removal_date <= cutoff:
                    upcoming.append({
                        "name": dep.name,
                        "removal_date": dep.removal_date,
                        "days_until": int((dep.removal_date - time.time()) / 86400),
                        "replacement": dep.replacement,
                    })

            return sorted(upcoming, key=lambda x: x["removal_date"])

    def get_statistics(self) -> Dict[str, Any]:
        """Get deprecation statistics.

        Returns:
            Statistics
        """
        with self._lock:
            return {
                "total_deprecations": len(self._deprecations),
                "total_warnings": len(self._warnings),
                "by_level": {
                    l.value: sum(1 for d in self._deprecations.values() if d.level == l)
                    for l in DeprecationLevel
                },
                "by_type": {
                    t.value: sum(1 for d in self._deprecations.values() if d.deprecation_type == t)
                    for t in DeprecationType
                },
                "total_usage": sum(d.usage_count for d in self._deprecations.values()),
            }

    def _register_known_deprecations(self) -> None:
        """Register known deprecations."""
        # Example deprecations
        self.register(
            name="/api/v0/metrics",
            deprecation_type=DeprecationType.ENDPOINT,
            message="API v0 endpoints are deprecated",
            replacement="/api/v1/metrics",
            level=DeprecationLevel.CRITICAL,
            deprecated_since="0.2.0",
            removal_version="1.0.0",
        )

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
                "component": "monitor_deprecation",
                "status": "healthy",
                "deprecations": len(self._deprecations),
            },
        )

        return True

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event",
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
            "actor": "monitor-deprecation",
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
_deprecation: Optional[MonitorDeprecationManager] = None


def get_deprecation() -> MonitorDeprecationManager:
    """Get or create the deprecation manager singleton.

    Returns:
        MonitorDeprecationManager instance
    """
    global _deprecation
    if _deprecation is None:
        _deprecation = MonitorDeprecationManager()
    return _deprecation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Deprecation Manager (Step 299)")
    parser.add_argument("--list", action="store_true", help="List all deprecations")
    parser.add_argument("--check", metavar="NAME", help="Check if item is deprecated")
    parser.add_argument("--warnings", action="store_true", help="Show recent warnings")
    parser.add_argument("--upcoming", type=int, default=30, metavar="DAYS", help="Show upcoming removals")
    parser.add_argument("--usage", action="store_true", help="Show usage statistics")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    deprecation = get_deprecation()

    if args.list:
        deps = deprecation.list_deprecations()
        if args.json:
            print(json.dumps(deps, indent=2))
        else:
            print("Deprecations:")
            for d in deps:
                print(f"  {d['name']}: [{d['level']}] {d['message']}")
                if d['replacement']:
                    print(f"    Replacement: {d['replacement']}")

    if args.check:
        is_dep = deprecation.is_deprecated(args.check)
        details = deprecation.get_deprecation(args.check)
        if args.json:
            print(json.dumps({"name": args.check, "deprecated": is_dep, "details": details}, indent=2))
        else:
            if is_dep:
                print(f"{args.check} is DEPRECATED")
                if details:
                    print(f"  Message: {details['message']}")
                    print(f"  Level: {details['level']}")
            else:
                print(f"{args.check} is not deprecated")

    if args.warnings:
        warnings_list = deprecation.get_recent_warnings(limit=20)
        if args.json:
            print(json.dumps(warnings_list, indent=2))
        else:
            print("Recent Warnings:")
            for w in warnings_list:
                print(f"  [{w['deprecation_id']}] {w['caller']}")

    if args.upcoming:
        upcoming = deprecation.check_for_upcoming_removals(days=args.upcoming)
        if args.json:
            print(json.dumps(upcoming, indent=2))
        else:
            print(f"Upcoming Removals (next {args.upcoming} days):")
            for u in upcoming:
                print(f"  {u['name']}: {u['days_until']} days")

    if args.usage:
        usage = deprecation.get_usage_stats()
        if args.json:
            print(json.dumps(usage, indent=2))
        else:
            print("Usage Statistics:")
            for name, count in sorted(usage.items(), key=lambda x: -x[1]):
                print(f"  {name}: {count} uses")

    if args.stats:
        stats = deprecation.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Deprecation Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
