#!/usr/bin/env python3
"""
deprecation_manager.py - Deprecation Manager (Step 99)

PBTSO Phase: SKILL, ITERATE

Provides:
- Deprecation tracking
- Migration guidance
- Sunset scheduling
- Warning emission
- Usage analytics for deprecated features

Bus Topics:
- code.deprecation.warn
- code.deprecation.sunset
- code.deprecation.usage

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import functools
import json
import os
import socket
import time
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class DeprecationLevel(Enum):
    """Deprecation severity level."""
    NOTICE = "notice"       # Informational, will be deprecated
    WARNING = "warning"     # Deprecated, will be removed
    ERROR = "error"         # Scheduled for removal, avoid use
    REMOVED = "removed"     # No longer available


@dataclass
class DeprecationConfig:
    """Configuration for deprecation manager."""
    emit_warnings: bool = True
    raise_on_removed: bool = True
    track_usage: bool = True
    log_to_bus: bool = True
    warning_frequency: str = "once"  # once, always, periodic
    warning_interval_s: int = 3600
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emit_warnings": self.emit_warnings,
            "raise_on_removed": self.raise_on_removed,
            "track_usage": self.track_usage,
            "warning_frequency": self.warning_frequency,
        }


# =============================================================================
# Agent Bus with File Locking
# =============================================================================

class LockedAgentBus:
    """Agent bus with file locking for safe concurrent writes."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        line = json.dumps(full_event, ensure_ascii=False, separators=(",", ":")) + "\n"

        fd = os.open(str(self.bus_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        return event_id


# =============================================================================
# Deprecation Types
# =============================================================================

@dataclass
class Deprecation:
    """A deprecation definition."""
    id: str
    name: str
    reason: str
    level: DeprecationLevel
    deprecated_since: str
    remove_in: Optional[str] = None
    sunset_date: Optional[float] = None
    replacement: Optional[str] = None
    migration_guide: Optional[str] = None
    usage_count: int = 0
    last_used: Optional[float] = None

    def is_sunset(self) -> bool:
        """Check if deprecation has reached sunset."""
        if self.sunset_date:
            return time.time() >= self.sunset_date
        return self.level == DeprecationLevel.REMOVED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "reason": self.reason,
            "level": self.level.value,
            "deprecated_since": self.deprecated_since,
            "remove_in": self.remove_in,
            "sunset_date": self.sunset_date,
            "sunset_iso": datetime.fromtimestamp(self.sunset_date, tz=timezone.utc).isoformat() if self.sunset_date else None,
            "replacement": self.replacement,
            "migration_guide": self.migration_guide,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
        }


@dataclass
class DeprecationWarning:
    """A deprecation warning instance."""
    deprecation_id: str
    name: str
    message: str
    level: DeprecationLevel
    timestamp: float = field(default_factory=time.time)
    caller: Optional[str] = None
    replacement: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deprecation_id": self.deprecation_id,
            "name": self.name,
            "message": self.message,
            "level": self.level.value,
            "timestamp": self.timestamp,
            "caller": self.caller,
            "replacement": self.replacement,
        }


@dataclass
class UsageRecord:
    """Record of deprecated feature usage."""
    deprecation_id: str
    count: int = 0
    first_used: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    callers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deprecation_id": self.deprecation_id,
            "count": self.count,
            "first_used": self.first_used,
            "last_used": self.last_used,
            "unique_callers": len(set(self.callers)),
        }


# =============================================================================
# Deprecation Manager
# =============================================================================

class DeprecationManager:
    """
    Manager for deprecation handling.

    PBTSO Phase: SKILL, ITERATE

    Features:
    - Deprecation registration and tracking
    - Warning emission
    - Migration guidance
    - Usage analytics
    - Sunset scheduling

    Usage:
        deprecation = DeprecationManager()

        # Register deprecation
        deprecation.register(
            name="old_function",
            reason="Replaced by new_function",
            deprecated_since="1.0.0",
            remove_in="2.0.0",
            replacement="new_function",
        )

        # Warn when deprecated feature is used
        deprecation.warn("old_function")
    """

    BUS_TOPICS = {
        "warn": "code.deprecation.warn",
        "sunset": "code.deprecation.sunset",
        "usage": "code.deprecation.usage",
    }

    # Global instance for decorator
    _instance: Optional["DeprecationManager"] = None

    def __init__(
        self,
        config: Optional[DeprecationConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or DeprecationConfig()
        self.bus = bus or LockedAgentBus()

        self._deprecations: Dict[str, Deprecation] = {}
        self._usage: Dict[str, UsageRecord] = {}
        self._warned: Dict[str, float] = {}  # Track warning times
        self._lock = Lock()

        # Set as global instance
        DeprecationManager._instance = self

    @classmethod
    def get_instance(cls) -> "DeprecationManager":
        """Get or create global instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # =========================================================================
    # Registration
    # =========================================================================

    def register(
        self,
        name: str,
        reason: str,
        deprecated_since: str,
        level: DeprecationLevel = DeprecationLevel.WARNING,
        remove_in: Optional[str] = None,
        sunset_date: Optional[float] = None,
        replacement: Optional[str] = None,
        migration_guide: Optional[str] = None,
    ) -> Deprecation:
        """Register a deprecation."""
        deprecation = Deprecation(
            id=f"dep-{uuid.uuid4().hex[:8]}",
            name=name,
            reason=reason,
            level=level,
            deprecated_since=deprecated_since,
            remove_in=remove_in,
            sunset_date=sunset_date,
            replacement=replacement,
            migration_guide=migration_guide,
        )

        with self._lock:
            self._deprecations[name] = deprecation

        return deprecation

    def update_level(self, name: str, level: DeprecationLevel) -> bool:
        """Update deprecation level."""
        with self._lock:
            if name not in self._deprecations:
                return False
            self._deprecations[name].level = level
            return True

    def set_sunset(self, name: str, sunset_date: float) -> bool:
        """Set sunset date for deprecation."""
        with self._lock:
            if name not in self._deprecations:
                return False
            self._deprecations[name].sunset_date = sunset_date

            self.bus.emit({
                "topic": self.BUS_TOPICS["sunset"],
                "kind": "deprecation",
                "actor": "deprecation-manager",
                "data": {
                    "name": name,
                    "sunset_date": sunset_date,
                    "sunset_iso": datetime.fromtimestamp(sunset_date, tz=timezone.utc).isoformat(),
                },
            })

            return True

    def get_deprecation(self, name: str) -> Optional[Deprecation]:
        """Get deprecation by name."""
        return self._deprecations.get(name)

    def list_deprecations(
        self,
        level: Optional[DeprecationLevel] = None,
        include_removed: bool = False,
    ) -> List[Deprecation]:
        """List all deprecations."""
        deprecations = list(self._deprecations.values())

        if level:
            deprecations = [d for d in deprecations if d.level == level]

        if not include_removed:
            deprecations = [d for d in deprecations if d.level != DeprecationLevel.REMOVED]

        return sorted(deprecations, key=lambda d: d.name)

    # =========================================================================
    # Warning
    # =========================================================================

    def warn(
        self,
        name: str,
        caller: Optional[str] = None,
    ) -> Optional[DeprecationWarning]:
        """Emit deprecation warning."""
        with self._lock:
            deprecation = self._deprecations.get(name)
            if not deprecation:
                return None

            # Track usage
            if self.config.track_usage:
                self._record_usage(name, caller)

            # Check if should warn
            if not self._should_warn(name):
                return None

            # Build message
            message = f"'{name}' is deprecated"
            if deprecation.reason:
                message += f": {deprecation.reason}"
            if deprecation.replacement:
                message += f". Use '{deprecation.replacement}' instead"
            if deprecation.remove_in:
                message += f". Will be removed in {deprecation.remove_in}"

            warning = DeprecationWarning(
                deprecation_id=deprecation.id,
                name=name,
                message=message,
                level=deprecation.level,
                caller=caller,
                replacement=deprecation.replacement,
            )

            # Emit warning
            if self.config.emit_warnings:
                warnings.warn(message, DeprecationWarning, stacklevel=3)

            # Log to bus
            if self.config.log_to_bus:
                self.bus.emit({
                    "topic": self.BUS_TOPICS["warn"],
                    "kind": "deprecation",
                    "level": "warning",
                    "actor": "deprecation-manager",
                    "data": warning.to_dict(),
                })

            # Check if removed
            if deprecation.level == DeprecationLevel.REMOVED and self.config.raise_on_removed:
                raise RuntimeError(f"'{name}' has been removed: {deprecation.reason}")

            self._warned[name] = time.time()
            return warning

    def _should_warn(self, name: str) -> bool:
        """Check if we should emit a warning."""
        if self.config.warning_frequency == "always":
            return True
        elif self.config.warning_frequency == "once":
            return name not in self._warned
        elif self.config.warning_frequency == "periodic":
            last_warned = self._warned.get(name)
            if last_warned is None:
                return True
            return (time.time() - last_warned) >= self.config.warning_interval_s
        return True

    def _record_usage(self, name: str, caller: Optional[str]) -> None:
        """Record usage of deprecated feature."""
        if name not in self._usage:
            self._usage[name] = UsageRecord(deprecation_id=self._deprecations[name].id)

        record = self._usage[name]
        record.count += 1
        record.last_used = time.time()
        if caller and caller not in record.callers[-100:]:
            record.callers.append(caller)

        # Update deprecation
        self._deprecations[name].usage_count = record.count
        self._deprecations[name].last_used = record.last_used

    # =========================================================================
    # Checking
    # =========================================================================

    def is_deprecated(self, name: str) -> bool:
        """Check if a feature is deprecated."""
        return name in self._deprecations

    def is_sunset(self, name: str) -> bool:
        """Check if a feature has reached sunset."""
        deprecation = self._deprecations.get(name)
        if not deprecation:
            return False
        return deprecation.is_sunset()

    def check_and_warn(self, name: str, caller: Optional[str] = None) -> bool:
        """Check if deprecated and emit warning if so."""
        if self.is_deprecated(name):
            self.warn(name, caller)
            return True
        return False

    def get_replacement(self, name: str) -> Optional[str]:
        """Get replacement for deprecated feature."""
        deprecation = self._deprecations.get(name)
        return deprecation.replacement if deprecation else None

    def get_migration_guide(self, name: str) -> Optional[str]:
        """Get migration guide for deprecated feature."""
        deprecation = self._deprecations.get(name)
        return deprecation.migration_guide if deprecation else None

    # =========================================================================
    # Usage Analytics
    # =========================================================================

    def get_usage(self, name: str) -> Optional[UsageRecord]:
        """Get usage record for a deprecation."""
        return self._usage.get(name)

    def get_usage_report(self) -> Dict[str, Any]:
        """Get usage report for all deprecations."""
        report = []
        for name, record in self._usage.items():
            deprecation = self._deprecations.get(name)
            if deprecation:
                report.append({
                    "name": name,
                    "level": deprecation.level.value,
                    "usage_count": record.count,
                    "last_used": record.last_used,
                    "unique_callers": len(set(record.callers)),
                })

        # Sort by usage count
        report.sort(key=lambda x: x["usage_count"], reverse=True)
        return {"deprecations": report, "total_usage": sum(r["usage_count"] for r in report)}

    def get_upcoming_sunsets(self, days: int = 30) -> List[Deprecation]:
        """Get deprecations with upcoming sunsets."""
        now = time.time()
        threshold = now + (days * 24 * 60 * 60)

        upcoming = []
        for dep in self._deprecations.values():
            if dep.sunset_date and now <= dep.sunset_date <= threshold:
                upcoming.append(dep)

        return sorted(upcoming, key=lambda d: d.sunset_date or 0)

    # =========================================================================
    # Utilities
    # =========================================================================

    def stats(self) -> Dict[str, Any]:
        """Get deprecation statistics."""
        by_level = defaultdict(int)
        for dep in self._deprecations.values():
            by_level[dep.level.value] += 1

        return {
            "total_deprecations": len(self._deprecations),
            "by_level": dict(by_level),
            "total_usage": sum(r.count for r in self._usage.values()),
            "active_warnings": len(self._warned),
            "config": self.config.to_dict(),
        }


# =============================================================================
# Decorator
# =============================================================================

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(
    reason: str = "",
    replacement: Optional[str] = None,
    since: str = "1.0.0",
    remove_in: Optional[str] = None,
    level: DeprecationLevel = DeprecationLevel.WARNING,
) -> Callable[[F], F]:
    """
    Decorator to mark a function/method as deprecated.

    Usage:
        @deprecated(reason="Use new_function instead", replacement="new_function")
        def old_function():
            pass
    """
    def decorator(func: F) -> F:
        # Register deprecation
        manager = DeprecationManager.get_instance()
        manager.register(
            name=func.__qualname__,
            reason=reason,
            deprecated_since=since,
            level=level,
            remove_in=remove_in,
            replacement=replacement,
        )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager.warn(func.__qualname__, caller=func.__module__)
            return func(*args, **kwargs)

        return wrapper  # type: ignore
    return decorator


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Deprecation Manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Deprecation Manager (Step 99)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    list_parser = subparsers.add_parser("list", help="List deprecations")
    list_parser.add_argument("--level", "-l", choices=["notice", "warning", "error", "removed"])
    list_parser.add_argument("--json", action="store_true")

    # check command
    check_parser = subparsers.add_parser("check", help="Check if deprecated")
    check_parser.add_argument("name", help="Feature name")

    # usage command
    usage_parser = subparsers.add_parser("usage", help="Show usage report")
    usage_parser.add_argument("--json", action="store_true")

    # sunsets command
    sunset_parser = subparsers.add_parser("sunsets", help="Show upcoming sunsets")
    sunset_parser.add_argument("--days", "-d", type=int, default=30)

    # stats command
    subparsers.add_parser("stats", help="Show statistics")

    # demo command
    subparsers.add_parser("demo", help="Run deprecation demo")

    args = parser.parse_args()
    manager = DeprecationManager()

    if args.command == "list":
        level = DeprecationLevel(args.level) if args.level else None
        deprecations = manager.list_deprecations(level=level, include_removed=True)

        if args.json:
            print(json.dumps([d.to_dict() for d in deprecations], indent=2))
        else:
            print("Deprecations:")
            for d in deprecations:
                level_icon = {
                    DeprecationLevel.NOTICE: "[N]",
                    DeprecationLevel.WARNING: "[W]",
                    DeprecationLevel.ERROR: "[E]",
                    DeprecationLevel.REMOVED: "[X]",
                }[d.level]
                print(f"  {level_icon} {d.name} (since {d.deprecated_since})")
                if d.replacement:
                    print(f"       Replace with: {d.replacement}")
        return 0

    elif args.command == "check":
        if manager.is_deprecated(args.name):
            dep = manager.get_deprecation(args.name)
            print(f"'{args.name}' is deprecated ({dep.level.value})")
            print(f"  Reason: {dep.reason}")
            if dep.replacement:
                print(f"  Replacement: {dep.replacement}")
            return 1
        else:
            print(f"'{args.name}' is not deprecated")
            return 0

    elif args.command == "usage":
        report = manager.get_usage_report()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(f"Total usage: {report['total_usage']}")
            print("\nDeprecations by usage:")
            for item in report["deprecations"][:20]:
                print(f"  {item['name']}: {item['usage_count']} uses ({item['level']})")
        return 0

    elif args.command == "sunsets":
        sunsets = manager.get_upcoming_sunsets(args.days)
        print(f"Upcoming sunsets (next {args.days} days):")
        for dep in sunsets:
            sunset_str = datetime.fromtimestamp(dep.sunset_date, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  {dep.name}: {sunset_str}")
        if not sunsets:
            print("  (none)")
        return 0

    elif args.command == "stats":
        stats = manager.stats()
        print(json.dumps(stats, indent=2))
        return 0

    elif args.command == "demo":
        print("Deprecation Manager Demo\n")

        # Register deprecations
        print("1. Registering deprecations...")
        manager.register(
            name="old_api",
            reason="Replaced by new_api for better performance",
            deprecated_since="1.0.0",
            remove_in="2.0.0",
            replacement="new_api",
            level=DeprecationLevel.WARNING,
        )

        manager.register(
            name="legacy_format",
            reason="Use JSON format instead",
            deprecated_since="0.9.0",
            level=DeprecationLevel.ERROR,
            sunset_date=time.time() + 7 * 24 * 60 * 60,  # 7 days
        )

        manager.register(
            name="experimental_feature",
            reason="Will be removed in next major version",
            deprecated_since="1.5.0",
            level=DeprecationLevel.NOTICE,
        )

        # List deprecations
        print("\nDeprecations:")
        for dep in manager.list_deprecations():
            print(f"  [{dep.level.value}] {dep.name}")

        # Test warnings
        print("\n2. Testing warnings...")
        manager.warn("old_api", caller="demo")
        manager.warn("old_api", caller="demo")  # Should not warn again (once mode)
        manager.warn("legacy_format", caller="demo")

        # Check deprecated decorator
        print("\n3. Testing decorator...")

        @deprecated(reason="Use new_function instead", replacement="new_function")
        def old_function() -> str:
            return "result"

        result = old_function()
        print(f"  old_function() returned: {result}")

        # Usage report
        print("\n4. Usage report:")
        report = manager.get_usage_report()
        for item in report["deprecations"]:
            print(f"  {item['name']}: {item['usage_count']} uses")

        # Upcoming sunsets
        print("\n5. Upcoming sunsets (30 days):")
        sunsets = manager.get_upcoming_sunsets(30)
        for dep in sunsets:
            sunset_str = datetime.fromtimestamp(dep.sunset_date, tz=timezone.utc).strftime("%Y-%m-%d")
            print(f"  {dep.name}: {sunset_str}")

        print("\nStatistics:")
        print(json.dumps(manager.stats(), indent=2))

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
