#!/usr/bin/env python3
"""
Step 149: Test Deprecation Manager

Deprecation handling for the Test Agent.

PBTSO Phase: PLAN, OBSERVE
Bus Topics:
- test.deprecation.notice (emits)
- test.deprecation.usage (emits)
- test.deprecation.sunset (emits)

Dependencies: Steps 101-148 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import functools
import json
import os
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


# ============================================================================
# Constants
# ============================================================================

class DeprecationStatus(Enum):
    """Deprecation status."""
    ACTIVE = "active"           # Currently in use
    DEPRECATED = "deprecated"   # Marked for removal
    SUNSET = "sunset"          # Scheduled for removal
    REMOVED = "removed"        # Already removed


class DeprecationSeverity(Enum):
    """Deprecation severity."""
    LOW = "low"         # Minor change
    MEDIUM = "medium"   # Significant change
    HIGH = "high"       # Breaking change
    CRITICAL = "critical"  # Immediate action required


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class DeprecationNotice:
    """
    A deprecation notice.

    Attributes:
        notice_id: Unique notice ID
        item_type: Type of deprecated item
        item_name: Name of deprecated item
        version_deprecated: Version when deprecated
        version_removed: Version when will be removed
        alternative: Alternative to use
        message: Deprecation message
        severity: Deprecation severity
        status: Current status
        usage_count: Times used since deprecation
    """
    notice_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    item_type: str = "function"  # function, class, parameter, endpoint, etc.
    item_name: str = ""
    version_deprecated: str = ""
    version_removed: Optional[str] = None
    alternative: Optional[str] = None
    message: str = ""
    severity: DeprecationSeverity = DeprecationSeverity.MEDIUM
    status: DeprecationStatus = DeprecationStatus.DEPRECATED
    created_at: float = field(default_factory=time.time)
    sunset_date: Optional[str] = None
    usage_count: int = 0
    last_used_at: Optional[float] = None
    migration_guide: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def full_message(self) -> str:
        """Get full deprecation message."""
        parts = [f"{self.item_type} '{self.item_name}' is deprecated"]

        if self.version_deprecated:
            parts.append(f"since version {self.version_deprecated}")

        if self.version_removed:
            parts.append(f"and will be removed in version {self.version_removed}")
        elif self.sunset_date:
            parts.append(f"and will be removed on {self.sunset_date}")

        if self.message:
            parts.append(f": {self.message}")

        if self.alternative:
            parts.append(f" Use '{self.alternative}' instead.")

        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "notice_id": self.notice_id,
            "item_type": self.item_type,
            "item_name": self.item_name,
            "version_deprecated": self.version_deprecated,
            "version_removed": self.version_removed,
            "alternative": self.alternative,
            "message": self.message,
            "severity": self.severity.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "sunset_date": self.sunset_date,
            "usage_count": self.usage_count,
            "last_used_at": self.last_used_at,
            "migration_guide": self.migration_guide,
            "tags": self.tags,
        }


@dataclass
class DeprecationPolicy:
    """
    Deprecation policy.

    Attributes:
        name: Policy name
        min_deprecation_period_days: Minimum deprecation period
        warning_period_days: Warning period before sunset
        require_alternative: Require alternative before deprecation
        block_deprecated_usage: Block usage of deprecated items
    """
    name: str = "default"
    min_deprecation_period_days: int = 90
    warning_period_days: int = 30
    require_alternative: bool = True
    block_deprecated_usage: bool = False
    enforce_in_tests: bool = True
    emit_warnings: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "min_deprecation_period_days": self.min_deprecation_period_days,
            "warning_period_days": self.warning_period_days,
            "require_alternative": self.require_alternative,
            "block_deprecated_usage": self.block_deprecated_usage,
        }


@dataclass
class DeprecationReport:
    """
    Deprecation usage report.

    Attributes:
        generated_at: Report generation timestamp
        total_deprecations: Total deprecation notices
        active_deprecations: Active deprecation count
        usage_summary: Usage summary
        upcoming_sunsets: Upcoming sunsets
    """
    generated_at: float = field(default_factory=time.time)
    total_deprecations: int = 0
    active_deprecations: int = 0
    by_status: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)
    usage_summary: List[Dict[str, Any]] = field(default_factory=list)
    upcoming_sunsets: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_at": datetime.fromtimestamp(self.generated_at, tz=timezone.utc).isoformat(),
            "total_deprecations": self.total_deprecations,
            "active_deprecations": self.active_deprecations,
            "by_status": self.by_status,
            "by_severity": self.by_severity,
            "usage_summary": self.usage_summary,
            "upcoming_sunsets": self.upcoming_sunsets,
        }


@dataclass
class DeprecationConfig:
    """
    Configuration for deprecation management.

    Attributes:
        output_dir: Output directory
        policy: Deprecation policy
        emit_warnings: Emit Python warnings
        track_usage: Track deprecated item usage
        log_file: Usage log file
    """
    output_dir: str = ".pluribus/test-agent/deprecation"
    policy: DeprecationPolicy = field(default_factory=DeprecationPolicy)
    emit_warnings: bool = True
    track_usage: bool = True
    log_file: str = "deprecation_usage.ndjson"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "emit_warnings": self.emit_warnings,
            "track_usage": self.track_usage,
            "policy": self.policy.to_dict(),
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class DeprecationBus:
    """Bus interface for deprecation with file locking."""

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


# ============================================================================
# Test Deprecation Manager
# ============================================================================

class TestDeprecationManager:
    """
    Deprecation manager for the Test Agent.

    Features:
    - Deprecation notice management
    - Usage tracking
    - Sunset scheduling
    - Migration guidance
    - Decorator support

    PBTSO Phase: PLAN, OBSERVE
    Bus Topics: test.deprecation.notice, test.deprecation.usage, test.deprecation.sunset
    """

    BUS_TOPICS = {
        "notice": "test.deprecation.notice",
        "usage": "test.deprecation.usage",
        "sunset": "test.deprecation.sunset",
    }

    def __init__(self, bus=None, config: Optional[DeprecationConfig] = None):
        """
        Initialize the deprecation manager.

        Args:
            bus: Optional bus instance
            config: Deprecation configuration
        """
        self.bus = bus or DeprecationBus()
        self.config = config or DeprecationConfig()
        self._notices: Dict[str, DeprecationNotice] = {}
        self._usage_log: List[Dict[str, Any]] = []

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Load existing notices
        self._load_notices()

    def deprecate(
        self,
        item_name: str,
        item_type: str = "function",
        version_deprecated: str = "",
        version_removed: Optional[str] = None,
        alternative: Optional[str] = None,
        message: str = "",
        severity: DeprecationSeverity = DeprecationSeverity.MEDIUM,
        sunset_date: Optional[str] = None,
        migration_guide: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> DeprecationNotice:
        """
        Create a deprecation notice.

        Args:
            item_name: Name of deprecated item
            item_type: Type of item (function, class, parameter, etc.)
            version_deprecated: Version when deprecated
            version_removed: Version when will be removed
            alternative: Alternative to use
            message: Deprecation message
            severity: Deprecation severity
            sunset_date: Date item will be removed
            migration_guide: Migration guide URL
            tags: Notice tags

        Returns:
            DeprecationNotice instance
        """
        # Validate policy requirements
        if self.config.policy.require_alternative and not alternative:
            raise ValueError("Alternative required by deprecation policy")

        notice = DeprecationNotice(
            item_name=item_name,
            item_type=item_type,
            version_deprecated=version_deprecated,
            version_removed=version_removed,
            alternative=alternative,
            message=message,
            severity=severity,
            sunset_date=sunset_date,
            migration_guide=migration_guide,
            tags=tags or [],
        )

        self._notices[f"{item_type}:{item_name}"] = notice
        self._save_notices()

        self._emit_event("notice", {
            "item_name": item_name,
            "item_type": item_type,
            "version_deprecated": version_deprecated,
            "severity": severity.value,
        })

        return notice

    def record_usage(self, item_name: str, item_type: str = "function") -> None:
        """
        Record usage of a deprecated item.

        Args:
            item_name: Name of deprecated item
            item_type: Type of item
        """
        key = f"{item_type}:{item_name}"

        if key in self._notices:
            notice = self._notices[key]
            notice.usage_count += 1
            notice.last_used_at = time.time()

            # Track usage
            if self.config.track_usage:
                usage_entry = {
                    "timestamp": time.time(),
                    "item_name": item_name,
                    "item_type": item_type,
                    "notice_id": notice.notice_id,
                }
                self._usage_log.append(usage_entry)
                self._log_usage(usage_entry)

            # Emit warning if configured
            if self.config.emit_warnings:
                warnings.warn(notice.full_message, DeprecationWarning, stacklevel=3)

            self._emit_event("usage", {
                "item_name": item_name,
                "item_type": item_type,
                "usage_count": notice.usage_count,
            })

            # Block if policy requires
            if self.config.policy.block_deprecated_usage:
                raise DeprecationWarning(notice.full_message)

    def deprecated(
        self,
        version: str = "",
        alternative: Optional[str] = None,
        message: str = "",
        severity: DeprecationSeverity = DeprecationSeverity.MEDIUM,
    ) -> Callable:
        """
        Decorator to mark a function as deprecated.

        Args:
            version: Version when deprecated
            alternative: Alternative function
            message: Deprecation message
            severity: Deprecation severity

        Returns:
            Decorator function
        """
        def decorator(fn: Callable) -> Callable:
            # Create notice
            self.deprecate(
                item_name=fn.__name__,
                item_type="function",
                version_deprecated=version,
                alternative=alternative,
                message=message,
                severity=severity,
            )

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                self.record_usage(fn.__name__, "function")
                return fn(*args, **kwargs)

            return wrapper

        return decorator

    def deprecated_class(
        self,
        version: str = "",
        alternative: Optional[str] = None,
        message: str = "",
    ) -> Callable:
        """
        Decorator to mark a class as deprecated.

        Args:
            version: Version when deprecated
            alternative: Alternative class
            message: Deprecation message

        Returns:
            Decorator function
        """
        def decorator(cls: type) -> type:
            # Create notice
            self.deprecate(
                item_name=cls.__name__,
                item_type="class",
                version_deprecated=version,
                alternative=alternative,
                message=message,
            )

            original_init = cls.__init__

            @functools.wraps(original_init)
            def new_init(self_instance, *args, **kwargs):
                self.record_usage(cls.__name__, "class")
                return original_init(self_instance, *args, **kwargs)

            cls.__init__ = new_init
            return cls

        return decorator

    def sunset(self, item_name: str, item_type: str = "function") -> bool:
        """
        Mark an item as sunset (no longer available).

        Args:
            item_name: Name of item
            item_type: Type of item

        Returns:
            True if item was sunset
        """
        key = f"{item_type}:{item_name}"

        if key in self._notices:
            notice = self._notices[key]
            notice.status = DeprecationStatus.SUNSET

            self._emit_event("sunset", {
                "item_name": item_name,
                "item_type": item_type,
            })

            self._save_notices()
            return True

        return False

    def get_notice(self, item_name: str, item_type: str = "function") -> Optional[DeprecationNotice]:
        """Get deprecation notice for an item."""
        key = f"{item_type}:{item_name}"
        return self._notices.get(key)

    def list_notices(
        self,
        status: Optional[DeprecationStatus] = None,
        severity: Optional[DeprecationSeverity] = None,
    ) -> List[DeprecationNotice]:
        """
        List deprecation notices.

        Args:
            status: Filter by status
            severity: Filter by severity

        Returns:
            List of notices
        """
        notices = list(self._notices.values())

        if status:
            notices = [n for n in notices if n.status == status]
        if severity:
            notices = [n for n in notices if n.severity == severity]

        return notices

    def generate_report(self) -> DeprecationReport:
        """Generate deprecation report."""
        report = DeprecationReport(
            total_deprecations=len(self._notices),
        )

        # Count by status
        for notice in self._notices.values():
            report.by_status[notice.status.value] = report.by_status.get(notice.status.value, 0) + 1
            report.by_severity[notice.severity.value] = report.by_severity.get(notice.severity.value, 0) + 1

            if notice.status == DeprecationStatus.DEPRECATED:
                report.active_deprecations += 1

        # Usage summary (top used deprecated items)
        sorted_notices = sorted(
            self._notices.values(),
            key=lambda n: n.usage_count,
            reverse=True,
        )
        report.usage_summary = [
            {
                "item_name": n.item_name,
                "item_type": n.item_type,
                "usage_count": n.usage_count,
                "last_used": datetime.fromtimestamp(n.last_used_at, tz=timezone.utc).isoformat() if n.last_used_at else None,
            }
            for n in sorted_notices[:10]
        ]

        # Upcoming sunsets
        now = time.time()
        for notice in self._notices.values():
            if notice.sunset_date:
                try:
                    sunset_dt = datetime.strptime(notice.sunset_date, "%Y-%m-%d")
                    if sunset_dt.timestamp() > now:
                        report.upcoming_sunsets.append({
                            "item_name": notice.item_name,
                            "item_type": notice.item_type,
                            "sunset_date": notice.sunset_date,
                            "alternative": notice.alternative,
                        })
                except ValueError:
                    pass

        report.upcoming_sunsets.sort(key=lambda x: x["sunset_date"])

        return report

    def check_for_sunsets(self) -> List[DeprecationNotice]:
        """Check for items that should be sunset."""
        now = time.time()
        sunset_notices = []

        for notice in self._notices.values():
            if notice.status != DeprecationStatus.DEPRECATED:
                continue

            if notice.sunset_date:
                try:
                    sunset_dt = datetime.strptime(notice.sunset_date, "%Y-%m-%d")
                    if sunset_dt.timestamp() <= now:
                        sunset_notices.append(notice)
                except ValueError:
                    pass

        return sunset_notices

    def _load_notices(self) -> None:
        """Load notices from disk."""
        notices_file = Path(self.config.output_dir) / "notices.json"

        if notices_file.exists():
            try:
                with open(notices_file) as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        data = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                for notice_dict in data:
                    notice = DeprecationNotice(
                        notice_id=notice_dict.get("notice_id", str(uuid.uuid4())),
                        item_type=notice_dict.get("item_type", "function"),
                        item_name=notice_dict.get("item_name", ""),
                        version_deprecated=notice_dict.get("version_deprecated", ""),
                        version_removed=notice_dict.get("version_removed"),
                        alternative=notice_dict.get("alternative"),
                        message=notice_dict.get("message", ""),
                        severity=DeprecationSeverity(notice_dict.get("severity", "medium")),
                        status=DeprecationStatus(notice_dict.get("status", "deprecated")),
                        created_at=notice_dict.get("created_at", time.time()),
                        sunset_date=notice_dict.get("sunset_date"),
                        usage_count=notice_dict.get("usage_count", 0),
                        tags=notice_dict.get("tags", []),
                    )
                    key = f"{notice.item_type}:{notice.item_name}"
                    self._notices[key] = notice

            except (IOError, json.JSONDecodeError):
                pass

    def _save_notices(self) -> None:
        """Save notices to disk."""
        notices_file = Path(self.config.output_dir) / "notices.json"

        notices_data = [n.to_dict() for n in self._notices.values()]

        try:
            with open(notices_file, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(notices_data, f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def _log_usage(self, usage_entry: Dict[str, Any]) -> None:
        """Log usage to file."""
        log_file = Path(self.config.output_dir) / self.config.log_file

        try:
            with open(log_file, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(usage_entry) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    async def generate_report_async(self) -> DeprecationReport:
        """Async version of generate_report."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_report)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.deprecation.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "deprecation",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Deprecation Manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Deprecation Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List deprecations")
    list_parser.add_argument("--status", choices=["active", "deprecated", "sunset", "removed"])
    list_parser.add_argument("--severity", choices=["low", "medium", "high", "critical"])

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate deprecation report")

    # Deprecate command
    deprecate_parser = subparsers.add_parser("deprecate", help="Deprecate an item")
    deprecate_parser.add_argument("name", help="Item name")
    deprecate_parser.add_argument("--type", default="function", help="Item type")
    deprecate_parser.add_argument("--version", help="Deprecated in version")
    deprecate_parser.add_argument("--alternative", help="Alternative to use")
    deprecate_parser.add_argument("--message", default="", help="Deprecation message")
    deprecate_parser.add_argument("--sunset", help="Sunset date (YYYY-MM-DD)")

    # Sunset command
    sunset_parser = subparsers.add_parser("sunset", help="Sunset a deprecated item")
    sunset_parser.add_argument("name", help="Item name")
    sunset_parser.add_argument("--type", default="function", help="Item type")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check for items to sunset")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/deprecation")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = DeprecationConfig(output_dir=args.output)
    manager = TestDeprecationManager(config=config)

    if args.command == "list":
        status = DeprecationStatus(args.status) if args.status else None
        severity = DeprecationSeverity(args.severity) if args.severity else None

        notices = manager.list_notices(status=status, severity=severity)

        if args.json:
            print(json.dumps([n.to_dict() for n in notices], indent=2))
        else:
            print(f"\nDeprecation Notices ({len(notices)}):")
            for notice in notices:
                status_str = f"[{notice.status.value.upper()}]"
                severity_str = f"({notice.severity.value})"
                print(f"\n  {status_str} {notice.item_type}:{notice.item_name} {severity_str}")
                print(f"    {notice.full_message}")
                if notice.usage_count > 0:
                    print(f"    Usage count: {notice.usage_count}")

    elif args.command == "report":
        report = manager.generate_report()

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print("\nDeprecation Report")
            print(f"\n  Total: {report.total_deprecations}")
            print(f"  Active: {report.active_deprecations}")

            print("\n  By Status:")
            for status, count in report.by_status.items():
                print(f"    {status}: {count}")

            print("\n  By Severity:")
            for severity, count in report.by_severity.items():
                print(f"    {severity}: {count}")

            if report.usage_summary:
                print("\n  Most Used Deprecated Items:")
                for item in report.usage_summary[:5]:
                    print(f"    {item['item_name']}: {item['usage_count']} uses")

            if report.upcoming_sunsets:
                print("\n  Upcoming Sunsets:")
                for item in report.upcoming_sunsets[:5]:
                    print(f"    {item['item_name']} -> {item['sunset_date']}")

    elif args.command == "deprecate":
        try:
            notice = manager.deprecate(
                item_name=args.name,
                item_type=args.type,
                version_deprecated=args.version or "",
                alternative=args.alternative,
                message=args.message,
                sunset_date=args.sunset,
            )
            print(f"Created deprecation notice: {notice.notice_id}")
            print(f"  {notice.full_message}")
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)

    elif args.command == "sunset":
        if manager.sunset(args.name, args.type):
            print(f"Sunset: {args.type}:{args.name}")
        else:
            print(f"Not found: {args.type}:{args.name}")

    elif args.command == "check":
        sunsets = manager.check_for_sunsets()

        if args.json:
            print(json.dumps([n.to_dict() for n in sunsets], indent=2))
        else:
            if sunsets:
                print(f"\nItems ready for sunset ({len(sunsets)}):")
                for notice in sunsets:
                    print(f"  - {notice.item_type}:{notice.item_name} (sunset: {notice.sunset_date})")
            else:
                print("No items ready for sunset")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
