#!/usr/bin/env python3
"""
deprecation_manager.py - Deprecation Manager (Step 49)

Manages deprecation lifecycle for APIs, features, and configurations.
Supports deprecation warnings, sunset periods, and migration guidance.

PBTSO Phase: TRANSITION

Bus Topics:
- a2a.research.deprecation.warn
- a2a.research.deprecation.sunset
- research.deprecation.register

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import fcntl
import functools
import json
import os
import socket
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class DeprecationPhase(Enum):
    """Phases of deprecation lifecycle."""
    ACTIVE = "active"           # Feature is current
    DEPRECATED = "deprecated"   # Feature deprecated, still works
    WARNING = "warning"         # Produces warnings
    SUNSET = "sunset"           # In sunset period
    REMOVED = "removed"         # Feature removed


class DeprecationType(Enum):
    """Types of deprecated items."""
    API_ENDPOINT = "api_endpoint"
    FUNCTION = "function"
    CLASS = "class"
    PARAMETER = "parameter"
    CONFIG_KEY = "config_key"
    FEATURE = "feature"
    BEHAVIOR = "behavior"


@dataclass
class DeprecationConfig:
    """Configuration for deprecation manager."""

    default_sunset_days: int = 90
    emit_warnings: bool = True
    warning_frequency: int = 1  # Warn every N calls
    log_deprecation_usage: bool = True
    strict_mode: bool = False  # Raise exceptions for deprecated items
    emit_to_bus: bool = True
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class DeprecationNotice:
    """A deprecation notice."""

    id: str
    name: str
    type: DeprecationType
    phase: DeprecationPhase
    message: str
    deprecated_in: str
    sunset_date: Optional[str] = None
    removed_in: Optional[str] = None
    replacement: Optional[str] = None
    migration_guide: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    call_count: int = 0
    last_warned: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "phase": self.phase.value,
            "message": self.message,
            "deprecated_in": self.deprecated_in,
            "sunset_date": self.sunset_date,
            "removed_in": self.removed_in,
            "replacement": self.replacement,
            "migration_guide": self.migration_guide,
            "call_count": self.call_count,
        }


@dataclass
class DeprecationUsage:
    """Record of deprecation usage."""

    deprecation_id: str
    timestamp: float
    caller_info: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deprecation_id": self.deprecation_id,
            "timestamp": self.timestamp,
            "caller_info": self.caller_info,
            "context": self.context,
        }


@dataclass
class DeprecationReport:
    """Report of deprecation usage."""

    generated_at: float
    total_deprecations: int = 0
    active_warnings: int = 0
    sunset_soon: int = 0
    most_used: List[Dict[str, Any]] = field(default_factory=list)
    by_type: Dict[str, int] = field(default_factory=dict)
    by_phase: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_at": self.generated_at,
            "total_deprecations": self.total_deprecations,
            "active_warnings": self.active_warnings,
            "sunset_soon": self.sunset_soon,
            "most_used": self.most_used,
            "by_type": self.by_type,
            "by_phase": self.by_phase,
        }


# ============================================================================
# Deprecation Manager
# ============================================================================


T = TypeVar("T")


class DeprecationManager:
    """
    Manages deprecation lifecycle for Research Agent.

    Features:
    - Deprecation registration
    - Warning emission
    - Usage tracking
    - Migration guidance
    - Sunset management

    PBTSO Phase: TRANSITION

    Example:
        deprecations = DeprecationManager()

        # Register deprecation
        deprecations.register(
            name="old_search",
            type=DeprecationType.FUNCTION,
            message="Use new_search instead",
            deprecated_in="1.5.0",
            replacement="new_search",
        )

        # Deprecation decorator
        @deprecations.deprecated(
            message="Use new_function",
            replacement="new_function",
        )
        def old_function():
            ...

        # Check and warn
        deprecations.warn_if_deprecated("old_search")
    """

    def __init__(
        self,
        config: Optional[DeprecationConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the deprecation manager.

        Args:
            config: Deprecation configuration
            bus: AgentBus for event emission
        """
        self.config = config or DeprecationConfig()
        self.bus = bus or AgentBus()

        # Deprecation registry
        self._deprecations: Dict[str, DeprecationNotice] = {}
        self._usage: List[DeprecationUsage] = []

        # Warning tracking
        self._warning_counts: Dict[str, int] = {}

        # Statistics
        self._stats = {
            "total_warnings": 0,
            "total_usages": 0,
            "suppressed_warnings": 0,
        }

    def register(
        self,
        name: str,
        type: DeprecationType,
        message: str,
        deprecated_in: str,
        sunset_date: Optional[str] = None,
        removed_in: Optional[str] = None,
        replacement: Optional[str] = None,
        migration_guide: Optional[str] = None,
    ) -> DeprecationNotice:
        """
        Register a deprecation notice.

        Args:
            name: Name of deprecated item
            type: Type of deprecation
            message: Deprecation message
            deprecated_in: Version deprecated
            sunset_date: Date of sunset (ISO format)
            removed_in: Version to be removed
            replacement: Suggested replacement
            migration_guide: URL or text for migration

        Returns:
            DeprecationNotice
        """
        notice = DeprecationNotice(
            id=str(uuid.uuid4())[:8],
            name=name,
            type=type,
            phase=DeprecationPhase.DEPRECATED,
            message=message,
            deprecated_in=deprecated_in,
            sunset_date=sunset_date,
            removed_in=removed_in,
            replacement=replacement,
            migration_guide=migration_guide,
        )

        # Determine phase based on sunset date
        if sunset_date:
            try:
                sunset = datetime.fromisoformat(sunset_date.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)

                if now > sunset:
                    notice.phase = DeprecationPhase.REMOVED
                elif (sunset - now).days < 30:
                    notice.phase = DeprecationPhase.SUNSET
                else:
                    notice.phase = DeprecationPhase.WARNING
            except ValueError:
                pass

        self._deprecations[name] = notice

        self._emit_event("research.deprecation.register", notice.to_dict())

        return notice

    def warn_if_deprecated(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[DeprecationNotice]:
        """
        Warn if an item is deprecated.

        Args:
            name: Name of item to check
            context: Additional context

        Returns:
            DeprecationNotice if deprecated, None otherwise
        """
        notice = self._deprecations.get(name)

        if not notice:
            return None

        if notice.phase == DeprecationPhase.ACTIVE:
            return None

        # Track usage
        notice.call_count += 1
        self._stats["total_usages"] += 1

        if self.config.log_deprecation_usage:
            self._record_usage(notice, context)

        # Check warning frequency
        self._warning_counts[name] = self._warning_counts.get(name, 0) + 1

        if self._warning_counts[name] % self.config.warning_frequency != 0:
            self._stats["suppressed_warnings"] += 1
            return notice

        # Emit warning
        if self.config.emit_warnings:
            self._emit_warning(notice)

        # Strict mode
        if self.config.strict_mode and notice.phase in [DeprecationPhase.SUNSET, DeprecationPhase.REMOVED]:
            raise DeprecationError(notice)

        return notice

    def deprecated(
        self,
        message: Optional[str] = None,
        deprecated_in: str = "0.0.0",
        replacement: Optional[str] = None,
        sunset_date: Optional[str] = None,
    ) -> Callable[[T], T]:
        """
        Decorator to mark functions/methods as deprecated.

        Args:
            message: Deprecation message
            deprecated_in: Version deprecated
            replacement: Suggested replacement
            sunset_date: Sunset date

        Example:
            @deprecations.deprecated(
                message="Use new_search",
                replacement="new_search",
            )
            def old_search():
                ...
        """
        def decorator(func: T) -> T:
            func_name = getattr(func, "__name__", str(func))
            msg = message or f"{func_name} is deprecated"

            # Register deprecation
            self.register(
                name=func_name,
                type=DeprecationType.FUNCTION,
                message=msg,
                deprecated_in=deprecated_in,
                replacement=replacement,
                sunset_date=sunset_date,
            )

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.warn_if_deprecated(func_name)
                return func(*args, **kwargs)

            return wrapper  # type: ignore

        return decorator

    def deprecated_param(
        self,
        param_name: str,
        message: Optional[str] = None,
        replacement: Optional[str] = None,
    ) -> Callable[[T], T]:
        """
        Decorator to mark function parameters as deprecated.

        Args:
            param_name: Name of deprecated parameter
            message: Deprecation message
            replacement: Suggested replacement

        Example:
            @deprecations.deprecated_param("old_param", replacement="new_param")
            def my_function(new_param=None, old_param=None):
                ...
        """
        def decorator(func: T) -> T:
            func_name = getattr(func, "__name__", str(func))
            msg = message or f"Parameter '{param_name}' in {func_name} is deprecated"

            name = f"{func_name}.{param_name}"
            self.register(
                name=name,
                type=DeprecationType.PARAMETER,
                message=msg,
                deprecated_in="0.0.0",
                replacement=replacement,
            )

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if param_name in kwargs:
                    self.warn_if_deprecated(name)
                return func(*args, **kwargs)

            return wrapper  # type: ignore

        return decorator

    def deprecated_class(
        self,
        message: Optional[str] = None,
        deprecated_in: str = "0.0.0",
        replacement: Optional[str] = None,
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to mark classes as deprecated.

        Args:
            message: Deprecation message
            deprecated_in: Version deprecated
            replacement: Suggested replacement

        Example:
            @deprecations.deprecated_class(replacement="NewClass")
            class OldClass:
                ...
        """
        def decorator(cls: Type[T]) -> Type[T]:
            class_name = cls.__name__
            msg = message or f"Class {class_name} is deprecated"

            self.register(
                name=class_name,
                type=DeprecationType.CLASS,
                message=msg,
                deprecated_in=deprecated_in,
                replacement=replacement,
            )

            original_init = cls.__init__

            @functools.wraps(original_init)
            def new_init(self, *args, **kwargs):
                self.warn_if_deprecated(class_name)
                original_init(self, *args, **kwargs)

            cls.__init__ = new_init  # type: ignore
            return cls

        return decorator

    def is_deprecated(self, name: str) -> bool:
        """Check if an item is deprecated."""
        notice = self._deprecations.get(name)
        return notice is not None and notice.phase != DeprecationPhase.ACTIVE

    def get_notice(self, name: str) -> Optional[DeprecationNotice]:
        """Get deprecation notice for an item."""
        return self._deprecations.get(name)

    def list_deprecations(
        self,
        phase: Optional[DeprecationPhase] = None,
        type: Optional[DeprecationType] = None,
    ) -> List[DeprecationNotice]:
        """
        List all deprecation notices.

        Args:
            phase: Filter by phase
            type: Filter by type

        Returns:
            List of DeprecationNotice
        """
        notices = list(self._deprecations.values())

        if phase:
            notices = [n for n in notices if n.phase == phase]

        if type:
            notices = [n for n in notices if n.type == type]

        return notices

    def get_sunset_soon(self, days: int = 30) -> List[DeprecationNotice]:
        """
        Get deprecations sunsetting soon.

        Args:
            days: Number of days threshold

        Returns:
            List of DeprecationNotice
        """
        now = datetime.now(timezone.utc)
        sunset_soon = []

        for notice in self._deprecations.values():
            if notice.sunset_date:
                try:
                    sunset = datetime.fromisoformat(notice.sunset_date.replace("Z", "+00:00"))
                    if 0 < (sunset - now).days <= days:
                        sunset_soon.append(notice)
                except ValueError:
                    pass

        return sunset_soon

    def generate_report(self) -> DeprecationReport:
        """Generate deprecation usage report."""
        notices = list(self._deprecations.values())

        # Count by type
        by_type: Dict[str, int] = {}
        for notice in notices:
            by_type[notice.type.value] = by_type.get(notice.type.value, 0) + 1

        # Count by phase
        by_phase: Dict[str, int] = {}
        for notice in notices:
            by_phase[notice.phase.value] = by_phase.get(notice.phase.value, 0) + 1

        # Most used
        sorted_by_usage = sorted(notices, key=lambda n: n.call_count, reverse=True)
        most_used = [
            {"name": n.name, "count": n.call_count, "type": n.type.value}
            for n in sorted_by_usage[:10]
        ]

        # Sunset soon
        sunset_soon = len(self.get_sunset_soon())

        # Active warnings
        active_warnings = sum(
            1 for n in notices
            if n.phase in [DeprecationPhase.WARNING, DeprecationPhase.DEPRECATED]
        )

        return DeprecationReport(
            generated_at=time.time(),
            total_deprecations=len(notices),
            active_warnings=active_warnings,
            sunset_soon=sunset_soon,
            most_used=most_used,
            by_type=by_type,
            by_phase=by_phase,
        )

    def generate_migration_guide(self, name: str) -> str:
        """
        Generate migration guide for a deprecation.

        Args:
            name: Deprecation name

        Returns:
            Migration guide text
        """
        notice = self._deprecations.get(name)
        if not notice:
            return f"No deprecation found for: {name}"

        lines = [
            f"# Migration Guide: {notice.name}",
            "",
            f"**Status:** {notice.phase.value.upper()}",
            f"**Deprecated in:** {notice.deprecated_in}",
        ]

        if notice.sunset_date:
            lines.append(f"**Sunset date:** {notice.sunset_date}")

        if notice.removed_in:
            lines.append(f"**Removed in:** {notice.removed_in}")

        lines.extend(["", "## Description", "", notice.message])

        if notice.replacement:
            lines.extend([
                "",
                "## Migration",
                "",
                f"Replace `{notice.name}` with `{notice.replacement}`.",
            ])

        if notice.migration_guide:
            lines.extend([
                "",
                "## Additional Guidance",
                "",
                notice.migration_guide,
            ])

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get deprecation statistics."""
        return {
            **self._stats,
            "total_registered": len(self._deprecations),
            "active_warnings": len([
                d for d in self._deprecations.values()
                if d.phase in [DeprecationPhase.WARNING, DeprecationPhase.DEPRECATED]
            ]),
        }

    def _record_usage(
        self,
        notice: DeprecationNotice,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record deprecation usage."""
        import traceback

        # Get caller info
        stack = traceback.extract_stack()
        caller_info = {}
        if len(stack) > 3:
            frame = stack[-4]
            caller_info = {
                "file": frame.filename,
                "line": frame.lineno,
                "function": frame.name,
            }

        usage = DeprecationUsage(
            deprecation_id=notice.id,
            timestamp=time.time(),
            caller_info=caller_info,
            context=context or {},
        )

        self._usage.append(usage)

        # Limit usage history
        if len(self._usage) > 10000:
            self._usage = self._usage[-5000:]

    def _emit_warning(self, notice: DeprecationNotice) -> None:
        """Emit deprecation warning."""
        self._stats["total_warnings"] += 1
        notice.last_warned = time.time()

        # Python warning
        msg = notice.message
        if notice.replacement:
            msg += f" Use {notice.replacement} instead."

        warnings.warn(msg, DeprecationWarning, stacklevel=4)

        # Bus event
        if self.config.emit_to_bus:
            self._emit_event("a2a.research.deprecation.warn", {
                "name": notice.name,
                "type": notice.type.value,
                "phase": notice.phase.value,
                "message": notice.message,
                "replacement": notice.replacement,
            }, level="warning")

    def _emit_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
    ) -> str:
        """Emit event with file locking."""
        if not self.config.emit_to_bus:
            return ""

        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "deprecation",
            "level": level,
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id


class DeprecationError(Exception):
    """Exception raised when using removed/sunset features in strict mode."""

    def __init__(self, notice: DeprecationNotice):
        self.notice = notice
        msg = f"{notice.name} is {notice.phase.value}"
        if notice.replacement:
            msg += f". Use {notice.replacement} instead."
        super().__init__(msg)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Deprecation Manager."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Deprecation Manager (Step 49)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command
    list_parser = subparsers.add_parser("list", help="List deprecations")
    list_parser.add_argument("--phase", choices=["active", "deprecated", "warning", "sunset", "removed"])
    list_parser.add_argument("--type")
    list_parser.add_argument("--json", action="store_true")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check deprecation status")
    check_parser.add_argument("name", help="Item name")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument("--json", action="store_true")

    # Sunset command
    sunset_parser = subparsers.add_parser("sunset", help="Show sunset soon")
    sunset_parser.add_argument("--days", type=int, default=30)

    # Migration command
    migration_parser = subparsers.add_parser("migration", help="Generate migration guide")
    migration_parser.add_argument("name", help="Deprecation name")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run deprecation demo")

    args = parser.parse_args()

    deprecations = DeprecationManager()

    if args.command == "list":
        phase = DeprecationPhase(args.phase) if args.phase else None
        dep_type = DeprecationType(args.type) if args.type else None

        notices = deprecations.list_deprecations(phase, dep_type)

        if args.json:
            print(json.dumps([n.to_dict() for n in notices], indent=2))
        else:
            print(f"Deprecations ({len(notices)}):")
            for notice in notices:
                print(f"  [{notice.phase.value}] {notice.name} ({notice.type.value})")
                print(f"    {notice.message}")
                if notice.replacement:
                    print(f"    -> {notice.replacement}")

    elif args.command == "check":
        notice = deprecations.get_notice(args.name)
        if notice:
            print(f"Name: {notice.name}")
            print(f"Type: {notice.type.value}")
            print(f"Phase: {notice.phase.value}")
            print(f"Message: {notice.message}")
            if notice.replacement:
                print(f"Replacement: {notice.replacement}")
            if notice.sunset_date:
                print(f"Sunset: {notice.sunset_date}")
        else:
            print(f"No deprecation found for: {args.name}")
            return 1

    elif args.command == "report":
        report = deprecations.generate_report()

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print("Deprecation Report")
            print(f"  Total: {report.total_deprecations}")
            print(f"  Active Warnings: {report.active_warnings}")
            print(f"  Sunset Soon: {report.sunset_soon}")

            if report.by_type:
                print("  By Type:")
                for t, count in report.by_type.items():
                    print(f"    {t}: {count}")

            if report.most_used:
                print("  Most Used:")
                for item in report.most_used[:5]:
                    print(f"    {item['name']}: {item['count']} calls")

    elif args.command == "sunset":
        notices = deprecations.get_sunset_soon(args.days)
        print(f"Sunsetting within {args.days} days ({len(notices)}):")
        for notice in notices:
            print(f"  {notice.name}: {notice.sunset_date}")

    elif args.command == "migration":
        guide = deprecations.generate_migration_guide(args.name)
        print(guide)

    elif args.command == "stats":
        stats = deprecations.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Deprecation Statistics:")
            print(f"  Registered: {stats['total_registered']}")
            print(f"  Active Warnings: {stats['active_warnings']}")
            print(f"  Total Warnings: {stats['total_warnings']}")
            print(f"  Total Usages: {stats['total_usages']}")

    elif args.command == "demo":
        print("Running deprecation demo...\n")

        # Register deprecations
        print("Registering deprecations:")

        deprecations.register(
            name="old_search_function",
            type=DeprecationType.FUNCTION,
            message="old_search_function is deprecated",
            deprecated_in="1.5.0",
            replacement="new_search_function",
            sunset_date="2025-12-31",
        )
        print("  - old_search_function")

        deprecations.register(
            name="legacy_config",
            type=DeprecationType.CONFIG_KEY,
            message="legacy_config key is deprecated",
            deprecated_in="1.0.0",
            removed_in="2.0.0",
            replacement="new_config",
        )
        print("  - legacy_config")

        # Use decorator
        @deprecations.deprecated(
            message="Use new_function instead",
            replacement="new_function",
        )
        def old_function():
            return "old"

        print("  - old_function (via decorator)")

        # Suppress warnings for demo
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            # Check deprecations
            print("\nChecking deprecations:")
            notice = deprecations.warn_if_deprecated("old_search_function")
            print(f"  old_search_function: {notice.phase.value if notice else 'not deprecated'}")

            # Call deprecated function
            old_function()
            print("  old_function called (warning suppressed)")

        # List all
        print("\nAll deprecations:")
        for notice in deprecations.list_deprecations():
            print(f"  [{notice.phase.value}] {notice.name} -> {notice.replacement}")

        # Generate report
        print("\n--- Deprecation Report ---")
        report = deprecations.generate_report()
        print(f"Total: {report.total_deprecations}")
        print(f"Active Warnings: {report.active_warnings}")

        # Migration guide
        print("\n--- Migration Guide ---")
        print(deprecations.generate_migration_guide("old_search_function"))

        print("\nDemo complete.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
