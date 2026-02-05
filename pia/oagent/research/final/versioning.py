#!/usr/bin/env python3
"""
versioning.py - API Versioning System (Step 48)

API versioning and compatibility management for Research Agent.
Supports semantic versioning, version negotiation, and compatibility checking.

PBTSO Phase: INTERFACE

Bus Topics:
- a2a.research.version.check
- a2a.research.version.negotiate
- research.version.register

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class VersionScheme(Enum):
    """Versioning schemes."""
    SEMANTIC = "semantic"  # Major.Minor.Patch
    CALVER = "calver"      # Calendar-based
    INTEGER = "integer"    # Simple integer


class CompatibilityLevel(Enum):
    """Compatibility levels."""
    FULL = "full"               # Fully compatible
    BACKWARD = "backward"       # Old clients work
    FORWARD = "forward"         # New clients work with old
    BREAKING = "breaking"       # Incompatible change


@dataclass
class VersionConfig:
    """Configuration for versioning system."""

    scheme: VersionScheme = VersionScheme.SEMANTIC
    current_version: str = "1.0.0"
    min_supported_version: str = "1.0.0"
    default_version: str = "1.0.0"
    version_header: str = "X-API-Version"
    version_param: str = "api_version"
    strict_mode: bool = False  # Reject unsupported versions
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
class SemanticVersion:
    """Semantic version representation."""

    major: int = 0
    minor: int = 0
    patch: int = 0
    prerelease: Optional[str] = None
    build: Optional[str] = None

    @classmethod
    def parse(cls, version_string: str) -> "SemanticVersion":
        """Parse a version string."""
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$"
        match = re.match(pattern, version_string)

        if not match:
            raise ValueError(f"Invalid semantic version: {version_string}")

        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4),
            build=match.group(5),
        )

    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: "SemanticVersion") -> bool:
        if (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch):
            return True
        if (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch):
            return False
        # Same version, check prerelease
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        return (self.prerelease or "") < (other.prerelease or "")

    def __le__(self, other: "SemanticVersion") -> bool:
        return self < other or self == other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch, self.prerelease))

    def is_compatible(self, other: "SemanticVersion") -> bool:
        """Check if versions are compatible (same major version)."""
        return self.major == other.major

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "prerelease": self.prerelease,
            "build": self.build,
            "string": str(self),
        }


@dataclass
class VersionedEndpoint:
    """An API endpoint with version information."""

    path: str
    method: str
    min_version: SemanticVersion
    max_version: Optional[SemanticVersion] = None
    deprecated_in: Optional[SemanticVersion] = None
    removed_in: Optional[SemanticVersion] = None
    handler: Optional[Callable] = None
    description: str = ""

    def is_available(self, version: SemanticVersion) -> bool:
        """Check if endpoint is available for version."""
        if version < self.min_version:
            return False
        if self.max_version and version > self.max_version:
            return False
        if self.removed_in and version >= self.removed_in:
            return False
        return True

    def is_deprecated(self, version: SemanticVersion) -> bool:
        """Check if endpoint is deprecated for version."""
        if self.deprecated_in and version >= self.deprecated_in:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "method": self.method,
            "min_version": str(self.min_version),
            "max_version": str(self.max_version) if self.max_version else None,
            "deprecated_in": str(self.deprecated_in) if self.deprecated_in else None,
            "removed_in": str(self.removed_in) if self.removed_in else None,
            "description": self.description,
        }


@dataclass
class VersionNegotiationResult:
    """Result of version negotiation."""

    success: bool
    negotiated_version: Optional[SemanticVersion] = None
    requested_version: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "negotiated_version": str(self.negotiated_version) if self.negotiated_version else None,
            "requested_version": self.requested_version,
            "error": self.error,
            "warnings": self.warnings,
        }


@dataclass
class ChangelogEntry:
    """A changelog entry."""

    version: SemanticVersion
    date: str
    changes: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)
    deprecations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": str(self.version),
            "date": self.date,
            "changes": self.changes,
            "breaking_changes": self.breaking_changes,
            "deprecations": self.deprecations,
        }


# ============================================================================
# Version Manager
# ============================================================================


T = TypeVar("T")


class VersionManager:
    """
    API version manager for Research Agent.

    Features:
    - Semantic versioning
    - Version negotiation
    - Endpoint versioning
    - Compatibility checking
    - Changelog management

    PBTSO Phase: INTERFACE

    Example:
        versions = VersionManager()

        # Register versioned endpoint
        @versions.versioned("1.0.0", deprecated_in="2.0.0")
        def search_v1(query: str):
            ...

        # Check version compatibility
        if versions.is_compatible("1.5.0", "2.0.0"):
            ...

        # Negotiate version
        result = versions.negotiate("1.5.0")
    """

    def __init__(
        self,
        config: Optional[VersionConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the version manager.

        Args:
            config: Version configuration
            bus: AgentBus for event emission
        """
        self.config = config or VersionConfig()
        self.bus = bus or AgentBus()

        # Parse configured versions
        self._current = SemanticVersion.parse(self.config.current_version)
        self._min_supported = SemanticVersion.parse(self.config.min_supported_version)
        self._default = SemanticVersion.parse(self.config.default_version)

        # Endpoint registry
        self._endpoints: Dict[str, List[VersionedEndpoint]] = {}

        # Changelog
        self._changelog: List[ChangelogEntry] = []

        # Statistics
        self._stats = {
            "negotiations": 0,
            "successful_negotiations": 0,
            "failed_negotiations": 0,
            "deprecated_calls": 0,
        }

    @property
    def current_version(self) -> SemanticVersion:
        """Get current API version."""
        return self._current

    @property
    def min_supported_version(self) -> SemanticVersion:
        """Get minimum supported version."""
        return self._min_supported

    def negotiate(
        self,
        requested: Optional[str] = None,
        accept_header: Optional[str] = None,
    ) -> VersionNegotiationResult:
        """
        Negotiate API version with client.

        Args:
            requested: Requested version string
            accept_header: Accept header with version info

        Returns:
            VersionNegotiationResult with negotiated version
        """
        self._stats["negotiations"] += 1

        # Parse requested version
        if requested:
            try:
                requested_version = SemanticVersion.parse(requested)
            except ValueError:
                self._stats["failed_negotiations"] += 1
                return VersionNegotiationResult(
                    success=False,
                    requested_version=requested,
                    error=f"Invalid version format: {requested}",
                )
        elif accept_header:
            # Parse Accept header for version
            version_match = re.search(r"version=(\d+\.\d+\.\d+)", accept_header)
            if version_match:
                requested_version = SemanticVersion.parse(version_match.group(1))
            else:
                requested_version = self._default
        else:
            requested_version = self._default

        # Check if supported
        if requested_version < self._min_supported:
            if self.config.strict_mode:
                self._stats["failed_negotiations"] += 1
                return VersionNegotiationResult(
                    success=False,
                    requested_version=str(requested_version),
                    error=f"Version {requested_version} is no longer supported. Minimum: {self._min_supported}",
                )
            else:
                # Upgrade to minimum
                result = VersionNegotiationResult(
                    success=True,
                    negotiated_version=self._min_supported,
                    requested_version=str(requested_version),
                    warnings=[f"Requested version {requested_version} upgraded to {self._min_supported}"],
                )
        elif requested_version > self._current:
            # Downgrade to current
            result = VersionNegotiationResult(
                success=True,
                negotiated_version=self._current,
                requested_version=str(requested_version),
                warnings=[f"Requested version {requested_version} not available, using {self._current}"],
            )
        else:
            result = VersionNegotiationResult(
                success=True,
                negotiated_version=requested_version,
                requested_version=str(requested_version),
            )

        self._stats["successful_negotiations"] += 1

        self._emit_event("a2a.research.version.negotiate", result.to_dict())

        return result

    def is_compatible(
        self,
        version1: str,
        version2: str,
    ) -> bool:
        """
        Check if two versions are compatible.

        Args:
            version1: First version
            version2: Second version

        Returns:
            True if versions are compatible
        """
        v1 = SemanticVersion.parse(version1)
        v2 = SemanticVersion.parse(version2)
        return v1.is_compatible(v2)

    def is_supported(self, version: str) -> bool:
        """Check if version is currently supported."""
        try:
            v = SemanticVersion.parse(version)
            return v >= self._min_supported and v <= self._current
        except ValueError:
            return False

    def register_endpoint(
        self,
        path: str,
        method: str,
        min_version: str,
        max_version: Optional[str] = None,
        deprecated_in: Optional[str] = None,
        removed_in: Optional[str] = None,
        handler: Optional[Callable] = None,
        description: str = "",
    ) -> VersionedEndpoint:
        """
        Register a versioned endpoint.

        Args:
            path: Endpoint path
            method: HTTP method
            min_version: Minimum version
            max_version: Maximum version
            deprecated_in: Version when deprecated
            removed_in: Version when removed
            handler: Endpoint handler
            description: Endpoint description

        Returns:
            VersionedEndpoint
        """
        endpoint = VersionedEndpoint(
            path=path,
            method=method,
            min_version=SemanticVersion.parse(min_version),
            max_version=SemanticVersion.parse(max_version) if max_version else None,
            deprecated_in=SemanticVersion.parse(deprecated_in) if deprecated_in else None,
            removed_in=SemanticVersion.parse(removed_in) if removed_in else None,
            handler=handler,
            description=description,
        )

        key = f"{method}:{path}"
        if key not in self._endpoints:
            self._endpoints[key] = []
        self._endpoints[key].append(endpoint)

        self._emit_event("research.version.register", {
            "path": path,
            "method": method,
            "min_version": min_version,
        })

        return endpoint

    def get_endpoint(
        self,
        path: str,
        method: str,
        version: str,
    ) -> Optional[VersionedEndpoint]:
        """
        Get endpoint for specific version.

        Args:
            path: Endpoint path
            method: HTTP method
            version: API version

        Returns:
            VersionedEndpoint if found and available
        """
        key = f"{method}:{path}"
        endpoints = self._endpoints.get(key, [])
        v = SemanticVersion.parse(version)

        for endpoint in endpoints:
            if endpoint.is_available(v):
                if endpoint.is_deprecated(v):
                    self._stats["deprecated_calls"] += 1
                return endpoint

        return None

    def list_endpoints(
        self,
        version: Optional[str] = None,
    ) -> List[VersionedEndpoint]:
        """
        List all endpoints, optionally filtered by version.

        Args:
            version: Optional version filter

        Returns:
            List of VersionedEndpoint
        """
        all_endpoints = []

        for endpoints in self._endpoints.values():
            for endpoint in endpoints:
                if version:
                    v = SemanticVersion.parse(version)
                    if endpoint.is_available(v):
                        all_endpoints.append(endpoint)
                else:
                    all_endpoints.append(endpoint)

        return all_endpoints

    def versioned(
        self,
        min_version: str,
        max_version: Optional[str] = None,
        deprecated_in: Optional[str] = None,
    ) -> Callable[[T], T]:
        """
        Decorator to version a function/endpoint.

        Args:
            min_version: Minimum version
            max_version: Maximum version
            deprecated_in: Version when deprecated

        Example:
            @versions.versioned("1.0.0", deprecated_in="2.0.0")
            def my_endpoint():
                ...
        """
        def decorator(func: T) -> T:
            # Store version info on function
            func._version_info = {  # type: ignore
                "min_version": min_version,
                "max_version": max_version,
                "deprecated_in": deprecated_in,
            }

            @wraps(func)
            def wrapper(*args, version: Optional[str] = None, **kwargs):
                # Check version if provided
                if version:
                    v = SemanticVersion.parse(version)
                    min_v = SemanticVersion.parse(min_version)

                    if v < min_v:
                        raise ValueError(f"Function requires version >= {min_version}")

                    if max_version:
                        max_v = SemanticVersion.parse(max_version)
                        if v > max_v:
                            raise ValueError(f"Function not available in version {version}")

                    if deprecated_in:
                        dep_v = SemanticVersion.parse(deprecated_in)
                        if v >= dep_v:
                            import warnings
                            warnings.warn(
                                f"This function is deprecated since version {deprecated_in}",
                                DeprecationWarning,
                            )

                return func(*args, **kwargs)

            return wrapper  # type: ignore

        return decorator

    def add_changelog_entry(
        self,
        version: str,
        date: str,
        changes: Optional[List[str]] = None,
        breaking_changes: Optional[List[str]] = None,
        deprecations: Optional[List[str]] = None,
    ) -> ChangelogEntry:
        """
        Add a changelog entry.

        Args:
            version: Version string
            date: Release date
            changes: List of changes
            breaking_changes: List of breaking changes
            deprecations: List of deprecations

        Returns:
            ChangelogEntry
        """
        entry = ChangelogEntry(
            version=SemanticVersion.parse(version),
            date=date,
            changes=changes or [],
            breaking_changes=breaking_changes or [],
            deprecations=deprecations or [],
        )

        self._changelog.append(entry)
        self._changelog.sort(key=lambda e: e.version, reverse=True)

        return entry

    def get_changelog(
        self,
        since_version: Optional[str] = None,
        limit: int = 10,
    ) -> List[ChangelogEntry]:
        """
        Get changelog entries.

        Args:
            since_version: Only entries after this version
            limit: Maximum entries to return

        Returns:
            List of ChangelogEntry
        """
        entries = self._changelog

        if since_version:
            since = SemanticVersion.parse(since_version)
            entries = [e for e in entries if e.version > since]

        return entries[:limit]

    def generate_changelog_markdown(self) -> str:
        """Generate changelog as Markdown."""
        lines = ["# Changelog\n"]

        for entry in self._changelog:
            lines.append(f"## [{entry.version}] - {entry.date}\n")

            if entry.breaking_changes:
                lines.append("### Breaking Changes\n")
                for change in entry.breaking_changes:
                    lines.append(f"- {change}")
                lines.append("")

            if entry.changes:
                lines.append("### Changes\n")
                for change in entry.changes:
                    lines.append(f"- {change}")
                lines.append("")

            if entry.deprecations:
                lines.append("### Deprecations\n")
                for dep in entry.deprecations:
                    lines.append(f"- {dep}")
                lines.append("")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get version manager statistics."""
        return {
            **self._stats,
            "current_version": str(self._current),
            "min_supported": str(self._min_supported),
            "registered_endpoints": sum(len(eps) for eps in self._endpoints.values()),
            "changelog_entries": len(self._changelog),
        }

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
            "kind": "version",
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


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Versioning."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Versioning System (Step 48)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Check command
    check_parser = subparsers.add_parser("check", help="Check version")
    check_parser.add_argument("version", help="Version to check")

    # Negotiate command
    negotiate_parser = subparsers.add_parser("negotiate", help="Negotiate version")
    negotiate_parser.add_argument("version", help="Requested version")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare versions")
    compare_parser.add_argument("v1", help="First version")
    compare_parser.add_argument("v2", help="Second version")

    # Current command
    current_parser = subparsers.add_parser("current", help="Show current version")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run versioning demo")

    args = parser.parse_args()

    config = VersionConfig(
        current_version="2.0.0",
        min_supported_version="1.0.0",
    )
    versions = VersionManager(config)

    if args.command == "check":
        supported = versions.is_supported(args.version)
        print(f"Version {args.version}: {'SUPPORTED' if supported else 'NOT SUPPORTED'}")
        print(f"  Current: {versions.current_version}")
        print(f"  Minimum: {versions.min_supported_version}")
        return 0 if supported else 1

    elif args.command == "negotiate":
        result = versions.negotiate(args.version)
        print(f"Negotiation: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"  Requested: {result.requested_version}")
        print(f"  Negotiated: {result.negotiated_version}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")
        if result.error:
            print(f"  Error: {result.error}")
        return 0 if result.success else 1

    elif args.command == "compare":
        v1 = SemanticVersion.parse(args.v1)
        v2 = SemanticVersion.parse(args.v2)

        if v1 < v2:
            print(f"{v1} < {v2}")
        elif v1 > v2:
            print(f"{v1} > {v2}")
        else:
            print(f"{v1} == {v2}")

        print(f"Compatible: {v1.is_compatible(v2)}")

    elif args.command == "current":
        print(f"Current Version: {versions.current_version}")
        print(f"Minimum Supported: {versions.min_supported_version}")

    elif args.command == "stats":
        stats = versions.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Versioning Statistics:")
            print(f"  Current Version: {stats['current_version']}")
            print(f"  Min Supported: {stats['min_supported']}")
            print(f"  Negotiations: {stats['negotiations']}")
            print(f"  Successful: {stats['successful_negotiations']}")
            print(f"  Failed: {stats['failed_negotiations']}")

    elif args.command == "demo":
        print("Running versioning demo...\n")

        # Parse versions
        print("Parsing semantic versions:")
        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("2.0.0-beta.1")
        v3 = SemanticVersion.parse("2.0.0")
        print(f"  {v1}, {v2}, {v3}")

        # Compare
        print(f"\nComparisons:")
        print(f"  {v1} < {v2}: {v1 < v2}")
        print(f"  {v2} < {v3}: {v2 < v3}")
        print(f"  {v1}.is_compatible({v3}): {v1.is_compatible(v3)}")

        # Version negotiation
        print("\nVersion negotiation:")
        result = versions.negotiate("1.5.0")
        print(f"  Requested 1.5.0: {result.negotiated_version}")

        result = versions.negotiate("0.5.0")
        print(f"  Requested 0.5.0: {result.negotiated_version} (warnings: {result.warnings})")

        result = versions.negotiate("3.0.0")
        print(f"  Requested 3.0.0: {result.negotiated_version} (warnings: {result.warnings})")

        # Register endpoints
        print("\nRegistering versioned endpoints:")
        versions.register_endpoint(
            path="/api/search",
            method="POST",
            min_version="1.0.0",
            description="Search endpoint",
        )
        versions.register_endpoint(
            path="/api/search/v2",
            method="POST",
            min_version="2.0.0",
            description="Search v2 endpoint",
        )
        versions.register_endpoint(
            path="/api/legacy",
            method="GET",
            min_version="1.0.0",
            deprecated_in="1.5.0",
            removed_in="2.0.0",
            description="Legacy endpoint",
        )

        endpoints = versions.list_endpoints()
        print(f"  Registered {len(endpoints)} endpoints")

        # Check endpoint availability
        print("\nEndpoint availability:")
        ep = versions.get_endpoint("/api/search", "POST", "1.0.0")
        print(f"  /api/search@1.0.0: {'available' if ep else 'not available'}")

        ep = versions.get_endpoint("/api/search/v2", "POST", "1.5.0")
        print(f"  /api/search/v2@1.5.0: {'available' if ep else 'not available'}")

        ep = versions.get_endpoint("/api/legacy", "GET", "1.8.0")
        print(f"  /api/legacy@1.8.0: {'available (deprecated)' if ep and ep.is_deprecated(SemanticVersion.parse('1.8.0')) else 'available'}")

        # Changelog
        print("\nChangelog:")
        versions.add_changelog_entry(
            "1.0.0",
            "2024-01-01",
            changes=["Initial release", "Basic search functionality"],
        )
        versions.add_changelog_entry(
            "1.5.0",
            "2024-06-01",
            changes=["Improved search", "New analysis features"],
            deprecations=["Deprecated /api/legacy endpoint"],
        )
        versions.add_changelog_entry(
            "2.0.0",
            "2025-01-01",
            changes=["Major rewrite", "New API structure"],
            breaking_changes=["Removed /api/legacy endpoint"],
        )

        print(versions.generate_changelog_markdown()[:500] + "...")

        print("\nDemo complete.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
