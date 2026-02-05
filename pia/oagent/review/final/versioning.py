#!/usr/bin/env python3
"""
Versioning System (Step 198)

API versioning system for the Review Agent with version routing,
compatibility checking, and deprecation support.

PBTSO Phase: BUILD
Bus Topics: review.version.request, review.version.deprecation

Versioning Features:
- Semantic versioning support
- Version routing
- Compatibility checking
- Version negotiation
- Deprecation headers

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900


# ============================================================================
# Types
# ============================================================================

class VersionStatus(Enum):
    """API version status."""
    CURRENT = "current"         # Recommended version
    SUPPORTED = "supported"     # Fully supported
    DEPRECATED = "deprecated"   # Still works but being phased out
    SUNSET = "sunset"           # No longer available


class CompatibilityLevel(Enum):
    """Compatibility between versions."""
    FULL = "full"               # Fully compatible
    BACKWARD = "backward"       # Backward compatible
    BREAKING = "breaking"       # Has breaking changes
    INCOMPATIBLE = "incompatible"  # Not compatible


@dataclass
class APIVersion:
    """
    API version definition.

    Attributes:
        major: Major version number
        minor: Minor version number
        patch: Patch version number
        status: Version status
        release_date: Release date
        sunset_date: Planned sunset date
        description: Version description
        changelog: Version changelog
    """
    major: int
    minor: int
    patch: int
    status: VersionStatus = VersionStatus.SUPPORTED
    release_date: str = ""
    sunset_date: Optional[str] = None
    description: str = ""
    changelog: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.release_date:
            self.release_date = datetime.now(timezone.utc).isoformat() + "Z"

    @property
    def version_string(self) -> str:
        """Get version string (e.g., '1.2.3')."""
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def api_version(self) -> str:
        """Get API version string (e.g., 'v1')."""
        return f"v{self.major}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version_string,
            "api_version": self.api_version,
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "status": self.status.value,
            "release_date": self.release_date,
            "sunset_date": self.sunset_date,
            "description": self.description,
            "changelog": self.changelog,
        }

    @classmethod
    def parse(cls, version_str: str) -> "APIVersion":
        """Parse version from string."""
        # Handle 'v1', 'v1.2', 'v1.2.3' formats
        version_str = version_str.lstrip("v")
        parts = version_str.split(".")

        major = int(parts[0]) if len(parts) > 0 else 1
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0

        return cls(major=major, minor=minor, patch=patch)

    def __lt__(self, other: "APIVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, APIVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))


@dataclass
class VersionCompatibility:
    """
    Version compatibility information.

    Attributes:
        from_version: Source version
        to_version: Target version
        level: Compatibility level
        breaking_changes: List of breaking changes
        migration_notes: Migration notes
    """
    from_version: str
    to_version: str
    level: CompatibilityLevel
    breaking_changes: List[str] = field(default_factory=list)
    migration_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "level": self.level.value,
            "breaking_changes": self.breaking_changes,
            "migration_notes": self.migration_notes,
        }


@dataclass
class VersionRequest:
    """
    Incoming version request.

    Attributes:
        requested_version: Requested version
        accept_header: Accept header value
        api_version_header: API-Version header
        user_agent: User agent string
    """
    requested_version: Optional[str] = None
    accept_header: Optional[str] = None
    api_version_header: Optional[str] = None
    user_agent: Optional[str] = None

    def get_version(self) -> Optional[str]:
        """Get requested version from headers."""
        # Priority: explicit header > accept header > URL version
        if self.api_version_header:
            return self.api_version_header
        if self.accept_header:
            # Parse 'application/vnd.review.v1+json'
            match = re.search(r'v(\d+(?:\.\d+)*)', self.accept_header)
            if match:
                return match.group(0)
        return self.requested_version


# ============================================================================
# Version Router
# ============================================================================

class VersionRouter:
    """
    Routes requests to appropriate version handlers.

    Example:
        router = VersionRouter()

        # Register version handlers
        router.register("v1", handler_v1)
        router.register("v2", handler_v2)

        # Route request
        handler = router.route("v1")
    """

    def __init__(self):
        """Initialize version router."""
        self._handlers: Dict[str, Callable] = {}
        self._default_version: Optional[str] = None

    def register(
        self,
        version: str,
        handler: Callable,
        is_default: bool = False,
    ) -> None:
        """
        Register a version handler.

        Args:
            version: Version string (e.g., 'v1', 'v2')
            handler: Handler callable
            is_default: Set as default version
        """
        self._handlers[version] = handler
        if is_default:
            self._default_version = version

    def route(self, version: Optional[str] = None) -> Optional[Callable]:
        """
        Route to appropriate handler.

        Args:
            version: Requested version

        Returns:
            Handler callable or None
        """
        if version and version in self._handlers:
            return self._handlers[version]
        if self._default_version:
            return self._handlers.get(self._default_version)
        return None

    def get_versions(self) -> List[str]:
        """Get all registered versions."""
        return list(self._handlers.keys())


# ============================================================================
# Versioning System
# ============================================================================

class VersioningSystem:
    """
    Complete API versioning system.

    Example:
        versioning = VersioningSystem()

        # Register versions
        versioning.add_version(APIVersion(1, 0, 0, status=VersionStatus.CURRENT))
        versioning.add_version(APIVersion(2, 0, 0, status=VersionStatus.SUPPORTED))

        # Check compatibility
        compat = versioning.check_compatibility("v1", "v2")

        # Resolve version from request
        version = versioning.resolve_version(request)
    """

    BUS_TOPICS = {
        "request": "review.version.request",
        "deprecation": "review.version.deprecation",
    }

    def __init__(
        self,
        bus_path: Optional[Path] = None,
    ):
        """Initialize versioning system."""
        self.bus_path = bus_path or self._get_bus_path()

        self._versions: Dict[str, APIVersion] = {}
        self._compatibility: Dict[tuple, VersionCompatibility] = {}
        self._current_version: Optional[APIVersion] = None
        self.router = VersionRouter()

        self._last_heartbeat = time.time()

        # Register default versions
        self._register_default_versions()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "versioning") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "versioning-system",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _register_default_versions(self) -> None:
        """Register default API versions."""
        # v1 - Initial version
        self.add_version(APIVersion(
            major=1,
            minor=0,
            patch=0,
            status=VersionStatus.SUPPORTED,
            description="Initial API version",
            changelog=["Initial release"],
        ))

        # v2 - Current version
        v2 = APIVersion(
            major=2,
            minor=0,
            patch=0,
            status=VersionStatus.CURRENT,
            description="Current API version with enhanced features",
            changelog=[
                "Added async review support",
                "Enhanced security scanning",
                "Improved performance",
            ],
        )
        self.add_version(v2)
        self._current_version = v2

        # Set up compatibility
        self.add_compatibility(VersionCompatibility(
            from_version="v1",
            to_version="v2",
            level=CompatibilityLevel.BACKWARD,
            breaking_changes=[
                "Response format changed for /api/review endpoint",
            ],
            migration_notes=[
                "Update client to handle new response format",
                "Use Accept header to specify version",
            ],
        ))

    def add_version(self, version: APIVersion) -> None:
        """Add a version."""
        self._versions[version.api_version] = version
        self._versions[version.version_string] = version

    def get_version(self, version_str: str) -> Optional[APIVersion]:
        """Get version by string."""
        return self._versions.get(version_str)

    def get_current_version(self) -> Optional[APIVersion]:
        """Get current (recommended) version."""
        return self._current_version

    def get_all_versions(self) -> List[APIVersion]:
        """Get all unique versions."""
        seen = set()
        versions = []
        for v in self._versions.values():
            if v.version_string not in seen:
                seen.add(v.version_string)
                versions.append(v)
        return sorted(versions, reverse=True)

    def add_compatibility(self, compat: VersionCompatibility) -> None:
        """Add compatibility information."""
        key = (compat.from_version, compat.to_version)
        self._compatibility[key] = compat

    def check_compatibility(
        self,
        from_version: str,
        to_version: str,
    ) -> VersionCompatibility:
        """
        Check compatibility between versions.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            VersionCompatibility
        """
        key = (from_version, to_version)
        if key in self._compatibility:
            return self._compatibility[key]

        # Parse versions and determine compatibility
        from_v = APIVersion.parse(from_version)
        to_v = APIVersion.parse(to_version)

        if from_v.major == to_v.major:
            level = CompatibilityLevel.FULL if from_v.minor == to_v.minor else CompatibilityLevel.BACKWARD
        elif from_v.major < to_v.major:
            level = CompatibilityLevel.BREAKING
        else:
            level = CompatibilityLevel.INCOMPATIBLE

        return VersionCompatibility(
            from_version=from_version,
            to_version=to_version,
            level=level,
        )

    def resolve_version(
        self,
        request: VersionRequest,
    ) -> Tuple[APIVersion, Dict[str, str]]:
        """
        Resolve version from request.

        Args:
            request: Version request

        Returns:
            Tuple of (resolved version, headers to add)
        """
        headers = {}
        requested = request.get_version()

        if requested:
            version = self.get_version(requested)
            if version:
                # Check if deprecated
                if version.status == VersionStatus.DEPRECATED:
                    headers["Deprecation"] = version.sunset_date or "true"
                    headers["Sunset"] = version.sunset_date or ""

                    self._emit_event(self.BUS_TOPICS["deprecation"], {
                        "requested_version": requested,
                        "status": version.status.value,
                        "sunset_date": version.sunset_date,
                    })

                return version, headers

        # Return current version
        version = self._current_version or list(self._versions.values())[0]
        headers["API-Version"] = version.api_version

        self._emit_event(self.BUS_TOPICS["request"], {
            "requested_version": requested,
            "resolved_version": version.api_version,
        })

        return version, headers

    def deprecate_version(
        self,
        version_str: str,
        sunset_date: str,
    ) -> bool:
        """
        Mark a version as deprecated.

        Args:
            version_str: Version to deprecate
            sunset_date: Sunset date (ISO format)

        Returns:
            True if version was deprecated
        """
        version = self.get_version(version_str)
        if not version:
            return False

        version.status = VersionStatus.DEPRECATED
        version.sunset_date = sunset_date

        self._emit_event(self.BUS_TOPICS["deprecation"], {
            "version": version_str,
            "sunset_date": sunset_date,
            "action": "deprecated",
        })

        return True

    def sunset_version(self, version_str: str) -> bool:
        """
        Mark a version as sunset (no longer available).

        Args:
            version_str: Version to sunset

        Returns:
            True if version was sunset
        """
        version = self.get_version(version_str)
        if not version:
            return False

        version.status = VersionStatus.SUNSET

        self._emit_event(self.BUS_TOPICS["deprecation"], {
            "version": version_str,
            "action": "sunset",
        })

        return True

    def get_version_headers(self, version: APIVersion) -> Dict[str, str]:
        """Get HTTP headers for a version."""
        headers = {
            "API-Version": version.api_version,
            "X-API-Version": version.version_string,
        }

        if version.status == VersionStatus.DEPRECATED:
            headers["Deprecation"] = version.sunset_date or "true"
            if version.sunset_date:
                headers["Sunset"] = version.sunset_date

        return headers

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        versions = self.get_all_versions()
        current = self.get_current_version()

        status = {
            "agent": "versioning-system",
            "healthy": True,
            "versions_registered": len(versions),
            "current_version": current.version_string if current else None,
            "deprecated_versions": len([v for v in versions if v.status == VersionStatus.DEPRECATED]),
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
    """CLI entry point for Versioning System."""
    import argparse

    parser = argparse.ArgumentParser(description="Versioning System (Step 198)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    subparsers.add_parser("list", help="List all versions")

    # Current command
    subparsers.add_parser("current", help="Show current version")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check compatibility")
    check_parser.add_argument("from_version", help="Source version")
    check_parser.add_argument("to_version", help="Target version")

    # Deprecate command
    deprecate_parser = subparsers.add_parser("deprecate", help="Deprecate version")
    deprecate_parser.add_argument("version", help="Version to deprecate")
    deprecate_parser.add_argument("--sunset", required=True, help="Sunset date")

    # Resolve command
    resolve_parser = subparsers.add_parser("resolve", help="Resolve version from request")
    resolve_parser.add_argument("--version", help="Requested version")
    resolve_parser.add_argument("--accept", help="Accept header")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    versioning = VersioningSystem()

    if args.command == "list":
        versions = versioning.get_all_versions()
        if args.json:
            print(json.dumps([v.to_dict() for v in versions], indent=2))
        else:
            print(f"API Versions: {len(versions)}")
            for v in versions:
                status = f" ({v.status.value})" if v.status != VersionStatus.SUPPORTED else ""
                current = " [CURRENT]" if v == versioning.get_current_version() else ""
                print(f"  {v.api_version} ({v.version_string}){status}{current}")
                if v.description:
                    print(f"    {v.description}")

    elif args.command == "current":
        current = versioning.get_current_version()
        if args.json:
            print(json.dumps(current.to_dict() if current else None, indent=2))
        else:
            if current:
                print(f"Current Version: {current.api_version} ({current.version_string})")
                print(f"  Status: {current.status.value}")
                print(f"  Released: {current.release_date}")
            else:
                print("No current version set")

    elif args.command == "check":
        compat = versioning.check_compatibility(args.from_version, args.to_version)
        if args.json:
            print(json.dumps(compat.to_dict(), indent=2))
        else:
            print(f"Compatibility: {args.from_version} -> {args.to_version}")
            print(f"  Level: {compat.level.value}")
            if compat.breaking_changes:
                print("  Breaking Changes:")
                for change in compat.breaking_changes:
                    print(f"    - {change}")
            if compat.migration_notes:
                print("  Migration Notes:")
                for note in compat.migration_notes:
                    print(f"    - {note}")

    elif args.command == "deprecate":
        success = versioning.deprecate_version(args.version, args.sunset)
        if success:
            print(f"Version {args.version} deprecated. Sunset: {args.sunset}")
        else:
            print(f"Version {args.version} not found")
            return 1

    elif args.command == "resolve":
        request = VersionRequest(
            requested_version=args.version,
            accept_header=args.accept,
        )
        version, headers = versioning.resolve_version(request)
        if args.json:
            print(json.dumps({
                "version": version.to_dict(),
                "headers": headers,
            }, indent=2))
        else:
            print(f"Resolved Version: {version.api_version} ({version.version_string})")
            if headers:
                print("Headers:")
                for k, v in headers.items():
                    print(f"  {k}: {v}")

    else:
        # Default: show status
        status = versioning.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Versioning: {status['versions_registered']} versions, current: {status['current_version']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
