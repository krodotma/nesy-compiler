#!/usr/bin/env python3
"""
Monitor Versioning - Step 298

API versioning system for the Monitor Agent.

PBTSO Phase: SKILL

Bus Topics:
- monitor.version.changed (emitted)
- monitor.version.deprecated (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class VersionStatus(Enum):
    """API version status."""
    CURRENT = "current"
    SUPPORTED = "supported"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"
    RETIRED = "retired"


class ChangeType(Enum):
    """API change types."""
    BREAKING = "breaking"
    FEATURE = "feature"
    FIX = "fix"
    DEPRECATION = "deprecation"


@dataclass
class SemanticVersion:
    """Semantic version representation.

    Attributes:
        major: Major version
        minor: Minor version
        patch: Patch version
        prerelease: Pre-release identifier
        build: Build metadata
    """
    major: int
    minor: int
    patch: int
    prerelease: str = ""
    build: str = ""

    def __str__(self) -> str:
        """Convert to string."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __lt__(self, other: "SemanticVersion") -> bool:
        """Compare versions."""
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        # Pre-release has lower precedence
        if self.prerelease and not other.prerelease:
            return True
        if not self.prerelease and other.prerelease:
            return False
        return self.prerelease < other.prerelease

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, SemanticVersion):
            return False
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse version string."""
        pattern = r"^v?(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$"
        match = re.match(pattern, version_str)
        if not match:
            raise ValueError(f"Invalid version: {version_str}")

        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
            prerelease=match.group(4) or "",
            build=match.group(5) or "",
        )

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
class APIVersion:
    """An API version definition.

    Attributes:
        version: Version identifier (e.g., "v1", "v2")
        semantic_version: Full semantic version
        status: Version status
        release_date: Release date
        sunset_date: Sunset date (if deprecated)
        changes: List of changes
        endpoints: Supported endpoints
    """
    version: str
    semantic_version: SemanticVersion
    status: VersionStatus
    release_date: float = field(default_factory=time.time)
    sunset_date: Optional[float] = None
    changes: List[Dict[str, Any]] = field(default_factory=list)
    endpoints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "semantic_version": self.semantic_version.to_dict(),
            "status": self.status.value,
            "release_date": self.release_date,
            "sunset_date": self.sunset_date,
            "changes": self.changes,
            "endpoint_count": len(self.endpoints),
        }


@dataclass
class VersionChange:
    """A version change entry.

    Attributes:
        change_type: Type of change
        description: Change description
        affected_endpoints: Affected endpoints
        migration_guide: Migration instructions
    """
    change_type: ChangeType
    description: str
    affected_endpoints: List[str] = field(default_factory=list)
    migration_guide: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.change_type.value,
            "description": self.description,
            "affected_endpoints": self.affected_endpoints,
            "migration_guide": self.migration_guide,
        }


class MonitorVersioning:
    """
    API versioning system for the Monitor Agent.

    Provides:
    - Semantic versioning
    - Version negotiation
    - Deprecation management
    - Version-specific routing
    - Migration guides

    Example:
        versioning = MonitorVersioning()

        # Get current version
        current = versioning.get_current_version()

        # Check version compatibility
        compatible = versioning.is_compatible("v1", "v2")

        # Resolve version from request
        resolved = versioning.resolve_version(
            requested="v1",
            accept_header="application/vnd.monitor.v2+json",
        )
    """

    BUS_TOPICS = {
        "changed": "monitor.version.changed",
        "deprecated": "monitor.version.deprecated",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        default_version: str = "v1",
        bus_dir: Optional[str] = None,
    ):
        """Initialize versioning system.

        Args:
            default_version: Default API version
            bus_dir: Bus directory
        """
        self._default_version = default_version
        self._last_heartbeat = time.time()

        # Version registry
        self._versions: Dict[str, APIVersion] = {}
        self._current_version: Optional[str] = None

        # Version aliases
        self._aliases: Dict[str, str] = {}

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register default versions
        self._register_default_versions()

    def register_version(
        self,
        version: str,
        semantic_version: str,
        status: VersionStatus = VersionStatus.CURRENT,
        changes: Optional[List[VersionChange]] = None,
        endpoints: Optional[List[str]] = None,
    ) -> APIVersion:
        """Register an API version.

        Args:
            version: Version identifier
            semantic_version: Semantic version string
            status: Version status
            changes: Version changes
            endpoints: Supported endpoints

        Returns:
            Registered version
        """
        sem_ver = SemanticVersion.parse(semantic_version)

        api_version = APIVersion(
            version=version,
            semantic_version=sem_ver,
            status=status,
            changes=[c.to_dict() for c in (changes or [])],
            endpoints=endpoints or [],
        )

        self._versions[version] = api_version

        if status == VersionStatus.CURRENT:
            self._current_version = version

        return api_version

    def deprecate_version(
        self,
        version: str,
        sunset_date: Optional[float] = None,
        replacement: Optional[str] = None,
    ) -> bool:
        """Deprecate an API version.

        Args:
            version: Version to deprecate
            sunset_date: Sunset timestamp
            replacement: Replacement version

        Returns:
            True if deprecated
        """
        if version not in self._versions:
            return False

        api_version = self._versions[version]
        api_version.status = VersionStatus.DEPRECATED
        api_version.sunset_date = sunset_date or (time.time() + 180 * 86400)  # 180 days

        self._emit_bus_event(
            self.BUS_TOPICS["deprecated"],
            {
                "version": version,
                "sunset_date": api_version.sunset_date,
                "replacement": replacement,
            },
            level="warning",
        )

        return True

    def retire_version(self, version: str) -> bool:
        """Retire an API version.

        Args:
            version: Version to retire

        Returns:
            True if retired
        """
        if version not in self._versions:
            return False

        self._versions[version].status = VersionStatus.RETIRED

        self._emit_bus_event(
            self.BUS_TOPICS["changed"],
            {
                "version": version,
                "status": VersionStatus.RETIRED.value,
            },
        )

        return True

    def get_version(self, version: str) -> Optional[APIVersion]:
        """Get API version details.

        Args:
            version: Version identifier

        Returns:
            API version or None
        """
        # Check aliases
        version = self._aliases.get(version, version)
        return self._versions.get(version)

    def get_current_version(self) -> Optional[APIVersion]:
        """Get current API version.

        Returns:
            Current version or None
        """
        if self._current_version:
            return self._versions.get(self._current_version)
        return None

    def list_versions(
        self,
        include_retired: bool = False,
    ) -> List[Dict[str, Any]]:
        """List all API versions.

        Args:
            include_retired: Include retired versions

        Returns:
            Version list
        """
        versions = []
        for version in self._versions.values():
            if not include_retired and version.status == VersionStatus.RETIRED:
                continue
            versions.append(version.to_dict())

        return sorted(versions, key=lambda v: v["version"], reverse=True)

    def resolve_version(
        self,
        requested: Optional[str] = None,
        accept_header: Optional[str] = None,
    ) -> Tuple[str, APIVersion]:
        """Resolve version from request.

        Args:
            requested: Explicitly requested version
            accept_header: Accept header value

        Returns:
            Tuple of (version_id, api_version)
        """
        # Try explicit request
        if requested:
            version = self._resolve_requested(requested)
            if version:
                return requested, version

        # Try Accept header
        if accept_header:
            version_id = self._parse_accept_header(accept_header)
            if version_id:
                version = self.get_version(version_id)
                if version:
                    return version_id, version

        # Fall back to default/current
        default = self._default_version
        version = self.get_version(default)
        if version:
            return default, version

        # Last resort: return first available
        for v_id, v in self._versions.items():
            if v.status != VersionStatus.RETIRED:
                return v_id, v

        raise ValueError("No available API version")

    def is_compatible(
        self,
        version_a: str,
        version_b: str,
    ) -> bool:
        """Check if two versions are compatible.

        Args:
            version_a: First version
            version_b: Second version

        Returns:
            True if compatible
        """
        va = self.get_version(version_a)
        vb = self.get_version(version_b)

        if not va or not vb:
            return False

        # Same major version is compatible
        return va.semantic_version.major == vb.semantic_version.major

    def get_migration_guide(
        self,
        from_version: str,
        to_version: str,
    ) -> Dict[str, Any]:
        """Get migration guide between versions.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            Migration guide
        """
        from_v = self.get_version(from_version)
        to_v = self.get_version(to_version)

        if not from_v or not to_v:
            return {"error": "Version not found"}

        # Collect all changes between versions
        changes = []
        for version in self._versions.values():
            if (
                from_v.semantic_version < version.semantic_version
                <= to_v.semantic_version
            ):
                for change in version.changes:
                    if change.get("migration_guide"):
                        changes.append({
                            "version": version.version,
                            **change,
                        })

        return {
            "from_version": from_version,
            "to_version": to_version,
            "breaking_changes": [
                c for c in changes if c.get("type") == ChangeType.BREAKING.value
            ],
            "all_changes": changes,
            "compatible": self.is_compatible(from_version, to_version),
        }

    def add_alias(self, alias: str, version: str) -> None:
        """Add a version alias.

        Args:
            alias: Alias name (e.g., "latest")
            version: Target version
        """
        self._aliases[alias] = version

    def get_deprecation_warnings(self, version: str) -> List[str]:
        """Get deprecation warnings for a version.

        Args:
            version: Version identifier

        Returns:
            List of warnings
        """
        api_version = self.get_version(version)
        if not api_version:
            return []

        warnings = []

        if api_version.status == VersionStatus.DEPRECATED:
            sunset = api_version.sunset_date
            if sunset:
                sunset_date = datetime.fromtimestamp(sunset).strftime("%Y-%m-%d")
                warnings.append(f"API version {version} is deprecated and will be retired on {sunset_date}")
            else:
                warnings.append(f"API version {version} is deprecated")

        if api_version.status == VersionStatus.SUNSET:
            warnings.append(f"API version {version} is in sunset period and will be retired soon")

        return warnings

    def get_statistics(self) -> Dict[str, Any]:
        """Get versioning statistics.

        Returns:
            Statistics
        """
        return {
            "total_versions": len(self._versions),
            "current_version": self._current_version,
            "default_version": self._default_version,
            "aliases": self._aliases,
            "by_status": {
                s.value: sum(1 for v in self._versions.values() if v.status == s)
                for s in VersionStatus
            },
        }

    def _resolve_requested(self, requested: str) -> Optional[APIVersion]:
        """Resolve explicitly requested version."""
        # Check aliases first
        requested = self._aliases.get(requested, requested)

        version = self._versions.get(requested)
        if version and version.status != VersionStatus.RETIRED:
            return version

        return None

    def _parse_accept_header(self, header: str) -> Optional[str]:
        """Parse version from Accept header."""
        # Parse: application/vnd.monitor.v2+json
        pattern = r"application/vnd\.monitor\.([a-z0-9]+)\+json"
        match = re.search(pattern, header, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _register_default_versions(self) -> None:
        """Register default API versions."""
        # Version 1 - Current
        self.register_version(
            version="v1",
            semantic_version="1.0.0",
            status=VersionStatus.CURRENT,
            endpoints=[
                "/api/v1/health",
                "/api/v1/metrics",
                "/api/v1/metrics/query",
                "/api/v1/alerts",
                "/api/v1/alerts/{id}",
                "/api/v1/dashboards",
                "/api/v1/reports",
            ],
        )

        # Add common aliases
        self.add_alias("latest", "v1")
        self.add_alias("stable", "v1")

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
                "component": "monitor_versioning",
                "status": "healthy",
                "current_version": self._current_version,
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
            "actor": "monitor-versioning",
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
_versioning: Optional[MonitorVersioning] = None


def get_versioning() -> MonitorVersioning:
    """Get or create the versioning system singleton.

    Returns:
        MonitorVersioning instance
    """
    global _versioning
    if _versioning is None:
        _versioning = MonitorVersioning()
    return _versioning


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Versioning (Step 298)")
    parser.add_argument("--list", action="store_true", help="List all versions")
    parser.add_argument("--current", action="store_true", help="Show current version")
    parser.add_argument("--resolve", metavar="VERSION", help="Resolve version")
    parser.add_argument("--compatible", nargs=2, metavar=("V1", "V2"), help="Check compatibility")
    parser.add_argument("--migration", nargs=2, metavar=("FROM", "TO"), help="Get migration guide")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    versioning = get_versioning()

    if args.list:
        versions = versioning.list_versions()
        if args.json:
            print(json.dumps(versions, indent=2))
        else:
            print("API Versions:")
            for v in versions:
                status = v["status"]
                print(f"  {v['version']}: {v['semantic_version']['string']} [{status}]")

    if args.current:
        current = versioning.get_current_version()
        if args.json:
            print(json.dumps(current.to_dict() if current else None, indent=2))
        else:
            if current:
                print(f"Current Version: {current.version}")
                print(f"  Semantic: {current.semantic_version}")
                print(f"  Endpoints: {len(current.endpoints)}")

    if args.resolve:
        try:
            version_id, version = versioning.resolve_version(requested=args.resolve)
            if args.json:
                print(json.dumps({"resolved": version_id, "version": version.to_dict()}, indent=2))
            else:
                print(f"Resolved: {version_id} -> {version.semantic_version}")
        except ValueError as e:
            print(f"Error: {e}")

    if args.compatible:
        v1, v2 = args.compatible
        compatible = versioning.is_compatible(v1, v2)
        if args.json:
            print(json.dumps({"v1": v1, "v2": v2, "compatible": compatible}))
        else:
            print(f"Compatibility {v1} <-> {v2}: {'compatible' if compatible else 'incompatible'}")

    if args.migration:
        from_v, to_v = args.migration
        guide = versioning.get_migration_guide(from_v, to_v)
        if args.json:
            print(json.dumps(guide, indent=2))
        else:
            print(f"Migration Guide: {from_v} -> {to_v}")
            print(f"  Compatible: {guide.get('compatible')}")
            print(f"  Breaking Changes: {len(guide.get('breaking_changes', []))}")

    if args.stats:
        stats = versioning.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Versioning Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
