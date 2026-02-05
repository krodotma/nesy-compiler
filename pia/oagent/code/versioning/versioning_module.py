#!/usr/bin/env python3
"""
versioning_module.py - Versioning Module (Step 98)

PBTSO Phase: SKILL, ITERATE

Provides:
- Semantic versioning
- API version management
- Version negotiation
- Backward compatibility
- Version lifecycle

Bus Topics:
- code.version.register
- code.version.negotiate
- code.version.deprecate

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

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
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class VersioningStrategy(Enum):
    """API versioning strategies."""
    URL_PATH = "url_path"        # /v1/resource
    HEADER = "header"           # Accept: application/vnd.api.v1+json
    QUERY_PARAM = "query_param"  # /resource?version=1
    MEDIA_TYPE = "media_type"    # Accept: application/vnd.api+json;version=1


class VersionStatus(Enum):
    """Version lifecycle status."""
    ALPHA = "alpha"
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"


@dataclass
class VersionConfig:
    """Configuration for versioning module."""
    strategy: VersioningStrategy = VersioningStrategy.URL_PATH
    default_version: str = "1.0.0"
    supported_versions: List[str] = field(default_factory=lambda: ["1.0.0"])
    version_header: str = "X-API-Version"
    version_param: str = "version"
    strict_mode: bool = False
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "default_version": self.default_version,
            "supported_versions": self.supported_versions,
            "strict_mode": self.strict_mode,
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
# Version Types
# =============================================================================

@dataclass
class SemanticVersion:
    """Semantic version representation."""
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    @classmethod
    def parse(cls, version_string: str) -> "SemanticVersion":
        """Parse a version string into SemanticVersion."""
        # Match semver pattern
        pattern = r"^v?(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.]+))?(?:\+([a-zA-Z0-9.]+))?$"
        match = re.match(pattern, version_string)
        if not match:
            raise ValueError(f"Invalid version string: {version_string}")

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
        return self._compare(other) < 0

    def __le__(self, other: "SemanticVersion") -> bool:
        return self._compare(other) <= 0

    def __gt__(self, other: "SemanticVersion") -> bool:
        return self._compare(other) > 0

    def __ge__(self, other: "SemanticVersion") -> bool:
        return self._compare(other) >= 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return self._compare(other) == 0

    def _compare(self, other: "SemanticVersion") -> int:
        if self.major != other.major:
            return self.major - other.major
        if self.minor != other.minor:
            return self.minor - other.minor
        if self.patch != other.patch:
            return self.patch - other.patch

        # Prerelease comparison
        if self.prerelease and not other.prerelease:
            return -1
        if not self.prerelease and other.prerelease:
            return 1
        if self.prerelease and other.prerelease:
            return (self.prerelease > other.prerelease) - (self.prerelease < other.prerelease)

        return 0

    def is_compatible_with(self, other: "SemanticVersion") -> bool:
        """Check if this version is compatible with another (same major version)."""
        return self.major == other.major

    def to_dict(self) -> Dict[str, Any]:
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
    """An API version definition."""
    version: SemanticVersion
    status: VersionStatus
    release_date: Optional[float] = None
    deprecation_date: Optional[float] = None
    sunset_date: Optional[float] = None
    changelog: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)

    def is_stable(self) -> bool:
        return self.status == VersionStatus.STABLE

    def is_deprecated(self) -> bool:
        return self.status in (VersionStatus.DEPRECATED, VersionStatus.SUNSET)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": str(self.version),
            "status": self.status.value,
            "release_date": self.release_date,
            "deprecation_date": self.deprecation_date,
            "sunset_date": self.sunset_date,
            "changelog": self.changelog,
            "breaking_changes": self.breaking_changes,
        }


@dataclass
class VersionedEndpoint:
    """An endpoint with version information."""
    path: str
    method: str
    handler: Callable
    min_version: SemanticVersion
    max_version: Optional[SemanticVersion] = None
    deprecated_in: Optional[SemanticVersion] = None
    removed_in: Optional[SemanticVersion] = None

    def supports_version(self, version: SemanticVersion) -> bool:
        """Check if endpoint supports the given version."""
        if version < self.min_version:
            return False
        if self.max_version and version > self.max_version:
            return False
        if self.removed_in and version >= self.removed_in:
            return False
        return True

    def is_deprecated_for(self, version: SemanticVersion) -> bool:
        """Check if endpoint is deprecated for the given version."""
        if self.deprecated_in and version >= self.deprecated_in:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "method": self.method,
            "min_version": str(self.min_version),
            "max_version": str(self.max_version) if self.max_version else None,
            "deprecated_in": str(self.deprecated_in) if self.deprecated_in else None,
        }


# =============================================================================
# Version Negotiator
# =============================================================================

class VersionNegotiator:
    """Negotiates API version from request."""

    def __init__(self, config: VersionConfig):
        self.config = config

    def negotiate(
        self,
        requested: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        path: Optional[str] = None,
    ) -> Tuple[SemanticVersion, bool]:
        """
        Negotiate version from request.

        Returns:
            Tuple of (version, is_explicit)
        """
        version_string = None

        if self.config.strategy == VersioningStrategy.URL_PATH:
            version_string = self._extract_from_path(path)
        elif self.config.strategy == VersioningStrategy.HEADER:
            version_string = self._extract_from_header(headers)
        elif self.config.strategy == VersioningStrategy.QUERY_PARAM:
            version_string = self._extract_from_params(params)
        elif self.config.strategy == VersioningStrategy.MEDIA_TYPE:
            version_string = self._extract_from_media_type(headers)

        if version_string is None and requested:
            version_string = requested

        if version_string:
            try:
                return SemanticVersion.parse(version_string), True
            except ValueError:
                pass

        # Return default
        return SemanticVersion.parse(self.config.default_version), False

    def _extract_from_path(self, path: Optional[str]) -> Optional[str]:
        """Extract version from URL path (/v1/...)."""
        if not path:
            return None

        match = re.match(r"^/v(\d+(?:\.\d+(?:\.\d+)?)?)/", path)
        if match:
            version = match.group(1)
            # Normalize to semver
            parts = version.split(".")
            while len(parts) < 3:
                parts.append("0")
            return ".".join(parts)
        return None

    def _extract_from_header(self, headers: Optional[Dict[str, str]]) -> Optional[str]:
        """Extract version from header."""
        if not headers:
            return None
        return headers.get(self.config.version_header)

    def _extract_from_params(self, params: Optional[Dict[str, str]]) -> Optional[str]:
        """Extract version from query params."""
        if not params:
            return None
        return params.get(self.config.version_param)

    def _extract_from_media_type(self, headers: Optional[Dict[str, str]]) -> Optional[str]:
        """Extract version from Accept media type."""
        if not headers:
            return None

        accept = headers.get("Accept", "")
        match = re.search(r"version=(\d+(?:\.\d+(?:\.\d+)?)?)", accept)
        if match:
            return match.group(1)
        return None


# =============================================================================
# Versioning Module
# =============================================================================

class VersioningModule:
    """
    Versioning module for API version management.

    PBTSO Phase: SKILL, ITERATE

    Features:
    - Semantic versioning
    - Version negotiation
    - Endpoint version routing
    - Deprecation management
    - Version lifecycle

    Usage:
        versioning = VersioningModule()

        # Register version
        versioning.register_version("1.0.0", VersionStatus.STABLE)
        versioning.register_version("2.0.0", VersionStatus.BETA)

        # Register endpoint
        versioning.register_endpoint("/users", "GET", handler, min_version="1.0.0")

        # Negotiate version
        version = versioning.negotiate_version(headers={"X-API-Version": "1.5.0"})
    """

    BUS_TOPICS = {
        "register": "code.version.register",
        "negotiate": "code.version.negotiate",
        "deprecate": "code.version.deprecate",
    }

    def __init__(
        self,
        config: Optional[VersionConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or VersionConfig()
        self.bus = bus or LockedAgentBus()

        self._versions: Dict[str, APIVersion] = {}
        self._endpoints: List[VersionedEndpoint] = []
        self._negotiator = VersionNegotiator(self.config)
        self._lock = Lock()

        # Register default versions
        for v in self.config.supported_versions:
            self.register_version(v, VersionStatus.STABLE)

    # =========================================================================
    # Version Management
    # =========================================================================

    def register_version(
        self,
        version: str,
        status: VersionStatus,
        changelog: Optional[List[str]] = None,
        breaking_changes: Optional[List[str]] = None,
    ) -> APIVersion:
        """Register an API version."""
        semver = SemanticVersion.parse(version)

        api_version = APIVersion(
            version=semver,
            status=status,
            release_date=time.time() if status != VersionStatus.ALPHA else None,
            changelog=changelog or [],
            breaking_changes=breaking_changes or [],
        )

        with self._lock:
            self._versions[str(semver)] = api_version

        self.bus.emit({
            "topic": self.BUS_TOPICS["register"],
            "kind": "version",
            "actor": "versioning-module",
            "data": api_version.to_dict(),
        })

        return api_version

    def deprecate_version(
        self,
        version: str,
        sunset_date: Optional[float] = None,
    ) -> bool:
        """Mark a version as deprecated."""
        with self._lock:
            api_version = self._versions.get(version)
            if not api_version:
                return False

            api_version.status = VersionStatus.DEPRECATED
            api_version.deprecation_date = time.time()
            api_version.sunset_date = sunset_date

        self.bus.emit({
            "topic": self.BUS_TOPICS["deprecate"],
            "kind": "version",
            "actor": "versioning-module",
            "data": {
                "version": version,
                "deprecation_date": api_version.deprecation_date,
                "sunset_date": sunset_date,
            },
        })

        return True

    def sunset_version(self, version: str) -> bool:
        """Mark a version as sunset (end of life)."""
        with self._lock:
            api_version = self._versions.get(version)
            if not api_version:
                return False

            api_version.status = VersionStatus.SUNSET
            api_version.sunset_date = time.time()

        return True

    def get_version(self, version: str) -> Optional[APIVersion]:
        """Get an API version."""
        return self._versions.get(version)

    def get_current_version(self) -> SemanticVersion:
        """Get the current (latest stable) version."""
        stable_versions = [
            v for v in self._versions.values()
            if v.is_stable()
        ]
        if stable_versions:
            return max(v.version for v in stable_versions)
        return SemanticVersion.parse(self.config.default_version)

    def list_versions(self, include_deprecated: bool = False) -> List[APIVersion]:
        """List all versions."""
        versions = list(self._versions.values())
        if not include_deprecated:
            versions = [v for v in versions if not v.is_deprecated()]
        return sorted(versions, key=lambda v: v.version, reverse=True)

    # =========================================================================
    # Endpoint Management
    # =========================================================================

    def register_endpoint(
        self,
        path: str,
        method: str,
        handler: Callable,
        min_version: str = "1.0.0",
        max_version: Optional[str] = None,
        deprecated_in: Optional[str] = None,
    ) -> VersionedEndpoint:
        """Register a versioned endpoint."""
        endpoint = VersionedEndpoint(
            path=path,
            method=method.upper(),
            handler=handler,
            min_version=SemanticVersion.parse(min_version),
            max_version=SemanticVersion.parse(max_version) if max_version else None,
            deprecated_in=SemanticVersion.parse(deprecated_in) if deprecated_in else None,
        )

        with self._lock:
            self._endpoints.append(endpoint)

        return endpoint

    def get_endpoint(
        self,
        path: str,
        method: str,
        version: SemanticVersion,
    ) -> Optional[VersionedEndpoint]:
        """Get endpoint for specific version."""
        for endpoint in self._endpoints:
            if endpoint.path == path and endpoint.method == method.upper():
                if endpoint.supports_version(version):
                    return endpoint
        return None

    def version_decorator(
        self,
        min_version: str = "1.0.0",
        max_version: Optional[str] = None,
        deprecated_in: Optional[str] = None,
    ) -> Callable:
        """Decorator for versioned functions."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args: Any, version: Optional[str] = None, **kwargs: Any) -> Any:
                if version:
                    req_version = SemanticVersion.parse(version)
                    min_v = SemanticVersion.parse(min_version)
                    max_v = SemanticVersion.parse(max_version) if max_version else None

                    if req_version < min_v:
                        raise ValueError(f"Version {version} not supported. Minimum: {min_version}")
                    if max_v and req_version > max_v:
                        raise ValueError(f"Version {version} not supported. Maximum: {max_version}")

                return func(*args, **kwargs)
            return wrapper
        return decorator

    # =========================================================================
    # Version Negotiation
    # =========================================================================

    def negotiate_version(
        self,
        requested: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        path: Optional[str] = None,
    ) -> Tuple[SemanticVersion, APIVersion, bool]:
        """
        Negotiate API version from request.

        Returns:
            Tuple of (version, api_version, is_explicit)
        """
        version, is_explicit = self._negotiator.negotiate(
            requested, headers, params, path,
        )

        # Find matching API version
        api_version = self._versions.get(str(version))
        if not api_version:
            # Find closest compatible version
            for v in sorted(self._versions.values(), key=lambda x: x.version, reverse=True):
                if v.version.is_compatible_with(version):
                    api_version = v
                    break

        if not api_version:
            api_version = self._versions.get(self.config.default_version)

        self.bus.emit({
            "topic": self.BUS_TOPICS["negotiate"],
            "kind": "version",
            "actor": "versioning-module",
            "data": {
                "requested": requested,
                "negotiated": str(version),
                "is_explicit": is_explicit,
            },
        })

        return version, api_version, is_explicit

    def is_supported(self, version: str) -> bool:
        """Check if a version is supported."""
        try:
            semver = SemanticVersion.parse(version)
            api_version = self._versions.get(str(semver))
            return api_version is not None and not api_version.is_deprecated()
        except ValueError:
            return False

    # =========================================================================
    # Utilities
    # =========================================================================

    def bump_version(
        self,
        current: str,
        bump_type: str = "patch",
    ) -> str:
        """Bump version number."""
        semver = SemanticVersion.parse(current)

        if bump_type == "major":
            return str(SemanticVersion(semver.major + 1, 0, 0))
        elif bump_type == "minor":
            return str(SemanticVersion(semver.major, semver.minor + 1, 0))
        else:  # patch
            return str(SemanticVersion(semver.major, semver.minor, semver.patch + 1))

    def stats(self) -> Dict[str, Any]:
        """Get versioning statistics."""
        return {
            "total_versions": len(self._versions),
            "stable_versions": sum(1 for v in self._versions.values() if v.is_stable()),
            "deprecated_versions": sum(1 for v in self._versions.values() if v.is_deprecated()),
            "endpoints": len(self._endpoints),
            "current_version": str(self.get_current_version()),
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Versioning Module."""
    import argparse

    parser = argparse.ArgumentParser(description="Versioning Module (Step 98)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    list_parser = subparsers.add_parser("list", help="List versions")
    list_parser.add_argument("--all", "-a", action="store_true", help="Include deprecated")
    list_parser.add_argument("--json", action="store_true")

    # check command
    check_parser = subparsers.add_parser("check", help="Check version")
    check_parser.add_argument("version", help="Version to check")

    # bump command
    bump_parser = subparsers.add_parser("bump", help="Bump version")
    bump_parser.add_argument("current", help="Current version")
    bump_parser.add_argument("--type", "-t", choices=["major", "minor", "patch"], default="patch")

    # parse command
    parse_parser = subparsers.add_parser("parse", help="Parse version string")
    parse_parser.add_argument("version", help="Version string")

    # stats command
    subparsers.add_parser("stats", help="Show versioning statistics")

    # demo command
    subparsers.add_parser("demo", help="Run versioning demo")

    args = parser.parse_args()
    versioning = VersioningModule()

    if args.command == "list":
        versions = versioning.list_versions(include_deprecated=args.all)
        if args.json:
            print(json.dumps([v.to_dict() for v in versions], indent=2))
        else:
            print("API Versions:")
            for v in versions:
                status_icon = "[OK]" if v.is_stable() else "[DEP]" if v.is_deprecated() else "[BETA]"
                print(f"  {status_icon} {v.version} - {v.status.value}")
        return 0

    elif args.command == "check":
        if versioning.is_supported(args.version):
            print(f"Version {args.version} is supported")
            return 0
        else:
            print(f"Version {args.version} is not supported")
            return 1

    elif args.command == "bump":
        new_version = versioning.bump_version(args.current, args.type)
        print(f"{args.current} -> {new_version}")
        return 0

    elif args.command == "parse":
        try:
            semver = SemanticVersion.parse(args.version)
            print(json.dumps(semver.to_dict(), indent=2))
            return 0
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    elif args.command == "stats":
        stats = versioning.stats()
        print(json.dumps(stats, indent=2))
        return 0

    elif args.command == "demo":
        print("Versioning Module Demo\n")

        # Register versions
        print("1. Registering versions...")
        versioning.register_version("1.0.0", VersionStatus.STABLE, ["Initial release"])
        versioning.register_version("1.1.0", VersionStatus.STABLE, ["Bug fixes", "New features"])
        versioning.register_version("2.0.0", VersionStatus.BETA, ["Major rewrite"], ["API changes"])

        # Deprecate old version
        versioning.deprecate_version("1.0.0")

        print("\nVersions:")
        for v in versioning.list_versions(include_deprecated=True):
            print(f"  {v.version}: {v.status.value}")

        # Version negotiation
        print("\n2. Version negotiation...")
        test_cases = [
            ({"X-API-Version": "1.1.0"}, None, None),
            (None, {"version": "2.0.0"}, None),
            (None, None, "/v1/users"),
            (None, None, None),  # Default
        ]

        for headers, params, path in test_cases:
            version, api_version, is_explicit = versioning.negotiate_version(
                headers=headers, params=params, path=path,
            )
            print(f"  Negotiated: {version} (explicit: {is_explicit})")

        # Version comparison
        print("\n3. Version comparison...")
        v1 = SemanticVersion.parse("1.0.0")
        v2 = SemanticVersion.parse("2.0.0")
        v3 = SemanticVersion.parse("1.5.0")

        print(f"  {v1} < {v2}: {v1 < v2}")
        print(f"  {v1} compatible with {v3}: {v1.is_compatible_with(v3)}")
        print(f"  {v1} compatible with {v2}: {v1.is_compatible_with(v2)}")

        # Bumping
        print("\n4. Version bumping...")
        current = "1.2.3"
        print(f"  Current: {current}")
        print(f"  Patch bump: {versioning.bump_version(current, 'patch')}")
        print(f"  Minor bump: {versioning.bump_version(current, 'minor')}")
        print(f"  Major bump: {versioning.bump_version(current, 'major')}")

        print("\nStatistics:")
        print(json.dumps(versioning.stats(), indent=2))

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
