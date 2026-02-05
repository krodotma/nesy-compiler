#!/usr/bin/env python3
"""
system.py - Versioning System (Step 248)

PBTSO Phase: PLAN
A2A Integration: API versioning via deploy.versioning.route

Provides:
- APIVersion: API version definition
- VersionPolicy: Versioning policy
- VersionRoute: Version routing rules
- VersionMigration: Version migration path
- CompatibilityReport: Compatibility analysis
- VersioningSystem: Complete API versioning

Bus Topics:
- deploy.versioning.route
- deploy.versioning.register
- deploy.versioning.deprecate
- deploy.versioning.migrate

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


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
    actor: str = "versioning-system"
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

class VersionStatus(Enum):
    """Version lifecycle status."""
    ALPHA = "alpha"
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"


class VersioningScheme(Enum):
    """API versioning schemes."""
    URL_PATH = "url_path"       # /v1/resource
    HEADER = "header"          # Accept-Version: v1
    QUERY_PARAM = "query_param"  # ?version=v1
    MEDIA_TYPE = "media_type"   # Accept: application/vnd.api+json;version=1


class CompatibilityLevel(Enum):
    """Compatibility levels between versions."""
    FULL = "full"              # Fully compatible
    BACKWARD = "backward"      # Backward compatible
    FORWARD = "forward"        # Forward compatible
    BREAKING = "breaking"      # Breaking changes


@dataclass
class APIVersion:
    """
    API version definition.

    Attributes:
        version_id: Unique version identifier
        version: Version string (e.g., "v1", "v2.0")
        name: Human-readable name
        status: Version lifecycle status
        released_at: Release timestamp
        deprecated_at: Deprecation timestamp
        sunset_at: Sunset timestamp
        changes: List of changes from previous version
        metadata: Additional metadata
    """
    version_id: str
    version: str
    name: str = ""
    status: VersionStatus = VersionStatus.STABLE
    released_at: float = field(default_factory=time.time)
    deprecated_at: float = 0.0
    sunset_at: float = 0.0
    changes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "version": self.version,
            "name": self.name,
            "status": self.status.value,
            "released_at": self.released_at,
            "deprecated_at": self.deprecated_at,
            "sunset_at": self.sunset_at,
            "changes": self.changes,
            "metadata": self.metadata,
        }

    @property
    def is_active(self) -> bool:
        """Check if version is active (not sunset)."""
        return self.status not in (VersionStatus.SUNSET,)

    @property
    def is_deprecated(self) -> bool:
        """Check if version is deprecated."""
        return self.status in (VersionStatus.DEPRECATED, VersionStatus.SUNSET)


@dataclass
class VersionPolicy:
    """
    Versioning policy.

    Attributes:
        policy_id: Unique policy identifier
        name: Policy name
        scheme: Versioning scheme
        default_version: Default API version
        allow_no_version: Allow requests without version
        deprecation_notice_days: Days notice before deprecation
        sunset_notice_days: Days notice before sunset
        header_name: Header name for header scheme
        query_param_name: Query parameter name
    """
    policy_id: str
    name: str
    scheme: VersioningScheme = VersioningScheme.URL_PATH
    default_version: str = "v1"
    allow_no_version: bool = True
    deprecation_notice_days: int = 90
    sunset_notice_days: int = 180
    header_name: str = "Accept-Version"
    query_param_name: str = "version"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "scheme": self.scheme.value,
            "default_version": self.default_version,
            "allow_no_version": self.allow_no_version,
            "deprecation_notice_days": self.deprecation_notice_days,
            "sunset_notice_days": self.sunset_notice_days,
            "header_name": self.header_name,
            "query_param_name": self.query_param_name,
        }


@dataclass
class VersionRoute:
    """
    Version routing rules.

    Attributes:
        route_id: Unique route identifier
        path_pattern: URL path pattern
        version: Target version
        handler: Handler identifier
        enabled: Whether route is enabled
        priority: Route priority (higher wins)
    """
    route_id: str
    path_pattern: str
    version: str
    handler: str = ""
    enabled: bool = True
    priority: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def matches(self, path: str) -> bool:
        """Check if route matches a path."""
        # Simple pattern matching
        pattern = self.path_pattern.replace("*", ".*").replace("{", "(?P<").replace("}", ">[^/]+)")
        return bool(re.match(f"^{pattern}$", path))


@dataclass
class VersionMigration:
    """
    Version migration path.

    Attributes:
        migration_id: Unique migration identifier
        from_version: Source version
        to_version: Target version
        compatibility: Compatibility level
        steps: Migration steps
        transform_fn: Request/response transformer
    """
    migration_id: str
    from_version: str
    to_version: str
    compatibility: CompatibilityLevel = CompatibilityLevel.BACKWARD
    steps: List[str] = field(default_factory=list)
    transform_fn: Optional[Callable] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "migration_id": self.migration_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "compatibility": self.compatibility.value,
            "steps": self.steps,
        }


@dataclass
class CompatibilityReport:
    """
    Compatibility analysis report.

    Attributes:
        report_id: Unique report identifier
        from_version: Source version
        to_version: Target version
        compatibility: Overall compatibility
        breaking_changes: List of breaking changes
        deprecated_features: Deprecated features
        new_features: New features
        recommendations: Migration recommendations
        generated_at: Generation timestamp
    """
    report_id: str
    from_version: str
    to_version: str
    compatibility: CompatibilityLevel = CompatibilityLevel.FULL
    breaking_changes: List[str] = field(default_factory=list)
    deprecated_features: List[str] = field(default_factory=list)
    new_features: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "compatibility": self.compatibility.value,
            "breaking_changes": self.breaking_changes,
            "deprecated_features": self.deprecated_features,
            "new_features": self.new_features,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at,
        }


# ==============================================================================
# Versioning System (Step 248)
# ==============================================================================

class VersioningSystem:
    """
    Versioning System - API versioning for deployments.

    PBTSO Phase: PLAN

    Responsibilities:
    - Register and manage API versions
    - Route requests to correct version
    - Handle version negotiation
    - Manage deprecation lifecycle
    - Generate compatibility reports

    Example:
        >>> system = VersioningSystem()
        >>> v1 = system.register_version("v1", "API Version 1")
        >>> v2 = system.register_version("v2", "API Version 2", changes=["New auth"])
        >>> route = system.resolve_version("/v1/users")
        >>> report = system.check_compatibility("v1", "v2")
    """

    BUS_TOPICS = {
        "route": "deploy.versioning.route",
        "register": "deploy.versioning.register",
        "deprecate": "deploy.versioning.deprecate",
        "migrate": "deploy.versioning.migrate",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        policy: Optional[VersionPolicy] = None,
        actor_id: str = "versioning-system",
    ):
        """
        Initialize the versioning system.

        Args:
            state_dir: Directory for state persistence
            policy: Versioning policy
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "versioning"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Policy
        self.policy = policy or VersionPolicy(
            policy_id=f"policy-{uuid.uuid4().hex[:8]}",
            name="default",
        )

        # Storage
        self._versions: Dict[str, APIVersion] = {}
        self._routes: Dict[str, VersionRoute] = {}
        self._migrations: Dict[str, VersionMigration] = {}

        self._load_state()

    def register_version(
        self,
        version: str,
        name: str = "",
        status: VersionStatus = VersionStatus.STABLE,
        changes: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> APIVersion:
        """
        Register a new API version.

        Args:
            version: Version string (e.g., "v1", "v2.0")
            name: Human-readable name
            status: Version status
            changes: Changes from previous version
            metadata: Additional metadata

        Returns:
            Created APIVersion
        """
        version_id = f"ver-{uuid.uuid4().hex[:12]}"

        api_version = APIVersion(
            version_id=version_id,
            version=version,
            name=name or f"API {version}",
            status=status,
            changes=changes or [],
            metadata=metadata or {},
        )

        self._versions[version] = api_version
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["register"],
            {
                "version_id": version_id,
                "version": version,
                "status": status.value,
            },
            actor=self.actor_id,
        )

        return api_version

    def register_route(
        self,
        path_pattern: str,
        version: str,
        handler: str = "",
        priority: int = 0,
    ) -> VersionRoute:
        """
        Register a version route.

        Args:
            path_pattern: URL pattern
            version: Target version
            handler: Handler identifier
            priority: Route priority

        Returns:
            Created VersionRoute
        """
        route_id = f"route-{uuid.uuid4().hex[:8]}"

        route = VersionRoute(
            route_id=route_id,
            path_pattern=path_pattern,
            version=version,
            handler=handler,
            priority=priority,
        )

        self._routes[route_id] = route
        self._save_state()

        return route

    def resolve_version(
        self,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, Optional[VersionRoute]]:
        """
        Resolve version for a request.

        Args:
            path: Request path
            headers: Request headers
            query_params: Query parameters

        Returns:
            Tuple of (version, matched_route)
        """
        headers = headers or {}
        query_params = query_params or {}
        version = None
        matched_route = None

        # Try to extract version based on scheme
        if self.policy.scheme == VersioningScheme.URL_PATH:
            # Extract from path: /v1/resource
            match = re.match(r"^/(v\d+(?:\.\d+)?)/", path)
            if match:
                version = match.group(1)

        elif self.policy.scheme == VersioningScheme.HEADER:
            version = headers.get(self.policy.header_name)

        elif self.policy.scheme == VersioningScheme.QUERY_PARAM:
            version = query_params.get(self.policy.query_param_name)

        elif self.policy.scheme == VersioningScheme.MEDIA_TYPE:
            accept = headers.get("Accept", "")
            match = re.search(r"version=(\d+(?:\.\d+)?)", accept)
            if match:
                version = f"v{match.group(1)}"

        # Fallback to default
        if not version:
            if self.policy.allow_no_version:
                version = self.policy.default_version
            else:
                version = None

        # Find matching route
        if version:
            matching_routes = [
                r for r in self._routes.values()
                if r.enabled and r.version == version and r.matches(path)
            ]
            if matching_routes:
                matched_route = max(matching_routes, key=lambda r: r.priority)

        _emit_bus_event(
            self.BUS_TOPICS["route"],
            {
                "path": path,
                "resolved_version": version,
                "route_id": matched_route.route_id if matched_route else None,
            },
            actor=self.actor_id,
        )

        return version or self.policy.default_version, matched_route

    def deprecate_version(
        self,
        version: str,
        sunset_days: Optional[int] = None,
    ) -> Optional[APIVersion]:
        """
        Deprecate an API version.

        Args:
            version: Version to deprecate
            sunset_days: Days until sunset

        Returns:
            Updated APIVersion or None
        """
        api_version = self._versions.get(version)
        if not api_version:
            return None

        api_version.status = VersionStatus.DEPRECATED
        api_version.deprecated_at = time.time()

        if sunset_days is None:
            sunset_days = self.policy.sunset_notice_days

        api_version.sunset_at = api_version.deprecated_at + (sunset_days * 86400)

        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["deprecate"],
            {
                "version": version,
                "deprecated_at": api_version.deprecated_at,
                "sunset_at": api_version.sunset_at,
            },
            level="warn",
            actor=self.actor_id,
        )

        return api_version

    def sunset_version(self, version: str) -> Optional[APIVersion]:
        """
        Mark a version as sunset (end of life).

        Args:
            version: Version to sunset

        Returns:
            Updated APIVersion or None
        """
        api_version = self._versions.get(version)
        if not api_version:
            return None

        api_version.status = VersionStatus.SUNSET
        self._save_state()

        # Disable routes for this version
        for route in self._routes.values():
            if route.version == version:
                route.enabled = False

        return api_version

    def register_migration(
        self,
        from_version: str,
        to_version: str,
        compatibility: CompatibilityLevel = CompatibilityLevel.BACKWARD,
        steps: Optional[List[str]] = None,
    ) -> VersionMigration:
        """
        Register a version migration path.

        Args:
            from_version: Source version
            to_version: Target version
            compatibility: Compatibility level
            steps: Migration steps

        Returns:
            Created VersionMigration
        """
        migration_id = f"migration-{uuid.uuid4().hex[:8]}"

        migration = VersionMigration(
            migration_id=migration_id,
            from_version=from_version,
            to_version=to_version,
            compatibility=compatibility,
            steps=steps or [],
        )

        self._migrations[migration_id] = migration
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["migrate"],
            {
                "migration_id": migration_id,
                "from_version": from_version,
                "to_version": to_version,
                "compatibility": compatibility.value,
            },
            actor=self.actor_id,
        )

        return migration

    def check_compatibility(
        self,
        from_version: str,
        to_version: str,
    ) -> CompatibilityReport:
        """
        Check compatibility between versions.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            CompatibilityReport
        """
        report_id = f"report-{uuid.uuid4().hex[:8]}"

        from_ver = self._versions.get(from_version)
        to_ver = self._versions.get(to_version)

        breaking_changes = []
        deprecated_features = []
        new_features = []
        recommendations = []

        # Analyze changes
        if to_ver and to_ver.changes:
            for change in to_ver.changes:
                change_lower = change.lower()
                if any(word in change_lower for word in ["remove", "break", "incompatible"]):
                    breaking_changes.append(change)
                elif any(word in change_lower for word in ["deprecate", "legacy"]):
                    deprecated_features.append(change)
                elif any(word in change_lower for word in ["add", "new", "feature"]):
                    new_features.append(change)

        # Determine overall compatibility
        if breaking_changes:
            compatibility = CompatibilityLevel.BREAKING
            recommendations.append("Review breaking changes before migration")
            recommendations.append("Plan for client updates")
        elif deprecated_features:
            compatibility = CompatibilityLevel.BACKWARD
            recommendations.append("Update deprecated feature usage")
        else:
            compatibility = CompatibilityLevel.FULL

        # Check for existing migration path
        migration = self._find_migration(from_version, to_version)
        if migration:
            compatibility = migration.compatibility
            recommendations.extend(migration.steps)

        return CompatibilityReport(
            report_id=report_id,
            from_version=from_version,
            to_version=to_version,
            compatibility=compatibility,
            breaking_changes=breaking_changes,
            deprecated_features=deprecated_features,
            new_features=new_features,
            recommendations=recommendations,
        )

    def _find_migration(
        self,
        from_version: str,
        to_version: str,
    ) -> Optional[VersionMigration]:
        """Find migration path between versions."""
        for migration in self._migrations.values():
            if migration.from_version == from_version and migration.to_version == to_version:
                return migration
        return None

    def get_version(self, version: str) -> Optional[APIVersion]:
        """Get a version by string."""
        return self._versions.get(version)

    def list_versions(
        self,
        status: Optional[VersionStatus] = None,
        active_only: bool = False,
    ) -> List[APIVersion]:
        """List versions with optional filters."""
        versions = list(self._versions.values())

        if status:
            versions = [v for v in versions if v.status == status]

        if active_only:
            versions = [v for v in versions if v.is_active]

        return sorted(versions, key=lambda v: v.released_at, reverse=True)

    def get_active_versions(self) -> List[str]:
        """Get list of active version strings."""
        return [v.version for v in self._versions.values() if v.is_active]

    def get_deprecated_versions(self) -> List[APIVersion]:
        """Get list of deprecated versions."""
        return [v for v in self._versions.values() if v.is_deprecated]

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "versions": {k: v.to_dict() for k, v in self._versions.items()},
            "routes": {k: v.to_dict() for k, v in self._routes.items()},
            "migrations": {k: v.to_dict() for k, v in self._migrations.items()},
            "policy": self.policy.to_dict(),
        }
        state_file = self.state_dir / "versioning_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "versioning_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            for k, v in state.get("versions", {}).items():
                v["status"] = VersionStatus(v.get("status", "stable"))
                self._versions[k] = APIVersion(**{
                    key: val for key, val in v.items()
                    if key in APIVersion.__dataclass_fields__
                })

            for k, v in state.get("routes", {}).items():
                self._routes[k] = VersionRoute(**v)

            for k, v in state.get("migrations", {}).items():
                v["compatibility"] = CompatibilityLevel(v.get("compatibility", "backward"))
                self._migrations[k] = VersionMigration(**{
                    key: val for key, val in v.items()
                    if key in VersionMigration.__dataclass_fields__
                })

        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for versioning system."""
    import argparse

    parser = argparse.ArgumentParser(description="Versioning System (Step 248)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # register command
    register_parser = subparsers.add_parser("register", help="Register a version")
    register_parser.add_argument("version", help="Version string (e.g., v1)")
    register_parser.add_argument("--name", "-n", default="", help="Version name")
    register_parser.add_argument("--status", "-s", default="stable",
                                choices=["alpha", "beta", "stable"])
    register_parser.add_argument("--changes", "-c", help="Comma-separated changes")
    register_parser.add_argument("--json", action="store_true", help="JSON output")

    # deprecate command
    deprecate_parser = subparsers.add_parser("deprecate", help="Deprecate a version")
    deprecate_parser.add_argument("version", help="Version to deprecate")
    deprecate_parser.add_argument("--sunset-days", "-d", type=int, help="Days until sunset")

    # resolve command
    resolve_parser = subparsers.add_parser("resolve", help="Resolve version for path")
    resolve_parser.add_argument("path", help="Request path")
    resolve_parser.add_argument("--header", "-H", help="Version header value")

    # compatibility command
    compat_parser = subparsers.add_parser("compatibility", help="Check compatibility")
    compat_parser.add_argument("from_version", help="Source version")
    compat_parser.add_argument("to_version", help="Target version")
    compat_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List versions")
    list_parser.add_argument("--active", "-a", action="store_true", help="Active only")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    system = VersioningSystem()

    if args.command == "register":
        changes = args.changes.split(",") if args.changes else []
        version = system.register_version(
            version=args.version,
            name=args.name,
            status=VersionStatus(args.status),
            changes=changes,
        )

        if args.json:
            print(json.dumps(version.to_dict(), indent=2))
        else:
            print(f"Registered version: {version.version}")
            print(f"  ID: {version.version_id}")
            print(f"  Status: {version.status.value}")

        return 0

    elif args.command == "deprecate":
        version = system.deprecate_version(
            version=args.version,
            sunset_days=args.sunset_days,
        )

        if version:
            sunset_date = datetime.fromtimestamp(version.sunset_at).strftime("%Y-%m-%d")
            print(f"Deprecated version: {args.version}")
            print(f"  Sunset date: {sunset_date}")
        else:
            print(f"Version not found: {args.version}")
            return 1

        return 0

    elif args.command == "resolve":
        headers = {}
        if args.header:
            headers[system.policy.header_name] = args.header

        version, route = system.resolve_version(args.path, headers)
        print(f"Resolved version: {version}")
        if route:
            print(f"  Route: {route.path_pattern}")
            print(f"  Handler: {route.handler}")

        return 0

    elif args.command == "compatibility":
        report = system.check_compatibility(args.from_version, args.to_version)

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Compatibility: {args.from_version} -> {args.to_version}")
            print(f"  Level: {report.compatibility.value}")
            if report.breaking_changes:
                print("  Breaking changes:")
                for change in report.breaking_changes:
                    print(f"    - {change}")
            if report.new_features:
                print("  New features:")
                for feature in report.new_features:
                    print(f"    - {feature}")
            if report.recommendations:
                print("  Recommendations:")
                for rec in report.recommendations:
                    print(f"    - {rec}")

        return 0

    elif args.command == "list":
        versions = system.list_versions(active_only=args.active)

        if args.json:
            print(json.dumps([v.to_dict() for v in versions], indent=2))
        else:
            if not versions:
                print("No versions found")
            else:
                for v in versions:
                    status_marker = ""
                    if v.status == VersionStatus.DEPRECATED:
                        status_marker = " [DEPRECATED]"
                    elif v.status == VersionStatus.SUNSET:
                        status_marker = " [SUNSET]"
                    print(f"{v.version}: {v.name} ({v.status.value}){status_marker}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
