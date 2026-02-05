#!/usr/bin/env python3
"""
Step 148: Test Versioning Module

API versioning system for the Test Agent.

PBTSO Phase: PLAN, BUILD
Bus Topics:
- test.version.register (emits)
- test.version.resolve (emits)
- test.version.deprecate (emits)

Dependencies: Steps 101-147 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# Constants
# ============================================================================

class VersionStrategy(Enum):
    """API versioning strategies."""
    URL_PATH = "url_path"       # /v1/resource
    HEADER = "header"           # X-API-Version: 1
    QUERY_PARAM = "query_param" # ?version=1
    ACCEPT_HEADER = "accept"    # Accept: application/vnd.api.v1+json


class VersionStatus(Enum):
    """Version status."""
    CURRENT = "current"
    SUPPORTED = "supported"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class APIVersion:
    """
    An API version definition.

    Attributes:
        version: Version string (e.g., "1", "2", "1.0")
        status: Version status
        introduced_at: When version was introduced
        deprecated_at: When version was deprecated
        sunset_at: When version will be removed
        changes: Changes from previous version
        supported_until: Support end date
    """
    version: str
    status: VersionStatus = VersionStatus.CURRENT
    introduced_at: float = field(default_factory=time.time)
    deprecated_at: Optional[float] = None
    sunset_at: Optional[float] = None
    changes: List[str] = field(default_factory=list)
    supported_until: Optional[str] = None
    migration_guide: Optional[str] = None

    @property
    def is_deprecated(self) -> bool:
        return self.status in (VersionStatus.DEPRECATED, VersionStatus.SUNSET)

    @property
    def is_sunset(self) -> bool:
        if self.sunset_at:
            return time.time() > self.sunset_at
        return self.status == VersionStatus.SUNSET

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "status": self.status.value,
            "introduced_at": datetime.fromtimestamp(self.introduced_at, tz=timezone.utc).isoformat(),
            "deprecated_at": datetime.fromtimestamp(self.deprecated_at, tz=timezone.utc).isoformat() if self.deprecated_at else None,
            "sunset_at": datetime.fromtimestamp(self.sunset_at, tz=timezone.utc).isoformat() if self.sunset_at else None,
            "changes": self.changes,
            "supported_until": self.supported_until,
            "is_deprecated": self.is_deprecated,
            "is_sunset": self.is_sunset,
        }


@dataclass
class VersionedEndpoint:
    """
    A versioned API endpoint.

    Attributes:
        path: Endpoint path (without version prefix)
        method: HTTP method
        handler: Request handler function
        versions: Supported versions
        introduced_in: Version where endpoint was introduced
        deprecated_in: Version where endpoint was deprecated
    """
    path: str
    method: str = "GET"
    handler: Optional[Callable] = None
    versions: List[str] = field(default_factory=list)
    introduced_in: str = "1"
    deprecated_in: Optional[str] = None
    removed_in: Optional[str] = None
    description: str = ""

    def supports_version(self, version: str) -> bool:
        """Check if endpoint supports a version."""
        if self.removed_in and self._compare_versions(version, self.removed_in) >= 0:
            return False
        if self.versions:
            return version in self.versions
        return self._compare_versions(version, self.introduced_in) >= 0

    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare two version strings."""
        def parse(v):
            return [int(x) for x in re.split(r'[.-]', v) if x.isdigit()]
        p1, p2 = parse(v1), parse(v2)
        for a, b in zip(p1, p2):
            if a != b:
                return a - b
        return len(p1) - len(p2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "method": self.method,
            "versions": self.versions,
            "introduced_in": self.introduced_in,
            "deprecated_in": self.deprecated_in,
            "removed_in": self.removed_in,
            "description": self.description,
        }


@dataclass
class VersionResolution:
    """
    Result of version resolution.

    Attributes:
        requested_version: Requested version
        resolved_version: Resolved version
        is_valid: Whether resolution was successful
        warnings: Any warnings
        endpoint: Resolved endpoint if found
    """
    requested_version: Optional[str] = None
    resolved_version: Optional[str] = None
    is_valid: bool = False
    warnings: List[str] = field(default_factory=list)
    endpoint: Optional[VersionedEndpoint] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requested_version": self.requested_version,
            "resolved_version": self.resolved_version,
            "is_valid": self.is_valid,
            "warnings": self.warnings,
        }


@dataclass
class VersionConfig:
    """
    Configuration for versioning.

    Attributes:
        output_dir: Output directory
        strategy: Versioning strategy
        default_version: Default API version
        strict_versioning: Require explicit version
        version_header: Header name for version
        version_param: Query parameter for version
    """
    output_dir: str = ".pluribus/test-agent/versioning"
    strategy: VersionStrategy = VersionStrategy.URL_PATH
    default_version: str = "1"
    strict_versioning: bool = False
    version_header: str = "X-API-Version"
    version_param: str = "version"
    include_deprecation_headers: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "default_version": self.default_version,
            "strict_versioning": self.strict_versioning,
            "version_header": self.version_header,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class VersionBus:
    """Bus interface for versioning with file locking."""

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
# Test Version Manager
# ============================================================================

class TestVersionManager:
    """
    API versioning manager for the Test Agent.

    Features:
    - Multiple versioning strategies
    - Version lifecycle management
    - Deprecation handling
    - Version negotiation
    - Migration support

    PBTSO Phase: PLAN, BUILD
    Bus Topics: test.version.register, test.version.resolve, test.version.deprecate
    """

    BUS_TOPICS = {
        "register": "test.version.register",
        "resolve": "test.version.resolve",
        "deprecate": "test.version.deprecate",
    }

    def __init__(self, bus=None, config: Optional[VersionConfig] = None):
        """
        Initialize the version manager.

        Args:
            bus: Optional bus instance
            config: Version configuration
        """
        self.bus = bus or VersionBus()
        self.config = config or VersionConfig()
        self._versions: Dict[str, APIVersion] = {}
        self._endpoints: Dict[str, Dict[str, VersionedEndpoint]] = {}  # path -> method -> endpoint
        self._current_version: str = self.config.default_version

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Register default version
        self.register_version(APIVersion(
            version=self.config.default_version,
            status=VersionStatus.CURRENT,
        ))

    def register_version(
        self,
        version: APIVersion,
    ) -> None:
        """
        Register an API version.

        Args:
            version: API version to register
        """
        self._versions[version.version] = version

        # Update current version if this is the highest
        if version.status == VersionStatus.CURRENT:
            self._current_version = version.version

        self._emit_event("register", {
            "version": version.version,
            "status": version.status.value,
        })

    def register_endpoint(
        self,
        path: str,
        method: str = "GET",
        handler: Optional[Callable] = None,
        introduced_in: str = "1",
        deprecated_in: Optional[str] = None,
        description: str = "",
    ) -> VersionedEndpoint:
        """
        Register a versioned endpoint.

        Args:
            path: Endpoint path
            method: HTTP method
            handler: Request handler
            introduced_in: Version introduced
            deprecated_in: Version deprecated
            description: Endpoint description

        Returns:
            VersionedEndpoint instance
        """
        endpoint = VersionedEndpoint(
            path=path,
            method=method,
            handler=handler,
            introduced_in=introduced_in,
            deprecated_in=deprecated_in,
            description=description,
        )

        if path not in self._endpoints:
            self._endpoints[path] = {}
        self._endpoints[path][method] = endpoint

        return endpoint

    def deprecate_version(
        self,
        version: str,
        sunset_date: Optional[str] = None,
        migration_guide: Optional[str] = None,
    ) -> bool:
        """
        Deprecate an API version.

        Args:
            version: Version to deprecate
            sunset_date: Date version will be removed
            migration_guide: Migration guide URL/text

        Returns:
            True if version was deprecated
        """
        if version not in self._versions:
            return False

        api_version = self._versions[version]
        api_version.status = VersionStatus.DEPRECATED
        api_version.deprecated_at = time.time()
        api_version.migration_guide = migration_guide

        if sunset_date:
            api_version.supported_until = sunset_date
            # Parse sunset date and set timestamp
            try:
                dt = datetime.strptime(sunset_date, "%Y-%m-%d")
                api_version.sunset_at = dt.timestamp()
            except ValueError:
                pass

        self._emit_event("deprecate", {
            "version": version,
            "sunset_date": sunset_date,
        })

        return True

    def sunset_version(self, version: str) -> bool:
        """
        Mark a version as sunset (no longer available).

        Args:
            version: Version to sunset

        Returns:
            True if version was sunset
        """
        if version not in self._versions:
            return False

        self._versions[version].status = VersionStatus.SUNSET
        return True

    def resolve_version(
        self,
        request_info: Dict[str, Any],
    ) -> VersionResolution:
        """
        Resolve API version from request.

        Args:
            request_info: Request information containing:
                - path: Request path
                - method: HTTP method
                - headers: Request headers
                - query_params: Query parameters

        Returns:
            VersionResolution with resolved version
        """
        resolution = VersionResolution()

        # Extract version based on strategy
        version = self._extract_version(request_info)
        resolution.requested_version = version

        # Use default if no version specified
        if not version:
            if self.config.strict_versioning:
                return resolution
            version = self.config.default_version

        # Check if version exists
        if version not in self._versions:
            resolution.warnings.append(f"Unknown version: {version}")
            return resolution

        api_version = self._versions[version]

        # Check version status
        if api_version.is_sunset:
            resolution.warnings.append(f"Version {version} has been sunset")
            return resolution

        if api_version.is_deprecated:
            resolution.warnings.append(
                f"Version {version} is deprecated. "
                f"Please migrate to version {self._current_version}"
            )
            if api_version.supported_until:
                resolution.warnings.append(f"Support ends: {api_version.supported_until}")

        resolution.resolved_version = version
        resolution.is_valid = True

        # Resolve endpoint if path provided
        path = request_info.get("path", "")
        method = request_info.get("method", "GET")

        # Strip version prefix from path for URL_PATH strategy
        if self.config.strategy == VersionStrategy.URL_PATH:
            path = self._strip_version_prefix(path)

        endpoint = self._get_endpoint(path, method, version)
        if endpoint:
            resolution.endpoint = endpoint
            if endpoint.deprecated_in and self._compare_versions(version, endpoint.deprecated_in) >= 0:
                resolution.warnings.append(f"Endpoint {path} is deprecated in v{endpoint.deprecated_in}")

        self._emit_event("resolve", {
            "requested": resolution.requested_version,
            "resolved": resolution.resolved_version,
            "valid": resolution.is_valid,
        })

        return resolution

    def _extract_version(self, request_info: Dict[str, Any]) -> Optional[str]:
        """Extract version from request based on strategy."""
        if self.config.strategy == VersionStrategy.URL_PATH:
            path = request_info.get("path", "")
            match = re.match(r'^/v(\d+(?:\.\d+)?)', path)
            if match:
                return match.group(1)

        elif self.config.strategy == VersionStrategy.HEADER:
            headers = request_info.get("headers", {})
            return headers.get(self.config.version_header)

        elif self.config.strategy == VersionStrategy.QUERY_PARAM:
            params = request_info.get("query_params", {})
            return params.get(self.config.version_param)

        elif self.config.strategy == VersionStrategy.ACCEPT_HEADER:
            accept = request_info.get("headers", {}).get("Accept", "")
            match = re.search(r'application/vnd\.api\.v(\d+(?:\.\d+)?)', accept)
            if match:
                return match.group(1)

        return None

    def _strip_version_prefix(self, path: str) -> str:
        """Strip version prefix from path."""
        return re.sub(r'^/v\d+(?:\.\d+)?', '', path)

    def _get_endpoint(
        self,
        path: str,
        method: str,
        version: str,
    ) -> Optional[VersionedEndpoint]:
        """Get endpoint for path/method/version."""
        if path not in self._endpoints:
            return None

        endpoint = self._endpoints[path].get(method)
        if endpoint and endpoint.supports_version(version):
            return endpoint

        return None

    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare two version strings."""
        def parse(v):
            return [int(x) for x in re.split(r'[.-]', v) if x.isdigit()]
        p1, p2 = parse(v1), parse(v2)
        for a, b in zip(p1, p2):
            if a != b:
                return a - b
        return len(p1) - len(p2)

    def versioned(
        self,
        version: str,
        deprecated_in: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to mark a function as version-specific.

        Args:
            version: Minimum version required
            deprecated_in: Version where function is deprecated

        Returns:
            Decorator function
        """
        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapper(*args, **kwargs):
                # In actual implementation, this would check request version
                return fn(*args, **kwargs)

            wrapper._version = version
            wrapper._deprecated_in = deprecated_in
            return wrapper

        return decorator

    def get_versions(self) -> List[APIVersion]:
        """Get all registered versions."""
        return sorted(
            self._versions.values(),
            key=lambda v: v.version,
            reverse=True,
        )

    def get_current_version(self) -> str:
        """Get current API version."""
        return self._current_version

    def get_endpoints(self, version: Optional[str] = None) -> List[VersionedEndpoint]:
        """Get all registered endpoints, optionally filtered by version."""
        endpoints = []
        for path_endpoints in self._endpoints.values():
            for endpoint in path_endpoints.values():
                if version is None or endpoint.supports_version(version):
                    endpoints.append(endpoint)
        return endpoints

    def get_deprecation_headers(self, version: str) -> Dict[str, str]:
        """Get deprecation headers for a version."""
        headers = {}

        if version in self._versions:
            api_version = self._versions[version]

            if api_version.is_deprecated:
                headers["Deprecation"] = "true"
                if api_version.deprecated_at:
                    dt = datetime.fromtimestamp(api_version.deprecated_at, tz=timezone.utc)
                    headers["Deprecation"] = dt.strftime("%a, %d %b %Y %H:%M:%S GMT")

                if api_version.sunset_at:
                    dt = datetime.fromtimestamp(api_version.sunset_at, tz=timezone.utc)
                    headers["Sunset"] = dt.strftime("%a, %d %b %Y %H:%M:%S GMT")

                if api_version.migration_guide:
                    headers["Link"] = f'<{api_version.migration_guide}>; rel="deprecation"'

        return headers

    async def resolve_version_async(
        self,
        request_info: Dict[str, Any],
    ) -> VersionResolution:
        """Async version of resolve_version."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.resolve_version, request_info)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.version.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "versioning",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Version Manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Version Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List versions")

    # Current command
    current_parser = subparsers.add_parser("current", help="Show current version")

    # Deprecate command
    deprecate_parser = subparsers.add_parser("deprecate", help="Deprecate a version")
    deprecate_parser.add_argument("version", help="Version to deprecate")
    deprecate_parser.add_argument("--sunset", help="Sunset date (YYYY-MM-DD)")

    # Endpoints command
    endpoints_parser = subparsers.add_parser("endpoints", help="List endpoints")
    endpoints_parser.add_argument("--version", help="Filter by version")

    # Resolve command
    resolve_parser = subparsers.add_parser("resolve", help="Resolve version from request")
    resolve_parser.add_argument("--path", default="/v1/test", help="Request path")
    resolve_parser.add_argument("--method", default="GET", help="HTTP method")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/versioning")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = VersionConfig(output_dir=args.output)
    manager = TestVersionManager(config=config)

    # Add some sample versions and endpoints for testing
    manager.register_version(APIVersion(version="2", status=VersionStatus.CURRENT))
    manager.register_version(APIVersion(version="1", status=VersionStatus.SUPPORTED))

    manager.register_endpoint("/tests", "GET", introduced_in="1", description="List tests")
    manager.register_endpoint("/tests/{id}", "GET", introduced_in="1", description="Get test")
    manager.register_endpoint("/coverage", "GET", introduced_in="2", description="Get coverage")

    if args.command == "list":
        versions = manager.get_versions()

        if args.json:
            print(json.dumps([v.to_dict() for v in versions], indent=2))
        else:
            print("\nAPI Versions:")
            for version in versions:
                status = f"[{version.status.value.upper()}]"
                deprecated = " (deprecated)" if version.is_deprecated else ""
                print(f"\n  {status} v{version.version}{deprecated}")
                if version.changes:
                    print("    Changes:")
                    for change in version.changes:
                        print(f"      - {change}")
                if version.supported_until:
                    print(f"    Support Until: {version.supported_until}")

    elif args.command == "current":
        current = manager.get_current_version()

        if args.json:
            print(json.dumps({"current_version": current}))
        else:
            print(f"Current Version: {current}")

    elif args.command == "deprecate":
        if manager.deprecate_version(args.version, sunset_date=args.sunset):
            print(f"Deprecated version {args.version}")
            if args.sunset:
                print(f"Sunset date: {args.sunset}")
        else:
            print(f"Version not found: {args.version}")

    elif args.command == "endpoints":
        endpoints = manager.get_endpoints(args.version)

        if args.json:
            print(json.dumps([e.to_dict() for e in endpoints], indent=2))
        else:
            version_filter = f" (v{args.version})" if args.version else ""
            print(f"\nEndpoints{version_filter}:")
            for endpoint in endpoints:
                print(f"\n  {endpoint.method} {endpoint.path}")
                print(f"    Introduced: v{endpoint.introduced_in}")
                if endpoint.deprecated_in:
                    print(f"    Deprecated: v{endpoint.deprecated_in}")
                if endpoint.description:
                    print(f"    {endpoint.description}")

    elif args.command == "resolve":
        request_info = {
            "path": args.path,
            "method": args.method,
            "headers": {},
            "query_params": {},
        }

        resolution = manager.resolve_version(request_info)

        if args.json:
            print(json.dumps(resolution.to_dict(), indent=2))
        else:
            print(f"\nVersion Resolution:")
            print(f"  Requested: {resolution.requested_version}")
            print(f"  Resolved: {resolution.resolved_version}")
            print(f"  Valid: {resolution.is_valid}")
            if resolution.warnings:
                print("  Warnings:")
                for warning in resolution.warnings:
                    print(f"    - {warning}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
