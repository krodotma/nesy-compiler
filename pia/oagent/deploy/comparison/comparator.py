#!/usr/bin/env python3
"""
comparator.py - Deployment Comparison (Step 227)

PBTSO Phase: VERIFY
A2A Integration: Compares deployments via deploy.comparison.*

Provides:
- ComparisonType: Types of comparisons
- DifferenceType: Types of differences
- ConfigDifference: Individual configuration difference
- DeploymentDiff: Complete deployment diff
- DeploymentComparator: Main comparator class

Bus Topics:
- deploy.comparison.compare
- deploy.comparison.diff

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ==============================================================================
# Bus Emission Helper with File Locking
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
    actor: str = "deployment-comparator"
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

class ComparisonType(Enum):
    """Types of comparisons."""
    VERSION = "version"
    CONFIG = "config"
    ENVIRONMENT = "environment"
    METRICS = "metrics"
    FULL = "full"


class DifferenceType(Enum):
    """Types of differences."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class SeverityLevel(Enum):
    """Severity of differences."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConfigDifference:
    """
    Individual configuration difference.

    Attributes:
        path: Configuration path (dot notation)
        diff_type: Type of difference
        old_value: Old value
        new_value: New value
        severity: Severity level
        description: Difference description
    """
    path: str
    diff_type: DifferenceType
    old_value: Any = None
    new_value: Any = None
    severity: SeverityLevel = SeverityLevel.INFO
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "diff_type": self.diff_type.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "severity": self.severity.value,
            "description": self.description,
        }


@dataclass
class VersionDiff:
    """
    Version difference.

    Attributes:
        old_version: Old version
        new_version: New version
        is_upgrade: Whether this is an upgrade
        is_rollback: Whether this is a rollback
        semver_change: Semantic version change type
    """
    old_version: str
    new_version: str
    is_upgrade: bool = False
    is_rollback: bool = False
    semver_change: str = ""  # major, minor, patch

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricsDiff:
    """
    Metrics difference between deployments.

    Attributes:
        metric_name: Metric name
        old_value: Old value
        new_value: New value
        change_pct: Percentage change
        improved: Whether metric improved
    """
    metric_name: str
    old_value: float = 0.0
    new_value: float = 0.0
    change_pct: float = 0.0
    improved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentDiff:
    """
    Complete deployment diff.

    Attributes:
        diff_id: Unique diff identifier
        source_deployment_id: Source deployment ID
        target_deployment_id: Target deployment ID
        service_name: Service name
        comparison_type: Type of comparison
        version_diff: Version differences
        config_diffs: Configuration differences
        metrics_diffs: Metrics differences
        environment_diffs: Environment differences
        summary: Diff summary
        created_at: Creation timestamp
    """
    diff_id: str
    source_deployment_id: str
    target_deployment_id: str
    service_name: str
    comparison_type: ComparisonType = ComparisonType.FULL
    version_diff: Optional[VersionDiff] = None
    config_diffs: List[ConfigDifference] = field(default_factory=list)
    metrics_diffs: List[MetricsDiff] = field(default_factory=list)
    environment_diffs: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diff_id": self.diff_id,
            "source_deployment_id": self.source_deployment_id,
            "target_deployment_id": self.target_deployment_id,
            "service_name": self.service_name,
            "comparison_type": self.comparison_type.value,
            "version_diff": self.version_diff.to_dict() if self.version_diff else None,
            "config_diffs": [d.to_dict() for d in self.config_diffs],
            "metrics_diffs": [d.to_dict() for d in self.metrics_diffs],
            "environment_diffs": self.environment_diffs,
            "summary": self.summary,
            "created_at": self.created_at,
        }

    @property
    def has_breaking_changes(self) -> bool:
        """Check if diff contains breaking changes."""
        if self.version_diff and self.version_diff.semver_change == "major":
            return True
        return any(
            d.severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL)
            for d in self.config_diffs
        )

    @property
    def total_changes(self) -> int:
        """Get total number of changes."""
        return (
            len([d for d in self.config_diffs if d.diff_type != DifferenceType.UNCHANGED])
            + len(self.metrics_diffs)
        )


# ==============================================================================
# Deployment Comparator (Step 227)
# ==============================================================================

class DeploymentComparator:
    """
    Deployment Comparator - compares deployments and configurations.

    PBTSO Phase: VERIFY

    Responsibilities:
    - Compare deployment versions
    - Diff deployment configurations
    - Compare deployment metrics
    - Identify breaking changes
    - Generate comparison reports

    Example:
        >>> comparator = DeploymentComparator()
        >>> diff = comparator.compare(
        ...     source_id="deploy-123",
        ...     target_id="deploy-456",
        ... )
        >>> print(f"Changes: {diff.total_changes}")
    """

    BUS_TOPICS = {
        "compare": "deploy.comparison.compare",
        "diff": "deploy.comparison.diff",
    }

    # Sensitive config paths that should be flagged
    SENSITIVE_PATHS = {
        "secrets",
        "credentials",
        "api_key",
        "password",
        "token",
        "certificate",
    }

    # Breaking change patterns
    BREAKING_PATTERNS = {
        "port",
        "protocol",
        "endpoint",
        "schema",
        "database",
        "api_version",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "deployment-comparator",
    ):
        """
        Initialize the comparator.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "comparison"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        self._diffs: Dict[str, DeploymentDiff] = {}
        self._history_tracker = None

    def _get_history_tracker(self):
        """Lazy load history tracker."""
        if self._history_tracker is None:
            from ..history.tracker import DeploymentHistoryTracker
            self._history_tracker = DeploymentHistoryTracker()
        return self._history_tracker

    def compare(
        self,
        source_id: str,
        target_id: str,
        comparison_type: ComparisonType = ComparisonType.FULL,
    ) -> DeploymentDiff:
        """
        Compare two deployments.

        Args:
            source_id: Source deployment ID (older)
            target_id: Target deployment ID (newer)
            comparison_type: Type of comparison

        Returns:
            DeploymentDiff with differences
        """
        diff_id = f"diff-{uuid.uuid4().hex[:12]}"

        tracker = self._get_history_tracker()
        source = tracker.get_deployment(source_id)
        target = tracker.get_deployment(target_id)

        if not source or not target:
            raise ValueError("One or both deployments not found")

        diff = DeploymentDiff(
            diff_id=diff_id,
            source_deployment_id=source_id,
            target_deployment_id=target_id,
            service_name=source.service_name,
            comparison_type=comparison_type,
        )

        _emit_bus_event(
            self.BUS_TOPICS["compare"],
            {
                "diff_id": diff_id,
                "source_id": source_id,
                "target_id": target_id,
                "comparison_type": comparison_type.value,
            },
            actor=self.actor_id,
        )

        # Compare versions
        if comparison_type in (ComparisonType.VERSION, ComparisonType.FULL):
            diff.version_diff = self._compare_versions(
                source.version, target.version
            )

        # Compare configs
        if comparison_type in (ComparisonType.CONFIG, ComparisonType.FULL):
            diff.config_diffs = self._compare_configs(
                source.config, target.config
            )

        # Compare environments
        if comparison_type in (ComparisonType.ENVIRONMENT, ComparisonType.FULL):
            diff.environment_diffs = {
                "source": source.environment,
                "target": target.environment,
                "changed": source.environment != target.environment,
            }

        # Compare metrics
        if comparison_type in (ComparisonType.METRICS, ComparisonType.FULL):
            diff.metrics_diffs = self._compare_metrics(source, target)

        # Build summary
        diff.summary = self._build_summary(diff)

        self._diffs[diff_id] = diff
        self._save_diff(diff)

        _emit_bus_event(
            self.BUS_TOPICS["diff"],
            {
                "diff_id": diff_id,
                "total_changes": diff.total_changes,
                "has_breaking_changes": diff.has_breaking_changes,
            },
            actor=self.actor_id,
        )

        return diff

    def compare_configs(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
    ) -> List[ConfigDifference]:
        """
        Compare two configuration dictionaries.

        Args:
            old_config: Old configuration
            new_config: New configuration

        Returns:
            List of ConfigDifference
        """
        return self._compare_configs(old_config, new_config)

    def _compare_versions(self, old_version: str, new_version: str) -> VersionDiff:
        """Compare two version strings."""
        diff = VersionDiff(
            old_version=old_version,
            new_version=new_version,
        )

        # Parse semver
        old_parts = self._parse_version(old_version)
        new_parts = self._parse_version(new_version)

        if old_parts and new_parts:
            if new_parts > old_parts:
                diff.is_upgrade = True
                if new_parts[0] > old_parts[0]:
                    diff.semver_change = "major"
                elif new_parts[1] > old_parts[1]:
                    diff.semver_change = "minor"
                else:
                    diff.semver_change = "patch"
            elif new_parts < old_parts:
                diff.is_rollback = True

        return diff

    def _parse_version(self, version: str) -> Optional[Tuple[int, ...]]:
        """Parse version string into tuple."""
        import re
        # Handle v prefix and extract numbers
        match = re.match(r'v?(\d+)\.(\d+)\.(\d+)', version)
        if match:
            return tuple(int(x) for x in match.groups())
        return None

    def _compare_configs(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
        prefix: str = "",
    ) -> List[ConfigDifference]:
        """Compare configuration dictionaries recursively."""
        diffs = []

        all_keys: Set[str] = set(old_config.keys()) | set(new_config.keys())

        for key in all_keys:
            path = f"{prefix}.{key}" if prefix else key

            if key not in old_config:
                diffs.append(ConfigDifference(
                    path=path,
                    diff_type=DifferenceType.ADDED,
                    new_value=new_config[key],
                    severity=self._get_severity(path),
                    description=f"Added: {path}",
                ))
            elif key not in new_config:
                diffs.append(ConfigDifference(
                    path=path,
                    diff_type=DifferenceType.REMOVED,
                    old_value=old_config[key],
                    severity=self._get_severity(path),
                    description=f"Removed: {path}",
                ))
            elif old_config[key] != new_config[key]:
                if isinstance(old_config[key], dict) and isinstance(new_config[key], dict):
                    # Recurse into nested dicts
                    diffs.extend(self._compare_configs(
                        old_config[key], new_config[key], path
                    ))
                else:
                    diffs.append(ConfigDifference(
                        path=path,
                        diff_type=DifferenceType.MODIFIED,
                        old_value=old_config[key],
                        new_value=new_config[key],
                        severity=self._get_severity(path),
                        description=f"Modified: {path}",
                    ))

        return diffs

    def _get_severity(self, path: str) -> SeverityLevel:
        """Determine severity based on config path."""
        path_lower = path.lower()

        # Check for sensitive paths
        for sensitive in self.SENSITIVE_PATHS:
            if sensitive in path_lower:
                return SeverityLevel.CRITICAL

        # Check for breaking change patterns
        for pattern in self.BREAKING_PATTERNS:
            if pattern in path_lower:
                return SeverityLevel.HIGH

        return SeverityLevel.INFO

    def _compare_metrics(self, source, target) -> List[MetricsDiff]:
        """Compare deployment metrics."""
        diffs = []

        # Compare durations
        if source.duration_ms and target.duration_ms:
            change_pct = ((target.duration_ms - source.duration_ms) / source.duration_ms) * 100
            diffs.append(MetricsDiff(
                metric_name="duration_ms",
                old_value=source.duration_ms,
                new_value=target.duration_ms,
                change_pct=change_pct,
                improved=change_pct < 0,  # Lower is better
            ))

        # Compare event counts
        source_events = len(source.events)
        target_events = len(target.events)
        if source_events and target_events:
            diffs.append(MetricsDiff(
                metric_name="event_count",
                old_value=source_events,
                new_value=target_events,
                change_pct=((target_events - source_events) / source_events) * 100,
            ))

        return diffs

    def _build_summary(self, diff: DeploymentDiff) -> Dict[str, Any]:
        """Build diff summary."""
        added = len([d for d in diff.config_diffs if d.diff_type == DifferenceType.ADDED])
        removed = len([d for d in diff.config_diffs if d.diff_type == DifferenceType.REMOVED])
        modified = len([d for d in diff.config_diffs if d.diff_type == DifferenceType.MODIFIED])

        return {
            "total_changes": diff.total_changes,
            "config_added": added,
            "config_removed": removed,
            "config_modified": modified,
            "has_breaking_changes": diff.has_breaking_changes,
            "version_changed": diff.version_diff is not None,
            "is_upgrade": diff.version_diff.is_upgrade if diff.version_diff else False,
            "is_rollback": diff.version_diff.is_rollback if diff.version_diff else False,
            "metrics_changed": len(diff.metrics_diffs),
            "environment_changed": diff.environment_diffs.get("changed", False),
            "critical_changes": len([
                d for d in diff.config_diffs
                if d.severity == SeverityLevel.CRITICAL
            ]),
            "high_severity_changes": len([
                d for d in diff.config_diffs
                if d.severity == SeverityLevel.HIGH
            ]),
        }

    def compare_with_current(
        self,
        service_name: str,
        environment: str,
        new_config: Dict[str, Any],
    ) -> List[ConfigDifference]:
        """
        Compare new config with currently deployed config.

        Args:
            service_name: Service name
            environment: Environment
            new_config: New configuration

        Returns:
            List of ConfigDifference
        """
        tracker = self._get_history_tracker()
        current = tracker.get_latest(service_name, environment)

        if not current:
            return [ConfigDifference(
                path="*",
                diff_type=DifferenceType.ADDED,
                description="No current deployment found - all config is new",
            )]

        return self._compare_configs(current.config or {}, new_config)

    def get_diff(self, diff_id: str) -> Optional[DeploymentDiff]:
        """Get a diff by ID."""
        return self._diffs.get(diff_id)

    def list_diffs(
        self,
        service_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[DeploymentDiff]:
        """List diffs."""
        diffs = list(self._diffs.values())

        if service_name:
            diffs = [d for d in diffs if d.service_name == service_name]

        diffs.sort(key=lambda d: d.created_at, reverse=True)
        return diffs[:limit]

    def _save_diff(self, diff: DeploymentDiff) -> None:
        """Save diff to disk."""
        diff_file = self.state_dir / f"{diff.diff_id}.json"
        with open(diff_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(diff.to_dict(), f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for deployment comparator."""
    import argparse

    parser = argparse.ArgumentParser(description="Deployment Comparator (Step 227)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two deployments")
    compare_parser.add_argument("source_id", help="Source deployment ID")
    compare_parser.add_argument("target_id", help="Target deployment ID")
    compare_parser.add_argument("--type", "-t", default="full",
                                 choices=["version", "config", "environment", "metrics", "full"])
    compare_parser.add_argument("--json", action="store_true", help="JSON output")

    # diff-config command
    config_parser = subparsers.add_parser("diff-config", help="Compare config files")
    config_parser.add_argument("old_file", help="Old config file (JSON)")
    config_parser.add_argument("new_file", help="New config file (JSON)")
    config_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List comparisons")
    list_parser.add_argument("--service", "-s", help="Filter by service")
    list_parser.add_argument("--limit", "-l", type=int, default=10, help="Limit")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # get command
    get_parser = subparsers.add_parser("get", help="Get comparison details")
    get_parser.add_argument("diff_id", help="Diff ID")
    get_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    comparator = DeploymentComparator()

    if args.command == "compare":
        try:
            diff = comparator.compare(
                source_id=args.source_id,
                target_id=args.target_id,
                comparison_type=ComparisonType(args.type.upper()),
            )

            if args.json:
                print(json.dumps(diff.to_dict(), indent=2))
            else:
                print(f"Diff: {diff.diff_id}")
                print(f"  Service: {diff.service_name}")
                print(f"  Total Changes: {diff.total_changes}")
                print(f"  Breaking Changes: {diff.has_breaking_changes}")
                if diff.version_diff:
                    print(f"  Version: {diff.version_diff.old_version} -> {diff.version_diff.new_version}")
                print(f"  Config Changes: {len(diff.config_diffs)}")

            return 0
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    elif args.command == "diff-config":
        try:
            with open(args.old_file) as f:
                old_config = json.load(f)
            with open(args.new_file) as f:
                new_config = json.load(f)

            diffs = comparator.compare_configs(old_config, new_config)

            if args.json:
                print(json.dumps([d.to_dict() for d in diffs], indent=2))
            else:
                if not diffs:
                    print("No differences found")
                else:
                    for d in diffs:
                        symbol = {"added": "+", "removed": "-", "modified": "~"}.get(d.diff_type.value, "?")
                        print(f"[{symbol}] {d.path}: {d.description}")
                        if d.diff_type == DifferenceType.MODIFIED:
                            print(f"    Old: {d.old_value}")
                            print(f"    New: {d.new_value}")

            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1

    elif args.command == "list":
        diffs = comparator.list_diffs(
            service_name=args.service,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps([d.to_dict() for d in diffs], indent=2))
        else:
            for d in diffs:
                print(f"{d.diff_id} ({d.service_name}) - {d.total_changes} changes")

        return 0

    elif args.command == "get":
        diff = comparator.get_diff(args.diff_id)
        if not diff:
            print(f"Diff not found: {args.diff_id}")
            return 1

        if args.json:
            print(json.dumps(diff.to_dict(), indent=2))
        else:
            print(f"Diff: {diff.diff_id}")
            print(f"  Source: {diff.source_deployment_id}")
            print(f"  Target: {diff.target_deployment_id}")
            print(f"  Changes: {diff.total_changes}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
