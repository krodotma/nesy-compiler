#!/usr/bin/env python3
"""
migration_tools.py - Migration Tools (Step 95)

PBTSO Phase: ITERATE, SKILL

Provides:
- Data migration definitions
- Migration versioning
- Rollback support
- Migration history
- Dry-run capability

Bus Topics:
- code.migration.run
- code.migration.rollback
- code.migration.status

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class MigrationStatus(Enum):
    """Migration execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"


class MigrationDirection(Enum):
    """Migration direction."""
    UP = "up"
    DOWN = "down"


@dataclass
class MigrationConfig:
    """Configuration for migration module."""
    migrations_dir: str = "/pluribus/pia/oagent/code/migrations"
    history_file: str = "/pluribus/.pluribus/migration_history.json"
    dry_run: bool = False
    allow_downgrade: bool = True
    batch_size: int = 100
    transaction_mode: bool = True
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "migrations_dir": self.migrations_dir,
            "history_file": self.history_file,
            "dry_run": self.dry_run,
            "allow_downgrade": self.allow_downgrade,
            "batch_size": self.batch_size,
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
# Migration Types
# =============================================================================

@dataclass
class MigrationStep:
    """A single step in a migration."""
    id: str
    description: str
    up: Callable[[], Coroutine[Any, Any, None]]
    down: Optional[Callable[[], Coroutine[Any, Any, None]]] = None
    validate: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "has_down": self.down is not None,
            "has_validate": self.validate is not None,
        }


@dataclass
class Migration:
    """A migration definition."""
    id: str
    version: str
    name: str
    description: str
    steps: List[MigrationStep] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def checksum(self) -> str:
        """Calculate migration checksum."""
        content = f"{self.version}:{self.name}:{len(self.steps)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "step_count": len(self.steps),
            "dependencies": self.dependencies,
            "checksum": self.checksum,
            "created_at": self.created_at,
        }


@dataclass
class MigrationResult:
    """Result of a migration execution."""
    migration_id: str
    version: str
    direction: MigrationDirection
    status: MigrationStatus
    duration_ms: float = 0.0
    steps_completed: int = 0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "migration_id": self.migration_id,
            "version": self.version,
            "direction": self.direction.value,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "steps_completed": self.steps_completed,
            "error": self.error,
            "timestamp": self.timestamp,
            "iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
        }


@dataclass
class MigrationHistory:
    """Migration execution history."""
    applied: List[MigrationResult] = field(default_factory=list)
    current_version: str = "0.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applied": [r.to_dict() for r in self.applied],
            "current_version": self.current_version,
            "total_migrations": len(self.applied),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MigrationHistory":
        """Create history from dict."""
        history = cls()
        history.current_version = data.get("current_version", "0.0.0")
        for entry in data.get("applied", []):
            history.applied.append(MigrationResult(
                migration_id=entry["migration_id"],
                version=entry["version"],
                direction=MigrationDirection(entry["direction"]),
                status=MigrationStatus(entry["status"]),
                duration_ms=entry.get("duration_ms", 0),
                steps_completed=entry.get("steps_completed", 0),
                error=entry.get("error"),
                timestamp=entry.get("timestamp", time.time()),
            ))
        return history


# =============================================================================
# Migration Module
# =============================================================================

class MigrationModule:
    """
    Migration module for data migrations.

    PBTSO Phase: ITERATE, SKILL

    Features:
    - Version-based migrations
    - Up/down migration support
    - Migration history tracking
    - Dry-run capability
    - Dependency resolution

    Usage:
        migrations = MigrationModule()

        migration = migrations.create_migration(
            version="1.0.0",
            name="initial_setup",
            description="Initial data setup",
        )

        migration.steps.append(MigrationStep(
            id="create_tables",
            description="Create database tables",
            up=async_create_tables,
            down=async_drop_tables,
        ))

        result = await migrations.run_migration(migration)
    """

    BUS_TOPICS = {
        "run": "code.migration.run",
        "rollback": "code.migration.rollback",
        "status": "code.migration.status",
    }

    def __init__(
        self,
        config: Optional[MigrationConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or MigrationConfig()
        self.bus = bus or LockedAgentBus()

        self._migrations: Dict[str, Migration] = {}
        self._history: Optional[MigrationHistory] = None
        self._lock = Lock()

        self._load_history()

    def _load_history(self) -> None:
        """Load migration history from file."""
        history_path = Path(self.config.history_file)
        if history_path.exists():
            try:
                data = json.loads(history_path.read_text())
                self._history = MigrationHistory.from_dict(data)
            except Exception:
                self._history = MigrationHistory()
        else:
            self._history = MigrationHistory()

    def _save_history(self) -> None:
        """Save migration history to file."""
        history_path = Path(self.config.history_file)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(self._history.to_dict(), indent=2))

    # =========================================================================
    # Migration Creation
    # =========================================================================

    def create_migration(
        self,
        version: str,
        name: str,
        description: str = "",
        dependencies: Optional[List[str]] = None,
    ) -> Migration:
        """Create a new migration."""
        migration = Migration(
            id=f"mig-{uuid.uuid4().hex[:12]}",
            version=version,
            name=name,
            description=description,
            dependencies=dependencies or [],
        )

        with self._lock:
            self._migrations[migration.id] = migration

        return migration

    def add_step(
        self,
        migration: Migration,
        step_id: str,
        description: str,
        up: Callable[[], Coroutine[Any, Any, None]],
        down: Optional[Callable[[], Coroutine[Any, Any, None]]] = None,
        validate: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None,
    ) -> MigrationStep:
        """Add a step to a migration."""
        step = MigrationStep(
            id=step_id,
            description=description,
            up=up,
            down=down,
            validate=validate,
        )
        migration.steps.append(step)
        return step

    def register_migration(self, migration: Migration) -> None:
        """Register a migration."""
        with self._lock:
            self._migrations[migration.id] = migration

    # =========================================================================
    # Migration Execution
    # =========================================================================

    async def run_migration(
        self,
        migration: Migration,
        dry_run: Optional[bool] = None,
    ) -> MigrationResult:
        """Run a migration."""
        dry_run = dry_run if dry_run is not None else self.config.dry_run

        self.bus.emit({
            "topic": self.BUS_TOPICS["run"],
            "kind": "migration",
            "actor": "migration-module",
            "data": {
                "migration_id": migration.id,
                "version": migration.version,
                "dry_run": dry_run,
            },
        })

        start = time.time()
        steps_completed = 0

        try:
            # Check dependencies
            for dep_version in migration.dependencies:
                if not self._is_version_applied(dep_version):
                    return MigrationResult(
                        migration_id=migration.id,
                        version=migration.version,
                        direction=MigrationDirection.UP,
                        status=MigrationStatus.FAILED,
                        error=f"Dependency not met: {dep_version}",
                    )

            # Run steps
            for step in migration.steps:
                if dry_run:
                    print(f"[DRY-RUN] Would execute: {step.description}")
                else:
                    await step.up()

                    # Validate if available
                    if step.validate:
                        valid = await step.validate()
                        if not valid:
                            raise ValueError(f"Validation failed for step: {step.id}")

                steps_completed += 1

            duration = (time.time() - start) * 1000

            result = MigrationResult(
                migration_id=migration.id,
                version=migration.version,
                direction=MigrationDirection.UP,
                status=MigrationStatus.COMPLETED,
                duration_ms=duration,
                steps_completed=steps_completed,
            )

            # Record in history
            if not dry_run:
                self._history.applied.append(result)
                self._history.current_version = migration.version
                self._save_history()

            return result

        except Exception as e:
            duration = (time.time() - start) * 1000

            result = MigrationResult(
                migration_id=migration.id,
                version=migration.version,
                direction=MigrationDirection.UP,
                status=MigrationStatus.FAILED,
                duration_ms=duration,
                steps_completed=steps_completed,
                error=str(e),
            )

            self.bus.emit({
                "topic": self.BUS_TOPICS["status"],
                "kind": "error",
                "level": "error",
                "actor": "migration-module",
                "data": result.to_dict(),
            })

            return result

    async def rollback_migration(
        self,
        migration: Migration,
        dry_run: Optional[bool] = None,
    ) -> MigrationResult:
        """Rollback a migration."""
        if not self.config.allow_downgrade:
            return MigrationResult(
                migration_id=migration.id,
                version=migration.version,
                direction=MigrationDirection.DOWN,
                status=MigrationStatus.FAILED,
                error="Downgrade not allowed",
            )

        dry_run = dry_run if dry_run is not None else self.config.dry_run

        self.bus.emit({
            "topic": self.BUS_TOPICS["rollback"],
            "kind": "migration",
            "actor": "migration-module",
            "data": {
                "migration_id": migration.id,
                "version": migration.version,
                "dry_run": dry_run,
            },
        })

        start = time.time()
        steps_completed = 0

        try:
            # Run steps in reverse
            for step in reversed(migration.steps):
                if step.down is None:
                    if dry_run:
                        print(f"[DRY-RUN] No down migration for: {step.description}")
                    continue

                if dry_run:
                    print(f"[DRY-RUN] Would rollback: {step.description}")
                else:
                    await step.down()

                steps_completed += 1

            duration = (time.time() - start) * 1000

            result = MigrationResult(
                migration_id=migration.id,
                version=migration.version,
                direction=MigrationDirection.DOWN,
                status=MigrationStatus.ROLLED_BACK,
                duration_ms=duration,
                steps_completed=steps_completed,
            )

            # Update history
            if not dry_run:
                self._history.applied.append(result)
                # Find previous version
                prev_version = self._get_previous_version(migration.version)
                self._history.current_version = prev_version
                self._save_history()

            return result

        except Exception as e:
            duration = (time.time() - start) * 1000

            return MigrationResult(
                migration_id=migration.id,
                version=migration.version,
                direction=MigrationDirection.DOWN,
                status=MigrationStatus.FAILED,
                duration_ms=duration,
                steps_completed=steps_completed,
                error=str(e),
            )

    async def migrate_to_version(
        self,
        target_version: str,
        dry_run: Optional[bool] = None,
    ) -> List[MigrationResult]:
        """Migrate to a specific version."""
        results = []
        current = self._history.current_version

        # Sort migrations by version
        sorted_migrations = sorted(
            self._migrations.values(),
            key=lambda m: m.version,
        )

        # Determine direction
        if self._compare_versions(target_version, current) > 0:
            # Upgrade
            for migration in sorted_migrations:
                if self._compare_versions(migration.version, current) > 0:
                    if self._compare_versions(migration.version, target_version) <= 0:
                        result = await self.run_migration(migration, dry_run)
                        results.append(result)
                        if result.status == MigrationStatus.FAILED:
                            break
        else:
            # Downgrade
            for migration in reversed(sorted_migrations):
                if self._compare_versions(migration.version, current) <= 0:
                    if self._compare_versions(migration.version, target_version) > 0:
                        result = await self.rollback_migration(migration, dry_run)
                        results.append(result)
                        if result.status == MigrationStatus.FAILED:
                            break

        return results

    def _compare_versions(self, v1: str, v2: str) -> int:
        """Compare two version strings."""
        def parse_version(v: str) -> List[int]:
            return [int(x) for x in v.split(".")]

        p1, p2 = parse_version(v1), parse_version(v2)

        for i in range(max(len(p1), len(p2))):
            n1 = p1[i] if i < len(p1) else 0
            n2 = p2[i] if i < len(p2) else 0
            if n1 > n2:
                return 1
            if n1 < n2:
                return -1
        return 0

    def _is_version_applied(self, version: str) -> bool:
        """Check if a version has been applied."""
        for result in self._history.applied:
            if result.version == version and result.status == MigrationStatus.COMPLETED:
                return True
        return False

    def _get_previous_version(self, version: str) -> str:
        """Get the version before the given version."""
        versions = [m.version for m in self._migrations.values()]
        versions.sort()

        idx = versions.index(version) if version in versions else 0
        if idx > 0:
            return versions[idx - 1]
        return "0.0.0"

    # =========================================================================
    # Status and Queries
    # =========================================================================

    def get_current_version(self) -> str:
        """Get current migration version."""
        return self._history.current_version

    def get_pending_migrations(self) -> List[Migration]:
        """Get migrations pending execution."""
        current = self._history.current_version
        pending = []

        for migration in self._migrations.values():
            if self._compare_versions(migration.version, current) > 0:
                pending.append(migration)

        return sorted(pending, key=lambda m: m.version)

    def get_applied_migrations(self) -> List[MigrationResult]:
        """Get applied migrations."""
        return [r for r in self._history.applied if r.status == MigrationStatus.COMPLETED]

    def get_migration(self, migration_id: str) -> Optional[Migration]:
        """Get migration by ID."""
        return self._migrations.get(migration_id)

    def list_migrations(self) -> List[Migration]:
        """List all registered migrations."""
        return sorted(self._migrations.values(), key=lambda m: m.version)

    def get_history(self) -> MigrationHistory:
        """Get migration history."""
        return self._history

    def stats(self) -> Dict[str, Any]:
        """Get migration statistics."""
        return {
            "current_version": self._history.current_version,
            "total_migrations": len(self._migrations),
            "applied_count": len(self.get_applied_migrations()),
            "pending_count": len(self.get_pending_migrations()),
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Migration Module."""
    import argparse

    parser = argparse.ArgumentParser(description="Migration Tools (Step 95)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status command
    subparsers.add_parser("status", help="Show migration status")

    # list command
    list_parser = subparsers.add_parser("list", help="List migrations")
    list_parser.add_argument("--pending", "-p", action="store_true")
    list_parser.add_argument("--applied", "-a", action="store_true")
    list_parser.add_argument("--json", action="store_true")

    # history command
    hist_parser = subparsers.add_parser("history", help="Show migration history")
    hist_parser.add_argument("--limit", "-n", type=int, default=20)
    hist_parser.add_argument("--json", action="store_true")

    # demo command
    subparsers.add_parser("demo", help="Run migration demo")

    args = parser.parse_args()
    migrations = MigrationModule()

    async def run_async():
        if args.command == "status":
            print(f"Current Version: {migrations.get_current_version()}")
            print(f"Total Migrations: {len(migrations.list_migrations())}")
            print(f"Applied: {len(migrations.get_applied_migrations())}")
            print(f"Pending: {len(migrations.get_pending_migrations())}")
            return 0

        elif args.command == "list":
            if args.pending:
                items = migrations.get_pending_migrations()
                print("Pending Migrations:")
            elif args.applied:
                items = migrations.get_applied_migrations()
                print("Applied Migrations:")
            else:
                items = migrations.list_migrations()
                print("All Migrations:")

            if args.json:
                print(json.dumps([m.to_dict() if hasattr(m, "to_dict") else m for m in items], indent=2))
            else:
                for item in items:
                    if isinstance(item, Migration):
                        print(f"  [{item.version}] {item.name}: {item.description}")
                    else:
                        print(f"  [{item.version}] {item.status.value}")
            return 0

        elif args.command == "history":
            history = migrations.get_history()
            entries = history.applied[-args.limit:]

            if args.json:
                print(json.dumps([e.to_dict() for e in entries], indent=2))
            else:
                print("Migration History:")
                for entry in entries:
                    ts = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                    print(f"  [{ts}] {entry.version} - {entry.direction.value}: {entry.status.value}")
            return 0

        elif args.command == "demo":
            print("Migration Module Demo\n")

            # Create sample migrations
            mig1 = migrations.create_migration(
                version="1.0.0",
                name="initial_setup",
                description="Initial setup migration",
            )

            async def setup_up():
                print("  Running: Creating initial structure...")
                await asyncio.sleep(0.1)

            async def setup_down():
                print("  Running: Removing initial structure...")
                await asyncio.sleep(0.1)

            migrations.add_step(
                mig1,
                "create_structure",
                "Create initial directory structure",
                up=setup_up,
                down=setup_down,
            )

            mig2 = migrations.create_migration(
                version="1.1.0",
                name="add_config",
                description="Add configuration files",
                dependencies=["1.0.0"],
            )

            async def config_up():
                print("  Running: Adding configuration...")
                await asyncio.sleep(0.1)

            async def config_down():
                print("  Running: Removing configuration...")
                await asyncio.sleep(0.1)

            migrations.add_step(
                mig2,
                "add_config",
                "Add configuration files",
                up=config_up,
                down=config_down,
            )

            print("Registered migrations:")
            for m in migrations.list_migrations():
                print(f"  [{m.version}] {m.name}")

            print("\nRunning migration 1.0.0...")
            result1 = await migrations.run_migration(mig1)
            print(f"  Status: {result1.status.value}")
            print(f"  Duration: {result1.duration_ms:.2f}ms")

            print("\nRunning migration 1.1.0...")
            result2 = await migrations.run_migration(mig2)
            print(f"  Status: {result2.status.value}")

            print(f"\nCurrent version: {migrations.get_current_version()}")

            print("\nRolling back 1.1.0...")
            rollback = await migrations.rollback_migration(mig2)
            print(f"  Status: {rollback.status.value}")

            print(f"\nFinal version: {migrations.get_current_version()}")

            print("\nStatistics:")
            print(json.dumps(migrations.stats(), indent=2))

            return 0

        return 1

    return asyncio.run(run_async())


if __name__ == "__main__":
    import sys
    sys.exit(main())
