#!/usr/bin/env python3
"""
Step 145: Test Migration Tools

Data migration utilities for the Test Agent.

PBTSO Phase: BUILD, VERIFY
Bus Topics:
- test.migration.start (emits)
- test.migration.complete (emits)
- test.migration.rollback (emits)

Dependencies: Steps 101-144 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ============================================================================
# Constants
# ============================================================================

class MigrationStatus(Enum):
    """Migration status."""
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


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class Migration:
    """
    A migration definition.

    Attributes:
        name: Migration name
        version: Migration version
        description: Migration description
        up_fn: Forward migration function
        down_fn: Rollback function
        dependencies: Required migrations
        checksum: Migration checksum
    """
    name: str
    version: str
    description: str = ""
    up_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    down_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    dependencies: List[str] = field(default_factory=list)
    checksum: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """Calculate checksum if not provided."""
        if not self.checksum:
            content = f"{self.name}:{self.version}:{self.description}"
            self.checksum = hashlib.md5(content.encode()).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "dependencies": self.dependencies,
            "checksum": self.checksum,
            "created_at": self.created_at,
        }


@dataclass
class MigrationResult:
    """
    Result of a migration.

    Attributes:
        migration_name: Migration name
        status: Migration status
        direction: Migration direction
        started_at: Start timestamp
        completed_at: Completion timestamp
        error: Error message if failed
        changes: Changes made
    """
    migration_name: str
    status: MigrationStatus = MigrationStatus.PENDING
    direction: MigrationDirection = MigrationDirection.UP
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error: Optional[str] = None
    changes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "migration_name": self.migration_name,
            "status": self.status.value,
            "direction": self.direction.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "changes": self.changes,
        }


@dataclass
class MigrationState:
    """
    Current migration state.

    Attributes:
        current_version: Current version
        applied_migrations: List of applied migrations
        last_migration_at: Last migration timestamp
    """
    current_version: str = "0.0.0"
    applied_migrations: List[str] = field(default_factory=list)
    last_migration_at: Optional[float] = None
    history: List[MigrationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_version": self.current_version,
            "applied_migrations": self.applied_migrations,
            "last_migration_at": self.last_migration_at,
            "history": [h.to_dict() for h in self.history[-10:]],
        }


@dataclass
class MigrationConfig:
    """
    Configuration for migrations.

    Attributes:
        output_dir: Output directory
        state_file: State file path
        backup_before: Create backup before migration
        auto_rollback: Auto rollback on failure
        dry_run: Preview changes without applying
    """
    output_dir: str = ".pluribus/test-agent/migrations"
    state_file: str = ".pluribus/test-agent/migration_state.json"
    backup_before: bool = True
    auto_rollback: bool = True
    dry_run: bool = False
    max_history: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_dir": self.output_dir,
            "backup_before": self.backup_before,
            "auto_rollback": self.auto_rollback,
            "dry_run": self.dry_run,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class MigrationBus:
    """Bus interface for migration with file locking."""

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
# Test Migration Manager
# ============================================================================

class TestMigrationManager:
    """
    Migration manager for the Test Agent.

    Features:
    - Version-based migrations
    - Rollback support
    - Dependency tracking
    - Backup and restore
    - Dry-run mode

    PBTSO Phase: BUILD, VERIFY
    Bus Topics: test.migration.start, test.migration.complete, test.migration.rollback
    """

    BUS_TOPICS = {
        "start": "test.migration.start",
        "complete": "test.migration.complete",
        "rollback": "test.migration.rollback",
    }

    def __init__(self, bus=None, config: Optional[MigrationConfig] = None):
        """
        Initialize the migration manager.

        Args:
            bus: Optional bus instance
            config: Migration configuration
        """
        self.bus = bus or MigrationBus()
        self.config = config or MigrationConfig()
        self._migrations: Dict[str, Migration] = {}
        self._state: MigrationState = MigrationState()

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Load state
        self._load_state()

        # Register built-in migrations
        self._register_builtin_migrations()

    def _register_builtin_migrations(self) -> None:
        """Register built-in migrations."""
        # Schema v1 migration
        self.register_migration(Migration(
            name="schema_v1",
            version="1.0.0",
            description="Initial schema setup",
            up_fn=self._migrate_schema_v1_up,
            down_fn=self._migrate_schema_v1_down,
        ))

        # Schema v2 migration
        self.register_migration(Migration(
            name="schema_v2",
            version="2.0.0",
            description="Add extended metrics",
            up_fn=self._migrate_schema_v2_up,
            down_fn=self._migrate_schema_v2_down,
            dependencies=["schema_v1"],
        ))

    def register_migration(self, migration: Migration) -> None:
        """Register a migration."""
        self._migrations[migration.name] = migration

    def migrate(
        self,
        target_version: Optional[str] = None,
        dry_run: Optional[bool] = None,
    ) -> List[MigrationResult]:
        """
        Run migrations to target version.

        Args:
            target_version: Target version (latest if not specified)
            dry_run: Preview changes without applying

        Returns:
            List of migration results
        """
        dry_run = dry_run if dry_run is not None else self.config.dry_run
        results = []

        # Determine migrations to run
        pending = self._get_pending_migrations(target_version)

        if not pending:
            return results

        self._emit_event("start", {
            "migrations": [m.name for m in pending],
            "target_version": target_version,
            "dry_run": dry_run,
        })

        # Create backup if configured
        backup_path = None
        if self.config.backup_before and not dry_run:
            backup_path = self._create_backup()

        # Run migrations
        for migration in pending:
            result = self._run_migration(migration, MigrationDirection.UP, dry_run)
            results.append(result)

            if result.status == MigrationStatus.FAILED:
                if self.config.auto_rollback and backup_path:
                    self._restore_backup(backup_path)
                break

            if not dry_run:
                self._state.applied_migrations.append(migration.name)
                self._state.current_version = migration.version
                self._state.last_migration_at = time.time()
                self._state.history.append(result)

        # Save state
        if not dry_run:
            self._save_state()

        self._emit_event("complete", {
            "migrations": len(results),
            "succeeded": sum(1 for r in results if r.status == MigrationStatus.COMPLETED),
            "failed": sum(1 for r in results if r.status == MigrationStatus.FAILED),
        })

        return results

    def rollback(
        self,
        count: int = 1,
        dry_run: Optional[bool] = None,
    ) -> List[MigrationResult]:
        """
        Rollback migrations.

        Args:
            count: Number of migrations to rollback
            dry_run: Preview changes without applying

        Returns:
            List of rollback results
        """
        dry_run = dry_run if dry_run is not None else self.config.dry_run
        results = []

        # Get migrations to rollback
        to_rollback = self._state.applied_migrations[-count:][::-1]

        for migration_name in to_rollback:
            migration = self._migrations.get(migration_name)
            if not migration:
                continue

            result = self._run_migration(migration, MigrationDirection.DOWN, dry_run)
            results.append(result)

            if result.status == MigrationStatus.FAILED:
                break

            if not dry_run:
                self._state.applied_migrations.remove(migration_name)
                self._state.history.append(result)

        # Update current version
        if not dry_run and self._state.applied_migrations:
            last_migration = self._migrations.get(self._state.applied_migrations[-1])
            if last_migration:
                self._state.current_version = last_migration.version
            self._save_state()

        self._emit_event("rollback", {
            "count": len(results),
            "succeeded": sum(1 for r in results if r.status == MigrationStatus.COMPLETED),
        })

        return results

    def _run_migration(
        self,
        migration: Migration,
        direction: MigrationDirection,
        dry_run: bool,
    ) -> MigrationResult:
        """Run a single migration."""
        result = MigrationResult(
            migration_name=migration.name,
            direction=direction,
        )

        try:
            result.status = MigrationStatus.RUNNING

            # Get migration function
            fn = migration.up_fn if direction == MigrationDirection.UP else migration.down_fn

            if fn is None:
                result.status = MigrationStatus.SKIPPED
                result.completed_at = time.time()
                return result

            if dry_run:
                result.status = MigrationStatus.COMPLETED
                result.changes.append(f"[DRY RUN] Would run {direction.value} migration: {migration.name}")
            else:
                # Create context for migration
                context = {
                    "output_dir": self.config.output_dir,
                    "state": self._state.to_dict(),
                    "migration": migration.to_dict(),
                }

                # Run migration
                success = fn(context)

                if success:
                    result.status = MigrationStatus.COMPLETED
                    result.changes.append(f"Applied {direction.value} migration: {migration.name}")
                else:
                    result.status = MigrationStatus.FAILED
                    result.error = "Migration function returned False"

        except Exception as e:
            result.status = MigrationStatus.FAILED
            result.error = str(e)

        result.completed_at = time.time()
        return result

    def _get_pending_migrations(self, target_version: Optional[str]) -> List[Migration]:
        """Get migrations pending to run."""
        # Sort migrations by version
        all_migrations = sorted(
            self._migrations.values(),
            key=lambda m: m.version,
        )

        pending = []
        for migration in all_migrations:
            if migration.name in self._state.applied_migrations:
                continue

            # Check dependencies
            deps_met = all(
                dep in self._state.applied_migrations
                for dep in migration.dependencies
            )
            if not deps_met:
                continue

            pending.append(migration)

            # Stop at target version
            if target_version and migration.version == target_version:
                break

        return pending

    def _create_backup(self) -> Path:
        """Create a backup of current state."""
        backup_dir = Path(self.config.output_dir) / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}.json"

        backup_data = {
            "state": self._state.to_dict(),
            "timestamp": time.time(),
        }

        with open(backup_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(backup_data, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return backup_path

    def _restore_backup(self, backup_path: Path) -> bool:
        """Restore from a backup."""
        try:
            with open(backup_path) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    backup_data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            state_dict = backup_data.get("state", {})
            self._state = MigrationState(
                current_version=state_dict.get("current_version", "0.0.0"),
                applied_migrations=state_dict.get("applied_migrations", []),
                last_migration_at=state_dict.get("last_migration_at"),
            )
            self._save_state()
            return True

        except (IOError, json.JSONDecodeError):
            return False

    def _load_state(self) -> None:
        """Load migration state from disk."""
        state_path = Path(self.config.state_file)

        if state_path.exists():
            try:
                with open(state_path) as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        state_dict = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                self._state = MigrationState(
                    current_version=state_dict.get("current_version", "0.0.0"),
                    applied_migrations=state_dict.get("applied_migrations", []),
                    last_migration_at=state_dict.get("last_migration_at"),
                )

            except (IOError, json.JSONDecodeError):
                pass

    def _save_state(self) -> None:
        """Save migration state to disk."""
        state_path = Path(self.config.state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        # Trim history
        if len(self._state.history) > self.config.max_history:
            self._state.history = self._state.history[-self.config.max_history:]

        with open(state_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(self._state.to_dict(), f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    # Built-in migration implementations
    def _migrate_schema_v1_up(self, context: Dict[str, Any]) -> bool:
        """Schema v1 up migration."""
        output_dir = Path(context["output_dir"])
        (output_dir / "data").mkdir(parents=True, exist_ok=True)
        (output_dir / "reports").mkdir(parents=True, exist_ok=True)
        return True

    def _migrate_schema_v1_down(self, context: Dict[str, Any]) -> bool:
        """Schema v1 down migration."""
        return True

    def _migrate_schema_v2_up(self, context: Dict[str, Any]) -> bool:
        """Schema v2 up migration."""
        output_dir = Path(context["output_dir"])
        (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
        return True

    def _migrate_schema_v2_down(self, context: Dict[str, Any]) -> bool:
        """Schema v2 down migration."""
        output_dir = Path(context["output_dir"])
        metrics_dir = output_dir / "metrics"
        if metrics_dir.exists():
            shutil.rmtree(metrics_dir)
        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        return {
            "current_version": self._state.current_version,
            "applied_count": len(self._state.applied_migrations),
            "pending_count": len(self._get_pending_migrations(None)),
            "last_migration": self._state.last_migration_at,
        }

    def list_migrations(self) -> List[Dict[str, Any]]:
        """List all migrations."""
        results = []
        for migration in sorted(self._migrations.values(), key=lambda m: m.version):
            results.append({
                **migration.to_dict(),
                "applied": migration.name in self._state.applied_migrations,
            })
        return results

    async def migrate_async(
        self,
        target_version: Optional[str] = None,
        dry_run: Optional[bool] = None,
    ) -> List[MigrationResult]:
        """Async version of migrate."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.migrate, target_version, dry_run
        )

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.migration.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "migration",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Migration Manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Migration Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run migrations")
    migrate_parser.add_argument("--target", help="Target version")
    migrate_parser.add_argument("--dry-run", action="store_true", help="Preview changes")

    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migrations")
    rollback_parser.add_argument("--count", type=int, default=1, help="Number to rollback")
    rollback_parser.add_argument("--dry-run", action="store_true", help="Preview changes")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show migration status")

    # List command
    list_parser = subparsers.add_parser("list", help="List migrations")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/migrations")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = MigrationConfig(output_dir=args.output)
    manager = TestMigrationManager(config=config)

    if args.command == "migrate":
        results = manager.migrate(
            target_version=args.target,
            dry_run=args.dry_run,
        )

        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            if not results:
                print("No migrations to run")
            else:
                print(f"\nMigrations ({'DRY RUN' if args.dry_run else 'APPLIED'}):")
                for result in results:
                    status = "[OK]" if result.status == MigrationStatus.COMPLETED else "[FAIL]"
                    print(f"  {status} {result.migration_name} ({result.duration_ms:.2f}ms)")
                    if result.error:
                        print(f"      Error: {result.error}")

    elif args.command == "rollback":
        results = manager.rollback(
            count=args.count,
            dry_run=args.dry_run,
        )

        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            if not results:
                print("Nothing to rollback")
            else:
                print(f"\nRollbacks ({'DRY RUN' if args.dry_run else 'APPLIED'}):")
                for result in results:
                    status = "[OK]" if result.status == MigrationStatus.COMPLETED else "[FAIL]"
                    print(f"  {status} {result.migration_name}")

    elif args.command == "status":
        status = manager.get_status()

        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("\nMigration Status:")
            print(f"  Current Version: {status['current_version']}")
            print(f"  Applied: {status['applied_count']}")
            print(f"  Pending: {status['pending_count']}")
            if status['last_migration']:
                dt = datetime.fromtimestamp(status['last_migration'])
                print(f"  Last Migration: {dt.isoformat()}")

    elif args.command == "list":
        migrations = manager.list_migrations()

        if args.json:
            print(json.dumps(migrations, indent=2))
        else:
            print("\nMigrations:")
            for m in migrations:
                status = "[APPLIED]" if m["applied"] else "[PENDING]"
                print(f"  {status} {m['name']} (v{m['version']})")
                print(f"      {m['description']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
