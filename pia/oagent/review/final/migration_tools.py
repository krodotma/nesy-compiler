#!/usr/bin/env python3
"""
Migration Tools (Step 195)

Data migration utilities for the Review Agent with versioned migrations,
rollback support, and migration tracking.

PBTSO Phase: BUILD, VERIFY
Bus Topics: review.migration.run, review.migration.complete, review.migration.rollback

Migration Features:
- Versioned migrations
- Forward and backward migrations
- Migration tracking
- Dry-run support
- Rollback capability

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Awaitable

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900


# ============================================================================
# Types
# ============================================================================

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
class Migration:
    """
    A single migration definition.

    Attributes:
        version: Migration version (semantic versioning)
        name: Migration name
        description: Migration description
        up: Forward migration function
        down: Backward migration function
        dependencies: Required migrations
        reversible: Whether migration can be rolled back
        batch: Batch number for grouping
    """
    version: str
    name: str
    description: str = ""
    up: Optional[Callable[..., Awaitable[bool]]] = None
    down: Optional[Callable[..., Awaitable[bool]]] = None
    dependencies: List[str] = field(default_factory=list)
    reversible: bool = True
    batch: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "dependencies": self.dependencies,
            "reversible": self.reversible,
            "batch": self.batch,
        }


@dataclass
class MigrationRecord:
    """
    Record of an executed migration.

    Attributes:
        version: Migration version
        name: Migration name
        status: Execution status
        direction: Migration direction
        started_at: Start timestamp
        completed_at: Completion timestamp
        duration_ms: Execution duration
        error: Error message (if failed)
        batch: Batch number
    """
    version: str
    name: str
    status: MigrationStatus = MigrationStatus.PENDING
    direction: MigrationDirection = MigrationDirection.UP
    started_at: str = ""
    completed_at: str = ""
    duration_ms: float = 0
    error: Optional[str] = None
    batch: int = 0

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "name": self.name,
            "status": self.status.value,
            "direction": self.direction.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
            "batch": self.batch,
        }


@dataclass
class MigrationResult:
    """
    Result of migration execution.

    Attributes:
        success: Whether all migrations succeeded
        records: Individual migration records
        total: Total migrations
        completed: Completed migrations
        failed: Failed migrations
        skipped: Skipped migrations
        duration_ms: Total execution time
    """
    success: bool
    records: List[MigrationRecord] = field(default_factory=list)
    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "records": [r.to_dict() for r in self.records],
            "summary": {
                "total": self.total,
                "completed": self.completed,
                "failed": self.failed,
                "skipped": self.skipped,
            },
            "duration_ms": round(self.duration_ms, 2),
        }


# ============================================================================
# Migration Runner
# ============================================================================

class MigrationRunner:
    """
    Executes migrations.

    Example:
        runner = MigrationRunner(state_path="/path/to/state.json")

        # Run all pending migrations
        result = await runner.migrate()

        # Rollback last batch
        result = await runner.rollback()

        # Migrate to specific version
        result = await runner.migrate_to("1.2.0")
    """

    def __init__(
        self,
        state_path: Optional[Path] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize migration runner.

        Args:
            state_path: Path to migration state file
            bus_path: Path to event bus file
        """
        self.state_path = state_path or self._get_state_path()
        self.bus_path = bus_path or self._get_bus_path()

        self._migrations: Dict[str, Migration] = {}
        self._state: Dict[str, MigrationRecord] = {}
        self._current_batch = 0

        self._load_state()

    def _get_state_path(self) -> Path:
        """Get default state path."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return pluribus_root / ".pluribus" / "review" / "migrations.json"

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "migration") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "migration-runner",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _load_state(self) -> None:
        """Load migration state from file."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        data = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                self._current_batch = data.get("current_batch", 0)
                for version, record_data in data.get("migrations", {}).items():
                    self._state[version] = MigrationRecord(
                        version=record_data["version"],
                        name=record_data["name"],
                        status=MigrationStatus(record_data["status"]),
                        direction=MigrationDirection(record_data["direction"]),
                        started_at=record_data.get("started_at", ""),
                        completed_at=record_data.get("completed_at", ""),
                        duration_ms=record_data.get("duration_ms", 0),
                        error=record_data.get("error"),
                        batch=record_data.get("batch", 0),
                    )
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_state(self) -> None:
        """Save migration state to file."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "current_batch": self._current_batch,
            "migrations": {v: r.to_dict() for v, r in self._state.items()},
            "updated_at": datetime.now(timezone.utc).isoformat() + "Z",
        }

        with open(self.state_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def register(self, migration: Migration) -> None:
        """Register a migration."""
        self._migrations[migration.version] = migration

    def get_pending(self) -> List[Migration]:
        """Get pending migrations."""
        pending = []
        for version in sorted(self._migrations.keys()):
            if version not in self._state or self._state[version].status != MigrationStatus.COMPLETED:
                pending.append(self._migrations[version])
        return pending

    def get_completed(self) -> List[MigrationRecord]:
        """Get completed migrations."""
        return [
            r for r in self._state.values()
            if r.status == MigrationStatus.COMPLETED
        ]

    async def migrate(
        self,
        dry_run: bool = False,
        steps: Optional[int] = None,
    ) -> MigrationResult:
        """
        Run pending migrations.

        Args:
            dry_run: Simulate without executing
            steps: Number of migrations to run (None = all)

        Returns:
            MigrationResult
        """
        start_time = time.time()
        pending = self.get_pending()

        if steps:
            pending = pending[:steps]

        result = MigrationResult(
            success=True,
            total=len(pending),
        )

        if not pending:
            return result

        self._current_batch += 1

        self._emit_event("review.migration.run", {
            "direction": "up",
            "migrations": [m.version for m in pending],
            "dry_run": dry_run,
            "batch": self._current_batch,
        })

        for migration in pending:
            record = await self._run_migration(migration, MigrationDirection.UP, dry_run)
            result.records.append(record)

            if record.status == MigrationStatus.COMPLETED:
                result.completed += 1
                self._state[migration.version] = record
            elif record.status == MigrationStatus.FAILED:
                result.failed += 1
                result.success = False
                self._state[migration.version] = record
                break  # Stop on failure
            elif record.status == MigrationStatus.SKIPPED:
                result.skipped += 1

        result.duration_ms = (time.time() - start_time) * 1000

        if not dry_run:
            self._save_state()

        self._emit_event("review.migration.complete", {
            "direction": "up",
            "success": result.success,
            "completed": result.completed,
            "failed": result.failed,
            "batch": self._current_batch,
        })

        return result

    async def migrate_to(
        self,
        target_version: str,
        dry_run: bool = False,
    ) -> MigrationResult:
        """
        Migrate to a specific version.

        Args:
            target_version: Target version
            dry_run: Simulate without executing

        Returns:
            MigrationResult
        """
        pending = []
        for version in sorted(self._migrations.keys()):
            if version <= target_version:
                if version not in self._state or self._state[version].status != MigrationStatus.COMPLETED:
                    pending.append(self._migrations[version])

        # Temporarily store pending list
        original_pending = self.get_pending()

        # Filter to only migrations up to target
        self._migrations_to_run = pending
        result = await self.migrate(dry_run=dry_run)

        return result

    async def rollback(
        self,
        steps: int = 1,
        dry_run: bool = False,
    ) -> MigrationResult:
        """
        Rollback migrations.

        Args:
            steps: Number of batches to rollback
            dry_run: Simulate without executing

        Returns:
            MigrationResult
        """
        start_time = time.time()

        # Get migrations to rollback (most recent first)
        to_rollback = []
        batches_to_rollback = set()

        for _ in range(steps):
            batch = self._current_batch - len(batches_to_rollback)
            if batch <= 0:
                break
            batches_to_rollback.add(batch)

        for version, record in sorted(self._state.items(), reverse=True):
            if record.batch in batches_to_rollback and record.status == MigrationStatus.COMPLETED:
                migration = self._migrations.get(version)
                if migration and migration.reversible:
                    to_rollback.append(migration)

        result = MigrationResult(
            success=True,
            total=len(to_rollback),
        )

        if not to_rollback:
            return result

        self._emit_event("review.migration.rollback", {
            "direction": "down",
            "migrations": [m.version for m in to_rollback],
            "dry_run": dry_run,
            "batches": list(batches_to_rollback),
        })

        for migration in to_rollback:
            record = await self._run_migration(migration, MigrationDirection.DOWN, dry_run)
            result.records.append(record)

            if record.status == MigrationStatus.COMPLETED:
                result.completed += 1
                self._state[migration.version].status = MigrationStatus.ROLLED_BACK
            elif record.status == MigrationStatus.FAILED:
                result.failed += 1
                result.success = False
                break
            elif record.status == MigrationStatus.SKIPPED:
                result.skipped += 1

        result.duration_ms = (time.time() - start_time) * 1000

        if not dry_run and result.success:
            self._current_batch -= steps
            self._save_state()

        return result

    async def _run_migration(
        self,
        migration: Migration,
        direction: MigrationDirection,
        dry_run: bool,
    ) -> MigrationRecord:
        """Run a single migration."""
        start_time = time.time()

        record = MigrationRecord(
            version=migration.version,
            name=migration.name,
            direction=direction,
            batch=self._current_batch,
        )

        # Check dependencies
        if direction == MigrationDirection.UP:
            for dep in migration.dependencies:
                if dep not in self._state or self._state[dep].status != MigrationStatus.COMPLETED:
                    record.status = MigrationStatus.SKIPPED
                    record.error = f"Dependency not met: {dep}"
                    return record

        # Check reversibility
        if direction == MigrationDirection.DOWN and not migration.reversible:
            record.status = MigrationStatus.SKIPPED
            record.error = "Migration is not reversible"
            return record

        if dry_run:
            record.status = MigrationStatus.SKIPPED
            record.error = "Dry run - not executed"
            return record

        record.status = MigrationStatus.RUNNING

        try:
            func = migration.up if direction == MigrationDirection.UP else migration.down
            if func:
                success = await func()
                if success:
                    record.status = MigrationStatus.COMPLETED
                else:
                    record.status = MigrationStatus.FAILED
                    record.error = "Migration returned False"
            else:
                # No function defined - consider it a schema-only migration
                record.status = MigrationStatus.COMPLETED

        except Exception as e:
            record.status = MigrationStatus.FAILED
            record.error = str(e)

        record.duration_ms = (time.time() - start_time) * 1000
        record.completed_at = datetime.now(timezone.utc).isoformat() + "Z"

        return record

    def get_status(self) -> Dict[str, Any]:
        """Get migration status."""
        pending = self.get_pending()
        completed = self.get_completed()

        return {
            "current_batch": self._current_batch,
            "registered": len(self._migrations),
            "completed": len(completed),
            "pending": len(pending),
            "pending_versions": [m.version for m in pending],
        }


# ============================================================================
# Migration Manager
# ============================================================================

class MigrationManager:
    """
    High-level migration management.

    Example:
        manager = MigrationManager()

        # Define migration
        @manager.migration("1.0.0", "Initial setup")
        async def migrate_1_0_0():
            # Migration logic
            return True

        # Run migrations
        result = await manager.run()
    """

    BUS_TOPICS = {
        "run": "review.migration.run",
        "complete": "review.migration.complete",
        "rollback": "review.migration.rollback",
    }

    def __init__(
        self,
        state_path: Optional[Path] = None,
        bus_path: Optional[Path] = None,
    ):
        """Initialize migration manager."""
        self.bus_path = bus_path or self._get_bus_path()
        self.runner = MigrationRunner(state_path, bus_path)
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "migration") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "migration-manager",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def migration(
        self,
        version: str,
        name: str,
        description: str = "",
        dependencies: Optional[List[str]] = None,
        reversible: bool = True,
    ) -> Callable:
        """
        Decorator to define a migration.

        Args:
            version: Migration version
            name: Migration name
            description: Description
            dependencies: Required migrations
            reversible: Can be rolled back
        """
        def decorator(func: Callable[..., Awaitable[bool]]) -> Callable:
            migration = Migration(
                version=version,
                name=name,
                description=description,
                up=func,
                dependencies=dependencies or [],
                reversible=reversible,
            )
            self.runner.register(migration)
            return func
        return decorator

    def register(self, migration: Migration) -> None:
        """Register a migration."""
        self.runner.register(migration)

    async def run(
        self,
        dry_run: bool = False,
        steps: Optional[int] = None,
    ) -> MigrationResult:
        """Run pending migrations."""
        return await self.runner.migrate(dry_run=dry_run, steps=steps)

    async def rollback(
        self,
        steps: int = 1,
        dry_run: bool = False,
    ) -> MigrationResult:
        """Rollback migrations."""
        return await self.runner.rollback(steps=steps, dry_run=dry_run)

    def status(self) -> Dict[str, Any]:
        """Get migration status."""
        return self.runner.get_status()

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        status = self.runner.get_status()
        result = {
            "agent": "migration-manager",
            "healthy": True,
            "migrations_registered": status["registered"],
            "migrations_pending": status["pending"],
            "current_batch": status["current_batch"],
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self._emit_event("a2a.heartbeat", result, kind="heartbeat")
        return result


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Migration Tools."""
    import argparse

    parser = argparse.ArgumentParser(description="Migration Tools (Step 195)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run migrations")
    run_parser.add_argument("--dry-run", action="store_true", help="Simulate only")
    run_parser.add_argument("--steps", type=int, help="Number of migrations")

    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migrations")
    rollback_parser.add_argument("--steps", type=int, default=1, help="Batches to rollback")
    rollback_parser.add_argument("--dry-run", action="store_true", help="Simulate only")

    # Status command
    subparsers.add_parser("status", help="Show migration status")

    # List command
    subparsers.add_parser("list", help="List migrations")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    manager = MigrationManager()

    # Register some demo migrations
    @manager.migration("0.1.0", "Create initial schema")
    async def migrate_0_1_0():
        print("  Creating initial schema...")
        return True

    @manager.migration("0.2.0", "Add review tables", dependencies=["0.1.0"])
    async def migrate_0_2_0():
        print("  Adding review tables...")
        return True

    @manager.migration("0.3.0", "Add metrics", dependencies=["0.2.0"])
    async def migrate_0_3_0():
        print("  Adding metrics...")
        return True

    if args.command == "run":
        result = asyncio.run(manager.run(dry_run=args.dry_run, steps=args.steps))

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "SUCCESS" if result.success else "FAILED"
            print(f"Migration: {status}")
            print(f"  Completed: {result.completed}/{result.total}")
            print(f"  Failed: {result.failed}")
            print(f"  Duration: {result.duration_ms:.1f}ms")
            for record in result.records:
                symbol = "+" if record.status == MigrationStatus.COMPLETED else "-"
                print(f"  [{symbol}] {record.version}: {record.name}")

        return 0 if result.success else 1

    elif args.command == "rollback":
        result = asyncio.run(manager.rollback(steps=args.steps, dry_run=args.dry_run))

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "SUCCESS" if result.success else "FAILED"
            print(f"Rollback: {status}")
            print(f"  Rolled back: {result.completed}/{result.total}")

        return 0 if result.success else 1

    elif args.command == "status":
        status = manager.status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Migration Status")
            print(f"  Current Batch: {status['current_batch']}")
            print(f"  Registered: {status['registered']}")
            print(f"  Completed: {status['completed']}")
            print(f"  Pending: {status['pending']}")
            if status['pending_versions']:
                print(f"  Pending Versions: {', '.join(status['pending_versions'])}")

    elif args.command == "list":
        status = manager.status()
        if args.json:
            migrations = [m.to_dict() for m in manager.runner._migrations.values()]
            print(json.dumps(migrations, indent=2))
        else:
            print(f"Migrations: {status['registered']}")
            for version in sorted(manager.runner._migrations.keys()):
                m = manager.runner._migrations[version]
                state = manager.runner._state.get(version)
                indicator = "+" if state and state.status == MigrationStatus.COMPLETED else " "
                print(f"  [{indicator}] {version}: {m.name}")

    else:
        # Default: show status
        status = manager.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Migration Manager: {status['migrations_registered']} migrations, {status['migrations_pending']} pending")

    return 0


if __name__ == "__main__":
    sys.exit(main())
