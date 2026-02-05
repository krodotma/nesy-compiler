#!/usr/bin/env python3
"""
migration_tools.py - Data Migration Tools (Step 45)

Data migration utilities for Research Agent including schema migrations,
data transformations, and version upgrades.

PBTSO Phase: TRANSITION

Bus Topics:
- a2a.research.migration.start
- a2a.research.migration.complete
- a2a.research.migration.error
- research.migration.rollback

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import socket
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
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
class MigrationConfig:
    """Configuration for migration system."""

    migrations_dir: str = ""
    history_file: str = ""
    backup_before_migrate: bool = True
    backup_dir: str = ""
    dry_run: bool = False
    batch_size: int = 1000
    timeout_seconds: int = 3600
    emit_to_bus: bool = True
    bus_path: Optional[str] = None

    def __post_init__(self):
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        if not self.migrations_dir:
            self.migrations_dir = f"{pluribus_root}/.pluribus/research/migrations"
        if not self.history_file:
            self.history_file = f"{pluribus_root}/.pluribus/research/migration_history.json"
        if not self.backup_dir:
            self.backup_dir = f"{pluribus_root}/.pluribus/research/backups"
        if self.bus_path is None:
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class MigrationRecord:
    """Record of a migration execution."""

    id: str
    version: str
    name: str
    status: MigrationStatus
    applied_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: float = 0
    error: Optional[str] = None
    checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "version": self.version,
            "name": self.name,
            "status": self.status.value,
            "applied_at": self.applied_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MigrationRecord":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            version=data["version"],
            name=data["name"],
            status=MigrationStatus(data["status"]),
            applied_at=data.get("applied_at"),
            completed_at=data.get("completed_at"),
            duration_ms=data.get("duration_ms", 0),
            error=data.get("error"),
            checksum=data.get("checksum"),
        )


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    success: bool
    records: List[MigrationRecord] = field(default_factory=list)
    applied: int = 0
    rolled_back: int = 0
    skipped: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "applied": self.applied,
            "rolled_back": self.rolled_back,
            "skipped": self.skipped,
            "error_count": len(self.errors),
        }


@dataclass
class DataTransformation:
    """A data transformation rule."""

    source_field: str
    target_field: str
    transform_fn: Callable[[Any], Any]
    default_value: Any = None
    required: bool = False

    def apply(self, data: Dict[str, Any]) -> Tuple[str, Any]:
        """Apply transformation to data."""
        value = data.get(self.source_field, self.default_value)

        if value is None and self.required:
            raise ValueError(f"Required field missing: {self.source_field}")

        if value is not None:
            value = self.transform_fn(value)

        return self.target_field, value


# ============================================================================
# Migration Base
# ============================================================================


class Migration(ABC):
    """
    Abstract base for migrations.

    Example:
        class AddTimestampField(Migration):
            version = "001"
            name = "add_timestamp_field"

            def up(self, context: MigrationContext) -> None:
                # Add timestamp to all records
                for record in context.get_all_records():
                    record["created_at"] = time.time()
                    context.update_record(record)

            def down(self, context: MigrationContext) -> None:
                # Remove timestamp from all records
                for record in context.get_all_records():
                    del record["created_at"]
                    context.update_record(record)
    """

    version: str = "000"
    name: str = "unnamed"
    description: str = ""
    dependencies: List[str] = []

    @abstractmethod
    def up(self, context: "MigrationContext") -> None:
        """Apply the migration."""
        pass

    @abstractmethod
    def down(self, context: "MigrationContext") -> None:
        """Rollback the migration."""
        pass

    def validate(self, context: "MigrationContext") -> bool:
        """Validate migration can be applied."""
        return True

    def get_checksum(self) -> str:
        """Get migration checksum for verification."""
        import inspect
        source = inspect.getsource(self.__class__)
        return hashlib.md5(source.encode()).hexdigest()[:16]


# ============================================================================
# Migration Context
# ============================================================================


class MigrationContext:
    """
    Context for migration execution.

    Provides access to data and utilities during migration.
    """

    def __init__(
        self,
        data_dir: str,
        config: MigrationConfig,
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self._modified_files: Set[str] = set()
        self._created_files: Set[str] = set()
        self._deleted_files: Set[str] = set()

    def get_file(self, path: str) -> Optional[Dict[str, Any]]:
        """Read a JSON file."""
        file_path = self.data_dir / path
        if file_path.exists():
            with open(file_path) as f:
                return json.load(f)
        return None

    def put_file(self, path: str, data: Dict[str, Any]) -> None:
        """Write a JSON file."""
        if self.config.dry_run:
            return

        file_path = self.data_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        if file_path.exists():
            self._modified_files.add(str(file_path))
        else:
            self._created_files.add(str(file_path))

    def delete_file(self, path: str) -> bool:
        """Delete a file."""
        if self.config.dry_run:
            return True

        file_path = self.data_dir / path
        if file_path.exists():
            file_path.unlink()
            self._deleted_files.add(str(file_path))
            return True
        return False

    def list_files(self, pattern: str = "*.json") -> List[str]:
        """List files matching pattern."""
        return [str(p.relative_to(self.data_dir)) for p in self.data_dir.glob(pattern)]

    def transform_file(
        self,
        path: str,
        transformations: List[DataTransformation],
    ) -> None:
        """Apply transformations to a file."""
        data = self.get_file(path)
        if not data:
            return

        new_data = {}
        for transform in transformations:
            field, value = transform.apply(data)
            new_data[field] = value

        # Preserve fields not in transformations
        for key, value in data.items():
            if key not in new_data:
                new_data[key] = value

        self.put_file(path, new_data)

    def batch_process(
        self,
        files: List[str],
        processor: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> int:
        """Process files in batches."""
        processed = 0

        for i in range(0, len(files), self.config.batch_size):
            batch = files[i:i + self.config.batch_size]

            for file_path in batch:
                data = self.get_file(file_path)
                if data:
                    new_data = processor(data)
                    self.put_file(file_path, new_data)
                    processed += 1

        return processed

    def execute_sql(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute SQL query (placeholder for FalkorDB integration)."""
        # This would integrate with FalkorDB on port 6380
        pass

    def log(self, message: str) -> None:
        """Log migration message."""
        print(f"[MIGRATION] {message}")


# ============================================================================
# Migration Manager
# ============================================================================


class MigrationManager:
    """
    Manager for data migrations.

    Features:
    - Version tracking
    - Up/down migrations
    - Automatic backups
    - Dry run mode
    - Batch processing

    PBTSO Phase: TRANSITION

    Example:
        manager = MigrationManager()

        # Register migrations
        manager.register(AddTimestampMigration())

        # Run migrations
        result = manager.migrate()

        # Rollback last migration
        result = manager.rollback(steps=1)
    """

    def __init__(
        self,
        config: Optional[MigrationConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the migration manager.

        Args:
            config: Migration configuration
            bus: AgentBus for event emission
        """
        self.config = config or MigrationConfig()
        self.bus = bus or AgentBus()

        self._migrations: Dict[str, Migration] = {}
        self._history: List[MigrationRecord] = []

        # Load history
        self._load_history()

        # Statistics
        self._stats = {
            "total_migrations": 0,
            "successful": 0,
            "failed": 0,
            "rolled_back": 0,
        }

    def register(self, migration: Migration) -> None:
        """Register a migration."""
        self._migrations[migration.version] = migration

    def get_pending(self) -> List[Migration]:
        """Get list of pending migrations."""
        applied_versions = {r.version for r in self._history if r.status == MigrationStatus.COMPLETED}
        pending = []

        for version, migration in sorted(self._migrations.items()):
            if version not in applied_versions:
                pending.append(migration)

        return pending

    def get_applied(self) -> List[MigrationRecord]:
        """Get list of applied migrations."""
        return [r for r in self._history if r.status == MigrationStatus.COMPLETED]

    def migrate(
        self,
        target_version: Optional[str] = None,
        data_dir: Optional[str] = None,
    ) -> MigrationResult:
        """
        Run pending migrations.

        Args:
            target_version: Optional target version
            data_dir: Data directory to migrate

        Returns:
            MigrationResult with execution details
        """
        data_path = data_dir or str(Path(self.config.migrations_dir).parent / "data")
        context = MigrationContext(data_path, self.config)

        pending = self.get_pending()
        if target_version:
            pending = [m for m in pending if m.version <= target_version]

        if not pending:
            return MigrationResult(success=True, skipped=len(self._migrations))

        self._emit_event("a2a.research.migration.start", {
            "count": len(pending),
            "versions": [m.version for m in pending],
            "dry_run": self.config.dry_run,
        })

        result = MigrationResult(success=True)

        # Backup if needed
        if self.config.backup_before_migrate and not self.config.dry_run:
            self._create_backup(data_path)

        for migration in pending:
            record = self._run_migration(migration, context, MigrationDirection.UP)
            result.records.append(record)

            if record.status == MigrationStatus.COMPLETED:
                result.applied += 1
                self._stats["successful"] += 1
            elif record.status == MigrationStatus.FAILED:
                result.errors.append(record.error or "Unknown error")
                result.success = False
                self._stats["failed"] += 1
                break
            else:
                result.skipped += 1

        self._save_history()

        self._emit_event("a2a.research.migration.complete", result.to_dict())

        return result

    def rollback(
        self,
        steps: int = 1,
        data_dir: Optional[str] = None,
    ) -> MigrationResult:
        """
        Rollback applied migrations.

        Args:
            steps: Number of migrations to rollback
            data_dir: Data directory

        Returns:
            MigrationResult with execution details
        """
        data_path = data_dir or str(Path(self.config.migrations_dir).parent / "data")
        context = MigrationContext(data_path, self.config)

        applied = self.get_applied()
        to_rollback = applied[-steps:] if steps <= len(applied) else applied

        if not to_rollback:
            return MigrationResult(success=True)

        result = MigrationResult(success=True)

        # Rollback in reverse order
        for record in reversed(to_rollback):
            migration = self._migrations.get(record.version)
            if not migration:
                result.errors.append(f"Migration {record.version} not found")
                continue

            rollback_record = self._run_migration(migration, context, MigrationDirection.DOWN)
            result.records.append(rollback_record)

            if rollback_record.status == MigrationStatus.COMPLETED:
                result.rolled_back += 1
                self._stats["rolled_back"] += 1

                # Update history
                record.status = MigrationStatus.ROLLED_BACK
            else:
                result.errors.append(rollback_record.error or "Unknown error")
                result.success = False
                break

        self._save_history()

        self._emit_event("research.migration.rollback", result.to_dict())

        return result

    def create_migration(
        self,
        name: str,
        description: str = "",
    ) -> str:
        """
        Create a new migration file.

        Args:
            name: Migration name
            description: Migration description

        Returns:
            Path to created migration file
        """
        # Generate version
        existing = sorted(self._migrations.keys())
        if existing:
            last_version = int(existing[-1])
            version = f"{last_version + 1:03d}"
        else:
            version = "001"

        # Create migration file
        migrations_dir = Path(self.config.migrations_dir)
        migrations_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{version}_{name}.py"
        file_path = migrations_dir / filename

        template = f'''#!/usr/bin/env python3
"""
Migration {version}: {name}

{description}
"""
from migration_tools import Migration, MigrationContext


class Migration{version}(Migration):
    version = "{version}"
    name = "{name}"
    description = """{description}"""

    def up(self, context: MigrationContext) -> None:
        """Apply the migration."""
        # TODO: Implement migration logic
        pass

    def down(self, context: MigrationContext) -> None:
        """Rollback the migration."""
        # TODO: Implement rollback logic
        pass
'''

        file_path.write_text(template)
        return str(file_path)

    def status(self) -> Dict[str, Any]:
        """Get migration status."""
        pending = self.get_pending()
        applied = self.get_applied()

        return {
            "total_registered": len(self._migrations),
            "applied": len(applied),
            "pending": len(pending),
            "pending_versions": [m.version for m in pending],
            "last_applied": applied[-1].to_dict() if applied else None,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get migration statistics."""
        return {
            **self._stats,
            "registered": len(self._migrations),
            "history_entries": len(self._history),
        }

    def _run_migration(
        self,
        migration: Migration,
        context: MigrationContext,
        direction: MigrationDirection,
    ) -> MigrationRecord:
        """Run a single migration."""
        record = MigrationRecord(
            id=str(uuid.uuid4())[:8],
            version=migration.version,
            name=migration.name,
            status=MigrationStatus.RUNNING,
            applied_at=time.time(),
            checksum=migration.get_checksum(),
        )

        self._stats["total_migrations"] += 1

        try:
            # Validate
            if not migration.validate(context):
                record.status = MigrationStatus.SKIPPED
                record.error = "Validation failed"
                return record

            # Execute
            start_time = time.time()

            if direction == MigrationDirection.UP:
                migration.up(context)
            else:
                migration.down(context)

            record.duration_ms = (time.time() - start_time) * 1000
            record.completed_at = time.time()
            record.status = MigrationStatus.COMPLETED

            if direction == MigrationDirection.UP:
                self._history.append(record)

        except Exception as e:
            record.status = MigrationStatus.FAILED
            record.error = str(e)
            record.completed_at = time.time()

            self._emit_event("a2a.research.migration.error", {
                "version": migration.version,
                "error": str(e),
            }, level="error")

        return record

    def _create_backup(self, data_dir: str) -> str:
        """Create backup before migration."""
        backup_dir = Path(self.config.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}"

        shutil.copytree(data_dir, backup_path)
        return str(backup_path)

    def _load_history(self) -> None:
        """Load migration history."""
        history_path = Path(self.config.history_file)

        if history_path.exists():
            with open(history_path) as f:
                data = json.load(f)
                self._history = [
                    MigrationRecord.from_dict(r)
                    for r in data.get("records", [])
                ]

    def _save_history(self) -> None:
        """Save migration history."""
        history_path = Path(self.config.history_file)
        history_path.parent.mkdir(parents=True, exist_ok=True)

        with open(history_path, "w") as f:
            json.dump({
                "records": [r.to_dict() for r in self._history],
                "updated_at": time.time(),
            }, f, indent=2)

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
            "kind": "migration",
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
# Data Transformer
# ============================================================================


class DataTransformer:
    """
    Utility for transforming data between versions.

    Example:
        transformer = DataTransformer()

        # Add transformations
        transformer.add_field_rename("old_name", "new_name")
        transformer.add_type_cast("count", int)
        transformer.add_default("created_at", lambda: time.time())

        # Transform data
        new_data = transformer.transform(old_data)
    """

    def __init__(self):
        self._transformations: List[DataTransformation] = []

    def add_transformation(self, transformation: DataTransformation) -> "DataTransformer":
        """Add a transformation."""
        self._transformations.append(transformation)
        return self

    def add_field_rename(self, old_name: str, new_name: str) -> "DataTransformer":
        """Add field rename transformation."""
        return self.add_transformation(DataTransformation(
            source_field=old_name,
            target_field=new_name,
            transform_fn=lambda x: x,
        ))

    def add_type_cast(
        self,
        field: str,
        target_type: Type,
        default: Any = None,
    ) -> "DataTransformer":
        """Add type casting transformation."""
        return self.add_transformation(DataTransformation(
            source_field=field,
            target_field=field,
            transform_fn=lambda x: target_type(x) if x is not None else default,
            default_value=default,
        ))

    def add_default(
        self,
        field: str,
        default_fn: Callable[[], Any],
    ) -> "DataTransformer":
        """Add default value transformation."""
        return self.add_transformation(DataTransformation(
            source_field=field,
            target_field=field,
            transform_fn=lambda x: x if x is not None else default_fn(),
            default_value=None,
        ))

    def add_computed(
        self,
        target_field: str,
        compute_fn: Callable[[Dict[str, Any]], Any],
    ) -> "DataTransformer":
        """Add computed field transformation."""
        self._transformations.append(DataTransformation(
            source_field="__full__",
            target_field=target_field,
            transform_fn=compute_fn,
        ))
        return self

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data using registered transformations."""
        result = dict(data)

        for transformation in self._transformations:
            if transformation.source_field == "__full__":
                result[transformation.target_field] = transformation.transform_fn(data)
            else:
                field, value = transformation.apply(data)
                result[field] = value

                # Remove old field if renamed
                if transformation.source_field != transformation.target_field:
                    result.pop(transformation.source_field, None)

        return result


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Migration Tools."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Migration Tools (Step 45)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run migrations")
    migrate_parser.add_argument("--target", help="Target version")
    migrate_parser.add_argument("--data-dir", help="Data directory")
    migrate_parser.add_argument("--dry-run", action="store_true")

    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migrations")
    rollback_parser.add_argument("--steps", type=int, default=1, help="Steps to rollback")
    rollback_parser.add_argument("--data-dir", help="Data directory")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create migration")
    create_parser.add_argument("name", help="Migration name")
    create_parser.add_argument("--description", "-d", default="", help="Description")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show migration status")
    status_parser.add_argument("--json", action="store_true")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run migration demo")

    args = parser.parse_args()

    config = MigrationConfig()
    if hasattr(args, "dry_run") and args.dry_run:
        config.dry_run = True

    manager = MigrationManager(config)

    if args.command == "migrate":
        result = manager.migrate(args.target, args.data_dir)

        print(f"Migration {'completed' if result.success else 'failed'}")
        print(f"  Applied: {result.applied}")
        print(f"  Skipped: {result.skipped}")

        if result.errors:
            print("  Errors:")
            for error in result.errors:
                print(f"    - {error}")

        return 0 if result.success else 1

    elif args.command == "rollback":
        result = manager.rollback(args.steps, args.data_dir)

        print(f"Rollback {'completed' if result.success else 'failed'}")
        print(f"  Rolled back: {result.rolled_back}")

        return 0 if result.success else 1

    elif args.command == "create":
        path = manager.create_migration(args.name, args.description)
        print(f"Created migration: {path}")

    elif args.command == "status":
        status = manager.status()

        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("Migration Status:")
            print(f"  Registered: {status['total_registered']}")
            print(f"  Applied: {status['applied']}")
            print(f"  Pending: {status['pending']}")
            if status['pending_versions']:
                print(f"  Pending versions: {', '.join(status['pending_versions'])}")
            if status['last_applied']:
                print(f"  Last applied: {status['last_applied']['version']} - {status['last_applied']['name']}")

    elif args.command == "demo":
        print("Running migration demo...\n")

        # Create demo migration
        class DemoMigration(Migration):
            version = "001"
            name = "demo_migration"
            description = "Demo migration for testing"

            def up(self, context: MigrationContext) -> None:
                context.log("Applying demo migration")
                # Demo: Create a test file
                context.put_file("demo_data.json", {
                    "migrated": True,
                    "version": self.version,
                    "timestamp": time.time(),
                })

            def down(self, context: MigrationContext) -> None:
                context.log("Rolling back demo migration")
                context.delete_file("demo_data.json")

        manager.register(DemoMigration())

        print("Status before migration:")
        print(json.dumps(manager.status(), indent=2))

        # Create temp data dir
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"\nRunning migration (dry_run=False)...")
            result = manager.migrate(data_dir=temp_dir)
            print(f"Migration result: {result.to_dict()}")

            # Check created file
            demo_file = Path(temp_dir) / "demo_data.json"
            if demo_file.exists():
                print(f"\nCreated file contents: {demo_file.read_text()}")

            print("\nStatus after migration:")
            print(json.dumps(manager.status(), indent=2))

        # Demo data transformer
        print("\n--- Data Transformer Demo ---")
        transformer = DataTransformer()
        transformer.add_field_rename("old_field", "new_field")
        transformer.add_type_cast("count", int, default=0)
        transformer.add_default("created_at", lambda: time.time())

        old_data = {
            "old_field": "value",
            "count": "42",
            "other": "preserved",
        }

        new_data = transformer.transform(old_data)
        print(f"Original: {old_data}")
        print(f"Transformed: {new_data}")

        print("\nDemo complete.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
