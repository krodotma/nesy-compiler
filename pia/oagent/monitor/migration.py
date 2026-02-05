#!/usr/bin/env python3
"""
Monitor Migration Tools - Step 295

Data migration utilities for the Monitor Agent.

PBTSO Phase: PLAN

Bus Topics:
- monitor.migration.started (emitted)
- monitor.migration.progress (emitted)
- monitor.migration.completed (emitted)
- monitor.migration.failed (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional


class MigrationStatus(Enum):
    """Migration status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class MigrationType(Enum):
    """Migration types."""
    SCHEMA = "schema"
    DATA = "data"
    CONFIG = "config"
    FULL = "full"


@dataclass
class MigrationStep:
    """A migration step.

    Attributes:
        name: Step name
        description: Step description
        up_func: Forward migration function
        down_func: Rollback function
        order: Execution order
    """
    name: str
    description: str
    up_func: Callable[[], Coroutine[Any, Any, bool]]
    down_func: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None
    order: int = 0


@dataclass
class Migration:
    """A migration definition.

    Attributes:
        migration_id: Unique migration ID
        version: Target version
        description: Migration description
        migration_type: Type of migration
        steps: Migration steps
        created_at: Creation timestamp
    """
    migration_id: str
    version: str
    description: str
    migration_type: MigrationType
    steps: List[MigrationStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "migration_id": self.migration_id,
            "version": self.version,
            "description": self.description,
            "type": self.migration_type.value,
            "steps": len(self.steps),
            "created_at": self.created_at,
        }


@dataclass
class MigrationResult:
    """Result of a migration.

    Attributes:
        migration_id: Migration ID
        status: Migration status
        version_from: Starting version
        version_to: Target version
        steps_completed: Number of completed steps
        steps_total: Total steps
        duration_ms: Migration duration
        error: Error message if failed
        started_at: Start timestamp
        completed_at: Completion timestamp
    """
    migration_id: str
    status: MigrationStatus
    version_from: str
    version_to: str
    steps_completed: int = 0
    steps_total: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "migration_id": self.migration_id,
            "status": self.status.value,
            "version_from": self.version_from,
            "version_to": self.version_to,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class MonitorMigrationTools:
    """
    Migration tools for the Monitor Agent.

    Provides:
    - Schema migrations
    - Data migrations
    - Configuration migrations
    - Rollback support
    - Migration history

    Example:
        migration = MonitorMigrationTools()

        # Create migration
        mig = migration.create_migration(
            version="0.4.0",
            description="Add new metric types",
        )

        # Add step
        migration.add_step(mig.migration_id, MigrationStep(
            name="add_histogram_table",
            description="Create histogram metrics table",
            up_func=create_histogram_table,
            down_func=drop_histogram_table,
        ))

        # Run migration
        result = await migration.run_migration(mig.migration_id)
    """

    BUS_TOPICS = {
        "started": "monitor.migration.started",
        "progress": "monitor.migration.progress",
        "completed": "monitor.migration.completed",
        "failed": "monitor.migration.failed",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        data_dir: Optional[str] = None,
        bus_dir: Optional[str] = None,
    ):
        """Initialize migration tools.

        Args:
            data_dir: Data directory
            bus_dir: Bus directory
        """
        self._last_heartbeat = time.time()

        # Data directory
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._data_dir = data_dir or os.path.join(pluribus_root, ".pluribus", "monitor")
        self._migrations_dir = os.path.join(self._data_dir, "migrations")
        Path(self._migrations_dir).mkdir(parents=True, exist_ok=True)

        # Bus path
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Migration registry
        self._migrations: Dict[str, Migration] = {}
        self._history: List[MigrationResult] = []
        self._current_version = "0.3.0"

        # Load history
        self._load_history()

    def create_migration(
        self,
        version: str,
        description: str,
        migration_type: MigrationType = MigrationType.DATA,
    ) -> Migration:
        """Create a new migration.

        Args:
            version: Target version
            description: Migration description
            migration_type: Type of migration

        Returns:
            Created migration
        """
        migration_id = f"mig-{version.replace('.', '')}-{uuid.uuid4().hex[:8]}"

        migration = Migration(
            migration_id=migration_id,
            version=version,
            description=description,
            migration_type=migration_type,
        )

        self._migrations[migration_id] = migration
        return migration

    def add_step(
        self,
        migration_id: str,
        step: MigrationStep,
    ) -> bool:
        """Add a step to a migration.

        Args:
            migration_id: Migration ID
            step: Step to add

        Returns:
            True if added
        """
        migration = self._migrations.get(migration_id)
        if not migration:
            return False

        # Set order if not specified
        if step.order == 0:
            step.order = len(migration.steps) + 1

        migration.steps.append(step)
        migration.steps.sort(key=lambda s: s.order)

        return True

    async def run_migration(
        self,
        migration_id: str,
        dry_run: bool = False,
    ) -> MigrationResult:
        """Run a migration.

        Args:
            migration_id: Migration ID
            dry_run: Only simulate

        Returns:
            Migration result
        """
        migration = self._migrations.get(migration_id)
        if not migration:
            return MigrationResult(
                migration_id=migration_id,
                status=MigrationStatus.FAILED,
                version_from=self._current_version,
                version_to="unknown",
                error="Migration not found",
            )

        result = MigrationResult(
            migration_id=migration_id,
            status=MigrationStatus.RUNNING,
            version_from=self._current_version,
            version_to=migration.version,
            steps_total=len(migration.steps),
        )

        self._emit_bus_event(
            self.BUS_TOPICS["started"],
            {
                "migration_id": migration_id,
                "version_from": result.version_from,
                "version_to": result.version_to,
                "dry_run": dry_run,
            },
        )

        completed_steps: List[MigrationStep] = []
        start_time = time.time()

        try:
            for i, step in enumerate(migration.steps):
                self._emit_bus_event(
                    self.BUS_TOPICS["progress"],
                    {
                        "migration_id": migration_id,
                        "step": step.name,
                        "progress": (i / len(migration.steps)) * 100,
                    },
                )

                if not dry_run:
                    success = await step.up_func()
                    if not success:
                        raise Exception(f"Step '{step.name}' failed")

                completed_steps.append(step)
                result.steps_completed += 1

            # Migration successful
            result.status = MigrationStatus.COMPLETED
            result.completed_at = time.time()
            result.duration_ms = (result.completed_at - start_time) * 1000

            if not dry_run:
                self._current_version = migration.version
                self._save_history(result)

            self._emit_bus_event(
                self.BUS_TOPICS["completed"],
                {
                    "migration_id": migration_id,
                    "version": migration.version,
                    "duration_ms": result.duration_ms,
                },
            )

        except Exception as e:
            result.status = MigrationStatus.FAILED
            result.error = str(e)
            result.completed_at = time.time()
            result.duration_ms = (result.completed_at - start_time) * 1000

            self._emit_bus_event(
                self.BUS_TOPICS["failed"],
                {
                    "migration_id": migration_id,
                    "error": str(e),
                    "steps_completed": result.steps_completed,
                },
                level="error",
            )

            # Attempt rollback
            if not dry_run:
                await self._rollback_steps(completed_steps)

        self._history.append(result)
        return result

    async def rollback_migration(
        self,
        migration_id: str,
    ) -> MigrationResult:
        """Rollback a migration.

        Args:
            migration_id: Migration ID

        Returns:
            Rollback result
        """
        migration = self._migrations.get(migration_id)
        if not migration:
            return MigrationResult(
                migration_id=migration_id,
                status=MigrationStatus.FAILED,
                version_from=self._current_version,
                version_to="unknown",
                error="Migration not found",
            )

        result = MigrationResult(
            migration_id=f"{migration_id}-rollback",
            status=MigrationStatus.RUNNING,
            version_from=migration.version,
            version_to=self._current_version,
            steps_total=len(migration.steps),
        )

        try:
            await self._rollback_steps(list(reversed(migration.steps)))
            result.status = MigrationStatus.ROLLED_BACK
            result.steps_completed = len(migration.steps)
        except Exception as e:
            result.status = MigrationStatus.FAILED
            result.error = str(e)

        result.completed_at = time.time()
        result.duration_ms = (result.completed_at - result.started_at) * 1000

        self._history.append(result)
        return result

    async def _rollback_steps(self, steps: List[MigrationStep]) -> None:
        """Rollback completed steps.

        Args:
            steps: Steps to rollback
        """
        for step in reversed(steps):
            if step.down_func:
                try:
                    await step.down_func()
                except Exception:
                    pass  # Best effort rollback

    def export_data(
        self,
        output_path: str,
        include_metrics: bool = True,
        include_alerts: bool = True,
        include_config: bool = True,
    ) -> bool:
        """Export data for migration.

        Args:
            output_path: Output path
            include_metrics: Include metrics
            include_alerts: Include alerts
            include_config: Include config

        Returns:
            True if exported
        """
        export_data: Dict[str, Any] = {
            "version": self._current_version,
            "exported_at": time.time(),
            "components": [],
        }

        if include_metrics:
            export_data["components"].append("metrics")
            export_data["metrics"] = self._export_component("metrics")

        if include_alerts:
            export_data["components"].append("alerts")
            export_data["alerts"] = self._export_component("alerts")

        if include_config:
            export_data["components"].append("config")
            export_data["config"] = self._export_component("config")

        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)
            return True
        except Exception:
            return False

    def import_data(
        self,
        input_path: str,
        overwrite: bool = False,
    ) -> bool:
        """Import data from export.

        Args:
            input_path: Input path
            overwrite: Overwrite existing data

        Returns:
            True if imported
        """
        try:
            with open(input_path, "r") as f:
                import_data = json.load(f)

            for component in import_data.get("components", []):
                data = import_data.get(component, {})
                self._import_component(component, data, overwrite)

            return True
        except Exception:
            return False

    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a backup before migration.

        Args:
            backup_name: Backup name

        Returns:
            Backup path
        """
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"

        backup_path = os.path.join(self._data_dir, "backups", backup_name)
        Path(backup_path).mkdir(parents=True, exist_ok=True)

        # Copy data files
        for item in ["config", "metrics", "alerts"]:
            src = os.path.join(self._data_dir, item)
            if os.path.exists(src):
                dst = os.path.join(backup_path, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

        return backup_path

    def restore_backup(self, backup_path: str) -> bool:
        """Restore from backup.

        Args:
            backup_path: Backup path

        Returns:
            True if restored
        """
        if not os.path.exists(backup_path):
            return False

        try:
            for item in os.listdir(backup_path):
                src = os.path.join(backup_path, item)
                dst = os.path.join(self._data_dir, item)

                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

            return True
        except Exception:
            return False

    def list_migrations(self) -> List[Dict[str, Any]]:
        """List all migrations.

        Returns:
            Migration info list
        """
        return [m.to_dict() for m in self._migrations.values()]

    def get_history(
        self,
        limit: int = 20,
        status_filter: Optional[MigrationStatus] = None,
    ) -> List[Dict[str, Any]]:
        """Get migration history.

        Args:
            limit: Maximum results
            status_filter: Filter by status

        Returns:
            History list
        """
        history = self._history
        if status_filter:
            history = [h for h in history if h.status == status_filter]
        return [h.to_dict() for h in reversed(history[-limit:])]

    def get_current_version(self) -> str:
        """Get current version.

        Returns:
            Current version
        """
        return self._current_version

    def get_pending_migrations(self) -> List[Dict[str, Any]]:
        """Get pending migrations.

        Returns:
            Pending migration list
        """
        return [
            m.to_dict()
            for m in self._migrations.values()
            if m.version > self._current_version
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get migration statistics.

        Returns:
            Statistics
        """
        return {
            "current_version": self._current_version,
            "total_migrations": len(self._migrations),
            "history_count": len(self._history),
            "by_status": {
                s.value: sum(1 for h in self._history if h.status == s)
                for s in MigrationStatus
            },
        }

    def _export_component(self, component: str) -> Dict[str, Any]:
        """Export a component's data."""
        # Placeholder - would read actual data files
        return {"component": component, "exported": True}

    def _import_component(
        self,
        component: str,
        data: Dict[str, Any],
        overwrite: bool,
    ) -> None:
        """Import a component's data."""
        # Placeholder - would write actual data files
        pass

    def _load_history(self) -> None:
        """Load migration history."""
        history_path = os.path.join(self._migrations_dir, "history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    data = json.load(f)
                    self._current_version = data.get("version", self._current_version)
            except Exception:
                pass

    def _save_history(self, result: MigrationResult) -> None:
        """Save migration result to history."""
        history_path = os.path.join(self._migrations_dir, "history.json")
        try:
            data = {
                "version": self._current_version,
                "last_migration": result.to_dict(),
                "updated_at": time.time(),
            }
            with open(history_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

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
                "component": "monitor_migration",
                "status": "healthy",
                "version": self._current_version,
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
            "actor": "monitor-migration",
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
_migration: Optional[MonitorMigrationTools] = None


def get_migration() -> MonitorMigrationTools:
    """Get or create the migration tools singleton.

    Returns:
        MonitorMigrationTools instance
    """
    global _migration
    if _migration is None:
        _migration = MonitorMigrationTools()
    return _migration


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Migration Tools (Step 295)")
    parser.add_argument("--list", action="store_true", help="List migrations")
    parser.add_argument("--history", action="store_true", help="Show migration history")
    parser.add_argument("--pending", action="store_true", help="Show pending migrations")
    parser.add_argument("--version", action="store_true", help="Show current version")
    parser.add_argument("--backup", metavar="NAME", help="Create backup")
    parser.add_argument("--export", metavar="PATH", help="Export data")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    migration = get_migration()

    if args.list:
        migrations = migration.list_migrations()
        if args.json:
            print(json.dumps(migrations, indent=2))
        else:
            print("Migrations:")
            for m in migrations:
                print(f"  {m['migration_id']}: {m['version']} - {m['description']}")

    if args.history:
        history = migration.get_history()
        if args.json:
            print(json.dumps(history, indent=2))
        else:
            print("Migration History:")
            for h in history:
                print(f"  {h['migration_id']}: {h['status']} ({h['version_from']} -> {h['version_to']})")

    if args.pending:
        pending = migration.get_pending_migrations()
        if args.json:
            print(json.dumps(pending, indent=2))
        else:
            print("Pending Migrations:")
            for p in pending:
                print(f"  {p['migration_id']}: {p['version']}")

    if args.version:
        version = migration.get_current_version()
        if args.json:
            print(json.dumps({"version": version}))
        else:
            print(f"Current Version: {version}")

    if args.backup:
        path = migration.create_backup(args.backup)
        if args.json:
            print(json.dumps({"backup_path": path}))
        else:
            print(f"Backup created: {path}")

    if args.export:
        success = migration.export_data(args.export)
        if args.json:
            print(json.dumps({"exported": success, "path": args.export}))
        else:
            print(f"Export: {'success' if success else 'failed'}")

    if args.stats:
        stats = migration.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Migration Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
