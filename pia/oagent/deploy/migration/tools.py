#!/usr/bin/env python3
"""
tools.py - Migration Tools (Step 245)

PBTSO Phase: TRANSFORM
A2A Integration: Data migration via deploy.migration.execute

Provides:
- MigrationStatus: Migration status enum
- Migration: Migration definition
- MigrationPlan: Migration plan with steps
- MigrationResult: Migration execution result
- DataTransformer: Data transformation utilities
- MigrationTools: Complete migration system

Bus Topics:
- deploy.migration.execute
- deploy.migration.rollback
- deploy.migration.status
- deploy.migration.complete

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


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
    actor: str = "migration-tools"
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

class MigrationStatus(Enum):
    """Migration status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    PAUSED = "paused"


class MigrationType(Enum):
    """Types of migrations."""
    SCHEMA = "schema"
    DATA = "data"
    CONFIG = "config"
    FILE = "file"
    DATABASE = "database"
    STATE = "state"


@dataclass
class MigrationStep:
    """
    Single migration step.

    Attributes:
        step_id: Unique step identifier
        name: Step name
        description: Step description
        up_fn: Forward migration function
        down_fn: Rollback function
        order: Execution order
        checksum: Content checksum
        timeout_s: Step timeout
    """
    step_id: str
    name: str
    description: str = ""
    up_fn: Optional[Callable] = None
    down_fn: Optional[Callable] = None
    order: int = 0
    checksum: str = ""
    timeout_s: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "order": self.order,
            "checksum": self.checksum,
            "timeout_s": self.timeout_s,
        }


@dataclass
class Migration:
    """
    Migration definition.

    Attributes:
        migration_id: Unique migration identifier
        name: Migration name
        version: Migration version
        migration_type: Type of migration
        description: Migration description
        steps: Migration steps
        dependencies: Required migrations
        created_at: Creation timestamp
        metadata: Additional metadata
    """
    migration_id: str
    name: str
    version: str
    migration_type: MigrationType = MigrationType.DATA
    description: str = ""
    steps: List[MigrationStep] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "migration_id": self.migration_id,
            "name": self.name,
            "version": self.version,
            "migration_type": self.migration_type.value,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "dependencies": self.dependencies,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class MigrationPlan:
    """
    Migration execution plan.

    Attributes:
        plan_id: Unique plan identifier
        migrations: Ordered list of migrations
        dry_run: Whether this is a dry run
        batch_size: Batch size for data migrations
        parallel: Run migrations in parallel
        stop_on_error: Stop on first error
        created_at: Plan creation timestamp
    """
    plan_id: str
    migrations: List[str] = field(default_factory=list)
    dry_run: bool = False
    batch_size: int = 1000
    parallel: bool = False
    stop_on_error: bool = True
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepResult:
    """Result of a migration step."""
    step_id: str
    status: MigrationStatus
    duration_ms: float
    records_affected: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "records_affected": self.records_affected,
            "error": self.error,
        }


@dataclass
class MigrationResult:
    """
    Migration execution result.

    Attributes:
        migration_id: Migration identifier
        status: Final status
        step_results: Individual step results
        duration_ms: Total duration
        records_affected: Total records affected
        started_at: Start timestamp
        completed_at: Completion timestamp
        error: Error message if failed
    """
    migration_id: str
    status: MigrationStatus = MigrationStatus.PENDING
    step_results: List[StepResult] = field(default_factory=list)
    duration_ms: float = 0.0
    records_affected: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "migration_id": self.migration_id,
            "status": self.status.value,
            "step_results": [s.to_dict() for s in self.step_results],
            "duration_ms": self.duration_ms,
            "records_affected": self.records_affected,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# ==============================================================================
# Data Transformer
# ==============================================================================

class DataTransformer:
    """
    Data transformation utilities for migrations.

    Provides common data transformation functions.
    """

    @staticmethod
    def rename_field(data: Dict, old_name: str, new_name: str) -> Dict:
        """Rename a field in a dictionary."""
        if old_name in data:
            data[new_name] = data.pop(old_name)
        return data

    @staticmethod
    def add_field(data: Dict, field_name: str, value: Any) -> Dict:
        """Add a field to a dictionary."""
        data[field_name] = value
        return data

    @staticmethod
    def remove_field(data: Dict, field_name: str) -> Dict:
        """Remove a field from a dictionary."""
        data.pop(field_name, None)
        return data

    @staticmethod
    def transform_field(
        data: Dict,
        field_name: str,
        transform_fn: Callable[[Any], Any],
    ) -> Dict:
        """Transform a field value."""
        if field_name in data:
            data[field_name] = transform_fn(data[field_name])
        return data

    @staticmethod
    def copy_field(data: Dict, from_field: str, to_field: str) -> Dict:
        """Copy a field to a new name."""
        if from_field in data:
            data[to_field] = data[from_field]
        return data

    @staticmethod
    def merge_fields(
        data: Dict,
        fields: List[str],
        target_field: str,
        separator: str = " ",
    ) -> Dict:
        """Merge multiple fields into one."""
        values = [str(data.get(f, "")) for f in fields]
        data[target_field] = separator.join(v for v in values if v)
        return data

    @staticmethod
    def split_field(
        data: Dict,
        field_name: str,
        target_fields: List[str],
        separator: str = " ",
    ) -> Dict:
        """Split a field into multiple fields."""
        if field_name in data:
            parts = str(data[field_name]).split(separator)
            for i, target in enumerate(target_fields):
                data[target] = parts[i] if i < len(parts) else ""
        return data

    @staticmethod
    def convert_type(data: Dict, field_name: str, type_fn: Callable) -> Dict:
        """Convert field type."""
        if field_name in data:
            try:
                data[field_name] = type_fn(data[field_name])
            except (ValueError, TypeError):
                pass
        return data


# ==============================================================================
# Migration Tools (Step 245)
# ==============================================================================

class MigrationTools:
    """
    Migration Tools - Data migration utilities for deployments.

    PBTSO Phase: TRANSFORM

    Responsibilities:
    - Define and manage migrations
    - Execute migrations with rollback support
    - Track migration history
    - Validate migration dependencies

    Example:
        >>> tools = MigrationTools()
        >>> migration = tools.create_migration(
        ...     name="add-user-email",
        ...     version="1.0.0",
        ...     migration_type=MigrationType.DATA
        ... )
        >>> async def up(ctx):
        ...     # Add email field
        ...     pass
        >>> tools.add_step(migration.migration_id, "add-email", up_fn=up)
        >>> result = await tools.execute(migration.migration_id)
    """

    BUS_TOPICS = {
        "execute": "deploy.migration.execute",
        "rollback": "deploy.migration.rollback",
        "status": "deploy.migration.status",
        "complete": "deploy.migration.complete",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "migration-tools",
    ):
        """
        Initialize migration tools.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "migrations"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Storage
        self._migrations: Dict[str, Migration] = {}
        self._history: Dict[str, MigrationResult] = {}
        self._applied: List[str] = []  # Applied migration IDs

        # Transformer
        self.transformer = DataTransformer()

        self._load_state()

    def create_migration(
        self,
        name: str,
        version: str,
        migration_type: MigrationType = MigrationType.DATA,
        description: str = "",
        dependencies: Optional[List[str]] = None,
    ) -> Migration:
        """
        Create a new migration.

        Args:
            name: Migration name
            version: Migration version
            migration_type: Type of migration
            description: Migration description
            dependencies: Required migration IDs

        Returns:
            Created Migration
        """
        migration_id = f"migration-{uuid.uuid4().hex[:12]}"

        migration = Migration(
            migration_id=migration_id,
            name=name,
            version=version,
            migration_type=migration_type,
            description=description,
            dependencies=dependencies or [],
        )

        self._migrations[migration_id] = migration
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["status"],
            {
                "action": "created",
                "migration_id": migration_id,
                "name": name,
                "version": version,
            },
            actor=self.actor_id,
        )

        return migration

    def add_step(
        self,
        migration_id: str,
        name: str,
        up_fn: Callable,
        down_fn: Optional[Callable] = None,
        description: str = "",
        timeout_s: int = 300,
    ) -> Optional[MigrationStep]:
        """
        Add a step to a migration.

        Args:
            migration_id: Migration ID
            name: Step name
            up_fn: Forward migration function
            down_fn: Rollback function
            description: Step description
            timeout_s: Step timeout

        Returns:
            Created MigrationStep or None
        """
        migration = self._migrations.get(migration_id)
        if not migration:
            return None

        step_id = f"step-{uuid.uuid4().hex[:8]}"
        order = len(migration.steps)

        step = MigrationStep(
            step_id=step_id,
            name=name,
            description=description,
            up_fn=up_fn,
            down_fn=down_fn,
            order=order,
            timeout_s=timeout_s,
        )

        migration.steps.append(step)
        self._save_state()

        return step

    def create_plan(
        self,
        migration_ids: List[str],
        dry_run: bool = False,
        batch_size: int = 1000,
        stop_on_error: bool = True,
    ) -> MigrationPlan:
        """
        Create a migration execution plan.

        Args:
            migration_ids: Migrations to include
            dry_run: Whether to perform dry run
            batch_size: Batch size for data migrations
            stop_on_error: Stop on first error

        Returns:
            MigrationPlan
        """
        plan_id = f"plan-{uuid.uuid4().hex[:12]}"

        # Resolve order based on dependencies
        ordered = self._resolve_order(migration_ids)

        return MigrationPlan(
            plan_id=plan_id,
            migrations=ordered,
            dry_run=dry_run,
            batch_size=batch_size,
            stop_on_error=stop_on_error,
        )

    def _resolve_order(self, migration_ids: List[str]) -> List[str]:
        """Resolve execution order based on dependencies."""
        # Simple topological sort
        ordered = []
        visited = set()

        def visit(mid: str):
            if mid in visited:
                return
            visited.add(mid)

            migration = self._migrations.get(mid)
            if migration:
                for dep in migration.dependencies:
                    if dep in migration_ids:
                        visit(dep)
            ordered.append(mid)

        for mid in migration_ids:
            visit(mid)

        return ordered

    async def execute(
        self,
        migration_id: str,
        dry_run: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> MigrationResult:
        """
        Execute a single migration.

        Args:
            migration_id: Migration ID to execute
            dry_run: Whether to perform dry run
            context: Execution context

        Returns:
            MigrationResult
        """
        migration = self._migrations.get(migration_id)
        if not migration:
            return MigrationResult(
                migration_id=migration_id,
                status=MigrationStatus.FAILED,
                error="Migration not found",
            )

        # Check if already applied
        if migration_id in self._applied:
            return MigrationResult(
                migration_id=migration_id,
                status=MigrationStatus.COMPLETED,
                error="Already applied",
            )

        # Check dependencies
        for dep in migration.dependencies:
            if dep not in self._applied:
                return MigrationResult(
                    migration_id=migration_id,
                    status=MigrationStatus.FAILED,
                    error=f"Dependency not satisfied: {dep}",
                )

        result = MigrationResult(
            migration_id=migration_id,
            started_at=time.time(),
        )
        result.status = MigrationStatus.RUNNING

        _emit_bus_event(
            self.BUS_TOPICS["execute"],
            {
                "migration_id": migration_id,
                "name": migration.name,
                "dry_run": dry_run,
            },
            actor=self.actor_id,
        )

        ctx = context or {}
        ctx["dry_run"] = dry_run

        try:
            # Execute steps in order
            for step in sorted(migration.steps, key=lambda s: s.order):
                step_result = await self._execute_step(step, ctx)
                result.step_results.append(step_result)
                result.records_affected += step_result.records_affected

                if step_result.status == MigrationStatus.FAILED:
                    result.status = MigrationStatus.FAILED
                    result.error = step_result.error
                    break

            if result.status != MigrationStatus.FAILED:
                result.status = MigrationStatus.COMPLETED
                if not dry_run:
                    self._applied.append(migration_id)

        except Exception as e:
            result.status = MigrationStatus.FAILED
            result.error = str(e)

        result.completed_at = time.time()
        result.duration_ms = (result.completed_at - result.started_at) * 1000

        self._history[migration_id] = result
        self._save_state()

        topic = self.BUS_TOPICS["complete"] if result.status == MigrationStatus.COMPLETED else self.BUS_TOPICS["status"]
        _emit_bus_event(
            topic,
            {
                "migration_id": migration_id,
                "status": result.status.value,
                "duration_ms": result.duration_ms,
                "records_affected": result.records_affected,
            },
            level="info" if result.status == MigrationStatus.COMPLETED else "error",
            actor=self.actor_id,
        )

        return result

    async def _execute_step(
        self,
        step: MigrationStep,
        ctx: Dict[str, Any],
    ) -> StepResult:
        """Execute a single migration step."""
        start_time = time.time()

        try:
            if step.up_fn:
                if asyncio.iscoroutinefunction(step.up_fn):
                    records = await asyncio.wait_for(
                        step.up_fn(ctx),
                        timeout=step.timeout_s
                    )
                else:
                    records = step.up_fn(ctx)

                records_affected = records if isinstance(records, int) else 0

                return StepResult(
                    step_id=step.step_id,
                    status=MigrationStatus.COMPLETED,
                    duration_ms=(time.time() - start_time) * 1000,
                    records_affected=records_affected,
                )
            else:
                return StepResult(
                    step_id=step.step_id,
                    status=MigrationStatus.COMPLETED,
                    duration_ms=(time.time() - start_time) * 1000,
                )

        except asyncio.TimeoutError:
            return StepResult(
                step_id=step.step_id,
                status=MigrationStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                error=f"Step timed out after {step.timeout_s}s",
            )

        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                status=MigrationStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def rollback(
        self,
        migration_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> MigrationResult:
        """
        Rollback a migration.

        Args:
            migration_id: Migration ID to rollback
            context: Execution context

        Returns:
            MigrationResult
        """
        migration = self._migrations.get(migration_id)
        if not migration:
            return MigrationResult(
                migration_id=migration_id,
                status=MigrationStatus.FAILED,
                error="Migration not found",
            )

        if migration_id not in self._applied:
            return MigrationResult(
                migration_id=migration_id,
                status=MigrationStatus.FAILED,
                error="Migration not applied",
            )

        result = MigrationResult(
            migration_id=migration_id,
            started_at=time.time(),
        )
        result.status = MigrationStatus.RUNNING

        _emit_bus_event(
            self.BUS_TOPICS["rollback"],
            {
                "migration_id": migration_id,
                "name": migration.name,
            },
            actor=self.actor_id,
        )

        ctx = context or {}

        try:
            # Execute rollback steps in reverse order
            for step in sorted(migration.steps, key=lambda s: s.order, reverse=True):
                if step.down_fn:
                    step_result = await self._execute_rollback_step(step, ctx)
                    result.step_results.append(step_result)
                    result.records_affected += step_result.records_affected

                    if step_result.status == MigrationStatus.FAILED:
                        result.status = MigrationStatus.FAILED
                        result.error = step_result.error
                        break

            if result.status != MigrationStatus.FAILED:
                result.status = MigrationStatus.ROLLED_BACK
                self._applied.remove(migration_id)

        except Exception as e:
            result.status = MigrationStatus.FAILED
            result.error = str(e)

        result.completed_at = time.time()
        result.duration_ms = (result.completed_at - result.started_at) * 1000

        self._history[f"{migration_id}_rollback"] = result
        self._save_state()

        return result

    async def _execute_rollback_step(
        self,
        step: MigrationStep,
        ctx: Dict[str, Any],
    ) -> StepResult:
        """Execute a rollback step."""
        start_time = time.time()

        try:
            if step.down_fn:
                if asyncio.iscoroutinefunction(step.down_fn):
                    records = await asyncio.wait_for(
                        step.down_fn(ctx),
                        timeout=step.timeout_s
                    )
                else:
                    records = step.down_fn(ctx)

                records_affected = records if isinstance(records, int) else 0

                return StepResult(
                    step_id=step.step_id,
                    status=MigrationStatus.ROLLED_BACK,
                    duration_ms=(time.time() - start_time) * 1000,
                    records_affected=records_affected,
                )

        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                status=MigrationStatus.FAILED,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

        return StepResult(
            step_id=step.step_id,
            status=MigrationStatus.ROLLED_BACK,
            duration_ms=(time.time() - start_time) * 1000,
        )

    async def execute_plan(self, plan: MigrationPlan) -> List[MigrationResult]:
        """
        Execute a migration plan.

        Args:
            plan: Migration plan to execute

        Returns:
            List of MigrationResults
        """
        results = []

        for migration_id in plan.migrations:
            result = await self.execute(
                migration_id,
                dry_run=plan.dry_run,
            )
            results.append(result)

            if plan.stop_on_error and result.status == MigrationStatus.FAILED:
                break

        return results

    def get_migration(self, migration_id: str) -> Optional[Migration]:
        """Get a migration by ID."""
        return self._migrations.get(migration_id)

    def list_migrations(
        self,
        applied_only: bool = False,
        pending_only: bool = False,
    ) -> List[Migration]:
        """List migrations."""
        migrations = list(self._migrations.values())

        if applied_only:
            migrations = [m for m in migrations if m.migration_id in self._applied]
        elif pending_only:
            migrations = [m for m in migrations if m.migration_id not in self._applied]

        return migrations

    def get_history(self, migration_id: str) -> Optional[MigrationResult]:
        """Get migration history."""
        return self._history.get(migration_id)

    def get_applied(self) -> List[str]:
        """Get list of applied migration IDs."""
        return list(self._applied)

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "migrations": {k: v.to_dict() for k, v in self._migrations.items()},
            "history": {k: v.to_dict() for k, v in self._history.items()},
            "applied": self._applied,
        }
        state_file = self.state_dir / "migration_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "migration_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            for k, v in state.get("migrations", {}).items():
                self._migrations[k] = Migration(
                    migration_id=v["migration_id"],
                    name=v["name"],
                    version=v["version"],
                    migration_type=MigrationType(v.get("migration_type", "data")),
                    description=v.get("description", ""),
                    dependencies=v.get("dependencies", []),
                )

            self._applied = state.get("applied", [])

        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for migration tools."""
    import argparse

    parser = argparse.ArgumentParser(description="Migration Tools (Step 245)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a migration")
    create_parser.add_argument("name", help="Migration name")
    create_parser.add_argument("--version", "-v", required=True, help="Version")
    create_parser.add_argument("--type", "-t", default="data",
                              choices=["schema", "data", "config", "file"])
    create_parser.add_argument("--description", "-d", default="", help="Description")
    create_parser.add_argument("--json", action="store_true", help="JSON output")

    # execute command
    exec_parser = subparsers.add_parser("execute", help="Execute a migration")
    exec_parser.add_argument("migration_id", help="Migration ID")
    exec_parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    exec_parser.add_argument("--json", action="store_true", help="JSON output")

    # rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback a migration")
    rollback_parser.add_argument("migration_id", help="Migration ID")
    rollback_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List migrations")
    list_parser.add_argument("--applied", "-a", action="store_true", help="Applied only")
    list_parser.add_argument("--pending", "-p", action="store_true", help="Pending only")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # status command
    status_parser = subparsers.add_parser("status", help="Migration status")
    status_parser.add_argument("migration_id", help="Migration ID")
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    tools = MigrationTools()

    if args.command == "create":
        migration = tools.create_migration(
            name=args.name,
            version=args.version,
            migration_type=MigrationType(args.type),
            description=args.description,
        )

        if args.json:
            print(json.dumps(migration.to_dict(), indent=2))
        else:
            print(f"Created migration: {migration.migration_id}")
            print(f"  Name: {migration.name}")
            print(f"  Version: {migration.version}")

        return 0

    elif args.command == "execute":
        result = asyncio.get_event_loop().run_until_complete(
            tools.execute(args.migration_id, dry_run=args.dry_run)
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "OK" if result.status == MigrationStatus.COMPLETED else "FAIL"
            print(f"[{status}] {args.migration_id}")
            print(f"  Status: {result.status.value}")
            print(f"  Duration: {result.duration_ms:.1f}ms")
            print(f"  Records: {result.records_affected}")
            if result.error:
                print(f"  Error: {result.error}")

        return 0 if result.status == MigrationStatus.COMPLETED else 1

    elif args.command == "rollback":
        result = asyncio.get_event_loop().run_until_complete(
            tools.rollback(args.migration_id)
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "OK" if result.status == MigrationStatus.ROLLED_BACK else "FAIL"
            print(f"[{status}] Rollback {args.migration_id}")
            print(f"  Status: {result.status.value}")
            if result.error:
                print(f"  Error: {result.error}")

        return 0 if result.status == MigrationStatus.ROLLED_BACK else 1

    elif args.command == "list":
        migrations = tools.list_migrations(
            applied_only=args.applied,
            pending_only=args.pending,
        )

        if args.json:
            print(json.dumps([m.to_dict() for m in migrations], indent=2))
        else:
            applied = tools.get_applied()
            if not migrations:
                print("No migrations found")
            else:
                for m in migrations:
                    status = "[APPLIED]" if m.migration_id in applied else "[PENDING]"
                    print(f"{status} {m.migration_id}: {m.name} (v{m.version})")

        return 0

    elif args.command == "status":
        migration = tools.get_migration(args.migration_id)
        history = tools.get_history(args.migration_id)

        if not migration:
            print(f"Migration not found: {args.migration_id}")
            return 1

        if args.json:
            output = migration.to_dict()
            output["history"] = history.to_dict() if history else None
            print(json.dumps(output, indent=2))
        else:
            applied = args.migration_id in tools.get_applied()
            print(f"Migration: {migration.migration_id}")
            print(f"  Name: {migration.name}")
            print(f"  Version: {migration.version}")
            print(f"  Applied: {'Yes' if applied else 'No'}")
            if history:
                print(f"  Last run: {history.status.value}")
                print(f"  Duration: {history.duration_ms:.1f}ms")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
