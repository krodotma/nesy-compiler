#!/usr/bin/env python3
"""
system.py - Backup System (Step 246)

PBTSO Phase: SEQUESTER
A2A Integration: Backup/restore via deploy.backup.create

Provides:
- BackupStatus: Backup status enum
- Backup: Backup definition
- BackupSchedule: Backup scheduling
- RestoreResult: Restore operation result
- BackupPolicy: Retention and lifecycle policy
- BackupSystem: Complete backup/restore system

Bus Topics:
- deploy.backup.create
- deploy.backup.restore
- deploy.backup.delete
- deploy.backup.schedule

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import asyncio
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import socket
import tarfile
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
    actor: str = "backup-system"
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

class BackupStatus(Enum):
    """Backup status enum."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    DELETED = "deleted"


class BackupType(Enum):
    """Types of backups."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class StorageBackend(Enum):
    """Backup storage backends."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"


@dataclass
class BackupPolicy:
    """
    Backup retention and lifecycle policy.

    Attributes:
        policy_id: Unique policy identifier
        name: Policy name
        retention_days: Days to retain backups
        max_backups: Maximum backups to keep
        compress: Whether to compress backups
        encrypt: Whether to encrypt backups
        verify: Whether to verify after backup
        schedule_cron: Cron expression for scheduling
    """
    policy_id: str
    name: str
    retention_days: int = 30
    max_backups: int = 10
    compress: bool = True
    encrypt: bool = False
    verify: bool = True
    schedule_cron: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BackupSchedule:
    """
    Backup scheduling configuration.

    Attributes:
        schedule_id: Unique schedule identifier
        name: Schedule name
        source_path: Path to backup
        policy_id: Associated policy
        backup_type: Type of backup
        enabled: Whether schedule is active
        last_run: Last run timestamp
        next_run: Next scheduled run
        cron_expression: Cron expression
    """
    schedule_id: str
    name: str
    source_path: str
    policy_id: str = ""
    backup_type: BackupType = BackupType.FULL
    enabled: bool = True
    last_run: float = 0.0
    next_run: float = 0.0
    cron_expression: str = "0 2 * * *"  # Daily at 2 AM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "name": self.name,
            "source_path": self.source_path,
            "policy_id": self.policy_id,
            "backup_type": self.backup_type.value,
            "enabled": self.enabled,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "cron_expression": self.cron_expression,
        }


@dataclass
class Backup:
    """
    Backup definition.

    Attributes:
        backup_id: Unique backup identifier
        name: Backup name
        source_path: Source path
        backup_path: Stored backup path
        backup_type: Type of backup
        status: Backup status
        size_bytes: Backup size in bytes
        checksum: Backup checksum
        created_at: Creation timestamp
        expires_at: Expiration timestamp
        metadata: Additional metadata
        parent_backup_id: Parent for incremental
    """
    backup_id: str
    name: str
    source_path: str
    backup_path: str = ""
    backup_type: BackupType = BackupType.FULL
    status: BackupStatus = BackupStatus.PENDING
    size_bytes: int = 0
    checksum: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_backup_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "name": self.name,
            "source_path": self.source_path,
            "backup_path": self.backup_path,
            "backup_type": self.backup_type.value,
            "status": self.status.value,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
            "parent_backup_id": self.parent_backup_id,
        }


@dataclass
class RestoreResult:
    """
    Restore operation result.

    Attributes:
        backup_id: Restored backup ID
        status: Restore status
        target_path: Restore target path
        files_restored: Number of files restored
        duration_ms: Restore duration
        error: Error message if failed
    """
    backup_id: str
    status: BackupStatus
    target_path: str = ""
    files_restored: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "status": self.status.value,
            "target_path": self.target_path,
            "files_restored": self.files_restored,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


# ==============================================================================
# Backup System (Step 246)
# ==============================================================================

class BackupSystem:
    """
    Backup System - Backup/restore capabilities for deployments.

    PBTSO Phase: SEQUESTER

    Responsibilities:
    - Create full and incremental backups
    - Restore from backups
    - Manage backup retention
    - Schedule automated backups
    - Verify backup integrity

    Example:
        >>> system = BackupSystem()
        >>> backup = await system.create_backup(
        ...     name="config-backup",
        ...     source_path="/etc/myapp",
        ...     backup_type=BackupType.FULL
        ... )
        >>> result = await system.restore(
        ...     backup_id=backup.backup_id,
        ...     target_path="/etc/myapp.restored"
        ... )
    """

    BUS_TOPICS = {
        "create": "deploy.backup.create",
        "restore": "deploy.backup.restore",
        "delete": "deploy.backup.delete",
        "schedule": "deploy.backup.schedule",
        "verify": "deploy.backup.verify",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        backup_dir: Optional[str] = None,
        actor_id: str = "backup-system",
    ):
        """
        Initialize the backup system.

        Args:
            state_dir: Directory for state persistence
            backup_dir: Directory for storing backups
            actor_id: Actor identifier for bus events
        """
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))

        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "backup"

        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            self.backup_dir = pluribus_root / ".pluribus" / "deploy" / "backup" / "data"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Storage
        self._backups: Dict[str, Backup] = {}
        self._policies: Dict[str, BackupPolicy] = {}
        self._schedules: Dict[str, BackupSchedule] = {}

        self._load_state()

    async def create_backup(
        self,
        name: str,
        source_path: str,
        backup_type: BackupType = BackupType.FULL,
        policy_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Backup:
        """
        Create a new backup.

        Args:
            name: Backup name
            source_path: Path to backup
            backup_type: Type of backup
            policy_id: Policy to apply
            metadata: Additional metadata

        Returns:
            Created Backup
        """
        backup_id = f"backup-{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{name}_{timestamp}.tar.gz"
        backup_path = str(self.backup_dir / backup_filename)

        # Get policy settings
        policy = self._policies.get(policy_id) if policy_id else None
        retention_days = policy.retention_days if policy else 30
        compress = policy.compress if policy else True

        backup = Backup(
            backup_id=backup_id,
            name=name,
            source_path=source_path,
            backup_path=backup_path,
            backup_type=backup_type,
            status=BackupStatus.PENDING,
            expires_at=time.time() + (retention_days * 86400),
            metadata=metadata or {},
        )

        self._backups[backup_id] = backup

        _emit_bus_event(
            self.BUS_TOPICS["create"],
            {
                "backup_id": backup_id,
                "name": name,
                "source_path": source_path,
                "backup_type": backup_type.value,
            },
            actor=self.actor_id,
        )

        # Create the backup
        try:
            backup.status = BackupStatus.IN_PROGRESS
            self._save_state()

            # Create tarball
            source = Path(source_path)
            if not source.exists():
                raise FileNotFoundError(f"Source not found: {source_path}")

            mode = "w:gz" if compress else "w"
            with tarfile.open(backup_path, mode) as tar:
                if source.is_file():
                    tar.add(source, arcname=source.name)
                else:
                    for item in source.iterdir():
                        tar.add(item, arcname=item.name)

            # Calculate size and checksum
            backup.size_bytes = os.path.getsize(backup_path)
            backup.checksum = self._calculate_checksum(backup_path)
            backup.status = BackupStatus.COMPLETED

            # Verify if policy requires it
            if policy and policy.verify:
                await self.verify_backup(backup_id)

        except Exception as e:
            backup.status = BackupStatus.FAILED
            backup.metadata["error"] = str(e)

        self._save_state()

        # Apply retention policy
        if policy:
            await self._apply_retention(policy)

        return backup

    async def restore(
        self,
        backup_id: str,
        target_path: str,
        overwrite: bool = False,
    ) -> RestoreResult:
        """
        Restore from a backup.

        Args:
            backup_id: Backup ID to restore
            target_path: Target restoration path
            overwrite: Whether to overwrite existing

        Returns:
            RestoreResult
        """
        backup = self._backups.get(backup_id)
        if not backup:
            return RestoreResult(
                backup_id=backup_id,
                status=BackupStatus.FAILED,
                error="Backup not found",
            )

        if backup.status != BackupStatus.COMPLETED:
            return RestoreResult(
                backup_id=backup_id,
                status=BackupStatus.FAILED,
                error=f"Backup not in completed status: {backup.status.value}",
            )

        start_time = time.time()
        target = Path(target_path)

        _emit_bus_event(
            self.BUS_TOPICS["restore"],
            {
                "backup_id": backup_id,
                "target_path": target_path,
            },
            actor=self.actor_id,
        )

        try:
            # Check target
            if target.exists() and not overwrite:
                return RestoreResult(
                    backup_id=backup_id,
                    status=BackupStatus.FAILED,
                    error="Target exists, use overwrite=True",
                )

            # Create target directory
            target.mkdir(parents=True, exist_ok=True)

            # Extract backup
            files_restored = 0
            with tarfile.open(backup.backup_path, "r:*") as tar:
                tar.extractall(target)
                files_restored = len(tar.getmembers())

            duration_ms = (time.time() - start_time) * 1000

            return RestoreResult(
                backup_id=backup_id,
                status=BackupStatus.COMPLETED,
                target_path=target_path,
                files_restored=files_restored,
                duration_ms=duration_ms,
            )

        except Exception as e:
            return RestoreResult(
                backup_id=backup_id,
                status=BackupStatus.FAILED,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )

    async def verify_backup(self, backup_id: str) -> bool:
        """
        Verify backup integrity.

        Args:
            backup_id: Backup ID to verify

        Returns:
            True if backup is valid
        """
        backup = self._backups.get(backup_id)
        if not backup:
            return False

        if not os.path.exists(backup.backup_path):
            backup.status = BackupStatus.FAILED
            backup.metadata["verify_error"] = "Backup file missing"
            return False

        # Verify checksum
        current_checksum = self._calculate_checksum(backup.backup_path)
        if current_checksum != backup.checksum:
            backup.status = BackupStatus.FAILED
            backup.metadata["verify_error"] = "Checksum mismatch"
            return False

        # Try to read the archive
        try:
            with tarfile.open(backup.backup_path, "r:*") as tar:
                tar.getmembers()
        except Exception as e:
            backup.status = BackupStatus.FAILED
            backup.metadata["verify_error"] = str(e)
            return False

        backup.metadata["verified_at"] = time.time()
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["verify"],
            {
                "backup_id": backup_id,
                "verified": True,
            },
            actor=self.actor_id,
        )

        return True

    async def delete_backup(self, backup_id: str, reason: str = "manual") -> bool:
        """
        Delete a backup.

        Args:
            backup_id: Backup ID to delete
            reason: Deletion reason

        Returns:
            True if deleted
        """
        backup = self._backups.get(backup_id)
        if not backup:
            return False

        # Delete backup file
        if os.path.exists(backup.backup_path):
            os.remove(backup.backup_path)

        backup.status = BackupStatus.DELETED
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["delete"],
            {
                "backup_id": backup_id,
                "reason": reason,
            },
            level="warn",
            actor=self.actor_id,
        )

        return True

    def create_policy(
        self,
        name: str,
        retention_days: int = 30,
        max_backups: int = 10,
        compress: bool = True,
        verify: bool = True,
    ) -> BackupPolicy:
        """
        Create a backup policy.

        Args:
            name: Policy name
            retention_days: Days to retain
            max_backups: Maximum backups
            compress: Compress backups
            verify: Verify after backup

        Returns:
            Created BackupPolicy
        """
        policy_id = f"policy-{uuid.uuid4().hex[:8]}"

        policy = BackupPolicy(
            policy_id=policy_id,
            name=name,
            retention_days=retention_days,
            max_backups=max_backups,
            compress=compress,
            verify=verify,
        )

        self._policies[policy_id] = policy
        self._save_state()

        return policy

    def create_schedule(
        self,
        name: str,
        source_path: str,
        policy_id: str,
        cron_expression: str = "0 2 * * *",
        backup_type: BackupType = BackupType.FULL,
    ) -> BackupSchedule:
        """
        Create a backup schedule.

        Args:
            name: Schedule name
            source_path: Path to backup
            policy_id: Policy to use
            cron_expression: Cron schedule
            backup_type: Type of backup

        Returns:
            Created BackupSchedule
        """
        schedule_id = f"schedule-{uuid.uuid4().hex[:8]}"

        schedule = BackupSchedule(
            schedule_id=schedule_id,
            name=name,
            source_path=source_path,
            policy_id=policy_id,
            backup_type=backup_type,
            cron_expression=cron_expression,
        )

        self._schedules[schedule_id] = schedule
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["schedule"],
            {
                "schedule_id": schedule_id,
                "name": name,
                "cron": cron_expression,
            },
            actor=self.actor_id,
        )

        return schedule

    async def run_schedule(self, schedule_id: str) -> Optional[Backup]:
        """
        Run a scheduled backup.

        Args:
            schedule_id: Schedule ID to run

        Returns:
            Created Backup or None
        """
        schedule = self._schedules.get(schedule_id)
        if not schedule or not schedule.enabled:
            return None

        backup = await self.create_backup(
            name=schedule.name,
            source_path=schedule.source_path,
            backup_type=schedule.backup_type,
            policy_id=schedule.policy_id,
        )

        schedule.last_run = time.time()
        self._save_state()

        return backup

    async def _apply_retention(self, policy: BackupPolicy) -> None:
        """Apply retention policy to backups."""
        now = time.time()

        # Get backups for this policy
        backups = [
            b for b in self._backups.values()
            if b.status == BackupStatus.COMPLETED
        ]

        # Sort by creation time (newest first)
        backups.sort(key=lambda b: b.created_at, reverse=True)

        # Delete expired backups
        for backup in backups:
            if backup.expires_at > 0 and now > backup.expires_at:
                await self.delete_backup(backup.backup_id, "expired")

        # Enforce max backups
        if len(backups) > policy.max_backups:
            for backup in backups[policy.max_backups:]:
                await self.delete_backup(backup.backup_id, "retention_limit")

    def _calculate_checksum(self, filepath: str) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_backup(self, backup_id: str) -> Optional[Backup]:
        """Get a backup by ID."""
        return self._backups.get(backup_id)

    def list_backups(
        self,
        status: Optional[BackupStatus] = None,
        source_path: Optional[str] = None,
    ) -> List[Backup]:
        """List backups with optional filters."""
        backups = list(self._backups.values())

        if status:
            backups = [b for b in backups if b.status == status]

        if source_path:
            backups = [b for b in backups if b.source_path == source_path]

        return sorted(backups, key=lambda b: b.created_at, reverse=True)

    def get_policy(self, policy_id: str) -> Optional[BackupPolicy]:
        """Get a policy by ID."""
        return self._policies.get(policy_id)

    def list_policies(self) -> List[BackupPolicy]:
        """List all policies."""
        return list(self._policies.values())

    def get_schedule(self, schedule_id: str) -> Optional[BackupSchedule]:
        """Get a schedule by ID."""
        return self._schedules.get(schedule_id)

    def list_schedules(self) -> List[BackupSchedule]:
        """List all schedules."""
        return list(self._schedules.values())

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "backups": {k: v.to_dict() for k, v in self._backups.items()},
            "policies": {k: v.to_dict() for k, v in self._policies.items()},
            "schedules": {k: v.to_dict() for k, v in self._schedules.items()},
        }
        state_file = self.state_dir / "backup_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "backup_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            for k, v in state.get("backups", {}).items():
                self._backups[k] = Backup(
                    backup_id=v["backup_id"],
                    name=v["name"],
                    source_path=v["source_path"],
                    backup_path=v.get("backup_path", ""),
                    backup_type=BackupType(v.get("backup_type", "full")),
                    status=BackupStatus(v.get("status", "pending")),
                    size_bytes=v.get("size_bytes", 0),
                    checksum=v.get("checksum", ""),
                    created_at=v.get("created_at", 0),
                    expires_at=v.get("expires_at", 0),
                )

            for k, v in state.get("policies", {}).items():
                self._policies[k] = BackupPolicy(**v)

            for k, v in state.get("schedules", {}).items():
                v["backup_type"] = BackupType(v.get("backup_type", "full"))
                self._schedules[k] = BackupSchedule(**{
                    key: val for key, val in v.items()
                    if key in BackupSchedule.__dataclass_fields__
                })

        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for backup system."""
    import argparse

    parser = argparse.ArgumentParser(description="Backup System (Step 246)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a backup")
    create_parser.add_argument("source_path", help="Path to backup")
    create_parser.add_argument("--name", "-n", required=True, help="Backup name")
    create_parser.add_argument("--type", "-t", default="full",
                              choices=["full", "incremental"])
    create_parser.add_argument("--policy", "-p", help="Policy ID")
    create_parser.add_argument("--json", action="store_true", help="JSON output")

    # restore command
    restore_parser = subparsers.add_parser("restore", help="Restore a backup")
    restore_parser.add_argument("backup_id", help="Backup ID")
    restore_parser.add_argument("--target", "-t", required=True, help="Target path")
    restore_parser.add_argument("--overwrite", "-o", action="store_true", help="Overwrite existing")
    restore_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List backups")
    list_parser.add_argument("--status", "-s", choices=["completed", "failed", "pending"])
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a backup")
    delete_parser.add_argument("backup_id", help="Backup ID")
    delete_parser.add_argument("--reason", "-r", default="manual", help="Deletion reason")

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a backup")
    verify_parser.add_argument("backup_id", help="Backup ID")

    # create-policy command
    policy_parser = subparsers.add_parser("create-policy", help="Create a policy")
    policy_parser.add_argument("name", help="Policy name")
    policy_parser.add_argument("--retention", "-r", type=int, default=30, help="Retention days")
    policy_parser.add_argument("--max-backups", "-m", type=int, default=10, help="Max backups")

    args = parser.parse_args()
    system = BackupSystem()

    if args.command == "create":
        backup = asyncio.get_event_loop().run_until_complete(
            system.create_backup(
                name=args.name,
                source_path=args.source_path,
                backup_type=BackupType(args.type),
                policy_id=args.policy,
            )
        )

        if args.json:
            print(json.dumps(backup.to_dict(), indent=2))
        else:
            status = "OK" if backup.status == BackupStatus.COMPLETED else "FAIL"
            print(f"[{status}] {backup.backup_id}")
            print(f"  Name: {backup.name}")
            print(f"  Status: {backup.status.value}")
            print(f"  Size: {backup.size_bytes} bytes")
            print(f"  Path: {backup.backup_path}")

        return 0 if backup.status == BackupStatus.COMPLETED else 1

    elif args.command == "restore":
        result = asyncio.get_event_loop().run_until_complete(
            system.restore(
                backup_id=args.backup_id,
                target_path=args.target,
                overwrite=args.overwrite,
            )
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "OK" if result.status == BackupStatus.COMPLETED else "FAIL"
            print(f"[{status}] Restored {args.backup_id}")
            print(f"  Target: {result.target_path}")
            print(f"  Files: {result.files_restored}")
            print(f"  Duration: {result.duration_ms:.1f}ms")
            if result.error:
                print(f"  Error: {result.error}")

        return 0 if result.status == BackupStatus.COMPLETED else 1

    elif args.command == "list":
        status = BackupStatus(args.status) if args.status else None
        backups = system.list_backups(status=status)

        if args.json:
            print(json.dumps([b.to_dict() for b in backups], indent=2))
        else:
            if not backups:
                print("No backups found")
            else:
                for b in backups:
                    ts = datetime.fromtimestamp(b.created_at).strftime("%Y-%m-%d %H:%M")
                    size_mb = b.size_bytes / (1024 * 1024)
                    print(f"{b.backup_id} [{b.status.value}] {b.name} ({size_mb:.1f}MB) - {ts}")

        return 0

    elif args.command == "delete":
        success = asyncio.get_event_loop().run_until_complete(
            system.delete_backup(args.backup_id, args.reason)
        )
        if success:
            print(f"Deleted backup: {args.backup_id}")
        else:
            print(f"Failed to delete: {args.backup_id}")
        return 0 if success else 1

    elif args.command == "verify":
        valid = asyncio.get_event_loop().run_until_complete(
            system.verify_backup(args.backup_id)
        )
        if valid:
            print(f"Backup verified: {args.backup_id}")
        else:
            print(f"Backup verification failed: {args.backup_id}")
        return 0 if valid else 1

    elif args.command == "create-policy":
        policy = system.create_policy(
            name=args.name,
            retention_days=args.retention,
            max_backups=args.max_backups,
        )
        print(f"Created policy: {policy.policy_id}")
        print(f"  Name: {policy.name}")
        print(f"  Retention: {policy.retention_days} days")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
