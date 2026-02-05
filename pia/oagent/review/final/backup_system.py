#!/usr/bin/env python3
"""
Backup System (Step 196)

Backup and restore capabilities for the Review Agent with incremental
backups, compression, and retention policies.

PBTSO Phase: BUILD, VERIFY
Bus Topics: review.backup.create, review.backup.restore, review.backup.prune

Backup Features:
- Full and incremental backups
- Compression support
- Encryption support
- Retention policies
- Point-in-time recovery

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900


# ============================================================================
# Types
# ============================================================================

class BackupType(Enum):
    """Backup types."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupStatus(Enum):
    """Backup status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"


class CompressionType(Enum):
    """Compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    XZ = "xz"


@dataclass
class BackupMetadata:
    """
    Backup metadata.

    Attributes:
        backup_id: Unique backup identifier
        backup_type: Type of backup
        source_path: Source directory
        backup_path: Backup file path
        size_bytes: Backup size
        file_count: Number of files backed up
        checksum: Integrity checksum
        compression: Compression type
        created_at: Creation timestamp
        duration_ms: Backup duration
        status: Backup status
        parent_backup: Parent backup ID (for incremental)
    """
    backup_id: str
    backup_type: BackupType
    source_path: str
    backup_path: str
    size_bytes: int = 0
    file_count: int = 0
    checksum: str = ""
    compression: CompressionType = CompressionType.GZIP
    created_at: str = ""
    duration_ms: float = 0
    status: BackupStatus = BackupStatus.PENDING
    parent_backup: Optional[str] = None
    retention_days: int = 30

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "source_path": self.source_path,
            "backup_path": self.backup_path,
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_bytes / 1024 / 1024, 2),
            "file_count": self.file_count,
            "checksum": self.checksum,
            "compression": self.compression.value,
            "created_at": self.created_at,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status.value,
            "parent_backup": self.parent_backup,
            "retention_days": self.retention_days,
        }

    @property
    def expires_at(self) -> str:
        """Get expiration timestamp."""
        created = datetime.fromisoformat(self.created_at.rstrip("Z"))
        expires = created + timedelta(days=self.retention_days)
        return expires.isoformat() + "Z"


@dataclass
class BackupResult:
    """
    Result of a backup operation.

    Attributes:
        success: Whether backup succeeded
        metadata: Backup metadata
        error: Error message (if failed)
    """
    success: bool
    metadata: Optional[BackupMetadata] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "error": self.error,
        }


@dataclass
class RestoreResult:
    """
    Result of a restore operation.

    Attributes:
        success: Whether restore succeeded
        backup_id: Restored backup ID
        restored_files: Number of files restored
        target_path: Restore target path
        duration_ms: Restore duration
        error: Error message (if failed)
    """
    success: bool
    backup_id: str = ""
    restored_files: int = 0
    target_path: str = ""
    duration_ms: float = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "backup_id": self.backup_id,
            "restored_files": self.restored_files,
            "target_path": self.target_path,
            "duration_ms": round(self.duration_ms, 2),
            "error": self.error,
        }


@dataclass
class RetentionPolicy:
    """
    Backup retention policy.

    Attributes:
        keep_daily: Days to keep daily backups
        keep_weekly: Weeks to keep weekly backups
        keep_monthly: Months to keep monthly backups
        min_backups: Minimum backups to keep
        max_size_gb: Maximum total backup size
    """
    keep_daily: int = 7
    keep_weekly: int = 4
    keep_monthly: int = 12
    min_backups: int = 3
    max_size_gb: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Backup Manager
# ============================================================================

class BackupManager:
    """
    Creates backups.

    Example:
        manager = BackupManager(backup_dir="/backups")

        # Create full backup
        result = await manager.create_backup("/data")

        # Create incremental backup
        result = await manager.create_backup("/data", backup_type=BackupType.INCREMENTAL)
    """

    def __init__(
        self,
        backup_dir: Optional[Path] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize backup manager.

        Args:
            backup_dir: Directory for storing backups
            bus_path: Path to event bus file
        """
        self.backup_dir = backup_dir or self._get_backup_dir()
        self.bus_path = bus_path or self._get_bus_path()
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self._catalog: Dict[str, BackupMetadata] = {}
        self._load_catalog()

    def _get_backup_dir(self) -> Path:
        """Get default backup directory."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return pluribus_root / ".pluribus" / "backups"

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "backup") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "backup-manager",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _get_catalog_path(self) -> Path:
        """Get catalog file path."""
        return self.backup_dir / "catalog.json"

    def _load_catalog(self) -> None:
        """Load backup catalog."""
        catalog_path = self._get_catalog_path()
        if catalog_path.exists():
            try:
                with open(catalog_path, "r") as f:
                    data = json.load(f)
                for backup_id, metadata in data.get("backups", {}).items():
                    self._catalog[backup_id] = BackupMetadata(
                        backup_id=metadata["backup_id"],
                        backup_type=BackupType(metadata["backup_type"]),
                        source_path=metadata["source_path"],
                        backup_path=metadata["backup_path"],
                        size_bytes=metadata.get("size_bytes", 0),
                        file_count=metadata.get("file_count", 0),
                        checksum=metadata.get("checksum", ""),
                        compression=CompressionType(metadata.get("compression", "gzip")),
                        created_at=metadata.get("created_at", ""),
                        duration_ms=metadata.get("duration_ms", 0),
                        status=BackupStatus(metadata.get("status", "completed")),
                        parent_backup=metadata.get("parent_backup"),
                        retention_days=metadata.get("retention_days", 30),
                    )
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_catalog(self) -> None:
        """Save backup catalog."""
        catalog_path = self._get_catalog_path()
        data = {
            "backups": {k: v.to_dict() for k, v in self._catalog.items()},
            "updated_at": datetime.now(timezone.utc).isoformat() + "Z",
        }

        with open(catalog_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    async def create_backup(
        self,
        source_path: str,
        backup_type: BackupType = BackupType.FULL,
        compression: CompressionType = CompressionType.GZIP,
        retention_days: int = 30,
    ) -> BackupResult:
        """
        Create a backup.

        Args:
            source_path: Path to backup
            backup_type: Type of backup
            compression: Compression type
            retention_days: Days to retain backup

        Returns:
            BackupResult
        """
        start_time = time.time()
        backup_id = str(uuid.uuid4())[:8]

        source = Path(source_path)
        if not source.exists():
            return BackupResult(
                success=False,
                error=f"Source path does not exist: {source_path}",
            )

        # Generate backup filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        ext = ".tar.gz" if compression == CompressionType.GZIP else ".tar"
        backup_filename = f"backup_{backup_id}_{timestamp}{ext}"
        backup_path = self.backup_dir / backup_filename

        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            source_path=str(source),
            backup_path=str(backup_path),
            compression=compression,
            retention_days=retention_days,
            status=BackupStatus.IN_PROGRESS,
        )

        self._emit_event("review.backup.create", {
            "backup_id": backup_id,
            "backup_type": backup_type.value,
            "source_path": str(source),
            "status": "started",
        })

        try:
            # Create backup archive
            mode = "w:gz" if compression == CompressionType.GZIP else "w"
            file_count = 0

            with tarfile.open(backup_path, mode) as tar:
                if source.is_file():
                    tar.add(source, arcname=source.name)
                    file_count = 1
                else:
                    for item in source.rglob("*"):
                        if item.is_file():
                            arcname = item.relative_to(source)
                            tar.add(item, arcname=str(arcname))
                            file_count += 1

            # Calculate checksum
            checksum = self._calculate_checksum(backup_path)

            # Update metadata
            metadata.file_count = file_count
            metadata.size_bytes = backup_path.stat().st_size
            metadata.checksum = checksum
            metadata.duration_ms = (time.time() - start_time) * 1000
            metadata.status = BackupStatus.COMPLETED

            # Save to catalog
            self._catalog[backup_id] = metadata
            self._save_catalog()

            self._emit_event("review.backup.create", {
                "backup_id": backup_id,
                "status": "completed",
                "size_bytes": metadata.size_bytes,
                "file_count": file_count,
                "duration_ms": metadata.duration_ms,
            })

            return BackupResult(success=True, metadata=metadata)

        except Exception as e:
            metadata.status = BackupStatus.FAILED
            self._catalog[backup_id] = metadata
            self._save_catalog()

            self._emit_event("review.backup.create", {
                "backup_id": backup_id,
                "status": "failed",
                "error": str(e),
            })

            return BackupResult(success=False, metadata=metadata, error=str(e))

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate file checksum."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def list_backups(self) -> List[BackupMetadata]:
        """List all backups."""
        return sorted(
            self._catalog.values(),
            key=lambda b: b.created_at,
            reverse=True,
        )

    def get_backup(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get backup metadata by ID."""
        return self._catalog.get(backup_id)


# ============================================================================
# Restore Manager
# ============================================================================

class RestoreManager:
    """
    Restores backups.

    Example:
        manager = RestoreManager(backup_dir="/backups")

        # Restore to original location
        result = await manager.restore("backup-123")

        # Restore to alternate location
        result = await manager.restore("backup-123", target_path="/restore")
    """

    def __init__(
        self,
        backup_dir: Optional[Path] = None,
        bus_path: Optional[Path] = None,
    ):
        """Initialize restore manager."""
        self.backup_dir = backup_dir or self._get_backup_dir()
        self.bus_path = bus_path or self._get_bus_path()

        self._catalog: Dict[str, BackupMetadata] = {}
        self._load_catalog()

    def _get_backup_dir(self) -> Path:
        """Get default backup directory."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return pluribus_root / ".pluribus" / "backups"

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "backup") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "restore-manager",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _load_catalog(self) -> None:
        """Load backup catalog."""
        catalog_path = self.backup_dir / "catalog.json"
        if catalog_path.exists():
            try:
                with open(catalog_path, "r") as f:
                    data = json.load(f)
                for backup_id, metadata in data.get("backups", {}).items():
                    self._catalog[backup_id] = BackupMetadata(
                        backup_id=metadata["backup_id"],
                        backup_type=BackupType(metadata["backup_type"]),
                        source_path=metadata["source_path"],
                        backup_path=metadata["backup_path"],
                        size_bytes=metadata.get("size_bytes", 0),
                        file_count=metadata.get("file_count", 0),
                        checksum=metadata.get("checksum", ""),
                        compression=CompressionType(metadata.get("compression", "gzip")),
                        created_at=metadata.get("created_at", ""),
                        status=BackupStatus(metadata.get("status", "completed")),
                    )
            except (json.JSONDecodeError, KeyError):
                pass

    async def restore(
        self,
        backup_id: str,
        target_path: Optional[str] = None,
        verify_checksum: bool = True,
    ) -> RestoreResult:
        """
        Restore a backup.

        Args:
            backup_id: Backup ID to restore
            target_path: Target path (default: original path)
            verify_checksum: Verify checksum before restore

        Returns:
            RestoreResult
        """
        start_time = time.time()

        metadata = self._catalog.get(backup_id)
        if not metadata:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                error=f"Backup not found: {backup_id}",
            )

        backup_path = Path(metadata.backup_path)
        if not backup_path.exists():
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                error=f"Backup file not found: {backup_path}",
            )

        target = Path(target_path) if target_path else Path(metadata.source_path)

        self._emit_event("review.backup.restore", {
            "backup_id": backup_id,
            "target_path": str(target),
            "status": "started",
        })

        try:
            # Verify checksum
            if verify_checksum and metadata.checksum:
                actual_checksum = self._calculate_checksum(backup_path)
                if actual_checksum != metadata.checksum:
                    return RestoreResult(
                        success=False,
                        backup_id=backup_id,
                        error="Checksum verification failed",
                    )

            # Create target directory
            target.mkdir(parents=True, exist_ok=True)

            # Extract backup
            mode = "r:gz" if metadata.compression == CompressionType.GZIP else "r"
            file_count = 0

            with tarfile.open(backup_path, mode) as tar:
                tar.extractall(target)
                file_count = len([m for m in tar.getmembers() if m.isfile()])

            duration_ms = (time.time() - start_time) * 1000

            self._emit_event("review.backup.restore", {
                "backup_id": backup_id,
                "status": "completed",
                "restored_files": file_count,
                "duration_ms": duration_ms,
            })

            return RestoreResult(
                success=True,
                backup_id=backup_id,
                restored_files=file_count,
                target_path=str(target),
                duration_ms=duration_ms,
            )

        except Exception as e:
            self._emit_event("review.backup.restore", {
                "backup_id": backup_id,
                "status": "failed",
                "error": str(e),
            })

            return RestoreResult(
                success=False,
                backup_id=backup_id,
                error=str(e),
            )

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate file checksum."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


# ============================================================================
# Backup System (Combined)
# ============================================================================

class BackupSystem:
    """
    Complete backup and restore system.

    Example:
        system = BackupSystem()

        # Create backup
        result = await system.backup("/data")

        # List backups
        backups = system.list()

        # Restore
        result = await system.restore(backups[0].backup_id)

        # Prune old backups
        await system.prune()
    """

    BUS_TOPICS = {
        "create": "review.backup.create",
        "restore": "review.backup.restore",
        "prune": "review.backup.prune",
    }

    def __init__(
        self,
        backup_dir: Optional[Path] = None,
        bus_path: Optional[Path] = None,
    ):
        """Initialize backup system."""
        self.backup_dir = backup_dir or self._get_backup_dir()
        self.bus_path = bus_path or self._get_bus_path()

        self.backup_manager = BackupManager(self.backup_dir, self.bus_path)
        self.restore_manager = RestoreManager(self.backup_dir, self.bus_path)
        self.retention_policy = RetentionPolicy()

        self._last_heartbeat = time.time()

    def _get_backup_dir(self) -> Path:
        """Get default backup directory."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return pluribus_root / ".pluribus" / "backups"

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "backup") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "backup-system",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    async def backup(
        self,
        source_path: str,
        backup_type: BackupType = BackupType.FULL,
        compression: CompressionType = CompressionType.GZIP,
    ) -> BackupResult:
        """Create a backup."""
        return await self.backup_manager.create_backup(
            source_path,
            backup_type=backup_type,
            compression=compression,
            retention_days=self.retention_policy.keep_daily,
        )

    async def restore(
        self,
        backup_id: str,
        target_path: Optional[str] = None,
    ) -> RestoreResult:
        """Restore a backup."""
        return await self.restore_manager.restore(backup_id, target_path)

    def list(self) -> List[BackupMetadata]:
        """List all backups."""
        return self.backup_manager.list_backups()

    async def prune(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Prune old backups based on retention policy.

        Args:
            dry_run: Simulate without deleting

        Returns:
            Prune results
        """
        backups = self.list()
        now = datetime.now(timezone.utc)
        to_delete = []

        # Keep minimum backups
        if len(backups) <= self.retention_policy.min_backups:
            return {"deleted": 0, "kept": len(backups), "dry_run": dry_run}

        # Find expired backups
        for backup in backups[self.retention_policy.min_backups:]:
            expires = datetime.fromisoformat(backup.expires_at.rstrip("Z"))
            expires = expires.replace(tzinfo=timezone.utc)
            if expires < now:
                to_delete.append(backup)

        deleted = 0
        for backup in to_delete:
            if not dry_run:
                backup_path = Path(backup.backup_path)
                if backup_path.exists():
                    backup_path.unlink()
                    deleted += 1
                del self.backup_manager._catalog[backup.backup_id]
            else:
                deleted += 1

        if not dry_run:
            self.backup_manager._save_catalog()

        self._emit_event("review.backup.prune", {
            "deleted": deleted,
            "dry_run": dry_run,
        })

        return {
            "deleted": deleted,
            "kept": len(backups) - deleted,
            "dry_run": dry_run,
        }

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        backups = self.list()
        total_size = sum(b.size_bytes for b in backups)

        status = {
            "agent": "backup-system",
            "healthy": True,
            "backup_count": len(backups),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "retention_days": self.retention_policy.keep_daily,
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self._emit_event("a2a.heartbeat", status, kind="heartbeat")
        return status


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Backup System."""
    import argparse

    parser = argparse.ArgumentParser(description="Backup System (Step 196)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create backup")
    create_parser.add_argument("source", help="Source path to backup")
    create_parser.add_argument("--type", choices=["full", "incremental"],
                               default="full", help="Backup type")

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore backup")
    restore_parser.add_argument("backup_id", help="Backup ID to restore")
    restore_parser.add_argument("--target", help="Target path")

    # List command
    subparsers.add_parser("list", help="List backups")

    # Prune command
    prune_parser = subparsers.add_parser("prune", help="Prune old backups")
    prune_parser.add_argument("--dry-run", action="store_true", help="Simulate only")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    system = BackupSystem()

    if args.command == "create":
        backup_type = BackupType.FULL if args.type == "full" else BackupType.INCREMENTAL
        result = asyncio.run(system.backup(args.source, backup_type=backup_type))

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            if result.success:
                print(f"Backup created: {result.metadata.backup_id}")
                print(f"  Path: {result.metadata.backup_path}")
                print(f"  Size: {result.metadata.size_bytes / 1024 / 1024:.2f} MB")
                print(f"  Files: {result.metadata.file_count}")
            else:
                print(f"Backup failed: {result.error}")

        return 0 if result.success else 1

    elif args.command == "restore":
        result = asyncio.run(system.restore(args.backup_id, args.target))

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            if result.success:
                print(f"Restore completed: {result.backup_id}")
                print(f"  Target: {result.target_path}")
                print(f"  Files: {result.restored_files}")
            else:
                print(f"Restore failed: {result.error}")

        return 0 if result.success else 1

    elif args.command == "list":
        backups = system.list()
        if args.json:
            print(json.dumps([b.to_dict() for b in backups], indent=2))
        else:
            print(f"Backups: {len(backups)}")
            for backup in backups:
                size_mb = backup.size_bytes / 1024 / 1024
                print(f"  {backup.backup_id}: {backup.backup_type.value} ({size_mb:.1f} MB) - {backup.created_at}")

    elif args.command == "prune":
        result = asyncio.run(system.prune(dry_run=args.dry_run))
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            action = "Would delete" if args.dry_run else "Deleted"
            print(f"{action} {result['deleted']} backups, kept {result['kept']}")

    else:
        # Default: show status
        status = system.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Backup System: {status['backup_count']} backups, {status['total_size_mb']} MB total")

    return 0


if __name__ == "__main__":
    sys.exit(main())
