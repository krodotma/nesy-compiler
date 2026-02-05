#!/usr/bin/env python3
"""
backup_system.py - Backup System (Step 96)

PBTSO Phase: VERIFY, ITERATE

Provides:
- File/directory backup
- Incremental backups
- Compression support
- Backup verification
- Point-in-time restore

Bus Topics:
- code.backup.create
- code.backup.restore
- code.backup.verify
- code.backup.status

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import socket
import tarfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class BackupStatus(Enum):
    """Backup status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"


class BackupType(Enum):
    """Backup type."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class CompressionType(Enum):
    """Compression type."""
    NONE = "none"
    GZIP = "gzip"
    BZ2 = "bz2"
    XZ = "xz"


@dataclass
class BackupConfig:
    """Configuration for backup module."""
    backup_dir: str = "/pluribus/.pluribus/backups"
    max_backups: int = 10
    compression: CompressionType = CompressionType.GZIP
    verify_after_backup: bool = True
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "__pycache__", "*.pyc", ".git", "node_modules", ".venv", "*.log",
    ])
    max_file_size_mb: int = 100
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_dir": self.backup_dir,
            "max_backups": self.max_backups,
            "compression": self.compression.value,
            "verify_after_backup": self.verify_after_backup,
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
# Backup Types
# =============================================================================

@dataclass
class FileEntry:
    """Entry for a backed up file."""
    path: str
    size: int
    mtime: float
    checksum: str
    compressed_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "mtime": self.mtime,
            "checksum": self.checksum,
            "compressed_size": self.compressed_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileEntry":
        return cls(
            path=data["path"],
            size=data["size"],
            mtime=data["mtime"],
            checksum=data["checksum"],
            compressed_size=data.get("compressed_size"),
        )


@dataclass
class BackupManifest:
    """Manifest describing a backup."""
    backup_id: str
    name: str
    created_at: float
    backup_type: BackupType
    source_path: str
    files: List[FileEntry]
    total_size: int
    compressed_size: int
    compression: CompressionType
    checksum: str
    parent_backup_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "name": self.name,
            "created_at": self.created_at,
            "iso": datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat(),
            "backup_type": self.backup_type.value,
            "source_path": self.source_path,
            "file_count": len(self.files),
            "total_size": self.total_size,
            "compressed_size": self.compressed_size,
            "compression": self.compression.value,
            "checksum": self.checksum,
            "parent_backup_id": self.parent_backup_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupManifest":
        return cls(
            backup_id=data["backup_id"],
            name=data["name"],
            created_at=data["created_at"],
            backup_type=BackupType(data["backup_type"]),
            source_path=data["source_path"],
            files=[FileEntry.from_dict(f) for f in data.get("files", [])],
            total_size=data["total_size"],
            compressed_size=data["compressed_size"],
            compression=CompressionType(data["compression"]),
            checksum=data["checksum"],
            parent_backup_id=data.get("parent_backup_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Backup:
    """A backup instance."""
    id: str
    manifest: BackupManifest
    archive_path: Path
    status: BackupStatus = BackupStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "manifest": self.manifest.to_dict(),
            "archive_path": str(self.archive_path),
            "status": self.status.value,
        }


@dataclass
class RestoreResult:
    """Result of a restore operation."""
    backup_id: str
    target_path: str
    status: BackupStatus
    files_restored: int
    duration_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backup_id": self.backup_id,
            "target_path": self.target_path,
            "status": self.status.value,
            "files_restored": self.files_restored,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


# =============================================================================
# Backup Module
# =============================================================================

class BackupModule:
    """
    Backup module for data protection.

    PBTSO Phase: VERIFY, ITERATE

    Features:
    - Full and incremental backups
    - Compression support
    - Backup verification
    - Point-in-time restore
    - Backup rotation

    Usage:
        backup = BackupModule()

        # Create backup
        result = backup.create_backup("/path/to/data", "my_backup")

        # Restore backup
        restore_result = backup.restore_backup(result.id, "/path/to/restore")
    """

    BUS_TOPICS = {
        "create": "code.backup.create",
        "restore": "code.backup.restore",
        "verify": "code.backup.verify",
        "status": "code.backup.status",
    }

    def __init__(
        self,
        config: Optional[BackupConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or BackupConfig()
        self.bus = bus or LockedAgentBus()

        self._backup_dir = Path(self.config.backup_dir)
        self._backup_dir.mkdir(parents=True, exist_ok=True)

        self._backups: Dict[str, Backup] = {}
        self._lock = Lock()

        # Load existing backups
        self._load_backups()

    def _load_backups(self) -> None:
        """Load existing backups from disk."""
        manifest_files = list(self._backup_dir.glob("*.manifest.json"))
        for manifest_path in manifest_files:
            try:
                data = json.loads(manifest_path.read_text())
                manifest = BackupManifest.from_dict(data)
                archive_path = manifest_path.with_suffix("").with_suffix(".tar.gz")

                backup = Backup(
                    id=manifest.backup_id,
                    manifest=manifest,
                    archive_path=archive_path,
                    status=BackupStatus.COMPLETED if archive_path.exists() else BackupStatus.CORRUPTED,
                )
                self._backups[backup.id] = backup
            except Exception:
                pass

    # =========================================================================
    # Backup Creation
    # =========================================================================

    def create_backup(
        self,
        source_path: Union[str, Path],
        name: str,
        backup_type: BackupType = BackupType.FULL,
        parent_backup_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Backup:
        """Create a new backup."""
        source = Path(source_path)
        if not source.exists():
            raise ValueError(f"Source path does not exist: {source}")

        backup_id = f"bak-{uuid.uuid4().hex[:12]}"
        timestamp = time.time()

        self.bus.emit({
            "topic": self.BUS_TOPICS["create"],
            "kind": "backup",
            "actor": "backup-module",
            "data": {
                "backup_id": backup_id,
                "source": str(source),
                "type": backup_type.value,
            },
        })

        # Collect files
        files: List[FileEntry] = []
        total_size = 0
        parent_files: Set[str] = set()

        # Get parent files for incremental backup
        if backup_type == BackupType.INCREMENTAL and parent_backup_id:
            parent = self._backups.get(parent_backup_id)
            if parent:
                parent_files = {f.path: f.checksum for f in parent.manifest.files}

        for file_path in self._collect_files(source):
            rel_path = str(file_path.relative_to(source))
            stat = file_path.stat()

            # Check size limit
            if stat.st_size > self.config.max_file_size_mb * 1024 * 1024:
                continue

            checksum = self._calculate_checksum(file_path)

            # Skip unchanged files for incremental backup
            if backup_type == BackupType.INCREMENTAL:
                if parent_files.get(rel_path) == checksum:
                    continue

            files.append(FileEntry(
                path=rel_path,
                size=stat.st_size,
                mtime=stat.st_mtime,
                checksum=checksum,
            ))
            total_size += stat.st_size

        # Create archive
        archive_name = f"{backup_id}.tar.gz"
        archive_path = self._backup_dir / archive_name

        compressed_size = self._create_archive(source, archive_path, files)

        # Calculate archive checksum
        archive_checksum = self._calculate_checksum(archive_path)

        # Create manifest
        manifest = BackupManifest(
            backup_id=backup_id,
            name=name,
            created_at=timestamp,
            backup_type=backup_type,
            source_path=str(source),
            files=files,
            total_size=total_size,
            compressed_size=compressed_size,
            compression=self.config.compression,
            checksum=archive_checksum,
            parent_backup_id=parent_backup_id,
            metadata=metadata or {},
        )

        # Save manifest
        manifest_path = self._backup_dir / f"{backup_id}.manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        # Create backup object
        backup = Backup(
            id=backup_id,
            manifest=manifest,
            archive_path=archive_path,
            status=BackupStatus.COMPLETED,
        )

        with self._lock:
            self._backups[backup_id] = backup

        # Verify if configured
        if self.config.verify_after_backup:
            if self.verify_backup(backup_id):
                backup.status = BackupStatus.VERIFIED

        # Rotate old backups
        self._rotate_backups()

        self.bus.emit({
            "topic": self.BUS_TOPICS["status"],
            "kind": "backup",
            "actor": "backup-module",
            "data": {
                "backup_id": backup_id,
                "status": backup.status.value,
                "files": len(files),
                "size": compressed_size,
            },
        })

        return backup

    def _collect_files(self, source: Path) -> List[Path]:
        """Collect files for backup."""
        files = []

        for path in source.rglob("*"):
            if path.is_file():
                # Check exclude patterns
                should_exclude = False
                for pattern in self.config.exclude_patterns:
                    if path.match(pattern):
                        should_exclude = True
                        break

                if not should_exclude:
                    files.append(path)

        return files

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _create_archive(
        self,
        source: Path,
        archive_path: Path,
        files: List[FileEntry],
    ) -> int:
        """Create tar archive with compression."""
        mode = "w:gz"
        if self.config.compression == CompressionType.BZ2:
            mode = "w:bz2"
        elif self.config.compression == CompressionType.XZ:
            mode = "w:xz"
        elif self.config.compression == CompressionType.NONE:
            mode = "w"

        with tarfile.open(archive_path, mode) as tar:
            for file_entry in files:
                file_path = source / file_entry.path
                tar.add(file_path, arcname=file_entry.path)

        return archive_path.stat().st_size

    def _rotate_backups(self) -> None:
        """Remove old backups to stay within limit."""
        with self._lock:
            if len(self._backups) <= self.config.max_backups:
                return

            # Sort by creation time
            sorted_backups = sorted(
                self._backups.values(),
                key=lambda b: b.manifest.created_at,
            )

            # Remove oldest backups
            to_remove = len(sorted_backups) - self.config.max_backups
            for backup in sorted_backups[:to_remove]:
                self.delete_backup(backup.id)

    # =========================================================================
    # Backup Restoration
    # =========================================================================

    def restore_backup(
        self,
        backup_id: str,
        target_path: Union[str, Path],
        overwrite: bool = False,
    ) -> RestoreResult:
        """Restore a backup to a target location."""
        backup = self._backups.get(backup_id)
        if not backup:
            return RestoreResult(
                backup_id=backup_id,
                target_path=str(target_path),
                status=BackupStatus.FAILED,
                files_restored=0,
                duration_ms=0,
                error="Backup not found",
            )

        target = Path(target_path)
        if target.exists() and not overwrite:
            return RestoreResult(
                backup_id=backup_id,
                target_path=str(target),
                status=BackupStatus.FAILED,
                files_restored=0,
                duration_ms=0,
                error="Target path exists and overwrite is False",
            )

        self.bus.emit({
            "topic": self.BUS_TOPICS["restore"],
            "kind": "backup",
            "actor": "backup-module",
            "data": {
                "backup_id": backup_id,
                "target": str(target),
            },
        })

        start = time.time()

        try:
            target.mkdir(parents=True, exist_ok=True)

            # Extract archive
            mode = "r:gz"
            if backup.manifest.compression == CompressionType.BZ2:
                mode = "r:bz2"
            elif backup.manifest.compression == CompressionType.XZ:
                mode = "r:xz"
            elif backup.manifest.compression == CompressionType.NONE:
                mode = "r"

            files_restored = 0
            with tarfile.open(backup.archive_path, mode) as tar:
                tar.extractall(target)
                files_restored = len(backup.manifest.files)

            duration = (time.time() - start) * 1000

            return RestoreResult(
                backup_id=backup_id,
                target_path=str(target),
                status=BackupStatus.COMPLETED,
                files_restored=files_restored,
                duration_ms=duration,
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return RestoreResult(
                backup_id=backup_id,
                target_path=str(target),
                status=BackupStatus.FAILED,
                files_restored=0,
                duration_ms=duration,
                error=str(e),
            )

    # =========================================================================
    # Backup Verification
    # =========================================================================

    def verify_backup(self, backup_id: str) -> bool:
        """Verify backup integrity."""
        backup = self._backups.get(backup_id)
        if not backup:
            return False

        if not backup.archive_path.exists():
            backup.status = BackupStatus.CORRUPTED
            return False

        self.bus.emit({
            "topic": self.BUS_TOPICS["verify"],
            "kind": "backup",
            "actor": "backup-module",
            "data": {"backup_id": backup_id},
        })

        # Verify archive checksum
        current_checksum = self._calculate_checksum(backup.archive_path)
        if current_checksum != backup.manifest.checksum:
            backup.status = BackupStatus.CORRUPTED
            return False

        # Try to read archive
        try:
            mode = "r:gz"
            if backup.manifest.compression == CompressionType.BZ2:
                mode = "r:bz2"
            elif backup.manifest.compression == CompressionType.XZ:
                mode = "r:xz"
            elif backup.manifest.compression == CompressionType.NONE:
                mode = "r"

            with tarfile.open(backup.archive_path, mode) as tar:
                names = tar.getnames()
                if len(names) != len(backup.manifest.files):
                    backup.status = BackupStatus.CORRUPTED
                    return False

        except Exception:
            backup.status = BackupStatus.CORRUPTED
            return False

        backup.status = BackupStatus.VERIFIED
        return True

    # =========================================================================
    # Backup Management
    # =========================================================================

    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        backup = self._backups.get(backup_id)
        if not backup:
            return False

        try:
            # Delete archive
            if backup.archive_path.exists():
                backup.archive_path.unlink()

            # Delete manifest
            manifest_path = self._backup_dir / f"{backup_id}.manifest.json"
            if manifest_path.exists():
                manifest_path.unlink()

            with self._lock:
                del self._backups[backup_id]

            return True
        except Exception:
            return False

    def get_backup(self, backup_id: str) -> Optional[Backup]:
        """Get backup by ID."""
        return self._backups.get(backup_id)

    def list_backups(self) -> List[Backup]:
        """List all backups."""
        return sorted(
            self._backups.values(),
            key=lambda b: b.manifest.created_at,
            reverse=True,
        )

    def get_latest_backup(self, name: Optional[str] = None) -> Optional[Backup]:
        """Get the latest backup."""
        backups = self.list_backups()
        if name:
            backups = [b for b in backups if b.manifest.name == name]
        return backups[0] if backups else None

    def stats(self) -> Dict[str, Any]:
        """Get backup statistics."""
        total_size = sum(
            b.manifest.compressed_size
            for b in self._backups.values()
        )
        return {
            "total_backups": len(self._backups),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / 1024 / 1024,
            "backup_dir": str(self._backup_dir),
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Backup System."""
    import argparse

    parser = argparse.ArgumentParser(description="Backup System (Step 96)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create command
    create_parser = subparsers.add_parser("create", help="Create backup")
    create_parser.add_argument("source", help="Source path to backup")
    create_parser.add_argument("--name", "-n", default="backup", help="Backup name")
    create_parser.add_argument("--type", "-t", choices=["full", "incremental"], default="full")

    # restore command
    restore_parser = subparsers.add_parser("restore", help="Restore backup")
    restore_parser.add_argument("backup_id", help="Backup ID")
    restore_parser.add_argument("target", help="Target path")
    restore_parser.add_argument("--overwrite", action="store_true")

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify backup")
    verify_parser.add_argument("backup_id", help="Backup ID")

    # list command
    list_parser = subparsers.add_parser("list", help="List backups")
    list_parser.add_argument("--json", action="store_true")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete backup")
    delete_parser.add_argument("backup_id", help="Backup ID")

    # stats command
    subparsers.add_parser("stats", help="Show backup statistics")

    args = parser.parse_args()
    backup_module = BackupModule()

    if args.command == "create":
        backup_type = BackupType.FULL if args.type == "full" else BackupType.INCREMENTAL
        backup = backup_module.create_backup(args.source, args.name, backup_type)
        print(f"Backup created: {backup.id}")
        print(f"  Status: {backup.status.value}")
        print(f"  Files: {len(backup.manifest.files)}")
        print(f"  Size: {backup.manifest.compressed_size / 1024:.2f} KB")
        return 0

    elif args.command == "restore":
        result = backup_module.restore_backup(
            args.backup_id,
            args.target,
            overwrite=args.overwrite,
        )
        print(f"Restore: {result.status.value}")
        print(f"  Files restored: {result.files_restored}")
        print(f"  Duration: {result.duration_ms:.2f}ms")
        if result.error:
            print(f"  Error: {result.error}")
        return 0 if result.status == BackupStatus.COMPLETED else 1

    elif args.command == "verify":
        valid = backup_module.verify_backup(args.backup_id)
        print(f"Verification: {'PASSED' if valid else 'FAILED'}")
        return 0 if valid else 1

    elif args.command == "list":
        backups = backup_module.list_backups()
        if args.json:
            print(json.dumps([b.to_dict() for b in backups], indent=2))
        else:
            print("Backups:")
            for b in backups:
                ts = datetime.fromtimestamp(b.manifest.created_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
                size = b.manifest.compressed_size / 1024
                print(f"  [{b.id}] {b.manifest.name} ({ts}) - {size:.1f}KB - {b.status.value}")
        return 0

    elif args.command == "delete":
        if backup_module.delete_backup(args.backup_id):
            print(f"Deleted backup: {args.backup_id}")
            return 0
        else:
            print(f"Failed to delete backup: {args.backup_id}")
            return 1

    elif args.command == "stats":
        stats = backup_module.stats()
        print(json.dumps(stats, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
