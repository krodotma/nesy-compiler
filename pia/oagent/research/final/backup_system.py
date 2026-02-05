#!/usr/bin/env python3
"""
backup_system.py - Backup/Restore System (Step 46)

Comprehensive backup and restore capabilities for Research Agent data.
Supports incremental backups, compression, and verification.

PBTSO Phase: PROTECT

Bus Topics:
- a2a.research.backup.start
- a2a.research.backup.complete
- a2a.research.backup.restore
- research.backup.verify

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class BackupType(Enum):
    """Type of backup."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupStatus(Enum):
    """Backup status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"


class CompressionType(Enum):
    """Compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"


@dataclass
class BackupConfig:
    """Configuration for backup system."""

    backup_dir: str = ""
    retention_days: int = 30
    max_backups: int = 100
    compression: CompressionType = CompressionType.GZIP
    verify_after_backup: bool = True
    include_metadata: bool = True
    chunk_size_mb: int = 100
    parallel_workers: int = 4
    emit_to_bus: bool = True
    bus_path: Optional[str] = None

    def __post_init__(self):
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        if not self.backup_dir:
            self.backup_dir = f"{pluribus_root}/.pluribus/research/backups"
        if self.bus_path is None:
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class BackupManifest:
    """Manifest describing a backup."""

    id: str
    name: str
    backup_type: BackupType
    status: BackupStatus
    created_at: float
    completed_at: Optional[float] = None
    source_path: str = ""
    backup_path: str = ""
    size_bytes: int = 0
    file_count: int = 0
    compression: CompressionType = CompressionType.GZIP
    checksum: str = ""
    parent_backup_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    files: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.backup_type.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "source_path": self.source_path,
            "backup_path": self.backup_path,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
            "compression": self.compression.value,
            "checksum": self.checksum,
            "parent_backup_id": self.parent_backup_id,
            "metadata": self.metadata,
            "files": self.files,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupManifest":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            backup_type=BackupType(data["type"]),
            status=BackupStatus(data["status"]),
            created_at=data["created_at"],
            completed_at=data.get("completed_at"),
            source_path=data.get("source_path", ""),
            backup_path=data.get("backup_path", ""),
            size_bytes=data.get("size_bytes", 0),
            file_count=data.get("file_count", 0),
            compression=CompressionType(data.get("compression", "gzip")),
            checksum=data.get("checksum", ""),
            parent_backup_id=data.get("parent_backup_id"),
            metadata=data.get("metadata", {}),
            files=data.get("files", []),
        )


@dataclass
class RestoreResult:
    """Result of a restore operation."""

    success: bool
    backup_id: str
    files_restored: int = 0
    bytes_restored: int = 0
    duration_ms: float = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "backup_id": self.backup_id,
            "files_restored": self.files_restored,
            "bytes_restored": self.bytes_restored,
            "duration_ms": self.duration_ms,
            "error_count": len(self.errors),
        }


@dataclass
class VerificationResult:
    """Result of backup verification."""

    valid: bool
    backup_id: str
    files_verified: int = 0
    files_missing: int = 0
    files_corrupted: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "backup_id": self.backup_id,
            "files_verified": self.files_verified,
            "files_missing": self.files_missing,
            "files_corrupted": self.files_corrupted,
        }


# ============================================================================
# Backup Manager
# ============================================================================


class BackupManager:
    """
    Backup and restore manager for Research Agent.

    Features:
    - Full and incremental backups
    - Compression support
    - Backup verification
    - Retention management
    - Restore functionality

    PBTSO Phase: PROTECT

    Example:
        backup = BackupManager()

        # Create full backup
        manifest = backup.create_backup(
            source="/data/research",
            name="daily_backup",
        )

        # List backups
        backups = backup.list_backups()

        # Restore
        result = backup.restore(manifest.id, target="/data/restore")

        # Verify
        result = backup.verify(manifest.id)
    """

    def __init__(
        self,
        config: Optional[BackupConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the backup manager.

        Args:
            config: Backup configuration
            bus: AgentBus for event emission
        """
        self.config = config or BackupConfig()
        self.bus = bus or AgentBus()

        # Ensure backup directory exists
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)

        # Manifest cache
        self._manifests: Dict[str, BackupManifest] = {}
        self._load_manifests()

        # Statistics
        self._stats = {
            "total_backups": 0,
            "total_restores": 0,
            "total_bytes_backed_up": 0,
            "total_bytes_restored": 0,
        }

    def create_backup(
        self,
        source: str,
        name: Optional[str] = None,
        backup_type: BackupType = BackupType.FULL,
        parent_id: Optional[str] = None,
    ) -> BackupManifest:
        """
        Create a backup.

        Args:
            source: Source directory to backup
            name: Backup name
            backup_type: Type of backup
            parent_id: Parent backup ID for incremental

        Returns:
            BackupManifest with backup details
        """
        source_path = Path(source)
        if not source_path.exists():
            raise ValueError(f"Source path does not exist: {source}")

        # Generate backup ID and name
        backup_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = name or f"backup_{timestamp}"

        # Create manifest
        manifest = BackupManifest(
            id=backup_id,
            name=backup_name,
            backup_type=backup_type,
            status=BackupStatus.IN_PROGRESS,
            created_at=time.time(),
            source_path=str(source_path),
            compression=self.config.compression,
            parent_backup_id=parent_id,
        )

        self._emit_event("a2a.research.backup.start", {
            "id": backup_id,
            "name": backup_name,
            "type": backup_type.value,
            "source": str(source_path),
        })

        try:
            # Create backup directory
            backup_dir = Path(self.config.backup_dir) / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Get files to backup
            if backup_type == BackupType.INCREMENTAL and parent_id:
                files_to_backup = self._get_incremental_files(source_path, parent_id)
            else:
                files_to_backup = self._get_all_files(source_path)

            # Create archive
            archive_path = backup_dir / f"{backup_name}.tar"
            if self.config.compression == CompressionType.GZIP:
                archive_path = archive_path.with_suffix(".tar.gz")

            self._create_archive(source_path, archive_path, files_to_backup, manifest)

            # Update manifest
            manifest.backup_path = str(archive_path)
            manifest.size_bytes = archive_path.stat().st_size
            manifest.file_count = len(files_to_backup)
            manifest.checksum = self._calculate_checksum(archive_path)
            manifest.completed_at = time.time()
            manifest.status = BackupStatus.COMPLETED

            # Save manifest
            self._save_manifest(manifest, backup_dir)
            self._manifests[backup_id] = manifest

            # Verify if configured
            if self.config.verify_after_backup:
                verify_result = self.verify(backup_id)
                if verify_result.valid:
                    manifest.status = BackupStatus.VERIFIED

            self._stats["total_backups"] += 1
            self._stats["total_bytes_backed_up"] += manifest.size_bytes

            self._emit_event("a2a.research.backup.complete", manifest.to_dict())

            # Cleanup old backups
            self._cleanup_old_backups()

        except Exception as e:
            manifest.status = BackupStatus.FAILED
            manifest.metadata["error"] = str(e)

            self._emit_event("a2a.research.backup.complete", {
                "id": backup_id,
                "status": "failed",
                "error": str(e),
            }, level="error")

        return manifest

    def restore(
        self,
        backup_id: str,
        target: str,
        overwrite: bool = False,
    ) -> RestoreResult:
        """
        Restore from a backup.

        Args:
            backup_id: Backup ID to restore
            target: Target directory
            overwrite: Whether to overwrite existing files

        Returns:
            RestoreResult with operation details
        """
        manifest = self._manifests.get(backup_id)
        if not manifest:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                errors=["Backup not found"],
            )

        target_path = Path(target)
        start_time = time.time()

        result = RestoreResult(
            success=True,
            backup_id=backup_id,
        )

        self._emit_event("a2a.research.backup.restore", {
            "backup_id": backup_id,
            "target": str(target_path),
        })

        try:
            archive_path = Path(manifest.backup_path)
            if not archive_path.exists():
                result.success = False
                result.errors.append(f"Archive not found: {archive_path}")
                return result

            # Create target directory
            target_path.mkdir(parents=True, exist_ok=True)

            # Extract archive
            mode = "r:gz" if self.config.compression == CompressionType.GZIP else "r"

            with tarfile.open(archive_path, mode) as tar:
                for member in tar.getmembers():
                    target_file = target_path / member.name

                    if target_file.exists() and not overwrite:
                        continue

                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    tar.extract(member, target_path)

                    result.files_restored += 1
                    result.bytes_restored += member.size

            # Handle incremental restore
            if manifest.backup_type == BackupType.INCREMENTAL and manifest.parent_backup_id:
                parent_result = self.restore(manifest.parent_backup_id, target, overwrite)
                result.files_restored += parent_result.files_restored
                result.bytes_restored += parent_result.bytes_restored

            result.duration_ms = (time.time() - start_time) * 1000

            self._stats["total_restores"] += 1
            self._stats["total_bytes_restored"] += result.bytes_restored

        except Exception as e:
            result.success = False
            result.errors.append(str(e))

        return result

    def verify(self, backup_id: str) -> VerificationResult:
        """
        Verify a backup's integrity.

        Args:
            backup_id: Backup ID to verify

        Returns:
            VerificationResult with verification details
        """
        manifest = self._manifests.get(backup_id)
        if not manifest:
            return VerificationResult(
                valid=False,
                backup_id=backup_id,
                errors=["Backup not found"],
            )

        result = VerificationResult(
            valid=True,
            backup_id=backup_id,
        )

        archive_path = Path(manifest.backup_path)

        # Check archive exists
        if not archive_path.exists():
            result.valid = False
            result.files_missing = 1
            result.errors.append("Archive file missing")
            return result

        # Verify checksum
        current_checksum = self._calculate_checksum(archive_path)
        if current_checksum != manifest.checksum:
            result.valid = False
            result.files_corrupted = 1
            result.errors.append(f"Checksum mismatch: expected {manifest.checksum}, got {current_checksum}")
            return result

        # Verify archive contents
        try:
            mode = "r:gz" if self.config.compression == CompressionType.GZIP else "r"

            with tarfile.open(archive_path, mode) as tar:
                for member in tar.getmembers():
                    result.files_verified += 1

        except Exception as e:
            result.valid = False
            result.errors.append(f"Archive error: {e}")

        # Update manifest status
        if result.valid:
            manifest.status = BackupStatus.VERIFIED
        else:
            manifest.status = BackupStatus.CORRUPTED

        self._emit_event("research.backup.verify", result.to_dict())

        return result

    def list_backups(
        self,
        backup_type: Optional[BackupType] = None,
        status: Optional[BackupStatus] = None,
        limit: int = 100,
    ) -> List[BackupManifest]:
        """
        List available backups.

        Args:
            backup_type: Filter by type
            status: Filter by status
            limit: Maximum number to return

        Returns:
            List of BackupManifest
        """
        backups = list(self._manifests.values())

        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]

        if status:
            backups = [b for b in backups if b.status == status]

        # Sort by creation time, newest first
        backups.sort(key=lambda b: b.created_at, reverse=True)

        return backups[:limit]

    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup.

        Args:
            backup_id: Backup ID to delete

        Returns:
            True if deleted successfully
        """
        manifest = self._manifests.get(backup_id)
        if not manifest:
            return False

        try:
            # Delete backup directory
            backup_dir = Path(self.config.backup_dir) / backup_id
            if backup_dir.exists():
                shutil.rmtree(backup_dir)

            # Remove from cache
            del self._manifests[backup_id]

            return True

        except Exception:
            return False

    def get_backup(self, backup_id: str) -> Optional[BackupManifest]:
        """Get a backup manifest by ID."""
        return self._manifests.get(backup_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get backup statistics."""
        return {
            **self._stats,
            "active_backups": len(self._manifests),
            "total_size_bytes": sum(m.size_bytes for m in self._manifests.values()),
        }

    def _get_all_files(self, source: Path) -> List[Dict[str, Any]]:
        """Get all files in source directory."""
        files = []

        for path in source.rglob("*"):
            if path.is_file():
                stat = path.stat()
                files.append({
                    "path": str(path.relative_to(source)),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "checksum": self._calculate_file_checksum(path),
                })

        return files

    def _get_incremental_files(
        self,
        source: Path,
        parent_id: str,
    ) -> List[Dict[str, Any]]:
        """Get files changed since parent backup."""
        parent = self._manifests.get(parent_id)
        if not parent:
            return self._get_all_files(source)

        # Build parent file lookup
        parent_files = {f["path"]: f for f in parent.files}

        changed_files = []
        for path in source.rglob("*"):
            if path.is_file():
                relative_path = str(path.relative_to(source))
                stat = path.stat()

                # Check if file is new or modified
                parent_file = parent_files.get(relative_path)
                if not parent_file or stat.st_mtime > parent_file.get("mtime", 0):
                    changed_files.append({
                        "path": relative_path,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "checksum": self._calculate_file_checksum(path),
                    })

        return changed_files

    def _create_archive(
        self,
        source: Path,
        archive_path: Path,
        files: List[Dict[str, Any]],
        manifest: BackupManifest,
    ) -> None:
        """Create backup archive."""
        mode = "w:gz" if self.config.compression == CompressionType.GZIP else "w"

        with tarfile.open(archive_path, mode) as tar:
            for file_info in files:
                file_path = source / file_info["path"]
                if file_path.exists():
                    tar.add(file_path, arcname=file_info["path"])

        manifest.files = files

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate file checksum."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _calculate_file_checksum(self, path: Path) -> str:
        """Calculate file checksum (shorter for efficiency)."""
        hasher = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def _save_manifest(self, manifest: BackupManifest, backup_dir: Path) -> None:
        """Save manifest to backup directory."""
        manifest_path = backup_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    def _load_manifests(self) -> None:
        """Load all manifests from backup directory."""
        backup_dir = Path(self.config.backup_dir)

        for subdir in backup_dir.iterdir():
            if subdir.is_dir():
                manifest_path = subdir / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path) as f:
                            data = json.load(f)
                            manifest = BackupManifest.from_dict(data)
                            self._manifests[manifest.id] = manifest
                    except Exception:
                        pass

    def _cleanup_old_backups(self) -> None:
        """Remove old backups based on retention policy."""
        backups = self.list_backups(limit=1000)

        # Remove by age
        cutoff_time = time.time() - (self.config.retention_days * 86400)
        for backup in backups:
            if backup.created_at < cutoff_time:
                self.delete_backup(backup.id)

        # Remove by count
        if len(self._manifests) > self.config.max_backups:
            sorted_backups = sorted(
                self._manifests.values(),
                key=lambda b: b.created_at,
            )
            for backup in sorted_backups[:-self.config.max_backups]:
                self.delete_backup(backup.id)

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
            "kind": "backup",
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
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Backup System."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Backup System (Step 46)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create command
    create_parser = subparsers.add_parser("create", help="Create backup")
    create_parser.add_argument("source", help="Source directory")
    create_parser.add_argument("--name", help="Backup name")
    create_parser.add_argument("--type", choices=["full", "incremental"], default="full")
    create_parser.add_argument("--parent", help="Parent backup ID for incremental")

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore backup")
    restore_parser.add_argument("backup_id", help="Backup ID")
    restore_parser.add_argument("target", help="Target directory")
    restore_parser.add_argument("--overwrite", action="store_true")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify backup")
    verify_parser.add_argument("backup_id", help="Backup ID")

    # List command
    list_parser = subparsers.add_parser("list", help="List backups")
    list_parser.add_argument("--type", choices=["full", "incremental"])
    list_parser.add_argument("--json", action="store_true")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete backup")
    delete_parser.add_argument("backup_id", help="Backup ID")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run backup demo")

    args = parser.parse_args()

    backup = BackupManager()

    if args.command == "create":
        backup_type = BackupType.INCREMENTAL if args.type == "incremental" else BackupType.FULL
        manifest = backup.create_backup(
            source=args.source,
            name=args.name,
            backup_type=backup_type,
            parent_id=args.parent,
        )

        print(f"Backup created: {manifest.id}")
        print(f"  Name: {manifest.name}")
        print(f"  Type: {manifest.backup_type.value}")
        print(f"  Status: {manifest.status.value}")
        print(f"  Files: {manifest.file_count}")
        print(f"  Size: {manifest.size_bytes:,} bytes")

    elif args.command == "restore":
        result = backup.restore(args.backup_id, args.target, args.overwrite)

        print(f"Restore {'completed' if result.success else 'failed'}")
        print(f"  Files restored: {result.files_restored}")
        print(f"  Bytes restored: {result.bytes_restored:,}")
        print(f"  Duration: {result.duration_ms:.2f}ms")

        if result.errors:
            print("  Errors:")
            for error in result.errors:
                print(f"    - {error}")

        return 0 if result.success else 1

    elif args.command == "verify":
        result = backup.verify(args.backup_id)

        print(f"Verification {'passed' if result.valid else 'failed'}")
        print(f"  Files verified: {result.files_verified}")
        print(f"  Files missing: {result.files_missing}")
        print(f"  Files corrupted: {result.files_corrupted}")

        return 0 if result.valid else 1

    elif args.command == "list":
        backup_type = BackupType(args.type) if args.type else None
        backups = backup.list_backups(backup_type=backup_type)

        if args.json:
            print(json.dumps([b.to_dict() for b in backups], indent=2))
        else:
            print(f"Backups ({len(backups)}):")
            for b in backups:
                created = datetime.fromtimestamp(b.created_at).strftime("%Y-%m-%d %H:%M")
                print(f"  [{b.id}] {b.name} ({b.backup_type.value}) - {created} - {b.status.value}")

    elif args.command == "delete":
        if backup.delete_backup(args.backup_id):
            print(f"Backup {args.backup_id} deleted")
        else:
            print(f"Failed to delete backup {args.backup_id}")
            return 1

    elif args.command == "stats":
        stats = backup.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Backup Statistics:")
            print(f"  Total backups: {stats['total_backups']}")
            print(f"  Total restores: {stats['total_restores']}")
            print(f"  Active backups: {stats['active_backups']}")
            print(f"  Total backed up: {stats['total_bytes_backed_up']:,} bytes")
            print(f"  Total restored: {stats['total_bytes_restored']:,} bytes")

    elif args.command == "demo":
        print("Running backup demo...\n")

        import tempfile

        # Create temp source directory with test data
        with tempfile.TemporaryDirectory() as source_dir:
            # Create test files
            for i in range(5):
                file_path = Path(source_dir) / f"test_file_{i}.txt"
                file_path.write_text(f"Test content {i}\n" * 100)

            subdir = Path(source_dir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.txt").write_text("Nested content")

            print(f"Created test data in {source_dir}")

            # Create backup
            manifest = backup.create_backup(source_dir, name="demo_backup")
            print(f"\nBackup created: {manifest.id}")
            print(f"  Status: {manifest.status.value}")
            print(f"  Files: {manifest.file_count}")
            print(f"  Size: {manifest.size_bytes:,} bytes")
            print(f"  Checksum: {manifest.checksum[:16]}...")

            # Verify backup
            print("\nVerifying backup...")
            verify_result = backup.verify(manifest.id)
            print(f"  Valid: {verify_result.valid}")
            print(f"  Files verified: {verify_result.files_verified}")

            # Restore to new location
            with tempfile.TemporaryDirectory() as restore_dir:
                print(f"\nRestoring to {restore_dir}...")
                restore_result = backup.restore(manifest.id, restore_dir)
                print(f"  Success: {restore_result.success}")
                print(f"  Files restored: {restore_result.files_restored}")

                # List restored files
                restored_files = list(Path(restore_dir).rglob("*"))
                print(f"  Restored files: {[str(f.relative_to(restore_dir)) for f in restored_files if f.is_file()]}")

            # Cleanup
            backup.delete_backup(manifest.id)
            print("\nBackup deleted")

        print("\nDemo complete.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
