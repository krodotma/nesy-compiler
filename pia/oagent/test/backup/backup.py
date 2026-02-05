#!/usr/bin/env python3
"""
Step 146: Test Backup System

Backup and restore capabilities for the Test Agent.

PBTSO Phase: SEQUESTER, VERIFY
Bus Topics:
- test.backup.create (emits)
- test.backup.restore (emits)
- test.backup.verify (emits)

Dependencies: Steps 101-145 (Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ============================================================================
# Constants
# ============================================================================

class BackupStatus(Enum):
    """Backup status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"


class BackupType(Enum):
    """Backup types."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class CompressionType(Enum):
    """Compression types."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class BackupManifest:
    """
    Backup manifest.

    Attributes:
        backup_id: Unique backup ID
        created_at: Creation timestamp
        files: List of files in backup
        checksums: File checksums
        metadata: Additional metadata
    """
    backup_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    backup_type: BackupType = BackupType.FULL
    files: List[str] = field(default_factory=list)
    checksums: Dict[str, str] = field(default_factory=dict)
    total_size_bytes: int = 0
    file_count: int = 0
    compression: CompressionType = CompressionType.GZIP
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_backup: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "created_at": self.created_at,
            "created_at_iso": datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat(),
            "backup_type": self.backup_type.value,
            "files": self.files,
            "checksums": self.checksums,
            "total_size_bytes": self.total_size_bytes,
            "file_count": self.file_count,
            "compression": self.compression.value,
            "metadata": self.metadata,
            "parent_backup": self.parent_backup,
        }


@dataclass
class Backup:
    """
    A backup instance.

    Attributes:
        backup_id: Unique backup ID
        name: Backup name
        path: Backup file path
        manifest: Backup manifest
        status: Backup status
    """
    backup_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    path: Optional[Path] = None
    manifest: BackupManifest = field(default_factory=BackupManifest)
    status: BackupStatus = BackupStatus.PENDING
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "name": self.name,
            "path": str(self.path) if self.path else None,
            "manifest": self.manifest.to_dict(),
            "status": self.status.value,
            "error": self.error,
        }


@dataclass
class BackupResult:
    """
    Result of a backup operation.

    Attributes:
        backup: The backup
        status: Operation status
        started_at: Start timestamp
        completed_at: Completion timestamp
        files_backed_up: Number of files backed up
        bytes_written: Bytes written
        error: Error message if failed
    """
    backup: Backup
    status: BackupStatus = BackupStatus.PENDING
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    files_backed_up: int = 0
    bytes_written: int = 0
    error: Optional[str] = None

    @property
    def duration_s(self) -> float:
        if self.completed_at:
            return self.completed_at - self.started_at
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup.backup_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "files_backed_up": self.files_backed_up,
            "bytes_written": self.bytes_written,
            "error": self.error,
        }


@dataclass
class RestoreResult:
    """
    Result of a restore operation.

    Attributes:
        backup_id: Backup ID restored
        status: Operation status
        started_at: Start timestamp
        completed_at: Completion timestamp
        files_restored: Number of files restored
        error: Error message if failed
    """
    backup_id: str
    status: BackupStatus = BackupStatus.PENDING
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    files_restored: int = 0
    error: Optional[str] = None

    @property
    def duration_s(self) -> float:
        if self.completed_at:
            return self.completed_at - self.started_at
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "files_restored": self.files_restored,
            "error": self.error,
        }


@dataclass
class BackupConfig:
    """
    Configuration for backups.

    Attributes:
        backup_dir: Backup storage directory
        source_dirs: Directories to backup
        compression: Compression type
        retention_days: Backup retention in days
        max_backups: Maximum backups to keep
        verify_after_backup: Verify backup integrity
        include_patterns: File patterns to include
        exclude_patterns: File patterns to exclude
    """
    backup_dir: str = ".pluribus/test-agent/backups"
    source_dirs: List[str] = field(default_factory=lambda: [".pluribus/test-agent"])
    compression: CompressionType = CompressionType.GZIP
    retention_days: int = 30
    max_backups: int = 10
    verify_after_backup: bool = True
    include_patterns: List[str] = field(default_factory=lambda: ["*"])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "__pycache__", ".git", "*.log"
    ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backup_dir": self.backup_dir,
            "source_dirs": self.source_dirs,
            "compression": self.compression.value,
            "retention_days": self.retention_days,
            "max_backups": self.max_backups,
            "verify_after_backup": self.verify_after_backup,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class BackupBus:
    """Bus interface for backup with file locking."""

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
# Test Backup Manager
# ============================================================================

class TestBackupManager:
    """
    Backup manager for the Test Agent.

    Features:
    - Full and incremental backups
    - Compression support
    - Backup verification
    - Retention policies
    - Restore operations

    PBTSO Phase: SEQUESTER, VERIFY
    Bus Topics: test.backup.create, test.backup.restore, test.backup.verify
    """

    BUS_TOPICS = {
        "create": "test.backup.create",
        "restore": "test.backup.restore",
        "verify": "test.backup.verify",
    }

    def __init__(self, bus=None, config: Optional[BackupConfig] = None):
        """
        Initialize the backup manager.

        Args:
            bus: Optional bus instance
            config: Backup configuration
        """
        self.bus = bus or BackupBus()
        self.config = config or BackupConfig()
        self._backups: Dict[str, Backup] = {}

        # Create backup directory
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)

        # Load existing backups
        self._load_backups()

    def create_backup(
        self,
        name: Optional[str] = None,
        backup_type: BackupType = BackupType.FULL,
        source_dirs: Optional[List[str]] = None,
    ) -> BackupResult:
        """
        Create a new backup.

        Args:
            name: Backup name
            backup_type: Type of backup
            source_dirs: Directories to backup

        Returns:
            BackupResult with backup outcome
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = name or f"backup_{timestamp}"

        backup = Backup(
            name=backup_name,
            manifest=BackupManifest(
                backup_type=backup_type,
                compression=self.config.compression,
            ),
        )

        result = BackupResult(backup=backup, status=BackupStatus.IN_PROGRESS)

        self._emit_event("create", {
            "backup_id": backup.backup_id,
            "name": backup_name,
            "type": backup_type.value,
        })

        try:
            sources = source_dirs or self.config.source_dirs

            # Determine backup file path
            ext = ".tar.gz" if self.config.compression == CompressionType.GZIP else ".tar"
            backup_path = Path(self.config.backup_dir) / f"{backup_name}{ext}"
            backup.path = backup_path

            # Create tarball
            mode = "w:gz" if self.config.compression == CompressionType.GZIP else "w"

            with tarfile.open(backup_path, mode) as tar:
                for source_dir in sources:
                    source_path = Path(source_dir)
                    if not source_path.exists():
                        continue

                    for file_path in source_path.rglob("*"):
                        if self._should_exclude(file_path):
                            continue

                        if file_path.is_file():
                            # Calculate checksum
                            checksum = self._calculate_checksum(file_path)
                            rel_path = str(file_path)
                            backup.manifest.checksums[rel_path] = checksum
                            backup.manifest.files.append(rel_path)
                            backup.manifest.total_size_bytes += file_path.stat().st_size
                            backup.manifest.file_count += 1

                            tar.add(file_path, arcname=rel_path)
                            result.files_backed_up += 1

            # Get backup size
            result.bytes_written = backup_path.stat().st_size

            # Save manifest
            manifest_path = backup_path.with_suffix(".manifest.json")
            with open(manifest_path, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(backup.manifest.to_dict(), f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Verify if configured
            if self.config.verify_after_backup:
                verified = self.verify_backup(backup.backup_id)
                if verified:
                    backup.status = BackupStatus.VERIFIED
                else:
                    backup.status = BackupStatus.CORRUPTED
                    result.status = BackupStatus.CORRUPTED
            else:
                backup.status = BackupStatus.COMPLETED
                result.status = BackupStatus.COMPLETED

            # Store backup
            self._backups[backup.backup_id] = backup

            # Cleanup old backups
            self._cleanup_old_backups()

        except Exception as e:
            backup.status = BackupStatus.FAILED
            backup.error = str(e)
            result.status = BackupStatus.FAILED
            result.error = str(e)

        result.completed_at = time.time()
        return result

    def restore_backup(
        self,
        backup_id: str,
        target_dir: Optional[str] = None,
        files: Optional[List[str]] = None,
    ) -> RestoreResult:
        """
        Restore from a backup.

        Args:
            backup_id: Backup ID to restore
            target_dir: Target directory for restore
            files: Specific files to restore (all if None)

        Returns:
            RestoreResult with restore outcome
        """
        result = RestoreResult(backup_id=backup_id, status=BackupStatus.IN_PROGRESS)

        self._emit_event("restore", {
            "backup_id": backup_id,
            "target_dir": target_dir,
        })

        try:
            backup = self._backups.get(backup_id)
            if not backup or not backup.path:
                raise ValueError(f"Backup not found: {backup_id}")

            if not backup.path.exists():
                raise ValueError(f"Backup file not found: {backup.path}")

            # Determine extraction directory
            extract_dir = Path(target_dir) if target_dir else Path(".")

            # Open and extract tarball
            mode = "r:gz" if self.config.compression == CompressionType.GZIP else "r"

            with tarfile.open(backup.path, mode) as tar:
                if files:
                    # Extract specific files
                    for file_name in files:
                        try:
                            tar.extract(file_name, path=extract_dir)
                            result.files_restored += 1
                        except KeyError:
                            pass  # File not in backup
                else:
                    # Extract all files
                    tar.extractall(path=extract_dir)
                    result.files_restored = len(backup.manifest.files)

            result.status = BackupStatus.COMPLETED

        except Exception as e:
            result.status = BackupStatus.FAILED
            result.error = str(e)

        result.completed_at = time.time()
        return result

    def verify_backup(self, backup_id: str) -> bool:
        """
        Verify backup integrity.

        Args:
            backup_id: Backup ID to verify

        Returns:
            True if backup is valid
        """
        backup = self._backups.get(backup_id)
        if not backup or not backup.path:
            return False

        self._emit_event("verify", {
            "backup_id": backup_id,
        })

        try:
            # Verify tarball is readable
            mode = "r:gz" if self.config.compression == CompressionType.GZIP else "r"

            with tarfile.open(backup.path, mode) as tar:
                # Check all files exist
                tar_files = set(tar.getnames())

                for manifest_file in backup.manifest.files:
                    if manifest_file not in tar_files:
                        return False

                # Verify checksums for a sample of files
                with tempfile.TemporaryDirectory() as temp_dir:
                    sample_files = backup.manifest.files[:5]  # Check first 5 files

                    for file_name in sample_files:
                        try:
                            tar.extract(file_name, path=temp_dir)
                            extracted_path = Path(temp_dir) / file_name
                            checksum = self._calculate_checksum(extracted_path)

                            expected = backup.manifest.checksums.get(file_name)
                            if expected and checksum != expected:
                                return False
                        except (KeyError, IOError):
                            return False

            return True

        except (tarfile.TarError, IOError):
            return False

    def list_backups(self) -> List[Backup]:
        """List all backups."""
        return sorted(
            self._backups.values(),
            key=lambda b: b.manifest.created_at,
            reverse=True,
        )

    def get_backup(self, backup_id: str) -> Optional[Backup]:
        """Get a specific backup."""
        return self._backups.get(backup_id)

    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup."""
        backup = self._backups.get(backup_id)
        if not backup:
            return False

        try:
            if backup.path and backup.path.exists():
                backup.path.unlink()

            # Also delete manifest
            manifest_path = backup.path.with_suffix(".manifest.json")
            if manifest_path.exists():
                manifest_path.unlink()

            del self._backups[backup_id]
            return True

        except IOError:
            return False

    def _should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded."""
        path_str = str(path)

        for pattern in self.config.exclude_patterns:
            if pattern in path_str:
                return True
            if path.match(pattern):
                return True

        return False

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _load_backups(self) -> None:
        """Load existing backups from disk."""
        backup_dir = Path(self.config.backup_dir)

        for manifest_path in backup_dir.glob("*.manifest.json"):
            try:
                with open(manifest_path) as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        manifest_dict = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                # Find corresponding backup file
                backup_name = manifest_path.stem.replace(".manifest", "")
                backup_path = backup_dir / f"{backup_name}.tar.gz"

                if not backup_path.exists():
                    backup_path = backup_dir / f"{backup_name}.tar"

                if backup_path.exists():
                    manifest = BackupManifest(
                        backup_id=manifest_dict.get("backup_id", str(uuid.uuid4())),
                        created_at=manifest_dict.get("created_at", time.time()),
                        backup_type=BackupType(manifest_dict.get("backup_type", "full")),
                        files=manifest_dict.get("files", []),
                        checksums=manifest_dict.get("checksums", {}),
                        total_size_bytes=manifest_dict.get("total_size_bytes", 0),
                        file_count=manifest_dict.get("file_count", 0),
                        compression=CompressionType(manifest_dict.get("compression", "gzip")),
                    )

                    backup = Backup(
                        backup_id=manifest.backup_id,
                        name=backup_name,
                        path=backup_path,
                        manifest=manifest,
                        status=BackupStatus.COMPLETED,
                    )

                    self._backups[backup.backup_id] = backup

            except (IOError, json.JSONDecodeError):
                pass

    def _cleanup_old_backups(self) -> None:
        """Clean up old backups based on retention policy."""
        backups = self.list_backups()

        # Remove by count
        if len(backups) > self.config.max_backups:
            for backup in backups[self.config.max_backups:]:
                self.delete_backup(backup.backup_id)

        # Remove by age
        cutoff = time.time() - (self.config.retention_days * 86400)
        for backup in backups:
            if backup.manifest.created_at < cutoff:
                self.delete_backup(backup.backup_id)

    async def create_backup_async(
        self,
        name: Optional[str] = None,
        backup_type: BackupType = BackupType.FULL,
        source_dirs: Optional[List[str]] = None,
    ) -> BackupResult:
        """Async version of create_backup."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.create_backup, name, backup_type, source_dirs
        )

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.backup.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "backup",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Backup Manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Backup Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a backup")
    create_parser.add_argument("--name", help="Backup name")
    create_parser.add_argument("--type", choices=["full", "incremental"], default="full")
    create_parser.add_argument("--source", nargs="*", help="Source directories")

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore a backup")
    restore_parser.add_argument("backup_id", help="Backup ID to restore")
    restore_parser.add_argument("--target", help="Target directory")
    restore_parser.add_argument("--files", nargs="*", help="Specific files to restore")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a backup")
    verify_parser.add_argument("backup_id", help="Backup ID to verify")

    # List command
    list_parser = subparsers.add_parser("list", help="List backups")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a backup")
    delete_parser.add_argument("backup_id", help="Backup ID to delete")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/backups")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = BackupConfig(backup_dir=args.output)
    manager = TestBackupManager(config=config)

    if args.command == "create":
        backup_type = BackupType(args.type)
        result = manager.create_backup(
            name=args.name,
            backup_type=backup_type,
            source_dirs=args.source,
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "[OK]" if result.status in (BackupStatus.COMPLETED, BackupStatus.VERIFIED) else "[FAIL]"
            print(f"\n{status} Backup: {result.backup.name}")
            print(f"  ID: {result.backup.backup_id}")
            print(f"  Files: {result.files_backed_up}")
            print(f"  Size: {result.bytes_written / 1024:.2f} KB")
            print(f"  Duration: {result.duration_s:.2f}s")
            if result.error:
                print(f"  Error: {result.error}")

    elif args.command == "restore":
        result = manager.restore_backup(
            args.backup_id,
            target_dir=args.target,
            files=args.files,
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "[OK]" if result.status == BackupStatus.COMPLETED else "[FAIL]"
            print(f"\n{status} Restore: {args.backup_id}")
            print(f"  Files Restored: {result.files_restored}")
            print(f"  Duration: {result.duration_s:.2f}s")
            if result.error:
                print(f"  Error: {result.error}")

    elif args.command == "verify":
        valid = manager.verify_backup(args.backup_id)

        if args.json:
            print(json.dumps({"backup_id": args.backup_id, "valid": valid}))
        else:
            status = "[VALID]" if valid else "[INVALID]"
            print(f"{status} Backup: {args.backup_id}")

    elif args.command == "list":
        backups = manager.list_backups()

        if args.json:
            print(json.dumps([b.to_dict() for b in backups], indent=2))
        else:
            print(f"\nBackups ({len(backups)}):")
            for backup in backups:
                dt = datetime.fromtimestamp(backup.manifest.created_at)
                status = f"[{backup.status.value.upper()}]"
                print(f"\n  {status} {backup.name}")
                print(f"      ID: {backup.backup_id}")
                print(f"      Created: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"      Files: {backup.manifest.file_count}")
                print(f"      Size: {backup.manifest.total_size_bytes / 1024:.2f} KB")

    elif args.command == "delete":
        if manager.delete_backup(args.backup_id):
            print(f"Deleted backup: {args.backup_id}")
        else:
            print(f"Backup not found: {args.backup_id}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
