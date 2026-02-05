#!/usr/bin/env python3
"""
Monitor Backup System - Step 296

Backup and restore capabilities for the Monitor Agent.

PBTSO Phase: PLAN

Bus Topics:
- monitor.backup.started (emitted)
- monitor.backup.completed (emitted)
- monitor.backup.failed (emitted)
- monitor.restore.started (emitted)
- monitor.restore.completed (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
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
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class BackupStatus(Enum):
    """Backup status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BackupType(Enum):
    """Backup types."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"


class BackupFormat(Enum):
    """Backup formats."""
    TAR_GZ = "tar.gz"
    ZIP = "zip"
    RAW = "raw"


@dataclass
class BackupManifest:
    """Backup manifest.

    Attributes:
        backup_id: Unique backup ID
        backup_type: Type of backup
        version: Monitor version
        components: Backed up components
        files: List of files
        total_size: Total size in bytes
        checksum: Backup checksum
        created_at: Creation timestamp
    """
    backup_id: str
    backup_type: BackupType
    version: str
    components: List[str]
    files: List[str] = field(default_factory=list)
    total_size: int = 0
    checksum: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "type": self.backup_type.value,
            "version": self.version,
            "components": self.components,
            "files": self.files,
            "total_size": self.total_size,
            "checksum": self.checksum,
            "created_at": self.created_at,
        }


@dataclass
class BackupResult:
    """Result of a backup operation.

    Attributes:
        backup_id: Backup ID
        status: Backup status
        path: Backup path
        size_bytes: Backup size
        duration_ms: Backup duration
        error: Error message if failed
        manifest: Backup manifest
    """
    backup_id: str
    status: BackupStatus
    path: str = ""
    size_bytes: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None
    manifest: Optional[BackupManifest] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "status": self.status.value,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "manifest": self.manifest.to_dict() if self.manifest else None,
        }


@dataclass
class RestoreResult:
    """Result of a restore operation.

    Attributes:
        backup_id: Backup ID
        status: Restore status
        components_restored: Restored components
        duration_ms: Restore duration
        error: Error message if failed
    """
    backup_id: str
    status: BackupStatus
    components_restored: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backup_id": self.backup_id,
            "status": self.status.value,
            "components_restored": self.components_restored,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class BackupPolicy:
    """Backup policy configuration.

    Attributes:
        name: Policy name
        backup_type: Type of backup
        schedule_cron: Cron schedule
        retention_days: Days to retain backups
        components: Components to backup
        enabled: Whether policy is enabled
    """
    name: str
    backup_type: BackupType
    schedule_cron: str
    retention_days: int = 30
    components: List[str] = field(default_factory=lambda: ["all"])
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.backup_type.value,
            "schedule": self.schedule_cron,
            "retention_days": self.retention_days,
            "components": self.components,
            "enabled": self.enabled,
        }


class MonitorBackupSystem:
    """
    Backup system for the Monitor Agent.

    Provides:
    - Full and incremental backups
    - Backup compression
    - Backup verification
    - Automated backup scheduling
    - Point-in-time restore

    Example:
        backup = MonitorBackupSystem()

        # Create backup
        result = await backup.create_backup(
            backup_type=BackupType.FULL,
            components=["metrics", "alerts", "config"],
        )

        # List backups
        backups = backup.list_backups()

        # Restore backup
        restore_result = await backup.restore(result.backup_id)
    """

    BUS_TOPICS = {
        "backup_started": "monitor.backup.started",
        "backup_completed": "monitor.backup.completed",
        "backup_failed": "monitor.backup.failed",
        "restore_started": "monitor.restore.started",
        "restore_completed": "monitor.restore.completed",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    # Default components
    DEFAULT_COMPONENTS = ["metrics", "alerts", "dashboards", "config", "logs"]

    def __init__(
        self,
        data_dir: Optional[str] = None,
        backup_dir: Optional[str] = None,
        bus_dir: Optional[str] = None,
    ):
        """Initialize backup system.

        Args:
            data_dir: Data directory
            backup_dir: Backup directory
            bus_dir: Bus directory
        """
        self._last_heartbeat = time.time()

        # Directories
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._data_dir = data_dir or os.path.join(pluribus_root, ".pluribus", "monitor")
        self._backup_dir = backup_dir or os.path.join(self._data_dir, "backups")
        Path(self._backup_dir).mkdir(parents=True, exist_ok=True)

        # Bus path
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Backup registry
        self._backups: Dict[str, BackupManifest] = {}
        self._policies: Dict[str, BackupPolicy] = {}
        self._history: List[BackupResult] = []

        # Load existing backups
        self._scan_backups()

    async def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        components: Optional[List[str]] = None,
        format: BackupFormat = BackupFormat.TAR_GZ,
        compress: bool = True,
    ) -> BackupResult:
        """Create a backup.

        Args:
            backup_type: Type of backup
            components: Components to backup
            format: Backup format
            compress: Whether to compress

        Returns:
            Backup result
        """
        backup_id = f"backup-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
        components = components or self.DEFAULT_COMPONENTS

        result = BackupResult(
            backup_id=backup_id,
            status=BackupStatus.RUNNING,
        )

        self._emit_bus_event(
            self.BUS_TOPICS["backup_started"],
            {
                "backup_id": backup_id,
                "type": backup_type.value,
                "components": components,
            },
        )

        start_time = time.time()

        try:
            # Create manifest
            manifest = BackupManifest(
                backup_id=backup_id,
                backup_type=backup_type,
                version="0.3.0",
                components=components,
            )

            # Create backup directory
            backup_path = os.path.join(self._backup_dir, backup_id)
            Path(backup_path).mkdir(parents=True, exist_ok=True)

            # Backup each component
            for component in components:
                component_files = await self._backup_component(
                    component,
                    backup_path,
                    backup_type,
                )
                manifest.files.extend(component_files)

            # Save manifest
            manifest_path = os.path.join(backup_path, "manifest.json")
            with open(manifest_path, "w") as f:
                json.dump(manifest.to_dict(), f, indent=2)

            # Create archive
            if format == BackupFormat.TAR_GZ:
                archive_path = f"{backup_path}.tar.gz"
                with tarfile.open(archive_path, "w:gz" if compress else "w") as tar:
                    tar.add(backup_path, arcname=backup_id)

                # Calculate checksum
                manifest.checksum = self._calculate_checksum(archive_path)
                manifest.total_size = os.path.getsize(archive_path)

                # Clean up temporary directory
                shutil.rmtree(backup_path)

                result.path = archive_path
                result.size_bytes = manifest.total_size
            else:
                result.path = backup_path
                result.size_bytes = self._get_dir_size(backup_path)

            result.status = BackupStatus.COMPLETED
            result.manifest = manifest
            result.duration_ms = (time.time() - start_time) * 1000

            self._backups[backup_id] = manifest

            self._emit_bus_event(
                self.BUS_TOPICS["backup_completed"],
                {
                    "backup_id": backup_id,
                    "size_bytes": result.size_bytes,
                    "duration_ms": result.duration_ms,
                },
            )

        except Exception as e:
            result.status = BackupStatus.FAILED
            result.error = str(e)
            result.duration_ms = (time.time() - start_time) * 1000

            self._emit_bus_event(
                self.BUS_TOPICS["backup_failed"],
                {
                    "backup_id": backup_id,
                    "error": str(e),
                },
                level="error",
            )

        self._history.append(result)
        return result

    async def restore(
        self,
        backup_id: str,
        components: Optional[List[str]] = None,
        verify: bool = True,
    ) -> RestoreResult:
        """Restore from a backup.

        Args:
            backup_id: Backup ID
            components: Components to restore (None = all)
            verify: Verify backup before restore

        Returns:
            Restore result
        """
        manifest = self._backups.get(backup_id)
        if not manifest:
            return RestoreResult(
                backup_id=backup_id,
                status=BackupStatus.FAILED,
                error="Backup not found",
            )

        result = RestoreResult(
            backup_id=backup_id,
            status=BackupStatus.RUNNING,
        )

        self._emit_bus_event(
            self.BUS_TOPICS["restore_started"],
            {"backup_id": backup_id},
        )

        start_time = time.time()

        try:
            # Find backup archive
            archive_path = os.path.join(self._backup_dir, f"{backup_id}.tar.gz")
            if not os.path.exists(archive_path):
                raise FileNotFoundError(f"Backup archive not found: {archive_path}")

            # Verify checksum
            if verify and manifest.checksum:
                actual_checksum = self._calculate_checksum(archive_path)
                if actual_checksum != manifest.checksum:
                    raise ValueError("Backup checksum mismatch")

            # Extract to temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(temp_dir)

                extracted_path = os.path.join(temp_dir, backup_id)

                # Restore components
                restore_components = components or manifest.components
                for component in restore_components:
                    if component in manifest.components:
                        await self._restore_component(component, extracted_path)
                        result.components_restored.append(component)

            result.status = BackupStatus.COMPLETED
            result.duration_ms = (time.time() - start_time) * 1000

            self._emit_bus_event(
                self.BUS_TOPICS["restore_completed"],
                {
                    "backup_id": backup_id,
                    "components": result.components_restored,
                    "duration_ms": result.duration_ms,
                },
            )

        except Exception as e:
            result.status = BackupStatus.FAILED
            result.error = str(e)
            result.duration_ms = (time.time() - start_time) * 1000

        return result

    def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup.

        Args:
            backup_id: Backup ID

        Returns:
            True if deleted
        """
        if backup_id not in self._backups:
            return False

        # Delete archive
        archive_path = os.path.join(self._backup_dir, f"{backup_id}.tar.gz")
        if os.path.exists(archive_path):
            os.remove(archive_path)

        # Delete directory if exists
        dir_path = os.path.join(self._backup_dir, backup_id)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        del self._backups[backup_id]
        return True

    def verify_backup(self, backup_id: str) -> bool:
        """Verify a backup's integrity.

        Args:
            backup_id: Backup ID

        Returns:
            True if valid
        """
        manifest = self._backups.get(backup_id)
        if not manifest:
            return False

        archive_path = os.path.join(self._backup_dir, f"{backup_id}.tar.gz")
        if not os.path.exists(archive_path):
            return False

        if not manifest.checksum:
            return True

        actual_checksum = self._calculate_checksum(archive_path)
        return actual_checksum == manifest.checksum

    def list_backups(
        self,
        backup_type: Optional[BackupType] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List available backups.

        Args:
            backup_type: Filter by type
            limit: Maximum results

        Returns:
            Backup list
        """
        backups = list(self._backups.values())
        if backup_type:
            backups = [b for b in backups if b.backup_type == backup_type]

        backups.sort(key=lambda b: b.created_at, reverse=True)

        return [b.to_dict() for b in backups[:limit]]

    def get_backup(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Get backup details.

        Args:
            backup_id: Backup ID

        Returns:
            Backup details or None
        """
        manifest = self._backups.get(backup_id)
        return manifest.to_dict() if manifest else None

    def register_policy(self, policy: BackupPolicy) -> None:
        """Register a backup policy.

        Args:
            policy: Backup policy
        """
        self._policies[policy.name] = policy

    def list_policies(self) -> List[Dict[str, Any]]:
        """List backup policies.

        Returns:
            Policy list
        """
        return [p.to_dict() for p in self._policies.values()]

    def cleanup_old_backups(self, retention_days: int = 30) -> int:
        """Clean up old backups.

        Args:
            retention_days: Days to retain

        Returns:
            Number of backups deleted
        """
        cutoff = time.time() - (retention_days * 86400)
        deleted = 0

        for backup_id, manifest in list(self._backups.items()):
            if manifest.created_at < cutoff:
                if self.delete_backup(backup_id):
                    deleted += 1

        return deleted

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get backup/restore history.

        Args:
            limit: Maximum results

        Returns:
            History list
        """
        return [h.to_dict() for h in reversed(self._history[-limit:])]

    def get_statistics(self) -> Dict[str, Any]:
        """Get backup statistics.

        Returns:
            Statistics
        """
        total_size = sum(
            os.path.getsize(os.path.join(self._backup_dir, f"{b.backup_id}.tar.gz"))
            for b in self._backups.values()
            if os.path.exists(os.path.join(self._backup_dir, f"{b.backup_id}.tar.gz"))
        )

        return {
            "total_backups": len(self._backups),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "policies": len(self._policies),
            "by_type": {
                t.value: sum(1 for b in self._backups.values() if b.backup_type == t)
                for t in BackupType
            },
        }

    async def _backup_component(
        self,
        component: str,
        backup_path: str,
        backup_type: BackupType,
    ) -> List[str]:
        """Backup a single component."""
        files = []
        component_src = os.path.join(self._data_dir, component)
        component_dst = os.path.join(backup_path, component)

        if os.path.exists(component_src):
            if os.path.isdir(component_src):
                shutil.copytree(component_src, component_dst)
                for root, _, filenames in os.walk(component_dst):
                    for filename in filenames:
                        files.append(os.path.join(root, filename))
            else:
                Path(component_dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(component_src, component_dst)
                files.append(component_dst)

        return files

    async def _restore_component(
        self,
        component: str,
        backup_path: str,
    ) -> None:
        """Restore a single component."""
        component_src = os.path.join(backup_path, component)
        component_dst = os.path.join(self._data_dir, component)

        if os.path.exists(component_src):
            # Remove existing
            if os.path.exists(component_dst):
                if os.path.isdir(component_dst):
                    shutil.rmtree(component_dst)
                else:
                    os.remove(component_dst)

            # Restore
            if os.path.isdir(component_src):
                shutil.copytree(component_src, component_dst)
            else:
                Path(component_dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(component_src, component_dst)

    def _scan_backups(self) -> None:
        """Scan backup directory for existing backups."""
        for filename in os.listdir(self._backup_dir):
            if filename.endswith(".tar.gz"):
                backup_id = filename[:-7]  # Remove .tar.gz
                manifest_path = None

                # Try to extract manifest
                archive_path = os.path.join(self._backup_dir, filename)
                try:
                    with tarfile.open(archive_path, "r:gz") as tar:
                        manifest_file = tar.extractfile(f"{backup_id}/manifest.json")
                        if manifest_file:
                            manifest_data = json.load(manifest_file)
                            manifest = BackupManifest(
                                backup_id=manifest_data["backup_id"],
                                backup_type=BackupType(manifest_data["type"]),
                                version=manifest_data["version"],
                                components=manifest_data["components"],
                                files=manifest_data.get("files", []),
                                total_size=manifest_data.get("total_size", 0),
                                checksum=manifest_data.get("checksum", ""),
                                created_at=manifest_data.get("created_at", 0),
                            )
                            self._backups[backup_id] = manifest
                except Exception:
                    pass

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_dir_size(self, path: str) -> int:
        """Get directory size in bytes."""
        total = 0
        for root, _, files in os.walk(path):
            for filename in files:
                total += os.path.getsize(os.path.join(root, filename))
        return total

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
                "component": "monitor_backup",
                "status": "healthy",
                "backups": len(self._backups),
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
            "actor": "monitor-backup",
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
_backup: Optional[MonitorBackupSystem] = None


def get_backup() -> MonitorBackupSystem:
    """Get or create the backup system singleton.

    Returns:
        MonitorBackupSystem instance
    """
    global _backup
    if _backup is None:
        _backup = MonitorBackupSystem()
    return _backup


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Backup System (Step 296)")
    parser.add_argument("--create", action="store_true", help="Create backup")
    parser.add_argument("--type", choices=["full", "incremental"], default="full", help="Backup type")
    parser.add_argument("--list", action="store_true", help="List backups")
    parser.add_argument("--restore", metavar="BACKUP_ID", help="Restore backup")
    parser.add_argument("--verify", metavar="BACKUP_ID", help="Verify backup")
    parser.add_argument("--delete", metavar="BACKUP_ID", help="Delete backup")
    parser.add_argument("--cleanup", type=int, metavar="DAYS", help="Clean up old backups")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    backup = get_backup()

    if args.create:
        backup_type = BackupType(args.type)
        result = asyncio.run(backup.create_backup(backup_type=backup_type))
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "success" if result.status == BackupStatus.COMPLETED else "failed"
            print(f"Backup: {status}")
            print(f"  ID: {result.backup_id}")
            print(f"  Path: {result.path}")
            print(f"  Size: {result.size_bytes} bytes")

    if args.list:
        backups = backup.list_backups()
        if args.json:
            print(json.dumps(backups, indent=2))
        else:
            print("Backups:")
            for b in backups:
                print(f"  {b['backup_id']}: {b['type']} ({b['total_size']} bytes)")

    if args.restore:
        result = asyncio.run(backup.restore(args.restore))
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "success" if result.status == BackupStatus.COMPLETED else "failed"
            print(f"Restore: {status}")
            print(f"  Components: {result.components_restored}")

    if args.verify:
        valid = backup.verify_backup(args.verify)
        if args.json:
            print(json.dumps({"backup_id": args.verify, "valid": valid}))
        else:
            print(f"Backup {args.verify}: {'valid' if valid else 'invalid'}")

    if args.delete:
        deleted = backup.delete_backup(args.delete)
        if args.json:
            print(json.dumps({"backup_id": args.delete, "deleted": deleted}))
        else:
            print(f"Delete: {'success' if deleted else 'failed'}")

    if args.cleanup:
        deleted = backup.cleanup_old_backups(args.cleanup)
        if args.json:
            print(json.dumps({"deleted": deleted}))
        else:
            print(f"Cleaned up {deleted} old backups")

    if args.stats:
        stats = backup.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Backup Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
