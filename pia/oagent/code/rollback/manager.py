#!/usr/bin/env python3
"""
manager.py - Rollback Manager (Step 64)

PBTSO Phase: ITERATE, VERIFY

Provides:
- Safe edit rollback to previous states
- Point-in-time recovery
- Selective file rollback
- Batch rollback operations
- State snapshot management

Bus Topics:
- code.rollback.create
- code.rollback.execute
- code.rollback.complete

Protocol: DKIN v30, PAIP v16
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# Types
# =============================================================================

class RollbackScope(Enum):
    """Scope of rollback operation."""
    FILE = "file"          # Single file
    DIRECTORY = "directory" # All files in directory
    BATCH = "batch"        # Specific batch of edits
    CHECKPOINT = "checkpoint" # To named checkpoint
    TIME = "time"          # To point in time


class RollbackStatus(Enum):
    """Status of rollback point."""
    ACTIVE = "active"       # Can be rolled back to
    APPLIED = "applied"     # Rollback was executed
    EXPIRED = "expired"     # Too old, cleaned up
    INVALIDATED = "invalidated" # Superseded by newer rollback


@dataclass
class FileSnapshot:
    """Snapshot of a single file's state."""
    path: str
    content_hash: str
    size: int
    mtime: float
    backup_path: Optional[str] = None
    existed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "content_hash": self.content_hash,
            "size": self.size,
            "mtime": self.mtime,
            "existed": self.existed,
        }


@dataclass
class RollbackPoint:
    """
    A point-in-time snapshot for rollback.

    Contains snapshots of all affected files at a specific moment.
    """
    id: str
    name: str
    description: str
    files: List[FileSnapshot]
    created_at: float
    scope: RollbackScope = RollbackScope.BATCH
    status: RollbackStatus = RollbackStatus.ACTIVE
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    applied_at: Optional[float] = None
    parent_id: Optional[str] = None  # For chained rollbacks

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def is_active(self) -> bool:
        return self.status == RollbackStatus.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "files": [f.to_dict() for f in self.files],
            "file_count": self.file_count,
            "total_size": self.total_size,
            "created_at": self.created_at,
            "scope": self.scope.value,
            "status": self.status.value,
            "tags": self.tags,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
        }


@dataclass
class RollbackResult:
    """Result of a rollback operation."""
    id: str
    rollback_point_id: str
    success: bool
    files_restored: int
    files_failed: int
    errors: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rollback_point_id": self.rollback_point_id,
            "success": self.success,
            "files_restored": self.files_restored,
            "files_failed": self.files_failed,
            "errors": self.errors,
            "elapsed_ms": self.elapsed_ms,
        }


# =============================================================================
# Rollback Manager
# =============================================================================

class RollbackManager:
    """
    Manage safe edit rollback operations.

    PBTSO Phase: ITERATE, VERIFY

    Features:
    - Create rollback points before edits
    - Restore files to previous states
    - Selective file restoration
    - Batch rollback operations
    - Automatic cleanup of old rollbacks

    Usage:
        manager = RollbackManager(working_dir)
        point = manager.create_point("before_refactor", files)
        # ... make changes ...
        result = manager.rollback(point.id)
    """

    BUS_TOPICS = {
        "create": "code.rollback.create",
        "execute": "code.rollback.execute",
        "complete": "code.rollback.complete",
    }

    def __init__(
        self,
        working_dir: Path,
        bus: Optional[Any] = None,
        storage_dir: Optional[Path] = None,
        max_rollback_points: int = 100,
        max_age_hours: int = 168,  # 1 week
    ):
        self.working_dir = Path(working_dir)
        self.bus = bus
        self.storage_dir = storage_dir or self.working_dir / ".pluribus" / "rollbacks"
        self.max_rollback_points = max_rollback_points
        self.max_age_hours = max_age_hours

        self._points: Dict[str, RollbackPoint] = {}
        self._ensure_storage()
        self._load_points()

    def _ensure_storage(self) -> None:
        """Ensure storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "snapshots").mkdir(exist_ok=True)
        (self.storage_dir / "metadata").mkdir(exist_ok=True)

    def _load_points(self) -> None:
        """Load existing rollback points from storage."""
        metadata_dir = self.storage_dir / "metadata"
        for meta_file in metadata_dir.glob("*.json"):
            try:
                with meta_file.open() as f:
                    data = json.load(f)

                files = [
                    FileSnapshot(
                        path=fd["path"],
                        content_hash=fd["content_hash"],
                        size=fd["size"],
                        mtime=fd["mtime"],
                        existed=fd.get("existed", True),
                    )
                    for fd in data["files"]
                ]

                point = RollbackPoint(
                    id=data["id"],
                    name=data["name"],
                    description=data["description"],
                    files=files,
                    created_at=data["created_at"],
                    scope=RollbackScope(data.get("scope", "batch")),
                    status=RollbackStatus(data.get("status", "active")),
                    tags=data.get("tags", []),
                    metadata=data.get("metadata", {}),
                    parent_id=data.get("parent_id"),
                )

                self._points[point.id] = point

            except Exception:
                pass  # Skip corrupted metadata files

    # =========================================================================
    # Point Creation
    # =========================================================================

    def create_point(
        self,
        name: str,
        files: List[str],
        description: str = "",
        scope: RollbackScope = RollbackScope.BATCH,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RollbackPoint:
        """
        Create a rollback point for specified files.

        Args:
            name: Human-readable name for the point
            files: List of file paths to snapshot
            description: Optional description
            scope: Scope of the rollback
            tags: Optional tags for organization
            metadata: Optional metadata

        Returns:
            Created RollbackPoint
        """
        point_id = f"rb-{uuid.uuid4().hex[:12]}"
        timestamp = time.time()

        # Create snapshots for each file
        snapshots: List[FileSnapshot] = []
        snapshot_dir = self.storage_dir / "snapshots" / point_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            full_path = self.working_dir / file_path
            snapshot = self._create_file_snapshot(full_path, file_path, snapshot_dir)
            if snapshot:
                snapshots.append(snapshot)

        point = RollbackPoint(
            id=point_id,
            name=name,
            description=description,
            files=snapshots,
            created_at=timestamp,
            scope=scope,
            tags=tags or [],
            metadata=metadata or {},
        )

        self._points[point.id] = point
        self._save_point(point)

        # Cleanup old points if needed
        self._cleanup_old_points()

        # Emit event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["create"],
                "kind": "rollback",
                "actor": "code-agent",
                "data": {
                    "point_id": point.id,
                    "name": name,
                    "file_count": len(snapshots),
                    "total_size": point.total_size,
                },
            })

        return point

    def _create_file_snapshot(
        self,
        full_path: Path,
        relative_path: str,
        snapshot_dir: Path,
    ) -> Optional[FileSnapshot]:
        """Create a snapshot of a single file."""
        if not full_path.exists():
            return FileSnapshot(
                path=relative_path,
                content_hash="",
                size=0,
                mtime=0,
                existed=False,
            )

        try:
            content = full_path.read_bytes()
            content_hash = hashlib.sha256(content).hexdigest()

            # Save backup
            backup_name = f"{relative_path.replace('/', '_')}_{content_hash[:8]}"
            backup_path = snapshot_dir / backup_name
            backup_path.write_bytes(content)

            stat = full_path.stat()

            return FileSnapshot(
                path=relative_path,
                content_hash=content_hash,
                size=stat.st_size,
                mtime=stat.st_mtime,
                backup_path=str(backup_path),
                existed=True,
            )

        except Exception:
            return None

    def _save_point(self, point: RollbackPoint) -> None:
        """Save rollback point metadata to storage."""
        meta_path = self.storage_dir / "metadata" / f"{point.id}.json"

        # Convert files to dicts with backup paths
        files_data = []
        for f in point.files:
            fd = f.to_dict()
            fd["backup_path"] = f.backup_path
            files_data.append(fd)

        data = {
            "id": point.id,
            "name": point.name,
            "description": point.description,
            "files": files_data,
            "created_at": point.created_at,
            "scope": point.scope.value,
            "status": point.status.value,
            "tags": point.tags,
            "metadata": point.metadata,
            "parent_id": point.parent_id,
        }

        with meta_path.open("w") as f:
            json.dump(data, f, indent=2)

    # =========================================================================
    # Rollback Execution
    # =========================================================================

    def rollback(
        self,
        point_id: str,
        files: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> RollbackResult:
        """
        Execute rollback to a specific point.

        Args:
            point_id: ID of the rollback point
            files: Optional list of specific files to rollback
            dry_run: If True, just validate without restoring

        Returns:
            RollbackResult with operation details
        """
        start_time = time.time()

        point = self._points.get(point_id)
        if not point:
            return RollbackResult(
                id=f"rbr-{uuid.uuid4().hex[:8]}",
                rollback_point_id=point_id,
                success=False,
                files_restored=0,
                files_failed=0,
                errors=[f"Rollback point not found: {point_id}"],
            )

        if not point.is_active:
            return RollbackResult(
                id=f"rbr-{uuid.uuid4().hex[:8]}",
                rollback_point_id=point_id,
                success=False,
                files_restored=0,
                files_failed=0,
                errors=[f"Rollback point is not active: {point.status.value}"],
            )

        # Emit start event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["execute"],
                "kind": "rollback",
                "actor": "code-agent",
                "data": {
                    "point_id": point_id,
                    "dry_run": dry_run,
                    "file_count": len(point.files),
                },
            })

        # Filter files if specified
        snapshots_to_restore = point.files
        if files:
            file_set = set(files)
            snapshots_to_restore = [s for s in point.files if s.path in file_set]

        # Create a new point before rollback (for forward rollback)
        if not dry_run:
            current_files = [s.path for s in snapshots_to_restore]
            self.create_point(
                name=f"pre_rollback_{point.name}",
                files=current_files,
                description=f"State before rolling back to {point.name}",
                metadata={"rollback_from": point_id},
            )

        files_restored = 0
        files_failed = 0
        errors: List[str] = []

        for snapshot in snapshots_to_restore:
            try:
                if not dry_run:
                    self._restore_file(snapshot)
                files_restored += 1
            except Exception as e:
                files_failed += 1
                errors.append(f"{snapshot.path}: {e}")

        # Update point status
        if not dry_run:
            point.status = RollbackStatus.APPLIED
            point.applied_at = time.time()
            self._save_point(point)

        result = RollbackResult(
            id=f"rbr-{uuid.uuid4().hex[:8]}",
            rollback_point_id=point_id,
            success=files_failed == 0,
            files_restored=files_restored,
            files_failed=files_failed,
            errors=errors,
            elapsed_ms=(time.time() - start_time) * 1000,
        )

        # Emit completion event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["complete"],
                "kind": "rollback",
                "actor": "code-agent",
                "data": result.to_dict(),
            })

        return result

    def _restore_file(self, snapshot: FileSnapshot) -> None:
        """Restore a single file from snapshot."""
        target_path = self.working_dir / snapshot.path

        if not snapshot.existed:
            # File didn't exist - delete it
            if target_path.exists():
                target_path.unlink()
            return

        if not snapshot.backup_path:
            raise ValueError(f"No backup path for {snapshot.path}")

        backup_path = Path(snapshot.backup_path)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")

        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy backup to target
        shutil.copy2(backup_path, target_path)

    # =========================================================================
    # Point Management
    # =========================================================================

    def get_point(self, point_id: str) -> Optional[RollbackPoint]:
        """Get a rollback point by ID."""
        return self._points.get(point_id)

    def list_points(
        self,
        status: Optional[RollbackStatus] = None,
        tags: Optional[List[str]] = None,
        scope: Optional[RollbackScope] = None,
    ) -> List[RollbackPoint]:
        """List rollback points with optional filters."""
        points = list(self._points.values())

        if status:
            points = [p for p in points if p.status == status]

        if tags:
            tag_set = set(tags)
            points = [p for p in points if tag_set & set(p.tags)]

        if scope:
            points = [p for p in points if p.scope == scope]

        # Sort by creation time, newest first
        points.sort(key=lambda p: p.created_at, reverse=True)

        return points

    def find_point_by_name(self, name: str) -> Optional[RollbackPoint]:
        """Find a rollback point by name."""
        for point in self._points.values():
            if point.name == name:
                return point
        return None

    def delete_point(self, point_id: str) -> bool:
        """Delete a rollback point and its snapshots."""
        point = self._points.get(point_id)
        if not point:
            return False

        # Delete snapshot directory
        snapshot_dir = self.storage_dir / "snapshots" / point_id
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)

        # Delete metadata
        meta_path = self.storage_dir / "metadata" / f"{point_id}.json"
        if meta_path.exists():
            meta_path.unlink()

        del self._points[point_id]
        return True

    def invalidate_point(self, point_id: str) -> bool:
        """Mark a rollback point as invalidated."""
        point = self._points.get(point_id)
        if not point:
            return False

        point.status = RollbackStatus.INVALIDATED
        self._save_point(point)
        return True

    # =========================================================================
    # Cleanup
    # =========================================================================

    def _cleanup_old_points(self) -> None:
        """Remove old rollback points that exceed limits."""
        # Clean up by age
        cutoff = time.time() - (self.max_age_hours * 3600)
        for point_id, point in list(self._points.items()):
            if point.created_at < cutoff and point.status == RollbackStatus.ACTIVE:
                point.status = RollbackStatus.EXPIRED
                self._save_point(point)

        # Clean up by count
        active_points = [p for p in self._points.values() if p.is_active]
        if len(active_points) > self.max_rollback_points:
            # Sort by age and remove oldest
            active_points.sort(key=lambda p: p.created_at)
            for point in active_points[:-self.max_rollback_points]:
                self.delete_point(point.id)

    def cleanup_expired(self, force: bool = False) -> int:
        """Clean up expired and invalidated points."""
        deleted = 0
        for point_id, point in list(self._points.items()):
            if point.status in (RollbackStatus.EXPIRED, RollbackStatus.INVALIDATED):
                if self.delete_point(point_id):
                    deleted += 1
            elif force and point.status == RollbackStatus.APPLIED:
                if self.delete_point(point_id):
                    deleted += 1
        return deleted

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_file_history(self, file_path: str) -> List[FileSnapshot]:
        """Get history of a file across all rollback points."""
        history: List[Tuple[float, FileSnapshot]] = []

        for point in self._points.values():
            for snapshot in point.files:
                if snapshot.path == file_path:
                    history.append((point.created_at, snapshot))

        history.sort(key=lambda x: x[0], reverse=True)
        return [h[1] for h in history]

    def diff_points(
        self,
        point_a_id: str,
        point_b_id: str,
    ) -> Dict[str, Any]:
        """Compare two rollback points."""
        point_a = self._points.get(point_a_id)
        point_b = self._points.get(point_b_id)

        if not point_a or not point_b:
            return {"error": "Point not found"}

        files_a = {f.path: f for f in point_a.files}
        files_b = {f.path: f for f in point_b.files}

        added = [p for p in files_b if p not in files_a]
        removed = [p for p in files_a if p not in files_b]
        modified = [
            p for p in files_a
            if p in files_b and files_a[p].content_hash != files_b[p].content_hash
        ]
        unchanged = [
            p for p in files_a
            if p in files_b and files_a[p].content_hash == files_b[p].content_hash
        ]

        return {
            "added": added,
            "removed": removed,
            "modified": modified,
            "unchanged": unchanged,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = 0
        for point in self._points.values():
            total_size += point.total_size

        return {
            "total_points": len(self._points),
            "active_points": len([p for p in self._points.values() if p.is_active]),
            "total_size_bytes": total_size,
            "storage_dir": str(self.storage_dir),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Rollback Manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Rollback Manager (Step 64)")
    parser.add_argument("--working-dir", default=".", help="Working directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # create command
    create_parser = subparsers.add_parser("create", help="Create rollback point")
    create_parser.add_argument("name", help="Point name")
    create_parser.add_argument("files", nargs="+", help="Files to snapshot")
    create_parser.add_argument("--description", default="", help="Description")
    create_parser.add_argument("--tags", nargs="*", help="Tags")

    # rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Execute rollback")
    rollback_parser.add_argument("point_id", help="Rollback point ID")
    rollback_parser.add_argument("--files", nargs="*", help="Specific files to rollback")
    rollback_parser.add_argument("--dry-run", action="store_true", help="Dry run")

    # list command
    list_parser = subparsers.add_parser("list", help="List rollback points")
    list_parser.add_argument("--status", choices=["active", "applied", "expired"])
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete rollback point")
    delete_parser.add_argument("point_id", help="Point ID to delete")

    # cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup old points")
    cleanup_parser.add_argument("--force", action="store_true", help="Also delete applied points")

    # stats command
    subparsers.add_parser("stats", help="Show statistics")

    args = parser.parse_args()

    manager = RollbackManager(Path(args.working_dir))

    if args.command == "create":
        point = manager.create_point(
            name=args.name,
            files=args.files,
            description=args.description,
            tags=args.tags,
        )
        print(f"Created rollback point: {point.id}")
        print(f"  Name: {point.name}")
        print(f"  Files: {point.file_count}")
        print(f"  Size: {point.total_size} bytes")
        return 0

    elif args.command == "rollback":
        result = manager.rollback(args.point_id, args.files, args.dry_run)
        if result.success:
            print(f"Rollback {'would restore' if args.dry_run else 'restored'} {result.files_restored} files")
        else:
            print(f"Rollback failed: {result.errors}")
        return 0 if result.success else 1

    elif args.command == "list":
        status = RollbackStatus(args.status) if args.status else None
        points = manager.list_points(status=status)

        if args.json:
            print(json.dumps([p.to_dict() for p in points], indent=2))
        else:
            for p in points:
                status_str = f"[{p.status.value}]"
                print(f"{p.id} {status_str} {p.name} ({p.file_count} files)")

        return 0

    elif args.command == "delete":
        if manager.delete_point(args.point_id):
            print(f"Deleted rollback point: {args.point_id}")
        else:
            print(f"Point not found: {args.point_id}")
            return 1
        return 0

    elif args.command == "cleanup":
        deleted = manager.cleanup_expired(args.force)
        print(f"Cleaned up {deleted} rollback points")
        return 0

    elif args.command == "stats":
        stats = manager.get_stats()
        print(f"Total points: {stats['total_points']}")
        print(f"Active points: {stats['active_points']}")
        print(f"Total size: {stats['total_size_bytes']} bytes")
        print(f"Storage: {stats['storage_dir']}")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
