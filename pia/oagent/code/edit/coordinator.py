#!/usr/bin/env python3
"""
coordinator.py - Multi-File Edit Coordinator (Step 56)

PBTSO Phase: DISTRIBUTE, ITERATE

Provides:
- Atomic multi-file edit operations
- Backup and rollback capabilities
- Dry-run mode for validation
- Edit batching and ordering

Bus Topics:
- code.edit.batch
- code.edit.applied
- code.edit.rollback

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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# =============================================================================
# Types
# =============================================================================

class EditOperation(Enum):
    """Types of file edit operations."""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"
    APPEND = "append"
    INSERT = "insert"
    REPLACE = "replace"


class EditStatus(Enum):
    """Status of an edit operation."""
    PENDING = "pending"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"


@dataclass
class FileEdit:
    """
    Represents a single file edit operation.

    Supports:
    - CREATE: Create new file with content
    - MODIFY: Replace file content
    - DELETE: Delete file
    - RENAME: Rename/move file
    - APPEND: Append to file
    - INSERT: Insert at specific position
    - REPLACE: Replace specific content
    """
    path: str
    operation: EditOperation
    content: Optional[str] = None
    new_path: Optional[str] = None  # For RENAME
    position: Optional[int] = None  # For INSERT (line number)
    search: Optional[str] = None    # For REPLACE
    replace: Optional[str] = None   # For REPLACE
    status: EditStatus = EditStatus.PENDING
    error: Optional[str] = None
    backup_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "operation": self.operation.value,
            "content": self.content[:100] + "..." if self.content and len(self.content) > 100 else self.content,
            "new_path": self.new_path,
            "position": self.position,
            "status": self.status.value,
            "error": self.error,
        }


@dataclass
class EditBatch:
    """A batch of related file edits."""
    id: str
    edits: List[FileEdit]
    description: str = ""
    created_at: float = field(default_factory=time.time)
    applied_at: Optional[float] = None
    rolled_back_at: Optional[float] = None
    dry_run: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "edits": [e.to_dict() for e in self.edits],
            "description": self.description,
            "created_at": self.created_at,
            "applied_at": self.applied_at,
            "rolled_back_at": self.rolled_back_at,
            "dry_run": self.dry_run,
        }


# =============================================================================
# Multi-File Edit Coordinator
# =============================================================================

class MultiFileEditCoordinator:
    """
    Coordinate edits across multiple files atomically.

    PBTSO Phases: DISTRIBUTE, ITERATE

    Features:
    - Atomic batch operations (all-or-nothing)
    - Automatic backup before modifications
    - Rollback on failure
    - Dry-run mode for validation
    - Conflict detection

    Usage:
        coordinator = MultiFileEditCoordinator(working_dir)
        coordinator.add_edit(FileEdit(...))
        coordinator.add_edit(FileEdit(...))
        result = coordinator.apply_all()
    """

    def __init__(
        self,
        working_dir: Path,
        bus: Optional[Any] = None,
        backup_dir: Optional[Path] = None,
        max_batch_size: int = 50,
    ):
        self.working_dir = Path(working_dir)
        self.bus = bus
        self.backup_dir = backup_dir or self.working_dir / ".pluribus" / "backups"
        self.max_batch_size = max_batch_size

        self.pending_edits: List[FileEdit] = []
        self.batches: Dict[str, EditBatch] = {}
        self._current_backup_dir: Optional[Path] = None

    # =========================================================================
    # Edit Management
    # =========================================================================

    def add_edit(self, edit: FileEdit) -> None:
        """Add an edit to the pending queue."""
        if len(self.pending_edits) >= self.max_batch_size:
            raise ValueError(f"Max batch size ({self.max_batch_size}) exceeded")
        self.pending_edits.append(edit)

    def add_edits(self, edits: List[FileEdit]) -> None:
        """Add multiple edits to the pending queue."""
        for edit in edits:
            self.add_edit(edit)

    def clear_pending(self) -> None:
        """Clear all pending edits."""
        self.pending_edits = []

    def create_batch(self, description: str = "") -> EditBatch:
        """Create a batch from pending edits."""
        batch = EditBatch(
            id=f"batch-{uuid.uuid4().hex[:8]}",
            edits=list(self.pending_edits),
            description=description,
        )
        self.batches[batch.id] = batch
        self.pending_edits = []
        return batch

    # =========================================================================
    # Apply Operations
    # =========================================================================

    def apply_all(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Apply all pending edits atomically.

        Args:
            dry_run: If True, validate but don't apply changes

        Returns:
            Dict with results for each edit
        """
        if not self.pending_edits:
            return {"edits": [], "dry_run": dry_run, "message": "No pending edits"}

        batch = self.create_batch(f"Batch of {len(self.pending_edits)} edits")
        batch.dry_run = dry_run

        # Emit batch start
        if self.bus:
            self.bus.emit({
                "topic": "code.edit.batch",
                "kind": "edit",
                "actor": "code-agent",
                "data": {
                    "batch_id": batch.id,
                    "edit_count": len(batch.edits),
                    "dry_run": dry_run,
                },
            })

        # Create backup
        if not dry_run:
            self._create_backup(batch)

        results = []
        failed = False

        # Sort edits for proper ordering (deletes last, creates first)
        sorted_edits = self._sort_edits(batch.edits)

        for edit in sorted_edits:
            try:
                if not dry_run:
                    self._apply_edit(edit)
                    edit.status = EditStatus.APPLIED
                else:
                    # Validate only
                    self._validate_edit(edit)
                    edit.status = EditStatus.PENDING

                results.append({
                    "path": edit.path,
                    "operation": edit.operation.value,
                    "status": "success" if not dry_run else "validated",
                })

            except Exception as e:
                edit.status = EditStatus.FAILED
                edit.error = str(e)
                results.append({
                    "path": edit.path,
                    "operation": edit.operation.value,
                    "status": "failed",
                    "error": str(e),
                })
                failed = True

                if not dry_run:
                    # Rollback on failure
                    self._rollback(batch)
                    break

        batch.applied_at = time.time() if not failed and not dry_run else None

        # Emit completion
        if self.bus:
            topic = "code.edit.applied" if not failed else "code.edit.rollback"
            self.bus.emit({
                "topic": topic,
                "kind": "edit",
                "actor": "code-agent",
                "data": {
                    "batch_id": batch.id,
                    "success": not failed,
                    "dry_run": dry_run,
                    "results": results,
                },
            })

        return {
            "batch_id": batch.id,
            "edits": results,
            "dry_run": dry_run,
            "success": not failed,
        }

    def apply_batch(self, batch_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """Apply a specific batch by ID."""
        batch = self.batches.get(batch_id)
        if not batch:
            return {"error": f"Batch not found: {batch_id}"}

        # Re-add edits to pending and apply
        self.pending_edits = list(batch.edits)
        return self.apply_all(dry_run)

    # =========================================================================
    # Internal Operations
    # =========================================================================

    def _apply_edit(self, edit: FileEdit) -> None:
        """Apply a single edit operation."""
        path = self.working_dir / edit.path

        if edit.operation == EditOperation.CREATE:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(edit.content or "")

        elif edit.operation == EditOperation.MODIFY:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {edit.path}")
            path.write_text(edit.content or "")

        elif edit.operation == EditOperation.DELETE:
            if path.exists():
                path.unlink()
            else:
                raise FileNotFoundError(f"File not found: {edit.path}")

        elif edit.operation == EditOperation.RENAME:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {edit.path}")
            new_path = self.working_dir / edit.new_path
            new_path.parent.mkdir(parents=True, exist_ok=True)
            path.rename(new_path)

        elif edit.operation == EditOperation.APPEND:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(edit.content or "")
            else:
                with path.open("a") as f:
                    f.write(edit.content or "")

        elif edit.operation == EditOperation.INSERT:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {edit.path}")
            lines = path.read_text().splitlines(keepends=True)
            position = edit.position or 0
            position = min(position, len(lines))
            content_lines = (edit.content or "").splitlines(keepends=True)
            lines = lines[:position] + content_lines + lines[position:]
            path.write_text("".join(lines))

        elif edit.operation == EditOperation.REPLACE:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {edit.path}")
            content = path.read_text()
            if edit.search and edit.search in content:
                content = content.replace(edit.search, edit.replace or "")
                path.write_text(content)
            else:
                raise ValueError(f"Search string not found in {edit.path}")

    def _validate_edit(self, edit: FileEdit) -> None:
        """Validate an edit operation without applying."""
        path = self.working_dir / edit.path

        if edit.operation == EditOperation.CREATE:
            if path.exists():
                raise FileExistsError(f"File already exists: {edit.path}")

        elif edit.operation in (EditOperation.MODIFY, EditOperation.DELETE,
                                EditOperation.RENAME, EditOperation.INSERT):
            if not path.exists():
                raise FileNotFoundError(f"File not found: {edit.path}")

        elif edit.operation == EditOperation.REPLACE:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {edit.path}")
            content = path.read_text()
            if edit.search and edit.search not in content:
                raise ValueError(f"Search string not found in {edit.path}")

    def _sort_edits(self, edits: List[FileEdit]) -> List[FileEdit]:
        """Sort edits for proper application order."""
        # Order: CREATE/INSERT -> MODIFY/APPEND/REPLACE -> RENAME -> DELETE
        priority = {
            EditOperation.CREATE: 0,
            EditOperation.INSERT: 1,
            EditOperation.APPEND: 2,
            EditOperation.MODIFY: 3,
            EditOperation.REPLACE: 4,
            EditOperation.RENAME: 5,
            EditOperation.DELETE: 6,
        }
        return sorted(edits, key=lambda e: priority.get(e.operation, 99))

    # =========================================================================
    # Backup and Rollback
    # =========================================================================

    def _create_backup(self, batch: EditBatch) -> None:
        """Create backups for all files that will be modified."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._current_backup_dir = self.backup_dir / f"{batch.id}_{timestamp}"
        self._current_backup_dir.mkdir(parents=True, exist_ok=True)

        for edit in batch.edits:
            if edit.operation in (EditOperation.MODIFY, EditOperation.DELETE,
                                  EditOperation.RENAME, EditOperation.INSERT,
                                  EditOperation.REPLACE):
                source = self.working_dir / edit.path
                if source.exists():
                    # Create relative backup path
                    backup_path = self._current_backup_dir / edit.path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, backup_path)
                    edit.backup_path = str(backup_path)

        # Save batch metadata
        meta_path = self._current_backup_dir / "batch.json"
        with meta_path.open("w") as f:
            json.dump(batch.to_dict(), f, indent=2)

    def _rollback(self, batch: EditBatch) -> None:
        """Rollback all applied edits in a batch."""
        if not self._current_backup_dir:
            return

        batch.rolled_back_at = time.time()

        for edit in reversed(batch.edits):
            if edit.status != EditStatus.APPLIED:
                continue

            try:
                path = self.working_dir / edit.path

                if edit.operation == EditOperation.CREATE:
                    # Remove created file
                    if path.exists():
                        path.unlink()

                elif edit.operation in (EditOperation.MODIFY, EditOperation.INSERT,
                                        EditOperation.REPLACE):
                    # Restore from backup
                    if edit.backup_path and Path(edit.backup_path).exists():
                        shutil.copy2(edit.backup_path, path)

                elif edit.operation == EditOperation.DELETE:
                    # Restore deleted file
                    if edit.backup_path and Path(edit.backup_path).exists():
                        path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(edit.backup_path, path)

                elif edit.operation == EditOperation.RENAME:
                    # Rename back
                    new_path = self.working_dir / edit.new_path
                    if new_path.exists():
                        new_path.rename(path)

                elif edit.operation == EditOperation.APPEND:
                    # Restore from backup
                    if edit.backup_path and Path(edit.backup_path).exists():
                        shutil.copy2(edit.backup_path, path)

                edit.status = EditStatus.ROLLED_BACK

            except Exception as e:
                edit.error = f"Rollback failed: {e}"

    def rollback_batch(self, batch_id: str) -> Dict[str, Any]:
        """Rollback a previously applied batch."""
        batch = self.batches.get(batch_id)
        if not batch:
            return {"error": f"Batch not found: {batch_id}"}

        if batch.rolled_back_at:
            return {"error": "Batch already rolled back"}

        # Find backup directory
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.name.startswith(batch_id):
                self._current_backup_dir = backup_dir
                break
        else:
            return {"error": "Backup not found for batch"}

        self._rollback(batch)

        return {
            "batch_id": batch_id,
            "rolled_back": True,
            "edits": [e.to_dict() for e in batch.edits],
        }

    # =========================================================================
    # Conflict Detection
    # =========================================================================

    def detect_conflicts(self, edits: Optional[List[FileEdit]] = None) -> List[Dict[str, Any]]:
        """
        Detect potential conflicts between edits.

        Returns list of conflicts with details.
        """
        edits = edits or self.pending_edits
        conflicts = []

        # Group edits by path
        by_path: Dict[str, List[FileEdit]] = {}
        for edit in edits:
            path = edit.path
            if path not in by_path:
                by_path[path] = []
            by_path[path].append(edit)

        # Check for conflicts
        for path, path_edits in by_path.items():
            if len(path_edits) > 1:
                # Multiple edits to same file
                operations = [e.operation.value for e in path_edits]

                # DELETE + anything else is a conflict
                if EditOperation.DELETE in [e.operation for e in path_edits]:
                    conflicts.append({
                        "path": path,
                        "type": "delete_conflict",
                        "operations": operations,
                        "message": "DELETE combined with other operations",
                    })

                # Multiple MODIFY/REPLACE without coordination
                if sum(1 for e in path_edits if e.operation in
                       (EditOperation.MODIFY, EditOperation.REPLACE)) > 1:
                    conflicts.append({
                        "path": path,
                        "type": "modify_conflict",
                        "operations": operations,
                        "message": "Multiple MODIFY/REPLACE operations on same file",
                    })

        return conflicts

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_file_hash(self, path: str) -> Optional[str]:
        """Get MD5 hash of a file."""
        full_path = self.working_dir / path
        if not full_path.exists():
            return None
        content = full_path.read_bytes()
        return hashlib.md5(content).hexdigest()

    def list_batches(self) -> List[Dict[str, Any]]:
        """List all batches."""
        return [b.to_dict() for b in self.batches.values()]

    def get_batch(self, batch_id: str) -> Optional[EditBatch]:
        """Get a batch by ID."""
        return self.batches.get(batch_id)


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Edit Coordinator."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-File Edit Coordinator (Step 56)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # apply command
    apply_parser = subparsers.add_parser("apply", help="Apply edits from JSON file")
    apply_parser.add_argument("edits_file", help="JSON file with edits")
    apply_parser.add_argument("--dry-run", action="store_true", help="Validate without applying")
    apply_parser.add_argument("--working-dir", default=".", help="Working directory")

    # rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback a batch")
    rollback_parser.add_argument("batch_id", help="Batch ID to rollback")
    rollback_parser.add_argument("--working-dir", default=".", help="Working directory")

    # list command
    list_parser = subparsers.add_parser("list", help="List batches")
    list_parser.add_argument("--working-dir", default=".", help="Working directory")

    args = parser.parse_args()

    coordinator = MultiFileEditCoordinator(Path(args.working_dir))

    if args.command == "apply":
        with open(args.edits_file) as f:
            edits_data = json.load(f)

        for edit_data in edits_data:
            edit = FileEdit(
                path=edit_data["path"],
                operation=EditOperation(edit_data["operation"]),
                content=edit_data.get("content"),
                new_path=edit_data.get("new_path"),
                position=edit_data.get("position"),
                search=edit_data.get("search"),
                replace=edit_data.get("replace"),
            )
            coordinator.add_edit(edit)

        result = coordinator.apply_all(dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
        return 0 if result.get("success", True) else 1

    elif args.command == "rollback":
        result = coordinator.rollback_batch(args.batch_id)
        print(json.dumps(result, indent=2))
        return 0 if "error" not in result else 1

    elif args.command == "list":
        batches = coordinator.list_batches()
        print(json.dumps(batches, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
