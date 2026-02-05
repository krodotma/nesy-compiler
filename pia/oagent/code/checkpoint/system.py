#!/usr/bin/env python3
"""
system.py - Checkpoint System (Step 65)

PBTSO Phase: ITERATE, DISTILL

Provides:
- Named checkpoint creation and management
- Automatic checkpointing at intervals
- Checkpoint branching for experiments
- Checkpoint comparison and diffing
- Incremental checkpoint storage

Bus Topics:
- code.checkpoint.created
- code.checkpoint.restored
- code.checkpoint.deleted

Protocol: DKIN v30, PAIP v16
"""

from __future__ import annotations

import gzip
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
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple


# =============================================================================
# Types
# =============================================================================

class CheckpointType(Enum):
    """Type of checkpoint."""
    MANUAL = "manual"          # User-created
    AUTO = "auto"              # Automatic (interval-based)
    PRE_EDIT = "pre_edit"      # Before edit operation
    POST_EDIT = "post_edit"    # After edit operation
    BRANCH = "branch"          # Branch point for experiments
    MILESTONE = "milestone"    # Named milestone


@dataclass
class FileState:
    """State of a file at checkpoint time."""
    path: str
    hash: str
    size: int
    mode: int
    mtime: float
    is_binary: bool = False
    compressed: bool = False
    delta_base: Optional[str] = None  # For incremental storage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "hash": self.hash,
            "size": self.size,
            "mode": self.mode,
            "mtime": self.mtime,
            "is_binary": self.is_binary,
            "compressed": self.compressed,
            "delta_base": self.delta_base,
        }


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    author: str = "code-agent"
    message: str = ""
    task_id: Optional[str] = None
    proposal_id: Optional[str] = None
    pbtso_phase: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "author": self.author,
            "message": self.message,
            "task_id": self.task_id,
            "proposal_id": self.proposal_id,
            "pbtso_phase": self.pbtso_phase,
            "custom": self.custom,
        }


@dataclass
class Checkpoint:
    """
    A complete checkpoint of workspace state.
    """
    id: str
    name: str
    checkpoint_type: CheckpointType
    files: List[FileState]
    created_at: float
    metadata: CheckpointMetadata
    parent_id: Optional[str] = None
    branch_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_size(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def is_branch_point(self) -> bool:
        return self.checkpoint_type == CheckpointType.BRANCH

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "checkpoint_type": self.checkpoint_type.value,
            "files": [f.to_dict() for f in self.files],
            "file_count": self.file_count,
            "total_size": self.total_size,
            "created_at": self.created_at,
            "metadata": self.metadata.to_dict(),
            "parent_id": self.parent_id,
            "branch_name": self.branch_name,
            "tags": self.tags,
        }


# =============================================================================
# Checkpoint System
# =============================================================================

class CheckpointSystem:
    """
    Manage workspace checkpoints for safe editing.

    PBTSO Phase: ITERATE, DISTILL

    Features:
    - Named checkpoint creation
    - Automatic checkpointing at intervals
    - Checkpoint branching for experiments
    - Incremental storage with compression
    - Checkpoint comparison and diffing

    Usage:
        system = CheckpointSystem(working_dir)
        cp = system.create("before_refactor", "Saving state before major refactor")
        # ... make changes ...
        system.restore(cp.id)
    """

    BUS_TOPICS = {
        "created": "code.checkpoint.created",
        "restored": "code.checkpoint.restored",
        "deleted": "code.checkpoint.deleted",
    }

    # File extensions to treat as binary
    BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf",
                        ".zip", ".tar", ".gz", ".whl", ".pyc", ".so", ".dll"}

    def __init__(
        self,
        working_dir: Path,
        bus: Optional[Any] = None,
        storage_dir: Optional[Path] = None,
        auto_checkpoint_interval: int = 300,  # seconds
        max_checkpoints: int = 50,
        compression_enabled: bool = True,
        incremental_enabled: bool = True,
    ):
        self.working_dir = Path(working_dir)
        self.bus = bus
        self.storage_dir = storage_dir or self.working_dir / ".pluribus" / "checkpoints"
        self.auto_checkpoint_interval = auto_checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.compression_enabled = compression_enabled
        self.incremental_enabled = incremental_enabled

        self._checkpoints: Dict[str, Checkpoint] = {}
        self._current_branch: str = "main"
        self._last_auto_checkpoint: float = 0
        self._file_cache: Dict[str, str] = {}  # path -> hash

        self._ensure_storage()
        self._load_checkpoints()

    def _ensure_storage(self) -> None:
        """Ensure storage directories exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        (self.storage_dir / "data").mkdir(exist_ok=True)
        (self.storage_dir / "metadata").mkdir(exist_ok=True)
        (self.storage_dir / "index").mkdir(exist_ok=True)

    def _load_checkpoints(self) -> None:
        """Load existing checkpoints from storage."""
        index_dir = self.storage_dir / "index"
        for index_file in index_dir.glob("*.json"):
            try:
                with index_file.open() as f:
                    data = json.load(f)

                files = [
                    FileState(
                        path=fd["path"],
                        hash=fd["hash"],
                        size=fd["size"],
                        mode=fd.get("mode", 0o644),
                        mtime=fd.get("mtime", 0),
                        is_binary=fd.get("is_binary", False),
                        compressed=fd.get("compressed", False),
                        delta_base=fd.get("delta_base"),
                    )
                    for fd in data["files"]
                ]

                metadata = CheckpointMetadata(
                    author=data.get("metadata", {}).get("author", "code-agent"),
                    message=data.get("metadata", {}).get("message", ""),
                    task_id=data.get("metadata", {}).get("task_id"),
                    proposal_id=data.get("metadata", {}).get("proposal_id"),
                    pbtso_phase=data.get("metadata", {}).get("pbtso_phase"),
                    custom=data.get("metadata", {}).get("custom", {}),
                )

                checkpoint = Checkpoint(
                    id=data["id"],
                    name=data["name"],
                    checkpoint_type=CheckpointType(data.get("checkpoint_type", "manual")),
                    files=files,
                    created_at=data["created_at"],
                    metadata=metadata,
                    parent_id=data.get("parent_id"),
                    branch_name=data.get("branch_name"),
                    tags=data.get("tags", []),
                )

                self._checkpoints[checkpoint.id] = checkpoint

            except Exception:
                pass  # Skip corrupted files

    # =========================================================================
    # Checkpoint Creation
    # =========================================================================

    def create(
        self,
        name: str,
        message: str = "",
        checkpoint_type: CheckpointType = CheckpointType.MANUAL,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        metadata: Optional[CheckpointMetadata] = None,
        tags: Optional[List[str]] = None,
    ) -> Checkpoint:
        """
        Create a new checkpoint.

        Args:
            name: Checkpoint name
            message: Description message
            checkpoint_type: Type of checkpoint
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            metadata: Additional metadata
            tags: Tags for organization

        Returns:
            Created Checkpoint
        """
        checkpoint_id = f"cp-{uuid.uuid4().hex[:12]}"
        timestamp = time.time()

        # Default patterns
        if include_patterns is None:
            include_patterns = ["**/*.py", "**/*.ts", "**/*.tsx", "**/*.js",
                              "**/*.json", "**/*.yaml", "**/*.yml", "**/*.md"]

        if exclude_patterns is None:
            exclude_patterns = ["**/__pycache__/**", "**/node_modules/**",
                              "**/.git/**", "**/.venv/**", "**/dist/**"]

        # Collect file states
        files = self._collect_files(include_patterns, exclude_patterns, checkpoint_id)

        # Build metadata
        if metadata is None:
            metadata = CheckpointMetadata(message=message)
        else:
            metadata.message = message

        # Find parent (most recent checkpoint on current branch)
        parent_id = self._find_parent_checkpoint()

        checkpoint = Checkpoint(
            id=checkpoint_id,
            name=name,
            checkpoint_type=checkpoint_type,
            files=files,
            created_at=timestamp,
            metadata=metadata,
            parent_id=parent_id,
            branch_name=self._current_branch,
            tags=tags or [],
        )

        self._checkpoints[checkpoint.id] = checkpoint
        self._save_checkpoint(checkpoint)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        # Emit event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["created"],
                "kind": "checkpoint",
                "actor": "code-agent",
                "data": {
                    "checkpoint_id": checkpoint.id,
                    "name": name,
                    "file_count": len(files),
                    "total_size": checkpoint.total_size,
                    "branch": self._current_branch,
                },
            })

        return checkpoint

    def _collect_files(
        self,
        include_patterns: List[str],
        exclude_patterns: List[str],
        checkpoint_id: str,
    ) -> List[FileState]:
        """Collect files matching patterns and create states."""
        files: List[FileState] = []
        processed: Set[str] = set()

        # Build exclude set
        excluded: Set[Path] = set()
        for pattern in exclude_patterns:
            excluded.update(self.working_dir.glob(pattern))

        # Process include patterns
        for pattern in include_patterns:
            for file_path in self.working_dir.glob(pattern):
                if file_path in excluded:
                    continue
                if not file_path.is_file():
                    continue

                rel_path = str(file_path.relative_to(self.working_dir))
                if rel_path in processed:
                    continue
                processed.add(rel_path)

                try:
                    file_state = self._create_file_state(file_path, rel_path, checkpoint_id)
                    if file_state:
                        files.append(file_state)
                except Exception:
                    pass  # Skip inaccessible files

        return files

    def _create_file_state(
        self,
        full_path: Path,
        rel_path: str,
        checkpoint_id: str,
    ) -> Optional[FileState]:
        """Create a file state and store content."""
        try:
            content = full_path.read_bytes()
            file_hash = hashlib.sha256(content).hexdigest()
            stat = full_path.stat()

            is_binary = full_path.suffix.lower() in self.BINARY_EXTENSIONS

            # Store content if not already stored
            data_path = self.storage_dir / "data" / file_hash[:2] / file_hash
            if not data_path.exists():
                data_path.parent.mkdir(parents=True, exist_ok=True)

                if self.compression_enabled and not is_binary:
                    with gzip.open(str(data_path) + ".gz", "wb") as f:
                        f.write(content)
                    compressed = True
                else:
                    data_path.write_bytes(content)
                    compressed = False
            else:
                compressed = (data_path.with_suffix(".gz")).exists()

            return FileState(
                path=rel_path,
                hash=file_hash,
                size=len(content),
                mode=stat.st_mode,
                mtime=stat.st_mtime,
                is_binary=is_binary,
                compressed=compressed,
            )

        except Exception:
            return None

    def _find_parent_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint on current branch."""
        branch_checkpoints = [
            cp for cp in self._checkpoints.values()
            if cp.branch_name == self._current_branch
        ]

        if not branch_checkpoints:
            return None

        branch_checkpoints.sort(key=lambda c: c.created_at, reverse=True)
        return branch_checkpoints[0].id

    def _save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint index."""
        index_path = self.storage_dir / "index" / f"{checkpoint.id}.json"
        with index_path.open("w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

    # =========================================================================
    # Checkpoint Restoration
    # =========================================================================

    def restore(
        self,
        checkpoint_id: str,
        files: Optional[List[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Restore workspace to checkpoint state.

        Args:
            checkpoint_id: ID of checkpoint to restore
            files: Optional specific files to restore
            dry_run: If True, just report what would change

        Returns:
            Dict with restoration results
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            return {"error": f"Checkpoint not found: {checkpoint_id}"}

        # Emit start event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["restored"],
                "kind": "checkpoint",
                "actor": "code-agent",
                "data": {
                    "checkpoint_id": checkpoint_id,
                    "dry_run": dry_run,
                },
            })

        # Filter files if specified
        files_to_restore = checkpoint.files
        if files:
            file_set = set(files)
            files_to_restore = [f for f in checkpoint.files if f.path in file_set]

        # Create pre-restore checkpoint
        if not dry_run:
            self.create(
                name=f"pre_restore_{checkpoint.name}",
                message=f"State before restoring to {checkpoint.name}",
                checkpoint_type=CheckpointType.PRE_EDIT,
            )

        restored = 0
        skipped = 0
        errors: List[str] = []

        for file_state in files_to_restore:
            try:
                if not dry_run:
                    self._restore_file(file_state)
                restored += 1
            except Exception as e:
                errors.append(f"{file_state.path}: {e}")
                skipped += 1

        return {
            "checkpoint_id": checkpoint_id,
            "restored": restored,
            "skipped": skipped,
            "errors": errors,
            "dry_run": dry_run,
        }

    def _restore_file(self, file_state: FileState) -> None:
        """Restore a single file from checkpoint."""
        target_path = self.working_dir / file_state.path

        # Find content in storage
        data_path = self.storage_dir / "data" / file_state.hash[:2] / file_state.hash

        if file_state.compressed:
            gz_path = Path(str(data_path) + ".gz")
            if gz_path.exists():
                with gzip.open(gz_path, "rb") as f:
                    content = f.read()
            else:
                raise FileNotFoundError(f"Compressed data not found: {gz_path}")
        else:
            if data_path.exists():
                content = data_path.read_bytes()
            else:
                raise FileNotFoundError(f"Data not found: {data_path}")

        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        target_path.write_bytes(content)

        # Restore mode
        os.chmod(target_path, file_state.mode)

    # =========================================================================
    # Branch Management
    # =========================================================================

    def create_branch(self, branch_name: str, from_checkpoint: Optional[str] = None) -> Checkpoint:
        """
        Create a new checkpoint branch.

        Args:
            branch_name: Name for the new branch
            from_checkpoint: Optional checkpoint to branch from

        Returns:
            Branch point checkpoint
        """
        if from_checkpoint:
            base = self._checkpoints.get(from_checkpoint)
            if not base:
                raise ValueError(f"Checkpoint not found: {from_checkpoint}")
            # Restore to base checkpoint first
            self.restore(from_checkpoint)

        # Create branch point checkpoint
        checkpoint = self.create(
            name=f"branch_{branch_name}",
            message=f"Branch point for {branch_name}",
            checkpoint_type=CheckpointType.BRANCH,
        )

        self._current_branch = branch_name
        checkpoint.branch_name = branch_name

        self._save_checkpoint(checkpoint)

        return checkpoint

    def switch_branch(self, branch_name: str) -> Optional[Checkpoint]:
        """
        Switch to a different branch.

        Restores to the latest checkpoint on that branch.
        """
        # Find latest checkpoint on branch
        branch_checkpoints = [
            cp for cp in self._checkpoints.values()
            if cp.branch_name == branch_name
        ]

        if not branch_checkpoints:
            return None

        branch_checkpoints.sort(key=lambda c: c.created_at, reverse=True)
        latest = branch_checkpoints[0]

        # Restore and switch
        self.restore(latest.id)
        self._current_branch = branch_name

        return latest

    def list_branches(self) -> List[str]:
        """List all branches."""
        branches: Set[str] = set()
        for cp in self._checkpoints.values():
            if cp.branch_name:
                branches.add(cp.branch_name)
        return sorted(branches)

    def merge_branch(self, branch_name: str, strategy: str = "ours") -> Dict[str, Any]:
        """
        Merge a branch into current branch.

        This is a simplified merge - production would use SemanticMerger.
        """
        # Get latest checkpoints from both branches
        current_checkpoints = [
            cp for cp in self._checkpoints.values()
            if cp.branch_name == self._current_branch
        ]
        source_checkpoints = [
            cp for cp in self._checkpoints.values()
            if cp.branch_name == branch_name
        ]

        if not current_checkpoints or not source_checkpoints:
            return {"error": "Branch not found"}

        # For now, just create a merge checkpoint
        checkpoint = self.create(
            name=f"merge_{branch_name}_into_{self._current_branch}",
            message=f"Merged {branch_name} into {self._current_branch}",
            checkpoint_type=CheckpointType.MILESTONE,
            tags=["merge", branch_name],
        )

        return {
            "merged": True,
            "checkpoint_id": checkpoint.id,
            "source_branch": branch_name,
            "target_branch": self._current_branch,
        }

    # =========================================================================
    # Automatic Checkpointing
    # =========================================================================

    def auto_checkpoint_if_needed(self) -> Optional[Checkpoint]:
        """Create automatic checkpoint if interval has passed."""
        now = time.time()

        if now - self._last_auto_checkpoint >= self.auto_checkpoint_interval:
            self._last_auto_checkpoint = now
            return self.create(
                name=f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                message="Automatic checkpoint",
                checkpoint_type=CheckpointType.AUTO,
            )

        return None

    def enable_auto_checkpoint(self, interval: Optional[int] = None) -> None:
        """Enable automatic checkpointing."""
        if interval:
            self.auto_checkpoint_interval = interval
        self._last_auto_checkpoint = time.time()

    def disable_auto_checkpoint(self) -> None:
        """Disable automatic checkpointing."""
        self.auto_checkpoint_interval = float("inf")

    # =========================================================================
    # Checkpoint Management
    # =========================================================================

    def get(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get checkpoint by ID."""
        return self._checkpoints.get(checkpoint_id)

    def find_by_name(self, name: str) -> Optional[Checkpoint]:
        """Find checkpoint by name."""
        for cp in self._checkpoints.values():
            if cp.name == name:
                return cp
        return None

    def list(
        self,
        branch: Optional[str] = None,
        checkpoint_type: Optional[CheckpointType] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[Checkpoint]:
        """List checkpoints with filters."""
        checkpoints = list(self._checkpoints.values())

        if branch:
            checkpoints = [c for c in checkpoints if c.branch_name == branch]

        if checkpoint_type:
            checkpoints = [c for c in checkpoints if c.checkpoint_type == checkpoint_type]

        if tags:
            tag_set = set(tags)
            checkpoints = [c for c in checkpoints if tag_set & set(c.tags)]

        checkpoints.sort(key=lambda c: c.created_at, reverse=True)

        return checkpoints[:limit]

    def delete(self, checkpoint_id: str, keep_data: bool = False) -> bool:
        """Delete a checkpoint."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            return False

        # Delete index
        index_path = self.storage_dir / "index" / f"{checkpoint_id}.json"
        if index_path.exists():
            index_path.unlink()

        # Optionally delete data (only if not used by other checkpoints)
        if not keep_data:
            other_hashes = set()
            for cp in self._checkpoints.values():
                if cp.id != checkpoint_id:
                    for f in cp.files:
                        other_hashes.add(f.hash)

            for f in checkpoint.files:
                if f.hash not in other_hashes:
                    data_path = self.storage_dir / "data" / f.hash[:2] / f.hash
                    if data_path.exists():
                        data_path.unlink()
                    gz_path = Path(str(data_path) + ".gz")
                    if gz_path.exists():
                        gz_path.unlink()

        del self._checkpoints[checkpoint_id]

        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["deleted"],
                "kind": "checkpoint",
                "actor": "code-agent",
                "data": {"checkpoint_id": checkpoint_id},
            })

        return True

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints exceeding max count."""
        # Keep milestones and branches
        keep_types = {CheckpointType.MILESTONE, CheckpointType.BRANCH}

        removable = [
            c for c in self._checkpoints.values()
            if c.checkpoint_type not in keep_types
        ]

        if len(removable) > self.max_checkpoints:
            removable.sort(key=lambda c: c.created_at)
            for cp in removable[:-self.max_checkpoints]:
                self.delete(cp.id)

    # =========================================================================
    # Comparison and Diffing
    # =========================================================================

    def diff(self, checkpoint_a: str, checkpoint_b: str) -> Dict[str, Any]:
        """Compare two checkpoints."""
        cp_a = self._checkpoints.get(checkpoint_a)
        cp_b = self._checkpoints.get(checkpoint_b)

        if not cp_a or not cp_b:
            return {"error": "Checkpoint not found"}

        files_a = {f.path: f for f in cp_a.files}
        files_b = {f.path: f for f in cp_b.files}

        added = [p for p in files_b if p not in files_a]
        removed = [p for p in files_a if p not in files_b]
        modified = [
            p for p in files_a
            if p in files_b and files_a[p].hash != files_b[p].hash
        ]

        return {
            "checkpoint_a": checkpoint_a,
            "checkpoint_b": checkpoint_b,
            "added": added,
            "removed": removed,
            "modified": modified,
            "unchanged": len(files_a) - len(removed) - len(modified),
        }

    def get_history(self, file_path: str) -> List[Tuple[Checkpoint, FileState]]:
        """Get history of a file across checkpoints."""
        history: List[Tuple[Checkpoint, FileState]] = []

        for cp in self._checkpoints.values():
            for f in cp.files:
                if f.path == file_path:
                    history.append((cp, f))

        history.sort(key=lambda x: x[0].created_at, reverse=True)
        return history

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = 0
        data_dir = self.storage_dir / "data"

        if data_dir.exists():
            for file_path in data_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

        return {
            "total_checkpoints": len(self._checkpoints),
            "branches": self.list_branches(),
            "current_branch": self._current_branch,
            "storage_size_bytes": total_size,
            "storage_dir": str(self.storage_dir),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Checkpoint System."""
    import argparse

    parser = argparse.ArgumentParser(description="Checkpoint System (Step 65)")
    parser.add_argument("--working-dir", default=".", help="Working directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # create command
    create_parser = subparsers.add_parser("create", help="Create checkpoint")
    create_parser.add_argument("name", help="Checkpoint name")
    create_parser.add_argument("--message", "-m", default="", help="Message")
    create_parser.add_argument("--tags", nargs="*", help="Tags")

    # restore command
    restore_parser = subparsers.add_parser("restore", help="Restore checkpoint")
    restore_parser.add_argument("checkpoint_id", help="Checkpoint ID or name")
    restore_parser.add_argument("--files", nargs="*", help="Specific files")
    restore_parser.add_argument("--dry-run", action="store_true")

    # list command
    list_parser = subparsers.add_parser("list", help="List checkpoints")
    list_parser.add_argument("--branch", help="Filter by branch")
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--json", action="store_true")

    # diff command
    diff_parser = subparsers.add_parser("diff", help="Compare checkpoints")
    diff_parser.add_argument("checkpoint_a", help="First checkpoint")
    diff_parser.add_argument("checkpoint_b", help="Second checkpoint")

    # branch command
    branch_parser = subparsers.add_parser("branch", help="Branch operations")
    branch_parser.add_argument("action", choices=["create", "switch", "list"])
    branch_parser.add_argument("name", nargs="?", help="Branch name")

    # stats command
    subparsers.add_parser("stats", help="Show statistics")

    args = parser.parse_args()

    system = CheckpointSystem(Path(args.working_dir))

    if args.command == "create":
        cp = system.create(
            name=args.name,
            message=args.message,
            tags=args.tags,
        )
        print(f"Created checkpoint: {cp.id}")
        print(f"  Name: {cp.name}")
        print(f"  Files: {cp.file_count}")
        print(f"  Size: {cp.total_size} bytes")
        return 0

    elif args.command == "restore":
        # Try as ID first, then as name
        cp = system.get(args.checkpoint_id) or system.find_by_name(args.checkpoint_id)
        if not cp:
            print(f"Checkpoint not found: {args.checkpoint_id}")
            return 1

        result = system.restore(cp.id, args.files, args.dry_run)
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1

        action = "Would restore" if args.dry_run else "Restored"
        print(f"{action} {result['restored']} files from checkpoint {cp.name}")
        return 0

    elif args.command == "list":
        checkpoints = system.list(branch=args.branch, limit=args.limit)

        if args.json:
            print(json.dumps([c.to_dict() for c in checkpoints], indent=2))
        else:
            for cp in checkpoints:
                ts = datetime.fromtimestamp(cp.created_at).strftime("%Y-%m-%d %H:%M")
                print(f"{cp.id} [{cp.branch_name}] {cp.name} ({ts}, {cp.file_count} files)")

        return 0

    elif args.command == "diff":
        result = system.diff(args.checkpoint_a, args.checkpoint_b)
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1

        print(f"Added: {len(result['added'])} files")
        print(f"Removed: {len(result['removed'])} files")
        print(f"Modified: {len(result['modified'])} files")
        print(f"Unchanged: {result['unchanged']} files")
        return 0

    elif args.command == "branch":
        if args.action == "list":
            branches = system.list_branches()
            for b in branches:
                marker = "*" if b == system._current_branch else " "
                print(f"{marker} {b}")
        elif args.action == "create" and args.name:
            cp = system.create_branch(args.name)
            print(f"Created branch {args.name} at checkpoint {cp.id}")
        elif args.action == "switch" and args.name:
            cp = system.switch_branch(args.name)
            if cp:
                print(f"Switched to branch {args.name} at checkpoint {cp.id}")
            else:
                print(f"Branch not found: {args.name}")
                return 1
        return 0

    elif args.command == "stats":
        stats = system.get_stats()
        print(f"Total checkpoints: {stats['total_checkpoints']}")
        print(f"Current branch: {stats['current_branch']}")
        print(f"Branches: {', '.join(stats['branches'])}")
        print(f"Storage size: {stats['storage_size_bytes']} bytes")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
