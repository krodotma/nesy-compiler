#!/usr/bin/env python3
"""
clone_manager.py - PAIP Clone Manager (Step 57)

PBTSO Phase: SEQUESTER

Provides:
- PAIP-isolated working copies for safe code execution
- Git-based cloning with shallow depth
- Patch generation and sync back
- Clone lifecycle management

Bus Topics:
- paip.clone.create
- paip.clone.ready
- paip.clone.sync
- paip.clone.cleanup

Protocol: DKIN v30, PAIP v16
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Types
# =============================================================================

class CloneStatus(Enum):
    """Status of a PAIP clone."""
    CREATING = "creating"
    READY = "ready"
    IN_USE = "in_use"
    SYNCING = "syncing"
    SYNCED = "synced"
    FAILED = "failed"
    CLEANED = "cleaned"


@dataclass
class PAIPClone:
    """Represents a PAIP-isolated working copy."""
    id: str
    task_id: str
    source_repo: Path
    clone_path: Path
    status: CloneStatus
    created_at: float
    updated_at: float
    branch: str = "main"
    commit_hash: Optional[str] = None
    patches: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "source_repo": str(self.source_repo),
            "clone_path": str(self.clone_path),
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "branch": self.branch,
            "commit_hash": self.commit_hash,
            "patches": self.patches,
            "error": self.error,
        }


# =============================================================================
# PAIP Clone Manager
# =============================================================================

class PAIPCloneManager:
    """
    Manage PAIP-isolated working copies.

    PBTSO Phase: SEQUESTER

    PAIP (Pluribus Agent Isolation Protocol) provides isolated
    environments for code execution and modification. Each task
    gets its own working copy to prevent cross-contamination.

    Features:
    - Shallow git clones for fast creation
    - Worktree support for shared object storage
    - Patch generation for controlled sync back
    - Automatic cleanup on completion

    Usage:
        manager = PAIPCloneManager(source_repo)
        clone_path = manager.create_clone("task-123")
        # ... make changes in clone_path ...
        manager.sync_back("task-123")
        manager.cleanup("task-123")
    """

    def __init__(
        self,
        source_repo: Path,
        bus: Optional[Any] = None,
        clone_base_dir: Optional[Path] = None,
        default_depth: int = 1,
        use_worktrees: bool = False,
    ):
        self.source_repo = Path(source_repo)
        self.bus = bus
        self.clone_base_dir = clone_base_dir or Path(tempfile.gettempdir()) / "paip-clones"
        self.default_depth = default_depth
        self.use_worktrees = use_worktrees

        self.clones: Dict[str, PAIPClone] = {}
        self._ensure_base_dir()

    def _ensure_base_dir(self) -> None:
        """Ensure clone base directory exists."""
        self.clone_base_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Clone Operations
    # =========================================================================

    def create_clone(
        self,
        task_id: str,
        branch: Optional[str] = None,
        depth: Optional[int] = None,
    ) -> Path:
        """
        Create a PAIP-isolated clone for a task.

        Args:
            task_id: ID of the task this clone is for
            branch: Branch to clone (default: current branch)
            depth: Clone depth (default: self.default_depth)

        Returns:
            Path to the clone directory
        """
        clone_id = f"paip-{task_id}-{uuid.uuid4().hex[:8]}"
        clone_path = self.clone_base_dir / clone_id
        now = time.time()

        clone = PAIPClone(
            id=clone_id,
            task_id=task_id,
            source_repo=self.source_repo,
            clone_path=clone_path,
            status=CloneStatus.CREATING,
            created_at=now,
            updated_at=now,
            branch=branch or self._get_current_branch(),
        )

        self.clones[task_id] = clone

        # Emit creation start
        if self.bus:
            self.bus.emit({
                "topic": "paip.clone.create",
                "kind": "paip",
                "actor": "code-agent",
                "data": {
                    "clone_id": clone_id,
                    "task_id": task_id,
                    "branch": clone.branch,
                },
            })

        try:
            if self.use_worktrees:
                self._create_worktree(clone, depth)
            else:
                self._create_clone(clone, depth)

            clone.status = CloneStatus.READY
            clone.commit_hash = self._get_commit_hash(clone_path)
            clone.updated_at = time.time()

            # Emit ready
            if self.bus:
                self.bus.emit({
                    "topic": "paip.clone.ready",
                    "kind": "paip",
                    "actor": "code-agent",
                    "data": {
                        "clone_id": clone_id,
                        "task_id": task_id,
                        "clone_path": str(clone_path),
                        "commit_hash": clone.commit_hash,
                    },
                })

            return clone_path

        except Exception as e:
            clone.status = CloneStatus.FAILED
            clone.error = str(e)
            clone.updated_at = time.time()
            raise

    def _create_clone(self, clone: PAIPClone, depth: Optional[int]) -> None:
        """Create a shallow git clone."""
        depth = depth or self.default_depth

        cmd = [
            "git", "clone",
            "--depth", str(depth),
            "--branch", clone.branch,
            "--single-branch",
            str(self.source_repo),
            str(clone.clone_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Clone failed: {result.stderr}")

    def _create_worktree(self, clone: PAIPClone, depth: Optional[int]) -> None:
        """Create a git worktree (shares objects with main repo)."""
        # First ensure a branch exists for the worktree
        worktree_branch = f"paip/{clone.task_id}"

        # Create branch from current HEAD
        subprocess.run(
            ["git", "branch", worktree_branch, "HEAD"],
            cwd=self.source_repo,
            capture_output=True,
        )

        # Create worktree
        result = subprocess.run(
            ["git", "worktree", "add", str(clone.clone_path), worktree_branch],
            cwd=self.source_repo,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Worktree creation failed: {result.stderr}")

    # =========================================================================
    # Sync Operations
    # =========================================================================

    def sync_back(self, task_id: str) -> bool:
        """
        Sync changes from clone back to main repo.

        Creates a patch from the clone and applies it to the source repo.

        Args:
            task_id: Task ID of the clone

        Returns:
            True if sync successful, False otherwise
        """
        clone = self.clones.get(task_id)
        if not clone:
            return False

        if clone.status not in (CloneStatus.READY, CloneStatus.IN_USE):
            return False

        clone.status = CloneStatus.SYNCING
        clone.updated_at = time.time()

        try:
            # Generate patch
            patch = self._generate_patch(clone)

            if not patch:
                clone.status = CloneStatus.SYNCED
                return True  # No changes to sync

            # Save patch
            patch_path = self.clone_base_dir / f"{clone.id}.patch"
            patch_path.write_text(patch)
            clone.patches.append(str(patch_path))

            # Apply patch to source
            self._apply_patch(patch)

            clone.status = CloneStatus.SYNCED
            clone.updated_at = time.time()

            # Emit sync complete
            if self.bus:
                self.bus.emit({
                    "topic": "paip.clone.sync",
                    "kind": "paip",
                    "actor": "code-agent",
                    "data": {
                        "clone_id": clone.id,
                        "task_id": task_id,
                        "patch_path": str(patch_path),
                        "patch_size": len(patch),
                    },
                })

            return True

        except Exception as e:
            clone.status = CloneStatus.FAILED
            clone.error = str(e)
            clone.updated_at = time.time()
            return False

    def _generate_patch(self, clone: PAIPClone) -> str:
        """Generate a patch from clone changes."""
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=clone.clone_path,
            capture_output=True,
            text=True,
        )

        return result.stdout

    def _apply_patch(self, patch: str) -> None:
        """Apply a patch to the source repo."""
        if not patch.strip():
            return

        result = subprocess.run(
            ["git", "apply", "--3way"],
            input=patch,
            cwd=self.source_repo,
            text=True,
            capture_output=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Patch apply failed: {result.stderr}")

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self, task_id: str) -> None:
        """
        Clean up a PAIP clone.

        Removes the clone directory and any associated resources.

        Args:
            task_id: Task ID of the clone to clean up
        """
        clone = self.clones.get(task_id)
        if not clone:
            return

        try:
            # Remove worktree if using worktrees
            if self.use_worktrees:
                subprocess.run(
                    ["git", "worktree", "remove", str(clone.clone_path), "--force"],
                    cwd=self.source_repo,
                    capture_output=True,
                )

                # Remove branch
                worktree_branch = f"paip/{task_id}"
                subprocess.run(
                    ["git", "branch", "-D", worktree_branch],
                    cwd=self.source_repo,
                    capture_output=True,
                )

            # Remove clone directory
            if clone.clone_path.exists():
                shutil.rmtree(clone.clone_path, ignore_errors=True)

            clone.status = CloneStatus.CLEANED
            clone.updated_at = time.time()

            # Emit cleanup
            if self.bus:
                self.bus.emit({
                    "topic": "paip.clone.cleanup",
                    "kind": "paip",
                    "actor": "code-agent",
                    "data": {
                        "clone_id": clone.id,
                        "task_id": task_id,
                    },
                })

        except Exception as e:
            clone.error = str(e)

        finally:
            del self.clones[task_id]

    def cleanup_all(self) -> int:
        """Clean up all clones. Returns number cleaned."""
        task_ids = list(self.clones.keys())
        for task_id in task_ids:
            self.cleanup(task_id)
        return len(task_ids)

    def cleanup_stale(self, max_age_s: int = 3600) -> int:
        """Clean up clones older than max_age_s. Returns number cleaned."""
        now = time.time()
        stale_ids = [
            task_id
            for task_id, clone in self.clones.items()
            if now - clone.created_at > max_age_s
        ]
        for task_id in stale_ids:
            self.cleanup(task_id)
        return len(stale_ids)

    # =========================================================================
    # Query
    # =========================================================================

    def get_clone(self, task_id: str) -> Optional[PAIPClone]:
        """Get clone by task ID."""
        return self.clones.get(task_id)

    def list_clones(self) -> List[PAIPClone]:
        """List all active clones."""
        return list(self.clones.values())

    def get_clone_path(self, task_id: str) -> Optional[Path]:
        """Get path to clone directory."""
        clone = self.clones.get(task_id)
        if clone and clone.clone_path.exists():
            return clone.clone_path
        return None

    # =========================================================================
    # Utilities
    # =========================================================================

    def _get_current_branch(self) -> str:
        """Get current branch of source repo."""
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=self.source_repo,
            capture_output=True,
            text=True,
        )
        branch = result.stdout.strip()
        return branch if branch else "main"

    def _get_commit_hash(self, repo_path: Path) -> str:
        """Get current commit hash."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()[:8]

    def get_diff(self, task_id: str) -> Optional[str]:
        """Get current diff for a clone."""
        clone = self.clones.get(task_id)
        if not clone or not clone.clone_path.exists():
            return None

        result = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=clone.clone_path,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get git status for a clone."""
        clone = self.clones.get(task_id)
        if not clone or not clone.clone_path.exists():
            return None

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=clone.clone_path,
            capture_output=True,
            text=True,
        )

        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        modified = [l[3:] for l in lines if l.startswith(" M")]
        added = [l[3:] for l in lines if l.startswith("A ")]
        deleted = [l[3:] for l in lines if l.startswith(" D")]
        untracked = [l[3:] for l in lines if l.startswith("??")]

        return {
            "modified": modified,
            "added": added,
            "deleted": deleted,
            "untracked": untracked,
            "total_changes": len(lines),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for PAIP Clone Manager."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="PAIP Clone Manager (Step 57)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a clone")
    create_parser.add_argument("task_id", help="Task ID")
    create_parser.add_argument("--source", required=True, help="Source repo path")
    create_parser.add_argument("--branch", help="Branch to clone")
    create_parser.add_argument("--depth", type=int, default=1, help="Clone depth")

    # sync command
    sync_parser = subparsers.add_parser("sync", help="Sync clone back")
    sync_parser.add_argument("task_id", help="Task ID")
    sync_parser.add_argument("--source", required=True, help="Source repo path")

    # cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up clone")
    cleanup_parser.add_argument("task_id", help="Task ID")
    cleanup_parser.add_argument("--source", required=True, help="Source repo path")

    # status command
    status_parser = subparsers.add_parser("status", help="Show clone status")
    status_parser.add_argument("task_id", help="Task ID")
    status_parser.add_argument("--source", required=True, help="Source repo path")

    args = parser.parse_args()

    manager = PAIPCloneManager(Path(args.source))

    if args.command == "create":
        clone_path = manager.create_clone(
            args.task_id,
            branch=args.branch,
            depth=args.depth,
        )
        print(f"Clone created: {clone_path}")
        return 0

    elif args.command == "sync":
        success = manager.sync_back(args.task_id)
        print(f"Sync {'successful' if success else 'failed'}")
        return 0 if success else 1

    elif args.command == "cleanup":
        manager.cleanup(args.task_id)
        print("Clone cleaned up")
        return 0

    elif args.command == "status":
        status = manager.get_status(args.task_id)
        if status:
            print(json.dumps(status, indent=2))
        else:
            print("Clone not found")
            return 1
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
