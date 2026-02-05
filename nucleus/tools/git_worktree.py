#!/usr/bin/env python3
"""
git_worktree.py - Git Worktree Management for Agent Isolation

Version: 1.0.0
Ring: 1 (Services)
Protocol: Worktree Protocol v1 / DKIN v30

This module provides git worktree management for agent isolation, replacing
the previous PAIP temp directory pattern. Benefits:
  - 10x faster than full repo clones (shared .git)
  - Automatic branch management per agent
  - Clean isolation without duplication
  - Easy cleanup and lifecycle management

Usage:
    from git_worktree import WorktreeManager

    wm = WorktreeManager("/pluribus")

    # Create a worktree for an agent
    path = wm.create("agent-claude-task-123", base_branch="main")

    # List worktrees
    worktrees = wm.list_worktrees()

    # Remove when done
    wm.remove("agent-claude-task-123")

Semops:
    PBWORKTREE: Worktree management operations
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger("nucleus.worktree")

VERSION = "1.0.0"

# Default worktree location
DEFAULT_WORKTREE_BASE = ".pluribus/worktrees"

# Maximum age for orphaned worktrees (24 hours)
MAX_WORKTREE_AGE_HOURS = 24


@dataclass
class Worktree:
    """Represents a git worktree."""
    path: str
    branch: str
    commit: str
    is_main: bool = False
    created_at: Optional[float] = None
    agent_id: Optional[str] = None


@dataclass
class WorktreeManager:
    """
    Manages git worktrees for agent isolation.

    Uses worktrees instead of full repo clones for:
    - Faster creation (no git clone needed)
    - Shared .git directory (saves disk space)
    - Clean branch per agent
    - Automatic cleanup
    """

    repo_path: str
    worktree_base: Optional[str] = None

    def __post_init__(self):
        """Initialize the worktree manager."""
        self.repo_path = os.path.abspath(self.repo_path)
        if self.worktree_base is None:
            self.worktree_base = os.path.join(self.repo_path, DEFAULT_WORKTREE_BASE)

        # Ensure worktree base exists
        os.makedirs(self.worktree_base, exist_ok=True)

    def _run_git(self, args: list[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
        """Run a git command and return the result."""
        cmd = ["git"] + args
        cwd = cwd or self.repo_path
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60
        )

    def list_worktrees(self) -> list[Worktree]:
        """List all worktrees for this repository."""
        result = self._run_git(["worktree", "list", "--porcelain"])
        if result.returncode != 0:
            logger.error(f"Failed to list worktrees: {result.stderr}")
            return []

        worktrees = []
        current: dict = {}

        for line in result.stdout.split("\n"):
            line = line.strip()
            if not line:
                if current:
                    wt = Worktree(
                        path=current.get("worktree", ""),
                        branch=current.get("branch", "").replace("refs/heads/", ""),
                        commit=current.get("HEAD", ""),
                        is_main=current.get("worktree", "") == self.repo_path
                    )
                    # Check metadata file for additional info
                    meta_path = Path(wt.path) / ".worktree_meta.json"
                    if meta_path.exists():
                        try:
                            meta = json.loads(meta_path.read_text())
                            wt.created_at = meta.get("created_at")
                            wt.agent_id = meta.get("agent_id")
                        except (json.JSONDecodeError, OSError):
                            pass
                    worktrees.append(wt)
                current = {}
                continue

            if line.startswith("worktree "):
                current["worktree"] = line[9:]
            elif line.startswith("HEAD "):
                current["HEAD"] = line[5:]
            elif line.startswith("branch "):
                current["branch"] = line[7:]
            elif line == "bare":
                current["bare"] = True
            elif line == "detached":
                current["detached"] = True

        return worktrees

    def create(
        self,
        name: str,
        base_branch: str = "main",
        agent_id: Optional[str] = None,
        force: bool = False
    ) -> str:
        """
        Create a new worktree for an agent.

        Args:
            name: Unique name for the worktree (used as branch name suffix)
            base_branch: Branch to base the worktree on
            agent_id: Optional agent identifier for metadata
            force: Force creation even if branch exists

        Returns:
            Path to the created worktree
        """
        # Sanitize name for use as branch
        safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)
        branch_name = f"worktree/{safe_name}"
        worktree_path = os.path.join(self.worktree_base, safe_name)

        # Check if worktree already exists
        if os.path.exists(worktree_path):
            if force:
                self.remove(name)
            else:
                logger.warning(f"Worktree {name} already exists at {worktree_path}")
                return worktree_path

        # Ensure base branch is up to date
        self._run_git(["fetch", "origin", base_branch])

        # Create the worktree with a new branch
        result = self._run_git([
            "worktree", "add",
            "-b", branch_name,
            worktree_path,
            f"origin/{base_branch}"
        ])

        if result.returncode != 0:
            # Branch might already exist, try without -b
            if "already exists" in result.stderr:
                result = self._run_git([
                    "worktree", "add",
                    worktree_path,
                    branch_name
                ])

            if result.returncode != 0:
                raise RuntimeError(f"Failed to create worktree: {result.stderr}")

        # Write metadata file
        meta = {
            "name": name,
            "branch": branch_name,
            "base_branch": base_branch,
            "agent_id": agent_id,
            "created_at": time.time(),
            "repo_path": self.repo_path
        }
        meta_path = Path(worktree_path) / ".worktree_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        logger.info(f"Created worktree '{name}' at {worktree_path}")
        return worktree_path

    def remove(self, name: str, delete_branch: bool = True) -> bool:
        """
        Remove a worktree.

        Args:
            name: Name of the worktree to remove
            delete_branch: Also delete the associated branch

        Returns:
            True if successful
        """
        safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)
        worktree_path = os.path.join(self.worktree_base, safe_name)
        branch_name = f"worktree/{safe_name}"

        # Remove the worktree
        result = self._run_git(["worktree", "remove", "--force", worktree_path])

        if result.returncode != 0:
            # Try manual cleanup if git worktree remove fails
            if os.path.exists(worktree_path):
                try:
                    shutil.rmtree(worktree_path)
                except OSError as e:
                    logger.error(f"Failed to remove worktree directory: {e}")
                    return False

            # Prune worktree list
            self._run_git(["worktree", "prune"])

        # Delete the branch if requested
        if delete_branch:
            self._run_git(["branch", "-D", branch_name])

        logger.info(f"Removed worktree '{name}'")
        return True

    def cleanup_orphaned(self, max_age_hours: float = MAX_WORKTREE_AGE_HOURS) -> list[str]:
        """
        Clean up worktrees older than max_age_hours.

        Returns:
            List of removed worktree names
        """
        removed = []
        cutoff = time.time() - (max_age_hours * 3600)

        for wt in self.list_worktrees():
            if wt.is_main:
                continue

            # Check age
            if wt.created_at and wt.created_at < cutoff:
                name = Path(wt.path).name
                if self.remove(name):
                    removed.append(name)

        # Also prune any dangling worktrees
        self._run_git(["worktree", "prune"])

        return removed

    def get_status(self, name: str) -> dict:
        """Get status of a worktree."""
        safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)
        worktree_path = os.path.join(self.worktree_base, safe_name)

        if not os.path.exists(worktree_path):
            return {"exists": False, "name": name}

        # Get git status
        result = self._run_git(["status", "--porcelain"], cwd=worktree_path)
        changes = len([l for l in result.stdout.split("\n") if l.strip()])

        # Get commit info
        result = self._run_git(["log", "-1", "--format=%H %s"], cwd=worktree_path)
        commit_info = result.stdout.strip()

        # Read metadata
        meta_path = Path(worktree_path) / ".worktree_meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        return {
            "exists": True,
            "name": name,
            "path": worktree_path,
            "branch": meta.get("branch", "unknown"),
            "uncommitted_changes": changes,
            "commit": commit_info,
            "created_at": meta.get("created_at"),
            "agent_id": meta.get("agent_id")
        }


def format_worktree_table(worktrees: list[Worktree]) -> str:
    """Format worktrees as a table."""
    lines = [
        "PATH                                    BRANCH                    COMMIT    AGE",
        "-" * 80
    ]
    for wt in worktrees:
        if wt.is_main:
            continue
        age = ""
        if wt.created_at:
            hours = (time.time() - wt.created_at) / 3600
            if hours < 1:
                age = f"{int(hours * 60)}m"
            elif hours < 24:
                age = f"{int(hours)}h"
            else:
                age = f"{int(hours / 24)}d"

        path = wt.path[-35:] if len(wt.path) > 35 else wt.path
        branch = wt.branch[-25:] if len(wt.branch) > 25 else wt.branch
        commit = wt.commit[:8]

        lines.append(f"{path:<40} {branch:<25} {commit:<10} {age}")

    return "\n".join(lines)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for PBWORKTREE operations."""
    if len(sys.argv) < 2:
        print(f"GIT WORKTREE MANAGER v{VERSION}")
        print("\nUsage:")
        print("  python3 git_worktree.py list                  # List all worktrees")
        print("  python3 git_worktree.py create <name> [base]  # Create worktree")
        print("  python3 git_worktree.py remove <name>         # Remove worktree")
        print("  python3 git_worktree.py status <name>         # Get worktree status")
        print("  python3 git_worktree.py cleanup [hours]       # Cleanup old worktrees")
        print("\nSemops: PBWORKTREE")
        sys.exit(1)

    cmd = sys.argv[1]

    # Default to current directory as repo path
    repo_path = os.environ.get("PLURIBUS_REPO_PATH", "/pluribus")
    wm = WorktreeManager(repo_path)

    if cmd == "list":
        worktrees = wm.list_worktrees()
        print(format_worktree_table(worktrees))
        print(f"\nTotal: {len([w for w in worktrees if not w.is_main])} worktrees")

    elif cmd == "create":
        if len(sys.argv) < 3:
            print("Usage: git_worktree.py create <name> [base_branch]")
            sys.exit(1)
        name = sys.argv[2]
        base = sys.argv[3] if len(sys.argv) > 3 else "main"
        agent_id = os.environ.get("PLURIBUS_ACTOR")
        try:
            path = wm.create(name, base_branch=base, agent_id=agent_id)
            print(f"Created worktree at: {path}")
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif cmd == "remove":
        if len(sys.argv) < 3:
            print("Usage: git_worktree.py remove <name>")
            sys.exit(1)
        name = sys.argv[2]
        if wm.remove(name):
            print(f"Removed worktree: {name}")
        else:
            print(f"Failed to remove worktree: {name}", file=sys.stderr)
            sys.exit(1)

    elif cmd == "status":
        if len(sys.argv) < 3:
            print("Usage: git_worktree.py status <name>")
            sys.exit(1)
        name = sys.argv[2]
        status = wm.get_status(name)
        print(json.dumps(status, indent=2))

    elif cmd == "cleanup":
        hours = float(sys.argv[2]) if len(sys.argv) > 2 else MAX_WORKTREE_AGE_HOURS
        removed = wm.cleanup_orphaned(max_age_hours=hours)
        if removed:
            print(f"Removed {len(removed)} orphaned worktrees:")
            for name in removed:
                print(f"  - {name}")
        else:
            print("No orphaned worktrees to clean up")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
