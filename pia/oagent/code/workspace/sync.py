#!/usr/bin/env python3
"""
sync.py - Workspace Sync (Step 68)

PBTSO Phase: DISTRIBUTE, SEQUESTER

Provides:
- Multi-agent workspace coordination
- File locking for exclusive access
- Change synchronization between agents
- Conflict prevention via pessimistic locking
- Workspace state broadcasting

Bus Topics:
- workspace.lock.acquired
- workspace.lock.released
- workspace.sync.push
- workspace.sync.pull

Protocol: DKIN v30, PAIP v16
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


# =============================================================================
# Types
# =============================================================================

class LockType(Enum):
    """Type of file lock."""
    EXCLUSIVE = "exclusive"   # Only one agent can write
    SHARED = "shared"         # Multiple agents can read
    INTENT = "intent"         # Signals intention to lock


class SyncState(Enum):
    """State of workspace synchronization."""
    SYNCED = "synced"         # All agents in sync
    PENDING = "pending"       # Changes pending sync
    CONFLICTED = "conflicted" # Sync conflict detected
    OFFLINE = "offline"       # Agent offline


class ChangeType(Enum):
    """Type of file change."""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


@dataclass
class WorkspaceLock:
    """
    Represents a lock on one or more files.
    """
    id: str
    agent_id: str
    files: Set[str]
    lock_type: LockType
    acquired_at: float
    expires_at: Optional[float] = None
    reason: str = ""

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def is_exclusive(self) -> bool:
        return self.lock_type == LockType.EXCLUSIVE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "files": list(self.files),
            "lock_type": self.lock_type.value,
            "acquired_at": self.acquired_at,
            "expires_at": self.expires_at,
            "reason": self.reason,
            "is_expired": self.is_expired,
        }


@dataclass
class FileChange:
    """
    Represents a file change for synchronization.
    """
    id: str
    path: str
    change_type: ChangeType
    agent_id: str
    timestamp: float
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    content: Optional[str] = None  # For small files
    size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "change_type": self.change_type.value,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "size": self.size,
        }


@dataclass
class AgentState:
    """State of an agent in the workspace."""
    agent_id: str
    last_seen: float
    sync_state: SyncState
    pending_changes: int = 0
    held_locks: List[str] = field(default_factory=list)

    @property
    def is_online(self) -> bool:
        return time.time() - self.last_seen < 30  # 30 second timeout

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "last_seen": self.last_seen,
            "sync_state": self.sync_state.value,
            "pending_changes": self.pending_changes,
            "held_locks": self.held_locks,
            "is_online": self.is_online,
        }


# =============================================================================
# Workspace Sync
# =============================================================================

class WorkspaceSync:
    """
    Coordinate workspace access between multiple agents.

    PBTSO Phase: DISTRIBUTE, SEQUESTER

    Features:
    - File-level locking for exclusive access
    - Change tracking and synchronization
    - Agent presence monitoring
    - Conflict prevention
    - Distributed state management

    Usage:
        sync = WorkspaceSync(working_dir, agent_id="code-agent")
        lock = sync.acquire_lock(["file.py"], LockType.EXCLUSIVE)
        # ... make changes ...
        sync.push_changes([FileChange(...)])
        sync.release_lock(lock.id)
    """

    BUS_TOPICS = {
        "lock_acquired": "workspace.lock.acquired",
        "lock_released": "workspace.lock.released",
        "sync_push": "workspace.sync.push",
        "sync_pull": "workspace.sync.pull",
        "agent_heartbeat": "workspace.agent.heartbeat",
    }

    def __init__(
        self,
        working_dir: Path,
        agent_id: str = "code-agent",
        bus: Optional[Any] = None,
        state_dir: Optional[Path] = None,
        lock_timeout_s: int = 300,
        heartbeat_interval_s: int = 10,
    ):
        self.working_dir = Path(working_dir)
        self.agent_id = agent_id
        self.bus = bus
        self.state_dir = state_dir or self.working_dir / ".pluribus" / "workspace"
        self.lock_timeout_s = lock_timeout_s
        self.heartbeat_interval_s = heartbeat_interval_s

        self._locks: Dict[str, WorkspaceLock] = {}
        self._agents: Dict[str, AgentState] = {}
        self._pending_changes: List[FileChange] = []
        self._file_versions: Dict[str, str] = {}  # path -> hash
        self._sync_sequence: int = 0

        self._ensure_state_dir()
        self._load_state()
        self._register_agent()

    def _ensure_state_dir(self) -> None:
        """Ensure state directory exists."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "locks").mkdir(exist_ok=True)
        (self.state_dir / "changes").mkdir(exist_ok=True)
        (self.state_dir / "agents").mkdir(exist_ok=True)

    def _load_state(self) -> None:
        """Load state from storage."""
        # Load locks
        locks_dir = self.state_dir / "locks"
        for lock_file in locks_dir.glob("*.json"):
            try:
                with lock_file.open() as f:
                    data = json.load(f)
                lock = WorkspaceLock(
                    id=data["id"],
                    agent_id=data["agent_id"],
                    files=set(data["files"]),
                    lock_type=LockType(data["lock_type"]),
                    acquired_at=data["acquired_at"],
                    expires_at=data.get("expires_at"),
                    reason=data.get("reason", ""),
                )
                if not lock.is_expired:
                    self._locks[lock.id] = lock
                else:
                    lock_file.unlink()  # Remove expired lock
            except Exception:
                pass

        # Load agents
        agents_dir = self.state_dir / "agents"
        for agent_file in agents_dir.glob("*.json"):
            try:
                with agent_file.open() as f:
                    data = json.load(f)
                agent = AgentState(
                    agent_id=data["agent_id"],
                    last_seen=data["last_seen"],
                    sync_state=SyncState(data["sync_state"]),
                    pending_changes=data.get("pending_changes", 0),
                    held_locks=data.get("held_locks", []),
                )
                self._agents[agent.agent_id] = agent
            except Exception:
                pass

    def _register_agent(self) -> None:
        """Register this agent."""
        self._agents[self.agent_id] = AgentState(
            agent_id=self.agent_id,
            last_seen=time.time(),
            sync_state=SyncState.SYNCED,
        )
        self._save_agent_state()

    def _save_agent_state(self) -> None:
        """Save this agent's state."""
        agent = self._agents.get(self.agent_id)
        if agent:
            agent.last_seen = time.time()
            agent_file = self.state_dir / "agents" / f"{self.agent_id}.json"
            with agent_file.open("w") as f:
                json.dump(agent.to_dict(), f, indent=2)

    # =========================================================================
    # Lock Management
    # =========================================================================

    def acquire_lock(
        self,
        files: List[str],
        lock_type: LockType = LockType.EXCLUSIVE,
        timeout_s: Optional[int] = None,
        reason: str = "",
        wait: bool = False,
        wait_timeout_s: int = 30,
    ) -> Optional[WorkspaceLock]:
        """
        Acquire a lock on files.

        Args:
            files: Files to lock
            lock_type: Type of lock to acquire
            timeout_s: Lock timeout (defaults to global setting)
            reason: Reason for acquiring lock
            wait: Whether to wait for conflicting locks
            wait_timeout_s: How long to wait

        Returns:
            WorkspaceLock if acquired, None if failed
        """
        file_set = set(files)
        timeout = timeout_s or self.lock_timeout_s

        # Check for conflicts
        start_time = time.time()
        while True:
            conflicts = self._check_lock_conflicts(file_set, lock_type)

            if not conflicts:
                break

            if not wait:
                return None

            if time.time() - start_time > wait_timeout_s:
                return None

            time.sleep(0.5)
            self._cleanup_expired_locks()

        # Create lock
        lock = WorkspaceLock(
            id=f"lock-{uuid.uuid4().hex[:12]}",
            agent_id=self.agent_id,
            files=file_set,
            lock_type=lock_type,
            acquired_at=time.time(),
            expires_at=time.time() + timeout if timeout else None,
            reason=reason,
        )

        self._locks[lock.id] = lock
        self._save_lock(lock)

        # Update agent state
        agent = self._agents.get(self.agent_id)
        if agent:
            agent.held_locks.append(lock.id)
            self._save_agent_state()

        # Emit event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["lock_acquired"],
                "kind": "lock",
                "actor": self.agent_id,
                "data": lock.to_dict(),
            })

        return lock

    def release_lock(self, lock_id: str) -> bool:
        """Release a lock."""
        lock = self._locks.get(lock_id)
        if not lock:
            return False

        if lock.agent_id != self.agent_id:
            return False  # Can't release another agent's lock

        del self._locks[lock_id]
        self._remove_lock_file(lock_id)

        # Update agent state
        agent = self._agents.get(self.agent_id)
        if agent and lock_id in agent.held_locks:
            agent.held_locks.remove(lock_id)
            self._save_agent_state()

        # Emit event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["lock_released"],
                "kind": "lock",
                "actor": self.agent_id,
                "data": {"lock_id": lock_id, "files": list(lock.files)},
            })

        return True

    def _check_lock_conflicts(
        self,
        files: Set[str],
        lock_type: LockType,
    ) -> List[WorkspaceLock]:
        """Check for conflicting locks."""
        conflicts = []

        for lock in self._locks.values():
            if lock.is_expired:
                continue

            if lock.agent_id == self.agent_id:
                continue  # Own locks don't conflict

            # Check for file overlap
            overlap = files & lock.files
            if not overlap:
                continue

            # Check lock type compatibility
            if lock.is_exclusive or lock_type == LockType.EXCLUSIVE:
                conflicts.append(lock)

        return conflicts

    def _save_lock(self, lock: WorkspaceLock) -> None:
        """Save lock to storage."""
        lock_file = self.state_dir / "locks" / f"{lock.id}.json"
        with lock_file.open("w") as f:
            json.dump(lock.to_dict(), f, indent=2)

    def _remove_lock_file(self, lock_id: str) -> None:
        """Remove lock file."""
        lock_file = self.state_dir / "locks" / f"{lock_id}.json"
        if lock_file.exists():
            lock_file.unlink()

    def _cleanup_expired_locks(self) -> None:
        """Remove expired locks."""
        expired = [lid for lid, lock in self._locks.items() if lock.is_expired]
        for lock_id in expired:
            del self._locks[lock_id]
            self._remove_lock_file(lock_id)

    def get_file_locks(self, path: str) -> List[WorkspaceLock]:
        """Get all locks affecting a file."""
        return [lock for lock in self._locks.values() if path in lock.files]

    def is_file_locked(self, path: str, exclude_self: bool = True) -> bool:
        """Check if a file is locked."""
        for lock in self._locks.values():
            if lock.is_expired:
                continue
            if path in lock.files:
                if not exclude_self or lock.agent_id != self.agent_id:
                    return True
        return False

    # =========================================================================
    # Change Synchronization
    # =========================================================================

    def push_changes(self, changes: List[FileChange]) -> bool:
        """
        Push changes to sync with other agents.

        Args:
            changes: List of changes to push

        Returns:
            True if push successful
        """
        for change in changes:
            change.agent_id = self.agent_id
            self._pending_changes.append(change)
            self._save_change(change)

        # Update file versions
        for change in changes:
            if change.new_hash:
                self._file_versions[change.path] = change.new_hash

        # Emit sync event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["sync_push"],
                "kind": "sync",
                "actor": self.agent_id,
                "data": {
                    "change_count": len(changes),
                    "sequence": self._sync_sequence,
                    "changes": [c.to_dict() for c in changes],
                },
            })

        self._sync_sequence += 1

        return True

    def pull_changes(self, since_sequence: Optional[int] = None) -> List[FileChange]:
        """
        Pull changes from other agents.

        Args:
            since_sequence: Only get changes after this sequence

        Returns:
            List of changes from other agents
        """
        changes = []

        # Load changes from storage
        changes_dir = self.state_dir / "changes"
        for change_file in sorted(changes_dir.glob("*.json")):
            try:
                with change_file.open() as f:
                    data = json.load(f)

                if data["agent_id"] == self.agent_id:
                    continue  # Skip own changes

                change = FileChange(
                    id=data["id"],
                    path=data["path"],
                    change_type=ChangeType(data["change_type"]),
                    agent_id=data["agent_id"],
                    timestamp=data["timestamp"],
                    old_hash=data.get("old_hash"),
                    new_hash=data.get("new_hash"),
                    size=data.get("size", 0),
                )
                changes.append(change)

            except Exception:
                pass

        # Filter by sequence if specified
        if since_sequence is not None:
            changes = [c for c in changes if c.timestamp > since_sequence]

        # Emit pull event
        if self.bus and changes:
            self.bus.emit({
                "topic": self.BUS_TOPICS["sync_pull"],
                "kind": "sync",
                "actor": self.agent_id,
                "data": {
                    "change_count": len(changes),
                },
            })

        return changes

    def _save_change(self, change: FileChange) -> None:
        """Save change to storage."""
        change_file = self.state_dir / "changes" / f"{change.id}.json"
        with change_file.open("w") as f:
            json.dump(change.to_dict(), f, indent=2)

    def track_file_change(
        self,
        path: str,
        change_type: ChangeType,
        old_content: Optional[str] = None,
        new_content: Optional[str] = None,
    ) -> FileChange:
        """Track a file change for synchronization."""
        old_hash = hashlib.sha256(old_content.encode()).hexdigest()[:16] if old_content else None
        new_hash = hashlib.sha256(new_content.encode()).hexdigest()[:16] if new_content else None

        change = FileChange(
            id=f"chg-{uuid.uuid4().hex[:12]}",
            path=path,
            change_type=change_type,
            agent_id=self.agent_id,
            timestamp=time.time(),
            old_hash=old_hash,
            new_hash=new_hash,
            size=len(new_content) if new_content else 0,
        )

        return change

    # =========================================================================
    # Agent Management
    # =========================================================================

    def heartbeat(self) -> None:
        """Send heartbeat to indicate agent is alive."""
        agent = self._agents.get(self.agent_id)
        if agent:
            agent.last_seen = time.time()
            self._save_agent_state()

            if self.bus:
                self.bus.emit({
                    "topic": self.BUS_TOPICS["agent_heartbeat"],
                    "kind": "heartbeat",
                    "actor": self.agent_id,
                    "data": agent.to_dict(),
                })

    def get_online_agents(self) -> List[AgentState]:
        """Get list of currently online agents."""
        self._load_state()  # Refresh state
        return [a for a in self._agents.values() if a.is_online]

    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get state of a specific agent."""
        return self._agents.get(agent_id)

    # =========================================================================
    # Conflict Detection
    # =========================================================================

    def check_conflicts(self, path: str) -> List[FileChange]:
        """Check for conflicting changes to a file."""
        conflicts = []

        # Get current file hash
        full_path = self.working_dir / path
        if full_path.exists():
            content = full_path.read_text()
            current_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        else:
            current_hash = None

        # Check pending changes from other agents
        for change in self.pull_changes():
            if change.path == path:
                if change.old_hash and change.old_hash != current_hash:
                    conflicts.append(change)

        return conflicts

    def resolve_conflict(
        self,
        path: str,
        resolution: str,  # "ours", "theirs", "manual"
        manual_content: Optional[str] = None,
    ) -> bool:
        """Resolve a sync conflict."""
        if resolution == "ours":
            # Keep local version, mark as resolved
            return True

        elif resolution == "theirs":
            # Apply remote change
            changes = [c for c in self.pull_changes() if c.path == path]
            if changes:
                latest = max(changes, key=lambda c: c.timestamp)
                if latest.content:
                    full_path = self.working_dir / path
                    full_path.write_text(latest.content)
                return True

        elif resolution == "manual" and manual_content:
            full_path = self.working_dir / path
            full_path.write_text(manual_content)
            return True

        return False

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_workspace_state(self) -> Dict[str, Any]:
        """Get current workspace state."""
        return {
            "agent_id": self.agent_id,
            "sync_state": self._agents.get(self.agent_id, AgentState("", 0, SyncState.OFFLINE)).sync_state.value,
            "active_locks": len(self._locks),
            "pending_changes": len(self._pending_changes),
            "online_agents": len(self.get_online_agents()),
            "file_versions": len(self._file_versions),
        }

    def list_locks(self) -> List[WorkspaceLock]:
        """List all active locks."""
        self._cleanup_expired_locks()
        return list(self._locks.values())

    def list_agents(self) -> List[AgentState]:
        """List all known agents."""
        return list(self._agents.values())


# =============================================================================
# Context Manager
# =============================================================================

class LockContext:
    """Context manager for file locks."""

    def __init__(
        self,
        sync: WorkspaceSync,
        files: List[str],
        lock_type: LockType = LockType.EXCLUSIVE,
        reason: str = "",
    ):
        self.sync = sync
        self.files = files
        self.lock_type = lock_type
        self.reason = reason
        self.lock: Optional[WorkspaceLock] = None

    def __enter__(self) -> WorkspaceLock:
        self.lock = self.sync.acquire_lock(
            self.files, self.lock_type, reason=self.reason, wait=True
        )
        if not self.lock:
            raise RuntimeError(f"Could not acquire lock on {self.files}")
        return self.lock

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock:
            self.sync.release_lock(self.lock.id)
        return False


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Workspace Sync."""
    import argparse

    parser = argparse.ArgumentParser(description="Workspace Sync (Step 68)")
    parser.add_argument("--working-dir", default=".", help="Working directory")
    parser.add_argument("--agent-id", default="cli-agent", help="Agent ID")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # lock command
    lock_parser = subparsers.add_parser("lock", help="Acquire lock")
    lock_parser.add_argument("files", nargs="+", help="Files to lock")
    lock_parser.add_argument("--type", choices=["exclusive", "shared"], default="exclusive")
    lock_parser.add_argument("--reason", default="", help="Lock reason")

    # unlock command
    unlock_parser = subparsers.add_parser("unlock", help="Release lock")
    unlock_parser.add_argument("lock_id", help="Lock ID to release")

    # locks command
    subparsers.add_parser("locks", help="List all locks")

    # agents command
    subparsers.add_parser("agents", help="List agents")

    # status command
    subparsers.add_parser("status", help="Show workspace status")

    # push command
    push_parser = subparsers.add_parser("push", help="Push changes")
    push_parser.add_argument("--file", required=True, help="File to track")
    push_parser.add_argument("--type", choices=["create", "modify", "delete"], default="modify")

    # pull command
    subparsers.add_parser("pull", help="Pull changes")

    args = parser.parse_args()

    sync = WorkspaceSync(Path(args.working_dir), agent_id=args.agent_id)

    if args.command == "lock":
        lock_type = LockType.EXCLUSIVE if args.type == "exclusive" else LockType.SHARED
        lock = sync.acquire_lock(args.files, lock_type, reason=args.reason)
        if lock:
            print(f"Acquired lock: {lock.id}")
            print(f"  Files: {list(lock.files)}")
            print(f"  Type: {lock.lock_type.value}")
        else:
            print("Failed to acquire lock (files may be locked by another agent)")
            return 1
        return 0

    elif args.command == "unlock":
        if sync.release_lock(args.lock_id):
            print(f"Released lock: {args.lock_id}")
        else:
            print(f"Failed to release lock: {args.lock_id}")
            return 1
        return 0

    elif args.command == "locks":
        locks = sync.list_locks()
        if not locks:
            print("No active locks")
        else:
            for lock in locks:
                print(f"{lock.id} [{lock.lock_type.value}] {lock.agent_id}: {list(lock.files)}")
        return 0

    elif args.command == "agents":
        agents = sync.list_agents()
        for agent in agents:
            status = "online" if agent.is_online else "offline"
            print(f"{agent.agent_id} [{status}] locks={len(agent.held_locks)} changes={agent.pending_changes}")
        return 0

    elif args.command == "status":
        state = sync.get_workspace_state()
        print(f"Agent ID: {state['agent_id']}")
        print(f"Sync State: {state['sync_state']}")
        print(f"Active Locks: {state['active_locks']}")
        print(f"Pending Changes: {state['pending_changes']}")
        print(f"Online Agents: {state['online_agents']}")
        return 0

    elif args.command == "push":
        change = sync.track_file_change(
            args.file,
            ChangeType(args.type),
        )
        sync.push_changes([change])
        print(f"Pushed change: {change.id}")
        return 0

    elif args.command == "pull":
        changes = sync.pull_changes()
        print(f"Pulled {len(changes)} changes")
        for c in changes:
            print(f"  {c.id}: {c.change_type.value} {c.path} by {c.agent_id}")
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
