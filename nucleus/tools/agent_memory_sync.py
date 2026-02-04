#!/usr/bin/env python3
"""
Cross-Agent Memory Synchronization (Step 18)
=============================================

Enables memory sharing and synchronization between Pluribus agents.

Features:
- Agent-specific namespaces with shared memory spaces
- CRDT-based conflict resolution (LWW, G-Counter, OR-Set)
- Vector clock synchronization
- Peer-to-peer and hub-spoke topologies
- Selective memory sharing based on permissions

Run with:
    python3 nucleus/tools/agent_memory_sync.py sync --agent claude --peer codex
    python3 nucleus/tools/agent_memory_sync.py share --memory-id abc123 --with codex
"""
from __future__ import annotations

import abc
import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generic, List, Optional,
    Set, Tuple, TypeVar, Union,
)

sys.dont_write_bytecode = True

logger = logging.getLogger("agent_memory_sync")


# =============================================================================
# Vector Clocks for Causality Tracking
# =============================================================================

@dataclass
class VectorClock:
    """
    Vector clock for tracking causal relationships.
    Maps agent_id -> logical timestamp.
    """
    clock: Dict[str, int] = field(default_factory=dict)

    def increment(self, agent_id: str) -> None:
        """Increment clock for an agent."""
        self.clock[agent_id] = self.clock.get(agent_id, 0) + 1

    def update(self, other: "VectorClock") -> None:
        """Merge with another clock (take max)."""
        for agent_id, ts in other.clock.items():
            self.clock[agent_id] = max(self.clock.get(agent_id, 0), ts)

    def get(self, agent_id: str) -> int:
        """Get timestamp for an agent."""
        return self.clock.get(agent_id, 0)

    def happens_before(self, other: "VectorClock") -> bool:
        """Check if this clock happens before other."""
        at_least_one_less = False
        for agent_id in set(self.clock.keys()) | set(other.clock.keys()):
            self_ts = self.clock.get(agent_id, 0)
            other_ts = other.clock.get(agent_id, 0)
            if self_ts > other_ts:
                return False
            if self_ts < other_ts:
                at_least_one_less = True
        return at_least_one_less

    def concurrent_with(self, other: "VectorClock") -> bool:
        """Check if clocks are concurrent (neither happens before)."""
        return not self.happens_before(other) and not other.happens_before(self)

    def to_dict(self) -> Dict[str, int]:
        return dict(self.clock)

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> "VectorClock":
        return cls(clock=dict(d))


# =============================================================================
# CRDTs for Conflict-Free Merging
# =============================================================================

class CRDT(abc.ABC):
    """Abstract base for Conflict-free Replicated Data Types."""

    @abc.abstractmethod
    def merge(self, other: "CRDT") -> "CRDT":
        """Merge with another CRDT."""
        pass

    @abc.abstractmethod
    def value(self) -> Any:
        """Get current value."""
        pass

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        pass


@dataclass
class LWWRegister(CRDT):
    """
    Last-Writer-Wins Register.
    Concurrent writes resolved by timestamp.
    """
    val: Any = None
    timestamp: float = 0.0
    agent_id: str = ""

    def set(self, value: Any, agent_id: str) -> None:
        self.val = value
        self.timestamp = time.time()
        self.agent_id = agent_id

    def merge(self, other: "LWWRegister") -> "LWWRegister":
        if other.timestamp > self.timestamp:
            return LWWRegister(val=other.val, timestamp=other.timestamp, agent_id=other.agent_id)
        elif other.timestamp == self.timestamp and other.agent_id > self.agent_id:
            # Tie-break by agent_id
            return LWWRegister(val=other.val, timestamp=other.timestamp, agent_id=other.agent_id)
        return self

    def value(self) -> Any:
        return self.val

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "lww", "value": self.val, "timestamp": self.timestamp, "agent_id": self.agent_id}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LWWRegister":
        return cls(val=d.get("value"), timestamp=d.get("timestamp", 0), agent_id=d.get("agent_id", ""))


@dataclass
class GCounter(CRDT):
    """
    Grow-only Counter.
    Each agent has its own counter that only increments.
    """
    counts: Dict[str, int] = field(default_factory=dict)

    def increment(self, agent_id: str, delta: int = 1) -> None:
        self.counts[agent_id] = self.counts.get(agent_id, 0) + delta

    def merge(self, other: "GCounter") -> "GCounter":
        merged = GCounter()
        for agent_id in set(self.counts.keys()) | set(other.counts.keys()):
            merged.counts[agent_id] = max(
                self.counts.get(agent_id, 0),
                other.counts.get(agent_id, 0)
            )
        return merged

    def value(self) -> int:
        return sum(self.counts.values())

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "gcounter", "counts": self.counts}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GCounter":
        return cls(counts=d.get("counts", {}))


@dataclass
class ORSet(CRDT):
    """
    Observed-Remove Set.
    Supports both add and remove with unique tags.
    """
    elements: Dict[str, Set[str]] = field(default_factory=dict)  # value -> set of tags
    tombstones: Set[str] = field(default_factory=set)  # removed tags

    def add(self, value: str, agent_id: str) -> None:
        tag = f"{agent_id}:{uuid.uuid4().hex[:8]}"
        if value not in self.elements:
            self.elements[value] = set()
        self.elements[value].add(tag)

    def remove(self, value: str) -> None:
        if value in self.elements:
            self.tombstones.update(self.elements[value])

    def merge(self, other: "ORSet") -> "ORSet":
        merged = ORSet()
        merged.tombstones = self.tombstones | other.tombstones

        for value in set(self.elements.keys()) | set(other.elements.keys()):
            tags = (self.elements.get(value, set()) | other.elements.get(value, set()))
            live_tags = tags - merged.tombstones
            if live_tags:
                merged.elements[value] = live_tags

        return merged

    def value(self) -> Set[str]:
        return set(self.elements.keys())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "orset",
            "elements": {k: list(v) for k, v in self.elements.items()},
            "tombstones": list(self.tombstones),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ORSet":
        elements = {k: set(v) for k, v in d.get("elements", {}).items()}
        tombstones = set(d.get("tombstones", []))
        return cls(elements=elements, tombstones=tombstones)


# =============================================================================
# Shared Memory Entry
# =============================================================================

class SharingLevel(Enum):
    """Memory sharing permissions."""
    PRIVATE = "private"      # Only owner can access
    SHARED = "shared"        # Specific agents can access
    PUBLIC = "public"        # All agents can access
    BROADCAST = "broadcast"  # Pushed to all agents


@dataclass
class SharedMemory:
    """Memory entry with sharing metadata."""
    id: str
    owner_agent: str
    content: Any
    sharing_level: SharingLevel
    shared_with: List[str]  # Agent IDs
    vector_clock: VectorClock
    crdt_type: str  # "lww", "gcounter", "orset"
    crdt_data: Dict[str, Any]
    created_at: float
    updated_at: float
    sync_version: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "owner_agent": self.owner_agent,
            "content": self.content,
            "sharing_level": self.sharing_level.value,
            "shared_with": self.shared_with,
            "vector_clock": self.vector_clock.to_dict(),
            "crdt_type": self.crdt_type,
            "crdt_data": self.crdt_data,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sync_version": self.sync_version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SharedMemory":
        return cls(
            id=d["id"],
            owner_agent=d["owner_agent"],
            content=d["content"],
            sharing_level=SharingLevel(d.get("sharing_level", "private")),
            shared_with=d.get("shared_with", []),
            vector_clock=VectorClock.from_dict(d.get("vector_clock", {})),
            crdt_type=d.get("crdt_type", "lww"),
            crdt_data=d.get("crdt_data", {}),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            sync_version=d.get("sync_version", 0),
        )


# =============================================================================
# Sync Protocol Messages
# =============================================================================

class SyncMessageType(Enum):
    """Types of sync protocol messages."""
    HELLO = "hello"          # Initial handshake
    CLOCK_SYNC = "clock_sync"  # Vector clock exchange
    PULL_REQUEST = "pull_request"  # Request updates
    PUSH_UPDATE = "push_update"  # Send updates
    ACK = "ack"              # Acknowledgment
    CONFLICT = "conflict"    # Conflict notification


@dataclass
class SyncMessage:
    """Message in sync protocol."""
    message_type: SyncMessageType
    sender_agent: str
    target_agent: str
    payload: Dict[str, Any]
    vector_clock: VectorClock
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_type": self.message_type.value,
            "sender_agent": self.sender_agent,
            "target_agent": self.target_agent,
            "payload": self.payload,
            "vector_clock": self.vector_clock.to_dict(),
            "timestamp": self.timestamp,
            "message_id": self.message_id,
        }


# =============================================================================
# Agent Memory Store
# =============================================================================

class AgentMemoryStore:
    """SQLite-based shared memory store."""

    def __init__(self, db_path: Path, agent_id: str):
        self.db_path = db_path
        self.agent_id = agent_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS shared_memories (
                id TEXT PRIMARY KEY,
                owner_agent TEXT NOT NULL,
                content TEXT NOT NULL,
                sharing_level TEXT DEFAULT 'private',
                shared_with TEXT DEFAULT '[]',
                vector_clock TEXT DEFAULT '{}',
                crdt_type TEXT DEFAULT 'lww',
                crdt_data TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                sync_version INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_shared_owner ON shared_memories(owner_agent);
            CREATE INDEX IF NOT EXISTS idx_shared_level ON shared_memories(sharing_level);

            -- Sync log for tracking what's been synced
            CREATE TABLE IF NOT EXISTS sync_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                peer_agent TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                sync_version INTEGER NOT NULL,
                synced_at REAL NOT NULL,
                direction TEXT NOT NULL  -- 'push' or 'pull'
            );

            CREATE INDEX IF NOT EXISTS idx_sync_peer ON sync_log(peer_agent);
        """)
        conn.commit()

    def save(self, memory: SharedMemory) -> bool:
        """Save or update a shared memory."""
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO shared_memories
                (id, owner_agent, content, sharing_level, shared_with,
                 vector_clock, crdt_type, crdt_data, created_at, updated_at, sync_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id,
                memory.owner_agent,
                json.dumps(memory.content),
                memory.sharing_level.value,
                json.dumps(memory.shared_with),
                json.dumps(memory.vector_clock.to_dict()),
                memory.crdt_type,
                json.dumps(memory.crdt_data),
                memory.created_at,
                memory.updated_at,
                memory.sync_version,
            ))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save shared memory: {e}")
            return False

    def get(self, memory_id: str) -> Optional[SharedMemory]:
        """Get a shared memory by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM shared_memories WHERE id = ?",
            (memory_id,),
        ).fetchone()

        if not row:
            return None

        return self._row_to_memory(row)

    def _row_to_memory(self, row: sqlite3.Row) -> SharedMemory:
        return SharedMemory(
            id=row["id"],
            owner_agent=row["owner_agent"],
            content=json.loads(row["content"]),
            sharing_level=SharingLevel(row["sharing_level"]),
            shared_with=json.loads(row["shared_with"] or "[]"),
            vector_clock=VectorClock.from_dict(json.loads(row["vector_clock"] or "{}")),
            crdt_type=row["crdt_type"],
            crdt_data=json.loads(row["crdt_data"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            sync_version=row["sync_version"],
        )

    def get_accessible(self, requesting_agent: str) -> List[SharedMemory]:
        """Get all memories accessible to an agent."""
        conn = self._get_conn()

        rows = conn.execute("""
            SELECT * FROM shared_memories
            WHERE owner_agent = ?
               OR sharing_level = 'public'
               OR sharing_level = 'broadcast'
               OR (sharing_level = 'shared' AND shared_with LIKE ?)
        """, (requesting_agent, f'%"{requesting_agent}"%')).fetchall()

        return [self._row_to_memory(row) for row in rows]

    def get_for_sync(self, peer_agent: str, since_version: int = 0) -> List[SharedMemory]:
        """Get memories that need to be synced with a peer."""
        conn = self._get_conn()

        rows = conn.execute("""
            SELECT * FROM shared_memories
            WHERE sync_version > ?
              AND (owner_agent = ?
                   OR sharing_level = 'public'
                   OR sharing_level = 'broadcast'
                   OR (sharing_level = 'shared' AND shared_with LIKE ?))
        """, (since_version, self.agent_id, f'%"{peer_agent}"%')).fetchall()

        return [self._row_to_memory(row) for row in rows]

    def log_sync(self, peer_agent: str, memory_id: str, sync_version: int, direction: str) -> None:
        """Log a sync operation."""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO sync_log (peer_agent, memory_id, sync_version, synced_at, direction)
            VALUES (?, ?, ?, ?, ?)
        """, (peer_agent, memory_id, sync_version, time.time(), direction))
        conn.commit()

    def get_last_sync_version(self, peer_agent: str) -> int:
        """Get the last synced version with a peer."""
        conn = self._get_conn()
        row = conn.execute("""
            SELECT MAX(sync_version) as max_version FROM sync_log WHERE peer_agent = ?
        """, (peer_agent,)).fetchone()
        return row["max_version"] or 0

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()


# =============================================================================
# Memory Sync Manager
# =============================================================================

class MemorySyncManager:
    """
    Manages memory synchronization between agents.
    """

    def __init__(self, store: AgentMemoryStore, agent_id: str):
        self.store = store
        self.agent_id = agent_id
        self.vector_clock = VectorClock()
        self._sync_callbacks: List[Callable[[SharedMemory], None]] = []

    def create_shared_memory(
        self,
        content: Any,
        sharing_level: SharingLevel = SharingLevel.PRIVATE,
        shared_with: Optional[List[str]] = None,
        crdt_type: str = "lww",
    ) -> SharedMemory:
        """Create a new shared memory."""
        self.vector_clock.increment(self.agent_id)

        memory = SharedMemory(
            id=uuid.uuid4().hex,
            owner_agent=self.agent_id,
            content=content,
            sharing_level=sharing_level,
            shared_with=shared_with or [],
            vector_clock=VectorClock(clock=dict(self.vector_clock.clock)),
            crdt_type=crdt_type,
            crdt_data=self._init_crdt_data(crdt_type, content),
            created_at=time.time(),
            updated_at=time.time(),
            sync_version=self.vector_clock.get(self.agent_id),
        )

        self.store.save(memory)
        return memory

    def _init_crdt_data(self, crdt_type: str, content: Any) -> Dict[str, Any]:
        """Initialize CRDT data based on type."""
        if crdt_type == "lww":
            return LWWRegister(val=content, timestamp=time.time(), agent_id=self.agent_id).to_dict()
        elif crdt_type == "gcounter":
            return GCounter().to_dict()
        elif crdt_type == "orset":
            return ORSet().to_dict()
        return {}

    def update_memory(self, memory_id: str, content: Any) -> Optional[SharedMemory]:
        """Update a shared memory."""
        memory = self.store.get(memory_id)
        if not memory:
            return None

        self.vector_clock.increment(self.agent_id)

        # Update based on CRDT type
        if memory.crdt_type == "lww":
            crdt = LWWRegister.from_dict(memory.crdt_data)
            crdt.set(content, self.agent_id)
            memory.crdt_data = crdt.to_dict()
            memory.content = content

        memory.vector_clock.update(self.vector_clock)
        memory.updated_at = time.time()
        memory.sync_version = self.vector_clock.get(self.agent_id)

        self.store.save(memory)
        return memory

    def share_with(self, memory_id: str, agent_ids: List[str]) -> bool:
        """Share a memory with specific agents."""
        memory = self.store.get(memory_id)
        if not memory or memory.owner_agent != self.agent_id:
            return False

        memory.sharing_level = SharingLevel.SHARED
        memory.shared_with = list(set(memory.shared_with + agent_ids))
        memory.updated_at = time.time()

        return self.store.save(memory)

    def sync_with_peer(self, peer_agent: str) -> Dict[str, int]:
        """
        Synchronize memories with a peer agent.

        Returns:
            Dict with counts of pushed, pulled, and conflicts
        """
        stats = {"pushed": 0, "pulled": 0, "conflicts": 0}

        # Get memories to push
        last_sync = self.store.get_last_sync_version(peer_agent)
        to_push = self.store.get_for_sync(peer_agent, last_sync)

        for memory in to_push:
            # In real implementation, this would send to peer
            # Here we just log it
            self.store.log_sync(peer_agent, memory.id, memory.sync_version, "push")
            stats["pushed"] += 1

        return stats

    def receive_update(self, memory: SharedMemory, from_agent: str) -> Optional[SharedMemory]:
        """
        Receive an update from another agent.
        Handles conflict resolution using CRDTs.
        """
        existing = self.store.get(memory.id)

        if not existing:
            # New memory, just save
            self.store.save(memory)
            self._notify_callbacks(memory)
            return memory

        # Merge using vector clock and CRDT
        if memory.vector_clock.happens_before(existing.vector_clock):
            # Incoming is older, ignore
            return existing

        if existing.vector_clock.happens_before(memory.vector_clock):
            # Incoming is newer, accept
            self.store.save(memory)
            self._notify_callbacks(memory)
            return memory

        # Concurrent updates - use CRDT merge
        merged = self._merge_memories(existing, memory)
        self.store.save(merged)
        self._notify_callbacks(merged)
        return merged

    def _merge_memories(self, local: SharedMemory, remote: SharedMemory) -> SharedMemory:
        """Merge two concurrent memory versions using CRDT."""
        crdt_type = local.crdt_type

        if crdt_type == "lww":
            local_crdt = LWWRegister.from_dict(local.crdt_data)
            remote_crdt = LWWRegister.from_dict(remote.crdt_data)
            merged_crdt = local_crdt.merge(remote_crdt)
        elif crdt_type == "gcounter":
            local_crdt = GCounter.from_dict(local.crdt_data)
            remote_crdt = GCounter.from_dict(remote.crdt_data)
            merged_crdt = local_crdt.merge(remote_crdt)
        elif crdt_type == "orset":
            local_crdt = ORSet.from_dict(local.crdt_data)
            remote_crdt = ORSet.from_dict(remote.crdt_data)
            merged_crdt = local_crdt.merge(remote_crdt)
        else:
            # Default to LWW behavior
            merged_crdt = LWWRegister.from_dict(remote.crdt_data)

        # Create merged memory
        merged = SharedMemory(
            id=local.id,
            owner_agent=local.owner_agent,
            content=merged_crdt.value(),
            sharing_level=local.sharing_level,
            shared_with=list(set(local.shared_with + remote.shared_with)),
            vector_clock=VectorClock(clock=dict(local.vector_clock.clock)),
            crdt_type=crdt_type,
            crdt_data=merged_crdt.to_dict(),
            created_at=min(local.created_at, remote.created_at),
            updated_at=max(local.updated_at, remote.updated_at),
            sync_version=max(local.sync_version, remote.sync_version),
        )
        merged.vector_clock.update(remote.vector_clock)

        return merged

    def on_sync(self, callback: Callable[[SharedMemory], None]) -> None:
        """Register callback for sync updates."""
        self._sync_callbacks.append(callback)

    def _notify_callbacks(self, memory: SharedMemory) -> None:
        """Notify registered callbacks of memory update."""
        for callback in self._sync_callbacks:
            try:
                callback(memory)
            except Exception as e:
                logger.error(f"Sync callback failed: {e}")


# =============================================================================
# CLI Interface
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-Agent Memory Synchronization"
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("PLURIBUS_ROOT", "/pluribus"),
        help="Pluribus root directory",
    )
    parser.add_argument(
        "--agent", "-a",
        default="claude",
        help="Current agent ID",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create command
    create_p = subparsers.add_parser("create", help="Create shared memory")
    create_p.add_argument("content", help="Memory content")
    create_p.add_argument("--level", choices=["private", "shared", "public"],
                          default="private", help="Sharing level")
    create_p.add_argument("--with", dest="shared_with", nargs="*",
                          help="Agent IDs to share with")

    # Share command
    share_p = subparsers.add_parser("share", help="Share memory with agents")
    share_p.add_argument("--memory-id", required=True, help="Memory ID")
    share_p.add_argument("--with", dest="agents", nargs="+", required=True,
                         help="Agent IDs to share with")

    # Sync command
    sync_p = subparsers.add_parser("sync", help="Sync with peer")
    sync_p.add_argument("--peer", required=True, help="Peer agent ID")

    # List command
    list_p = subparsers.add_parser("list", help="List accessible memories")
    list_p.add_argument("--for-agent", help="Agent to check access for")

    # Stats command
    subparsers.add_parser("stats", help="Show sync statistics")

    args = parser.parse_args()
    root = Path(args.root)

    # Initialize store and manager
    db_path = root / ".pluribus" / "memory" / f"shared_{args.agent}.sqlite3"
    store = AgentMemoryStore(db_path, args.agent)
    manager = MemorySyncManager(store, args.agent)

    result: Any = None

    try:
        if args.command == "create":
            try:
                content = json.loads(args.content)
            except json.JSONDecodeError:
                content = args.content

            level = SharingLevel(args.level)
            memory = manager.create_shared_memory(
                content=content,
                sharing_level=level,
                shared_with=args.shared_with or [],
            )
            result = {"success": True, "memory": memory.to_dict()}

        elif args.command == "share":
            success = manager.share_with(args.memory_id, args.agents)
            result = {"success": success, "memory_id": args.memory_id, "shared_with": args.agents}

        elif args.command == "sync":
            stats = manager.sync_with_peer(args.peer)
            result = {"peer": args.peer, "stats": stats}

        elif args.command == "list":
            for_agent = args.for_agent or args.agent
            memories = store.get_accessible(for_agent)
            result = {
                "agent": for_agent,
                "count": len(memories),
                "memories": [m.to_dict() for m in memories[:50]],
            }

        elif args.command == "stats":
            conn = store._get_conn()
            rows = conn.execute("""
                SELECT peer_agent, direction, COUNT(*) as count
                FROM sync_log
                GROUP BY peer_agent, direction
            """).fetchall()
            result = {
                "agent": args.agent,
                "sync_history": [
                    {"peer": r["peer_agent"], "direction": r["direction"], "count": r["count"]}
                    for r in rows
                ],
            }

        print(json.dumps(result, indent=2))

    finally:
        store.close()

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
