#!/usr/bin/env python3
"""Mem0 Adapter for Pluribus Agent Long-Term Memory.

Provides MCP-compatible memory operations for agents:
- store: Persist memories with metadata and embeddings
- retrieve: Semantic search over stored memories
- update: Modify existing memories
- delete: Remove memories
- list: Browse memories by agent/session

Integrates with:
- Bus events: Auto-extract and store memories from bus activity
- MCP protocol: Exposes memory tools via JSON-RPC
- Temporal tracking: Memories have created_at, accessed_at timestamps

Effects: R(file), W(file)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_ndjson(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def simple_hash_embed(text: str, dim: int = 64) -> list[float]:
    """Simple deterministic pseudo-embedding for testing.

    In production, replace with actual embedding model (OpenAI, local, etc).
    This uses a hash-based approach that provides consistent vectors for identical text.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [float(b) / 255.0 for b in h]
    # Extend to desired dimension
    while len(raw) < dim:
        h = hashlib.sha256(h).digest()
        raw.extend(float(b) / 255.0 for b in h)
    return raw[:dim]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class Memory:
    """A single memory record."""
    id: str
    agent_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)
    created_at: str = ""
    accessed_at: str = ""
    access_count: int = 0
    session_id: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": "mem0_memory",
            "id": self.id,
            "agent_id": self.agent_id,
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "access_count": self.access_count,
            "session_id": self.session_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        return cls(
            id=str(d.get("id") or ""),
            agent_id=str(d.get("agent_id") or ""),
            content=str(d.get("content") or ""),
            metadata=d.get("metadata") if isinstance(d.get("metadata"), dict) else {},
            embedding=d.get("embedding") if isinstance(d.get("embedding"), list) else [],
            created_at=str(d.get("created_at") or ""),
            accessed_at=str(d.get("accessed_at") or ""),
            access_count=int(d.get("access_count") or 0),
            session_id=d.get("session_id") if isinstance(d.get("session_id"), str) else None,
            tags=d.get("tags") if isinstance(d.get("tags"), list) else [],
        )


class Mem0Service:
    """Service layer for Mem0 memory operations."""

    def __init__(self, root: Path | str, embed_dim: int = 64):
        self.root = Path(root) if isinstance(root, str) else root
        self.embed_dim = embed_dim

    @property
    def memories_path(self) -> Path:
        return self.root / ".pluribus" / "memory" / "mem0_store.ndjson"

    @property
    def index_path(self) -> Path:
        return self.root / ".pluribus" / "memory" / "mem0_index.ndjson"

    def _load_all_memories(self) -> dict[str, Memory]:
        """Load all memories indexed by ID."""
        memories: dict[str, Memory] = {}
        for obj in iter_ndjson(self.memories_path):
            if obj.get("kind") == "mem0_memory":
                mem = Memory.from_dict(obj)
                if mem.id:
                    memories[mem.id] = mem
        return memories

    def _write_memory(self, memory: Memory) -> None:
        """Append a memory record to storage."""
        append_ndjson(self.memories_path, memory.to_dict())

    def _log_access(self, memory_id: str, action: str) -> None:
        """Log memory access for analytics."""
        append_ndjson(self.index_path, {
            "kind": "mem0_access",
            "memory_id": memory_id,
            "action": action,
            "ts": time.time(),
            "iso": now_iso_utc(),
        })

    def store(
        self,
        *,
        agent_id: str,
        content: str,
        metadata: dict | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Store a new memory.

        Args:
            agent_id: The agent storing the memory
            content: The memory content (text)
            metadata: Optional metadata dict
            session_id: Optional session identifier
            tags: Optional list of tags for categorization

        Returns:
            Dict with memory id and status
        """
        if not agent_id:
            return {"error": "agent_id is required"}
        if not content:
            return {"error": "content is required"}

        memory_id = str(uuid.uuid4())
        now = now_iso_utc()
        embedding = simple_hash_embed(content, self.embed_dim)

        memory = Memory(
            id=memory_id,
            agent_id=agent_id,
            content=content,
            metadata=metadata or {},
            embedding=embedding,
            created_at=now,
            accessed_at=now,
            access_count=0,
            session_id=session_id,
            tags=tags or [],
        )

        self._write_memory(memory)
        self._log_access(memory_id, "store")

        return {
            "id": memory_id,
            "agent_id": agent_id,
            "created_at": now,
            "status": "stored",
        }

    def retrieve(
        self,
        *,
        query: str,
        agent_id: str | None = None,
        limit: int = 10,
        threshold: float = 0.3,
        tags: list[str] | None = None,
    ) -> dict:
        """Retrieve memories by semantic similarity.

        Args:
            query: The query text to search for
            agent_id: Optional filter by agent
            limit: Maximum results to return
            threshold: Minimum similarity threshold
            tags: Optional filter by tags (any match)

        Returns:
            Dict with list of matching memories
        """
        if not query:
            return {"error": "query is required", "memories": []}

        query_embedding = simple_hash_embed(query, self.embed_dim)
        all_memories = self._load_all_memories()

        scored: list[tuple[float, Memory]] = []
        for mem in all_memories.values():
            # Filter by agent if specified
            if agent_id and mem.agent_id != agent_id:
                continue
            # Filter by tags if specified
            if tags and not any(t in mem.tags for t in tags):
                continue

            score = cosine_similarity(query_embedding, mem.embedding)
            if score >= threshold:
                scored.append((score, mem))

        # Sort by similarity (descending)
        scored.sort(key=lambda x: -x[0])
        results = scored[:limit]

        # Log access for retrieved memories
        for _, mem in results:
            self._log_access(mem.id, "retrieve")

        return {
            "memories": [
                {
                    "id": mem.id,
                    "agent_id": mem.agent_id,
                    "content": mem.content,
                    "metadata": mem.metadata,
                    "similarity": round(score, 4),
                    "tags": mem.tags,
                    "created_at": mem.created_at,
                }
                for score, mem in results
            ],
            "count": len(results),
            "query": query,
        }

    def update(
        self,
        *,
        memory_id: str,
        content: str | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        """Update an existing memory.

        Args:
            memory_id: The ID of the memory to update
            content: New content (optional)
            metadata: New metadata (optional, merged)
            tags: New tags (optional, replaces)

        Returns:
            Dict with update status
        """
        if not memory_id:
            return {"error": "memory_id is required"}

        all_memories = self._load_all_memories()
        if memory_id not in all_memories:
            return {"error": f"memory not found: {memory_id}"}

        mem = all_memories[memory_id]
        now = now_iso_utc()

        if content is not None:
            mem.content = content
            mem.embedding = simple_hash_embed(content, self.embed_dim)
        if metadata is not None:
            mem.metadata = {**mem.metadata, **metadata}
        if tags is not None:
            mem.tags = tags

        mem.accessed_at = now
        mem.access_count += 1

        self._write_memory(mem)
        self._log_access(memory_id, "update")

        return {
            "id": memory_id,
            "status": "updated",
            "accessed_at": now,
        }

    def delete(self, *, memory_id: str) -> dict:
        """Mark a memory as deleted.

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            Dict with deletion status
        """
        if not memory_id:
            return {"error": "memory_id is required"}

        # Write a deletion marker
        append_ndjson(self.memories_path, {
            "kind": "mem0_deletion",
            "memory_id": memory_id,
            "deleted_at": now_iso_utc(),
            "ts": time.time(),
        })
        self._log_access(memory_id, "delete")

        return {"id": memory_id, "status": "deleted"}

    def list_memories(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """List memories with optional filtering.

        Args:
            agent_id: Optional filter by agent
            session_id: Optional filter by session
            limit: Maximum results
            offset: Skip this many results

        Returns:
            Dict with list of memories
        """
        all_memories = self._load_all_memories()

        # Get deleted IDs
        deleted_ids: set[str] = set()
        for obj in iter_ndjson(self.memories_path):
            if obj.get("kind") == "mem0_deletion":
                mid = obj.get("memory_id")
                if isinstance(mid, str):
                    deleted_ids.add(mid)

        # Filter
        filtered: list[Memory] = []
        for mem in all_memories.values():
            if mem.id in deleted_ids:
                continue
            if agent_id and mem.agent_id != agent_id:
                continue
            if session_id and mem.session_id != session_id:
                continue
            filtered.append(mem)

        # Sort by created_at (newest first)
        filtered.sort(key=lambda m: m.created_at, reverse=True)

        # Paginate
        page = filtered[offset : offset + limit]

        return {
            "memories": [
                {
                    "id": m.id,
                    "agent_id": m.agent_id,
                    "content": m.content[:200] + ("..." if len(m.content) > 200 else ""),
                    "tags": m.tags,
                    "created_at": m.created_at,
                }
                for m in page
            ],
            "total": len(filtered),
            "offset": offset,
            "limit": limit,
        }

    def tools_list(self) -> dict:
        """Return MCP tools list."""
        return {
            "tools": [
                {
                    "name": "mem0_store",
                    "description": "Store a memory for an agent",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string", "description": "Agent identifier"},
                            "content": {"type": "string", "description": "Memory content"},
                            "metadata": {"type": "object", "description": "Optional metadata"},
                            "session_id": {"type": "string", "description": "Optional session ID"},
                            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"},
                        },
                        "required": ["agent_id", "content"],
                    },
                },
                {
                    "name": "mem0_retrieve",
                    "description": "Retrieve memories by semantic similarity",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "agent_id": {"type": "string", "description": "Optional agent filter"},
                            "limit": {"type": "integer", "default": 10},
                            "threshold": {"type": "number", "default": 0.3},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "mem0_update",
                    "description": "Update an existing memory",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string"},
                            "content": {"type": "string"},
                            "metadata": {"type": "object"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["memory_id"],
                    },
                },
                {
                    "name": "mem0_delete",
                    "description": "Delete a memory",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string"},
                        },
                        "required": ["memory_id"],
                    },
                },
                {
                    "name": "mem0_list",
                    "description": "List memories with optional filtering",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "agent_id": {"type": "string"},
                            "session_id": {"type": "string"},
                            "limit": {"type": "integer", "default": 50},
                            "offset": {"type": "integer", "default": 0},
                        },
                    },
                },
            ]
        }

    def tools_call(self, name: str, args: dict) -> dict:
        """Dispatch an MCP tool call."""
        if name == "mem0_store":
            return self.store(
                agent_id=str(args.get("agent_id") or ""),
                content=str(args.get("content") or ""),
                metadata=args.get("metadata") if isinstance(args.get("metadata"), dict) else None,
                session_id=args.get("session_id") if isinstance(args.get("session_id"), str) else None,
                tags=args.get("tags") if isinstance(args.get("tags"), list) else None,
            )
        if name == "mem0_retrieve":
            return self.retrieve(
                query=str(args.get("query") or ""),
                agent_id=args.get("agent_id") if isinstance(args.get("agent_id"), str) else None,
                limit=int(args.get("limit") or 10),
                threshold=float(args.get("threshold") or 0.3),
                tags=args.get("tags") if isinstance(args.get("tags"), list) else None,
            )
        if name == "mem0_update":
            return self.update(
                memory_id=str(args.get("memory_id") or ""),
                content=args.get("content") if isinstance(args.get("content"), str) else None,
                metadata=args.get("metadata") if isinstance(args.get("metadata"), dict) else None,
                tags=args.get("tags") if isinstance(args.get("tags"), list) else None,
            )
        if name == "mem0_delete":
            return self.delete(memory_id=str(args.get("memory_id") or ""))
        if name == "mem0_list":
            return self.list_memories(
                agent_id=args.get("agent_id") if isinstance(args.get("agent_id"), str) else None,
                session_id=args.get("session_id") if isinstance(args.get("session_id"), str) else None,
                limit=int(args.get("limit") or 50),
                offset=int(args.get("offset") or 0),
            )
        return {"error": f"unknown tool: {name}"}


def extract_memories_from_bus_event(event: dict) -> list[dict]:
    """Extract potential memories from a bus event.

    This function analyzes bus events and extracts content that should
    be persisted as agent memories.

    Returns a list of memory candidates with agent_id, content, metadata.
    """
    candidates: list[dict] = []

    topic = str(event.get("topic") or "")
    actor = str(event.get("actor") or "")
    data = event.get("data") if isinstance(event.get("data"), dict) else {}

    # Skip internal/metric events
    if event.get("kind") in ("metric", "heartbeat"):
        return []

    # Extract from dialogos outputs
    if topic.startswith("dialogos.cell.output"):
        content = str(data.get("content") or "")
        provider = str(data.get("provider") or "unknown")
        if len(content) > 50:  # Only meaningful outputs
            candidates.append({
                "agent_id": actor or provider,
                "content": content[:2000],
                "metadata": {
                    "source": "dialogos",
                    "provider": provider,
                    "topic": topic,
                    "event_id": event.get("id"),
                },
                "tags": ["dialogos", "llm_output"],
            })

    # Extract from task completions
    if topic.endswith(".complete") or topic.endswith(".done"):
        summary = data.get("summary") or data.get("result") or data.get("output")
        if isinstance(summary, str) and len(summary) > 20:
            candidates.append({
                "agent_id": actor,
                "content": summary[:2000],
                "metadata": {
                    "source": "task_completion",
                    "topic": topic,
                    "event_id": event.get("id"),
                    "status": data.get("status"),
                },
                "tags": ["task", "completion"],
            })

    # Extract from research findings
    if "research" in topic or "sota" in topic:
        findings = data.get("findings") or data.get("insights") or data.get("content")
        if isinstance(findings, str) and len(findings) > 30:
            candidates.append({
                "agent_id": actor,
                "content": findings[:2000],
                "metadata": {
                    "source": "research",
                    "topic": topic,
                    "event_id": event.get("id"),
                },
                "tags": ["research", "sota"],
            })

    return candidates


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def default_bus_dir() -> Path:
    return Path(os.environ.get("PLURIBUS_BUS_DIR") or "/pluribus/.pluribus/bus").expanduser().resolve()


def emit_bus_event(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    """Emit an event to the bus."""
    events_path = bus_dir / "events.ndjson"
    ensure_dir(events_path.parent)
    evt = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    append_ndjson(events_path, evt)


def run_bus_watcher(
    *,
    root: Path,
    bus_dir: Path,
    actor: str,
    poll_s: float = 0.5,
) -> None:
    """Watch bus events and auto-extract memories."""
    service = Mem0Service(root)
    events_path = bus_dir / "events.ndjson"
    ensure_dir(events_path.parent)
    events_path.touch(exist_ok=True)

    processed_ids: set[str] = set()

    # Load already processed
    for obj in iter_ndjson(service.index_path):
        if obj.get("kind") == "mem0_access" and obj.get("action") == "bus_extract":
            eid = obj.get("event_id")
            if isinstance(eid, str):
                processed_ids.add(eid)

    print(f"[mem0] watching {events_path}", file=sys.stderr)

    with events_path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_s)
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            if not isinstance(event, dict):
                continue

            event_id = str(event.get("id") or "")
            if event_id in processed_ids:
                continue

            candidates = extract_memories_from_bus_event(event)
            for cand in candidates:
                result = service.store(
                    agent_id=cand["agent_id"],
                    content=cand["content"],
                    metadata=cand.get("metadata"),
                    tags=cand.get("tags"),
                )
                if "error" not in result:
                    # Log that we extracted from this event
                    append_ndjson(service.index_path, {
                        "kind": "mem0_access",
                        "action": "bus_extract",
                        "event_id": event_id,
                        "memory_id": result.get("id"),
                        "ts": time.time(),
                    })
                    emit_bus_event(
                        bus_dir,
                        topic="mem0.memory.stored",
                        kind="metric",
                        level="info",
                        actor=actor,
                        data={
                            "memory_id": result.get("id"),
                            "agent_id": cand["agent_id"],
                            "source_event_id": event_id,
                        },
                    )
            if event_id:
                processed_ids.add(event_id)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mem0_adapter.py", description="Mem0 adapter for Pluribus agent memory.")
    p.add_argument("--root", help="Pluribus root directory")
    p.add_argument("--bus-dir", default=None, help="Bus directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    store = sub.add_parser("store", help="Store a memory")
    store.add_argument("--agent-id", required=True)
    store.add_argument("--content", required=True)
    store.add_argument("--tags", default="", help="Comma-separated tags")
    store.add_argument("--session-id", default=None)

    retrieve = sub.add_parser("retrieve", help="Retrieve memories")
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--agent-id", default=None)
    retrieve.add_argument("--limit", type=int, default=10)

    list_cmd = sub.add_parser("list", help="List memories")
    list_cmd.add_argument("--agent-id", default=None)
    list_cmd.add_argument("--limit", type=int, default=50)

    watch = sub.add_parser("watch", help="Watch bus and auto-extract memories")
    watch.add_argument("--actor", default="mem0-adapter")
    watch.add_argument("--poll", type=float, default=0.5)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    service = Mem0Service(root)

    if args.cmd == "store":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
        result = service.store(
            agent_id=args.agent_id,
            content=args.content,
            tags=tags,
            session_id=args.session_id,
        )
        print(json.dumps(result, indent=2))
        return 0 if "error" not in result else 1

    if args.cmd == "retrieve":
        result = service.retrieve(
            query=args.query,
            agent_id=args.agent_id,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "list":
        result = service.list_memories(
            agent_id=args.agent_id,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "watch":
        bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else default_bus_dir()
        run_bus_watcher(root=root, bus_dir=bus_dir, actor=args.actor, poll_s=args.poll)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
