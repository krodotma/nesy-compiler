#!/usr/bin/env python3
"""Graphiti Bridge for Pluribus Temporal Knowledge Graph.

Provides temporal fact management:
- Fact versioning with valid_from/valid_to timestamps
- Entity linking and disambiguation
- Automatic fact extraction from bus events
- Time-aware queries (facts valid at a specific time)

Schema:
  Entity: id, name, type, properties
  Fact: subject, predicate, object, valid_from, valid_to, confidence, source

Effects: R(file), W(file)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
try:
    import fcntl
except ImportError:
    fcntl = None
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

try:
    from core.bus_consumer import BusConsumer
except (ImportError, ValueError):
    try:
        from core.bus_consumer import BusConsumer
    except ImportError:
        # Fallback for direct execution
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from core.bus_consumer import BusConsumer

# Try to add membrane/graphiti to path for graphiti_core
_GRAPHITI_CORE_AVAILABLE = False
try:
    root_dir = Path(__file__).resolve().parents[2]
    graphiti_dir = root_dir / "membrane" / "graphiti"
    if graphiti_dir.exists():
        sys.path.append(str(graphiti_dir))
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    _GRAPHITI_CORE_AVAILABLE = True
except ImportError:
    pass

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_iso_ts(iso: str) -> float:
    """Parse ISO timestamp to epoch float."""
    try:
        # Handle Z suffix
        iso = iso.replace("Z", "+00:00")
        if "+" not in iso and "-" not in iso[10:]:
            iso = iso + "+00:00"
        dt = datetime.fromisoformat(iso)
        return dt.timestamp()
    except Exception:
        return 0.0


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class BatchNDJSONWriter:
    """Buffered NDJSON writer that keeps file handle open."""
    def __init__(self, path: Path, buffer_size: int = 100):
        self.path = path
        self.buffer_size = buffer_size
        self.buffer: list[dict] = []
        self.last_flush = time.time()
        self._handle = None

    def write(self, obj: dict):
        self.buffer.append(obj)
        if len(self.buffer) >= self.buffer_size or (time.time() - self.last_flush > 5):
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        
        ensure_dir(self.path.parent)
        # Use low-level open for flock compatibility and atomic append
        fd = os.open(str(self.path), os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o666)
        
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            
            for item in self.buffer:
                line = json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                os.write(fd, line.encode("utf-8"))
            
            os.fsync(fd)
            self.buffer = []
            self.last_flush = time.time()
        except Exception:
            # Re-raise or handle as needed
            raise
        finally:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def close(self):
        self.flush()
        if self._handle:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()


# Global writer cache for watch mode
_WRITERS: dict[str, BatchNDJSONWriter] = {}


def append_ndjson(path: Path, obj: dict, buffered: bool = False) -> None:
    if buffered:
        path_str = str(path.resolve())
        if path_str not in _WRITERS:
            _WRITERS[path_str] = BatchNDJSONWriter(path)
        _WRITERS[path_str].write(obj)
    else:
        ensure_dir(path.parent)
        fd = os.open(str(path), os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o666)
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
            os.write(fd, line.encode("utf-8"))
        finally:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


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


def stable_id(text: str) -> str:
    """Generate a stable ID from text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Entity:
    """A knowledge graph entity."""
    id: str
    name: str
    entity_type: str
    properties: dict = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    provenance_id: str | None = None  # Link to source event UUID

    def to_dict(self) -> dict:
        return {
            "kind": "graphiti_entity",
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "properties": self.properties,
            "aliases": self.aliases,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provenance_id": self.provenance_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(
            id=str(d.get("id") or ""),
            name=str(d.get("name") or ""),
            entity_type=str(d.get("entity_type") or d.get("type") or "entity"),
            properties=d.get("properties") if isinstance(d.get("properties"), dict) else {},
            aliases=d.get("aliases") if isinstance(d.get("aliases"), list) else [],
            created_at=str(d.get("created_at") or ""),
            updated_at=str(d.get("updated_at") or ""),
            provenance_id=d.get("provenance_id"),
        )


@dataclass
class Fact:
    """A temporal fact (triple with time bounds)."""
    id: str
    subject_id: str
    predicate: str
    object_id: str
    valid_from: str  # ISO timestamp
    valid_to: str | None  # ISO timestamp or None (still valid)
    confidence: float
    source: str
    properties: dict = field(default_factory=dict)
    created_at: str = ""
    superseded_by: str | None = None  # ID of newer version
    provenance_id: str | None = None  # Link to source event UUID

    def to_dict(self) -> dict:
        return {
            "kind": "graphiti_fact",
            "id": self.id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "confidence": self.confidence,
            "source": self.source,
            "properties": self.properties,
            "created_at": self.created_at,
            "superseded_by": self.superseded_by,
            "provenance_id": self.provenance_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(
            id=str(d.get("id") or ""),
            subject_id=str(d.get("subject_id") or d.get("subject") or ""),
            predicate=str(d.get("predicate") or ""),
            object_id=str(d.get("object_id") or d.get("object") or ""),
            valid_from=str(d.get("valid_from") or ""),
            valid_to=d.get("valid_to") if isinstance(d.get("valid_to"), str) else None,
            confidence=float(d.get("confidence") or 1.0),
            source=str(d.get("source") or ""),
            properties=d.get("properties") if isinstance(d.get("properties"), dict) else {},
            created_at=str(d.get("created_at") or ""),
            superseded_by=d.get("superseded_by") if isinstance(d.get("superseded_by"), str) else None,
            provenance_id=d.get("provenance_id"),
        )

    def is_valid_at(self, ts: float) -> bool:
        """Check if fact was valid at the given timestamp."""
        from_ts = parse_iso_ts(self.valid_from) if self.valid_from else 0.0
        to_ts = parse_iso_ts(self.valid_to) if self.valid_to else float("inf")
        return from_ts <= ts <= to_ts


class GraphitiService:
    """Service layer for temporal knowledge graph operations."""

    def __init__(self, root: Path | str, buffered: bool = False):
        self.root = Path(root) if isinstance(root, str) else root
        self.buffered = buffered
        self._core_instance = None
        self._init_core()

    def _init_core(self):
        if not _GRAPHITI_CORE_AVAILABLE:
            return
        
        neo4j_uri = os.environ.get("NEO4J_URI")
        if neo4j_uri:
            try:
                # Use environment variables for auth
                self._core_instance = Graphiti(
                    neo4j_uri,
                    os.environ.get("NEO4J_USER", "neo4j"),
                    os.environ.get("NEO4J_PASSWORD", "password")
                )
            except Exception:
                pass

    @property
    def entities_path(self) -> Path:
        return self.root / ".pluribus" / "kg" / "graphiti_entities.ndjson"

    @property
    def facts_path(self) -> Path:
        return self.root / ".pluribus" / "kg" / "graphiti_facts.ndjson"

    @property
    def index_path(self) -> Path:
        return self.root / ".pluribus" / "kg" / "graphiti_index.ndjson"

    def _load_entities(self) -> dict[str, Entity]:
        """Load all entities indexed by ID."""
        entities: dict[str, Entity] = {}
        for obj in iter_ndjson(self.entities_path):
            if obj.get("kind") == "graphiti_entity":
                ent = Entity.from_dict(obj)
                if ent.id:
                    entities[ent.id] = ent
        return entities

    def _load_facts(self) -> dict[str, Fact]:
        """Load all facts indexed by ID."""
        facts: dict[str, Fact] = {}
        for obj in iter_ndjson(self.facts_path):
            if obj.get("kind") == "graphiti_fact":
                fact = Fact.from_dict(obj)
                if fact.id:
                    facts[fact.id] = fact
        return facts

    def _find_entity_by_name(self, name: str, entities: dict[str, Entity]) -> Entity | None:
        """Find entity by name or alias."""
        name_lower = name.lower().strip()
        for ent in entities.values():
            if ent.name.lower() == name_lower:
                return ent
            if name_lower in [a.lower() for a in ent.aliases]:
                return ent
        return None

    def add_entity(
        self,
        *,
        name: str,
        entity_type: str = "entity",
        properties: dict | None = None,
        aliases: list[str] | None = None,
        provenance_id: str | None = None,
    ) -> dict:
        """Add or update an entity.

        If an entity with the same name exists, updates it instead.
        """
        if not name:
            return {"error": "name is required"}

        entities = self._load_entities()
        existing = self._find_entity_by_name(name, entities)
        now = now_iso_utc()

        if existing:
            # Update existing entity
            existing.entity_type = entity_type or existing.entity_type
            if properties:
                existing.properties = {**existing.properties, **properties}
            if aliases:
                existing.aliases = list(set(existing.aliases + aliases))
            existing.updated_at = now
            existing.provenance_id = provenance_id or existing.provenance_id
            append_ndjson(self.entities_path, existing.to_dict(), buffered=self.buffered)
            return {"id": existing.id, "name": existing.name, "status": "updated"}

        # Create new entity
        entity_id = str(uuid.uuid4())
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            aliases=aliases or [],
            created_at=now,
            updated_at=now,
            provenance_id=provenance_id,
        )
        append_ndjson(self.entities_path, entity.to_dict(), buffered=self.buffered)
        return {"id": entity_id, "name": name, "status": "created"}

    def add_fact(
        self,
        *,
        subject: str,
        predicate: str,
        object_value: str,
        valid_from: str | None = None,
        valid_to: str | None = None,
        confidence: float = 1.0,
        source: str = "",
        properties: dict | None = None,
        provenance_id: str | None = None,
    ) -> dict:
        """Add a temporal fact.

        Subject and object can be entity IDs or names.
        If names, entities are created/resolved automatically.
        """
        if not subject or not predicate or not object_value:
            return {"error": "subject, predicate, and object are required"}

        entities = self._load_entities()
        now = now_iso_utc()

        # Resolve subject/object to IDs if needed
        subject_id = None
        if subject:
            ent = self._find_entity_by_name(subject, entities)
            subject_id = ent.id if ent else entities.get(subject, Entity(id=subject, name="", entity_type="")).id

        # Resolve or create object entity
        object_entity = self._find_entity_by_name(object_value, entities)
        if not object_entity:
            object_id = entities.get(object_value, Entity(id=object_value, name="", entity_type="")).id
        else:
            object_id = object_entity.id

        # Check for existing similar fact to supersede
        facts = self._load_facts()
        for existing_fact in facts.values():
            if (
                existing_fact.subject_id == subject_id
                and existing_fact.predicate == predicate
                and existing_fact.valid_to is None
                and existing_fact.superseded_by is None
            ):
                # Supersede the old fact
                existing_fact.valid_to = valid_from or now
                append_ndjson(self.facts_path, existing_fact.to_dict(), buffered=self.buffered)


    def query_facts(
        self,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        object_value: str | None = None,
        at_time: str | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> dict:
        """Query facts with optional temporal filter.

        Args:
            subject: Filter by subject (ID or name)
            predicate: Filter by predicate
            object_value: Filter by object (ID or name)
            at_time: Return only facts valid at this ISO timestamp
            include_superseded: Include superseded facts
            limit: Maximum results
        """
        entities = self._load_entities()
        facts = self._load_facts()

        # Resolve subject/object to IDs if needed
        subject_id = None
        if subject:
            ent = self._find_entity_by_name(subject, entities)
            subject_id = ent.id if ent else entities.get(subject, Entity(id=subject, name="", entity_type="")).id

        object_id = None
        if object_value:
            ent = self._find_entity_by_name(object_value, entities)
            object_id = ent.id if ent else entities.get(object_value, Entity(id=object_value, name="", entity_type="")).id

        at_ts = parse_iso_ts(at_time) if at_time else None

        results: list[dict] = []
        for fact in facts.values():
            # Filter by subject
            if subject_id and fact.subject_id != subject_id:
                continue
            # Filter by predicate
            if predicate and fact.predicate != predicate:
                continue
            # Filter by object
            if object_id and fact.object_id != object_id:
                continue
            # Filter superseded
            if not include_superseded and fact.superseded_by:
                continue
            # Filter by time validity
            if at_ts is not None and not fact.is_valid_at(at_ts):
                continue

            # Resolve entity names for display
            subj_ent = entities.get(fact.subject_id)
            obj_ent = entities.get(fact.object_id)

            results.append({
                "id": fact.id,
                "subject": {
                    "id": fact.subject_id,
                    "name": subj_ent.name if subj_ent else fact.subject_id,
                },
                "predicate": fact.predicate,
                "object": {
                    "id": fact.object_id,
                    "name": obj_ent.name if obj_ent else fact.object_id,
                },
                "valid_from": fact.valid_from,
                "valid_to": fact.valid_to,
                "confidence": fact.confidence,
                "source": fact.source,
            })
            if len(results) >= limit:
                break

        return {"facts": results, "count": len(results)}

    def get_entity(self, *, entity_id: str | None = None, name: str | None = None) -> dict:
        """Get an entity by ID or name."""
        entities = self._load_entities()

        if entity_id and entity_id in entities:
            ent = entities[entity_id]
            return {"entity": ent.to_dict(), "found": True}

        if name:
            ent = self._find_entity_by_name(name, entities)
            if ent:
                return {"entity": ent.to_dict(), "found": True}

        return {"entity": None, "found": False}

    def list_entities(
        self,
        *,
        entity_type: str | None = None,
        limit: int = 50,
    ) -> dict:
        """List entities with optional filtering."""
        entities = self._load_entities()

        results: list[dict] = []
        for ent in entities.values():
            if entity_type and ent.entity_type != entity_type:
                continue
            results.append({
                "id": ent.id,
                "name": ent.name,
                "entity_type": ent.entity_type,
                "aliases": ent.aliases[:3],  # Limit aliases in list
            })
            if len(results) >= limit:
                break

        return {"entities": results, "count": len(results)}

    def invalidate_fact(self, *, fact_id: str, reason: str = "") -> dict:
        """Mark a fact as no longer valid (set valid_to to now)."""
        if not fact_id:
            return {"error": "fact_id is required"}

        facts = self._load_facts()
        if fact_id not in facts:
            return {"error": f"fact not found: {fact_id}"}

        fact = facts[fact_id]
        if fact.valid_to is not None:
            return {"error": "fact already invalidated"}

        now = now_iso_utc()
        fact.valid_to = now
        fact.properties["invalidation_reason"] = reason
        append_ndjson(self.facts_path, fact.to_dict())

        return {"id": fact_id, "valid_to": now, "status": "invalidated"}

    def tools_list(self) -> dict:
        """Return MCP tools list."""
        return {
            "tools": [
                {
                    "name": "graphiti_add_entity",
                    "description": "Add or update an entity in the knowledge graph",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "entity_type": {"type": "string", "default": "entity"},
                            "properties": {"type": "object"},
                            "aliases": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["name"],
                    },
                },
                {
                    "name": "graphiti_add_fact",
                    "description": "Add a temporal fact to the knowledge graph",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string", "description": "Entity name or ID"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string", "description": "Entity name or ID"},
                            "valid_from": {"type": "string", "description": "ISO timestamp"},
                            "valid_to": {"type": "string", "description": "ISO timestamp"},
                            "confidence": {"type": "number", "default": 1.0},
                            "source": {"type": "string"},
                            "properties": {"type": "object"},
                        },
                        "required": ["subject", "predicate", "object"],
                    },
                },
                {
                    "name": "graphiti_query",
                    "description": "Query facts with optional temporal filter",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "at_time": {"type": "string", "description": "ISO timestamp for time-travel query"},
                            "include_superseded": {"type": "boolean", "default": False},
                            "limit": {"type": "integer", "default": 50},
                        },
                    },
                },
                {
                    "name": "graphiti_get_entity",
                    "description": "Get an entity by ID or name",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "entity_id": {"type": "string"},
                            "name": {"type": "string"},
                        },
                    },
                },
                {
                    "name": "graphiti_list_entities",
                    "description": "List entities with optional type filter",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "entity_type": {"type": "string"},
                            "limit": {"type": "integer", "default": 50},
                        },
                    },
                },
                {
                    "name": "graphiti_invalidate",
                    "description": "Mark a fact as no longer valid",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "fact_id": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["fact_id"],
                    },
                },
            ]
        }

    def tools_call(self, name: str, args: dict) -> dict:
        """Dispatch an MCP tool call."""
        if name == "graphiti_add_entity":
            return self.add_entity(
                name=str(args.get("name") or ""),
                entity_type=str(args.get("entity_type") or "entity"),
                properties=args.get("properties") if isinstance(args.get("properties"), dict) else None,
                aliases=args.get("aliases") if isinstance(args.get("aliases"), list) else None,
            )
        if name == "graphiti_add_fact":
            return self.add_fact(
                subject=str(args.get("subject") or ""),
                predicate=str(args.get("predicate") or ""),
                object_value=str(args.get("object") or ""),
                valid_from=args.get("valid_from") if isinstance(args.get("valid_from"), str) else None,
                valid_to=args.get("valid_to") if isinstance(args.get("valid_to"), str) else None,
                confidence=float(args.get("confidence") or 1.0),
                source=str(args.get("source") or ""),
                properties=args.get("properties") if isinstance(args.get("properties"), dict) else None,
            )
        if name == "graphiti_query":
            return self.query_facts(
                subject=args.get("subject") if isinstance(args.get("subject"), str) else None,
                predicate=args.get("predicate") if isinstance(args.get("predicate"), str) else None,
                object_value=args.get("object") if isinstance(args.get("object"), str) else None,
                at_time=args.get("at_time") if isinstance(args.get("at_time"), str) else None,
                include_superseded=bool(args.get("include_superseded")),
                limit=int(args.get("limit") or 50),
            )
        if name == "graphiti_get_entity":
            return self.get_entity(
                entity_id=args.get("entity_id") if isinstance(args.get("entity_id"), str) else None,
                name=args.get("name") if isinstance(args.get("name"), str) else None,
            )
        if name == "graphiti_list_entities":
            return self.list_entities(
                entity_type=args.get("entity_type") if isinstance(args.get("entity_type"), str) else None,
                limit=int(args.get("limit") or 50),
            )
        if name == "graphiti_invalidate":
            return self.invalidate_fact(
                fact_id=str(args.get("fact_id") or ""),
                reason=str(args.get("reason") or ""),
            )
        return {"error": f"unknown tool: {name}"}


def _truncate_text(text: str, max_len: int = 100) -> str:
    """Truncate text to max_len, adding ellipsis if truncated."""
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def extract_facts_from_dialogos_record(record: dict) -> list[dict]:
    """Extract rich facts from dialogos trace records.

    Analyzes dialogos session/turn records and extracts comprehensive facts:
    - Session lifecycle (start/end)
    - Prompt/Response pairs
    - Tool usage
    - Errors
    - Turn relationships

    Args:
        record: A dialogos trace record dict

    Returns:
        List of fact dicts with subject, predicate, object, valid_from, confidence, source
    """
    facts: list[dict] = []

    # Extract common fields
    record_type = str(record.get("type") or record.get("kind") or "")
    actor = str(record.get("actor") or record.get("agent") or record.get("user") or "unknown")
    session_id = str(record.get("session_id") or record.get("session") or "")
    turn_id = str(record.get("turn_id") or record.get("turn") or record.get("id") or "")
    req_id = str(record.get("req_id") or record.get("request_id") or "")
    timestamp = str(record.get("timestamp") or record.get("ts") or record.get("iso") or now_iso_utc())
    data = record.get("data") if isinstance(record.get("data"), dict) else {}

    # Normalize timestamp to ISO format if it's a float
    if isinstance(record.get("ts"), (int, float)):
        try:
            timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record["ts"]))
        except Exception:
            timestamp = now_iso_utc()

    source = f"dialogos:{record_type}"

    # 1. Session Facts
    if record_type in ("session_start", "session.start", "start"):
        if actor and session_id:
            facts.append({
                "subject": actor,
                "predicate": "started_session",
                "object": f"session:{session_id}",
                "valid_from": timestamp,
                "confidence": 1.0,
                "source": source,
                "properties": {"session_id": session_id},
            })
            # Session belongs to actor
            facts.append({
                "subject": f"session:{session_id}",
                "predicate": "belongs_to_actor",
                "object": actor,
                "valid_from": timestamp,
                "confidence": 1.0,
                "source": source,
            })

    elif record_type in ("session_end", "session.end", "end"):
        if actor and session_id:
            facts.append({
                "subject": actor,
                "predicate": "ended_session",
                "object": f"session:{session_id}",
                "valid_from": timestamp,
                "confidence": 1.0,
                "source": source,
                "properties": {"session_id": session_id},
            })

    # 2. Prompt/Response Facts
    elif record_type in ("prompt", "user_message", "request", "turn.user"):
        prompt = record.get("prompt") or record.get("message") or record.get("content") or data.get("prompt") or ""
        prompt_summary = _truncate_text(prompt, 100)
        if actor and prompt_summary:
            facts.append({
                "subject": actor,
                "predicate": "asked",
                "object": prompt_summary,
                "valid_from": timestamp,
                "confidence": 1.0,
                "source": source,
                "properties": {"turn_id": turn_id, "req_id": req_id} if turn_id or req_id else {},
            })
        # Session contains turn
        if session_id and turn_id:
            facts.append({
                "subject": f"session:{session_id}",
                "predicate": "contains_turn",
                "object": f"turn:{turn_id}",
                "valid_from": timestamp,
                "confidence": 1.0,
                "source": source,
            })

    elif record_type in ("response", "assistant_message", "completion", "turn.assistant"):
        assistant = str(record.get("assistant") or record.get("model") or "assistant")
        if req_id:
            facts.append({
                "subject": assistant,
                "predicate": "responded_to",
                "object": f"request:{req_id}",
                "valid_from": timestamp,
                "confidence": 1.0,
                "source": source,
                "properties": {"turn_id": turn_id},
            })
        # Session contains turn
        if session_id and turn_id:
            facts.append({
                "subject": f"session:{session_id}",
                "predicate": "contains_turn",
                "object": f"turn:{turn_id}",
                "valid_from": timestamp,
                "confidence": 1.0,
                "source": source,
            })

    # 3. Tool Usage Facts
    elif record_type in ("tool_call", "tool.call", "function_call", "tool_use"):
        tool_name = str(
            record.get("tool") or record.get("tool_name") or record.get("function") or data.get("tool") or ""
        )
        output = record.get("output") or record.get("result") or data.get("output") or ""
        output_summary = _truncate_text(str(output), 100)

        if actor and tool_name:
            facts.append({
                "subject": actor,
                "predicate": "used_tool",
                "object": f"tool:{tool_name}",
                "valid_from": timestamp,
                "confidence": 1.0,
                "source": source,
                "properties": {"turn_id": turn_id, "session_id": session_id},
            })
        if tool_name and output_summary:
            facts.append({
                "subject": f"tool:{tool_name}",
                "predicate": "produced_output",
                "object": output_summary,
                "valid_from": timestamp,
                "confidence": 0.9,
                "source": source,
            })

    # 4. Error Facts
    elif record_type in ("error", "exception", "failure"):
        error_type = str(
            record.get("error_type")
            or record.get("error_code")
            or record.get("exception_type")
            or data.get("error_type")
            or "unknown_error"
        )
        error_message = str(record.get("error") or record.get("message") or data.get("error") or "")

        if session_id:
            facts.append({
                "subject": f"session:{session_id}",
                "predicate": "encountered_error",
                "object": error_type,
                "valid_from": timestamp,
                "confidence": 1.0,
                "source": source,
                "properties": {"message": _truncate_text(error_message, 200)},
            })
        facts.append({
            "subject": f"error:{error_type}",
            "predicate": "occurred_at",
            "object": timestamp,
            "valid_from": timestamp,
            "confidence": 1.0,
            "source": source,
            "properties": {"session_id": session_id, "turn_id": turn_id},
        })

    # 5. Turn Relationship Facts (follows)
    prev_turn_id = str(record.get("prev_turn_id") or record.get("previous_turn") or record.get("parent_turn") or "")
    if turn_id and prev_turn_id:
        facts.append({
            "subject": f"turn:{turn_id}",
            "predicate": "follows",
            "object": f"turn:{prev_turn_id}",
            "valid_from": timestamp,
            "confidence": 1.0,
            "source": source,
        })

    # Handle generic dialogos records with prompt/response in data
    if not facts and data:
        # Try to extract from nested data
        if data.get("prompt") or data.get("message"):
            prompt_summary = _truncate_text(str(data.get("prompt") or data.get("message")), 100)
            if actor and prompt_summary:
                facts.append({
                    "subject": actor,
                    "predicate": "asked",
                    "object": prompt_summary,
                    "valid_from": timestamp,
                    "confidence": 0.8,
                    "source": source,
                })
        if data.get("response") or data.get("completion"):
            resp_summary = _truncate_text(str(data.get("response") or data.get("completion")), 100)
            if resp_summary:
                facts.append({
                    "subject": "assistant",
                    "predicate": "responded_with",
                    "object": resp_summary,
                    "valid_from": timestamp,
                    "confidence": 0.8,
                    "source": source,
                })
        if data.get("tool") or data.get("tool_name"):
            tool_name = str(data.get("tool") or data.get("tool_name"))
            facts.append({
                "subject": actor or "agent",
                "predicate": "used_tool",
                "object": f"tool:{tool_name}",
                "valid_from": timestamp,
                "confidence": 0.8,
                "source": source,
            })

    return facts


def extract_facts_from_bus_event(event: dict) -> list[dict]:
    """Extract potential facts from a bus event.

    Analyzes bus events and extracts subject-predicate-object triples.
    Handles dialogos.* topics with specialized extraction.
    """
    candidates: list[dict] = []

    topic = str(event.get("topic") or "")
    actor = str(event.get("actor") or "")
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    iso = str(event.get("iso") or now_iso_utc())

    # Skip internal events
    if event.get("kind") in ("metric", "heartbeat"):
        return []

    # Handle dialogos.* topics specially
    if topic.startswith("dialogos."):
        # Build a dialogos record from the bus event
        dialogos_record = {
            "type": topic.replace("dialogos.", ""),
            "actor": actor,
            "timestamp": iso,
            "data": data,
            **data,  # Merge data fields to top level for extraction
        }
        return extract_facts_from_dialogos_record(dialogos_record)

    # Extract from task events
    if ".task." in topic or topic.endswith(".complete"):
        task_id = data.get("task_id") or data.get("req_id")
        status = data.get("status")
        if task_id and status:
            candidates.append({
                "subject": f"task:{task_id}",
                "predicate": "has_status",
                "object": str(status),
                "valid_from": iso,
                "confidence": 1.0,
                "source": f"bus:{topic}",
            })

    # Extract from agent communication
    if ".request" in topic or ".response" in topic:
        req_id = data.get("req_id") or data.get("request_id")
        if req_id and actor:
            predicate = "sent_request" if ".request" in topic else "sent_response"
            candidates.append({
                "subject": actor,
                "predicate": predicate,
                "object": f"request:{req_id}",
                "valid_from": iso,
                "confidence": 0.9,
                "source": f"bus:{topic}",
            })

    # Extract from file operations
    if "file" in topic.lower():
        file_path = data.get("path") or data.get("file_path")
        action = data.get("action") or data.get("operation")
        if file_path and action and actor:
            candidates.append({
                "subject": actor,
                "predicate": f"performed_{action}",
                "object": str(file_path),
                "valid_from": iso,
                "confidence": 1.0,
                "source": f"bus:{topic}",
            })

    # Extract from sota/research events
    if "sota" in topic or "research" in topic:
        tool_name = data.get("tool") or data.get("name")
        status = data.get("status") or data.get("result")
        if tool_name and status:
            candidates.append({
                "subject": str(tool_name),
                "predicate": "evaluation_result",
                "object": str(status),
                "valid_from": iso,
                "confidence": 0.8,
                "source": f"bus:{topic}",
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
    """Watch bus events and auto-extract facts."""
    service = GraphitiService(root, buffered=True)
    events_path = bus_dir / "events.ndjson"
    ensure_dir(events_path.parent)
    events_path.touch(exist_ok=True)

    processed_ids: set[str] = set()

    # Load already processed
    for obj in iter_ndjson(service.index_path):
        if obj.get("kind") == "graphiti_extraction":
            eid = obj.get("event_id")
            if isinstance(eid, str):
                processed_ids.add(eid)

    print(f"[graphiti] watching {events_path}", file=sys.stderr)

    consumer = BusConsumer(events_path)

    def process_event(event):
        # Flush any pending writes from buffered services
        for writer in _WRITERS.values():
            writer.flush()

        event_id = str(event.get("id") or "")
        if event_id in processed_ids:
            return

        candidates = extract_facts_from_bus_event(event)
        for cand in candidates:
            result = service.add_fact(
                subject=cand["subject"],
                predicate=cand["predicate"],
                object_value=cand["object"],
                valid_from=cand.get("valid_from"),
                confidence=cand.get("confidence", 1.0),
                source=cand.get("source", ""),
                provenance_id=event_id,
            )
            if "error" not in result:
                append_ndjson(service.index_path, {
                    "kind": "graphiti_extraction",
                    "event_id": event_id,
                    "fact_id": result.get("id"),
                    "ts": time.time(),
                }, buffered=True)
                emit_bus_event(
                    bus_dir,
                    topic="graphiti.fact.extracted",
                    kind="metric",
                    level="info",
                    actor=actor,
                    data={
                        "fact_id": result.get("id"),
                        "subject": cand["subject"],
                        "predicate": cand["predicate"],
                        "object": cand["object"],
                        "source_event_id": event_id,
                    },
                )
        if event_id:
            processed_ids.add(event_id)

    consumer.tail(callback=process_event, poll_s=poll_s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="graphiti_bridge.py", description="Graphiti bridge for Pluribus temporal KG.")
    p.add_argument("--root", help="Pluribus root directory")
    p.add_argument("--bus-dir", default=None, help="Bus directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    add_entity = sub.add_parser("add-entity", help="Add an entity")
    add_entity.add_argument("--name", required=True)
    add_entity.add_argument("--type", default="entity")
    add_entity.add_argument("--aliases", default="", help="Comma-separated aliases")

    add_fact = sub.add_parser("add-fact", help="Add a fact")
    add_fact.add_argument("--subject", required=True)
    add_fact.add_argument("--predicate", required=True)
    add_fact.add_argument("--object", required=True)
    add_fact.add_argument("--source", default="cli")
    add_fact.add_argument("--confidence", type=float, default=1.0)

    query = sub.add_parser("query", help="Query facts")
    query.add_argument("--subject", default=None)
    query.add_argument("--predicate", default=None)
    query.add_argument("--object", default=None)
    query.add_argument("--at-time", default=None, help="ISO timestamp for time-travel")
    query.add_argument("--limit", type=int, default=50)

    list_cmd = sub.add_parser("list-entities", help="List entities")
    list_cmd.add_argument("--type", default=None)
    list_cmd.add_argument("--limit", type=int, default=50)

    watch = sub.add_parser("watch", help="Watch bus and auto-extract facts")
    watch.add_argument("--actor", default="graphiti-bridge")
    watch.add_argument("--poll", type=float, default=0.5)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    service = GraphitiService(root)

    if args.cmd == "add-entity":
        aliases = [a.strip() for a in args.aliases.split(",") if a.strip()] if args.aliases else None
        result = service.add_entity(name=args.name, entity_type=args.type, aliases=aliases)
        print(json.dumps(result, indent=2))
        return 0 if "error" not in result else 1

    if args.cmd == "add-fact":
        result = service.add_fact(
            subject=args.subject,
            predicate=args.predicate,
            object_value=args.object,
            source=args.source,
            confidence=args.confidence,
        )
        print(json.dumps(result, indent=2))
        return 0 if "error" not in result else 1

    if args.cmd == "query":
        result = service.query_facts(
            subject=args.subject,
            predicate=args.predicate,
            object_value=args.object,
            at_time=args.at_time,
            limit=args.limit,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "list-entities":
        result = service.list_entities(entity_type=args.type, limit=args.limit)
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "watch":
        bus_dir = Path(args.bus_dir).expanduser().resolve() if args.bus_dir else default_bus_dir()
        run_bus_watcher(root=root, bus_dir=bus_dir, actor=args.actor, poll_s=args.poll)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
