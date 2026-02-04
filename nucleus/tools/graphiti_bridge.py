#!/usr/bin/env python3
"""
Graphiti Bridge (Local Temporal KG)
=================================

File-backed temporal knowledge graph used by dialogos indexer, MCP, and tests.
Stores entities and facts as append-only NDJSON in .pluribus/kg.

FalkorDB Integration (Phase 1 Step 1):
- Parameterized Cypher queries (secure)
- query_cypher() for raw Cypher execution
- FalkorDB-first with NDJSON fallback
- Simple query result caching
- Connection health monitoring
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

sys.dont_write_bytecode = True


class QueryCache:
    """Simple LRU cache for Cypher query results."""

    def __init__(self, maxsize: int = 100, ttl_seconds: float = 60.0):
        self._cache: OrderedDict[str, Tuple[float, Any]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._lock = Lock()

    def _make_key(self, query: str, params: Optional[Dict] = None) -> str:
        key_str = query + json.dumps(params or {}, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, params: Optional[Dict] = None) -> Optional[Any]:
        key = self._make_key(query, params)
        with self._lock:
            if key not in self._cache:
                return None
            ts, value = self._cache[key]
            if time.time() - ts > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def set(self, query: str, params: Optional[Dict], value: Any) -> None:
        key = self._make_key(query, params)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = (time.time(), value)

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()


class FalkorDBError(Exception):
    """FalkorDB-specific error."""
    pass


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_iso_ts(value: str) -> float:
    if not value:
        return 0.0
    value = value.strip()
    if not value:
        return 0.0
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return 0.0


@dataclass
class Entity:
    id: str
    name: str
    entity_type: str = "entity"
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_iso_utc)
    updated_at: str = field(default_factory=now_iso_utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "graphiti_entity",
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "properties": self.properties,
            "aliases": self.aliases,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            id=str(data.get("id") or ""),
            name=str(data.get("name") or ""),
            entity_type=str(data.get("entity_type") or "entity"),
            properties=dict(data.get("properties") or {}),
            aliases=list(data.get("aliases") or []),
            created_at=str(data.get("created_at") or now_iso_utc()),
            updated_at=str(data.get("updated_at") or now_iso_utc()),
        )


@dataclass
class Fact:
    id: str
    subject_id: str
    predicate: str
    object_id: str
    valid_from: str
    valid_to: Optional[str] = None
    confidence: float = 1.0
    source: str = "unknown"
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso_utc)
    provenance_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
            "provenance_id": self.provenance_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fact":
        return cls(
            id=str(data.get("id") or ""),
            subject_id=str(data.get("subject_id") or ""),
            predicate=str(data.get("predicate") or ""),
            object_id=str(data.get("object_id") or ""),
            valid_from=str(data.get("valid_from") or now_iso_utc()),
            valid_to=data.get("valid_to"),
            confidence=float(data.get("confidence", 1.0)),
            source=str(data.get("source") or "unknown"),
            properties=dict(data.get("properties") or {}),
            created_at=str(data.get("created_at") or now_iso_utc()),
            provenance_id=data.get("provenance_id"),
        )

    def is_valid_at(self, ts: float) -> bool:
        if ts <= 0:
            return False
        start = parse_iso_ts(self.valid_from)
        if start and ts < start:
            return False
        if self.valid_to:
            end = parse_iso_ts(self.valid_to)
            if end and ts > end:
                return False
        return True


class _ClientStub:
    def close(self) -> None:
        return None


class GraphitiService:
    """Temporal KG with pluggable backend (NDJSON, FalkorDB).

    FalkorDB Integration Features:
    - Parameterized Cypher queries (prevents injection)
    - query_cypher() for raw Cypher execution
    - Query result caching with TTL
    - Connection health monitoring
    - DR-only fallback to NDJSON
    """

    def __init__(self, root: Optional[Path | str] = None, backend: Optional[str] = None):
        base = Path(root or os.environ.get("PLURIBUS_ROOT", "/pluribus")).expanduser().resolve()
        self.root = base
        self.kg_dir = self.root / ".pluribus" / "kg"
        self.entities_path = self.kg_dir / "graphiti_entities.ndjson"
        self.facts_path = self.kg_dir / "graphiti_facts.ndjson"
        self.kg_dir.mkdir(parents=True, exist_ok=True)
        self.entities_path.touch(exist_ok=True)
        self.facts_path.touch(exist_ok=True)
        self.client = _ClientStub()

        # Backend selection: check env or use provided backend
        self._backend_type = backend or os.environ.get("PLURIBUS_GRAPH_BACKEND", "ndjson")
        self._falkordb = None
        self._falkordb_graph = None
        self._query_cache = QueryCache(maxsize=100, ttl_seconds=60.0)
        self._connection_healthy = False
        self._last_health_check = 0.0

        if self._backend_type == "falkordb":
            self._init_falkordb()

    @staticmethod
    def _dr_enabled() -> bool:
        return os.environ.get("PLURIBUS_DR_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

    def _init_falkordb(self) -> None:
        """Initialize FalkorDB connection with health check."""
        try:
            from falkordb import FalkorDB
            host = os.environ.get("FALKORDB_HOST", "localhost")
            port = int(os.environ.get("FALKORDB_PORT", "6379"))
            database = os.environ.get("FALKORDB_DATABASE", "pluribus_kg")

            self._falkordb = FalkorDB(host=host, port=port)
            self._falkordb_graph = self._falkordb.select_graph(database)
            # Test connection
            self._falkordb_graph.query("RETURN 1")
            self._connection_healthy = True
            self._last_health_check = time.time()
        except Exception as e:
            # Fallback to NDJSON if FalkorDB unavailable
            fallback = os.environ.get("PLURIBUS_GRAPH_FALLBACK", "ndjson").strip().lower()
            if fallback in {"ndjson", "dr"}:
                if self._dr_enabled():
                    self._backend_type = "ndjson"
                    self._falkordb = None
                    self._falkordb_graph = None
                    self._connection_healthy = False
                    return
                raise FalkorDBError(
                    "FalkorDB connection failed and NDJSON fallback is disabled unless PLURIBUS_DR_MODE=1."
                )
            raise FalkorDBError(f"FalkorDB connection failed: {e}")

    @property
    def using_falkordb(self) -> bool:
        """Check if FalkorDB backend is active."""
        return self._backend_type == "falkordb" and self._falkordb_graph is not None

    def check_health(self) -> Dict[str, Any]:
        """Check FalkorDB connection health."""
        if not self.using_falkordb:
            return {"healthy": False, "backend": "ndjson", "reason": "FalkorDB not configured"}

        try:
            result = self._falkordb_graph.query("RETURN 1 AS health")
            self._connection_healthy = True
            self._last_health_check = time.time()

            # Get graph stats
            node_count = self._falkordb_graph.query("MATCH (n) RETURN count(n) AS cnt")
            edge_count = self._falkordb_graph.query("MATCH ()-[r]->() RETURN count(r) AS cnt")

            return {
                "healthy": True,
                "backend": "falkordb",
                "last_check": self._last_health_check,
                "node_count": node_count.result_set[0][0] if node_count.result_set else 0,
                "edge_count": edge_count.result_set[0][0] if edge_count.result_set else 0,
                "cache_size": len(self._query_cache._cache),
            }
        except Exception as e:
            self._connection_healthy = False
            return {"healthy": False, "backend": "falkordb", "error": str(e)}

    def query_cypher(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        timeout_ms: int = 5000,
    ) -> List[Dict[str, Any]]:
        """Execute parameterized Cypher query against FalkorDB.

        Args:
            cypher: Cypher query string with $param placeholders
            params: Dictionary of parameter values
            use_cache: Whether to use query cache (default True)
            timeout_ms: Query timeout in milliseconds

        Returns:
            List of result dictionaries with column names as keys

        Raises:
            FalkorDBError: If FalkorDB backend not available or query fails
        """
        if not self.using_falkordb:
            raise FalkorDBError("FalkorDB backend not available. Set PLURIBUS_GRAPH_BACKEND=falkordb")

        # Check cache
        if use_cache:
            cached = self._query_cache.get(cypher, params)
            if cached is not None:
                return cached

        try:
            # Execute with parameters
            result = self._falkordb_graph.query(cypher, params or {}, timeout=timeout_ms)

            # Convert to list of dicts
            rows: List[Dict[str, Any]] = []
            if result.result_set:
                # FalkorDB returns headers as list of column info
                # Each header item might be a string or a list like [0, 'column_name']
                raw_headers = result.header if hasattr(result, 'header') else []
                headers: List[str] = []
                for h in raw_headers:
                    if isinstance(h, str):
                        headers.append(h)
                    elif isinstance(h, (list, tuple)) and len(h) >= 2:
                        headers.append(str(h[1]))  # Second element is column name
                    else:
                        headers.append(str(h))

                for row in result.result_set:
                    if headers and len(headers) == len(row):
                        row_dict = {}
                        for i, val in enumerate(row):
                            # Handle nested structures
                            if hasattr(val, 'properties'):  # Node or Edge
                                row_dict[headers[i]] = dict(val.properties) if val.properties else {}
                            else:
                                row_dict[headers[i]] = val
                        rows.append(row_dict)
                    elif len(row) == 1:
                        rows.append({"value": row[0]})
                    else:
                        rows.append({"values": list(row)})

            # Cache result
            if use_cache:
                self._query_cache.set(cypher, params, rows)

            return rows

        except Exception as e:
            self._connection_healthy = False
            raise FalkorDBError(f"Cypher query failed: {e}")

    def query_cypher_single(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Execute Cypher query and return single result or None."""
        results = self.query_cypher(cypher, params, use_cache=True)
        return results[0] if results else None

    def invalidate_cache(self) -> None:
        """Invalidate query cache (call after writes)."""
        self._query_cache.invalidate()

    def _read_ndjson(self, path: Path) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not path.exists():
            return items
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
        return items

    def _append_ndjson(self, path: Path, payload: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _latest_entities(self) -> Dict[str, Entity]:
        items = self._read_ndjson(self.entities_path)
        latest: Dict[str, Entity] = {}
        for item in items:
            if item.get("kind") != "graphiti_entity":
                continue
            entity = Entity.from_dict(item)
            if entity.id:
                latest[entity.id] = entity
        return latest

    def _entity_name_map(self, entities: Dict[str, Entity]) -> Dict[str, str]:
        name_map: Dict[str, str] = {}
        for ent in entities.values():
            for name in [ent.name, *ent.aliases]:
                if not name:
                    continue
                name_map[name.lower()] = ent.id
        return name_map

    def add_entity(
        self,
        *,
        name: str,
        entity_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not name:
            return {"error": "name required"}

        entities = self._latest_entities()
        name_map = self._entity_name_map(entities)
        key = name.lower()
        existing_id = name_map.get(key)
        status = "created"
        if existing_id and existing_id in entities:
            existing = entities[existing_id]
            merged_aliases = sorted(set(existing.aliases + (aliases or [])))
            merged_props = dict(existing.properties)
            if properties:
                merged_props.update(properties)
            entity = Entity(
                id=existing.id,
                name=existing.name,
                entity_type=entity_type or existing.entity_type,
                properties=merged_props,
                aliases=merged_aliases,
                created_at=existing.created_at,
                updated_at=now_iso_utc(),
            )
            status = "updated"
        else:
            entity = Entity(
                id=uuid.uuid4().hex,
                name=name,
                entity_type=entity_type or "entity",
                properties=properties or {},
                aliases=aliases or [],
                created_at=now_iso_utc(),
                updated_at=now_iso_utc(),
            )

        # Write to backend
        if self.using_falkordb:
            self._add_entity_falkordb(entity)
        self._append_ndjson(self.entities_path, entity.to_dict())  # Always write to NDJSON as backup
        return {"status": status, "id": entity.id, "name": entity.name}

    def _add_entity_falkordb(self, entity: Entity) -> None:
        """Add entity to FalkorDB using parameterized query."""
        if not self._falkordb_graph:
            return
        try:
            # Use parameterized query to prevent injection
            # Note: FalkorDB doesn't support parameterized labels, so we validate entity_type
            entity_type = (entity.entity_type or "entity").replace(" ", "_")
            if not entity_type.isidentifier():
                entity_type = "entity"

            query = f"""
            MERGE (n:{entity_type} {{id: $id}})
            SET n.name = $name,
                n.entity_type = $entity_type,
                n.created_at = $created_at,
                n.updated_at = $updated_at
            RETURN n.id
            """
            params = {
                "id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type or "entity",
                "created_at": entity.created_at,
                "updated_at": entity.updated_at,
            }
            self._falkordb_graph.query(query, params)
            self.invalidate_cache()  # Invalidate cache after write
        except Exception as e:
            # Log but don't fail - NDJSON is already written as backup
            pass

    def get_entity(self, *, entity_id: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
        entities = self._latest_entities()
        if entity_id:
            ent = entities.get(entity_id)
            if not ent:
                return {"found": False}
            return {"found": True, "entity": ent.to_dict()}
        if name:
            name_map = self._entity_name_map(entities)
            ent_id = name_map.get(name.lower())
            if ent_id and ent_id in entities:
                return {"found": True, "entity": entities[ent_id].to_dict()}
            return {"found": False}
        return {"found": False, "error": "entity_id or name required"}

    def list_entities(self, *, entity_type: Optional[str] = None) -> Dict[str, Any]:
        entities = list(self._latest_entities().values())
        if entity_type:
            entities = [e for e in entities if e.entity_type == entity_type]
        return {
            "count": len(entities),
            "entities": [e.to_dict() for e in entities],
        }

    def _ensure_entity(self, name: str) -> Entity:
        entities = self._latest_entities()
        name_map = self._entity_name_map(entities)
        ent_id = name_map.get(name.lower())
        if ent_id and ent_id in entities:
            return entities[ent_id]
        result = self.add_entity(name=name)
        return Entity(id=result["id"], name=name)

    def add_fact(
        self,
        *,
        subject: str,
        predicate: str,
        object_value: str,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        confidence: float = 1.0,
        source: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        provenance_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not subject or not predicate or not object_value:
            return {"error": "subject, predicate, and object_value required"}

        subject_ent = self._ensure_entity(subject)
        object_ent = self._ensure_entity(object_value)
        fact = Fact(
            id=uuid.uuid4().hex,
            subject_id=subject_ent.id,
            predicate=predicate,
            object_id=object_ent.id,
            valid_from=valid_from or now_iso_utc(),
            valid_to=valid_to,
            confidence=confidence,
            source=source or "graphiti",
            properties=properties or {},
            created_at=now_iso_utc(),
            provenance_id=provenance_id,
        )

        # Write to backend
        if self.using_falkordb:
            self._add_fact_falkordb(fact, subject_ent, object_ent)
        self._append_ndjson(self.facts_path, fact.to_dict())  # Always write to NDJSON as backup
        return {
            "status": "created",
            "id": fact.id,
            "predicate": fact.predicate,
            "provenance_id": fact.provenance_id,
        }

    def _add_fact_falkordb(self, fact: Fact, subject: Entity, obj: Entity) -> None:
        """Add fact as edge to FalkorDB using parameterized query."""
        if not self._falkordb_graph:
            return
        try:
            # Validate and normalize predicate for use as relationship type
            predicate = fact.predicate.replace(' ', '_').replace('-', '_').upper()
            if not predicate.replace('_', '').isalnum():
                predicate = "RELATES_TO"

            # Use parameterized query for node properties
            # Note: FalkorDB doesn't support parameterized relationship types
            query = f"""
            MERGE (s:entity {{id: $subject_id}})
            ON CREATE SET s.name = $subject_name
            MERGE (o:entity {{id: $object_id}})
            ON CREATE SET o.name = $object_name
            MERGE (s)-[r:{predicate}]->(o)
            SET r.id = $fact_id,
                r.confidence = $confidence,
                r.valid_from = $valid_from,
                r.valid_to = $valid_to,
                r.source = $source,
                r.created_at = $created_at
            RETURN r.id
            """
            params = {
                "subject_id": subject.id,
                "subject_name": subject.name,
                "object_id": obj.id,
                "object_name": obj.name,
                "fact_id": fact.id,
                "confidence": fact.confidence,
                "valid_from": fact.valid_from,
                "valid_to": fact.valid_to or "",
                "source": fact.source,
                "created_at": fact.created_at,
            }
            self._falkordb_graph.query(query, params)
            self.invalidate_cache()  # Invalidate cache after write
        except Exception as e:
            # Log but don't fail - NDJSON is already written as backup
            pass

    def _latest_facts(self) -> Dict[str, Fact]:
        items = self._read_ndjson(self.facts_path)
        latest: Dict[str, Fact] = {}
        for item in items:
            if item.get("kind") != "graphiti_fact":
                continue
            fact = Fact.from_dict(item)
            if fact.id:
                latest[fact.id] = fact
        return latest

    def invalidate_fact(self, *, fact_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        facts = self._latest_facts()
        if fact_id not in facts:
            return {"error": "fact not found"}
        fact = facts[fact_id]
        updated = Fact(
            id=fact.id,
            subject_id=fact.subject_id,
            predicate=fact.predicate,
            object_id=fact.object_id,
            valid_from=fact.valid_from,
            valid_to=now_iso_utc(),
            confidence=fact.confidence,
            source=fact.source,
            properties=dict(fact.properties),
            created_at=fact.created_at,
            provenance_id=fact.provenance_id,
        )
        if reason:
            updated.properties["invalidated_reason"] = reason
        self._append_ndjson(self.facts_path, updated.to_dict())
        return {"status": "invalidated", "id": fact_id, "valid_to": updated.valid_to}

    def query_facts_falkordb(
        self,
        *,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_value: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Query facts using FalkorDB Cypher (faster for large graphs)."""
        if not self.using_falkordb:
            return {"error": "FalkorDB not available", "count": 0, "facts": []}

        try:
            # Build Cypher query dynamically based on filters
            where_clauses = []
            params: Dict[str, Any] = {"limit": limit}

            if subject:
                where_clauses.append("(s.name CONTAINS $subject OR s.id = $subject)")
                params["subject"] = subject
            if object_value:
                where_clauses.append("(o.name CONTAINS $object OR o.id = $object)")
                params["object"] = object_value

            where_str = " AND ".join(where_clauses) if where_clauses else "1=1"

            # Handle predicate filter (relationship type)
            if predicate:
                pred_normalized = predicate.replace(' ', '_').replace('-', '_').upper()
                query = f"""
                MATCH (s)-[r:{pred_normalized}]->(o)
                WHERE {where_str}
                RETURN s.id AS subject_id, s.name AS subject_name, s.entity_type AS subject_type,
                       type(r) AS predicate, r.id AS fact_id, r.confidence AS confidence,
                       r.valid_from AS valid_from, r.valid_to AS valid_to, r.source AS source,
                       o.id AS object_id, o.name AS object_name, o.entity_type AS object_type
                LIMIT $limit
                """
            else:
                query = f"""
                MATCH (s)-[r]->(o)
                WHERE {where_str}
                RETURN s.id AS subject_id, s.name AS subject_name, s.entity_type AS subject_type,
                       type(r) AS predicate, r.id AS fact_id, r.confidence AS confidence,
                       r.valid_from AS valid_from, r.valid_to AS valid_to, r.source AS source,
                       o.id AS object_id, o.name AS object_name, o.entity_type AS object_type
                LIMIT $limit
                """

            results = self.query_cypher(query, params, use_cache=True)

            facts = []
            for row in results:
                facts.append({
                    "id": row.get("fact_id") or "",
                    "subject": {
                        "id": row.get("subject_id") or "",
                        "name": row.get("subject_name") or "",
                        "type": row.get("subject_type") or "entity",
                    },
                    "predicate": row.get("predicate") or "",
                    "object": {
                        "id": row.get("object_id") or "",
                        "name": row.get("object_name") or "",
                        "type": row.get("object_type") or "entity",
                    },
                    "confidence": row.get("confidence") if row.get("confidence") is not None else 1.0,
                    "valid_from": row.get("valid_from") or "",
                    "valid_to": row.get("valid_to") or "",
                    "source": row.get("source") or "falkordb",
                })

            return {"count": len(facts), "facts": facts, "source": "falkordb"}

        except FalkorDBError:
            # Fall back to NDJSON
            return self._query_facts_ndjson(
                subject=subject, predicate=predicate, object_value=object_value
            )

    def _query_facts_ndjson(
        self,
        *,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_value: Optional[str] = None,
        at_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query facts from NDJSON (original implementation)."""
        facts = list(self._latest_facts().values())
        entities = self._latest_entities()
        name_map = self._entity_name_map(entities)
        at_ts = parse_iso_ts(at_time) if at_time else 0.0

        def entity_for_id(ent_id: str) -> Entity:
            return entities.get(ent_id) or Entity(id=ent_id, name=ent_id)

        filtered = []
        for fact in facts:
            subj = entity_for_id(fact.subject_id)
            obj = entity_for_id(fact.object_id)

            if subject:
                match = subject.lower() == subj.name.lower() or subject.lower() in {a.lower() for a in subj.aliases}
                match = match or subject == subj.id
                if not match:
                    continue
            if predicate and predicate != fact.predicate:
                continue
            if object_value:
                match_obj = object_value.lower() == obj.name.lower() or object_value == obj.id
                if not match_obj:
                    continue
            if at_ts and not fact.is_valid_at(at_ts):
                continue

            filtered.append({
                "id": fact.id,
                "subject": {
                    "id": subj.id,
                    "name": subj.name,
                    "type": subj.entity_type,
                },
                "predicate": fact.predicate,
                "object": {
                    "id": obj.id,
                    "name": obj.name,
                    "type": obj.entity_type,
                },
                "valid_from": fact.valid_from,
                "valid_to": fact.valid_to,
                "confidence": fact.confidence,
                "source": fact.source,
                "properties": fact.properties,
                "created_at": fact.created_at,
                "provenance_id": fact.provenance_id,
            })

        return {"count": len(filtered), "facts": filtered, "source": "ndjson"}

    def query_facts(
        self,
        *,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_value: Optional[str] = None,
        at_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query facts - uses FalkorDB if available, falls back to NDJSON."""
        # Use FalkorDB for faster queries when available (unless at_time filter is used)
        if self.using_falkordb and not at_time:
            result = self.query_facts_falkordb(
                subject=subject, predicate=predicate, object_value=object_value
            )
            if result.get("source") == "falkordb":
                return result

        # Fall back to NDJSON for at_time queries or when FalkorDB unavailable
        return self._query_facts_ndjson(
            subject=subject, predicate=predicate, object_value=object_value, at_time=at_time
        )

    def tools_list(self) -> Dict[str, Any]:
        return {
            "tools": [
                {"name": "graphiti_add_entity"},
                {"name": "graphiti_add_fact"},
                {"name": "graphiti_query"},
                {"name": "graphiti_get_entity"},
                {"name": "graphiti_list_entities"},
                {"name": "graphiti_invalidate"},
            ]
        }

    def tools_call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name == "graphiti_add_entity":
            return self.add_entity(
                name=arguments.get("name", ""),
                entity_type=arguments.get("entity_type"),
                properties=arguments.get("properties"),
                aliases=arguments.get("aliases"),
            )
        if name == "graphiti_add_fact":
            return self.add_fact(
                subject=arguments.get("subject", ""),
                predicate=arguments.get("predicate", ""),
                object_value=arguments.get("object_value") or arguments.get("object", ""),
                valid_from=arguments.get("valid_from"),
                valid_to=arguments.get("valid_to"),
                confidence=float(arguments.get("confidence", 1.0)),
                source=arguments.get("source"),
                properties=arguments.get("properties"),
                provenance_id=arguments.get("provenance_id"),
            )
        if name == "graphiti_query":
            return self.query_facts(
                subject=arguments.get("subject"),
                predicate=arguments.get("predicate"),
                object_value=arguments.get("object"),
                at_time=arguments.get("at_time"),
            )
        if name == "graphiti_get_entity":
            return self.get_entity(
                entity_id=arguments.get("entity_id"),
                name=arguments.get("name"),
            )
        if name == "graphiti_list_entities":
            return self.list_entities(entity_type=arguments.get("entity_type"))
        if name == "graphiti_invalidate":
            return self.invalidate_fact(
                fact_id=arguments.get("fact_id", ""),
                reason=arguments.get("reason"),
            )
        return {"error": f"unknown tool: {name}"}

    def query(self, subject: Optional[str] = None, predicate: Optional[str] = None, object_val: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        res = self.query_facts(subject=subject, predicate=predicate, object_value=object_val)
        rows = []
        for fact in res["facts"][:limit]:
            rows.append({
                "s.name": fact["subject"]["name"],
                "f.predicate": fact["predicate"],
                "object": fact["object"]["name"],
            })
        return rows


def extract_facts_from_bus_event(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    kind = event.get("kind")
    if kind == "metric":
        return []

    topic = str(event.get("topic") or "")
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    actor = str(event.get("actor") or "unknown")
    iso = event.get("iso") or now_iso_utc()
    event_id = event.get("id")
    facts: List[Dict[str, Any]] = []

    task_id = data.get("task_id")
    status = data.get("status") or data.get("state")
    if task_id and status:
        facts.append({
            "subject": f"task:{task_id}",
            "predicate": "has_status",
            "object": status,
            "valid_from": iso,
            "confidence": 1.0,
            "source": f"bus:{topic}",
            "provenance_id": event_id,
        })

    if kind == "request":
        req_id = data.get("req_id") or event.get("req_id")
        if req_id:
            facts.append({
                "subject": f"actor:{actor}",
                "predicate": "sent_request",
                "object": f"request:{req_id}",
                "valid_from": iso,
                "confidence": 1.0,
                "source": f"bus:{topic}",
                "provenance_id": event_id,
            })

    event_type = event.get("event_type")
    session_id = data.get("session_id") or event.get("session_id")
    if topic == "dialogos.session_start" or event_type == "session_start":
        if session_id:
            facts.append({
                "subject": actor,
                "predicate": "started_session",
                "object": f"session:{session_id}",
                "valid_from": iso,
                "confidence": 1.0,
                "source": f"dialogos:{topic}",
                "provenance_id": event_id,
            })

    if topic == "dialogos.session_end" or event_type == "session_end":
        if session_id:
            facts.append({
                "subject": actor,
                "predicate": "ended_session",
                "object": f"session:{session_id}",
                "valid_from": iso,
                "confidence": 1.0,
                "source": f"dialogos:{topic}",
                "provenance_id": event_id,
            })

    return facts


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run", help="Run a minimal ingestion pass over the bus")
    q = subparsers.add_parser("query", help="Query facts")
    q.add_argument("--subject", "-s")
    q.add_argument("--predicate", "-p")
    q.add_argument("--object", "-o")
    q.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()
    service = GraphitiService()

    if args.command == "run":
        bus_path = service.root / ".pluribus" / "bus" / "events.ndjson"
        if not bus_path.exists():
            print("bus events not found", file=sys.stderr)
            return 1
        for line in bus_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            for fact in extract_facts_from_bus_event(event):
                service.add_fact(
                    subject=fact["subject"],
                    predicate=fact["predicate"],
                    object_value=fact["object"],
                    valid_from=fact.get("valid_from"),
                    confidence=fact.get("confidence", 1.0),
                    source=fact.get("source"),
                    provenance_id=fact.get("provenance_id"),
                )
        return 0

    if args.command == "query":
        rows = service.query(subject=args.subject, predicate=args.predicate, object_val=args.object, limit=args.limit)
        print(json.dumps(rows, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
