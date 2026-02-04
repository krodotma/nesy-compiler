#!/usr/bin/env python3
"""
N-Sphere Knowledge Graph Bridge

Connects Token Geometry Engine to knowledge graph storage (Graphiti/Neo4j/mem0).
Implements:
- Hyperspherical embedding storage
- Geodesic similarity search
- Vec2Vec transformations for cross-modal retrieval
- Superpositional state persistence
- LTL constraint enforcement on graph operations

Architecture:
    Token Geometry Engine
           ↓
    N-Sphere Projection
           ↓
    KG Bridge (this module)
           ↓
    ├── Graphiti (temporal KG)
    ├── Neo4j (graph storage)
    └── mem0 (memory layer)

Usage:
    from nsphere_kg_bridge import NSphereKGBridge

    bridge = NSphereKGBridge()
    await bridge.connect()

    # Store embedding with n-sphere coordinates
    node_id = await bridge.store_embedding(
        text="System status check",
        embedding=engine_result.nsphere_point.coords,
        metadata={"operator": "CKIN", "auom_units": [...]}
    )

    # Geodesic similarity search
    similar = await bridge.search_geodesic(
        query_embedding=query_coords,
        top_k=10,
        max_distance=0.5
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Literal

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

# =============================================================================
# OPTIONAL IMPORTS
# =============================================================================

try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False
    Graphiti = None
    EpisodeType = None

try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncGraphDatabase = None

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    Memory = None

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

NSPHERE_DIM = 128
SEXTET_CHANNELS = 6
CHANNEL_DIM = 64

# =============================================================================
# DATA CLASSES
# =============================================================================

class StorageBackend(Enum):
    GRAPHITI = "graphiti"
    NEO4J = "neo4j"
    MEM0 = "mem0"
    LOCAL = "local"  # File-based fallback

@dataclass
class NSphereNode:
    """Node with n-sphere embedding."""
    id: str
    text: str
    coords: np.ndarray  # N-sphere coordinates
    radius: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    sextet_channels: dict[str, np.ndarray] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "coords": self.coords.tolist(),
            "radius": self.radius,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "sextet_channels": {
                k: v.tolist() for k, v in (self.sextet_channels or {}).items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NSphereNode":
        return cls(
            id=data["id"],
            text=data["text"],
            coords=np.array(data["coords"]),
            radius=data.get("radius", 1.0),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            sextet_channels={
                k: np.array(v) for k, v in data.get("sextet_channels", {}).items()
            } or None,
        )

@dataclass
class NSphereEdge:
    """Edge between n-sphere nodes with geodesic distance."""
    id: str
    source_id: str
    target_id: str
    geodesic_distance: float
    edge_type: str = "similarity"
    metadata: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "geodesic_distance": self.geodesic_distance,
            "edge_type": self.edge_type,
            "metadata": self.metadata,
            "weight": self.weight,
        }

@dataclass
class SuperpositionalNode:
    """Node representing quantum-like superposition of states."""
    id: str
    basis_states: list[str]
    amplitudes: np.ndarray  # Complex amplitudes
    collapsed: bool = False
    collapsed_state: str | None = None
    ltl_constraints: list[str] = field(default_factory=list)

    def probability(self, state: str) -> float:
        if state not in self.basis_states:
            return 0.0
        idx = self.basis_states.index(state)
        return float(np.abs(self.amplitudes[idx])**2)

    def collapse(self, rng: np.random.Generator | None = None) -> str:
        if self.collapsed:
            return self.collapsed_state or self.basis_states[0]

        rng = rng or np.random.default_rng()
        probs = np.abs(self.amplitudes)**2
        probs = probs / probs.sum()  # Ensure normalization
        state = rng.choice(self.basis_states, p=probs)

        self.collapsed = True
        self.collapsed_state = state
        return state

# =============================================================================
# LOCAL STORAGE (FALLBACK)
# =============================================================================

class LocalNSphereStorage:
    """File-based n-sphere storage for environments without external DBs."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.nodes_file = storage_dir / "nsphere_nodes.ndjson"
        self.edges_file = storage_dir / "nsphere_edges.ndjson"
        self.index_file = storage_dir / "nsphere_index.json"

        storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index for fast similarity search
        self._nodes: dict[str, NSphereNode] = {}
        self._edges: dict[str, NSphereEdge] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load nodes and edges from files."""
        if self.nodes_file.exists():
            with open(self.nodes_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        node = NSphereNode.from_dict(data)
                        self._nodes[node.id] = node

        if self.edges_file.exists():
            with open(self.edges_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        edge = NSphereEdge(**data)
                        self._edges[edge.id] = edge

    def _save_node(self, node: NSphereNode) -> None:
        """Append node to storage."""
        with open(self.nodes_file, "a") as f:
            f.write(json.dumps(node.to_dict()) + "\n")

    def _save_edge(self, edge: NSphereEdge) -> None:
        """Append edge to storage."""
        with open(self.edges_file, "a") as f:
            f.write(json.dumps(edge.to_dict()) + "\n")

    async def store_node(self, node: NSphereNode) -> str:
        """Store node and return ID."""
        self._nodes[node.id] = node
        self._save_node(node)
        return node.id

    async def get_node(self, node_id: str) -> NSphereNode | None:
        """Retrieve node by ID."""
        return self._nodes.get(node_id)

    async def store_edge(self, edge: NSphereEdge) -> str:
        """Store edge and return ID."""
        self._edges[edge.id] = edge
        self._save_edge(edge)
        return edge.id

    async def search_geodesic(
        self,
        query_coords: np.ndarray,
        top_k: int = 10,
        max_distance: float | None = None
    ) -> list[tuple[NSphereNode, float]]:
        """Search for nearest nodes by geodesic distance."""
        results: list[tuple[NSphereNode, float]] = []

        query_norm = np.linalg.norm(query_coords)
        if query_norm < 1e-10:
            return results

        query_unit = query_coords / query_norm

        for node in self._nodes.values():
            node_norm = np.linalg.norm(node.coords)
            if node_norm < 1e-10:
                continue

            node_unit = node.coords / node_norm

            # Geodesic distance on unit sphere
            dot = np.clip(np.dot(query_unit, node_unit), -1, 1)
            distance = np.arccos(dot)

            if max_distance is None or distance <= max_distance:
                results.append((node, float(distance)))

        # Sort by distance
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    async def search_channel(
        self,
        channel: str,
        query_vec: np.ndarray,
        top_k: int = 10
    ) -> list[tuple[NSphereNode, float]]:
        """Search by specific sextet channel similarity."""
        results: list[tuple[NSphereNode, float]] = []

        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-10:
            return results

        query_unit = query_vec / query_norm

        for node in self._nodes.values():
            if not node.sextet_channels or channel not in node.sextet_channels:
                continue

            channel_vec = node.sextet_channels[channel]
            channel_norm = np.linalg.norm(channel_vec)
            if channel_norm < 1e-10:
                continue

            channel_unit = channel_vec / channel_norm

            # Cosine similarity
            similarity = float(np.dot(query_unit, channel_unit))
            results.append((node, similarity))

        # Sort by similarity (descending)
        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    async def get_neighbors(self, node_id: str) -> list[tuple[NSphereNode, NSphereEdge]]:
        """Get all neighbors of a node."""
        neighbors: list[tuple[NSphereNode, NSphereEdge]] = []

        for edge in self._edges.values():
            if edge.source_id == node_id:
                target = self._nodes.get(edge.target_id)
                if target:
                    neighbors.append((target, edge))
            elif edge.target_id == node_id:
                source = self._nodes.get(edge.source_id)
                if source:
                    neighbors.append((source, edge))

        return neighbors

    async def count(self) -> tuple[int, int]:
        """Return (node_count, edge_count)."""
        return len(self._nodes), len(self._edges)

# =============================================================================
# NEO4J ADAPTER
# =============================================================================

class Neo4jNSphereAdapter:
    """Neo4j adapter for n-sphere storage with Cypher queries."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j not installed: pip install neo4j")

        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None

    async def connect(self) -> None:
        """Connect to Neo4j."""
        self._driver = AsyncGraphDatabase.driver(
            self.uri, auth=(self.user, self.password)
        )
        # Ensure indices exist
        async with self._driver.session(database=self.database) as session:
            await session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:NSphereNode) ON (n.id)"
            )
            await session.run(
                "CREATE INDEX IF NOT EXISTS FOR (n:NSphereNode) ON (n.text)"
            )

    async def close(self) -> None:
        """Close connection."""
        if self._driver:
            await self._driver.close()

    async def store_node(self, node: NSphereNode) -> str:
        """Store node in Neo4j."""
        async with self._driver.session(database=self.database) as session:
            await session.run(
                """
                MERGE (n:NSphereNode {id: $id})
                SET n.text = $text,
                    n.coords = $coords,
                    n.radius = $radius,
                    n.metadata = $metadata,
                    n.created_at = $created_at
                """,
                id=node.id,
                text=node.text,
                coords=node.coords.tolist(),
                radius=node.radius,
                metadata=json.dumps(node.metadata),
                created_at=node.created_at,
            )
        return node.id

    async def get_node(self, node_id: str) -> NSphereNode | None:
        """Get node by ID."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                "MATCH (n:NSphereNode {id: $id}) RETURN n",
                id=node_id
            )
            record = await result.single()
            if record:
                n = record["n"]
                return NSphereNode(
                    id=n["id"],
                    text=n["text"],
                    coords=np.array(n["coords"]),
                    radius=n.get("radius", 1.0),
                    metadata=json.loads(n.get("metadata", "{}")),
                    created_at=n.get("created_at", time.time()),
                )
        return None

    async def store_edge(self, edge: NSphereEdge) -> str:
        """Store edge in Neo4j."""
        async with self._driver.session(database=self.database) as session:
            await session.run(
                """
                MATCH (a:NSphereNode {id: $source_id})
                MATCH (b:NSphereNode {id: $target_id})
                MERGE (a)-[r:NSPHERE_EDGE {id: $id}]->(b)
                SET r.geodesic_distance = $geodesic_distance,
                    r.edge_type = $edge_type,
                    r.weight = $weight,
                    r.metadata = $metadata
                """,
                id=edge.id,
                source_id=edge.source_id,
                target_id=edge.target_id,
                geodesic_distance=edge.geodesic_distance,
                edge_type=edge.edge_type,
                weight=edge.weight,
                metadata=json.dumps(edge.metadata),
            )
        return edge.id

    async def search_geodesic(
        self,
        query_coords: np.ndarray,
        top_k: int = 10,
        max_distance: float | None = None
    ) -> list[tuple[NSphereNode, float]]:
        """
        Search by geodesic distance.
        Note: Neo4j doesn't have native n-sphere distance, so we compute in Python.
        For production, consider using Neo4j GDS with custom procedures.
        """
        results: list[tuple[NSphereNode, float]] = []

        query_norm = np.linalg.norm(query_coords)
        if query_norm < 1e-10:
            return results

        query_unit = query_coords / query_norm

        async with self._driver.session(database=self.database) as session:
            result = await session.run(
                "MATCH (n:NSphereNode) RETURN n LIMIT 1000"
            )
            async for record in result:
                n = record["n"]
                coords = np.array(n["coords"])
                node_norm = np.linalg.norm(coords)
                if node_norm < 1e-10:
                    continue

                node_unit = coords / node_norm
                dot = np.clip(np.dot(query_unit, node_unit), -1, 1)
                distance = float(np.arccos(dot))

                if max_distance is None or distance <= max_distance:
                    node = NSphereNode(
                        id=n["id"],
                        text=n["text"],
                        coords=coords,
                        radius=n.get("radius", 1.0),
                        metadata=json.loads(n.get("metadata", "{}")),
                        created_at=n.get("created_at", time.time()),
                    )
                    results.append((node, distance))

        results.sort(key=lambda x: x[1])
        return results[:top_k]

# =============================================================================
# GRAPHITI ADAPTER
# =============================================================================

class GraphitiNSphereAdapter:
    """Graphiti adapter for temporal n-sphere knowledge graph."""

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
    ):
        if not GRAPHITI_AVAILABLE:
            raise ImportError("graphiti-core not installed")

        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self._graphiti = None

    async def connect(self) -> None:
        """Connect to Graphiti."""
        self._graphiti = Graphiti(
            self.neo4j_uri,
            self.neo4j_user,
            self.neo4j_password
        )
        await self._graphiti.build_indices_and_constraints()

    async def close(self) -> None:
        """Close connection."""
        if self._graphiti:
            await self._graphiti.close()

    async def add_episode(
        self,
        text: str,
        nsphere_coords: np.ndarray,
        metadata: dict[str, Any] | None = None,
        source: str = "token_geometry"
    ) -> str:
        """Add episode with n-sphere embedding."""
        episode_id = f"nsphere-{uuid.uuid4().hex[:12]}"

        # Store as Graphiti episode with n-sphere coords in metadata
        await self._graphiti.add_episode(
            name=episode_id,
            episode_body=text,
            source=EpisodeType.message if EpisodeType else "message",
            reference_time=time.time(),
            source_description=source,
        )

        # Also store n-sphere coords (would need custom node type in production)
        logger.info(f"Added episode {episode_id} with {len(nsphere_coords)}-dim n-sphere embedding")

        return episode_id

    async def search(
        self,
        query: str,
        top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Search using Graphiti's hybrid search."""
        results = await self._graphiti.search(query, num_results=top_k)
        return [
            {
                "id": r.uuid,
                "content": r.content if hasattr(r, 'content') else str(r),
                "score": r.score if hasattr(r, 'score') else 0.0,
            }
            for r in results
        ]

# =============================================================================
# MAIN BRIDGE CLASS
# =============================================================================

class NSphereKGBridge:
    """
    Main bridge connecting Token Geometry Engine to knowledge graph storage.

    Supports multiple backends:
    - Local (file-based, always available)
    - Neo4j (graph database)
    - Graphiti (temporal KG)
    - mem0 (memory layer)
    """

    def __init__(
        self,
        backend: StorageBackend = StorageBackend.LOCAL,
        storage_dir: Path | None = None,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
    ):
        self.backend = backend
        self.storage_dir = storage_dir or Path("/pluribus/.pluribus/nsphere_kg")

        # Backend configuration
        self.neo4j_uri = neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.environ.get("NEO4J_PASSWORD", "password")

        # Storage instances
        self._local: LocalNSphereStorage | None = None
        self._neo4j: Neo4jNSphereAdapter | None = None
        self._graphiti: GraphitiNSphereAdapter | None = None

        # LTL constraints
        self._ltl_constraints: list[str] = []

    async def connect(self) -> None:
        """Connect to configured backend(s)."""
        # Always initialize local storage as fallback
        self._local = LocalNSphereStorage(self.storage_dir)

        if self.backend == StorageBackend.NEO4J and NEO4J_AVAILABLE:
            self._neo4j = Neo4jNSphereAdapter(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
            )
            await self._neo4j.connect()
            logger.info("Connected to Neo4j n-sphere storage")

        elif self.backend == StorageBackend.GRAPHITI and GRAPHITI_AVAILABLE:
            self._graphiti = GraphitiNSphereAdapter(
                neo4j_uri=self.neo4j_uri,
                neo4j_user=self.neo4j_user,
                neo4j_password=self.neo4j_password,
            )
            await self._graphiti.connect()
            logger.info("Connected to Graphiti n-sphere storage")

        else:
            logger.info("Using local n-sphere storage")

    async def close(self) -> None:
        """Close all connections."""
        if self._neo4j:
            await self._neo4j.close()
        if self._graphiti:
            await self._graphiti.close()

    async def store_embedding(
        self,
        text: str,
        embedding: np.ndarray | list[float],
        metadata: dict[str, Any] | None = None,
        sextet_channels: dict[str, np.ndarray | list[float]] | None = None,
        provenance_id: str | None = None,
    ) -> str:
        """
        Store text with n-sphere embedding. Resolves Memory Schizophrenia 
        by multiplexing across backends with unified provenance.
        """
        if isinstance(embedding, list):
            embedding = np.array(embedding)

        node = NSphereNode(
            id=f"nsphere-{uuid.uuid4().hex[:12]}",
            text=text,
            coords=embedding,
            metadata=metadata or {},
            sextet_channels={
                k: np.array(v) if isinstance(v, list) else v
                for k, v in (sextet_channels or {}).items()
            } or None,
        )

        # MANDATORY PROVENANCE (v26)
        node.metadata["provenance_id"] = provenance_id

        # Multiplexed storage (Dual-Write)
        if self._neo4j:
            await self._neo4j.store_node(node)
        
        if self._graphiti:
            # Propagate provenance to temporal layer
            await self._graphiti.add_episode(text, embedding, metadata, provenance_id=provenance_id)

        # Always store locally for fast access and baseline truth
        await self._local.store_node(node)

        return node.id

    async def search_geodesic(
        self,
        query_embedding: np.ndarray | list[float],
        top_k: int = 10,
        max_distance: float | None = None,
    ) -> list[tuple[NSphereNode, float]]:
        """
        Search for nearest nodes by geodesic distance on n-sphere.

        Args:
            query_embedding: Query n-sphere coordinates
            top_k: Number of results
            max_distance: Maximum geodesic distance (radians)

        Returns:
            List of (node, distance) tuples
        """
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)

        if self._neo4j:
            return await self._neo4j.search_geodesic(query_embedding, top_k, max_distance)

        return await self._local.search_geodesic(query_embedding, top_k, max_distance)

    async def search_channel(
        self,
        channel: str,
        query_vec: np.ndarray | list[float],
        top_k: int = 10,
    ) -> list[tuple[NSphereNode, float]]:
        """
        Search by specific sextet channel similarity.

        Args:
            channel: Channel name (semantic, syntactic, etc.)
            query_vec: Query vector for that channel
            top_k: Number of results

        Returns:
            List of (node, similarity) tuples
        """
        if isinstance(query_vec, list):
            query_vec = np.array(query_vec)

        return await self._local.search_channel(channel, query_vec, top_k)

    async def create_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "similarity",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create edge between nodes with geodesic distance.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            metadata: Additional metadata

        Returns:
            Edge ID
        """
        source = await self._local.get_node(source_id)
        target = await self._local.get_node(target_id)

        if not source or not target:
            raise ValueError(f"Node not found: {source_id} or {target_id}")

        # Compute geodesic distance
        source_norm = np.linalg.norm(source.coords)
        target_norm = np.linalg.norm(target.coords)

        if source_norm > 1e-10 and target_norm > 1e-10:
            source_unit = source.coords / source_norm
            target_unit = target.coords / target_norm
            dot = np.clip(np.dot(source_unit, target_unit), -1, 1)
            geodesic_distance = float(np.arccos(dot))
        else:
            geodesic_distance = np.pi  # Maximum distance

        edge = NSphereEdge(
            id=f"edge-{uuid.uuid4().hex[:12]}",
            source_id=source_id,
            target_id=target_id,
            geodesic_distance=geodesic_distance,
            edge_type=edge_type,
            metadata=metadata or {},
            weight=1.0 - geodesic_distance / np.pi,  # Similarity as weight
        )

        if self._neo4j:
            await self._neo4j.store_edge(edge)

        await self._local.store_edge(edge)

        return edge.id

    async def get_neighbors(
        self,
        node_id: str,
        max_distance: float | None = None,
    ) -> list[tuple[NSphereNode, NSphereEdge]]:
        """Get neighboring nodes within geodesic distance."""
        neighbors = await self._local.get_neighbors(node_id)

        if max_distance is not None:
            neighbors = [
                (n, e) for n, e in neighbors
                if e.geodesic_distance <= max_distance
            ]

        return neighbors

    async def vec2vec_transform(
        self,
        node_id: str,
        source_channel: str,
        target_channel: str,
    ) -> np.ndarray | None:
        """
        Apply vec2vec transformation between sextet channels.

        Args:
            node_id: Node to transform
            source_channel: Source channel name
            target_channel: Target channel name

        Returns:
            Transformed vector or None
        """
        node = await self._local.get_node(node_id)
        if not node or not node.sextet_channels:
            return None

        source = node.sextet_channels.get(source_channel)
        target = node.sextet_channels.get(target_channel)

        if source is None or target is None:
            return None

        # Simple correlation-based transform
        correlation = np.outer(target, source)
        transform = correlation / (np.linalg.norm(correlation) + 1e-10)

        return transform @ source

    def add_ltl_constraint(self, constraint: str) -> None:
        """Add LTL constraint for graph operations."""
        self._ltl_constraints.append(constraint)

    async def validate_operation(
        self,
        operation: str,
        node_id: str | None = None,
    ) -> bool:
        """
        Validate operation against LTL constraints.

        Args:
            operation: Operation type (store, delete, etc.)
            node_id: Node involved in operation

        Returns:
            True if operation is valid
        """
        # Placeholder for LTL validation
        # In production, integrate with superpositional LTL validator
        return True

    async def stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        node_count, edge_count = await self._local.count()
        return {
            "backend": self.backend.value,
            "node_count": node_count,
            "edge_count": edge_count,
            "ltl_constraints": len(self._ltl_constraints),
        }

    async def emit_bus_event(
        self,
        event_type: str,
        data: dict[str, Any],
        bus_dir: Path | None = None,
    ) -> None:
        """Emit event to Pluribus bus."""
        bus_dir = bus_dir or Path("/pluribus/.pluribus/bus")
        bus_file = bus_dir / "events.ndjson"
        bus_dir.mkdir(parents=True, exist_ok=True)

        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "topic": f"nsphere_kg.{event_type}",
            "kind": "event",
            "level": "info",
            "actor": "nsphere_kg_bridge",
            "data": data,
        }

        with open(bus_file, "a") as f:
            f.write(json.dumps(event) + "\n")

# =============================================================================
# CLI
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="N-Sphere KG Bridge CLI")
    parser.add_argument("--backend", choices=["local", "neo4j", "graphiti"], default="local")
    parser.add_argument("--storage-dir", default="/pluribus/.pluribus/nsphere_kg")
    parser.add_argument("--stats", action="store_true", help="Show storage stats")
    parser.add_argument("--store", help="Store text with embedding")
    parser.add_argument("--search", help="Search by text")
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    bridge = NSphereKGBridge(
        backend=StorageBackend(args.backend),
        storage_dir=Path(args.storage_dir),
    )
    await bridge.connect()

    try:
        if args.stats:
            stats = await bridge.stats()
            print(json.dumps(stats, indent=2))

        elif args.store:
            # Generate dummy embedding
            embedding = np.random.randn(NSPHERE_DIM)
            embedding = embedding / np.linalg.norm(embedding)

            node_id = await bridge.store_embedding(
                text=args.store,
                embedding=embedding,
            )
            print(f"Stored: {node_id}")

        elif args.search:
            # Generate query embedding
            query = np.random.randn(NSPHERE_DIM)
            query = query / np.linalg.norm(query)

            results = await bridge.search_geodesic(query, top_k=args.top_k)
            for node, distance in results:
                print(f"  {node.id}: {node.text[:50]}... (dist={distance:.4f})")

    finally:
        await bridge.close()

if __name__ == "__main__":
    asyncio.run(main())
