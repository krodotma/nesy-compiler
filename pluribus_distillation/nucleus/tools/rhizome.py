#!/usr/bin/env python3
"""
rhizome.py - Pre-Git Semantic DAG Layer

The rhizome is a semantic layer that exists *before* git crystallization.
It enables etymon→embedding→retrieval flows for portal inception.

Ring: 1 (Infrastructure)
Protocol: DKIN v28 | PAIP v16 | Citizen v1

Usage:
    python3 rhizome.py ingest <file> [--store] [--emit-bus] [--tag <tag>]...
    python3 rhizome.py search <query> [--semantic] [--limit N]
    python3 rhizome.py export <node-id> [--format md|json]
    python3 rhizome.py status
    python3 rhizome.py gc [--dry-run]

Bus Topics:
    rhizome.node.created
    rhizome.dag.updated
    rhizome.search.executed
"""

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Set


# Configuration
RHIZOME_DIR = Path(os.environ.get("PLURIBUS_RHIZOME_DIR", ".pluribus/rhizome"))
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", ".pluribus/bus"))


@dataclass
class RhizomeNode:
    """A node in the rhizome DAG."""
    id: str
    content_hash: str
    source_path: str
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    created_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RhizomeNode":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RhizomeDag:
    """
    The Rhizome DAG - a semantic layer before git crystallization.
    
    Supports:
    - CRUD operations on nodes
    - Parent/child relationships
    - Tag-based retrieval
    - Content-hash deduplication
    """

    def __init__(self, root_dir: Path = None):
        self.root_dir = root_dir or RHIZOME_DIR
        self.nodes_file = self.root_dir / "nodes.ndjson"
        self.index_file = self.root_dir / "index.json"
        self._nodes: Dict[str, RhizomeNode] = {}
        self._hash_index: Dict[str, str] = {}  # content_hash -> node_id
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> set of node_ids
        self._load()

    def _load(self):
        """Load nodes from disk."""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        if self.nodes_file.exists():
            for line in self.nodes_file.read_text().strip().split("\n"):
                if line:
                    try:
                        data = json.loads(line)
                        node = RhizomeNode.from_dict(data)
                        self._nodes[node.id] = node
                        self._hash_index[node.content_hash] = node.id
                        for tag in node.tags:
                            if tag not in self._tag_index:
                                self._tag_index[tag] = set()
                            self._tag_index[tag].add(node.id)
                    except json.JSONDecodeError:
                        continue

    def _save_node(self, node: RhizomeNode):
        """Append a node to the NDJSON file."""
        with open(self.nodes_file, "a") as f:
            f.write(json.dumps(node.to_dict()) + "\n")

    def _rebuild_nodes_file(self):
        """Rebuild the nodes file (for compaction/gc)."""
        with open(self.nodes_file, "w") as f:
            for node in self._nodes.values():
                f.write(json.dumps(node.to_dict()) + "\n")

    def get(self, node_id: str) -> Optional[RhizomeNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_by_hash(self, content_hash: str) -> Optional[RhizomeNode]:
        """Get a node by content hash."""
        node_id = self._hash_index.get(content_hash)
        return self._nodes.get(node_id) if node_id else None

    def get_by_tag(self, tag: str) -> List[RhizomeNode]:
        """Get all nodes with a given tag."""
        node_ids = self._tag_index.get(tag, set())
        return [self._nodes[nid] for nid in node_ids if nid in self._nodes]

    def create(
        self,
        source_path: str,
        content: bytes,
        tags: List[str] = None,
        parents: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> RhizomeNode:
        """
        Create a new rhizome node.
        
        Returns existing node if content hash matches (deduplication).
        """
        content_hash = hashlib.sha256(content).hexdigest()
        
        # Deduplication
        existing = self.get_by_hash(content_hash)
        if existing:
            # Update tags if new ones provided
            if tags:
                for tag in tags:
                    if tag not in existing.tags:
                        existing.tags.append(tag)
                        if tag not in self._tag_index:
                            self._tag_index[tag] = set()
                        self._tag_index[tag].add(existing.id)
            return existing

        node = RhizomeNode(
            id=uuid.uuid4().hex[:16],
            content_hash=content_hash,
            source_path=source_path,
            tags=tags or [],
            parents=parents or [],
            metadata=metadata or {},
        )
        
        self._nodes[node.id] = node
        self._hash_index[content_hash] = node.id
        for tag in node.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(node.id)
        
        self._save_node(node)
        return node

    def add_child(self, parent_id: str, child_id: str):
        """Add a parent-child relationship."""
        parent = self._nodes.get(parent_id)
        child = self._nodes.get(child_id)
        if parent and child:
            if child_id not in parent.children:
                parent.children.append(child_id)
            if parent_id not in child.parents:
                child.parents.append(parent_id)
            self._rebuild_nodes_file()

    def delete(self, node_id: str) -> bool:
        """Delete a node (mark as deleted, actual removal on gc)."""
        if node_id in self._nodes:
            node = self._nodes.pop(node_id)
            self._hash_index.pop(node.content_hash, None)
            for tag in node.tags:
                if tag in self._tag_index:
                    self._tag_index[tag].discard(node_id)
            self._rebuild_nodes_file()
            return True
        return False

    def gc(self, dry_run: bool = False) -> int:
        """Garbage collect orphaned nodes."""
        # For now, just compact the file
        if not dry_run:
            self._rebuild_nodes_file()
        return 0

    def stats(self) -> Dict[str, Any]:
        """Return DAG statistics."""
        return {
            "node_count": len(self._nodes),
            "tag_count": len(self._tag_index),
            "tags": {tag: len(ids) for tag, ids in self._tag_index.items()},
        }


def emit_bus_event(topic: str, data: Dict[str, Any], level: str = "info"):
    """Emit event to the Pluribus bus."""
    bus_path = BUS_DIR / "events.ndjson"
    bus_path.parent.mkdir(parents=True, exist_ok=True)
    
    event = {
        "id": uuid.uuid4().hex,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "kind": "event",
        "level": level,
        "actor": "rhizome",
        "data": data,
    }
    
    with open(bus_path, "a") as f:
        f.write(json.dumps(event) + "\n")


def cmd_ingest(args):
    """Ingest a file into the rhizome."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1
    
    content = file_path.read_bytes()
    dag = RhizomeDag(Path(args.root) / ".pluribus" / "rhizome" if args.root else None)
    
    node = dag.create(
        source_path=str(file_path),
        content=content,
        tags=args.tag or [],
        metadata={"size": len(content), "ingested_at": datetime.now(timezone.utc).isoformat()},
    )
    
    print(f"Node: {node.id}")
    print(f"Hash: {node.content_hash[:16]}...")
    print(f"Tags: {', '.join(node.tags) or '(none)'}")
    
    if args.emit_bus:
        emit_bus_event("rhizome.node.created", {
            "node_id": node.id,
            "content_hash": node.content_hash,
            "source_path": str(file_path),
            "tags": node.tags,
        })
        print("Bus event emitted: rhizome.node.created")
    
    return 0


def cmd_search(args):
    """Search the rhizome by tag or query."""
    dag = RhizomeDag(Path(args.root) / ".pluribus" / "rhizome" if args.root else None)
    
    # For now, simple tag-based search
    # TODO: Add semantic search with embeddings
    results = []
    
    if args.query.startswith("tag:"):
        tag = args.query[4:]
        results = dag.get_by_tag(tag)
    else:
        # Fuzzy search on source paths
        query_lower = args.query.lower()
        for node in dag._nodes.values():
            if query_lower in node.source_path.lower():
                results.append(node)
    
    limit = args.limit or 10
    for node in results[:limit]:
        print(f"{node.id}: {node.source_path} [{', '.join(node.tags)}]")
    
    print(f"\n{len(results)} results (showing {min(len(results), limit)})")
    return 0


def cmd_status(args):
    """Show rhizome status."""
    dag = RhizomeDag(Path(args.root) / ".pluribus" / "rhizome" if args.root else None)
    stats = dag.stats()
    
    print("Rhizome Status")
    print(f"  Nodes: {stats['node_count']}")
    print(f"  Tags:  {stats['tag_count']}")
    if stats["tags"]:
        print("  Tag distribution:")
        for tag, count in sorted(stats["tags"].items(), key=lambda x: -x[1])[:10]:
            print(f"    {tag}: {count}")
    return 0


def cmd_gc(args):
    """Garbage collect the rhizome."""
    dag = RhizomeDag(Path(args.root) / ".pluribus" / "rhizome" if args.root else None)
    removed = dag.gc(dry_run=args.dry_run)
    
    if args.dry_run:
        print(f"[Dry run] Would remove {removed} orphaned nodes")
    else:
        print(f"Removed {removed} orphaned nodes")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Rhizome - Pre-Git Semantic DAG")
    parser.add_argument("--root", help="Repository root directory")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a file")
    p_ingest.add_argument("file", help="File to ingest")
    p_ingest.add_argument("--store", action="store_true", help="Store content (default: hash only)")
    p_ingest.add_argument("--emit-bus", action="store_true", help="Emit bus event")
    p_ingest.add_argument("--tag", action="append", help="Add tag (repeatable)")
    
    # search
    p_search = subparsers.add_parser("search", help="Search the rhizome")
    p_search.add_argument("query", help="Search query (or tag:<tagname>)")
    p_search.add_argument("--semantic", action="store_true", help="Use semantic search")
    p_search.add_argument("--limit", type=int, help="Max results")
    
    # status
    subparsers.add_parser("status", help="Show status")
    
    # gc
    p_gc = subparsers.add_parser("gc", help="Garbage collect")
    p_gc.add_argument("--dry-run", action="store_true", help="Dry run")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        return cmd_ingest(args)
    elif args.command == "search":
        return cmd_search(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "gc":
        return cmd_gc(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
