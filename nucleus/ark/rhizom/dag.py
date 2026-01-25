#!/usr/bin/env python3
"""
dag.py - RhizomDAG: Semantic commit graph with etymology and CMP

The Rhizom extends the traditional git commit DAG with:
- Etymology: semantic origin/purpose for each commit
- CMP: fitness score tracking
- H* entropy: 8-dimensional state vector
- Lineage: clade membership and ancestry
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterator
from pathlib import Path
from datetime import datetime


@dataclass
class RhizomNode:
    """A node in the Rhizom semantic DAG."""
    sha: str
    etymology: str = ""
    cmp: float = 0.5
    entropy: Dict[str, float] = field(default_factory=dict)
    parents: List[str] = field(default_factory=list)
    lineage_tags: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    witness_id: Optional[str] = None
    # DGM-compatible: soft compaction (mark, don't delete)
    compacted: bool = False  # True = low-CMP dead-end, excluded from active queries
    stepping_stone: bool = False  # True = stepping stone (low CMP but high descendant impact)
    
    def to_dict(self) -> Dict:
        return {
            "sha": self.sha,
            "etymology": self.etymology,
            "cmp": self.cmp,
            "entropy": self.entropy,
            "parents": self.parents,
            "lineage_tags": self.lineage_tags,
            "timestamp": self.timestamp,
            "witness_id": self.witness_id,
            "compacted": self.compacted,
            "stepping_stone": self.stepping_stone
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RhizomNode":
        # Handle legacy nodes without new fields
        data.setdefault("compacted", False)
        data.setdefault("stepping_stone", False)
        return cls(**data)


class RhizomDAG:
    """
    Semantic DAG for ARK commit lineage.
    
    Extends standard git history with:
    - Etymology tracking (semantic origin)
    - CMP scoring (cumulative fitness)
    - H* entropy vectors (8-dimensional state)
    - Lineage/clade membership
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path / ".ark" / "rhizom.json"
        self.nodes: Dict[str, RhizomNode] = {}
        self._load()
    
    def _load(self) -> None:
        """Load DAG from storage."""
        if self.storage_path.exists():
            data = json.loads(self.storage_path.read_text())
            for sha, node_data in data.get("nodes", {}).items():
                self.nodes[sha] = RhizomNode.from_dict(node_data)
    
    def _save(self) -> None:
        """Save DAG to storage."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "nodes": {sha: node.to_dict() for sha, node in self.nodes.items()}
        }
        self.storage_path.write_text(json.dumps(data, indent=2))
    
    def insert(self, node: RhizomNode) -> None:
        """Insert or update a node in the DAG."""
        self.nodes[node.sha] = node
        self._save()
    
    def get(self, sha: str) -> Optional[RhizomNode]:
        """Get a node by SHA."""
        return self.nodes.get(sha)
    
    def ancestry(self, sha: str, depth: int = 10) -> List[RhizomNode]:
        """Get ancestors of a node up to depth."""
        result = []
        current_sha = sha
        
        for _ in range(depth):
            node = self.nodes.get(current_sha)
            if not node or not node.parents:
                break
            
            parent_sha = node.parents[0]
            parent_node = self.nodes.get(parent_sha)
            if parent_node:
                result.append(parent_node)
                current_sha = parent_sha
            else:
                break
        
        return result
    
    def query_by_etymology(self, term: str) -> List[RhizomNode]:
        """Find nodes containing term in etymology."""
        return [n for n in self.nodes.values() if term.lower() in n.etymology.lower()]
    
    def query_by_clade(self, clade: str) -> List[RhizomNode]:
        """Find nodes belonging to a clade."""
        return [n for n in self.nodes.values() if clade in n.lineage_tags]
    
    def query_by_cmp(self, min_cmp: float) -> List[RhizomNode]:
        """Find nodes with CMP above threshold."""
        return [n for n in self.nodes.values() if n.cmp >= min_cmp]
    
    def cmp_trend(self, limit: int = 10) -> List[float]:
        """Get recent CMP trend (most recent first)."""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.timestamp,
            reverse=True
        )
        return [n.cmp for n in sorted_nodes[:limit]]
    
    def entropy_average(self) -> Dict[str, float]:
        """Calculate average entropy across all nodes."""
        if not self.nodes:
            return {}
        
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        
        for node in self.nodes.values():
            for key, value in node.entropy.items():
                totals[key] = totals.get(key, 0) + value
                counts[key] = counts.get(key, 0) + 1
        
        return {k: totals[k] / counts[k] for k in totals}
    
    def get_children(self, sha: str) -> List[str]:
        """Get child SHAs (nodes that have this SHA as parent)."""
        return [n.sha for n in self.nodes.values() if sha in n.parents]
    
    def active_nodes(self) -> Iterator[RhizomNode]:
        """Iterate over non-compacted nodes only."""
        return (n for n in self.nodes.values() if not n.compacted)
    
    def compact(self, cmp_threshold: float = 0.3, preserve_stepping_stones: bool = True) -> int:
        """
        Soft-compact the Rhizom DAG.
        
        DGM-compatible: marks low-CMP dead-ends as compacted but does NOT delete.
        This preserves stepping stones for open-ended evolution.
        
        Args:
            cmp_threshold: Nodes below this CMP are candidates for compaction
            preserve_stepping_stones: If True, skip nodes with high-CMP descendants
            
        Returns:
            Number of nodes marked as compacted
        """
        compacted_count = 0
        
        # Identify dead-ends (nodes with no children)
        all_parents = set()
        for node in self.nodes.values():
            all_parents.update(node.parents)
        
        dead_ends = [sha for sha in self.nodes if sha not in all_parents]
        
        for sha in dead_ends:
            node = self.nodes[sha]
            
            # Skip already compacted
            if node.compacted:
                continue
            
            # Skip if above threshold
            if node.cmp >= cmp_threshold:
                continue
            
            # Check for stepping stone potential
            if preserve_stepping_stones:
                # A stepping stone has low self-CMP but enabled high-CMP descendants
                # Since this is a dead-end, check ancestry instead
                ancestors = self.ancestry(sha, depth=5)
                if any(a.cmp > 0.7 for a in ancestors):
                    # This low-CMP node came from a high-CMP lineage - might be exploration
                    node.stepping_stone = True
                    continue
            
            # Mark as compacted
            node.compacted = True
            compacted_count += 1
        
        if compacted_count > 0:
            self._save()
        
        return compacted_count
    
    def uncompact(self, sha: str) -> bool:
        """Restore a compacted node to active status."""
        node = self.nodes.get(sha)
        if node and node.compacted:
            node.compacted = False
            self._save()
            return True
        return False
    
    def __len__(self) -> int:
        return len(self.nodes)
    
    def __iter__(self) -> Iterator[RhizomNode]:
        return iter(self.nodes.values())

