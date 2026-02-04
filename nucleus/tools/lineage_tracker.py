#!/usr/bin/env python3
"""
lineage_tracker.py - Lineage DAG Tracker for CMP Computation

DUALITY-BIND Phase 1: Lineage Foundation

This module builds and maintains the lineage tree from bus events,
enabling CMP (Clade Meta-Productivity) computation across agent lineages.

Ring: 1 (Operator)
Protocol: DKIN v29 | PAIP v15 | Citizen v1

Usage:
    python3 lineage_tracker.py build          # Build tree from bus events
    python3 lineage_tracker.py show <id>      # Show lineage subtree
    python3 lineage_tracker.py cmp <id>       # Compute CMP for lineage
    python3 lineage_tracker.py stats          # Show lineage statistics
"""

import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

# Configuration
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", ".pluribus/bus"))
LINEAGE_DIR = Path(os.environ.get("PLURIBUS_LINEAGE_DIR", ".pluribus/lineage"))

# CMP Discount Factor (temporal decay for descendant contributions)
GAMMA = 0.95  # ~5% decay per generation
# Spectral smoothing factor across siblings
ALPHA_SIBLING = 0.1


@dataclass
class LineageNode:
    """A node in the lineage DAG."""
    lineage_id: str
    parent_id: Optional[str] = None
    actor: str = "unknown"
    created_ts: float = field(default_factory=time.time)
    mutation_op: Optional[str] = None
    
    # Metrics (aggregated from events)
    event_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    reward_sum: float = 0.0
    
    # CMP components
    raw_cmp: float = 0.0  # Before descendant aggregation
    computed_cmp: float = 0.0  # After descendant aggregation
    descendant_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageNode":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LineageTracker:
    """
    Builds and maintains the lineage DAG from bus events.
    
    The lineage DAG tracks:
    - Parent-child relationships between agent lineages
    - Metrics per lineage (events, successes, failures, rewards)
    - CMP computation with temporal discount and sibling smoothing
    """
    
    def __init__(self, bus_dir: Path = None, lineage_dir: Path = None):
        self.bus_dir = bus_dir or BUS_DIR
        self.lineage_dir = lineage_dir or LINEAGE_DIR
        self.bus_path = self.bus_dir / "events.ndjson"
        self.lineage_path = self.lineage_dir / "lineages.ndjson"
        self.lineage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory tree
        self.nodes: Dict[str, LineageNode] = {}
        self.children: Dict[str, List[str]] = defaultdict(list)  # parent_id -> [child_ids]
        self.roots: Set[str] = set()  # Lineages with no parent
    
    def _load_state(self):
        """Load lineage state from disk."""
        if not self.lineage_path.exists():
            return
        
        for line in self.lineage_path.read_text().strip().split("\n"):
            if line:
                try:
                    data = json.loads(line)
                    node = LineageNode.from_dict(data)
                    self.nodes[node.lineage_id] = node
                    if node.parent_id:
                        self.children[node.parent_id].append(node.lineage_id)
                    else:
                        self.roots.add(node.lineage_id)
                except (json.JSONDecodeError, TypeError):
                    continue
    
    def _save_state(self):
        """Save lineage state to disk."""
        with open(self.lineage_path, "w") as f:
            for node in self.nodes.values():
                f.write(json.dumps(node.to_dict()) + "\n")
    
    def build_from_bus(self) -> int:
        """
        Build lineage tree from bus events.
        
        Returns:
            Number of new lineages discovered
        """
        if not self.bus_path.exists():
            return 0
        
        self._load_state()
        new_count = 0
        
        for line in self.bus_path.read_text().strip().split("\n"):
            if not line:
                continue
            
            try:
                event = json.loads(line)
                lineage_id = event.get("lineage_id")
                
                if not lineage_id:
                    continue
                
                # Create or update lineage node
                if lineage_id not in self.nodes:
                    self.nodes[lineage_id] = LineageNode(
                        lineage_id=lineage_id,
                        parent_id=event.get("parent_lineage_id"),
                        actor=event.get("actor", "unknown"),
                        created_ts=event.get("ts", time.time()),
                        mutation_op=event.get("mutation_op"),
                    )
                    
                    parent_id = event.get("parent_lineage_id")
                    if parent_id:
                        self.children[parent_id].append(lineage_id)
                        if parent_id in self.nodes:
                            self.nodes[parent_id].descendant_ids.append(lineage_id)
                    else:
                        self.roots.add(lineage_id)
                    
                    new_count += 1
                
                # Update metrics
                node = self.nodes[lineage_id]
                node.event_count += 1
                
                # Extract success/failure from event data
                data = event.get("data", {})
                if data.get("success") or data.get("status") == "success":
                    node.success_count += 1
                    node.reward_sum += data.get("reward", 1.0)
                elif data.get("failure") or data.get("status") in ("failure", "error"):
                    node.failure_count += 1
                    node.reward_sum += data.get("reward", -0.5)
                
            except (json.JSONDecodeError, TypeError):
                continue
        
        self._save_state()
        return new_count
    
    def compute_raw_cmp(self, lineage_id: str) -> float:
        """
        Compute raw CMP for a lineage (before descendant aggregation).
        
        raw_cmp = success_rate * (1 + log(1 + reward_sum))
        """
        if lineage_id not in self.nodes:
            return 0.0
        
        node = self.nodes[lineage_id]
        total = node.success_count + node.failure_count
        
        if total == 0:
            return 0.5  # Neutral prior for unobserved lineages
        
        success_rate = node.success_count / total
        import math
        reward_factor = 1 + math.log(1 + max(0, node.reward_sum))
        
        return success_rate * reward_factor
    
    def compute_cmp(self, lineage_id: str, depth: int = 0, max_depth: int = 10) -> float:
        """
        Compute CMP with descendant aggregation.
        
        CMP(lineage) = raw_cmp + GAMMA * mean(CMP(descendants))
        
        Args:
            lineage_id: The lineage to compute CMP for
            depth: Current recursion depth
            max_depth: Maximum recursion depth
            
        Returns:
            CMP score in [0, ~2] range
        """
        if lineage_id not in self.nodes:
            return 0.0
        
        if depth >= max_depth:
            return self.compute_raw_cmp(lineage_id)
        
        node = self.nodes[lineage_id]
        raw = self.compute_raw_cmp(lineage_id)
        
        # Get descendant CMPs
        descendant_cmps = []
        for child_id in self.children.get(lineage_id, []):
            child_cmp = self.compute_cmp(child_id, depth + 1, max_depth)
            descendant_cmps.append(child_cmp)
        
        # Aggregate with temporal discount
        if descendant_cmps:
            # Spectral smoothing across siblings
            mean_cmp = sum(descendant_cmps) / len(descendant_cmps)
            smoothed = mean_cmp  # Could add spectral smoothing here
            
            computed = raw + GAMMA * smoothed
        else:
            computed = raw
        
        node.raw_cmp = raw
        node.computed_cmp = computed
        
        return computed
    
    def compute_all_cmp(self) -> Dict[str, float]:
        """Compute CMP for all lineages, starting from roots."""
        self._load_state()
        
        # Compute from roots down
        results = {}
        for root_id in self.roots:
            self._compute_subtree_cmp(root_id, results)
        
        self._save_state()
        return results
    
    def _compute_subtree_cmp(self, lineage_id: str, results: Dict[str, float]):
        """Recursively compute CMP for a subtree."""
        cmp = self.compute_cmp(lineage_id)
        results[lineage_id] = cmp
        
        for child_id in self.children.get(lineage_id, []):
            self._compute_subtree_cmp(child_id, results)
    
    def get_lineage(self, lineage_id: str) -> Optional[LineageNode]:
        """Get a lineage node by ID."""
        self._load_state()
        return self.nodes.get(lineage_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the lineage tree."""
        self._load_state()
        
        if not self.nodes:
            return {"total": 0, "roots": 0, "max_depth": 0, "avg_cmp": 0.0}
        
        # Compute max depth
        def get_depth(lid: str, visited: Set[str]) -> int:
            if lid in visited:
                return 0
            visited.add(lid)
            children = self.children.get(lid, [])
            if not children:
                return 1
            return 1 + max(get_depth(c, visited) for c in children)
        
        max_depth = max(get_depth(r, set()) for r in self.roots) if self.roots else 0
        
        # Compute CMPs
        cmp_values = [n.computed_cmp for n in self.nodes.values() if n.computed_cmp > 0]
        avg_cmp = sum(cmp_values) / len(cmp_values) if cmp_values else 0.0
        
        return {
            "total": len(self.nodes),
            "roots": len(self.roots),
            "max_depth": max_depth,
            "avg_cmp": round(avg_cmp, 4),
            "top_5_cmp": sorted(
                [(n.lineage_id[:12], round(n.computed_cmp, 3)) for n in self.nodes.values()],
                key=lambda x: x[1], reverse=True
            )[:5],
        }


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_build(args):
    """Build lineage tree from bus events."""
    tracker = LineageTracker()
    new_count = tracker.build_from_bus()
    print(f"Built lineage tree: {new_count} new lineages discovered")
    print(f"Total lineages: {len(tracker.nodes)}")
    print(f"Root lineages: {len(tracker.roots)}")
    return 0


def cmd_show(args):
    """Show a lineage subtree."""
    tracker = LineageTracker()
    tracker._load_state()
    
    node = tracker.get_lineage(args.lineage_id)
    if not node:
        print(f"Lineage not found: {args.lineage_id}")
        return 1
    
    print(f"Lineage: {node.lineage_id}")
    print(f"  Parent: {node.parent_id or '(root)'}")
    print(f"  Actor: {node.actor}")
    print(f"  Mutation: {node.mutation_op or '(none)'}")
    print(f"  Events: {node.event_count}")
    print(f"  Success/Fail: {node.success_count}/{node.failure_count}")
    print(f"  Raw CMP: {node.raw_cmp:.4f}")
    print(f"  Computed CMP: {node.computed_cmp:.4f}")
    print(f"  Descendants: {len(tracker.children.get(node.lineage_id, []))}")
    
    return 0


def cmd_cmp(args):
    """Compute CMP for a lineage."""
    tracker = LineageTracker()
    tracker.build_from_bus()
    
    cmp = tracker.compute_cmp(args.lineage_id)
    print(f"CMP({args.lineage_id[:12]}...) = {cmp:.4f}")
    
    return 0


def cmd_stats(args):
    """Show lineage statistics."""
    tracker = LineageTracker()
    tracker.build_from_bus()
    tracker.compute_all_cmp()
    
    stats = tracker.get_stats()
    print(f"Lineage Statistics")
    print(f"  Total lineages: {stats['total']}")
    print(f"  Root lineages: {stats['roots']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Avg CMP: {stats['avg_cmp']}")
    print(f"  Top 5 by CMP:")
    for lid, cmp in stats.get('top_5_cmp', []):
        print(f"    {lid}: {cmp}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Lineage Tracker for CMP Computation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # build
    subparsers.add_parser("build", help="Build lineage tree from bus events")
    
    # show
    p_show = subparsers.add_parser("show", help="Show lineage subtree")
    p_show.add_argument("lineage_id", help="Lineage ID")
    
    # cmp
    p_cmp = subparsers.add_parser("cmp", help="Compute CMP for lineage")
    p_cmp.add_argument("lineage_id", help="Lineage ID")
    
    # stats
    subparsers.add_parser("stats", help="Show lineage statistics")
    
    args = parser.parse_args()
    
    if args.command == "build":
        return cmd_build(args)
    elif args.command == "show":
        return cmd_show(args)
    elif args.command == "cmp":
        return cmd_cmp(args)
    elif args.command == "stats":
        return cmd_stats(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
