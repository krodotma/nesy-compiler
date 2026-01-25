#!/usr/bin/env python3
"""
lineage.py - LineageTracker: Ancestry and provenance tracking

Tracks evolutionary lineage, computes ancestry paths,
and enables CMP-based clade selection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import random

from nucleus.ark.rhizom.dag import RhizomDAG, RhizomNode
from nucleus.ark.ribosome.clade import Clade


@dataclass
class LineageInfo:
    """Information about a node's lineage."""
    sha: str
    depth: int               # Distance from root
    ancestors: List[str]     # Ancestor SHAs
    clades: List[str]        # Clade memberships
    cmp_trajectory: List[float]  # CMP scores over ancestry


class LineageTracker:
    """
    Tracks evolutionary lineage and ancestry.
    
    Features:
    - Ancestry path computation
    - CMP trajectory analysis
    - Thompson Sampling for clade selection
    - HGT (Horizontal Gene Transfer) detection
    """
    
    def __init__(self, rhizom: RhizomDAG):
        self.rhizom = rhizom
        self.clades: Dict[str, Clade] = {}
    
    def register_clade(self, clade: Clade) -> None:
        """Register a clade for tracking."""
        self.clades[clade.name] = clade
    
    def get_lineage(self, sha: str, max_depth: int = 50) -> LineageInfo:
        """Get complete lineage information for a commit."""
        ancestors = []
        cmp_trajectory = []
        clades_found = set()
        
        current_sha = sha
        for depth in range(max_depth):
            node = self.rhizom.get(current_sha)
            if not node:
                break
            
            if depth > 0:  # Don't include self
                ancestors.append(current_sha)
            
            cmp_trajectory.append(node.cmp)
            clades_found.update(node.lineage_tags)
            
            if not node.parents:
                break
            current_sha = node.parents[0]
        
        return LineageInfo(
            sha=sha,
            depth=len(ancestors),
            ancestors=ancestors,
            clades=list(clades_found),
            cmp_trajectory=cmp_trajectory
        )
    
    def find_common_ancestor(self, sha1: str, sha2: str) -> Optional[str]:
        """Find the most recent common ancestor of two commits."""
        lineage1 = self.get_lineage(sha1)
        lineage2 = self.get_lineage(sha2)
        
        ancestors1 = set([sha1] + lineage1.ancestors)
        
        # Walk sha2's ancestry until we find overlap
        for ancestor in [sha2] + lineage2.ancestors:
            if ancestor in ancestors1:
                return ancestor
        
        return None
    
    def cmp_delta(self, sha: str, depth: int = 5) -> float:
        """Calculate CMP trend over recent ancestry."""
        lineage = self.get_lineage(sha, max_depth=depth)
        
        if len(lineage.cmp_trajectory) < 2:
            return 0.0
        
        # Compute weighted average delta (more recent = higher weight)
        total_delta = 0.0
        total_weight = 0.0
        
        for i in range(1, len(lineage.cmp_trajectory)):
            delta = lineage.cmp_trajectory[i-1] - lineage.cmp_trajectory[i]
            weight = 1.0 / i  # More recent = higher weight
            total_delta += delta * weight
            total_weight += weight
        
        return total_delta / total_weight if total_weight > 0 else 0.0
    
    def select_clade(self, candidates: Optional[List[str]] = None) -> Optional[Clade]:
        """
        Select a clade using Thompson Sampling.
        
        Each clade has a Beta(α, β) prior updated by merge outcomes.
        """
        clades_to_consider = candidates or list(self.clades.keys())
        
        if not clades_to_consider:
            return None
        
        # Sample from each clade's Beta distribution
        samples = []
        for name in clades_to_consider:
            clade = self.clades.get(name)
            if clade:
                fitness = clade.sample_fitness()
                samples.append((clade, fitness))
        
        if not samples:
            return None
        
        # Return clade with highest sampled fitness
        return max(samples, key=lambda x: x[1])[0]
    
    def detect_hgt(self, sha: str) -> List[Tuple[str, str]]:
        """
        Detect Horizontal Gene Transfer events.
        
        HGT occurs when a commit has multiple parents from different clades.
        Returns list of (source_clade, target_clade) transfers.
        """
        node = self.rhizom.get(sha)
        if not node or len(node.parents) < 2:
            return []
        
        # Get clades of each parent
        parent_clades = []
        for parent_sha in node.parents:
            parent_node = self.rhizom.get(parent_sha)
            if parent_node:
                parent_clades.append((parent_sha, set(parent_node.lineage_tags)))
        
        # Find cross-clade transfers
        transfers = []
        for i, (sha1, clades1) in enumerate(parent_clades):
            for sha2, clades2 in parent_clades[i+1:]:
                if clades1 and clades2 and not clades1.intersection(clades2):
                    # Different clades - HGT detected
                    for c1 in clades1:
                        for c2 in clades2:
                            transfers.append((c1, c2))
        
        return transfers
    
    def prune_dead_ends(self) -> List[str]:
        """
        Identify dead-end lineages (no descendants, low CMP).
        
        These are candidates for eventual pruning/compaction.
        """
        dead_ends = []
        all_parents = set()
        
        # Collect all parent references
        for node in self.rhizom:
            all_parents.update(node.parents)
        
        # Find leaves (no children) with low CMP
        for node in self.rhizom:
            if node.sha not in all_parents and node.cmp < 0.3:
                dead_ends.append(node.sha)
        
        return dead_ends
    
    def entropy_trajectory(self, sha: str, dimension: str = "h_struct") -> List[float]:
        """Get entropy trajectory for a specific dimension."""
        lineage = self.get_lineage(sha)
        trajectory = []
        
        for ancestor_sha in [sha] + lineage.ancestors:
            node = self.rhizom.get(ancestor_sha)
            if node and dimension in node.entropy:
                trajectory.append(node.entropy[dimension])
        
        return trajectory
