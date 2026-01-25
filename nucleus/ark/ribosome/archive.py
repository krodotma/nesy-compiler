#!/usr/bin/env python3
"""
archive.py - DGM-style CladeArchive (Gene Pool)

Implements the Darwin Gödel Machine archive mechanism:
- Tree structure of ALL past agent/clade versions
- Never discards ("stepping stones" preserved)
- Diversity-weighted parent selection
- Descendant CMP aggregation (HGM-style lineage fitness)

References:
- Sakana AI Darwin Gödel Machine (2024)
- Huxley-Gödel Machine CMP metric
- Schmidhuber Gödel Machine (2006)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
import random
import math
import json
from pathlib import Path

from .clade import Clade
from ..core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CladeArchive:
    """
    DGM-style gene pool for agent lineages.
    
    Key DGM principles:
    1. NEVER delete clades (stepping stones preserved)
    2. Tree structure tracks full evolutionary history
    3. Diversity-weighted selection prevents premature convergence
    4. Descendant CMP aggregation for lineage-level fitness
    """
    
    # All clades ever created (never pruned)
    nodes: Dict[str, Clade] = field(default_factory=dict)
    
    # Parent -> Children mapping
    children: Dict[str, List[str]] = field(default_factory=dict)
    
    # Child count per clade (for diversity weighting)
    child_counts: Dict[str, int] = field(default_factory=dict)
    
    # Root clades (no parent)
    roots: Set[str] = field(default_factory=set)
    
    # Stepping stones: low self-CMP but high descendant-CMP
    stepping_stones: Set[str] = field(default_factory=set)
    
    # Cached lineage CMP (recomputed on changes)
    _lineage_cmp_cache: Dict[str, float] = field(default_factory=dict)
    _cache_valid: bool = False
    
    # Archive metadata
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0.0"
    
    def add_clade(
        self, 
        clade: Clade, 
        parent_id: Optional[str] = None
    ) -> None:
        """
        Add a clade to the archive. NEVER removes existing clades.
        
        Args:
            clade: Clade to add
            parent_id: Optional parent clade ID
        """
        clade_id = clade.name
        
        # Always add, never overwrite (immutable history)
        if clade_id in self.nodes:
            logger.warning("Clade %s already exists, skipping", clade_id)
            return
        
        self.nodes[clade_id] = clade
        self.child_counts[clade_id] = 0
        
        if parent_id:
            # Add to parent's children
            if parent_id not in self.children:
                self.children[parent_id] = []
            self.children[parent_id].append(clade_id)
            
            # Increment parent's child count
            if parent_id in self.child_counts:
                self.child_counts[parent_id] += 1
            
            # Set clade's parent reference
            clade.parent = parent_id
        else:
            # Root clade
            self.roots.add(clade_id)
        
        # Invalidate cache
        self._cache_valid = False
        
        logger.info("Added clade %s (parent=%s)", clade_id, parent_id)
    
    def get_children(self, clade_id: str) -> List[str]:
        """Get direct children of a clade."""
        return self.children.get(clade_id, [])
    
    def get_descendants(self, clade_id: str) -> Set[str]:
        """Get all descendants (recursive)."""
        result = set()
        stack = [clade_id]
        
        while stack:
            current = stack.pop()
            for child in self.get_children(current):
                if child not in result:
                    result.add(child)
                    stack.append(child)
        
        return result
    
    def get_ancestors(self, clade_id: str) -> List[str]:
        """Get ancestry chain from clade to root."""
        result = []
        current = clade_id
        
        while current in self.nodes:
            parent = self.nodes[current].parent
            if parent:
                result.append(parent)
                current = parent
            else:
                break
        
        return result
    
    def compute_lineage_cmp(
        self, 
        clade_id: str,
        discount: float = 0.9
    ) -> float:
        """
        HGM-style CMP: aggregate descendants with temporal discount.
        
        CMP_lineage(n) = cmp(n) + γ * mean(CMP_lineage(children))
        
        This measures "productive lineage" not just self-performance.
        """
        if self._cache_valid and clade_id in self._lineage_cmp_cache:
            return self._lineage_cmp_cache[clade_id]
        
        if clade_id not in self.nodes:
            return 0.0
        
        clade = self.nodes[clade_id]
        base_cmp = clade.cmp
        
        children = self.get_children(clade_id)
        if not children:
            self._lineage_cmp_cache[clade_id] = base_cmp
            return base_cmp
        
        # Recursive aggregation with discount
        descendant_scores = [
            self.compute_lineage_cmp(child, discount) 
            for child in children
        ]
        
        avg_descendant = sum(descendant_scores) / len(descendant_scores)
        lineage_cmp = base_cmp + discount * avg_descendant
        
        self._lineage_cmp_cache[clade_id] = lineage_cmp
        return lineage_cmp
    
    def recompute_all_lineage_cmp(self) -> None:
        """Recompute lineage CMP for all clades."""
        self._lineage_cmp_cache.clear()
        
        # Start from leaves and work up for efficiency
        for clade_id in self.nodes:
            self.compute_lineage_cmp(clade_id)
        
        self._cache_valid = True
        logger.debug("Recomputed lineage CMP for %d clades", len(self.nodes))
    
    def detect_stepping_stones(self, threshold: float = 2.0) -> Set[str]:
        """
        Detect stepping stones: clades with low self-CMP but high descendant impact.
        
        A clade is a stepping stone if:
        lineage_cmp / self_cmp > threshold
        
        These are preserved even though they look "bad" individually.
        """
        self.recompute_all_lineage_cmp()
        
        stepping_stones = set()
        for clade_id, clade in self.nodes.items():
            if clade.cmp < 0.01:  # Avoid division by zero
                continue
            
            lineage_cmp = self._lineage_cmp_cache.get(clade_id, clade.cmp)
            ratio = lineage_cmp / clade.cmp
            
            if ratio > threshold:
                stepping_stones.add(clade_id)
                logger.info(
                    "Detected stepping stone: %s (self=%.3f, lineage=%.3f, ratio=%.2f)",
                    clade_id, clade.cmp, lineage_cmp, ratio
                )
        
        self.stepping_stones = stepping_stones
        return stepping_stones
    
    def select_parent(
        self, 
        temperature: float = 1.0,
        diversity_weight: float = 0.3,
        performance_weight: float = 0.7
    ) -> Optional[Clade]:
        """
        DGM-style parent selection: blend of performance + diversity.
        
        Args:
            temperature: Softmax temperature (higher = more exploration)
            diversity_weight: Weight for diversity bonus (inverse child count)
            performance_weight: Weight for CMP-based selection
            
        Returns:
            Selected parent clade, or None if archive empty
        """
        if not self.nodes:
            return None
        
        # Ensure lineage CMP is computed
        self.recompute_all_lineage_cmp()
        
        weights = []
        clade_list = list(self.nodes.values())
        
        for clade in clade_list:
            clade_id = clade.name
            
            # Performance component (lineage CMP)
            lineage_cmp = self._lineage_cmp_cache.get(clade_id, clade.cmp)
            perf_score = lineage_cmp ** temperature
            
            # Diversity component (inverse child count)
            # Clades with fewer children get bonus to encourage exploration
            num_children = self.child_counts.get(clade_id, 0)
            diversity_score = 1.0 / (1.0 + num_children)
            
            # Thompson sampling component
            thompson_score = clade.sample_fitness()
            
            # Combined weight
            weight = (
                performance_weight * perf_score +
                diversity_weight * diversity_score
            ) * thompson_score
            
            weights.append(max(weight, 1e-10))  # Prevent zero weights
        
        # Softmax normalization
        total = sum(weights)
        probs = [w / total for w in weights]
        
        # Sample
        selected = random.choices(clade_list, weights=probs, k=1)[0]
        
        logger.debug(
            "Selected parent %s (children=%d, lineage_cmp=%.3f)",
            selected.name,
            self.child_counts.get(selected.name, 0),
            self._lineage_cmp_cache.get(selected.name, 0)
        )
        
        return selected
    
    def get_archive_stats(self) -> Dict[str, Any]:
        """Get archive statistics."""
        self.recompute_all_lineage_cmp()
        
        if not self.nodes:
            return {"total_clades": 0}
        
        cmps = [c.cmp for c in self.nodes.values()]
        lineage_cmps = list(self._lineage_cmp_cache.values())
        
        return {
            "total_clades": len(self.nodes),
            "root_clades": len(self.roots),
            "stepping_stones": len(self.stepping_stones),
            "avg_cmp": sum(cmps) / len(cmps),
            "max_cmp": max(cmps),
            "avg_lineage_cmp": sum(lineage_cmps) / len(lineage_cmps) if lineage_cmps else 0,
            "max_lineage_cmp": max(lineage_cmps) if lineage_cmps else 0,
            "avg_child_count": sum(self.child_counts.values()) / len(self.child_counts) if self.child_counts else 0,
            "max_depth": self._compute_max_depth(),
        }
    
    def _compute_max_depth(self) -> int:
        """Compute maximum tree depth."""
        max_depth = 0
        for clade_id in self.nodes:
            depth = len(self.get_ancestors(clade_id))
            max_depth = max(max_depth, depth)
        return max_depth
    
    def to_dict(self) -> Dict:
        """Serialize archive to dictionary."""
        return {
            "version": self.version,
            "created": self.created,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "children": self.children,
            "child_counts": self.child_counts,
            "roots": list(self.roots),
            "stepping_stones": list(self.stepping_stones),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "CladeArchive":
        """Deserialize archive from dictionary."""
        archive = cls(
            version=data.get("version", "1.0.0"),
            created=data.get("created", datetime.utcnow().isoformat()),
            children=data.get("children", {}),
            child_counts=data.get("child_counts", {}),
            roots=set(data.get("roots", [])),
            stepping_stones=set(data.get("stepping_stones", [])),
        )
        
        # Deserialize clades
        for name, clade_data in data.get("nodes", {}).items():
            archive.nodes[name] = Clade.from_dict(clade_data)
        
        return archive
    
    def save(self, path: Path) -> None:
        """Save archive to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved archive to %s", path)
    
    @classmethod
    def load(cls, path: Path) -> "CladeArchive":
        """Load archive from JSON file."""
        with open(path) as f:
            data = json.load(f)
        logger.info("Loaded archive from %s", path)
        return cls.from_dict(data)


def create_child_clade(
    archive: CladeArchive,
    parent_id: str,
    child_name: str,
    mutation_traits: Optional[Dict[str, float]] = None
) -> Clade:
    """
    Helper: Create a child clade from a parent with trait inheritance.
    
    Args:
        archive: Clade archive
        parent_id: Parent clade ID
        child_name: Name for new child clade
        mutation_traits: Optional trait mutations to apply
        
    Returns:
        New child clade (already added to archive)
    """
    parent = archive.nodes.get(parent_id)
    if not parent:
        raise ValueError(f"Parent clade {parent_id} not found")
    
    # Create child with inherited traits
    child = Clade(
        name=child_name,
        parent=parent_id,
        cmp=parent.cmp * 0.9,  # Start slightly lower
        initial_entropy=parent.initial_entropy,
    )
    
    # Inherit traits with variation
    child.inherit_traits(parent)
    
    # Apply mutations
    if mutation_traits:
        for trait, value in mutation_traits.items():
            child.add_trait(trait, value)
    
    # Add to archive
    archive.add_clade(child, parent_id)
    
    return child
