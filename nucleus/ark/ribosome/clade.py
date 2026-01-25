#!/usr/bin/env python3
"""
clade.py - Clade: An evolutionary branch with traits

A Clade represents a branch/lineage with:
- Traits: inherited characteristics
- CMP score: cumulative fitness
- Thompson Sampling: selection statistics
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
import random


@dataclass
class Clade:
    """
    An evolutionary branch in the ARK lineage.
    
    Clades are "branches with DNA" - they carry traits,
    fitness scores, and selection statistics for Thompson Sampling.
    """
    
    # Unique name
    name: str = ""
    
    # Git branch reference
    branch_ref: str = ""
    
    # Parent clade (for ancestry)
    parent: Optional[str] = None
    
    # Cumulative Meta-Priority (fitness)
    cmp: float = 0.5
    
    # Traits inherited or acquired
    traits: Dict[str, float] = field(default_factory=dict)
    
    # Thompson Sampling statistics
    # α (alpha): successful merges
    # β (beta): rejected merges
    alpha: float = 1.0
    beta: float = 1.0
    
    # Entropy at creation
    initial_entropy: float = 0.5
    
    # Commit count in this clade
    commit_count: int = 0
    
    # Child count (DGM diversity tracking)
    child_count: int = 0
    
    # Member genes/SHAs in this clade
    members: List[str] = field(default_factory=list)
    
    # Metadata
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: Optional[str] = None
    
    def sample_fitness(self) -> float:
        """
        Thompson Sampling: Sample from Beta(α, β) distribution.
        Returns expected win rate for clade selection.
        """
        return random.betavariate(self.alpha, self.beta)
    
    def record_success(self) -> None:
        """Record a successful merge/operation."""
        self.alpha += 1
        self.last_updated = datetime.utcnow().isoformat()
    
    def record_failure(self) -> None:
        """Record a failed merge/operation."""
        self.beta += 1
        self.last_updated = datetime.utcnow().isoformat()
    
    def expected_fitness(self) -> float:
        """Expected value of Beta distribution."""
        return self.alpha / (self.alpha + self.beta)
    
    def add_trait(self, trait_name: str, value: float) -> None:
        """Add or update a trait."""
        self.traits[trait_name] = value
    
    def inherit_traits(self, parent_clade: "Clade", inheritance_rate: float = 0.8) -> None:
        """Inherit traits from parent with some variation."""
        for trait, value in parent_clade.traits.items():
            # Inherit with random variation
            variation = (random.random() - 0.5) * 0.2
            self.traits[trait] = max(0, min(1, value * inheritance_rate + variation))
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "branch_ref": self.branch_ref,
            "parent": self.parent,
            "cmp": self.cmp,
            "traits": self.traits,
            "alpha": self.alpha,
            "beta": self.beta,
            "initial_entropy": self.initial_entropy,
            "commit_count": self.commit_count,
            "child_count": self.child_count,
            "members": self.members,
            "created": self.created,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Clade":
        """Deserialize from dictionary."""
        return cls(**data)
