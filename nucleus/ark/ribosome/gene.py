#!/usr/bin/env python3
"""
gene.py - Gene: The fundamental unit of ARK lineage

A Gene represents a discrete unit of functionality (file, module, service)
with semantic etymology and fitness tracking.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
import hashlib


@dataclass
class Gene:
    """
    A fundamental unit of functionality in the ARK genome.
    
    Corresponds to a file or module with:
    - Etymology: semantic origin/purpose
    - Fitness: CMP-derived score
    - Lineage: mutation history
    """
    
    # Unique identifier (content hash)
    oid: str = ""
    
    # File path relative to repo root
    path: str = ""
    
    # Semantic origin/purpose
    etymology: str = ""
    
    # Fitness score (0.0 - 1.0)
    fitness: float = 0.5
    
    # Complexity metrics
    complexity: float = 0.0
    lines: int = 0
    
    # Dependencies
    imports: List[str] = field(default_factory=list)
    imported_by: List[str] = field(default_factory=list)
    
    # Mutation tracking
    mutations: int = 0
    last_mutated: Optional[str] = None
    
    # Metadata
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    @classmethod
    def from_file(cls, path: str, content: str, etymology: str = "") -> "Gene":
        """Create a Gene from file content."""
        oid = hashlib.sha256(content.encode()).hexdigest()[:16]
        lines = len(content.split("\n"))
        
        # Extract imports (Python)
        imports = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                imports.append(line)
        
        return cls(
            oid=oid,
            path=path,
            etymology=etymology or f"Gene from {path}",
            lines=lines,
            imports=imports
        )
    
    def calculate_inertia(self) -> float:
        """
        Calculate inertia score based on:
        - Number of dependents (imported_by)
        - Mutation frequency (fewer = higher inertia)
        - Complexity (higher = higher inertia)
        """
        dependent_score = min(len(self.imported_by) / 10.0, 1.0)
        stability_score = 1.0 / (1.0 + self.mutations * 0.1)
        complexity_score = min(self.complexity / 20.0, 1.0)
        
        return (dependent_score * 0.4 + stability_score * 0.4 + complexity_score * 0.2)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "oid": self.oid,
            "path": self.path,
            "etymology": self.etymology,
            "fitness": self.fitness,
            "complexity": self.complexity,
            "lines": self.lines,
            "imports": self.imports,
            "imported_by": self.imported_by,
            "mutations": self.mutations,
            "last_mutated": self.last_mutated,
            "created": self.created
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Gene":
        """Deserialize from dictionary."""
        return cls(**data)
