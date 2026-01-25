#!/usr/bin/env python3
"""
genome.py - OrganismGenome: The complete blueprint for a Pluribus instance

Defines the structure, constitution, and evolution parameters
for an entire ARK-managed repository.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

from nucleus.ark.ribosome.gene import Gene
from nucleus.ark.ribosome.clade import Clade


@dataclass
class Constitution:
    """
    Constitutional constraints for the genome.
    Defines what is allowed/forbidden at the deepest level.
    """
    # DNA Axioms
    axioms: List[str] = field(default_factory=lambda: [
        "Entelecheia: Every change must serve a purpose",
        "Inertia: High-centrality nodes resist change",
        "Witness: Every mutation requires attestation",
        "Hysteresis: Past states influence the present",
        "Infinity: Viable behaviors recur infinitely"
    ])
    
    # Forbidden patterns (AST-level)
    forbidden_patterns: List[str] = field(default_factory=lambda: [
        "AbstractFactoryFactory",
        "GodClass",
        "GlobalMutableState"
    ])
    
    # Required patterns for core modules
    required_patterns: List[str] = field(default_factory=lambda: [
        "Docstring",
        "TypeHints"
    ])
    
    # Maximum allowed entropy
    entropy_threshold: float = 0.7
    
    # Minimum CMP for promotion
    cmp_threshold: float = 0.6


@dataclass
class OrganismGenome:
    """
    The complete blueprint for a Pluribus instance.
    
    Contains:
    - Constitution: fundamental rules
    - Genes: all tracked files
    - Clades: all evolutionary branches
    - Evolution parameters: mutation rates, etc.
    """
    
    # Genome identifier
    name: str = "pluribus"
    version: str = "1.0.0"
    
    # Constitutional constraints
    constitution: Constitution = field(default_factory=Constitution)
    
    # All genes (files) in the genome
    genes: Dict[str, Gene] = field(default_factory=dict)
    
    # All clades (branches)
    clades: Dict[str, Clade] = field(default_factory=dict)
    
    # Evolution parameters
    mutation_rate: float = 0.05
    crossover_rate: float = 0.1
    selection_pressure: float = 0.7
    
    # Current state
    total_fitness: float = 0.5
    generation: int = 0
    
    # Metadata
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_evolved: Optional[str] = None
    
    def register_gene(self, gene: Gene) -> None:
        """Add or update a gene in the genome."""
        self.genes[gene.path] = gene
        self._recalculate_fitness()
    
    def register_clade(self, clade: Clade) -> None:
        """Add or update a clade in the genome."""
        self.clades[clade.name] = clade
    
    def get_gene(self, path: str) -> Optional[Gene]:
        """Get a gene by path."""
        return self.genes.get(path)
    
    def get_clade(self, name: str) -> Optional[Clade]:
        """Get a clade by name."""
        return self.clades.get(name)
    
    def select_clade(self) -> Optional[Clade]:
        """
        Select a clade using Thompson Sampling.
        Returns the clade with highest sampled fitness.
        """
        if not self.clades:
            return None
        
        samples = [(c, c.sample_fitness()) for c in self.clades.values()]
        return max(samples, key=lambda x: x[1])[0]
    
    def _recalculate_fitness(self) -> None:
        """Recalculate total genome fitness from genes."""
        if not self.genes:
            self.total_fitness = 0.5
            return
        
        total = sum(g.fitness for g in self.genes.values())
        self.total_fitness = total / len(self.genes)
    
    def advance_generation(self) -> None:
        """Advance to next generation."""
        self.generation += 1
        self.last_evolved = datetime.utcnow().isoformat()
    
    def entropy_status(self) -> Dict[str, int]:
        """Count genes by fitness level."""
        low = sum(1 for g in self.genes.values() if g.fitness < 0.4)
        mid = sum(1 for g in self.genes.values() if 0.4 <= g.fitness < 0.7)
        high = sum(1 for g in self.genes.values() if g.fitness >= 0.7)
        return {"low": low, "medium": mid, "high": high}
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "constitution": {
                "axioms": self.constitution.axioms,
                "forbidden_patterns": self.constitution.forbidden_patterns,
                "required_patterns": self.constitution.required_patterns,
                "entropy_threshold": self.constitution.entropy_threshold,
                "cmp_threshold": self.constitution.cmp_threshold
            },
            "genes": {k: v.to_dict() for k, v in self.genes.items()},
            "clades": {k: v.to_dict() for k, v in self.clades.items()},
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "selection_pressure": self.selection_pressure,
            "total_fitness": self.total_fitness,
            "generation": self.generation,
            "created": self.created,
            "last_evolved": self.last_evolved
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "OrganismGenome":
        """Deserialize from dictionary."""
        genome = cls(
            name=data.get("name", "pluribus"),
            version=data.get("version", "1.0.0"),
            mutation_rate=data.get("mutation_rate", 0.05),
            crossover_rate=data.get("crossover_rate", 0.1),
            selection_pressure=data.get("selection_pressure", 0.7),
            total_fitness=data.get("total_fitness", 0.5),
            generation=data.get("generation", 0),
            created=data.get("created", datetime.utcnow().isoformat()),
            last_evolved=data.get("last_evolved")
        )
        
        # Load constitution
        if "constitution" in data:
            c = data["constitution"]
            genome.constitution = Constitution(
                axioms=c.get("axioms", []),
                forbidden_patterns=c.get("forbidden_patterns", []),
                required_patterns=c.get("required_patterns", []),
                entropy_threshold=c.get("entropy_threshold", 0.7),
                cmp_threshold=c.get("cmp_threshold", 0.6)
            )
        
        # Load genes
        for path, gdata in data.get("genes", {}).items():
            genome.genes[path] = Gene.from_dict(gdata)
        
        # Load clades
        for name, cdata in data.get("clades", {}).items():
            genome.clades[name] = Clade.from_dict(cdata)
        
        return genome
