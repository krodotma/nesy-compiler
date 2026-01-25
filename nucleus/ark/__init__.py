# ARK: Autonomous Reactive Kernel
# The negentropic evolutionary version control system for Pluribus

"""
ARK unifies all Pluribus holons (Rhizom, Portal, Ingest, Inception, Ribosome)
into a single evolution-aware VCS built on isomorphic-git patterns.

Core Concepts:
- Negentropic Filtering: Only purposeful, stable code enters lineage
- LTL-Guided Synthesis: Correct-by-construction commits
- DNA Hyperautomata: Triplet gates (Inertia, Entelecheia, Homeostasis)
- Clade/CMP Evolution: Gödelian self-improvement with decidable guards

Package Structure:
- core/: Repository, commit, config primitives
- gates/: DNA triplet gates (Inertia, Entelecheia, Homeostasis)
- rhizom/: Semantic DAG for commit lineage
- ribosome/: Gene, Clade, OrganismGenome schemas
- neural/: Thrash detection and feature extraction
- synthesis/: LTL specs and grammar-guided synthesis
- portal/: Ingestion and distillation pipeline
- commands/: CLI command implementations
"""

__version__ = "0.1.0"
__author__ = "Pluribus Evolution Swarm"

from nucleus.ark.core.repository import ArkRepository
from nucleus.ark.core.context import ArkCommitContext
from nucleus.ark.ribosome.gene import Gene
from nucleus.ark.ribosome.clade import Clade
from nucleus.ark.ribosome.genome import OrganismGenome

__all__ = [
    "ArkRepository",
    "ArkCommitContext", 
    "Gene",
    "Clade",
    "OrganismGenome",
]
