#!/usr/bin/env python3
"""
distill.py - DistillationPipeline: Full negentropic transformation

The complete distillation pipeline:
1. IngestPipeline for initial filtering
2. Rhizom integration for lineage tracking
3. CMP scoring for fitness evaluation
4. Crystallization for ISO-layer promotion
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from nucleus.ark.portal.ingest import IngestPipeline, IngestReport
from nucleus.ark.rhizom.dag import RhizomDAG, RhizomNode
from nucleus.ark.rhizom.etymology import EtymologyExtractor
from nucleus.ark.ribosome.gene import Gene

logger = logging.getLogger("ARK.Portal.Distill")


@dataclass
class DistillationReport:
    """Report from a distillation run."""
    source: str
    target: str
    started: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed: Optional[str] = None
    ingest_report: Optional[IngestReport] = None
    genes_created: int = 0
    total_cmp: float = 0.0
    entropy_reduction: float = 0.0
    status: str = "pending"


class DistillationPipeline:
    """
    Full distillation pipeline: Entropic → Negentropic.
    
    Combines:
    - IngestPipeline for filtering
    - RhizomDAG for lineage
    - Gene creation for tracking
    - CMP scoring for fitness
    """
    
    def __init__(self, source_root: str, target_root: str):
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        self.ingest = IngestPipeline(source_root, target_root)
        
        # Initialize Rhizom at target
        self.rhizom = RhizomDAG(self.target_root)
        
        # Gene registry
        self.genes: Dict[str, Gene] = {}
        
        # Etymology extractor
        self.etymology_extractor = EtymologyExtractor()
    
    def run(self, purpose: str = "Negentropic distillation") -> DistillationReport:
        """
        Run the complete distillation pipeline.
        
        Args:
            purpose: High-level purpose for the distillation
        
        Returns:
            DistillationReport with complete metrics
        """
        report = DistillationReport(
            source=str(self.source_root),
            target=str(self.target_root)
        )
        
        logger.info(f"🔬 Starting distillation: {self.source_root} → {self.target_root}")
        
        # Phase 1: Ingestion
        try:
            ingest_report = self.ingest.run(purpose=purpose)
            report.ingest_report = ingest_report
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            report.status = "failed"
            return report
        
        # Phase 2: Gene creation
        for result in ingest_report.results:
            if not result.accepted or not result.target_path:
                continue
            
            try:
                content = Path(result.target_path).read_text()
                gene = Gene.from_file(
                    path=result.target_path,
                    content=content,
                    etymology=result.etymology.primary if result.etymology else ""
                )
                
                # Calculate fitness from entropy
                avg_entropy = sum(result.entropy.values()) / len(result.entropy)
                gene.fitness = 1.0 - avg_entropy  # Lower entropy = higher fitness
                
                self.genes[gene.path] = gene
                report.genes_created += 1
                report.total_cmp += gene.fitness
                
            except Exception as e:
                logger.warning(f"Failed to create gene for {result.target_path}: {e}")
        
        # Phase 3: Rhizom node creation (mark this distillation run)
        distillation_node = RhizomNode(
            sha=f"distill-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            etymology=purpose,
            cmp=report.total_cmp / max(report.genes_created, 1),
            entropy=self._aggregate_entropy(ingest_report),
            lineage_tags=["distillation", "portal"]
        )
        self.rhizom.insert(distillation_node)
        
        # Finalize
        report.entropy_reduction = ingest_report.total_entropy_reduction
        report.completed = datetime.utcnow().isoformat()
        report.status = "completed"
        
        logger.info(f"✅ Distillation complete: {report.genes_created} genes, CMP={report.total_cmp:.2f}")
        
        return report
    
    def _aggregate_entropy(self, report: IngestReport) -> Dict[str, float]:
        """Aggregate entropy across all accepted files."""
        if not report.accepted_files:
            return {"h_total": 0.5}
        
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        
        for result in report.results:
            if not result.accepted:
                continue
            for key, value in result.entropy.items():
                totals[key] = totals.get(key, 0) + value
                counts[key] = counts.get(key, 0) + 1
        
        return {k: totals[k] / counts[k] for k in totals}
    
    def get_gene(self, path: str) -> Optional[Gene]:
        """Get a gene by path."""
        return self.genes.get(path)
    
    def list_genes(self, min_fitness: float = 0.0) -> List[Gene]:
        """List genes with fitness above threshold."""
        return [g for g in self.genes.values() if g.fitness >= min_fitness]


class CrystallizationGate:
    """
    Gate for promoting code to ISO (production) layer.
    
    Requirements:
    1. Gene fitness > 0.6
    2. No TODO/FIXME markers
    3. Has docstrings
    4. Test coverage exists (future)
    """
    
    def __init__(self, fitness_threshold: float = 0.6):
        self.fitness_threshold = fitness_threshold
    
    def check(self, gene: Gene, content: str) -> tuple[bool, str]:
        """
        Check if a gene is ready for crystallization.
        
        Returns:
            (passed, reason)
        """
        # Fitness check
        if gene.fitness < self.fitness_threshold:
            return False, f"Fitness too low: {gene.fitness:.2f} < {self.fitness_threshold}"
        
        # TODO/FIXME check
        if 'TODO' in content or 'FIXME' in content:
            return False, "Contains TODO/FIXME markers"
        
        # Docstring check (for Python)
        if gene.path.endswith('.py') and '"""' not in content and "'''" not in content:
            return False, "Missing docstrings"
        
        return True, "Ready for crystallization"
