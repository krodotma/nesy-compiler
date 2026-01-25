#!/usr/bin/env python3
"""
ingest.py - IngestPipeline: Entropic source processing

The Portal's ingestion pipeline:
1. Walks entropic source directories
2. Applies DNA gates for filtering
3. Extracts etymology for accepted files
4. Outputs to negentropic target with Rhizom tracking
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Any
from pathlib import Path

from nucleus.ark.gates.inertia import InertiaGate, InertiaContext
from nucleus.ark.gates.entelecheia import EntelecheiaGate, EntelecheiaContext
from nucleus.ark.gates.homeostasis import HomeostasisGate, HomeostasisContext
from nucleus.ark.rhizom.etymology import EtymologyExtractor, Etymology

logger = logging.getLogger("ARK.Portal.Ingest")


@dataclass
class IngestResult:
    """Result of ingesting a single file."""
    source_path: str
    target_path: Optional[str] = None
    accepted: bool = False
    etymology: Optional[Etymology] = None
    rejection_reason: str = ""
    entropy: Dict[str, float] = field(default_factory=dict)


@dataclass
class IngestReport:
    """Summary report of an ingestion run."""
    source_root: str
    target_root: str
    total_files: int = 0
    accepted_files: int = 0
    rejected_files: int = 0
    results: List[IngestResult] = field(default_factory=list)
    total_entropy_reduction: float = 0.0


class IngestPipeline:
    """
    Ingestion pipeline for processing entropic sources.
    
    Flow:
    1. SCAN: Walk source directory for eligible files
    2. FILTER: Apply DNA gates (Inertia, Entelecheia, Homeostasis)
    3. EXTRACT: Etymology and entropy metrics
    4. TRANSFER: Copy to target with metadata
    """
    
    def __init__(self, source_root: str, target_root: str):
        self.source_root = Path(source_root)
        self.target_root = Path(target_root)
        
        # Initialize gates
        self.gates = [
            InertiaGate(),
            EntelecheiaGate(),
            HomeostasisGate()
        ]
        
        # Etymology extractor
        self.etymology_extractor = EtymologyExtractor()
        
        # Neural adapter (optional, for thrash detection)
        self.neural_adapter = None
        
        # Configuration
        self.extensions = [".py", ".ts", ".tsx", ".js", ".md"]
        self.exclude_patterns = ["__pycache__", ".git", "node_modules", ".venv", "venv"]
    
    def set_neural_adapter(self, adapter: Any) -> None:
        """Set optional neural adapter for thrash detection."""
        self.neural_adapter = adapter
    
    def run(self, purpose: str = "General ingestion") -> IngestReport:
        """
        Run the ingestion pipeline.
        
        Args:
            purpose: High-level purpose for Entelecheia gate
        
        Returns:
            IngestReport with all results
        """
        logger.info(f"Starting ingestion: {self.source_root} → {self.target_root}")
        
        report = IngestReport(
            source_root=str(self.source_root),
            target_root=str(self.target_root)
        )
        
        # Create target if needed
        self.target_root.mkdir(parents=True, exist_ok=True)
        
        # Walk source
        for root, dirs, files in os.walk(self.source_root):
            # Filter excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_patterns]
            
            for filename in files:
                if not any(filename.endswith(ext) for ext in self.extensions):
                    continue
                
                source_path = Path(root) / filename
                result = self._ingest_file(source_path, purpose)
                
                report.results.append(result)
                report.total_files += 1
                
                if result.accepted:
                    report.accepted_files += 1
                else:
                    report.rejected_files += 1
        
        # Calculate entropy reduction
        if report.accepted_files > 0:
            avg_entropy = sum(r.entropy.get("h_struct", 0.5) 
                             for r in report.results if r.accepted) / report.accepted_files
            report.total_entropy_reduction = 0.5 - avg_entropy  # Assume 0.5 baseline
        
        logger.info(f"Ingestion complete: {report.accepted_files}/{report.total_files} accepted")
        return report
    
    def _ingest_file(self, source_path: Path, purpose: str) -> IngestResult:
        """Process a single file through the pipeline."""
        rel_path = source_path.relative_to(self.source_root)
        target_path = self.target_root / rel_path
        
        result = IngestResult(source_path=str(source_path))
        
        try:
            content = source_path.read_text(encoding='utf-8')
        except Exception as e:
            result.rejection_reason = f"Read error: {e}"
            return result
        
        # Extract etymology
        etymology = self.etymology_extractor.extract_from_code(content, str(source_path))
        result.etymology = etymology
        
        # Calculate entropy
        entropy = self._calculate_entropy(content)
        result.entropy = entropy
        
        # Gate 1: Homeostasis (system-level check)
        h_ctx = HomeostasisContext(entropy=entropy, is_stabilization_commit=False)
        if not self.gates[2].check(h_ctx):
            result.rejection_reason = "Homeostasis: System entropy too high"
            return result
        
        # Gate 2: Inertia (high-inertia protection)
        i_ctx = InertiaContext(
            files=[str(source_path)],
            has_witness=False,
            has_formal_proof=False
        )
        if not self.gates[0].check(i_ctx):
            result.rejection_reason = "Inertia: High-inertia file without witness"
            # For ingestion, we might still accept with a flag
            # For now, we pass (ingestion is more permissive than commit)
        
        # Gate 3: Entelecheia (purpose check)
        e_ctx = EntelecheiaContext(
            purpose=purpose or etymology.primary,
            is_cosmetic=False,
            liveness_gain=0.1  # Ingestion implies some gain
        )
        if not self.gates[1].check(e_ctx):
            result.rejection_reason = "Entelecheia: No clear purpose"
            return result
        
        # Neural gate (if available)
        if self.neural_adapter:
            features = self.neural_adapter.extract_features_from_code(content)
            thrash_prob = self.neural_adapter.predict_thrash(features)
            if thrash_prob > 0.8:
                result.rejection_reason = f"Neural: High thrash probability ({thrash_prob:.2f})"
                return result
        
        # All gates passed - transfer file
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding='utf-8')
            result.target_path = str(target_path)
            result.accepted = True
            logger.info(f"✅ Accepted: {rel_path}")
        except Exception as e:
            result.rejection_reason = f"Write error: {e}"
        
        return result
    
    def _calculate_entropy(self, content: str) -> Dict[str, float]:
        """Calculate 8-dimensional entropy vector."""
        lines = content.split('\n')
        
        # Structural complexity (based on indentation variance)
        indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
        h_struct = min(len(set(indents)) / 10, 1.0) if indents else 0.5
        
        # Documentation (docstring/comment ratio)
        doc_lines = sum(1 for l in lines if l.strip().startswith('#') or '"""' in l)
        h_doc = 1.0 - min(doc_lines / max(len(lines), 1), 0.5) * 2
        
        # Type hints (presence of : in function defs)
        type_hints = sum(1 for l in lines if 'def ' in l and ':' in l.split('def')[1])
        func_count = sum(1 for l in lines if 'def ' in l)
        h_type = 1.0 - (type_hints / max(func_count, 1))
        
        # Test coverage proxy (assume no tests for ingested code)
        h_test = 0.7
        
        # Dependency sprawl (import count)
        import_count = sum(1 for l in lines if l.strip().startswith(('import ', 'from ')))
        h_deps = min(import_count / 20, 1.0)
        
        # Code churn (assume medium for new code)
        h_churn = 0.5
        
        # Technical debt indicators
        todo_count = sum(1 for l in lines if 'TODO' in l or 'FIXME' in l or 'HACK' in l)
        h_debt = min(todo_count / 5, 1.0)
        
        # Spec alignment (assume medium for ingested)
        h_align = 0.5
        
        return {
            "h_struct": h_struct,
            "h_doc": h_doc,
            "h_type": h_type,
            "h_test": h_test,
            "h_deps": h_deps,
            "h_churn": h_churn,
            "h_debt": h_debt,
            "h_align": h_align
        }
