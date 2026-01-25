#!/usr/bin/env python3
"""
proposal_generator.py - Generates refactoring proposals from observer analysis.

Part of the pluribus_evolution refiner subsystem.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RefactoringProposal:
    """A proposed refactoring to the primary trunk."""
    id: str
    proposal_type: str  # extract_function, rename, decompose, consolidate
    priority: int  # 1 (highest) - 5 (lowest)
    target_files: list[str]
    description: str
    rationale: str
    estimated_impact: str  # low, medium, high
    generated_at: str
    clade_compatible: bool = True  # Can be merged via Clade-Weave


@dataclass
class ProposalBatch:
    """Batch of related proposals."""
    batch_id: str
    theme: str  # e.g., "reduce_complexity", "improve_cohesion"
    proposals: list[RefactoringProposal] = field(default_factory=list)
    generated_at: str = ""


class ProposalGenerator:
    """
    Generates refactoring proposals based on observer findings.

    Takes input from:
    - CodeAnalyzer (patterns, antipatterns)
    - DriftDetector (genotype/phenotype drift)
    - VectorProfiler (semantic clustering)

    Produces proposals that can be:
    - Reviewed by agents/humans
    - Synthesized into patches
    - Merged via Clade-Weave
    """

    def __init__(self):
        self.proposals: list[RefactoringProposal] = []

    def from_code_patterns(
        self,
        patterns: list[dict],
        threshold_confidence: float = 0.7
    ) -> list[RefactoringProposal]:
        """Generate proposals from CodeAnalyzer patterns."""
        proposals = []

        for pattern in patterns:
            if pattern.get("confidence", 0) < threshold_confidence:
                continue

            pattern_type = pattern.get("pattern_type", "")

            if pattern_type == "large_function":
                proposals.append(RefactoringProposal(
                    id=uuid.uuid4().hex[:12],
                    proposal_type="decompose",
                    priority=2,
                    target_files=[pattern.get("location", "").split(":")[0]],
                    description=f"Decompose large function: {pattern.get('description', '')}",
                    rationale="Large functions (>50 lines) are harder to test and maintain",
                    estimated_impact="medium",
                    generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                ))

            elif pattern_type == "god_class":
                proposals.append(RefactoringProposal(
                    id=uuid.uuid4().hex[:12],
                    proposal_type="decompose",
                    priority=1,
                    target_files=[pattern.get("location", "").split(":")[0]],
                    description=f"Split god class into focused components",
                    rationale="God classes violate single responsibility principle",
                    estimated_impact="high",
                    generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                ))

            elif pattern_type == "deep_nesting":
                proposals.append(RefactoringProposal(
                    id=uuid.uuid4().hex[:12],
                    proposal_type="extract_function",
                    priority=3,
                    target_files=[pattern.get("location", "").split(":")[0]],
                    description=f"Extract deeply nested logic: {pattern.get('description', '')}",
                    rationale="Deep nesting increases cognitive complexity",
                    estimated_impact="low",
                    generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                ))

        self.proposals.extend(proposals)
        return proposals

    def from_drift_signals(
        self,
        signals: list[dict],
        threshold_severity: float = 0.3
    ) -> list[RefactoringProposal]:
        """Generate proposals from DriftDetector signals."""
        proposals = []

        for signal in signals:
            if signal.get("severity", 0) < threshold_severity:
                continue

            drift_type = signal.get("drift_type", "")

            if drift_type == "schema":
                proposals.append(RefactoringProposal(
                    id=uuid.uuid4().hex[:12],
                    proposal_type="consolidate",
                    priority=2,
                    target_files=[signal.get("source_path", ""), signal.get("target_path", "")],
                    description="Reconcile schema drift between spec and implementation",
                    rationale=signal.get("description", ""),
                    estimated_impact="medium",
                    generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                ))

            elif drift_type == "protocol":
                proposals.append(RefactoringProposal(
                    id=uuid.uuid4().hex[:12],
                    proposal_type="consolidate",
                    priority=1,
                    target_files=[signal.get("target_path", "")],
                    description="Update implementation to match latest protocol version",
                    rationale=signal.get("description", ""),
                    estimated_impact="high",
                    generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                ))

        self.proposals.extend(proposals)
        return proposals

    def from_vector_outliers(
        self,
        outliers: list[dict]
    ) -> list[RefactoringProposal]:
        """Generate proposals from VectorProfiler outliers."""
        proposals = []

        for outlier in outliers:
            proposals.append(RefactoringProposal(
                id=uuid.uuid4().hex[:12],
                proposal_type="relocate",
                priority=4,
                target_files=[outlier.get("path", "")],
                description=f"Consider relocating outlier file with unusual semantics",
                rationale=f"File has unusual tag combination: {outlier.get('semantic_tags', [])}",
                estimated_impact="low",
                generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ))

        self.proposals.extend(proposals)
        return proposals

    def create_batch(self, theme: str) -> ProposalBatch:
        """Create a batch of proposals with a common theme."""
        return ProposalBatch(
            batch_id=uuid.uuid4().hex[:12],
            theme=theme,
            proposals=self.proposals.copy(),
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        )

    def to_bus_event(self, batch: ProposalBatch) -> dict:
        """Convert proposal batch to bus event."""
        return {
            "topic": "evolution.refiner.proposal",
            "kind": "proposal",
            "level": "info",
            "data": {
                "batch_id": batch.batch_id,
                "theme": batch.theme,
                "proposal_count": len(batch.proposals),
                "priorities": {
                    str(p): sum(1 for x in batch.proposals if x.priority == p)
                    for p in range(1, 6)
                },
                "generated_at": batch.generated_at
            }
        }


if __name__ == "__main__":
    generator = ProposalGenerator()

    # Example: Generate from mock patterns
    mock_patterns = [
        {
            "pattern_type": "large_function",
            "location": "nucleus/tools/lens_laser_synth.py:150",
            "description": "synthesize() is 200 lines",
            "confidence": 0.9
        }
    ]

    proposals = generator.from_code_patterns(mock_patterns)
    batch = generator.create_batch("complexity_reduction")

    print(f"Generated {len(batch.proposals)} proposals")
    for p in batch.proposals:
        print(f"  [{p.priority}] {p.proposal_type}: {p.description[:60]}...")
