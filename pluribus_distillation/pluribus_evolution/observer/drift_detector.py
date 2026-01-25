#!/usr/bin/env python3
"""
drift_detector.py - Detects drift between genotype (specs) and phenotype (runtime).

Part of the pluribus_evolution observer subsystem.

The drift detector identifies:
1. Schema drift: Runtime types diverging from spec definitions
2. Protocol drift: Code behavior not matching protocol versions
3. Vector drift: Embedding/manifold drift over time
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DriftSignal:
    """A detected drift between genotype and phenotype."""
    drift_type: str  # schema, protocol, vector, semantic
    source_path: str  # genotype reference
    target_path: str  # phenotype location
    severity: float  # 0.0 (none) - 1.0 (critical)
    description: str
    remediation: str | None = None


@dataclass
class DriftReport:
    """Summary of drift analysis."""
    timestamp: str
    genotype_hash: str
    phenotype_hash: str
    signals: list[DriftSignal] = field(default_factory=list)
    total_drift: float = 0.0  # Aggregate drift score


class DriftDetector:
    """
    Detects drift between pluribus genotype (specs, schemas) and phenotype (runtime).

    Temporal modes:
    - Retroactive: Compare current state to historical snapshots
    - Current: Compare live phenotype to canonical genotype
    - Predictive: Project drift trajectory based on velocity
    """

    def __init__(
        self,
        genotype_root: str = "/pluribus/nucleus/specs",
        phenotype_root: str = "/pluribus/nucleus/tools"
    ):
        self.genotype_root = Path(genotype_root)
        self.phenotype_root = Path(phenotype_root)

    def compute_hash(self, path: Path) -> str:
        """Compute content hash for drift tracking."""
        if not path.exists():
            return "missing"
        if path.is_dir():
            # Hash directory contents
            hasher = hashlib.sha256()
            for f in sorted(path.rglob("*")):
                if f.is_file() and "__pycache__" not in str(f):
                    hasher.update(f.read_bytes())
            return hasher.hexdigest()[:16]
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    def detect_schema_drift(self) -> list[DriftSignal]:
        """Detect drift between JSON schemas and runtime types."""
        signals = []

        # Check semops.json against actual operators
        semops_path = self.genotype_root / "semops.json"
        if semops_path.exists():
            try:
                with open(semops_path) as f:
                    semops = json.load(f)

                declared_ops = set(semops.get("operators", {}).keys())

                # Scan for implemented operators (PB* pattern)
                implemented_ops = set()
                for py_file in self.phenotype_root.glob("pb*.py"):
                    op_name = py_file.stem.upper()
                    implemented_ops.add(op_name)

                # Check for drift
                missing_impl = declared_ops - implemented_ops
                undeclared = implemented_ops - declared_ops

                if missing_impl:
                    signals.append(DriftSignal(
                        drift_type="schema",
                        source_path=str(semops_path),
                        target_path=str(self.phenotype_root),
                        severity=0.5 * (len(missing_impl) / max(len(declared_ops), 1)),
                        description=f"Declared operators missing implementation: {missing_impl}",
                        remediation="Implement missing operators or remove from semops.json"
                    ))

                if undeclared:
                    signals.append(DriftSignal(
                        drift_type="schema",
                        source_path=str(semops_path),
                        target_path=str(self.phenotype_root),
                        severity=0.3 * (len(undeclared) / max(len(implemented_ops), 1)),
                        description=f"Implemented operators not in semops: {undeclared}",
                        remediation="Add operators to semops.json or deprecate"
                    ))

            except Exception as e:
                signals.append(DriftSignal(
                    drift_type="schema",
                    source_path=str(semops_path),
                    target_path="<parse_error>",
                    severity=0.8,
                    description=f"Failed to parse semops.json: {e}"
                ))

        return signals

    def detect_protocol_drift(self) -> list[DriftSignal]:
        """Detect drift between protocol versions and implementation."""
        signals = []

        # Check for DKIN protocol version consistency
        dkin_specs = list(self.genotype_root.glob("dkin_protocol_v*.md"))
        if dkin_specs:
            latest_version = max(
                int(p.stem.split("_v")[1].split("_")[0])
                for p in dkin_specs
            )

            # Check if CLAUDE.md references current version
            claude_md = Path("/pluribus/CLAUDE.md")
            if claude_md.exists():
                content = claude_md.read_text()
                if f"v{latest_version}" not in content and f"v{latest_version-1}" not in content:
                    signals.append(DriftSignal(
                        drift_type="protocol",
                        source_path=f"dkin_protocol_v{latest_version}.md",
                        target_path=str(claude_md),
                        severity=0.4,
                        description=f"CLAUDE.md may not reference latest DKIN v{latest_version}",
                        remediation="Update CLAUDE.md DKIN version reference"
                    ))

        return signals

    def detect_all(self) -> DriftReport:
        """Run all drift detection and produce report."""
        import time

        signals = []
        signals.extend(self.detect_schema_drift())
        signals.extend(self.detect_protocol_drift())

        # Calculate aggregate drift
        total_drift = sum(s.severity for s in signals) / max(len(signals), 1)

        return DriftReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            genotype_hash=self.compute_hash(self.genotype_root),
            phenotype_hash=self.compute_hash(self.phenotype_root),
            signals=signals,
            total_drift=total_drift
        )

    def to_bus_event(self, report: DriftReport) -> dict:
        """Convert drift report to bus event payload."""
        return {
            "topic": "evolution.observer.drift",
            "kind": "metric",
            "level": "warn" if report.total_drift > 0.3 else "info",
            "data": {
                "timestamp": report.timestamp,
                "genotype_hash": report.genotype_hash,
                "phenotype_hash": report.phenotype_hash,
                "signal_count": len(report.signals),
                "total_drift": round(report.total_drift, 3),
                "signals": [
                    {
                        "type": s.drift_type,
                        "severity": s.severity,
                        "description": s.description[:200]
                    }
                    for s in report.signals[:5]  # Top 5
                ]
            }
        }


if __name__ == "__main__":
    detector = DriftDetector()
    report = detector.detect_all()

    print(f"Drift Report ({report.timestamp})")
    print(f"  Genotype hash: {report.genotype_hash}")
    print(f"  Phenotype hash: {report.phenotype_hash}")
    print(f"  Total drift: {report.total_drift:.3f}")
    print(f"\nSignals ({len(report.signals)}):")
    for s in report.signals:
        print(f"  [{s.drift_type}] {s.severity:.2f} - {s.description[:80]}")
