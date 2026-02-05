#!/usr/bin/env python3
"""
Code Quality Scorer (Step 163)

Computes quality scores for code across multiple dimensions:
- Maintainability
- Reliability
- Security
- Testability
- Documentation
- Performance

PBTSO Phase: VERIFY, DISTILL
Bus Topics: review.quality.score, review.quality.dimensions

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================================
# Types
# ============================================================================

class QualityDimension(Enum):
    """Dimensions of code quality."""
    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"
    SECURITY = "security"
    TESTABILITY = "testability"
    DOCUMENTATION = "documentation"
    PERFORMANCE = "performance"
    COMPLEXITY = "complexity"
    DUPLICATION = "duplication"


class QualityGrade(Enum):
    """Quality grade levels (A-F)."""
    A = "A"  # 90-100%
    B = "B"  # 80-89%
    C = "C"  # 70-79%
    D = "D"  # 60-69%
    F = "F"  # Below 60%

    @classmethod
    def from_score(cls, score: float) -> "QualityGrade":
        """Convert score to grade."""
        if score >= 90:
            return cls.A
        elif score >= 80:
            return cls.B
        elif score >= 70:
            return cls.C
        elif score >= 60:
            return cls.D
        else:
            return cls.F


@dataclass
class DimensionScore:
    """
    Score for a single quality dimension.

    Attributes:
        dimension: The quality dimension
        score: Score from 0-100
        grade: Letter grade
        weight: Weight in overall score
        findings: Number of findings affecting this dimension
        details: Detailed breakdown
    """
    dimension: QualityDimension
    score: float
    grade: QualityGrade
    weight: float
    findings: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension.value,
            "score": round(self.score, 2),
            "grade": self.grade.value,
            "weight": self.weight,
            "findings": self.findings,
            "details": self.details,
        }


@dataclass
class QualityScore:
    """
    Overall quality score for code.

    Attributes:
        score_id: Unique score identifier
        overall_score: Weighted overall score (0-100)
        overall_grade: Overall letter grade
        dimensions: Scores by dimension
        files_analyzed: Number of files analyzed
        lines_analyzed: Lines of code analyzed
        issue_density: Issues per 1000 lines
        technical_debt_hours: Estimated hours to fix issues
        created_at: Timestamp
    """
    score_id: str
    overall_score: float
    overall_grade: QualityGrade
    dimensions: List[DimensionScore]
    files_analyzed: int = 0
    lines_analyzed: int = 0
    issue_density: float = 0.0
    technical_debt_hours: float = 0.0
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score_id": self.score_id,
            "overall_score": round(self.overall_score, 2),
            "overall_grade": self.overall_grade.value,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "files_analyzed": self.files_analyzed,
            "lines_analyzed": self.lines_analyzed,
            "issue_density": round(self.issue_density, 2),
            "technical_debt_hours": round(self.technical_debt_hours, 1),
            "created_at": self.created_at,
        }

    def to_markdown(self) -> str:
        """Convert to markdown report."""
        lines = [
            "# Code Quality Report",
            "",
            f"**Overall Score:** {self.overall_score:.1f}/100 ({self.overall_grade.value})",
            "",
            "## Summary",
            "",
            f"- Files Analyzed: {self.files_analyzed}",
            f"- Lines of Code: {self.lines_analyzed:,}",
            f"- Issue Density: {self.issue_density:.2f} per 1K lines",
            f"- Technical Debt: {self.technical_debt_hours:.1f} hours",
            "",
            "## Dimension Scores",
            "",
            "| Dimension | Score | Grade | Weight | Findings |",
            "|-----------|-------|-------|--------|----------|",
        ]

        for dim in sorted(self.dimensions, key=lambda d: -d.score):
            lines.append(
                f"| {dim.dimension.value.title()} | {dim.score:.1f} | {dim.grade.value} | {dim.weight:.0%} | {dim.findings} |"
            )

        lines.extend([
            "",
            "_Generated by Code Quality Scorer_",
        ])

        return "\n".join(lines)

    def get_dimension(self, dimension: QualityDimension) -> Optional[DimensionScore]:
        """Get score for a specific dimension."""
        for d in self.dimensions:
            if d.dimension == dimension:
                return d
        return None


@dataclass
class ScoringConfig:
    """Configuration for quality scoring."""
    weights: Dict[QualityDimension, float] = field(default_factory=lambda: {
        QualityDimension.MAINTAINABILITY: 0.20,
        QualityDimension.RELIABILITY: 0.20,
        QualityDimension.SECURITY: 0.25,
        QualityDimension.TESTABILITY: 0.10,
        QualityDimension.DOCUMENTATION: 0.10,
        QualityDimension.PERFORMANCE: 0.05,
        QualityDimension.COMPLEXITY: 0.05,
        QualityDimension.DUPLICATION: 0.05,
    })
    # Issue impact on score (deduction per issue type)
    issue_impacts: Dict[str, float] = field(default_factory=lambda: {
        "critical": 10.0,
        "high": 5.0,
        "medium": 2.0,
        "low": 0.5,
        "info": 0.1,
    })
    # Technical debt estimates (hours per issue type)
    debt_estimates: Dict[str, float] = field(default_factory=lambda: {
        "critical": 4.0,
        "high": 2.0,
        "medium": 1.0,
        "low": 0.25,
        "info": 0.1,
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "weights": {k.value: v for k, v in self.weights.items()},
            "issue_impacts": self.issue_impacts,
            "debt_estimates": self.debt_estimates,
        }


# ============================================================================
# Code Quality Scorer
# ============================================================================

class CodeQualityScorer:
    """
    Computes quality scores for code.

    Takes results from various analyzers and computes weighted scores
    across multiple quality dimensions.

    Example:
        scorer = CodeQualityScorer()

        # Add findings from analyzers
        scorer.add_static_analysis_results(static_result)
        scorer.add_security_scan_results(security_result)
        scorer.add_smell_detection_results(smell_result)

        # Compute score
        score = scorer.compute_score(files_analyzed=10, lines_analyzed=5000)
        print(score.to_markdown())
    """

    BUS_TOPICS = {
        "score": "review.quality.score",
        "dimensions": "review.quality.dimensions",
        "trend": "review.quality.trend",
    }

    def __init__(
        self,
        config: Optional[ScoringConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the quality scorer.

        Args:
            config: Scoring configuration
            bus_path: Path to event bus file
        """
        self.config = config or ScoringConfig()
        self.bus_path = bus_path or self._get_bus_path()

        # Findings by dimension
        self._findings: Dict[QualityDimension, List[Dict[str, Any]]] = {
            dim: [] for dim in QualityDimension
        }

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "quality") -> str:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "quality-scorer",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def add_static_analysis_results(self, result: Any) -> None:
        """
        Add static analysis results.

        Maps issues to maintainability and reliability dimensions.
        """
        for issue in getattr(result, "issues", []):
            severity = getattr(issue, "severity", None)
            if hasattr(severity, "value"):
                severity = severity.value

            finding = {
                "type": "static_analysis",
                "severity": severity or "info",
                "rule": getattr(issue, "rule", "unknown"),
                "file": getattr(issue, "file", ""),
                "line": getattr(issue, "line", 0),
            }

            # Static issues affect maintainability
            self._findings[QualityDimension.MAINTAINABILITY].append(finding)

            # Error severity affects reliability
            if severity == "error":
                self._findings[QualityDimension.RELIABILITY].append(finding)

    def add_security_scan_results(self, result: Any) -> None:
        """
        Add security scan results.

        Maps vulnerabilities to security dimension.
        """
        for vuln in getattr(result, "vulnerabilities", []):
            severity = getattr(vuln, "severity", None)
            if hasattr(severity, "value"):
                severity = severity.value

            finding = {
                "type": "security",
                "severity": severity or "medium",
                "category": str(getattr(vuln, "category", "")),
                "cwe": getattr(vuln, "cwe", ""),
                "file": getattr(vuln, "file", ""),
                "line": getattr(vuln, "line", 0),
            }

            self._findings[QualityDimension.SECURITY].append(finding)

    def add_smell_detection_results(self, result: Any) -> None:
        """
        Add code smell detection results.

        Maps smells to maintainability and complexity dimensions.
        """
        for smell in getattr(result, "smells", []):
            smell_type = getattr(smell, "smell_type", None)
            if hasattr(smell_type, "value"):
                smell_type = smell_type.value

            severity = getattr(smell, "severity", None)
            if hasattr(severity, "value"):
                severity = severity.value

            finding = {
                "type": "smell",
                "severity": severity or "low",
                "smell_type": smell_type or "unknown",
                "file": getattr(smell, "file", ""),
                "line": getattr(smell, "line", 0),
            }

            self._findings[QualityDimension.MAINTAINABILITY].append(finding)

            # Complexity smells
            if smell_type in ("long_method", "long_class", "complex_conditional"):
                self._findings[QualityDimension.COMPLEXITY].append(finding)

            # Duplication smells
            if smell_type in ("duplicate_code", "copy_paste"):
                self._findings[QualityDimension.DUPLICATION].append(finding)

    def add_architecture_results(self, result: Any) -> None:
        """
        Add architecture check results.

        Maps violations to maintainability dimension.
        """
        for violation in getattr(result, "violations", []):
            severity = getattr(violation, "severity", None)
            if hasattr(severity, "value"):
                severity = severity.value

            finding = {
                "type": "architecture",
                "severity": severity or "medium",
                "violation_type": str(getattr(violation, "violation_type", "")),
                "file": getattr(violation, "file", ""),
                "line": getattr(violation, "line", 0),
            }

            self._findings[QualityDimension.MAINTAINABILITY].append(finding)
            self._findings[QualityDimension.RELIABILITY].append(finding)

    def add_doc_check_results(self, result: Any) -> None:
        """
        Add documentation check results.

        Maps issues to documentation dimension.
        """
        for issue in getattr(result, "issues", []):
            severity = getattr(issue, "severity", None)
            if hasattr(severity, "value"):
                severity = severity.value

            finding = {
                "type": "documentation",
                "severity": severity or "info",
                "issue_type": str(getattr(issue, "issue_type", "")),
                "file": getattr(issue, "file", ""),
                "line": getattr(issue, "line", 0),
            }

            self._findings[QualityDimension.DOCUMENTATION].append(finding)

    def add_test_coverage(self, coverage_percent: float) -> None:
        """
        Add test coverage information.

        Args:
            coverage_percent: Test coverage percentage (0-100)
        """
        # Lower coverage adds findings to testability
        if coverage_percent < 80:
            finding = {
                "type": "coverage",
                "severity": "low" if coverage_percent >= 60 else "medium",
                "coverage": coverage_percent,
            }
            # Add multiple findings for low coverage
            penalty_count = max(1, int((80 - coverage_percent) / 10))
            for _ in range(penalty_count):
                self._findings[QualityDimension.TESTABILITY].append(finding)

    def compute_score(
        self,
        files_analyzed: int = 0,
        lines_analyzed: int = 0,
    ) -> QualityScore:
        """
        Compute the quality score.

        Args:
            files_analyzed: Number of files analyzed
            lines_analyzed: Lines of code analyzed

        Returns:
            QualityScore with all dimensions

        Emits:
            review.quality.score
            review.quality.dimensions
        """
        score_id = str(uuid.uuid4())[:8]

        self._emit_event(self.BUS_TOPICS["score"], {
            "score_id": score_id,
            "files_analyzed": files_analyzed,
            "status": "started",
        })

        # Compute dimension scores
        dimension_scores = []
        total_weighted_score = 0.0
        total_weight = 0.0
        total_debt_hours = 0.0

        for dimension, weight in self.config.weights.items():
            findings = self._findings[dimension]
            dim_score = self._compute_dimension_score(dimension, findings, weight)
            dimension_scores.append(dim_score)

            total_weighted_score += dim_score.score * weight
            total_weight += weight

            # Calculate debt for this dimension
            for finding in findings:
                severity = finding.get("severity", "info")
                total_debt_hours += self.config.debt_estimates.get(severity, 0.1)

        # Normalize overall score
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 100.0
        overall_score = max(0.0, min(100.0, overall_score))

        # Calculate issue density
        total_findings = sum(len(f) for f in self._findings.values())
        issue_density = (total_findings / (lines_analyzed / 1000)) if lines_analyzed > 0 else 0.0

        score = QualityScore(
            score_id=score_id,
            overall_score=overall_score,
            overall_grade=QualityGrade.from_score(overall_score),
            dimensions=dimension_scores,
            files_analyzed=files_analyzed,
            lines_analyzed=lines_analyzed,
            issue_density=issue_density,
            technical_debt_hours=total_debt_hours,
        )

        # Emit completion
        self._emit_event(self.BUS_TOPICS["score"], {
            "score_id": score_id,
            "overall_score": score.overall_score,
            "overall_grade": score.overall_grade.value,
            "status": "completed",
        })

        # Emit dimension details
        self._emit_event(self.BUS_TOPICS["dimensions"], {
            "score_id": score_id,
            "dimensions": [d.to_dict() for d in dimension_scores],
        })

        return score

    def _compute_dimension_score(
        self,
        dimension: QualityDimension,
        findings: List[Dict[str, Any]],
        weight: float,
    ) -> DimensionScore:
        """Compute score for a single dimension."""
        # Start with perfect score
        score = 100.0

        # Deduct based on findings
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for finding in findings:
            severity = finding.get("severity", "info")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            impact = self.config.issue_impacts.get(severity, 0.1)
            score -= impact

        # Apply diminishing returns for many issues
        if len(findings) > 10:
            additional_penalty = math.log10(len(findings) - 9) * 5
            score -= additional_penalty

        # Clamp score
        score = max(0.0, min(100.0, score))

        return DimensionScore(
            dimension=dimension,
            score=score,
            grade=QualityGrade.from_score(score),
            weight=weight,
            findings=len(findings),
            details={
                "severity_counts": severity_counts,
                "total_findings": len(findings),
            },
        )

    def clear(self) -> None:
        """Clear all findings."""
        self._findings = {dim: [] for dim in QualityDimension}


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Code Quality Scorer."""
    import argparse

    parser = argparse.ArgumentParser(description="Code Quality Scorer (Step 163)")
    parser.add_argument("--files", type=int, default=0, help="Files analyzed")
    parser.add_argument("--lines", type=int, default=0, help="Lines analyzed")
    parser.add_argument("--demo", action="store_true", help="Run with demo data")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown")

    args = parser.parse_args()

    scorer = CodeQualityScorer()

    if args.demo:
        # Add some demo findings
        class MockIssue:
            def __init__(self, severity, file, line, rule):
                self.severity = severity
                self.file = file
                self.line = line
                self.rule = rule

        class MockResult:
            def __init__(self, issues):
                self.issues = issues

        # Add demo static analysis issues
        static_issues = [
            MockIssue("error", "test.py", 10, "E001"),
            MockIssue("warning", "test.py", 20, "W001"),
            MockIssue("info", "test.py", 30, "I001"),
        ]
        scorer.add_static_analysis_results(MockResult(static_issues))

        args.files = args.files or 5
        args.lines = args.lines or 1000

    score = scorer.compute_score(
        files_analyzed=args.files,
        lines_analyzed=args.lines,
    )

    if args.json:
        print(json.dumps(score.to_dict(), indent=2))
    elif args.markdown:
        print(score.to_markdown())
    else:
        print(f"Quality Score: {score.overall_score:.1f}/100 ({score.overall_grade.value})")
        print(f"  Files: {score.files_analyzed}")
        print(f"  Lines: {score.lines_analyzed:,}")
        print(f"  Issue Density: {score.issue_density:.2f}/1K lines")
        print(f"  Technical Debt: {score.technical_debt_hours:.1f}h")
        print("\nDimensions:")
        for dim in sorted(score.dimensions, key=lambda d: -d.score):
            print(f"  {dim.dimension.value}: {dim.score:.1f} ({dim.grade.value})")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
