#!/usr/bin/env python3
"""
Review Priority Calculator (Step 167)

Calculates review priority based on various factors including
file types, change size, author history, and code complexity.

PBTSO Phase: PLAN
Bus Topics: review.priority.calculate, review.priority.factors

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import json
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

class PriorityLevel(Enum):
    """Priority levels for reviews."""
    CRITICAL = "critical"  # 90-100
    HIGH = "high"          # 70-89
    MEDIUM = "medium"      # 40-69
    LOW = "low"            # 20-39
    MINIMAL = "minimal"    # 0-19

    @classmethod
    def from_score(cls, score: float) -> "PriorityLevel":
        """Convert score to priority level."""
        if score >= 90:
            return cls.CRITICAL
        elif score >= 70:
            return cls.HIGH
        elif score >= 40:
            return cls.MEDIUM
        elif score >= 20:
            return cls.LOW
        else:
            return cls.MINIMAL


class FactorCategory(Enum):
    """Categories of priority factors."""
    SIZE = "size"
    COMPLEXITY = "complexity"
    RISK = "risk"
    URGENCY = "urgency"
    CONTEXT = "context"
    AUTHOR = "author"


@dataclass
class PriorityFactor:
    """
    A single factor affecting review priority.

    Attributes:
        name: Factor name
        category: Factor category
        value: Computed value (0-100)
        weight: Weight in final calculation
        details: Additional details
    """
    name: str
    category: FactorCategory
    value: float
    weight: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "value": round(self.value, 2),
            "weight": self.weight,
            "weighted_value": round(self.value * self.weight, 2),
            "details": self.details,
        }


@dataclass
class PriorityFactors:
    """Collection of priority factors."""
    factors: List[PriorityFactor] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "factors": [f.to_dict() for f in self.factors],
            "total_weight": sum(f.weight for f in self.factors),
        }

    def get_by_category(self, category: FactorCategory) -> List[PriorityFactor]:
        """Get factors by category."""
        return [f for f in self.factors if f.category == category]


@dataclass
class ReviewPriority:
    """
    Calculated review priority.

    Attributes:
        priority_id: Unique ID
        pr_id: Pull request ID
        score: Priority score (0-100)
        level: Priority level
        factors: Contributing factors
        recommendation: Review recommendation
        estimated_review_time: Estimated time in minutes
        created_at: Calculation timestamp
    """
    priority_id: str
    pr_id: str
    score: float
    level: PriorityLevel
    factors: PriorityFactors
    recommendation: str = ""
    estimated_review_time: int = 0
    created_at: str = ""

    def __post_init__(self):
        if not self.priority_id:
            self.priority_id = str(uuid.uuid4())[:8]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "priority_id": self.priority_id,
            "pr_id": self.pr_id,
            "score": round(self.score, 2),
            "level": self.level.value,
            "factors": self.factors.to_dict(),
            "recommendation": self.recommendation,
            "estimated_review_time": self.estimated_review_time,
            "created_at": self.created_at,
        }


@dataclass
class PriorityConfig:
    """Configuration for priority calculation."""
    # Weights for factor categories (should sum to 1.0)
    category_weights: Dict[FactorCategory, float] = field(default_factory=lambda: {
        FactorCategory.SIZE: 0.15,
        FactorCategory.COMPLEXITY: 0.20,
        FactorCategory.RISK: 0.30,
        FactorCategory.URGENCY: 0.15,
        FactorCategory.CONTEXT: 0.10,
        FactorCategory.AUTHOR: 0.10,
    })

    # Size thresholds
    small_change_lines: int = 50
    medium_change_lines: int = 200
    large_change_lines: int = 500

    # High-risk file patterns
    high_risk_patterns: List[str] = field(default_factory=lambda: [
        "**/auth/**",
        "**/security/**",
        "**/payment/**",
        "**/api/**",
        "**/*config*",
        "**/*secret*",
    ])

    # Critical file patterns
    critical_files: List[str] = field(default_factory=lambda: [
        "package.json",
        "requirements.txt",
        "pyproject.toml",
        "Dockerfile",
        ".env*",
    ])


# ============================================================================
# Priority Calculator
# ============================================================================

class PriorityCalculator:
    """
    Calculates review priority based on multiple factors.

    Analyzes PR metadata and changes to determine review priority.

    Example:
        calculator = PriorityCalculator()

        # Calculate priority for a PR
        priority = calculator.calculate(
            pr_id="123",
            files_changed=["auth/login.py", "utils/helpers.py"],
            lines_added=150,
            lines_deleted=30,
            labels=["security"],
        )

        print(f"Priority: {priority.level.value} ({priority.score:.1f})")
    """

    BUS_TOPICS = {
        "calculate": "review.priority.calculate",
        "factors": "review.priority.factors",
    }

    def __init__(
        self,
        config: Optional[PriorityConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the priority calculator.

        Args:
            config: Priority configuration
            bus_path: Path to event bus file
        """
        self.config = config or PriorityConfig()
        self.bus_path = bus_path or self._get_bus_path()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "priority") -> str:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "priority-calculator",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def calculate(
        self,
        pr_id: str,
        files_changed: List[str],
        lines_added: int = 0,
        lines_deleted: int = 0,
        labels: Optional[List[str]] = None,
        author: Optional[str] = None,
        is_draft: bool = False,
        is_urgent: bool = False,
        blocking_issues: int = 0,
        security_issues: int = 0,
    ) -> ReviewPriority:
        """
        Calculate review priority.

        Args:
            pr_id: Pull request ID
            files_changed: List of changed file paths
            lines_added: Lines added
            lines_deleted: Lines deleted
            labels: PR labels
            author: PR author
            is_draft: Whether PR is draft
            is_urgent: Whether PR is marked urgent
            blocking_issues: Number of blocking issues found
            security_issues: Number of security issues found

        Returns:
            ReviewPriority with score and factors

        Emits:
            review.priority.calculate
            review.priority.factors
        """
        self._emit_event(self.BUS_TOPICS["calculate"], {
            "pr_id": pr_id,
            "files_count": len(files_changed),
            "status": "started",
        })

        factors = PriorityFactors()
        labels = labels or []

        # Calculate SIZE factors
        size_factor = self._calculate_size_factor(lines_added, lines_deleted, len(files_changed))
        factors.factors.append(size_factor)

        # Calculate COMPLEXITY factors
        complexity_factor = self._calculate_complexity_factor(files_changed)
        factors.factors.append(complexity_factor)

        # Calculate RISK factors
        risk_factors = self._calculate_risk_factors(
            files_changed, labels, blocking_issues, security_issues
        )
        factors.factors.extend(risk_factors)

        # Calculate URGENCY factors
        urgency_factor = self._calculate_urgency_factor(labels, is_urgent, is_draft)
        factors.factors.append(urgency_factor)

        # Calculate CONTEXT factors
        context_factor = self._calculate_context_factor(files_changed, labels)
        factors.factors.append(context_factor)

        # Calculate AUTHOR factors (simplified)
        author_factor = self._calculate_author_factor(author)
        factors.factors.append(author_factor)

        # Compute weighted score
        total_score = 0.0
        total_weight = 0.0

        for factor in factors.factors:
            weight = self.config.category_weights.get(factor.category, 0.1)
            total_score += factor.value * weight
            total_weight += weight

        final_score = (total_score / total_weight) if total_weight > 0 else 50.0
        final_score = max(0.0, min(100.0, final_score))

        # Generate recommendation
        recommendation = self._generate_recommendation(
            PriorityLevel.from_score(final_score),
            factors,
            lines_added + lines_deleted,
        )

        # Estimate review time
        estimated_time = self._estimate_review_time(
            lines_added + lines_deleted,
            len(files_changed),
            final_score,
        )

        priority = ReviewPriority(
            priority_id=str(uuid.uuid4())[:8],
            pr_id=pr_id,
            score=final_score,
            level=PriorityLevel.from_score(final_score),
            factors=factors,
            recommendation=recommendation,
            estimated_review_time=estimated_time,
        )

        # Emit events
        self._emit_event(self.BUS_TOPICS["factors"], {
            "pr_id": pr_id,
            "factors": factors.to_dict(),
        })

        self._emit_event(self.BUS_TOPICS["calculate"], {
            "pr_id": pr_id,
            "score": priority.score,
            "level": priority.level.value,
            "status": "completed",
        })

        return priority

    def _calculate_size_factor(
        self,
        lines_added: int,
        lines_deleted: int,
        files_count: int,
    ) -> PriorityFactor:
        """Calculate size-based priority factor."""
        total_lines = lines_added + lines_deleted

        # Larger changes need more careful review
        if total_lines <= self.config.small_change_lines:
            value = 30.0
        elif total_lines <= self.config.medium_change_lines:
            value = 50.0
        elif total_lines <= self.config.large_change_lines:
            value = 70.0
        else:
            value = 90.0

        # Adjust for file count
        if files_count > 20:
            value = min(100, value + 10)

        return PriorityFactor(
            name="change_size",
            category=FactorCategory.SIZE,
            value=value,
            weight=self.config.category_weights[FactorCategory.SIZE],
            details={
                "lines_added": lines_added,
                "lines_deleted": lines_deleted,
                "files_count": files_count,
            },
        )

    def _calculate_complexity_factor(
        self,
        files_changed: List[str],
    ) -> PriorityFactor:
        """Calculate complexity-based priority factor."""
        # More complex file types get higher priority
        complex_extensions = {".py", ".ts", ".tsx", ".go", ".rs", ".java"}
        config_extensions = {".json", ".yaml", ".yml", ".toml"}

        complex_count = sum(1 for f in files_changed if Path(f).suffix in complex_extensions)
        config_count = sum(1 for f in files_changed if Path(f).suffix in config_extensions)

        total = len(files_changed) or 1
        complex_ratio = complex_count / total
        config_ratio = config_count / total

        value = 30.0 + (complex_ratio * 40) + (config_ratio * 20)

        return PriorityFactor(
            name="file_complexity",
            category=FactorCategory.COMPLEXITY,
            value=min(100, value),
            weight=self.config.category_weights[FactorCategory.COMPLEXITY],
            details={
                "complex_files": complex_count,
                "config_files": config_count,
                "total_files": len(files_changed),
            },
        )

    def _calculate_risk_factors(
        self,
        files_changed: List[str],
        labels: List[str],
        blocking_issues: int,
        security_issues: int,
    ) -> List[PriorityFactor]:
        """Calculate risk-based priority factors."""
        factors = []

        # Check for high-risk paths
        high_risk_count = 0
        for file_path in files_changed:
            path_lower = file_path.lower()
            if any(p in path_lower for p in ["auth", "security", "payment", "api", "config", "secret"]):
                high_risk_count += 1

        if high_risk_count > 0:
            value = min(100, 50 + (high_risk_count * 10))
            factors.append(PriorityFactor(
                name="high_risk_paths",
                category=FactorCategory.RISK,
                value=value,
                weight=self.config.category_weights[FactorCategory.RISK] * 0.4,
                details={"high_risk_count": high_risk_count},
            ))

        # Check for security label
        security_labels = {"security", "vulnerability", "cve"}
        has_security_label = any(l.lower() in security_labels for l in labels)

        if has_security_label or security_issues > 0:
            value = 90.0 if security_issues > 0 else 70.0
            factors.append(PriorityFactor(
                name="security_risk",
                category=FactorCategory.RISK,
                value=value,
                weight=self.config.category_weights[FactorCategory.RISK] * 0.4,
                details={
                    "has_security_label": has_security_label,
                    "security_issues": security_issues,
                },
            ))

        # Blocking issues
        if blocking_issues > 0:
            factors.append(PriorityFactor(
                name="blocking_issues",
                category=FactorCategory.RISK,
                value=min(100, 60 + (blocking_issues * 10)),
                weight=self.config.category_weights[FactorCategory.RISK] * 0.2,
                details={"blocking_count": blocking_issues},
            ))

        # If no risk factors, add a baseline
        if not factors:
            factors.append(PriorityFactor(
                name="baseline_risk",
                category=FactorCategory.RISK,
                value=30.0,
                weight=self.config.category_weights[FactorCategory.RISK],
                details={},
            ))

        return factors

    def _calculate_urgency_factor(
        self,
        labels: List[str],
        is_urgent: bool,
        is_draft: bool,
    ) -> PriorityFactor:
        """Calculate urgency-based priority factor."""
        value = 50.0

        # Urgent labels increase priority
        urgent_labels = {"urgent", "hotfix", "critical", "blocker", "p0", "p1"}
        has_urgent_label = any(l.lower() in urgent_labels for l in labels)

        if is_urgent or has_urgent_label:
            value = 90.0
        elif is_draft:
            value = 20.0

        return PriorityFactor(
            name="urgency",
            category=FactorCategory.URGENCY,
            value=value,
            weight=self.config.category_weights[FactorCategory.URGENCY],
            details={
                "is_urgent": is_urgent,
                "has_urgent_label": has_urgent_label,
                "is_draft": is_draft,
            },
        )

    def _calculate_context_factor(
        self,
        files_changed: List[str],
        labels: List[str],
    ) -> PriorityFactor:
        """Calculate context-based priority factor."""
        # Check for test files
        test_files = sum(1 for f in files_changed if "test" in f.lower())
        total = len(files_changed) or 1
        test_ratio = test_files / total

        # Test-only changes get lower priority
        if test_ratio > 0.8:
            value = 30.0
        elif test_ratio > 0.5:
            value = 50.0
        else:
            value = 60.0

        # Documentation changes get lower priority
        doc_labels = {"documentation", "docs", "typo"}
        if any(l.lower() in doc_labels for l in labels):
            value = max(20, value - 20)

        return PriorityFactor(
            name="context",
            category=FactorCategory.CONTEXT,
            value=value,
            weight=self.config.category_weights[FactorCategory.CONTEXT],
            details={
                "test_files": test_files,
                "test_ratio": round(test_ratio, 2),
            },
        )

    def _calculate_author_factor(
        self,
        author: Optional[str],
    ) -> PriorityFactor:
        """Calculate author-based priority factor."""
        # In a full implementation, this would check author history
        # For now, use a baseline
        return PriorityFactor(
            name="author",
            category=FactorCategory.AUTHOR,
            value=50.0,
            weight=self.config.category_weights[FactorCategory.AUTHOR],
            details={"author": author or "unknown"},
        )

    def _generate_recommendation(
        self,
        level: PriorityLevel,
        factors: PriorityFactors,
        total_lines: int,
    ) -> str:
        """Generate a review recommendation."""
        if level == PriorityLevel.CRITICAL:
            return "Immediate review required. High-risk changes detected."
        elif level == PriorityLevel.HIGH:
            return "Priority review needed. Review within 4 hours."
        elif level == PriorityLevel.MEDIUM:
            return "Standard review. Review within 1 business day."
        elif level == PriorityLevel.LOW:
            return "Low priority. Review when available."
        else:
            return "Minimal changes. Quick review sufficient."

    def _estimate_review_time(
        self,
        total_lines: int,
        files_count: int,
        priority_score: float,
    ) -> int:
        """Estimate review time in minutes."""
        # Base time: 1 minute per 20 lines
        base_time = max(5, total_lines // 20)

        # Add time for each file (context switching)
        file_time = files_count * 2

        # Multiply by complexity factor based on priority
        complexity_multiplier = 1.0 + (priority_score / 200)

        return int((base_time + file_time) * complexity_multiplier)


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Priority Calculator."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Priority Calculator (Step 167)")
    parser.add_argument("--pr", required=True, help="PR ID")
    parser.add_argument("--files", nargs="*", default=[], help="Changed files")
    parser.add_argument("--added", type=int, default=0, help="Lines added")
    parser.add_argument("--deleted", type=int, default=0, help="Lines deleted")
    parser.add_argument("--labels", nargs="*", default=[], help="PR labels")
    parser.add_argument("--urgent", action="store_true", help="Mark as urgent")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    calculator = PriorityCalculator()

    priority = calculator.calculate(
        pr_id=args.pr,
        files_changed=args.files,
        lines_added=args.added,
        lines_deleted=args.deleted,
        labels=args.labels,
        is_urgent=args.urgent,
    )

    if args.json:
        print(json.dumps(priority.to_dict(), indent=2))
    else:
        print(f"Review Priority for {priority.pr_id}")
        print(f"  Score: {priority.score:.1f}/100")
        print(f"  Level: {priority.level.value}")
        print(f"  Est. Time: {priority.estimated_review_time} minutes")
        print(f"  Recommendation: {priority.recommendation}")
        print("\nFactors:")
        for factor in priority.factors.factors:
            print(f"  {factor.name}: {factor.value:.1f} (weight: {factor.weight:.2f})")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
