#!/usr/bin/env python3
"""
Review Comment Generator (Step 157)

Generates human-readable review comments from analysis results.

PBTSO Phase: DISTILL
Bus Topics: review.comments.generate, review.comments.posted

Aggregates findings from:
- Static Analysis
- Security Scanner
- Code Smell Detector
- Architecture Checker
- Documentation Checker

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

class CommentSeverity(Enum):
    """Severity levels for review comments."""
    BLOCKER = "blocker"    # Must fix before merge
    CRITICAL = "critical"  # Should fix before merge
    MAJOR = "major"        # Should fix soon
    MINOR = "minor"        # Nice to fix
    SUGGESTION = "suggestion"  # Optional improvement

    @property
    def priority(self) -> int:
        """Get numeric priority (higher = more severe)."""
        return {
            self.BLOCKER: 5,
            self.CRITICAL: 4,
            self.MAJOR: 3,
            self.MINOR: 2,
            self.SUGGESTION: 1,
        }.get(self, 0)

    @property
    def emoji(self) -> str:
        """Get emoji for severity."""
        return {
            self.BLOCKER: "[X]",
            self.CRITICAL: "[!]",
            self.MAJOR: "[*]",
            self.MINOR: "[-]",
            self.SUGGESTION: "[i]",
        }.get(self, "")


class CommentCategory(Enum):
    """Categories of review comments."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    BEST_PRACTICE = "best_practice"
    BUG = "bug"


@dataclass
class ReviewComment:
    """
    A review comment for a specific location.

    Attributes:
        file: File path
        line: Line number
        end_line: End line for multi-line comments
        severity: Comment severity
        category: Comment category
        title: Short title
        body: Full comment body (markdown)
        suggestion: Code suggestion if available
        source: Source of the finding (analyzer name)
        rule_id: Rule/check ID that triggered this
    """
    file: str
    line: int
    end_line: Optional[int]
    severity: CommentSeverity
    category: CommentCategory
    title: str
    body: str
    suggestion: Optional[str] = None
    source: str = "review-agent"
    rule_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["severity"] = self.severity.value
        result["category"] = self.category.value
        return result

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"### {self.severity.emoji} {self.title}",
            f"**Location:** `{self.file}:{self.line}`",
            f"**Category:** {self.category.value.replace('_', ' ').title()}",
            "",
            self.body,
        ]

        if self.suggestion:
            lines.extend([
                "",
                "**Suggestion:**",
                "```",
                self.suggestion,
                "```",
            ])

        if self.rule_id:
            lines.append(f"\n_Rule: {self.rule_id}_")

        return "\n".join(lines)


@dataclass
class GeneratedReview:
    """Complete generated review."""
    comments: List[ReviewComment] = field(default_factory=list)
    summary: str = ""
    overall_severity: CommentSeverity = CommentSeverity.SUGGESTION
    blocker_count: int = 0
    critical_count: int = 0
    major_count: int = 0
    minor_count: int = 0
    suggestion_count: int = 0
    files_reviewed: int = 0
    duration_ms: float = 0
    should_block_merge: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "comments": [c.to_dict() for c in self.comments],
            "summary": self.summary,
            "overall_severity": self.overall_severity.value,
            "blocker_count": self.blocker_count,
            "critical_count": self.critical_count,
            "major_count": self.major_count,
            "minor_count": self.minor_count,
            "suggestion_count": self.suggestion_count,
            "files_reviewed": self.files_reviewed,
            "duration_ms": self.duration_ms,
            "should_block_merge": self.should_block_merge,
        }

    def to_markdown(self) -> str:
        """Convert to full markdown review."""
        lines = [
            "# Code Review Summary",
            "",
            self.summary,
            "",
            "## Statistics",
            "",
            f"- **Files Reviewed:** {self.files_reviewed}",
            f"- **Total Comments:** {len(self.comments)}",
            f"  - Blockers: {self.blocker_count}",
            f"  - Critical: {self.critical_count}",
            f"  - Major: {self.major_count}",
            f"  - Minor: {self.minor_count}",
            f"  - Suggestions: {self.suggestion_count}",
            "",
        ]

        if self.should_block_merge:
            lines.extend([
                "## [X] Merge Blocked",
                "",
                "This PR has blocking issues that must be resolved before merge.",
                "",
            ])

        # Group comments by file
        by_file: Dict[str, List[ReviewComment]] = {}
        for comment in self.comments:
            if comment.file not in by_file:
                by_file[comment.file] = []
            by_file[comment.file].append(comment)

        if by_file:
            lines.append("## Comments by File")
            lines.append("")

            for file_path, comments in sorted(by_file.items()):
                lines.append(f"### `{file_path}`")
                lines.append("")

                for comment in sorted(comments, key=lambda c: c.line):
                    lines.append(comment.to_markdown())
                    lines.append("")
                    lines.append("---")
                    lines.append("")

        lines.append("")
        lines.append("_Generated by Review Agent_")

        return "\n".join(lines)


# ============================================================================
# Comment Generator
# ============================================================================

class CommentGenerator:
    """
    Generates review comments from analysis results.

    Aggregates findings from multiple analyzers and produces
    unified, actionable review comments.

    Example:
        generator = CommentGenerator()

        # Add findings from different analyzers
        generator.add_static_issues(static_result.issues)
        generator.add_security_vulnerabilities(security_result.vulnerabilities)
        generator.add_code_smells(smell_result.smells)

        # Generate review
        review = generator.generate()
        print(review.to_markdown())
    """

    # Mapping from source severities to comment severities
    SEVERITY_MAP = {
        # Static analysis
        "error": CommentSeverity.MAJOR,
        "warning": CommentSeverity.MINOR,
        "info": CommentSeverity.SUGGESTION,

        # Security
        "critical": CommentSeverity.BLOCKER,
        "high": CommentSeverity.CRITICAL,
        "medium": CommentSeverity.MAJOR,
        "low": CommentSeverity.MINOR,

        # Code smells
        "high": CommentSeverity.MAJOR,
        "medium": CommentSeverity.MINOR,
        "low": CommentSeverity.SUGGESTION,
    }

    def __init__(self, bus_path: Optional[Path] = None):
        """
        Initialize the comment generator.

        Args:
            bus_path: Path to event bus file
        """
        self.bus_path = bus_path or self._get_bus_path()
        self._pending_comments: List[ReviewComment] = []
        self._files_seen: set = set()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any]) -> None:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "comment",
            "actor": "comment-generator",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def add_static_issues(self, issues: List[Any]) -> None:
        """
        Add static analysis issues.

        Args:
            issues: List of StaticAnalysisIssue objects
        """
        for issue in issues:
            self._files_seen.add(issue.file)

            severity = self.SEVERITY_MAP.get(
                issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity),
                CommentSeverity.MINOR
            )

            body = issue.message
            if hasattr(issue, "fix_suggestion") and issue.fix_suggestion:
                suggestion = issue.fix_suggestion
            else:
                suggestion = None

            self._pending_comments.append(ReviewComment(
                file=issue.file,
                line=issue.line,
                end_line=getattr(issue, "end_line", None),
                severity=severity,
                category=CommentCategory.STYLE,
                title=f"Static Analysis: {issue.rule}",
                body=body,
                suggestion=suggestion,
                source=getattr(issue, "source", "static-analyzer"),
                rule_id=issue.rule,
            ))

    def add_security_vulnerabilities(self, vulnerabilities: List[Any]) -> None:
        """
        Add security vulnerabilities.

        Args:
            vulnerabilities: List of SecurityVulnerability objects
        """
        for vuln in vulnerabilities:
            self._files_seen.add(vuln.file)

            severity = self.SEVERITY_MAP.get(
                vuln.severity.value if hasattr(vuln.severity, "value") else str(vuln.severity),
                CommentSeverity.MAJOR
            )

            body_parts = [vuln.description]
            if hasattr(vuln, "cwe") and vuln.cwe:
                body_parts.append(f"\n**CWE:** {vuln.cwe}")
            if hasattr(vuln, "owasp") and vuln.owasp:
                body_parts.append(f"**OWASP:** {vuln.owasp}")

            suggestion = getattr(vuln, "remediation", None)

            self._pending_comments.append(ReviewComment(
                file=vuln.file,
                line=vuln.line,
                end_line=None,
                severity=severity,
                category=CommentCategory.SECURITY,
                title=f"Security: {vuln.category.value if hasattr(vuln.category, 'value') else str(vuln.category)}",
                body="\n".join(body_parts),
                suggestion=suggestion,
                source="security-scanner",
                rule_id=getattr(vuln, "cwe", None),
            ))

    def add_code_smells(self, smells: List[Any]) -> None:
        """
        Add code smells.

        Args:
            smells: List of CodeSmell objects
        """
        for smell in smells:
            self._files_seen.add(smell.file)

            severity = self.SEVERITY_MAP.get(
                smell.severity.value if hasattr(smell.severity, "value") else str(smell.severity),
                CommentSeverity.MINOR
            )

            self._pending_comments.append(ReviewComment(
                file=smell.file,
                line=smell.line,
                end_line=smell.end_line,
                severity=severity,
                category=CommentCategory.MAINTAINABILITY,
                title=f"Code Smell: {smell.smell_type.value if hasattr(smell.smell_type, 'value') else str(smell.smell_type)}",
                body=smell.description,
                suggestion=smell.suggestion,
                source="smell-detector",
                rule_id=smell.smell_type.value if hasattr(smell.smell_type, "value") else None,
            ))

    def add_architecture_violations(self, violations: List[Any]) -> None:
        """
        Add architecture violations.

        Args:
            violations: List of ArchitectureViolation objects
        """
        for violation in violations:
            self._files_seen.add(violation.file)

            severity_val = violation.severity.value if hasattr(violation.severity, "value") else str(violation.severity)
            if severity_val == "critical":
                severity = CommentSeverity.BLOCKER
            elif severity_val == "error":
                severity = CommentSeverity.CRITICAL
            else:
                severity = CommentSeverity.MAJOR

            self._pending_comments.append(ReviewComment(
                file=violation.file,
                line=violation.line,
                end_line=None,
                severity=severity,
                category=CommentCategory.ARCHITECTURE,
                title=f"Architecture: {violation.violation_type.value if hasattr(violation.violation_type, 'value') else str(violation.violation_type)}",
                body=violation.description,
                suggestion=violation.suggestion,
                source="architecture-checker",
                rule_id=violation.violation_type.value if hasattr(violation.violation_type, "value") else None,
            ))

    def add_doc_issues(self, issues: List[Any]) -> None:
        """
        Add documentation issues.

        Args:
            issues: List of DocIssue objects
        """
        for issue in issues:
            self._files_seen.add(issue.file)

            severity_val = issue.severity.value if hasattr(issue.severity, "value") else str(issue.severity)
            if severity_val == "error":
                severity = CommentSeverity.MAJOR
            elif severity_val == "warning":
                severity = CommentSeverity.MINOR
            else:
                severity = CommentSeverity.SUGGESTION

            self._pending_comments.append(ReviewComment(
                file=issue.file,
                line=issue.line,
                end_line=None,
                severity=severity,
                category=CommentCategory.DOCUMENTATION,
                title=f"Documentation: {issue.issue_type.value if hasattr(issue.issue_type, 'value') else str(issue.issue_type)}",
                body=issue.description,
                suggestion=issue.suggestion,
                source="doc-checker",
                rule_id=issue.issue_type.value if hasattr(issue.issue_type, "value") else None,
            ))

    def generate(self) -> GeneratedReview:
        """
        Generate the final review from all added findings.

        Returns:
            GeneratedReview with all comments and summary

        Emits:
            review.comments.generate (start)
            review.comments.posted (completion)
        """
        start_time = time.time()

        # Emit start event
        self._emit_event("review.comments.generate", {
            "pending_comments": len(self._pending_comments),
            "files": list(self._files_seen)[:20],
            "status": "started",
        })

        # Sort comments by severity (most severe first) then by file and line
        comments = sorted(
            self._pending_comments,
            key=lambda c: (-c.severity.priority, c.file, c.line)
        )

        # Calculate counts
        blocker_count = sum(1 for c in comments if c.severity == CommentSeverity.BLOCKER)
        critical_count = sum(1 for c in comments if c.severity == CommentSeverity.CRITICAL)
        major_count = sum(1 for c in comments if c.severity == CommentSeverity.MAJOR)
        minor_count = sum(1 for c in comments if c.severity == CommentSeverity.MINOR)
        suggestion_count = sum(1 for c in comments if c.severity == CommentSeverity.SUGGESTION)

        # Determine overall severity
        if blocker_count > 0:
            overall_severity = CommentSeverity.BLOCKER
        elif critical_count > 0:
            overall_severity = CommentSeverity.CRITICAL
        elif major_count > 0:
            overall_severity = CommentSeverity.MAJOR
        elif minor_count > 0:
            overall_severity = CommentSeverity.MINOR
        else:
            overall_severity = CommentSeverity.SUGGESTION

        # Should block merge?
        should_block = blocker_count > 0 or critical_count > 0

        # Generate summary
        summary = self._generate_summary(comments, should_block)

        review = GeneratedReview(
            comments=comments,
            summary=summary,
            overall_severity=overall_severity,
            blocker_count=blocker_count,
            critical_count=critical_count,
            major_count=major_count,
            minor_count=minor_count,
            suggestion_count=suggestion_count,
            files_reviewed=len(self._files_seen),
            duration_ms=(time.time() - start_time) * 1000,
            should_block_merge=should_block,
        )

        # Emit completion
        self._emit_event("review.comments.posted", {
            "comment_count": len(comments),
            "blocker_count": blocker_count,
            "critical_count": critical_count,
            "should_block_merge": should_block,
            "files_reviewed": len(self._files_seen),
            "status": "completed",
        })

        return review

    def _generate_summary(self, comments: List[ReviewComment], should_block: bool) -> str:
        """Generate a summary of the review."""
        if not comments:
            return "No issues found. Code looks good!"

        # Count by category
        by_category: Dict[str, int] = {}
        for comment in comments:
            cat = comment.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        parts = []

        if should_block:
            parts.append("**This PR has blocking issues that must be resolved before merge.**")
            parts.append("")

        parts.append(f"Found **{len(comments)}** issues across **{len(self._files_seen)}** files:")
        parts.append("")

        for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
            parts.append(f"- **{cat.replace('_', ' ').title()}**: {count} issues")

        # Highlight key issues
        blockers = [c for c in comments if c.severity == CommentSeverity.BLOCKER]
        if blockers:
            parts.append("")
            parts.append("**Key Blockers:**")
            for b in blockers[:3]:
                parts.append(f"- `{b.file}:{b.line}`: {b.title}")

        return "\n".join(parts)

    def clear(self) -> None:
        """Clear all pending comments."""
        self._pending_comments = []
        self._files_seen = set()


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Comment Generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Comment Generator (Step 157)")
    parser.add_argument("--static", help="JSON file with static analysis results")
    parser.add_argument("--security", help="JSON file with security scan results")
    parser.add_argument("--smells", help="JSON file with code smell results")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown (default)")

    args = parser.parse_args()

    generator = CommentGenerator()

    # Load results from files if provided
    # (In practice, this would be integrated with the actual result objects)

    review = generator.generate()

    if args.json:
        print(json.dumps(review.to_dict(), indent=2))
    else:
        print(review.to_markdown())

    return 1 if review.should_block_merge else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
