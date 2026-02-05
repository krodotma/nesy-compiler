#!/usr/bin/env python3
"""
Review Report Generator (Step 172)

Generates comprehensive review reports in multiple formats.

PBTSO Phase: DISTILL, DISTRIBUTE
Bus Topics: review.report.generate, review.report.export

Supports:
- Markdown reports
- HTML reports
- JSON reports
- PDF-ready output
- Executive summaries
- Technical deep-dives

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import html
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable


# ============================================================================
# Types
# ============================================================================

class ReportFormat(Enum):
    """Report output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    TEXT = "text"


class ReportType(Enum):
    """Types of reports."""
    FULL = "full"
    SUMMARY = "summary"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    SECURITY = "security"
    COMPLIANCE = "compliance"


class ReportSection(Enum):
    """Report sections."""
    OVERVIEW = "overview"
    METRICS = "metrics"
    FINDINGS = "findings"
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    DOCUMENTATION = "documentation"
    RECOMMENDATIONS = "recommendations"
    TRENDS = "trends"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    format: ReportFormat = ReportFormat.MARKDOWN
    report_type: ReportType = ReportType.FULL
    include_sections: List[ReportSection] = field(default_factory=lambda: list(ReportSection))
    include_code_snippets: bool = True
    max_findings_per_section: int = 50
    include_metrics_charts: bool = False
    project_name: Optional[str] = None
    author: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format": self.format.value,
            "report_type": self.report_type.value,
            "include_sections": [s.value for s in self.include_sections],
            "include_code_snippets": self.include_code_snippets,
            "max_findings_per_section": self.max_findings_per_section,
            "include_metrics_charts": self.include_metrics_charts,
            "project_name": self.project_name,
            "author": self.author,
        }


@dataclass
class Finding:
    """A single review finding."""
    file: str
    line: int
    severity: str
    category: str
    title: str
    description: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MetricData:
    """Metric data for report."""
    name: str
    value: float
    unit: str = ""
    trend: Optional[str] = None
    target: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ReportData:
    """Data to include in the report."""
    review_id: str
    files_reviewed: List[str] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    metrics: List[MetricData] = field(default_factory=list)
    quality_score: float = 0.0
    decision: str = "comment"
    duration_ms: float = 0.0
    started_at: str = ""
    completed_at: str = ""
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "review_id": self.review_id,
            "files_reviewed": self.files_reviewed,
            "findings": [f.to_dict() for f in self.findings],
            "metrics": [m.to_dict() for m in self.metrics],
            "quality_score": self.quality_score,
            "decision": self.decision,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "summary": self.summary,
            "recommendations": self.recommendations,
        }


@dataclass
class GeneratedReport:
    """A generated report."""
    report_id: str
    format: ReportFormat
    report_type: ReportType
    content: str
    generated_at: str
    data: ReportData
    file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "format": self.format.value,
            "report_type": self.report_type.value,
            "content_length": len(self.content),
            "generated_at": self.generated_at,
            "data": self.data.to_dict(),
            "file_path": self.file_path,
        }

    def save(self, path: Path) -> str:
        """Save report to file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Determine extension
        ext = {
            ReportFormat.MARKDOWN: ".md",
            ReportFormat.HTML: ".html",
            ReportFormat.JSON: ".json",
            ReportFormat.TEXT: ".txt",
        }.get(self.format, ".txt")

        file_path = path / f"review_report_{self.report_id}{ext}"

        with open(file_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(self.content)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        self.file_path = str(file_path)
        return self.file_path


# ============================================================================
# Report Formatters
# ============================================================================

class MarkdownFormatter:
    """Formats reports as Markdown."""

    def format(self, data: ReportData, config: ReportConfig) -> str:
        """Format report data as Markdown."""
        lines = [
            f"# Code Review Report",
            "",
        ]

        if config.project_name:
            lines.append(f"**Project:** {config.project_name}")
        lines.extend([
            f"**Review ID:** {data.review_id}",
            f"**Decision:** {data.decision.replace('_', ' ').title()}",
            f"**Quality Score:** {data.quality_score:.1f}/100",
            "",
        ])

        # Overview section
        if ReportSection.OVERVIEW in config.include_sections:
            lines.extend(self._format_overview(data))

        # Metrics section
        if ReportSection.METRICS in config.include_sections:
            lines.extend(self._format_metrics(data))

        # Findings section
        if ReportSection.FINDINGS in config.include_sections:
            lines.extend(self._format_findings(data, config))

        # Security section (if applicable)
        if ReportSection.SECURITY in config.include_sections:
            security_findings = [f for f in data.findings if f.category == "security"]
            if security_findings:
                lines.extend(self._format_security(security_findings, config))

        # Recommendations section
        if ReportSection.RECOMMENDATIONS in config.include_sections and data.recommendations:
            lines.extend(self._format_recommendations(data))

        lines.extend([
            "",
            "---",
            f"_Generated: {datetime.now(timezone.utc).isoformat()}Z_",
        ])

        return "\n".join(lines)

    def _format_overview(self, data: ReportData) -> List[str]:
        """Format overview section."""
        lines = [
            "## Overview",
            "",
        ]

        if data.summary:
            lines.extend([data.summary, ""])

        lines.extend([
            f"- **Files Reviewed:** {len(data.files_reviewed)}",
            f"- **Total Findings:** {len(data.findings)}",
            f"- **Duration:** {data.duration_ms / 1000:.1f} seconds",
        ])

        # Count by severity
        severity_counts: Dict[str, int] = {}
        for f in data.findings:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

        if severity_counts:
            lines.append("")
            lines.append("### Findings by Severity")
            lines.append("")
            for sev in ["blocker", "critical", "major", "minor", "suggestion"]:
                if sev in severity_counts:
                    lines.append(f"- **{sev.title()}:** {severity_counts[sev]}")

        lines.append("")
        return lines

    def _format_metrics(self, data: ReportData) -> List[str]:
        """Format metrics section."""
        if not data.metrics:
            return []

        lines = [
            "## Metrics",
            "",
            "| Metric | Value | Target | Trend |",
            "|--------|-------|--------|-------|",
        ]

        for m in data.metrics:
            target = f"{m.target:.1f}" if m.target is not None else "-"
            trend = m.trend or "-"
            lines.append(f"| {m.name} | {m.value:.2f} {m.unit} | {target} | {trend} |")

        lines.append("")
        return lines

    def _format_findings(self, data: ReportData, config: ReportConfig) -> List[str]:
        """Format findings section."""
        if not data.findings:
            return ["## Findings", "", "No issues found.", ""]

        lines = [
            "## Findings",
            "",
        ]

        # Group by category
        by_category: Dict[str, List[Finding]] = {}
        for f in data.findings:
            if f.category not in by_category:
                by_category[f.category] = []
            by_category[f.category].append(f)

        for category, findings in sorted(by_category.items()):
            lines.append(f"### {category.replace('_', ' ').title()}")
            lines.append("")

            for finding in findings[:config.max_findings_per_section]:
                severity_icon = {
                    "blocker": "[X]",
                    "critical": "[!]",
                    "major": "[*]",
                    "minor": "[-]",
                    "suggestion": "[i]",
                }.get(finding.severity, "[-]")

                lines.extend([
                    f"#### {severity_icon} {finding.title}",
                    f"**File:** `{finding.file}:{finding.line}`",
                    "",
                    finding.description,
                ])

                if config.include_code_snippets and finding.code_snippet:
                    lines.extend([
                        "",
                        "```",
                        finding.code_snippet,
                        "```",
                    ])

                if finding.suggestion:
                    lines.extend([
                        "",
                        f"**Suggestion:** {finding.suggestion}",
                    ])

                lines.extend(["", "---", ""])

            if len(findings) > config.max_findings_per_section:
                lines.append(f"_... and {len(findings) - config.max_findings_per_section} more_")
                lines.append("")

        return lines

    def _format_security(self, findings: List[Finding], config: ReportConfig) -> List[str]:
        """Format security section."""
        lines = [
            "## Security Analysis",
            "",
        ]

        critical = [f for f in findings if f.severity in ("blocker", "critical")]
        if critical:
            lines.extend([
                "### Critical Security Issues",
                "",
            ])
            for f in critical[:5]:
                lines.extend([
                    f"- **{f.title}** (`{f.file}:{f.line}`)",
                    f"  {f.description}",
                    "",
                ])

        lines.append("")
        return lines

    def _format_recommendations(self, data: ReportData) -> List[str]:
        """Format recommendations section."""
        lines = [
            "## Recommendations",
            "",
        ]

        for i, rec in enumerate(data.recommendations, 1):
            lines.append(f"{i}. {rec}")

        lines.append("")
        return lines


class HTMLFormatter:
    """Formats reports as HTML."""

    def format(self, data: ReportData, config: ReportConfig) -> str:
        """Format report data as HTML."""
        findings_html = self._format_findings_html(data, config)
        metrics_html = self._format_metrics_html(data)

        severity_counts = {}
        for f in data.findings:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Code Review Report - {html.escape(data.review_id)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        .header {{ border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 20px; }}
        .metric-card {{ display: inline-block; padding: 15px; margin: 10px; background: #f5f5f5; border-radius: 8px; }}
        .finding {{ border-left: 4px solid #ccc; padding: 10px; margin: 10px 0; background: #fafafa; }}
        .finding.blocker {{ border-color: #d32f2f; }}
        .finding.critical {{ border-color: #f57c00; }}
        .finding.major {{ border-color: #ffc107; }}
        .finding.minor {{ border-color: #2196f3; }}
        .severity {{ padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
        .severity.blocker {{ background: #ffcdd2; color: #b71c1c; }}
        .severity.critical {{ background: #ffe0b2; color: #e65100; }}
        .severity.major {{ background: #fff9c4; color: #f57f17; }}
        .severity.minor {{ background: #bbdefb; color: #1565c0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        code {{ background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Code Review Report</h1>
        <p><strong>Review ID:</strong> {html.escape(data.review_id)}</p>
        <p><strong>Decision:</strong> {html.escape(data.decision.replace('_', ' ').title())}</p>
        <p><strong>Quality Score:</strong> {data.quality_score:.1f}/100</p>
    </div>

    <h2>Overview</h2>
    <div class="metric-card">
        <strong>Files Reviewed</strong><br>
        {len(data.files_reviewed)}
    </div>
    <div class="metric-card">
        <strong>Total Findings</strong><br>
        {len(data.findings)}
    </div>
    <div class="metric-card">
        <strong>Duration</strong><br>
        {data.duration_ms / 1000:.1f}s
    </div>

    {metrics_html}

    <h2>Findings</h2>
    {findings_html}

    <footer>
        <hr>
        <p><em>Generated: {datetime.now(timezone.utc).isoformat()}Z</em></p>
    </footer>
</body>
</html>"""

    def _format_findings_html(self, data: ReportData, config: ReportConfig) -> str:
        """Format findings as HTML."""
        if not data.findings:
            return "<p>No issues found.</p>"

        lines = []
        for finding in data.findings[:config.max_findings_per_section]:
            snippet_html = ""
            if config.include_code_snippets and finding.code_snippet:
                snippet_html = f"<pre><code>{html.escape(finding.code_snippet)}</code></pre>"

            suggestion_html = ""
            if finding.suggestion:
                suggestion_html = f"<p><strong>Suggestion:</strong> {html.escape(finding.suggestion)}</p>"

            lines.append(f"""
            <div class="finding {html.escape(finding.severity)}">
                <span class="severity {html.escape(finding.severity)}">{html.escape(finding.severity.upper())}</span>
                <strong>{html.escape(finding.title)}</strong>
                <p><code>{html.escape(finding.file)}:{finding.line}</code></p>
                <p>{html.escape(finding.description)}</p>
                {snippet_html}
                {suggestion_html}
            </div>
            """)

        return "\n".join(lines)

    def _format_metrics_html(self, data: ReportData) -> str:
        """Format metrics as HTML table."""
        if not data.metrics:
            return ""

        rows = []
        for m in data.metrics:
            target = f"{m.target:.1f}" if m.target is not None else "-"
            trend = m.trend or "-"
            rows.append(f"""
            <tr>
                <td>{html.escape(m.name)}</td>
                <td>{m.value:.2f} {html.escape(m.unit)}</td>
                <td>{target}</td>
                <td>{html.escape(trend)}</td>
            </tr>
            """)

        return f"""
        <h2>Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Target</th>
                    <th>Trend</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
        """


class JSONFormatter:
    """Formats reports as JSON."""

    def format(self, data: ReportData, config: ReportConfig) -> str:
        """Format report data as JSON."""
        return json.dumps(data.to_dict(), indent=2)


class TextFormatter:
    """Formats reports as plain text."""

    def format(self, data: ReportData, config: ReportConfig) -> str:
        """Format report data as plain text."""
        lines = [
            "=" * 60,
            "CODE REVIEW REPORT",
            "=" * 60,
            "",
            f"Review ID: {data.review_id}",
            f"Decision: {data.decision.replace('_', ' ').title()}",
            f"Quality Score: {data.quality_score:.1f}/100",
            f"Files Reviewed: {len(data.files_reviewed)}",
            f"Total Findings: {len(data.findings)}",
            "",
            "-" * 60,
            "FINDINGS",
            "-" * 60,
            "",
        ]

        for finding in data.findings[:config.max_findings_per_section]:
            lines.extend([
                f"[{finding.severity.upper()}] {finding.title}",
                f"  File: {finding.file}:{finding.line}",
                f"  {finding.description}",
            ])
            if finding.suggestion:
                lines.append(f"  Suggestion: {finding.suggestion}")
            lines.append("")

        lines.extend([
            "-" * 60,
            f"Generated: {datetime.now(timezone.utc).isoformat()}Z",
        ])

        return "\n".join(lines)


# ============================================================================
# Report Generator
# ============================================================================

class ReportGenerator:
    """
    Generates comprehensive review reports.

    Supports multiple output formats and report types.

    Example:
        generator = ReportGenerator()

        data = ReportData(
            review_id="abc123",
            files_reviewed=["file.py"],
            findings=[Finding(...)],
            quality_score=75.0,
        )

        report = await generator.generate(data)
        report.save(Path("/output"))
    """

    BUS_TOPICS = {
        "generate": "review.report.generate",
        "export": "review.report.export",
    }

    FORMATTERS = {
        ReportFormat.MARKDOWN: MarkdownFormatter,
        ReportFormat.HTML: HTMLFormatter,
        ReportFormat.JSON: JSONFormatter,
        ReportFormat.TEXT: TextFormatter,
    }

    def __init__(
        self,
        config: Optional[ReportConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the report generator.

        Args:
            config: Report configuration
            bus_path: Path to event bus file
        """
        self.config = config or ReportConfig()
        self.bus_path = bus_path or self._get_bus_path()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "report") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "report-generator",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    async def generate(
        self,
        data: ReportData,
        config: Optional[ReportConfig] = None,
    ) -> GeneratedReport:
        """
        Generate a review report.

        Args:
            data: Report data
            config: Optional config override

        Returns:
            GeneratedReport with formatted content

        Emits:
            review.report.generate
        """
        config = config or self.config
        report_id = str(uuid.uuid4())[:8]

        self._emit_event(self.BUS_TOPICS["generate"], {
            "report_id": report_id,
            "review_id": data.review_id,
            "format": config.format.value,
            "report_type": config.report_type.value,
            "status": "generating",
        })

        # Get formatter
        formatter_class = self.FORMATTERS.get(config.format, MarkdownFormatter)
        formatter = formatter_class()

        # Generate content
        content = formatter.format(data, config)

        report = GeneratedReport(
            report_id=report_id,
            format=config.format,
            report_type=config.report_type,
            content=content,
            generated_at=datetime.now(timezone.utc).isoformat() + "Z",
            data=data,
        )

        self._emit_event(self.BUS_TOPICS["generate"], {
            "report_id": report_id,
            "content_length": len(content),
            "status": "completed",
        })

        return report

    async def export(
        self,
        report: GeneratedReport,
        output_path: Path,
    ) -> str:
        """
        Export report to file.

        Args:
            report: Generated report
            output_path: Output directory path

        Returns:
            Path to saved file

        Emits:
            review.report.export
        """
        file_path = report.save(output_path)

        self._emit_event(self.BUS_TOPICS["export"], {
            "report_id": report.report_id,
            "file_path": file_path,
            "format": report.format.value,
        })

        return file_path


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Report Generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Report Generator (Step 172)")
    parser.add_argument("--review-id", default="demo", help="Review ID")
    parser.add_argument("--format", choices=["markdown", "html", "json", "text"],
                        default="markdown", help="Output format")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--demo", action="store_true", help="Generate demo report")

    args = parser.parse_args()

    config = ReportConfig(
        format=ReportFormat[args.format.upper()],
    )

    generator = ReportGenerator(config)

    # Create demo data
    data = ReportData(
        review_id=args.review_id,
        files_reviewed=["src/main.py", "src/utils.py"],
        findings=[
            Finding(
                file="src/main.py",
                line=42,
                severity="major",
                category="security",
                title="SQL Injection Vulnerability",
                description="User input is directly interpolated into SQL query.",
                suggestion="Use parameterized queries instead.",
            ),
            Finding(
                file="src/utils.py",
                line=15,
                severity="minor",
                category="style",
                title="Missing docstring",
                description="Function lacks documentation.",
                suggestion="Add a docstring describing the function purpose.",
            ),
        ],
        metrics=[
            MetricData(name="Code Coverage", value=75.5, unit="%", target=80.0, trend="up"),
            MetricData(name="Complexity", value=12.3, unit="avg", target=10.0, trend="stable"),
        ],
        quality_score=72.5,
        decision="request_changes",
        duration_ms=5000,
        summary="Review found 2 issues requiring attention.",
        recommendations=[
            "Address security vulnerability before merge",
            "Add documentation for public functions",
        ],
    )

    report = asyncio.run(generator.generate(data, config))

    if args.output:
        output_path = Path(args.output).parent
        file_path = asyncio.run(generator.export(report, output_path))
        print(f"Report saved to: {file_path}")
    else:
        print(report.content)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
