#!/usr/bin/env python3
"""
Step 110: Coverage Reporter

Generates coverage reports in various formats.

PBTSO Phase: VERIFY, OBSERVE
Bus Topics:
- test.coverage.report (emits)

Dependencies: Step 109 (Coverage Analyzer)
"""
from __future__ import annotations

import html
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# Import from analyzer
try:
    from .analyzer import CoverageData, FileCoverage
except ImportError:
    # Standalone usage - define minimal types
    @dataclass
    class FileCoverage:
        file_path: str
        total_lines: int = 0
        covered_lines: int = 0
        line_coverage_percent: float = 0.0

    @dataclass
    class CoverageData:
        id: str
        timestamp: float
        source: str
        files: List[FileCoverage] = field(default_factory=list)


# ============================================================================
# Data Types
# ============================================================================

class ReportFormat(Enum):
    """Supported report formats."""
    TEXT = "text"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    BADGE = "badge"  # SVG badge
    LCOV = "lcov"
    COBERTURA = "cobertura"


@dataclass
class CoverageReport:
    """Generated coverage report."""
    report_id: str
    format: ReportFormat
    content: str
    coverage_data: CoverageData
    generated_at: float = field(default_factory=time.time)
    output_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "format": self.format.value,
            "generated_at": self.generated_at,
            "output_path": self.output_path,
            "line_coverage": self.coverage_data.line_coverage_percent,
            "metadata": self.metadata,
        }


# ============================================================================
# Report Templates
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coverage Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4a90d9;
            padding-bottom: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .coverage-high {{ color: #28a745; }}
        .coverage-medium {{ color: #ffc107; }}
        .coverage-low {{ color: #dc3545; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #4a90d9;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #4a90d9);
            transition: width 0.3s ease;
        }}
        .timestamp {{
            color: #999;
            font-size: 0.8em;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Coverage Report</h1>
        <p class="timestamp">Generated: {timestamp}</p>

        <div class="summary">
            <div class="metric">
                <div class="metric-value {line_class}">{line_coverage:.1f}%</div>
                <div class="metric-label">Line Coverage</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {line_coverage}%"></div>
                </div>
            </div>
            <div class="metric">
                <div class="metric-value {func_class}">{func_coverage:.1f}%</div>
                <div class="metric-label">Function Coverage</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {func_coverage}%"></div>
                </div>
            </div>
            <div class="metric">
                <div class="metric-value {branch_class}">{branch_coverage:.1f}%</div>
                <div class="metric-label">Branch Coverage</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {branch_coverage}%"></div>
                </div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_files}</div>
                <div class="metric-label">Files Analyzed</div>
            </div>
        </div>

        <h2>File Coverage</h2>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Lines</th>
                    <th>Coverage</th>
                </tr>
            </thead>
            <tbody>
                {file_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""

BADGE_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="a">
    <rect width="{width}" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <rect width="61" height="20" fill="#555"/>
    <rect x="61" width="{value_width}" height="20" fill="{color}"/>
    <rect width="{width}" height="20" fill="url(#b)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="30.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
    <text x="30.5" y="14">coverage</text>
    <text x="{text_x}" y="15" fill="#010101" fill-opacity=".3">{value}%</text>
    <text x="{text_x}" y="14">{value}%</text>
  </g>
</svg>
"""


# ============================================================================
# Coverage Reporter
# ============================================================================

class CoverageReporter:
    """
    Generates coverage reports in various formats.

    PBTSO Phase: VERIFY, OBSERVE
    Bus Topics: test.coverage.report
    """

    BUS_TOPICS = {
        "report": "test.coverage.report",
    }

    def __init__(self, bus=None):
        """
        Initialize the coverage reporter.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus

    def generate(
        self,
        data: CoverageData,
        format: ReportFormat = ReportFormat.TEXT,
        output_path: Optional[str] = None,
        threshold: float = 80.0,
    ) -> CoverageReport:
        """
        Generate a coverage report.

        Args:
            data: Coverage data to report on
            format: Output format
            output_path: Optional file path to write report
            threshold: Coverage threshold for highlighting

        Returns:
            CoverageReport with generated content
        """
        # Generate content based on format
        if format == ReportFormat.TEXT:
            content = self._generate_text(data, threshold)
        elif format == ReportFormat.JSON:
            content = self._generate_json(data)
        elif format == ReportFormat.HTML:
            content = self._generate_html(data, threshold)
        elif format == ReportFormat.MARKDOWN:
            content = self._generate_markdown(data, threshold)
        elif format == ReportFormat.BADGE:
            content = self._generate_badge(data)
        elif format == ReportFormat.LCOV:
            content = self._generate_lcov(data)
        else:
            content = self._generate_text(data, threshold)

        report = CoverageReport(
            report_id=str(uuid.uuid4()),
            format=format,
            content=content,
            coverage_data=data,
            output_path=output_path,
            metadata={
                "threshold": threshold,
                "source": data.source,
            },
        )

        # Write to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(content)
            report.output_path = output_path

        # Emit report event
        self._emit_event("report", {
            "report_id": report.report_id,
            "format": format.value,
            "output_path": output_path,
            "line_coverage": data.line_coverage_percent,
        })

        return report

    def _generate_text(self, data: CoverageData, threshold: float) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 70,
            "COVERAGE REPORT",
            "=" * 70,
            "",
            f"Generated: {datetime.now().isoformat()}",
            f"Source: {data.source}",
            "",
            "-" * 70,
            "SUMMARY",
            "-" * 70,
            f"Line Coverage:     {data.line_coverage_percent:6.1f}%  ({data.covered_lines}/{data.total_lines} lines)",
            f"Function Coverage: {data.function_coverage_percent:6.1f}%  ({data.covered_functions}/{data.total_functions} functions)",
            f"Branch Coverage:   {data.branch_coverage_percent:6.1f}%  ({data.covered_branches}/{data.total_branches} branches)",
            "",
            f"Files analyzed: {len(data.files)}",
            f"Threshold: {threshold}%",
            "",
        ]

        # Files below threshold
        uncovered = data.get_uncovered_files(threshold)
        if uncovered:
            lines.append("-" * 70)
            lines.append(f"FILES BELOW {threshold}% THRESHOLD")
            lines.append("-" * 70)
            for f in sorted(uncovered, key=lambda x: x.line_coverage_percent):
                status = "[!]" if f.line_coverage_percent < threshold / 2 else "[*]"
                lines.append(f"{status} {f.line_coverage_percent:5.1f}%  {f.file_path}")
            lines.append("")

        # All files
        lines.append("-" * 70)
        lines.append("ALL FILES")
        lines.append("-" * 70)
        lines.append(f"{'Coverage':<10} {'Lines':<15} {'File'}")
        lines.append("-" * 70)

        for f in sorted(data.files, key=lambda x: x.file_path):
            lines.append(f"{f.line_coverage_percent:5.1f}%    {f.covered_lines:4d}/{f.total_lines:<4d}       {f.file_path}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def _generate_json(self, data: CoverageData) -> str:
        """Generate JSON report."""
        return json.dumps(data.to_dict(), indent=2)

    def _generate_html(self, data: CoverageData, threshold: float) -> str:
        """Generate HTML report."""
        # Determine coverage classes
        def get_class(value: float) -> str:
            if value >= 80:
                return "coverage-high"
            elif value >= 50:
                return "coverage-medium"
            return "coverage-low"

        # Generate file rows
        file_rows = []
        for f in sorted(data.files, key=lambda x: x.line_coverage_percent):
            coverage_class = get_class(f.line_coverage_percent)
            row = f"""
                <tr>
                    <td>{html.escape(f.file_path)}</td>
                    <td>{f.covered_lines}/{f.total_lines}</td>
                    <td class="{coverage_class}">{f.line_coverage_percent:.1f}%</td>
                </tr>
            """
            file_rows.append(row)

        return HTML_TEMPLATE.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            line_coverage=data.line_coverage_percent,
            func_coverage=data.function_coverage_percent,
            branch_coverage=data.branch_coverage_percent,
            line_class=get_class(data.line_coverage_percent),
            func_class=get_class(data.function_coverage_percent),
            branch_class=get_class(data.branch_coverage_percent),
            total_files=len(data.files),
            file_rows="\n".join(file_rows),
        )

    def _generate_markdown(self, data: CoverageData, threshold: float) -> str:
        """Generate Markdown report."""
        lines = [
            "# Coverage Report",
            "",
            f"> Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
            "| Metric | Coverage | Details |",
            "|--------|----------|---------|",
            f"| Lines | **{data.line_coverage_percent:.1f}%** | {data.covered_lines}/{data.total_lines} |",
            f"| Functions | **{data.function_coverage_percent:.1f}%** | {data.covered_functions}/{data.total_functions} |",
            f"| Branches | **{data.branch_coverage_percent:.1f}%** | {data.covered_branches}/{data.total_branches} |",
            "",
        ]

        # Files below threshold
        uncovered = data.get_uncovered_files(threshold)
        if uncovered:
            lines.extend([
                f"## Files Below {threshold}% Threshold",
                "",
                "| File | Coverage |",
                "|------|----------|",
            ])
            for f in sorted(uncovered, key=lambda x: x.line_coverage_percent):
                emoji = "!!!" if f.line_coverage_percent < threshold / 2 else "!"
                lines.append(f"| {f.file_path} | {f.line_coverage_percent:.1f}% {emoji} |")
            lines.append("")

        # All files
        lines.extend([
            "## All Files",
            "",
            "| File | Lines | Coverage |",
            "|------|-------|----------|",
        ])
        for f in sorted(data.files, key=lambda x: -x.line_coverage_percent):
            lines.append(f"| {f.file_path} | {f.covered_lines}/{f.total_lines} | {f.line_coverage_percent:.1f}% |")

        return "\n".join(lines)

    def _generate_badge(self, data: CoverageData) -> str:
        """Generate SVG badge."""
        coverage = data.line_coverage_percent

        # Determine color
        if coverage >= 90:
            color = "#4c1"  # Bright green
        elif coverage >= 80:
            color = "#97CA00"  # Green
        elif coverage >= 70:
            color = "#a3c51c"  # Yellow-green
        elif coverage >= 50:
            color = "#dfb317"  # Yellow
        else:
            color = "#e05d44"  # Red

        # Calculate widths
        value_str = f"{coverage:.0f}"
        value_width = len(value_str) * 8 + 10
        width = 61 + value_width
        text_x = 61 + value_width / 2

        return BADGE_TEMPLATE.format(
            width=width,
            value_width=value_width,
            color=color,
            text_x=text_x,
            value=value_str,
        )

    def _generate_lcov(self, data: CoverageData) -> str:
        """Generate LCOV format report."""
        lines = []

        for f in data.files:
            lines.append(f"SF:{f.file_path}")

            # Function coverage
            for func in f.functions:
                lines.append(f"FN:{func.start_line},{func.name}")
            for func in f.functions:
                lines.append(f"FNDA:{func.hit_count},{func.name}")
            lines.append(f"FNF:{f.total_functions}")
            lines.append(f"FNH:{f.covered_functions}")

            # Line coverage
            for line in f.lines:
                lines.append(f"DA:{line.line_number},{line.hit_count}")
            lines.append(f"LF:{f.total_lines}")
            lines.append(f"LH:{f.covered_lines}")

            # Branch coverage
            for branch in f.branches:
                lines.append(f"BRDA:{branch.line_number},{branch.branch_id},0,{branch.taken}")
                lines.append(f"BRDA:{branch.line_number},{branch.branch_id},1,{branch.not_taken}")
            lines.append(f"BRF:{f.total_branches * 2}")
            lines.append(f"BRH:{f.covered_branches}")

            lines.append("end_of_record")

        return "\n".join(lines)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        if self.bus:
            topic = self.BUS_TOPICS.get(event_type, f"test.coverage.{event_type}")
            self.bus.emit({
                "topic": topic,
                "kind": "coverage_report",
                "actor": "test-agent",
                "data": data,
            })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Coverage Reporter."""
    import argparse

    from .analyzer import CoverageAnalyzer

    parser = argparse.ArgumentParser(description="Coverage Reporter")
    parser.add_argument("path", help="Path to coverage file")
    parser.add_argument("-f", "--format", choices=["text", "json", "html", "markdown", "badge", "lcov"],
                        default="text", help="Output format")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--threshold", type=float, default=80.0, help="Coverage threshold")

    args = parser.parse_args()

    # Analyze coverage first
    analyzer = CoverageAnalyzer()
    data = analyzer.analyze(args.path, threshold=args.threshold)

    # Generate report
    reporter = CoverageReporter()
    format_enum = ReportFormat(args.format)

    report = reporter.generate(
        data=data,
        format=format_enum,
        output_path=args.output,
        threshold=args.threshold,
    )

    # Print to stdout if no output file
    if not args.output:
        print(report.content)
    else:
        print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
