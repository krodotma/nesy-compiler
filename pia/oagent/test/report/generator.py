#!/usr/bin/env python3
"""
Step 121: Test Report Generator

Generates comprehensive test reports in multiple formats.

PBTSO Phase: OBSERVE, VERIFY
Bus Topics:
- test.report.generate (subscribes)
- test.report.complete (emits)

Dependencies: Steps 101-120 (All Test Components)
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import html


# ============================================================================
# Constants
# ============================================================================

class ReportFormat(Enum):
    """Supported report formats."""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    JUNIT = "junit"
    CONSOLE = "console"
    CSV = "csv"


class ReportSection(Enum):
    """Sections that can be included in reports."""
    SUMMARY = "summary"
    TEST_RESULTS = "test_results"
    COVERAGE = "coverage"
    MUTATION = "mutation"
    FLAKY = "flaky"
    REGRESSION = "regression"
    TIMING = "timing"
    FAILURES = "failures"
    TRENDS = "trends"


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class TestResultData:
    """Test result data for reporting."""
    test_name: str
    status: str
    duration_ms: float
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "tags": self.tags,
        }


@dataclass
class CoverageData:
    """Coverage data for reporting."""
    total_lines: int = 0
    covered_lines: int = 0
    total_branches: int = 0
    covered_branches: int = 0
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def line_coverage(self) -> float:
        """Get line coverage percentage."""
        if self.total_lines == 0:
            return 0.0
        return (self.covered_lines / self.total_lines) * 100

    @property
    def branch_coverage(self) -> float:
        """Get branch coverage percentage."""
        if self.total_branches == 0:
            return 0.0
        return (self.covered_branches / self.total_branches) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_lines": self.total_lines,
            "covered_lines": self.covered_lines,
            "line_coverage": self.line_coverage,
            "total_branches": self.total_branches,
            "covered_branches": self.covered_branches,
            "branch_coverage": self.branch_coverage,
            "files": self.files,
        }


@dataclass
class ReportConfig:
    """
    Configuration for report generation.

    Attributes:
        formats: Output formats to generate
        sections: Sections to include
        output_dir: Output directory
        title: Report title
        include_passed: Include passed tests in detail
        include_skipped: Include skipped tests in detail
        max_failures: Maximum failures to show in detail
        compare_to: Previous run ID for comparison
        custom_metadata: Custom metadata to include
    """
    formats: List[ReportFormat] = field(default_factory=lambda: [
        ReportFormat.JSON, ReportFormat.HTML, ReportFormat.MARKDOWN
    ])
    sections: List[ReportSection] = field(default_factory=lambda: [
        ReportSection.SUMMARY, ReportSection.TEST_RESULTS,
        ReportSection.COVERAGE, ReportSection.FAILURES
    ])
    output_dir: str = ".pluribus/test-agent/reports"
    title: str = "Test Report"
    include_passed: bool = False
    include_skipped: bool = False
    max_failures: int = 50
    compare_to: Optional[str] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "formats": [f.value for f in self.formats],
            "sections": [s.value for s in self.sections],
            "output_dir": self.output_dir,
            "title": self.title,
            "include_passed": self.include_passed,
            "max_failures": self.max_failures,
        }


@dataclass
class ReportData:
    """Complete data for a test report."""
    run_id: str
    timestamp: float
    test_results: List[TestResultData] = field(default_factory=list)
    coverage: Optional[CoverageData] = None
    mutation_score: Optional[float] = None
    flaky_tests: List[str] = field(default_factory=list)
    regressions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tests(self) -> int:
        """Get total number of tests."""
        return len(self.test_results)

    @property
    def passed(self) -> int:
        """Get number of passed tests."""
        return sum(1 for t in self.test_results if t.status == "passed")

    @property
    def failed(self) -> int:
        """Get number of failed tests."""
        return sum(1 for t in self.test_results if t.status == "failed")

    @property
    def skipped(self) -> int:
        """Get number of skipped tests."""
        return sum(1 for t in self.test_results if t.status == "skipped")

    @property
    def error(self) -> int:
        """Get number of errored tests."""
        return sum(1 for t in self.test_results if t.status == "error")

    @property
    def duration_s(self) -> float:
        """Get total duration in seconds."""
        return sum(t.duration_ms for t in self.test_results) / 1000

    @property
    def success_rate(self) -> float:
        """Get success rate percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "error": self.error,
            "duration_s": self.duration_s,
            "success_rate": self.success_rate,
            "test_results": [t.to_dict() for t in self.test_results],
            "coverage": self.coverage.to_dict() if self.coverage else None,
            "mutation_score": self.mutation_score,
            "flaky_tests": self.flaky_tests,
            "regressions": self.regressions,
            "metadata": self.metadata,
        }


@dataclass
class ReportResult:
    """Result of report generation."""
    run_id: str
    generated_at: float
    formats_generated: List[ReportFormat] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "generated_at": self.generated_at,
            "formats_generated": [f.value for f in self.formats_generated],
            "output_files": self.output_files,
            "errors": self.errors,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class ReportBus:
    """Bus interface for report generation with file locking."""

    HEARTBEAT_INTERVAL = 300  # 5 minutes
    HEARTBEAT_TIMEOUT = 900   # 15 minutes

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                # Acquire exclusive lock for write
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError as e:
            # Log but don't fail on bus write errors
            pass

    def heartbeat(self, agent_id: str) -> None:
        """Send A2A heartbeat if interval has passed."""
        now = time.time()
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.emit({
                "topic": "a2a.heartbeat",
                "kind": "heartbeat",
                "actor": agent_id,
                "data": {"status": "alive", "interval_s": self.HEARTBEAT_INTERVAL},
            })
            self._last_heartbeat = now


# ============================================================================
# Test Report Generator
# ============================================================================

class TestReportGenerator:
    """
    Generates comprehensive test reports in multiple formats.

    Supported formats:
    - JSON: Machine-readable full report
    - HTML: Human-readable with styling
    - Markdown: Documentation-friendly
    - JUnit: CI/CD compatible XML
    - Console: Terminal output
    - CSV: Spreadsheet compatible

    PBTSO Phase: OBSERVE, VERIFY
    Bus Topics: test.report.generate, test.report.complete
    """

    BUS_TOPICS = {
        "generate": "test.report.generate",
        "complete": "test.report.complete",
        "error": "test.report.error",
    }

    def __init__(self, bus=None):
        """
        Initialize the test report generator.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus or ReportBus()
        self._format_handlers: Dict[ReportFormat, Callable] = {
            ReportFormat.JSON: self._generate_json,
            ReportFormat.HTML: self._generate_html,
            ReportFormat.MARKDOWN: self._generate_markdown,
            ReportFormat.JUNIT: self._generate_junit,
            ReportFormat.CONSOLE: self._generate_console,
            ReportFormat.CSV: self._generate_csv,
        }

    def generate(
        self,
        data: ReportData,
        config: Optional[ReportConfig] = None,
    ) -> ReportResult:
        """
        Generate test reports in configured formats.

        Args:
            data: Report data
            config: Report configuration

        Returns:
            ReportResult with generated file paths
        """
        config = config or ReportConfig()

        result = ReportResult(
            run_id=data.run_id,
            generated_at=time.time(),
        )

        # Create output directory
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Emit generation start
        self._emit_event("generate", {
            "run_id": data.run_id,
            "formats": [f.value for f in config.formats],
        })

        # Generate each format
        for fmt in config.formats:
            try:
                handler = self._format_handlers.get(fmt)
                if handler:
                    output_file = handler(data, config, output_path)
                    if output_file:
                        result.output_files.append(str(output_file))
                        result.formats_generated.append(fmt)
            except Exception as e:
                result.errors.append(f"{fmt.value}: {str(e)}")
                self._emit_event("error", {
                    "run_id": data.run_id,
                    "format": fmt.value,
                    "error": str(e),
                })

        # Emit completion
        self._emit_event("complete", {
            "run_id": data.run_id,
            "formats_generated": [f.value for f in result.formats_generated],
            "output_files": result.output_files,
            "errors": result.errors,
        })

        return result

    async def generate_async(
        self,
        data: ReportData,
        config: Optional[ReportConfig] = None,
    ) -> ReportResult:
        """Async version of report generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, data, config)

    def _generate_json(
        self,
        data: ReportData,
        config: ReportConfig,
        output_path: Path,
    ) -> Path:
        """Generate JSON report."""
        output_file = output_path / f"report_{data.run_id}.json"

        report_data = data.to_dict()
        report_data["config"] = config.to_dict()
        report_data["generated_at"] = time.time()

        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2)

        return output_file

    def _generate_html(
        self,
        data: ReportData,
        config: ReportConfig,
        output_path: Path,
    ) -> Path:
        """Generate HTML report."""
        output_file = output_path / f"report_{data.run_id}.html"

        # Status color mapping
        status_colors = {
            "passed": "#28a745",
            "failed": "#dc3545",
            "skipped": "#6c757d",
            "error": "#ffc107",
        }

        # Determine overall status
        if data.failed > 0 or data.error > 0:
            overall_status = "FAILED"
            overall_color = "#dc3545"
        else:
            overall_status = "PASSED"
            overall_color = "#28a745"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(config.title)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            background: {overall_color};
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .summary-card {{
            background: #fff;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-card h3 {{
            margin: 0;
            font-size: 2em;
            color: #333;
        }}
        .summary-card p {{
            margin: 5px 0 0 0;
            color: #666;
        }}
        .section {{
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .status {{
            padding: 3px 8px;
            border-radius: 4px;
            color: white;
            font-size: 0.85em;
        }}
        .status-passed {{ background: {status_colors['passed']}; }}
        .status-failed {{ background: {status_colors['failed']}; }}
        .status-skipped {{ background: {status_colors['skipped']}; }}
        .status-error {{ background: {status_colors['error']}; }}
        .error-details {{
            background: #fff5f5;
            border-left: 4px solid #dc3545;
            padding: 10px;
            margin-top: 5px;
            font-family: monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            overflow-x: auto;
        }}
        .coverage-bar {{
            height: 20px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
        }}
        .coverage-fill {{
            height: 100%;
            background: #28a745;
            transition: width 0.3s;
        }}
        .meta {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{html.escape(config.title)}</h1>
        <span class="status-badge">{overall_status}</span>
        <p class="meta">
            Run ID: {data.run_id}<br>
            Generated: {datetime.fromtimestamp(data.timestamp).strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>

    <div class="summary">
        <div class="summary-card">
            <h3>{data.total_tests}</h3>
            <p>Total Tests</p>
        </div>
        <div class="summary-card">
            <h3 style="color: {status_colors['passed']}">{data.passed}</h3>
            <p>Passed</p>
        </div>
        <div class="summary-card">
            <h3 style="color: {status_colors['failed']}">{data.failed}</h3>
            <p>Failed</p>
        </div>
        <div class="summary-card">
            <h3 style="color: {status_colors['skipped']}">{data.skipped}</h3>
            <p>Skipped</p>
        </div>
        <div class="summary-card">
            <h3>{data.duration_s:.2f}s</h3>
            <p>Duration</p>
        </div>
        <div class="summary-card">
            <h3>{data.success_rate:.1f}%</h3>
            <p>Success Rate</p>
        </div>
    </div>
"""

        # Coverage section
        if data.coverage and ReportSection.COVERAGE in config.sections:
            html_content += f"""
    <div class="section">
        <h2>Coverage</h2>
        <p>Line Coverage: {data.coverage.line_coverage:.1f}% ({data.coverage.covered_lines}/{data.coverage.total_lines})</p>
        <div class="coverage-bar">
            <div class="coverage-fill" style="width: {data.coverage.line_coverage}%"></div>
        </div>
        <p>Branch Coverage: {data.coverage.branch_coverage:.1f}% ({data.coverage.covered_branches}/{data.coverage.total_branches})</p>
        <div class="coverage-bar">
            <div class="coverage-fill" style="width: {data.coverage.branch_coverage}%"></div>
        </div>
    </div>
"""

        # Test results section
        if ReportSection.TEST_RESULTS in config.sections:
            html_content += """
    <div class="section">
        <h2>Test Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Test</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>File</th>
                </tr>
            </thead>
            <tbody>
"""

            # Filter and sort results
            results_to_show = data.test_results
            if not config.include_passed:
                results_to_show = [t for t in results_to_show if t.status != "passed"]
            if not config.include_skipped:
                results_to_show = [t for t in results_to_show if t.status != "skipped"]

            # Limit failures
            shown = 0
            for test in results_to_show:
                if test.status in ("failed", "error") and shown >= config.max_failures:
                    continue
                if test.status in ("failed", "error"):
                    shown += 1

                html_content += f"""
                <tr>
                    <td>{html.escape(test.test_name)}</td>
                    <td><span class="status status-{test.status}">{test.status.upper()}</span></td>
                    <td>{test.duration_ms:.0f}ms</td>
                    <td>{html.escape(test.file_path or '')}</td>
                </tr>
"""

                if test.error_message:
                    html_content += f"""
                <tr>
                    <td colspan="4">
                        <div class="error-details">{html.escape(test.error_message)}</div>
                    </td>
                </tr>
"""

            html_content += """
            </tbody>
        </table>
    </div>
"""

        # Flaky tests section
        if data.flaky_tests and ReportSection.FLAKY in config.sections:
            html_content += """
    <div class="section">
        <h2>Flaky Tests</h2>
        <ul>
"""
            for test in data.flaky_tests:
                html_content += f"            <li>{html.escape(test)}</li>\n"
            html_content += """
        </ul>
    </div>
"""

        html_content += """
</body>
</html>
"""

        with open(output_file, "w") as f:
            f.write(html_content)

        return output_file

    def _generate_markdown(
        self,
        data: ReportData,
        config: ReportConfig,
        output_path: Path,
    ) -> Path:
        """Generate Markdown report."""
        output_file = output_path / f"report_{data.run_id}.md"

        # Determine status icon
        if data.failed > 0 or data.error > 0:
            status_icon = "[FAIL]"
        else:
            status_icon = "[PASS]"

        lines = [
            f"# {config.title}",
            "",
            f"**Status**: {status_icon}",
            f"**Run ID**: `{data.run_id}`",
            f"**Generated**: {datetime.fromtimestamp(data.timestamp).strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Tests | {data.total_tests} |",
            f"| Passed | {data.passed} |",
            f"| Failed | {data.failed} |",
            f"| Skipped | {data.skipped} |",
            f"| Errors | {data.error} |",
            f"| Duration | {data.duration_s:.2f}s |",
            f"| Success Rate | {data.success_rate:.1f}% |",
        ]

        # Coverage
        if data.coverage and ReportSection.COVERAGE in config.sections:
            lines.extend([
                "",
                "## Coverage",
                "",
                f"- **Line Coverage**: {data.coverage.line_coverage:.1f}% ({data.coverage.covered_lines}/{data.coverage.total_lines})",
                f"- **Branch Coverage**: {data.coverage.branch_coverage:.1f}% ({data.coverage.covered_branches}/{data.coverage.total_branches})",
            ])

        # Failures
        if ReportSection.FAILURES in config.sections:
            failures = [t for t in data.test_results if t.status in ("failed", "error")]
            if failures:
                lines.extend([
                    "",
                    "## Failures",
                    "",
                ])

                for test in failures[:config.max_failures]:
                    lines.append(f"### {test.test_name}")
                    lines.append("")
                    lines.append(f"- **Status**: {test.status}")
                    lines.append(f"- **Duration**: {test.duration_ms:.0f}ms")
                    if test.file_path:
                        lines.append(f"- **File**: `{test.file_path}`")
                    if test.error_message:
                        lines.append("")
                        lines.append("```")
                        lines.append(test.error_message)
                        lines.append("```")
                    lines.append("")

        # Flaky tests
        if data.flaky_tests and ReportSection.FLAKY in config.sections:
            lines.extend([
                "",
                "## Flaky Tests",
                "",
            ])
            for test in data.flaky_tests:
                lines.append(f"- `{test}`")

        # Regressions
        if data.regressions and ReportSection.REGRESSION in config.sections:
            lines.extend([
                "",
                "## Regressions",
                "",
            ])
            for reg in data.regressions:
                lines.append(f"- **{reg.get('test_name', 'Unknown')}**: {reg.get('type', 'unknown')}")

        with open(output_file, "w") as f:
            f.write("\n".join(lines))

        return output_file

    def _generate_junit(
        self,
        data: ReportData,
        config: ReportConfig,
        output_path: Path,
    ) -> Path:
        """Generate JUnit XML report."""
        output_file = output_path / f"report_{data.run_id}.xml"

        # Group tests by file/suite
        suites: Dict[str, List[TestResultData]] = {}
        for test in data.test_results:
            suite_name = test.file_path or "default"
            if suite_name not in suites:
                suites[suite_name] = []
            suites[suite_name].append(test)

        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuites name="{html.escape(config.title)}" tests="{data.total_tests}" '
            f'failures="{data.failed}" errors="{data.error}" '
            f'skipped="{data.skipped}" time="{data.duration_s:.3f}">',
        ]

        for suite_name, tests in suites.items():
            suite_failures = sum(1 for t in tests if t.status == "failed")
            suite_errors = sum(1 for t in tests if t.status == "error")
            suite_skipped = sum(1 for t in tests if t.status == "skipped")
            suite_time = sum(t.duration_ms for t in tests) / 1000

            xml_lines.append(
                f'  <testsuite name="{html.escape(suite_name)}" tests="{len(tests)}" '
                f'failures="{suite_failures}" errors="{suite_errors}" '
                f'skipped="{suite_skipped}" time="{suite_time:.3f}">'
            )

            for test in tests:
                test_time = test.duration_ms / 1000
                xml_lines.append(
                    f'    <testcase name="{html.escape(test.test_name)}" '
                    f'classname="{html.escape(suite_name)}" time="{test_time:.3f}">'
                )

                if test.status == "failed":
                    message = html.escape(test.error_message or "Test failed")
                    xml_lines.append(f'      <failure message="{message}">')
                    if test.stack_trace:
                        xml_lines.append(f'        {html.escape(test.stack_trace)}')
                    xml_lines.append('      </failure>')
                elif test.status == "error":
                    message = html.escape(test.error_message or "Test error")
                    xml_lines.append(f'      <error message="{message}">')
                    if test.stack_trace:
                        xml_lines.append(f'        {html.escape(test.stack_trace)}')
                    xml_lines.append('      </error>')
                elif test.status == "skipped":
                    xml_lines.append('      <skipped/>')

                xml_lines.append('    </testcase>')

            xml_lines.append('  </testsuite>')

        xml_lines.append('</testsuites>')

        with open(output_file, "w") as f:
            f.write("\n".join(xml_lines))

        return output_file

    def _generate_console(
        self,
        data: ReportData,
        config: ReportConfig,
        output_path: Path,
    ) -> Optional[Path]:
        """Generate console output (prints to stdout)."""
        # Determine status
        if data.failed > 0 or data.error > 0:
            status = "[FAIL]"
        else:
            status = "[PASS]"

        print(f"\n{'='*60}")
        print(f"{config.title} {status}")
        print(f"{'='*60}")
        print(f"Run ID: {data.run_id}")
        print(f"Duration: {data.duration_s:.2f}s")
        print()
        print(f"Total: {data.total_tests} | Passed: {data.passed} | "
              f"Failed: {data.failed} | Skipped: {data.skipped} | "
              f"Errors: {data.error}")
        print(f"Success Rate: {data.success_rate:.1f}%")

        if data.coverage:
            print(f"\nCoverage: {data.coverage.line_coverage:.1f}% lines, "
                  f"{data.coverage.branch_coverage:.1f}% branches")

        # Show failures
        failures = [t for t in data.test_results if t.status in ("failed", "error")]
        if failures:
            print(f"\nFailures ({len(failures)}):")
            for test in failures[:config.max_failures]:
                print(f"  [FAIL] {test.test_name}")
                if test.error_message:
                    # Truncate long messages
                    msg = test.error_message[:200]
                    if len(test.error_message) > 200:
                        msg += "..."
                    print(f"         {msg}")

        print(f"{'='*60}\n")

        return None  # Console output doesn't produce a file

    def _generate_csv(
        self,
        data: ReportData,
        config: ReportConfig,
        output_path: Path,
    ) -> Path:
        """Generate CSV report."""
        output_file = output_path / f"report_{data.run_id}.csv"

        import csv

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "test_name", "status", "duration_ms", "file_path",
                "line_number", "error_message", "tags"
            ])

            # Data rows
            for test in data.test_results:
                writer.writerow([
                    test.test_name,
                    test.status,
                    test.duration_ms,
                    test.file_path or "",
                    test.line_number or "",
                    test.error_message or "",
                    ",".join(test.tags),
                ])

        return output_file

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.report.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "report_generation",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Report Generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Report Generator")
    parser.add_argument("input", help="Input JSON file with test results")
    parser.add_argument("--formats", nargs="*",
                       choices=["json", "html", "markdown", "junit", "console", "csv"],
                       default=["json", "html", "markdown"])
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/reports")
    parser.add_argument("--title", default="Test Report")
    parser.add_argument("--include-passed", action="store_true")
    parser.add_argument("--max-failures", type=int, default=50)

    args = parser.parse_args()

    # Load input data
    try:
        with open(args.input) as f:
            input_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading input: {e}")
        exit(1)

    # Convert to ReportData
    test_results = []
    for result in input_data.get("results", input_data.get("test_results", [])):
        test_results.append(TestResultData(
            test_name=result.get("test_name", result.get("name", "")),
            status=result.get("status", "unknown"),
            duration_ms=result.get("duration_ms", result.get("duration_s", 0) * 1000),
            file_path=result.get("file_path"),
            error_message=result.get("error_message"),
            stack_trace=result.get("stack_trace"),
        ))

    data = ReportData(
        run_id=input_data.get("run_id", str(uuid.uuid4())),
        timestamp=input_data.get("timestamp", time.time()),
        test_results=test_results,
    )

    config = ReportConfig(
        formats=[ReportFormat(f) for f in args.formats],
        output_dir=args.output,
        title=args.title,
        include_passed=args.include_passed,
        max_failures=args.max_failures,
    )

    generator = TestReportGenerator()
    result = generator.generate(data, config)

    print(f"\nReport generated:")
    for path in result.output_files:
        print(f"  - {path}")

    if result.errors:
        print(f"\nErrors:")
        for error in result.errors:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
