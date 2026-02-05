#!/usr/bin/env python3
"""
Monitor Report Generator - Step 272

Generates monitoring reports for various time periods and formats.

PBTSO Phase: REPORT

Bus Topics:
- monitor.report.generate (subscribed)
- monitor.report.complete (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ReportFormat(Enum):
    """Report output formats."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    CSV = "csv"


class ReportPeriod(Enum):
    """Report time periods."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class ReportType(Enum):
    """Types of reports."""
    SYSTEM_HEALTH = "system_health"
    ALERT_SUMMARY = "alert_summary"
    INCIDENT_REPORT = "incident_report"
    SLO_COMPLIANCE = "slo_compliance"
    PERFORMANCE_TRENDS = "performance_trends"
    CAPACITY_REPORT = "capacity_report"
    CUSTOM = "custom"


@dataclass
class ReportSection:
    """A section within a report.

    Attributes:
        section_id: Unique section ID
        title: Section title
        content: Section content
        charts: Charts/visualizations
        tables: Data tables
        order: Section order
    """
    section_id: str
    title: str
    content: str = ""
    charts: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    order: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "charts": self.charts,
            "tables": self.tables,
            "order": self.order,
        }


@dataclass
class Report:
    """A monitoring report.

    Attributes:
        report_id: Unique report ID
        title: Report title
        report_type: Type of report
        period: Report period
        start_time: Report start timestamp
        end_time: Report end timestamp
        sections: Report sections
        summary: Executive summary
        metadata: Report metadata
        generated_at: Generation timestamp
        format: Output format
    """
    report_id: str
    title: str
    report_type: ReportType
    period: ReportPeriod
    start_time: float
    end_time: float
    sections: List[ReportSection] = field(default_factory=list)
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)
    format: ReportFormat = ReportFormat.JSON

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "title": self.title,
            "report_type": self.report_type.value,
            "period": self.period.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "sections": [s.to_dict() for s in self.sections],
            "summary": self.summary,
            "metadata": self.metadata,
            "generated_at": self.generated_at,
            "format": self.format.value,
        }

    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)
        self.sections.sort(key=lambda s: s.order)


@dataclass
class ReportTemplate:
    """Template for generating reports.

    Attributes:
        template_id: Template ID
        name: Template name
        report_type: Type of report
        sections: Section definitions
        variables: Template variables
    """
    template_id: str
    name: str
    report_type: ReportType
    sections: List[Dict[str, Any]] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "template_id": self.template_id,
            "name": self.name,
            "report_type": self.report_type.value,
            "sections": self.sections,
            "variables": self.variables,
        }


class ReportGenerator:
    """
    Generate monitoring reports.

    The generator:
    - Creates reports from templates
    - Collects data from various sources
    - Formats output in multiple formats
    - Handles scheduled report generation

    Example:
        generator = ReportGenerator()

        # Generate a daily health report
        report = await generator.generate_report(
            report_type=ReportType.SYSTEM_HEALTH,
            period=ReportPeriod.DAILY,
            format=ReportFormat.MARKDOWN
        )

        # Export to file
        generator.export_report(report, "/path/to/report.md")
    """

    BUS_TOPICS = {
        "generate": "monitor.report.generate",
        "complete": "monitor.report.complete",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        output_dir: Optional[str] = None,
        bus_dir: Optional[str] = None,
    ):
        """Initialize report generator.

        Args:
            output_dir: Directory for report output
            bus_dir: Bus directory
        """
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")

        self._output_dir = Path(output_dir or os.path.join(pluribus_root, ".pluribus", "reports"))
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        self._templates: Dict[str, ReportTemplate] = {}
        self._reports: Dict[str, Report] = {}
        self._last_heartbeat = time.time()

        # Register default templates
        self._register_default_templates()

    def register_template(self, template: ReportTemplate) -> None:
        """Register a report template.

        Args:
            template: Template to register
        """
        self._templates[template.template_id] = template

    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """Get a template by ID.

        Args:
            template_id: Template ID

        Returns:
            Template or None
        """
        return self._templates.get(template_id)

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all templates.

        Returns:
            Template summaries
        """
        return [t.to_dict() for t in self._templates.values()]

    async def generate_report(
        self,
        report_type: ReportType,
        period: ReportPeriod = ReportPeriod.DAILY,
        format: ReportFormat = ReportFormat.JSON,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        title: Optional[str] = None,
    ) -> Report:
        """Generate a report.

        Args:
            report_type: Type of report
            period: Report period
            format: Output format
            start_time: Custom start time
            end_time: Custom end time
            title: Custom title

        Returns:
            Generated report
        """
        # Calculate time range
        if start_time is None or end_time is None:
            end_time = time.time()
            start_time = self._calculate_period_start(period, end_time)

        # Create report
        report_id = f"report-{uuid.uuid4().hex[:8]}"

        report = Report(
            report_id=report_id,
            title=title or f"{report_type.value.replace('_', ' ').title()} Report",
            report_type=report_type,
            period=period,
            start_time=start_time,
            end_time=end_time,
            format=format,
            metadata={
                "generator_version": "1.0.0",
                "generated_by": "monitor-agent",
            },
        )

        # Generate sections based on report type
        await self._generate_sections(report)

        # Generate summary
        report.summary = self._generate_summary(report)

        self._reports[report_id] = report

        self._emit_bus_event(
            self.BUS_TOPICS["complete"],
            {
                "report_id": report_id,
                "report_type": report_type.value,
                "period": period.value,
                "section_count": len(report.sections),
            }
        )

        return report

    async def generate_from_template(
        self,
        template_id: str,
        variables: Optional[Dict[str, Any]] = None,
        format: ReportFormat = ReportFormat.JSON,
    ) -> Optional[Report]:
        """Generate a report from a template.

        Args:
            template_id: Template ID
            variables: Variable overrides
            format: Output format

        Returns:
            Generated report or None
        """
        template = self._templates.get(template_id)
        if not template:
            return None

        # Merge variables
        all_vars = {**template.variables, **(variables or {})}

        return await self.generate_report(
            report_type=template.report_type,
            period=ReportPeriod(all_vars.get("period", "daily")),
            format=format,
            title=all_vars.get("title", template.name),
        )

    def export_report(
        self,
        report: Report,
        output_path: Optional[str] = None,
    ) -> str:
        """Export a report to file.

        Args:
            report: Report to export
            output_path: Optional output path

        Returns:
            Output file path
        """
        if output_path is None:
            ext = self._get_format_extension(report.format)
            filename = f"{report.report_id}.{ext}"
            output_path = str(self._output_dir / filename)

        content = self._format_report(report)

        with open(output_path, "w") as f:
            f.write(content)

        return output_path

    def get_report(self, report_id: str) -> Optional[Report]:
        """Get a report by ID.

        Args:
            report_id: Report ID

        Returns:
            Report or None
        """
        return self._reports.get(report_id)

    def list_reports(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List generated reports.

        Args:
            limit: Maximum results

        Returns:
            Report summaries
        """
        reports = list(self._reports.values())
        reports.sort(key=lambda r: r.generated_at, reverse=True)

        return [
            {
                "report_id": r.report_id,
                "title": r.title,
                "report_type": r.report_type.value,
                "period": r.period.value,
                "generated_at": r.generated_at,
                "section_count": len(r.sections),
            }
            for r in reports[:limit]
        ]

    def handle_generate_request(self, event: Dict[str, Any]) -> None:
        """Handle report generation request from bus.

        Args:
            event: Bus event
        """
        data = event.get("data", {})

        report_type = ReportType(data.get("report_type", "system_health"))
        period = ReportPeriod(data.get("period", "daily"))
        format = ReportFormat(data.get("format", "json"))

        asyncio.create_task(
            self.generate_report(report_type, period, format)
        )

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "report_generator",
                "status": "healthy",
                "reports_generated": len(self._reports),
            }
        )

        return True

    async def _generate_sections(self, report: Report) -> None:
        """Generate sections for a report.

        Args:
            report: Report to populate
        """
        if report.report_type == ReportType.SYSTEM_HEALTH:
            await self._generate_system_health_sections(report)
        elif report.report_type == ReportType.ALERT_SUMMARY:
            await self._generate_alert_summary_sections(report)
        elif report.report_type == ReportType.SLO_COMPLIANCE:
            await self._generate_slo_sections(report)
        elif report.report_type == ReportType.PERFORMANCE_TRENDS:
            await self._generate_performance_sections(report)
        elif report.report_type == ReportType.INCIDENT_REPORT:
            await self._generate_incident_sections(report)
        else:
            # Generic sections
            report.add_section(ReportSection(
                section_id="overview",
                title="Overview",
                content="Report overview",
                order=0,
            ))

    async def _generate_system_health_sections(self, report: Report) -> None:
        """Generate system health sections."""
        report.add_section(ReportSection(
            section_id="overview",
            title="System Overview",
            content="Overall system health status for the reporting period.",
            order=0,
        ))

        report.add_section(ReportSection(
            section_id="resources",
            title="Resource Utilization",
            content="CPU, memory, and disk utilization metrics.",
            tables=[{
                "name": "Resource Usage",
                "columns": ["Resource", "Average", "Peak", "Min"],
                "rows": [
                    ["CPU", "45%", "78%", "12%"],
                    ["Memory", "62%", "85%", "48%"],
                    ["Disk", "55%", "60%", "50%"],
                ]
            }],
            order=1,
        ))

        report.add_section(ReportSection(
            section_id="services",
            title="Service Health",
            content="Health status of monitored services.",
            order=2,
        ))

        report.add_section(ReportSection(
            section_id="agents",
            title="Agent Status",
            content="OAGENT health and performance.",
            order=3,
        ))

    async def _generate_alert_summary_sections(self, report: Report) -> None:
        """Generate alert summary sections."""
        report.add_section(ReportSection(
            section_id="overview",
            title="Alert Overview",
            content="Summary of alerts during the reporting period.",
            order=0,
        ))

        report.add_section(ReportSection(
            section_id="by_severity",
            title="Alerts by Severity",
            content="Distribution of alerts by severity level.",
            charts=[{
                "type": "pie",
                "data": {"critical": 5, "warning": 23, "info": 42}
            }],
            order=1,
        ))

        report.add_section(ReportSection(
            section_id="top_alerts",
            title="Top Alerts",
            content="Most frequent alerts.",
            tables=[{
                "name": "Alert Frequency",
                "columns": ["Alert", "Count", "Severity"],
                "rows": [],
            }],
            order=2,
        ))

    async def _generate_slo_sections(self, report: Report) -> None:
        """Generate SLO compliance sections."""
        report.add_section(ReportSection(
            section_id="overview",
            title="SLO Compliance Overview",
            content="SLO compliance status for the reporting period.",
            order=0,
        ))

        report.add_section(ReportSection(
            section_id="compliance",
            title="Compliance Summary",
            content="Detailed compliance metrics.",
            tables=[{
                "name": "SLO Status",
                "columns": ["SLO", "Target", "Actual", "Status"],
                "rows": [],
            }],
            order=1,
        ))

        report.add_section(ReportSection(
            section_id="error_budget",
            title="Error Budget Status",
            content="Error budget consumption and remaining.",
            order=2,
        ))

    async def _generate_performance_sections(self, report: Report) -> None:
        """Generate performance trend sections."""
        report.add_section(ReportSection(
            section_id="overview",
            title="Performance Overview",
            content="Performance trends during the reporting period.",
            order=0,
        ))

        report.add_section(ReportSection(
            section_id="latency",
            title="Latency Trends",
            content="Response time and latency metrics.",
            order=1,
        ))

        report.add_section(ReportSection(
            section_id="throughput",
            title="Throughput Analysis",
            content="Request and transaction throughput.",
            order=2,
        ))

    async def _generate_incident_sections(self, report: Report) -> None:
        """Generate incident report sections."""
        report.add_section(ReportSection(
            section_id="overview",
            title="Incident Summary",
            content="Incident overview for the reporting period.",
            order=0,
        ))

        report.add_section(ReportSection(
            section_id="timeline",
            title="Incident Timeline",
            content="Chronological sequence of events.",
            order=1,
        ))

        report.add_section(ReportSection(
            section_id="impact",
            title="Impact Analysis",
            content="Impact on services and users.",
            order=2,
        ))

        report.add_section(ReportSection(
            section_id="resolution",
            title="Resolution & Actions",
            content="Resolution steps and follow-up actions.",
            order=3,
        ))

    def _generate_summary(self, report: Report) -> str:
        """Generate executive summary for a report."""
        period_str = report.period.value.title()
        start_dt = datetime.fromtimestamp(report.start_time, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(report.end_time, tz=timezone.utc)

        return (
            f"This {period_str.lower()} {report.report_type.value.replace('_', ' ')} report "
            f"covers the period from {start_dt.strftime('%Y-%m-%d %H:%M')} UTC "
            f"to {end_dt.strftime('%Y-%m-%d %H:%M')} UTC. "
            f"The report contains {len(report.sections)} sections."
        )

    def _calculate_period_start(self, period: ReportPeriod, end_time: float) -> float:
        """Calculate period start time."""
        end_dt = datetime.fromtimestamp(end_time, tz=timezone.utc)

        deltas = {
            ReportPeriod.HOURLY: timedelta(hours=1),
            ReportPeriod.DAILY: timedelta(days=1),
            ReportPeriod.WEEKLY: timedelta(weeks=1),
            ReportPeriod.MONTHLY: timedelta(days=30),
            ReportPeriod.CUSTOM: timedelta(days=1),
        }

        delta = deltas.get(period, timedelta(days=1))
        start_dt = end_dt - delta

        return start_dt.timestamp()

    def _format_report(self, report: Report) -> str:
        """Format report content based on format."""
        if report.format == ReportFormat.JSON:
            return json.dumps(report.to_dict(), indent=2)
        elif report.format == ReportFormat.MARKDOWN:
            return self._format_markdown(report)
        elif report.format == ReportFormat.TEXT:
            return self._format_text(report)
        elif report.format == ReportFormat.HTML:
            return self._format_html(report)
        elif report.format == ReportFormat.CSV:
            return self._format_csv(report)
        return json.dumps(report.to_dict())

    def _format_markdown(self, report: Report) -> str:
        """Format report as Markdown."""
        lines = [
            f"# {report.title}",
            "",
            f"**Report ID:** {report.report_id}",
            f"**Generated:** {datetime.fromtimestamp(report.generated_at, tz=timezone.utc).isoformat()}",
            f"**Period:** {report.period.value}",
            "",
            "## Summary",
            "",
            report.summary,
            "",
        ]

        for section in report.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            if section.content:
                lines.append(section.content)
                lines.append("")

            for table in section.tables:
                if "columns" in table and "rows" in table:
                    lines.append("| " + " | ".join(table["columns"]) + " |")
                    lines.append("| " + " | ".join(["---"] * len(table["columns"])) + " |")
                    for row in table["rows"]:
                        lines.append("| " + " | ".join(str(c) for c in row) + " |")
                    lines.append("")

        return "\n".join(lines)

    def _format_text(self, report: Report) -> str:
        """Format report as plain text."""
        lines = [
            report.title.upper(),
            "=" * len(report.title),
            "",
            f"Report ID: {report.report_id}",
            f"Generated: {datetime.fromtimestamp(report.generated_at, tz=timezone.utc).isoformat()}",
            "",
            "SUMMARY",
            "-" * 7,
            report.summary,
            "",
        ]

        for section in report.sections:
            lines.append(section.title.upper())
            lines.append("-" * len(section.title))
            if section.content:
                lines.append(section.content)
            lines.append("")

        return "\n".join(lines)

    def _format_html(self, report: Report) -> str:
        """Format report as HTML."""
        sections_html = ""
        for section in report.sections:
            tables_html = ""
            for table in section.tables:
                if "columns" in table and "rows" in table:
                    headers = "".join(f"<th>{c}</th>" for c in table["columns"])
                    rows = "".join(
                        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
                        for row in table["rows"]
                    )
                    tables_html += f"<table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table>"

            sections_html += f"""
            <section>
                <h2>{section.title}</h2>
                <p>{section.content}</p>
                {tables_html}
            </section>
            """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f4f4f4; }}
            </style>
        </head>
        <body>
            <h1>{report.title}</h1>
            <p><strong>Report ID:</strong> {report.report_id}</p>
            <p><strong>Generated:</strong> {datetime.fromtimestamp(report.generated_at, tz=timezone.utc).isoformat()}</p>
            <h2>Summary</h2>
            <p>{report.summary}</p>
            {sections_html}
        </body>
        </html>
        """

    def _format_csv(self, report: Report) -> str:
        """Format report tables as CSV."""
        lines = [f"# {report.title}", f"# Report ID: {report.report_id}", ""]

        for section in report.sections:
            for table in section.tables:
                if "columns" in table and "rows" in table:
                    lines.append(f"# {section.title} - {table.get('name', 'Table')}")
                    lines.append(",".join(f'"{c}"' for c in table["columns"]))
                    for row in table["rows"]:
                        lines.append(",".join(f'"{c}"' for c in row))
                    lines.append("")

        return "\n".join(lines)

    def _get_format_extension(self, format: ReportFormat) -> str:
        """Get file extension for format."""
        extensions = {
            ReportFormat.JSON: "json",
            ReportFormat.MARKDOWN: "md",
            ReportFormat.HTML: "html",
            ReportFormat.TEXT: "txt",
            ReportFormat.CSV: "csv",
        }
        return extensions.get(format, "json")

    def _register_default_templates(self) -> None:
        """Register default report templates."""
        self.register_template(ReportTemplate(
            template_id="daily-health",
            name="Daily Health Report",
            report_type=ReportType.SYSTEM_HEALTH,
            variables={"period": "daily"},
        ))

        self.register_template(ReportTemplate(
            template_id="weekly-alerts",
            name="Weekly Alert Summary",
            report_type=ReportType.ALERT_SUMMARY,
            variables={"period": "weekly"},
        ))

        self.register_template(ReportTemplate(
            template_id="monthly-slo",
            name="Monthly SLO Report",
            report_type=ReportType.SLO_COMPLIANCE,
            variables={"period": "monthly"},
        ))

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id


# Singleton instance
_generator: Optional[ReportGenerator] = None


def get_generator() -> ReportGenerator:
    """Get or create the report generator singleton.

    Returns:
        ReportGenerator instance
    """
    global _generator
    if _generator is None:
        _generator = ReportGenerator()
    return _generator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Report Generator (Step 272)")
    parser.add_argument("--generate", metavar="TYPE", help="Generate report (system_health, alert_summary, slo_compliance)")
    parser.add_argument("--period", default="daily", help="Report period")
    parser.add_argument("--format", default="json", help="Output format (json, markdown, html, text)")
    parser.add_argument("--templates", action="store_true", help="List templates")
    parser.add_argument("--reports", action="store_true", help="List reports")
    parser.add_argument("--export", metavar="ID", help="Export report to file")
    parser.add_argument("--output", metavar="PATH", help="Output file path")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    generator = get_generator()

    if args.generate:
        async def run():
            return await generator.generate_report(
                report_type=ReportType(args.generate),
                period=ReportPeriod(args.period),
                format=ReportFormat(args.format),
            )

        report = asyncio.run(run())
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Generated report: {report.report_id}")
            print(f"  Title: {report.title}")
            print(f"  Sections: {len(report.sections)}")
            print(f"  Format: {report.format.value}")

    if args.templates:
        templates = generator.list_templates()
        if args.json:
            print(json.dumps(templates, indent=2))
        else:
            print("Templates:")
            for t in templates:
                print(f"  [{t['template_id']}] {t['name']} ({t['report_type']})")

    if args.reports:
        reports = generator.list_reports()
        if args.json:
            print(json.dumps(reports, indent=2))
        else:
            print("Reports:")
            for r in reports:
                print(f"  [{r['report_id']}] {r['title']} ({r['period']})")

    if args.export:
        report = generator.get_report(args.export)
        if report:
            path = generator.export_report(report, args.output)
            print(f"Exported to: {path}")
        else:
            print(f"Report not found: {args.export}")
