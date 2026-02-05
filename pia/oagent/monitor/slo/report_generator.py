#!/usr/bin/env python3
"""
Report Generator - Step 268

Generates automated monitoring reports.

PBTSO Phase: DISTILL

Bus Topics:
- monitor.report.generate (emitted)
- monitor.report.complete (emitted)

Protocol: DKIN v30, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class ReportType(Enum):
    """Types of reports."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    INCIDENT = "incident"
    SLO = "slo"
    CAPACITY = "capacity"
    COST = "cost"
    CUSTOM = "custom"


class ReportFormat(Enum):
    """Report output formats."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"


@dataclass
class ReportSection:
    """Report section.

    Attributes:
        title: Section title
        content: Section content (can be string, dict, or list)
        section_type: Type of section (text, table, chart, metrics)
        order: Section order
    """
    title: str
    content: Any
    section_type: str = "text"
    order: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "section_type": self.section_type,
            "order": self.order,
        }


@dataclass
class Report:
    """Generated report.

    Attributes:
        report_id: Unique report ID
        report_type: Type of report
        title: Report title
        summary: Executive summary
        sections: Report sections
        period_start: Report period start
        period_end: Report period end
        generated_at: Generation timestamp
        metadata: Additional metadata
    """
    report_id: str
    report_type: ReportType
    title: str
    summary: str
    sections: List[ReportSection] = field(default_factory=list)
    period_start: float = 0.0
    period_end: float = 0.0
    generated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in sorted(self.sections, key=lambda x: x.order)],
            "period_start": self.period_start,
            "period_end": self.period_end,
            "generated_at": self.generated_at,
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        lines = [
            f"# {self.title}",
            "",
            f"**Report ID:** {self.report_id}",
            f"**Type:** {self.report_type.value}",
            f"**Period:** {datetime.fromtimestamp(self.period_start).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(self.period_end).strftime('%Y-%m-%d')}",
            f"**Generated:** {datetime.fromtimestamp(self.generated_at).strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            self.summary,
            "",
        ]

        for section in sorted(self.sections, key=lambda x: x.order):
            lines.append(f"## {section.title}")
            lines.append("")

            if section.section_type == "text":
                lines.append(str(section.content))
            elif section.section_type == "table":
                lines.extend(self._format_table(section.content))
            elif section.section_type == "metrics":
                lines.extend(self._format_metrics(section.content))
            elif section.section_type == "list":
                for item in section.content:
                    lines.append(f"- {item}")
            else:
                lines.append(str(section.content))

            lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Convert to HTML format."""
        sections_html = []
        for section in sorted(self.sections, key=lambda x: x.order):
            if section.section_type == "table":
                content = self._table_to_html(section.content)
            elif section.section_type == "metrics":
                content = self._metrics_to_html(section.content)
            elif section.section_type == "list":
                content = "<ul>" + "".join(f"<li>{item}</li>" for item in section.content) + "</ul>"
            else:
                content = f"<p>{section.content}</p>"

            sections_html.append(f"<section><h2>{section.title}</h2>{content}</section>")

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px; padding: 20px; background: #f0f0f0; border-radius: 8px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ color: #666; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <div class="meta">
        <p><strong>Report ID:</strong> {self.report_id}</p>
        <p><strong>Period:</strong> {datetime.fromtimestamp(self.period_start).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(self.period_end).strftime('%Y-%m-%d')}</p>
    </div>
    <section class="summary">
        <h2>Executive Summary</h2>
        <p>{self.summary}</p>
    </section>
    {''.join(sections_html)}
</body>
</html>"""

    def to_text(self) -> str:
        """Convert to plain text format."""
        lines = [
            "=" * 60,
            self.title.center(60),
            "=" * 60,
            "",
            f"Report ID: {self.report_id}",
            f"Type: {self.report_type.value}",
            f"Period: {datetime.fromtimestamp(self.period_start).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(self.period_end).strftime('%Y-%m-%d')}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
            self.summary,
            "",
        ]

        for section in sorted(self.sections, key=lambda x: x.order):
            lines.append(section.title.upper())
            lines.append("-" * 40)
            lines.append(str(section.content))
            lines.append("")

        return "\n".join(lines)

    def _format_table(self, content: List[Dict[str, Any]]) -> List[str]:
        """Format table for Markdown."""
        if not content:
            return ["*No data*"]

        headers = list(content[0].keys())
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]

        for row in content:
            values = [str(row.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(values) + " |")

        return lines

    def _format_metrics(self, content: Dict[str, Any]) -> List[str]:
        """Format metrics for Markdown."""
        lines = []
        for key, value in content.items():
            if isinstance(value, float):
                lines.append(f"- **{key}:** {value:.2f}")
            else:
                lines.append(f"- **{key}:** {value}")
        return lines

    def _table_to_html(self, content: List[Dict[str, Any]]) -> str:
        """Format table for HTML."""
        if not content:
            return "<p><em>No data</em></p>"

        headers = list(content[0].keys())
        header_row = "".join(f"<th>{h}</th>" for h in headers)
        rows = []
        for row in content:
            cells = "".join(f"<td>{row.get(h, '')}</td>" for h in headers)
            rows.append(f"<tr>{cells}</tr>")

        return f"<table><tr>{header_row}</tr>{''.join(rows)}</table>"

    def _metrics_to_html(self, content: Dict[str, Any]) -> str:
        """Format metrics for HTML."""
        metrics = []
        for key, value in content.items():
            if isinstance(value, float):
                formatted = f"{value:.2f}"
            else:
                formatted = str(value)
            metrics.append(f'<div class="metric"><div class="metric-value">{formatted}</div><div class="metric-label">{key}</div></div>')
        return "".join(metrics)


@dataclass
class ReportTemplate:
    """Report template definition.

    Attributes:
        name: Template name
        report_type: Report type
        title_template: Title template string
        sections: Section definitions
        data_sources: Data sources to query
    """
    name: str
    report_type: ReportType
    title_template: str
    sections: List[Dict[str, Any]] = field(default_factory=list)
    data_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ReportGenerator:
    """
    Generate automated monitoring reports.

    The generator:
    - Creates reports from templates
    - Supports multiple output formats
    - Integrates with monitoring data sources
    - Schedules periodic reports

    Example:
        generator = ReportGenerator()

        # Generate a daily report
        report = generator.generate_daily_report()
        print(report.to_markdown())

        # Generate custom report
        report = generator.generate(
            report_type=ReportType.SLO,
            title="SLO Compliance Report",
            sections=[
                ReportSection(title="Overview", content="...")
            ]
        )
    """

    BUS_TOPICS = {
        "generate": "monitor.report.generate",
        "complete": "monitor.report.complete",
    }

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

        # Templates
        self._templates: Dict[str, ReportTemplate] = {}
        self._register_default_templates()

        # Data providers
        self._data_providers: Dict[str, Callable[[], Any]] = {}

        # Bus path
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        report_type: ReportType,
        title: str,
        sections: List[ReportSection],
        period_start: Optional[float] = None,
        period_end: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Report:
        """Generate a report.

        Args:
            report_type: Type of report
            title: Report title
            sections: Report sections
            period_start: Period start timestamp
            period_end: Period end timestamp
            metadata: Additional metadata

        Returns:
            Generated report
        """
        now = time.time()
        report_id = str(uuid.uuid4())[:8]

        # Default period to last 24 hours
        if period_end is None:
            period_end = now
        if period_start is None:
            period_start = period_end - 86400

        # Calculate summary
        summary = self._generate_summary(sections)

        report = Report(
            report_id=report_id,
            report_type=report_type,
            title=title,
            summary=summary,
            sections=sections,
            period_start=period_start,
            period_end=period_end,
            metadata=metadata or {},
        )

        # Emit event
        self._emit_bus_event(
            self.BUS_TOPICS["complete"],
            {
                "report_id": report_id,
                "report_type": report_type.value,
                "title": title,
            }
        )

        return report

    def generate_from_template(
        self,
        template_name: str,
        period_start: Optional[float] = None,
        period_end: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Report]:
        """Generate report from template.

        Args:
            template_name: Template name
            period_start: Period start
            period_end: Period end
            context: Additional context for template

        Returns:
            Generated report or None
        """
        template = self._templates.get(template_name)
        if not template:
            return None

        context = context or {}
        now = datetime.now()

        # Generate title
        title = template.title_template.format(
            date=now.strftime("%Y-%m-%d"),
            month=now.strftime("%B %Y"),
            week=now.strftime("Week %W"),
            **context
        )

        # Build sections
        sections = []
        for i, section_def in enumerate(template.sections):
            section_title = section_def.get("title", f"Section {i + 1}")
            section_type = section_def.get("type", "text")
            data_source = section_def.get("data_source")

            if data_source and data_source in self._data_providers:
                content = self._data_providers[data_source]()
            else:
                content = section_def.get("content", "")

            sections.append(ReportSection(
                title=section_title,
                content=content,
                section_type=section_type,
                order=i,
            ))

        return self.generate(
            report_type=template.report_type,
            title=title,
            sections=sections,
            period_start=period_start,
            period_end=period_end,
            metadata={"template": template_name},
        )

    def generate_daily_report(
        self,
        date: Optional[datetime] = None
    ) -> Report:
        """Generate daily operations report.

        Args:
            date: Report date (default: yesterday)

        Returns:
            Daily report
        """
        if date is None:
            date = datetime.now() - timedelta(days=1)

        period_start = date.replace(hour=0, minute=0, second=0).timestamp()
        period_end = period_start + 86400

        sections = [
            ReportSection(
                title="Health Summary",
                content={
                    "services_healthy": 0,
                    "services_degraded": 0,
                    "services_down": 0,
                    "uptime_percent": 99.9,
                },
                section_type="metrics",
                order=0,
            ),
            ReportSection(
                title="Incidents",
                content="No incidents reported.",
                section_type="text",
                order=1,
            ),
            ReportSection(
                title="Key Metrics",
                content={
                    "avg_response_time_ms": 0.0,
                    "error_rate_percent": 0.0,
                    "requests_total": 0,
                },
                section_type="metrics",
                order=2,
            ),
            ReportSection(
                title="Alerts Summary",
                content=[],
                section_type="table",
                order=3,
            ),
        ]

        return self.generate(
            report_type=ReportType.DAILY,
            title=f"Daily Operations Report - {date.strftime('%Y-%m-%d')}",
            sections=sections,
            period_start=period_start,
            period_end=period_end,
        )

    def generate_weekly_report(
        self,
        week_start: Optional[datetime] = None
    ) -> Report:
        """Generate weekly summary report.

        Args:
            week_start: Week start date

        Returns:
            Weekly report
        """
        if week_start is None:
            today = datetime.now()
            week_start = today - timedelta(days=today.weekday() + 7)

        period_start = week_start.replace(hour=0, minute=0, second=0).timestamp()
        period_end = period_start + (7 * 86400)

        sections = [
            ReportSection(
                title="Week at a Glance",
                content={
                    "total_uptime_percent": 99.95,
                    "incidents_count": 0,
                    "alerts_count": 0,
                    "deployments_count": 0,
                },
                section_type="metrics",
                order=0,
            ),
            ReportSection(
                title="Daily Breakdown",
                content=[],
                section_type="table",
                order=1,
            ),
            ReportSection(
                title="Top Issues",
                content=[],
                section_type="list",
                order=2,
            ),
            ReportSection(
                title="Recommendations",
                content="No recommendations at this time.",
                section_type="text",
                order=3,
            ),
        ]

        return self.generate(
            report_type=ReportType.WEEKLY,
            title=f"Weekly Summary Report - Week of {week_start.strftime('%Y-%m-%d')}",
            sections=sections,
            period_start=period_start,
            period_end=period_end,
        )

    def generate_slo_report(
        self,
        period_days: int = 30
    ) -> Report:
        """Generate SLO compliance report.

        Args:
            period_days: Period in days

        Returns:
            SLO report
        """
        now = time.time()
        period_start = now - (period_days * 86400)

        sections = [
            ReportSection(
                title="SLO Compliance Summary",
                content={
                    "total_slos": 0,
                    "compliant": 0,
                    "at_risk": 0,
                    "breached": 0,
                    "compliance_rate": 100.0,
                },
                section_type="metrics",
                order=0,
            ),
            ReportSection(
                title="SLO Details",
                content=[],
                section_type="table",
                order=1,
            ),
            ReportSection(
                title="Error Budget Status",
                content=[],
                section_type="table",
                order=2,
            ),
            ReportSection(
                title="Breaches",
                content="No breaches in reporting period.",
                section_type="text",
                order=3,
            ),
        ]

        return self.generate(
            report_type=ReportType.SLO,
            title=f"SLO Compliance Report - {period_days} Day Period",
            sections=sections,
            period_start=period_start,
            period_end=now,
        )

    def generate_capacity_report(self) -> Report:
        """Generate capacity planning report.

        Returns:
            Capacity report
        """
        now = time.time()
        period_start = now - (30 * 86400)

        sections = [
            ReportSection(
                title="Capacity Overview",
                content={
                    "total_resources": 0,
                    "critical": 0,
                    "warning": 0,
                    "healthy": 0,
                },
                section_type="metrics",
                order=0,
            ),
            ReportSection(
                title="Resource Utilization",
                content=[],
                section_type="table",
                order=1,
            ),
            ReportSection(
                title="Scaling Recommendations",
                content=[],
                section_type="table",
                order=2,
            ),
            ReportSection(
                title="30-Day Forecast",
                content=[],
                section_type="table",
                order=3,
            ),
        ]

        return self.generate(
            report_type=ReportType.CAPACITY,
            title="Capacity Planning Report",
            sections=sections,
            period_start=period_start,
            period_end=now,
        )

    def generate_cost_report(
        self,
        period_days: int = 30
    ) -> Report:
        """Generate cost analysis report.

        Args:
            period_days: Period in days

        Returns:
            Cost report
        """
        now = time.time()
        period_start = now - (period_days * 86400)

        sections = [
            ReportSection(
                title="Cost Summary",
                content={
                    "total_cost": 0.0,
                    "budget": 0.0,
                    "variance": 0.0,
                    "variance_percent": 0.0,
                },
                section_type="metrics",
                order=0,
            ),
            ReportSection(
                title="Cost by Category",
                content=[],
                section_type="table",
                order=1,
            ),
            ReportSection(
                title="Cost by Resource",
                content=[],
                section_type="table",
                order=2,
            ),
            ReportSection(
                title="Cost Optimization Opportunities",
                content=[],
                section_type="list",
                order=3,
            ),
        ]

        return self.generate(
            report_type=ReportType.COST,
            title=f"Cost Analysis Report - {period_days} Day Period",
            sections=sections,
            period_start=period_start,
            period_end=now,
        )

    def register_template(self, template: ReportTemplate) -> None:
        """Register a report template.

        Args:
            template: Template to register
        """
        self._templates[template.name] = template

    def register_data_provider(
        self,
        name: str,
        provider: Callable[[], Any]
    ) -> None:
        """Register a data provider.

        Args:
            name: Provider name
            provider: Provider function
        """
        self._data_providers[name] = provider

    def save_report(
        self,
        report: Report,
        format: ReportFormat = ReportFormat.MARKDOWN
    ) -> str:
        """Save report to file.

        Args:
            report: Report to save
            format: Output format

        Returns:
            File path
        """
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report.report_type.value}_{date_str}_{report.report_id}"

        if format == ReportFormat.JSON:
            content = json.dumps(report.to_dict(), indent=2)
            ext = "json"
        elif format == ReportFormat.MARKDOWN:
            content = report.to_markdown()
            ext = "md"
        elif format == ReportFormat.HTML:
            content = report.to_html()
            ext = "html"
        else:
            content = report.to_text()
            ext = "txt"

        filepath = self._output_dir / f"{filename}.{ext}"
        filepath.write_text(content)

        return str(filepath)

    def list_templates(self) -> List[str]:
        """List available templates.

        Returns:
            Template names
        """
        return list(self._templates.keys())

    def _register_default_templates(self) -> None:
        """Register default templates."""
        self.register_template(ReportTemplate(
            name="daily",
            report_type=ReportType.DAILY,
            title_template="Daily Operations Report - {date}",
            sections=[
                {"title": "Health Summary", "type": "metrics", "data_source": "health_summary"},
                {"title": "Incidents", "type": "table", "data_source": "incidents"},
                {"title": "Alerts", "type": "table", "data_source": "alerts"},
            ],
        ))

        self.register_template(ReportTemplate(
            name="weekly",
            report_type=ReportType.WEEKLY,
            title_template="Weekly Summary Report - {week}",
            sections=[
                {"title": "Week Overview", "type": "metrics", "data_source": "week_summary"},
                {"title": "Daily Breakdown", "type": "table", "data_source": "daily_breakdown"},
                {"title": "Top Issues", "type": "list", "data_source": "top_issues"},
            ],
        ))

        self.register_template(ReportTemplate(
            name="slo",
            report_type=ReportType.SLO,
            title_template="SLO Compliance Report - {month}",
            sections=[
                {"title": "Compliance Summary", "type": "metrics", "data_source": "slo_summary"},
                {"title": "SLO Details", "type": "table", "data_source": "slo_details"},
                {"title": "Error Budgets", "type": "table", "data_source": "error_budgets"},
            ],
        ))

    def _generate_summary(self, sections: List[ReportSection]) -> str:
        """Generate executive summary from sections.

        Args:
            sections: Report sections

        Returns:
            Summary text
        """
        # Simple summary generation
        summary_parts = []

        for section in sections:
            if section.section_type == "metrics" and isinstance(section.content, dict):
                for key, value in section.content.items():
                    if "percent" in key.lower() or "rate" in key.lower():
                        if isinstance(value, (int, float)):
                            summary_parts.append(f"{key}: {value:.1f}%")

        if summary_parts:
            return "Key metrics: " + ", ".join(summary_parts[:5])

        return "Report generated successfully."

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to bus."""
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
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

        return event_id


# Singleton
_generator: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """Get or create the report generator singleton."""
    global _generator
    if _generator is None:
        _generator = ReportGenerator()
    return _generator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Report Generator (Step 268)")
    parser.add_argument("--daily", action="store_true", help="Generate daily report")
    parser.add_argument("--weekly", action="store_true", help="Generate weekly report")
    parser.add_argument("--slo", action="store_true", help="Generate SLO report")
    parser.add_argument("--capacity", action="store_true", help="Generate capacity report")
    parser.add_argument("--cost", action="store_true", help="Generate cost report")
    parser.add_argument("--format", choices=["json", "markdown", "html", "text"], default="markdown")
    parser.add_argument("--save", action="store_true", help="Save report to file")
    parser.add_argument("--templates", action="store_true", help="List templates")

    args = parser.parse_args()

    generator = get_report_generator()

    if args.templates:
        print("Available Templates:")
        for name in generator.list_templates():
            print(f"  - {name}")
        exit(0)

    report = None

    if args.daily:
        report = generator.generate_daily_report()
    elif args.weekly:
        report = generator.generate_weekly_report()
    elif args.slo:
        report = generator.generate_slo_report()
    elif args.capacity:
        report = generator.generate_capacity_report()
    elif args.cost:
        report = generator.generate_cost_report()

    if report:
        format_map = {
            "json": ReportFormat.JSON,
            "markdown": ReportFormat.MARKDOWN,
            "html": ReportFormat.HTML,
            "text": ReportFormat.TEXT,
        }
        fmt = format_map[args.format]

        if args.save:
            filepath = generator.save_report(report, fmt)
            print(f"Report saved to: {filepath}")
        else:
            if fmt == ReportFormat.JSON:
                print(json.dumps(report.to_dict(), indent=2))
            elif fmt == ReportFormat.MARKDOWN:
                print(report.to_markdown())
            elif fmt == ReportFormat.HTML:
                print(report.to_html())
            else:
                print(report.to_text())
