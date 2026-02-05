#!/usr/bin/env python3
"""
generator.py - Deployment Report Generator (Step 222)

PBTSO Phase: DISTILL
A2A Integration: Generates deployment reports via deploy.reports.*

Provides:
- ReportType: Types of deployment reports
- ReportFormat: Output formats
- ReportSection: Report section data
- DeploymentReport: Complete report
- DeploymentReportGenerator: Main generator class

Bus Topics:
- deploy.reports.generate
- deploy.reports.complete
- deploy.reports.export

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ==============================================================================
# Bus Emission Helper with File Locking
# ==============================================================================

def _get_bus_path() -> Path:
    """Get the bus event file path."""
    pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    kind: str = "event",
    level: str = "info",
    actor: str = "report-generator"
) -> str:
    """Emit an event to the Pluribus bus with file locking."""
    bus_path = _get_bus_path()
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat() + "Z",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }

    try:
        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class ReportType(Enum):
    """Types of deployment reports."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    DEPLOYMENT = "deployment"
    SERVICE = "service"
    ENVIRONMENT = "environment"
    INCIDENT = "incident"
    CUSTOM = "custom"


class ReportFormat(Enum):
    """Report output formats."""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"


class ReportStatus(Enum):
    """Report generation status."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ReportSection:
    """
    Report section data.

    Attributes:
        section_id: Unique section identifier
        title: Section title
        content: Section content
        data: Structured data for section
        order: Section order
        charts: Chart definitions
    """
    section_id: str
    title: str
    content: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    order: int = 0
    charts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentSummary:
    """
    Deployment summary for reports.

    Attributes:
        total_deployments: Total number of deployments
        successful: Successful deployments
        failed: Failed deployments
        rollbacks: Rollback count
        avg_duration_ms: Average deployment duration
        services: List of services deployed
        environments: Target environments
    """
    total_deployments: int = 0
    successful: int = 0
    failed: int = 0
    rollbacks: int = 0
    avg_duration_ms: float = 0.0
    services: List[str] = field(default_factory=list)
    environments: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def success_rate(self) -> float:
        if self.total_deployments == 0:
            return 0.0
        return (self.successful / self.total_deployments) * 100


@dataclass
class DeploymentReport:
    """
    Complete deployment report.

    Attributes:
        report_id: Unique report identifier
        report_type: Type of report
        title: Report title
        description: Report description
        sections: Report sections
        summary: Deployment summary
        start_date: Report start date
        end_date: Report end date
        generated_at: Generation timestamp
        status: Report status
        format: Output format
        file_path: Output file path
    """
    report_id: str
    report_type: ReportType = ReportType.DAILY
    title: str = ""
    description: str = ""
    sections: List[ReportSection] = field(default_factory=list)
    summary: DeploymentSummary = field(default_factory=DeploymentSummary)
    start_date: str = ""
    end_date: str = ""
    generated_at: float = field(default_factory=time.time)
    status: ReportStatus = ReportStatus.PENDING
    format: ReportFormat = ReportFormat.JSON
    file_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "title": self.title,
            "description": self.description,
            "sections": [s.to_dict() for s in self.sections],
            "summary": self.summary.to_dict(),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "generated_at": self.generated_at,
            "status": self.status.value,
            "format": self.format.value,
            "file_path": self.file_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentReport":
        data = dict(data)
        if "report_type" in data:
            data["report_type"] = ReportType(data["report_type"])
        if "status" in data:
            data["status"] = ReportStatus(data["status"])
        if "format" in data:
            data["format"] = ReportFormat(data["format"])
        if "sections" in data:
            data["sections"] = [
                ReportSection(**s) if isinstance(s, dict) else s
                for s in data["sections"]
            ]
        if "summary" in data and isinstance(data["summary"], dict):
            data["summary"] = DeploymentSummary(**data["summary"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==============================================================================
# Deployment Report Generator (Step 222)
# ==============================================================================

class DeploymentReportGenerator:
    """
    Deployment Report Generator - generates comprehensive deployment reports.

    PBTSO Phase: DISTILL

    Responsibilities:
    - Generate daily/weekly/monthly deployment reports
    - Create deployment-specific reports
    - Aggregate metrics and trends
    - Export reports in multiple formats
    - Schedule automatic report generation

    Example:
        >>> generator = DeploymentReportGenerator()
        >>> report = generator.generate_daily_report()
        >>> generator.export(report, ReportFormat.HTML)
    """

    BUS_TOPICS = {
        "generate": "deploy.reports.generate",
        "complete": "deploy.reports.complete",
        "export": "deploy.reports.export",
        "failed": "deploy.reports.failed",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "report-generator",
    ):
        """
        Initialize the report generator.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "reports"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        self._reports: Dict[str, DeploymentReport] = {}
        self._load_reports()

    def generate_daily_report(
        self,
        date: Optional[datetime] = None,
        services: Optional[List[str]] = None,
    ) -> DeploymentReport:
        """
        Generate a daily deployment report.

        Args:
            date: Date for report (defaults to today)
            services: Filter by services

        Returns:
            Generated DeploymentReport
        """
        if date is None:
            date = datetime.now(timezone.utc)

        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)

        return self._generate_report(
            report_type=ReportType.DAILY,
            title=f"Daily Deployment Report - {start_date.strftime('%Y-%m-%d')}",
            start_date=start_date,
            end_date=end_date,
            services=services,
        )

    def generate_weekly_report(
        self,
        week_start: Optional[datetime] = None,
        services: Optional[List[str]] = None,
    ) -> DeploymentReport:
        """
        Generate a weekly deployment report.

        Args:
            week_start: Start of week (defaults to current week)
            services: Filter by services

        Returns:
            Generated DeploymentReport
        """
        if week_start is None:
            today = datetime.now(timezone.utc)
            week_start = today - timedelta(days=today.weekday())

        start_date = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=7)

        return self._generate_report(
            report_type=ReportType.WEEKLY,
            title=f"Weekly Deployment Report - Week of {start_date.strftime('%Y-%m-%d')}",
            start_date=start_date,
            end_date=end_date,
            services=services,
        )

    def generate_monthly_report(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
        services: Optional[List[str]] = None,
    ) -> DeploymentReport:
        """
        Generate a monthly deployment report.

        Args:
            year: Year for report
            month: Month for report
            services: Filter by services

        Returns:
            Generated DeploymentReport
        """
        now = datetime.now(timezone.utc)
        if year is None:
            year = now.year
        if month is None:
            month = now.month

        start_date = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end_date = datetime(year, month + 1, 1, tzinfo=timezone.utc)

        return self._generate_report(
            report_type=ReportType.MONTHLY,
            title=f"Monthly Deployment Report - {start_date.strftime('%B %Y')}",
            start_date=start_date,
            end_date=end_date,
            services=services,
        )

    def generate_deployment_report(
        self,
        deployment_id: str,
    ) -> DeploymentReport:
        """
        Generate a report for a specific deployment.

        Args:
            deployment_id: Deployment identifier

        Returns:
            Generated DeploymentReport
        """
        report_id = f"report-{uuid.uuid4().hex[:12]}"

        report = DeploymentReport(
            report_id=report_id,
            report_type=ReportType.DEPLOYMENT,
            title=f"Deployment Report - {deployment_id}",
            status=ReportStatus.GENERATING,
        )

        _emit_bus_event(
            self.BUS_TOPICS["generate"],
            {"report_id": report_id, "type": "deployment", "deployment_id": deployment_id},
            actor=self.actor_id,
        )

        try:
            # Build sections
            sections = [
                ReportSection(
                    section_id="overview",
                    title="Deployment Overview",
                    order=1,
                    data={"deployment_id": deployment_id},
                ),
                ReportSection(
                    section_id="timeline",
                    title="Deployment Timeline",
                    order=2,
                    data={"phases": []},
                ),
                ReportSection(
                    section_id="metrics",
                    title="Performance Metrics",
                    order=3,
                    data={},
                ),
                ReportSection(
                    section_id="issues",
                    title="Issues and Warnings",
                    order=4,
                    data={"issues": []},
                ),
            ]

            report.sections = sections
            report.status = ReportStatus.COMPLETE
            report.generated_at = time.time()

            self._reports[report_id] = report
            self._save_report(report)

            _emit_bus_event(
                self.BUS_TOPICS["complete"],
                {"report_id": report_id, "status": "complete"},
                actor=self.actor_id,
            )

        except Exception as e:
            report.status = ReportStatus.FAILED
            report.metadata["error"] = str(e)

            _emit_bus_event(
                self.BUS_TOPICS["failed"],
                {"report_id": report_id, "error": str(e)},
                level="error",
                actor=self.actor_id,
            )

        return report

    def generate_service_report(
        self,
        service_name: str,
        days: int = 30,
    ) -> DeploymentReport:
        """
        Generate a report for a specific service.

        Args:
            service_name: Service name
            days: Number of days to include

        Returns:
            Generated DeploymentReport
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        return self._generate_report(
            report_type=ReportType.SERVICE,
            title=f"Service Deployment Report - {service_name}",
            start_date=start_date,
            end_date=end_date,
            services=[service_name],
        )

    def _generate_report(
        self,
        report_type: ReportType,
        title: str,
        start_date: datetime,
        end_date: datetime,
        services: Optional[List[str]] = None,
    ) -> DeploymentReport:
        """Generate a report."""
        report_id = f"report-{uuid.uuid4().hex[:12]}"

        report = DeploymentReport(
            report_id=report_id,
            report_type=report_type,
            title=title,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            status=ReportStatus.GENERATING,
            metadata={"services_filter": services or []},
        )

        _emit_bus_event(
            self.BUS_TOPICS["generate"],
            {
                "report_id": report_id,
                "type": report_type.value,
                "start_date": report.start_date,
                "end_date": report.end_date,
            },
            actor=self.actor_id,
        )

        try:
            # Build summary
            summary = self._build_summary(start_date, end_date, services)
            report.summary = summary

            # Build sections
            sections = self._build_sections(start_date, end_date, services, summary)
            report.sections = sections

            report.status = ReportStatus.COMPLETE
            report.generated_at = time.time()

            self._reports[report_id] = report
            self._save_report(report)

            _emit_bus_event(
                self.BUS_TOPICS["complete"],
                {
                    "report_id": report_id,
                    "status": "complete",
                    "sections": len(sections),
                    "summary": summary.to_dict(),
                },
                actor=self.actor_id,
            )

        except Exception as e:
            report.status = ReportStatus.FAILED
            report.metadata["error"] = str(e)

            _emit_bus_event(
                self.BUS_TOPICS["failed"],
                {"report_id": report_id, "error": str(e)},
                level="error",
                actor=self.actor_id,
            )

        return report

    def _build_summary(
        self,
        start_date: datetime,
        end_date: datetime,
        services: Optional[List[str]],
    ) -> DeploymentSummary:
        """Build deployment summary."""
        # In a real implementation, this would query metrics and history
        # For now, return placeholder data
        return DeploymentSummary(
            total_deployments=0,
            successful=0,
            failed=0,
            rollbacks=0,
            avg_duration_ms=0.0,
            services=services or [],
            environments=["staging", "prod"],
        )

    def _build_sections(
        self,
        start_date: datetime,
        end_date: datetime,
        services: Optional[List[str]],
        summary: DeploymentSummary,
    ) -> List[ReportSection]:
        """Build report sections."""
        sections = []

        # Executive Summary
        sections.append(ReportSection(
            section_id="executive-summary",
            title="Executive Summary",
            order=1,
            content=self._generate_executive_summary(summary),
            data=summary.to_dict(),
        ))

        # Deployment Statistics
        sections.append(ReportSection(
            section_id="statistics",
            title="Deployment Statistics",
            order=2,
            data={
                "total": summary.total_deployments,
                "successful": summary.successful,
                "failed": summary.failed,
                "success_rate": summary.success_rate,
            },
            charts=[
                {"type": "pie", "data": "deployment_status"},
                {"type": "line", "data": "deployments_over_time"},
            ],
        ))

        # Performance Metrics
        sections.append(ReportSection(
            section_id="performance",
            title="Performance Metrics",
            order=3,
            data={
                "avg_duration_ms": summary.avg_duration_ms,
            },
            charts=[
                {"type": "bar", "data": "duration_by_service"},
            ],
        ))

        # Service Breakdown
        if summary.services:
            sections.append(ReportSection(
                section_id="services",
                title="Service Breakdown",
                order=4,
                data={"services": summary.services},
            ))

        # Environment Summary
        sections.append(ReportSection(
            section_id="environments",
            title="Environment Summary",
            order=5,
            data={"environments": summary.environments},
        ))

        # Issues and Incidents
        sections.append(ReportSection(
            section_id="issues",
            title="Issues and Incidents",
            order=6,
            data={"issues": [], "incidents": []},
        ))

        # Recommendations
        sections.append(ReportSection(
            section_id="recommendations",
            title="Recommendations",
            order=7,
            content=self._generate_recommendations(summary),
        ))

        return sections

    def _generate_executive_summary(self, summary: DeploymentSummary) -> str:
        """Generate executive summary text."""
        return f"""
During this reporting period:
- Total deployments: {summary.total_deployments}
- Successful: {summary.successful} ({summary.success_rate:.1f}%)
- Failed: {summary.failed}
- Rollbacks: {summary.rollbacks}
- Average duration: {summary.avg_duration_ms:.0f}ms
        """.strip()

    def _generate_recommendations(self, summary: DeploymentSummary) -> str:
        """Generate recommendations based on summary."""
        recommendations = []

        if summary.success_rate < 95:
            recommendations.append(
                "Consider reviewing failed deployments to identify common patterns."
            )

        if summary.rollbacks > 0:
            recommendations.append(
                f"Investigate {summary.rollbacks} rollbacks to improve deployment reliability."
            )

        if summary.avg_duration_ms > 300000:  # 5 minutes
            recommendations.append(
                "Deployment duration is high. Consider optimizing build and deploy steps."
            )

        if not recommendations:
            recommendations.append("Deployments are performing well. Keep up the good work!")

        return "\n".join(f"- {r}" for r in recommendations)

    def export(
        self,
        report: DeploymentReport,
        format: ReportFormat = ReportFormat.JSON,
    ) -> str:
        """
        Export a report to file.

        Args:
            report: Report to export
            format: Output format

        Returns:
            File path
        """
        filename = f"{report.report_id}.{format.value}"
        file_path = self.state_dir / filename

        if format == ReportFormat.JSON:
            content = json.dumps(report.to_dict(), indent=2)
        elif format == ReportFormat.MARKDOWN:
            content = self._to_markdown(report)
        elif format == ReportFormat.HTML:
            content = self._to_html(report)
        elif format == ReportFormat.TEXT:
            content = self._to_text(report)
        else:
            content = json.dumps(report.to_dict(), indent=2)

        with open(file_path, "w") as f:
            f.write(content)

        report.file_path = str(file_path)
        report.format = format

        _emit_bus_event(
            self.BUS_TOPICS["export"],
            {
                "report_id": report.report_id,
                "format": format.value,
                "file_path": str(file_path),
            },
            actor=self.actor_id,
        )

        return str(file_path)

    def _to_markdown(self, report: DeploymentReport) -> str:
        """Convert report to Markdown."""
        lines = [
            f"# {report.title}",
            "",
            f"*Generated: {datetime.fromtimestamp(report.generated_at).isoformat()}*",
            "",
            f"Period: {report.start_date} to {report.end_date}",
            "",
        ]

        for section in sorted(report.sections, key=lambda s: s.order):
            lines.append(f"## {section.title}")
            lines.append("")
            if section.content:
                lines.append(section.content)
                lines.append("")
            if section.data:
                lines.append("```json")
                lines.append(json.dumps(section.data, indent=2))
                lines.append("```")
                lines.append("")

        return "\n".join(lines)

    def _to_html(self, report: DeploymentReport) -> str:
        """Convert report to HTML."""
        sections_html = ""
        for section in sorted(report.sections, key=lambda s: s.order):
            sections_html += f"""
            <section>
                <h2>{section.title}</h2>
                <div class="content">{section.content}</div>
                <pre>{json.dumps(section.data, indent=2)}</pre>
            </section>
            """

        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
        section {{ margin: 20px 0; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
        .meta {{ color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>{report.title}</h1>
    <p class="meta">Generated: {datetime.fromtimestamp(report.generated_at).isoformat()}</p>
    <p class="meta">Period: {report.start_date} to {report.end_date}</p>
    {sections_html}
</body>
</html>
        """.strip()

    def _to_text(self, report: DeploymentReport) -> str:
        """Convert report to plain text."""
        lines = [
            "=" * 60,
            report.title.center(60),
            "=" * 60,
            "",
            f"Generated: {datetime.fromtimestamp(report.generated_at).isoformat()}",
            f"Period: {report.start_date} to {report.end_date}",
            "",
        ]

        for section in sorted(report.sections, key=lambda s: s.order):
            lines.append("-" * 40)
            lines.append(section.title)
            lines.append("-" * 40)
            if section.content:
                lines.append(section.content)
            lines.append("")

        return "\n".join(lines)

    def get_report(self, report_id: str) -> Optional[DeploymentReport]:
        """Get a report by ID."""
        return self._reports.get(report_id)

    def list_reports(
        self,
        report_type: Optional[ReportType] = None,
        limit: int = 100,
    ) -> List[DeploymentReport]:
        """List reports."""
        reports = list(self._reports.values())

        if report_type:
            reports = [r for r in reports if r.report_type == report_type]

        reports.sort(key=lambda r: r.generated_at, reverse=True)
        return reports[:limit]

    def delete_report(self, report_id: str) -> bool:
        """Delete a report."""
        if report_id not in self._reports:
            return False

        report = self._reports[report_id]
        if report.file_path and Path(report.file_path).exists():
            Path(report.file_path).unlink()

        del self._reports[report_id]

        report_file = self.state_dir / f"{report_id}.json"
        if report_file.exists():
            report_file.unlink()

        return True

    def _save_report(self, report: DeploymentReport) -> None:
        """Save report to disk."""
        report_file = self.state_dir / f"{report.report_id}.json"
        with open(report_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(report.to_dict(), f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_reports(self) -> None:
        """Load reports from disk."""
        for report_file in self.state_dir.glob("report-*.json"):
            try:
                with open(report_file, "r") as f:
                    data = json.load(f)
                    report = DeploymentReport.from_dict(data)
                    self._reports[report.report_id] = report
            except (json.JSONDecodeError, IOError):
                pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for report generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Deployment Report Generator (Step 222)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # daily command
    daily_parser = subparsers.add_parser("daily", help="Generate daily report")
    daily_parser.add_argument("--date", "-d", help="Date (YYYY-MM-DD)")
    daily_parser.add_argument("--format", "-f", default="json",
                              choices=["json", "markdown", "html", "text"])
    daily_parser.add_argument("--json", action="store_true", help="JSON output")

    # weekly command
    weekly_parser = subparsers.add_parser("weekly", help="Generate weekly report")
    weekly_parser.add_argument("--week-start", help="Week start date (YYYY-MM-DD)")
    weekly_parser.add_argument("--format", "-f", default="json",
                               choices=["json", "markdown", "html", "text"])
    weekly_parser.add_argument("--json", action="store_true", help="JSON output")

    # monthly command
    monthly_parser = subparsers.add_parser("monthly", help="Generate monthly report")
    monthly_parser.add_argument("--year", type=int, help="Year")
    monthly_parser.add_argument("--month", type=int, help="Month")
    monthly_parser.add_argument("--format", "-f", default="json",
                                choices=["json", "markdown", "html", "text"])
    monthly_parser.add_argument("--json", action="store_true", help="JSON output")

    # deployment command
    deploy_parser = subparsers.add_parser("deployment", help="Generate deployment report")
    deploy_parser.add_argument("deployment_id", help="Deployment ID")
    deploy_parser.add_argument("--format", "-f", default="json",
                               choices=["json", "markdown", "html", "text"])
    deploy_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List reports")
    list_parser.add_argument("--type", "-t", help="Filter by type")
    list_parser.add_argument("--limit", "-l", type=int, default=10, help="Limit results")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # get command
    get_parser = subparsers.add_parser("get", help="Get a report")
    get_parser.add_argument("report_id", help="Report ID")
    get_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    generator = DeploymentReportGenerator()

    if args.command == "daily":
        date = None
        if args.date:
            date = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc)

        report = generator.generate_daily_report(date=date)
        file_path = generator.export(report, ReportFormat(args.format))

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Generated: {report.report_id}")
            print(f"  Status: {report.status.value}")
            print(f"  File: {file_path}")

        return 0

    elif args.command == "weekly":
        week_start = None
        if args.week_start:
            week_start = datetime.fromisoformat(args.week_start).replace(tzinfo=timezone.utc)

        report = generator.generate_weekly_report(week_start=week_start)
        file_path = generator.export(report, ReportFormat(args.format))

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Generated: {report.report_id}")
            print(f"  Status: {report.status.value}")
            print(f"  File: {file_path}")

        return 0

    elif args.command == "monthly":
        report = generator.generate_monthly_report(year=args.year, month=args.month)
        file_path = generator.export(report, ReportFormat(args.format))

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Generated: {report.report_id}")
            print(f"  Status: {report.status.value}")
            print(f"  File: {file_path}")

        return 0

    elif args.command == "deployment":
        report = generator.generate_deployment_report(args.deployment_id)
        file_path = generator.export(report, ReportFormat(args.format))

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Generated: {report.report_id}")
            print(f"  Status: {report.status.value}")
            print(f"  File: {file_path}")

        return 0

    elif args.command == "list":
        report_type = ReportType(args.type) if args.type else None
        reports = generator.list_reports(report_type=report_type, limit=args.limit)

        if args.json:
            print(json.dumps([r.to_dict() for r in reports], indent=2))
        else:
            for r in reports:
                print(f"{r.report_id} ({r.report_type.value}) - {r.status.value}")

        return 0

    elif args.command == "get":
        report = generator.get_report(args.report_id)
        if not report:
            print(f"Report not found: {args.report_id}")
            return 1

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(f"Report: {report.report_id}")
            print(f"  Title: {report.title}")
            print(f"  Type: {report.report_type.value}")
            print(f"  Status: {report.status.value}")
            print(f"  Sections: {len(report.sections)}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
