#!/usr/bin/env python3
"""
main.py - Deployment CLI (Step 230)

PBTSO Phase: DISTRIBUTE
A2A Integration: Complete CLI interface for deployment operations

Provides:
- DeployCLI: Main CLI class
- main: CLI entry point

Bus Topics:
- deploy.cli.command

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
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
    actor: str = "deploy-cli"
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
# Output Formatters
# ==============================================================================

class OutputFormatter:
    """Format CLI output."""

    @staticmethod
    def table(headers: List[str], rows: List[List[str]]) -> str:
        """Format as table."""
        if not rows:
            return "No data"

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        # Format header
        header_line = " | ".join(
            str(h).ljust(widths[i]) for i, h in enumerate(headers)
        )
        separator = "-+-".join("-" * w for w in widths)

        # Format rows
        row_lines = []
        for row in rows:
            row_line = " | ".join(
                str(row[i] if i < len(row) else "").ljust(widths[i])
                for i in range(len(headers))
            )
            row_lines.append(row_line)

        return f"{header_line}\n{separator}\n" + "\n".join(row_lines)

    @staticmethod
    def json_output(data: Any) -> str:
        """Format as JSON."""
        return json.dumps(data, indent=2, default=str)

    @staticmethod
    def status_icon(success: bool) -> str:
        """Get status icon."""
        return "OK" if success else "FAIL"


# ==============================================================================
# Deployment CLI (Step 230)
# ==============================================================================

class DeployCLI:
    """
    Deployment CLI - Complete CLI interface for deployment operations.

    PBTSO Phase: DISTRIBUTE

    Provides unified CLI access to all deploy agent components:
    - Deployment orchestration
    - History and audit
    - Scheduling
    - Approvals
    - Metrics
    - Reports
    - Notifications
    - CI/CD integration

    Example:
        >>> cli = DeployCLI()
        >>> cli.run(["deploy", "api", "--version", "v2.0.0"])
    """

    VERSION = "0.1.0"
    ACTOR_ID = "deploy-cli"

    def __init__(self):
        """Initialize the CLI."""
        self.parser = self._create_parser()
        self.formatter = OutputFormatter()

        # Lazy-loaded components
        self._orchestrator = None
        self._history_tracker = None
        self._scheduler = None
        self._approval_gate = None
        self._metrics_collector = None
        self._report_generator = None
        self._notifier = None
        self._integration_hub = None
        self._comparator = None

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            prog="deploy",
            description="Pluribus Deployment CLI (Step 230)",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Commands:
  deploy          Deploy a service
  rollback        Rollback a deployment
  status          Get deployment status
  history         View deployment history
  schedule        Manage scheduled deployments
  approve         Manage approvals
  metrics         View deployment metrics
  reports         Generate reports
  notify          Send notifications
  compare         Compare deployments
  integration     Manage CI/CD integrations
  api             Manage API server

Examples:
  deploy api --version v2.0.0 --env prod
  deploy rollback api --to v1.9.0
  deploy history --service api --limit 10
  deploy schedule create api --cron "0 2 * * *"
            """,
        )

        parser.add_argument("--version", "-V", action="version", version=f"%(prog)s {self.VERSION}")
        parser.add_argument("--json", "-j", action="store_true", help="JSON output")
        parser.add_argument("--quiet", "-q", action="store_true", help="Quiet output")
        parser.add_argument("--verbose", "-v", action="count", default=0, help="Verbose output")

        subparsers = parser.add_subparsers(dest="command", help="Command to run")

        # deploy command
        deploy_parser = subparsers.add_parser("deploy", help="Deploy a service")
        deploy_parser.add_argument("service", help="Service name")
        deploy_parser.add_argument("--version", "-v", required=True, help="Version to deploy")
        deploy_parser.add_argument("--env", "-e", default="staging", help="Target environment")
        deploy_parser.add_argument("--strategy", "-s", default="blue_green",
                                   choices=["blue_green", "canary", "rolling"])
        deploy_parser.add_argument("--wait", "-w", action="store_true", help="Wait for completion")
        deploy_parser.add_argument("--timeout", "-t", type=int, default=600, help="Timeout in seconds")

        # rollback command
        rollback_parser = subparsers.add_parser("rollback", help="Rollback a deployment")
        rollback_parser.add_argument("service", help="Service name")
        rollback_parser.add_argument("--to", required=True, help="Version to rollback to")
        rollback_parser.add_argument("--env", "-e", default="prod", help="Environment")

        # status command
        status_parser = subparsers.add_parser("status", help="Get deployment status")
        status_parser.add_argument("deployment_id", nargs="?", help="Deployment ID")
        status_parser.add_argument("--service", "-s", help="Service name")
        status_parser.add_argument("--env", "-e", help="Environment")

        # history command
        history_parser = subparsers.add_parser("history", help="View deployment history")
        history_parser.add_argument("--service", "-s", help="Filter by service")
        history_parser.add_argument("--env", "-e", help="Filter by environment")
        history_parser.add_argument("--status", help="Filter by status")
        history_parser.add_argument("--limit", "-l", type=int, default=10, help="Limit results")
        history_parser.add_argument("--stats", action="store_true", help="Show statistics")
        history_parser.add_argument("--days", "-d", type=int, default=30, help="Days for stats")

        # schedule command
        schedule_parser = subparsers.add_parser("schedule", help="Manage schedules")
        schedule_subparsers = schedule_parser.add_subparsers(dest="schedule_command")

        schedule_list = schedule_subparsers.add_parser("list", help="List schedules")
        schedule_list.add_argument("--service", "-s", help="Filter by service")

        schedule_create = schedule_subparsers.add_parser("create", help="Create schedule")
        schedule_create.add_argument("service", help="Service name")
        schedule_create.add_argument("--name", "-n", required=True, help="Schedule name")
        schedule_create.add_argument("--version", "-v", default="latest", help="Version")
        schedule_create.add_argument("--cron", "-c", help="Cron expression")
        schedule_create.add_argument("--at", help="One-time schedule (ISO timestamp)")
        schedule_create.add_argument("--env", "-e", default="staging", help="Environment")

        schedule_cancel = schedule_subparsers.add_parser("cancel", help="Cancel schedule")
        schedule_cancel.add_argument("schedule_id", help="Schedule ID")

        schedule_upcoming = schedule_subparsers.add_parser("upcoming", help="Upcoming deployments")
        schedule_upcoming.add_argument("--hours", type=int, default=24, help="Hours ahead")

        # approve command
        approve_parser = subparsers.add_parser("approve", help="Manage approvals")
        approve_subparsers = approve_parser.add_subparsers(dest="approve_command")

        approve_list = approve_subparsers.add_parser("list", help="List approvals")
        approve_list.add_argument("--pending", "-p", action="store_true", help="Pending only")

        approve_accept = approve_subparsers.add_parser("accept", help="Approve request")
        approve_accept.add_argument("request_id", help="Request ID")
        approve_accept.add_argument("--approver", "-a", required=True, help="Approver ID")
        approve_accept.add_argument("--comment", "-c", default="", help="Comment")

        approve_reject = approve_subparsers.add_parser("reject", help="Reject request")
        approve_reject.add_argument("request_id", help="Request ID")
        approve_reject.add_argument("--approver", "-a", required=True, help="Approver ID")
        approve_reject.add_argument("--comment", "-c", required=True, help="Rejection reason")

        # metrics command
        metrics_parser = subparsers.add_parser("metrics", help="View metrics")
        metrics_parser.add_argument("--service", "-s", help="Filter by service")
        metrics_parser.add_argument("--hours", type=int, default=24, help="Time window")
        metrics_parser.add_argument("--name", "-n", help="Metric name")

        # reports command
        reports_parser = subparsers.add_parser("reports", help="Generate reports")
        reports_subparsers = reports_parser.add_subparsers(dest="reports_command")

        reports_daily = reports_subparsers.add_parser("daily", help="Daily report")
        reports_daily.add_argument("--date", "-d", help="Date (YYYY-MM-DD)")
        reports_daily.add_argument("--format", "-f", default="json",
                                   choices=["json", "markdown", "html"])

        reports_weekly = reports_subparsers.add_parser("weekly", help="Weekly report")
        reports_weekly.add_argument("--format", "-f", default="json",
                                    choices=["json", "markdown", "html"])

        reports_list = reports_subparsers.add_parser("list", help="List reports")
        reports_list.add_argument("--limit", "-l", type=int, default=10, help="Limit")

        # notify command
        notify_parser = subparsers.add_parser("notify", help="Send notifications")
        notify_parser.add_argument("--type", "-t", required=True, help="Notification type")
        notify_parser.add_argument("--service", "-s", required=True, help="Service name")
        notify_parser.add_argument("--message", "-m", help="Custom message")

        # compare command
        compare_parser = subparsers.add_parser("compare", help="Compare deployments")
        compare_parser.add_argument("source", help="Source deployment ID")
        compare_parser.add_argument("target", help="Target deployment ID")
        compare_parser.add_argument("--type", "-t", default="full",
                                    choices=["version", "config", "full"])

        # integration command
        integration_parser = subparsers.add_parser("integration", help="CI/CD integrations")
        integration_subparsers = integration_parser.add_subparsers(dest="integration_command")

        integration_list = integration_subparsers.add_parser("list", help="List integrations")

        integration_add = integration_subparsers.add_parser("add", help="Add integration")
        integration_add.add_argument("--name", "-n", required=True, help="Integration name")
        integration_add.add_argument("--type", "-t", required=True, help="Integration type")
        integration_add.add_argument("--url", "-u", help="Endpoint URL")

        integration_trigger = integration_subparsers.add_parser("trigger", help="Trigger pipeline")
        integration_trigger.add_argument("integration_id", help="Integration ID")
        integration_trigger.add_argument("--service", "-s", required=True, help="Service")
        integration_trigger.add_argument("--version", "-v", required=True, help="Version")

        # api command
        api_parser = subparsers.add_parser("api", help="API server")
        api_subparsers = api_parser.add_subparsers(dest="api_command")

        api_serve = api_subparsers.add_parser("serve", help="Start API server")
        api_serve.add_argument("--host", "-H", default="0.0.0.0", help="Bind host")
        api_serve.add_argument("--port", "-p", type=int, default=8080, help="Bind port")

        api_key = api_subparsers.add_parser("create-key", help="Create API key")
        api_key.add_argument("--name", "-n", required=True, help="Key name")

        return parser

    def _get_orchestrator(self):
        """Lazy load orchestrator."""
        if self._orchestrator is None:
            from ..orchestrator_v2 import DeployOrchestratorV2
            self._orchestrator = DeployOrchestratorV2()
        return self._orchestrator

    def _get_history_tracker(self):
        """Lazy load history tracker."""
        if self._history_tracker is None:
            from ..history.tracker import DeploymentHistoryTracker
            self._history_tracker = DeploymentHistoryTracker()
        return self._history_tracker

    def _get_scheduler(self):
        """Lazy load scheduler."""
        if self._scheduler is None:
            from ..scheduler.scheduler import DeploymentScheduler
            self._scheduler = DeploymentScheduler()
        return self._scheduler

    def _get_approval_gate(self):
        """Lazy load approval gate."""
        if self._approval_gate is None:
            from ..approval.gate import DeploymentApprovalGate
            self._approval_gate = DeploymentApprovalGate()
        return self._approval_gate

    def _get_metrics_collector(self):
        """Lazy load metrics collector."""
        if self._metrics_collector is None:
            from ..metrics.collector import DeploymentMetricsCollector
            self._metrics_collector = DeploymentMetricsCollector()
        return self._metrics_collector

    def _get_report_generator(self):
        """Lazy load report generator."""
        if self._report_generator is None:
            from ..reports.generator import DeploymentReportGenerator
            self._report_generator = DeploymentReportGenerator()
        return self._report_generator

    def _get_comparator(self):
        """Lazy load comparator."""
        if self._comparator is None:
            from ..comparison.comparator import DeploymentComparator
            self._comparator = DeploymentComparator()
        return self._comparator

    def _get_integration_hub(self):
        """Lazy load integration hub."""
        if self._integration_hub is None:
            from ..integration.hub import DeploymentIntegrationHub
            self._integration_hub = DeploymentIntegrationHub()
        return self._integration_hub

    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the CLI.

        Args:
            args: Command line arguments

        Returns:
            Exit code
        """
        parsed = self.parser.parse_args(args)

        _emit_bus_event(
            "deploy.cli.command",
            {"command": parsed.command, "args": vars(parsed)},
            actor=self.ACTOR_ID,
        )

        if not parsed.command:
            self.parser.print_help()
            return 0

        try:
            if parsed.command == "deploy":
                return self._cmd_deploy(parsed)
            elif parsed.command == "rollback":
                return self._cmd_rollback(parsed)
            elif parsed.command == "status":
                return self._cmd_status(parsed)
            elif parsed.command == "history":
                return self._cmd_history(parsed)
            elif parsed.command == "schedule":
                return self._cmd_schedule(parsed)
            elif parsed.command == "approve":
                return self._cmd_approve(parsed)
            elif parsed.command == "metrics":
                return self._cmd_metrics(parsed)
            elif parsed.command == "reports":
                return self._cmd_reports(parsed)
            elif parsed.command == "notify":
                return self._cmd_notify(parsed)
            elif parsed.command == "compare":
                return self._cmd_compare(parsed)
            elif parsed.command == "integration":
                return self._cmd_integration(parsed)
            elif parsed.command == "api":
                return self._cmd_api(parsed)
            else:
                self.parser.print_help()
                return 1

        except Exception as e:
            if parsed.verbose:
                import traceback
                traceback.print_exc()
            else:
                print(f"Error: {e}", file=sys.stderr)
            return 1

    def _output(self, data: Any, parsed: argparse.Namespace) -> None:
        """Output data based on format preference."""
        if parsed.quiet:
            return

        if parsed.json:
            print(self.formatter.json_output(data))
        elif isinstance(data, str):
            print(data)
        else:
            print(self.formatter.json_output(data))

    def _cmd_deploy(self, parsed: argparse.Namespace) -> int:
        """Handle deploy command."""
        from ..orchestrator_v2 import PipelineConfigV2, DeploymentType

        config = PipelineConfigV2(
            name=f"{parsed.service}-deploy",
            service_name=parsed.service,
            version=parsed.version,
            deployment_type=DeploymentType.FULL,
            strategy=parsed.strategy,
            target_environments=[parsed.env],
        )

        orchestrator = self._get_orchestrator()

        print(f"Deploying {parsed.service}:{parsed.version} to {parsed.env}...")

        state = asyncio.get_event_loop().run_until_complete(
            orchestrator.run_pipeline(config)
        )

        if parsed.json:
            self._output(state.to_dict(), parsed)
        else:
            status = self.formatter.status_icon(state.current_phase.value == "complete")
            print(f"[{status}] {state.pipeline_id}")
            print(f"  Service: {state.config.service_name}")
            print(f"  Version: {state.config.version}")
            print(f"  Phase: {state.current_phase.value}")
            if state.metrics:
                print(f"  Duration: {state.metrics.get('total_duration_ms', 0):.0f}ms")
            if state.error:
                print(f"  Error: {state.error}")

        return 0 if state.current_phase.value == "complete" else 1

    def _cmd_rollback(self, parsed: argparse.Namespace) -> int:
        """Handle rollback command."""
        tracker = self._get_history_tracker()

        print(f"Rolling back {parsed.service} to {parsed.to}...")

        # Find current deployment
        current = tracker.get_latest(parsed.service, parsed.env)
        if not current:
            print("No current deployment found")
            return 1

        # Record rollback
        rollback = tracker.record_rollback(
            original_deployment_id=current.deployment_id,
            rollback_to_version=parsed.to,
        )

        if rollback:
            if parsed.json:
                self._output(rollback.to_dict(), parsed)
            else:
                print(f"Rollback initiated: {rollback.deployment_id}")
            return 0
        else:
            print("Rollback failed")
            return 1

    def _cmd_status(self, parsed: argparse.Namespace) -> int:
        """Handle status command."""
        orchestrator = self._get_orchestrator()

        if parsed.deployment_id:
            pipeline = orchestrator.get_pipeline(parsed.deployment_id)
            if not pipeline:
                print(f"Deployment not found: {parsed.deployment_id}")
                return 1

            if parsed.json:
                self._output(pipeline.to_dict(), parsed)
            else:
                print(f"Deployment: {pipeline.pipeline_id}")
                print(f"  Service: {pipeline.config.service_name}")
                print(f"  Version: {pipeline.config.version}")
                print(f"  Phase: {pipeline.current_phase.value}")
        else:
            pipelines = orchestrator.list_pipelines(
                service_name=parsed.service,
                limit=5,
            )

            if parsed.json:
                self._output([p.to_dict() for p in pipelines], parsed)
            else:
                for p in pipelines:
                    print(f"{p.pipeline_id} ({p.config.service_name}) - {p.current_phase.value}")

        return 0

    def _cmd_history(self, parsed: argparse.Namespace) -> int:
        """Handle history command."""
        tracker = self._get_history_tracker()

        if parsed.stats:
            stats = tracker.get_statistics(
                service_name=parsed.service,
                environment=parsed.env,
                days=parsed.days,
            )

            if parsed.json:
                self._output(stats, parsed)
            else:
                print(f"Deployment Statistics ({parsed.days} days)")
                print(f"  Total: {stats['total']}")
                print(f"  Succeeded: {stats['succeeded']}")
                print(f"  Failed: {stats['failed']}")
                print(f"  Success Rate: {stats['success_rate']:.1f}%")
                print(f"  Avg Duration: {stats['avg_duration_ms']:.0f}ms")
        else:
            from ..history.tracker import DeploymentStatus
            status = DeploymentStatus(parsed.status.upper()) if parsed.status else None

            deployments = tracker.query_deployments(
                service_name=parsed.service,
                environment=parsed.env,
                status=status,
                limit=parsed.limit,
            )

            if parsed.json:
                self._output([d.to_dict() for d in deployments], parsed)
            else:
                if not deployments:
                    print("No deployments found")
                else:
                    headers = ["ID", "Service", "Version", "Status", "Date"]
                    rows = [
                        [
                            d.deployment_id[:20],
                            d.service_name,
                            d.version,
                            d.status.value,
                            datetime.fromtimestamp(d.started_at).strftime("%Y-%m-%d %H:%M"),
                        ]
                        for d in deployments
                    ]
                    print(self.formatter.table(headers, rows))

        return 0

    def _cmd_schedule(self, parsed: argparse.Namespace) -> int:
        """Handle schedule command."""
        scheduler = self._get_scheduler()

        if parsed.schedule_command == "list":
            schedules = scheduler.list_schedules(service_name=parsed.service)

            if parsed.json:
                self._output([s.to_dict() for s in schedules], parsed)
            else:
                if not schedules:
                    print("No schedules found")
                else:
                    for s in schedules:
                        next_run = ""
                        if s.next_run:
                            next_run = datetime.fromtimestamp(s.next_run).strftime("%Y-%m-%d %H:%M")
                        print(f"{s.schedule_id} ({s.service_name}) - {s.status.value} - {next_run}")

        elif parsed.schedule_command == "create":
            if parsed.cron:
                schedule = scheduler.schedule_cron(
                    name=parsed.name,
                    service_name=parsed.service,
                    version=parsed.version,
                    cron_expression=parsed.cron,
                    environment=parsed.env,
                )
            elif parsed.at:
                scheduled_time = datetime.fromisoformat(parsed.at).timestamp()
                schedule = scheduler.schedule_once(
                    name=parsed.name,
                    service_name=parsed.service,
                    version=parsed.version,
                    scheduled_time=scheduled_time,
                    environment=parsed.env,
                )
            else:
                print("Error: Must specify --cron or --at")
                return 1

            if parsed.json:
                self._output(schedule.to_dict(), parsed)
            else:
                print(f"Created: {schedule.schedule_id}")

        elif parsed.schedule_command == "cancel":
            success = scheduler.cancel_schedule(parsed.schedule_id)
            if success:
                print(f"Cancelled: {parsed.schedule_id}")
            else:
                print(f"Failed to cancel: {parsed.schedule_id}")
                return 1

        elif parsed.schedule_command == "upcoming":
            upcoming = scheduler.get_upcoming(hours=parsed.hours)

            if parsed.json:
                self._output([s.to_dict() for s in upcoming], parsed)
            else:
                if not upcoming:
                    print("No upcoming deployments")
                else:
                    for s in upcoming:
                        next_run = datetime.fromtimestamp(s.next_run).strftime("%Y-%m-%d %H:%M")
                        print(f"{next_run} - {s.service_name}:{s.version}")

        return 0

    def _cmd_approve(self, parsed: argparse.Namespace) -> int:
        """Handle approve command."""
        gate = self._get_approval_gate()

        if parsed.approve_command == "list":
            if parsed.pending:
                requests = gate.list_pending()
            else:
                requests = gate.list_requests()

            if parsed.json:
                self._output([r.to_dict() for r in requests], parsed)
            else:
                if not requests:
                    print("No approval requests")
                else:
                    for r in requests:
                        print(f"{r.request_id} ({r.service_name}:{r.version}) - {r.status.value}")

        elif parsed.approve_command == "accept":
            try:
                request = gate.approve(
                    request_id=parsed.request_id,
                    approver_id=parsed.approver,
                    comment=parsed.comment,
                )
                if parsed.json:
                    self._output(request.to_dict(), parsed)
                else:
                    print(f"Approved: {request.request_id}")
            except ValueError as e:
                print(f"Error: {e}")
                return 1

        elif parsed.approve_command == "reject":
            try:
                request = gate.reject(
                    request_id=parsed.request_id,
                    approver_id=parsed.approver,
                    comment=parsed.comment,
                )
                if parsed.json:
                    self._output(request.to_dict(), parsed)
                else:
                    print(f"Rejected: {request.request_id}")
            except ValueError as e:
                print(f"Error: {e}")
                return 1

        return 0

    def _cmd_metrics(self, parsed: argparse.Namespace) -> int:
        """Handle metrics command."""
        collector = self._get_metrics_collector()

        summary = collector.get_summary(
            service_name=parsed.service,
            hours=parsed.hours,
        )

        if parsed.json:
            self._output(summary, parsed)
        else:
            print(f"Metrics Summary ({parsed.hours}h)")
            for name, stats in summary.get("metrics", {}).items():
                print(f"  {name}:")
                print(f"    count={stats['count']}, avg={stats['avg']:.2f}, p95={stats['p95']:.2f}")

        return 0

    def _cmd_reports(self, parsed: argparse.Namespace) -> int:
        """Handle reports command."""
        generator = self._get_report_generator()

        if parsed.reports_command == "daily":
            from ..reports.generator import ReportFormat
            date = None
            if parsed.date:
                date = datetime.fromisoformat(parsed.date)

            report = generator.generate_daily_report(date=date)
            file_path = generator.export(report, ReportFormat(parsed.format))

            if parsed.json:
                self._output(report.to_dict(), parsed)
            else:
                print(f"Generated: {report.report_id}")
                print(f"  File: {file_path}")

        elif parsed.reports_command == "weekly":
            from ..reports.generator import ReportFormat

            report = generator.generate_weekly_report()
            file_path = generator.export(report, ReportFormat(parsed.format))

            if parsed.json:
                self._output(report.to_dict(), parsed)
            else:
                print(f"Generated: {report.report_id}")
                print(f"  File: {file_path}")

        elif parsed.reports_command == "list":
            reports = generator.list_reports(limit=parsed.limit)

            if parsed.json:
                self._output([r.to_dict() for r in reports], parsed)
            else:
                for r in reports:
                    print(f"{r.report_id} ({r.report_type.value}) - {r.status.value}")

        return 0

    def _cmd_notify(self, parsed: argparse.Namespace) -> int:
        """Handle notify command."""
        from ..notifications.notifier import DeploymentNotificationSystem, NotificationType

        notifier = DeploymentNotificationSystem()

        notification = asyncio.get_event_loop().run_until_complete(
            notifier.notify(
                notification_type=NotificationType(parsed.type),
                service_name=parsed.service,
                variables={"message": parsed.message or ""},
            )
        )

        if parsed.json:
            self._output(notification.to_dict(), parsed)
        else:
            print(f"Sent: {notification.notification_id}")
            print(f"  Status: {notification.status.value}")

        return 0

    def _cmd_compare(self, parsed: argparse.Namespace) -> int:
        """Handle compare command."""
        comparator = self._get_comparator()

        try:
            from ..comparison.comparator import ComparisonType

            diff = comparator.compare(
                source_id=parsed.source,
                target_id=parsed.target,
                comparison_type=ComparisonType(parsed.type.upper()),
            )

            if parsed.json:
                self._output(diff.to_dict(), parsed)
            else:
                print(f"Comparison: {diff.diff_id}")
                print(f"  Changes: {diff.total_changes}")
                print(f"  Breaking: {diff.has_breaking_changes}")
                if diff.version_diff:
                    print(f"  Version: {diff.version_diff.old_version} -> {diff.version_diff.new_version}")

            return 0
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    def _cmd_integration(self, parsed: argparse.Namespace) -> int:
        """Handle integration command."""
        hub = self._get_integration_hub()

        if parsed.integration_command == "list":
            integrations = hub.list_integrations()

            if parsed.json:
                self._output([i.to_dict() for i in integrations], parsed)
            else:
                for i in integrations:
                    print(f"{i.integration_id} ({i.name}) - {i.status.value}")

        elif parsed.integration_command == "add":
            from ..integration.hub import IntegrationType

            config = hub.register_integration(
                name=parsed.name,
                integration_type=IntegrationType(parsed.type),
                endpoint_url=parsed.url or "",
            )

            if parsed.json:
                self._output(config.to_dict(), parsed)
            else:
                print(f"Added: {config.integration_id}")

        elif parsed.integration_command == "trigger":
            event = asyncio.get_event_loop().run_until_complete(
                hub.trigger_pipeline(
                    integration_id=parsed.integration_id,
                    service_name=parsed.service,
                    version=parsed.version,
                    environment="staging",
                )
            )

            if parsed.json:
                self._output(event.to_dict(), parsed)
            else:
                print(f"Triggered: {event.event_id}")
                print(f"  Status: {event.status}")

        return 0

    def _cmd_api(self, parsed: argparse.Namespace) -> int:
        """Handle API command."""
        from ..api.server import DeploymentAPI, create_app

        if parsed.api_command == "serve":
            server, api = create_app(parsed.host, parsed.port)
            print(f"Starting API server on {parsed.host}:{parsed.port}")

            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\nShutting down...")
                server.shutdown()

        elif parsed.api_command == "create-key":
            api = DeploymentAPI()
            key = api.create_api_key(parsed.name)
            print(f"Created API key: {key}")

        return 0


def main() -> int:
    """Main CLI entry point."""
    cli = DeployCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
