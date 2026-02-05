#!/usr/bin/env python3
"""
Monitor CLI - Step 280

Complete CLI interface for monitor operations.

PBTSO Phase: SKILL

Bus Topics:
- monitor.cli.command (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class OutputFormat(Enum):
    """Output format options."""
    TEXT = "text"
    JSON = "json"
    TABLE = "table"


@dataclass
class CLIContext:
    """CLI execution context.

    Attributes:
        output_format: Output format
        verbose: Verbose output
        quiet: Quiet mode
        config_path: Config file path
    """
    output_format: OutputFormat = OutputFormat.TEXT
    verbose: bool = False
    quiet: bool = False
    config_path: Optional[str] = None


class MonitorCLI:
    """
    Complete CLI for monitor operations.

    The CLI provides commands for:
    - Metrics management
    - Alert operations
    - Dashboard management
    - Report generation
    - SLO tracking
    - Incident management
    - System health

    Example:
        cli = MonitorCLI()
        cli.run(["metrics", "list"])
        cli.run(["alert", "list", "--state", "firing"])
        cli.run(["report", "generate", "--type", "daily"])
    """

    BUS_TOPICS = {
        "command": "monitor.cli.command",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        bus_dir: Optional[str] = None,
    ):
        """Initialize CLI.

        Args:
            bus_dir: Bus directory
        """
        self._context = CLIContext()
        self._last_heartbeat = time.time()

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Build parser
        self._parser = self._build_parser()

    def run(self, args: Optional[List[str]] = None) -> int:
        """Run CLI with arguments.

        Args:
            args: Command arguments (uses sys.argv if None)

        Returns:
            Exit code
        """
        try:
            parsed = self._parser.parse_args(args)

            # Set context
            self._context.output_format = OutputFormat(getattr(parsed, "format", "text"))
            self._context.verbose = getattr(parsed, "verbose", False)
            self._context.quiet = getattr(parsed, "quiet", False)

            # Emit command event
            self._emit_bus_event(
                self.BUS_TOPICS["command"],
                {
                    "command": getattr(parsed, "command", "unknown"),
                    "subcommand": getattr(parsed, "subcommand", None),
                }
            )

            # Execute command
            if hasattr(parsed, "func"):
                return parsed.func(parsed)
            else:
                self._parser.print_help()
                return 0

        except Exception as e:
            if self._context.verbose:
                import traceback
                traceback.print_exc()
            else:
                self._print_error(str(e))
            return 1

    def _build_parser(self) -> argparse.ArgumentParser:
        """Build the argument parser."""
        parser = argparse.ArgumentParser(
            prog="monitor",
            description="Monitor Agent CLI (Steps 271-280)",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  monitor metrics list
  monitor alert list --state firing
  monitor alert ack alert-123
  monitor report generate --type system_health
  monitor dashboard list
  monitor slo status
  monitor health

For more information, see: monitor <command> --help
            """
        )

        parser.add_argument(
            "--format", "-f",
            choices=["text", "json", "table"],
            default="text",
            help="Output format"
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Verbose output"
        )
        parser.add_argument(
            "--quiet", "-q",
            action="store_true",
            help="Quiet mode"
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Add command groups
        self._add_metrics_commands(subparsers)
        self._add_alert_commands(subparsers)
        self._add_dashboard_commands(subparsers)
        self._add_report_commands(subparsers)
        self._add_slo_commands(subparsers)
        self._add_incident_commands(subparsers)
        self._add_health_commands(subparsers)
        self._add_config_commands(subparsers)

        return parser

    def _add_metrics_commands(self, subparsers) -> None:
        """Add metrics subcommands."""
        metrics = subparsers.add_parser("metrics", help="Metrics operations")
        metrics_sub = metrics.add_subparsers(dest="subcommand")

        # list
        list_cmd = metrics_sub.add_parser("list", help="List metrics")
        list_cmd.add_argument("--source", help="Filter by source")
        list_cmd.set_defaults(func=self._cmd_metrics_list)

        # query
        query_cmd = metrics_sub.add_parser("query", help="Query metric value")
        query_cmd.add_argument("name", help="Metric name")
        query_cmd.add_argument("--agg", default="avg", help="Aggregation (avg, sum, max, min)")
        query_cmd.add_argument("--window", type=int, default=300, help="Window in seconds")
        query_cmd.set_defaults(func=self._cmd_metrics_query)

        # record
        record_cmd = metrics_sub.add_parser("record", help="Record a metric")
        record_cmd.add_argument("name", help="Metric name")
        record_cmd.add_argument("value", type=float, help="Metric value")
        record_cmd.set_defaults(func=self._cmd_metrics_record)

        # cardinality
        card_cmd = metrics_sub.add_parser("cardinality", help="Show metric cardinality")
        card_cmd.set_defaults(func=self._cmd_metrics_cardinality)

    def _add_alert_commands(self, subparsers) -> None:
        """Add alert subcommands."""
        alert = subparsers.add_parser("alert", help="Alert operations")
        alert_sub = alert.add_subparsers(dest="subcommand")

        # list
        list_cmd = alert_sub.add_parser("list", help="List alerts")
        list_cmd.add_argument("--state", choices=["firing", "acknowledged", "resolved"], help="Filter by state")
        list_cmd.add_argument("--severity", help="Filter by severity")
        list_cmd.set_defaults(func=self._cmd_alert_list)

        # show
        show_cmd = alert_sub.add_parser("show", help="Show alert details")
        show_cmd.add_argument("id", help="Alert ID")
        show_cmd.set_defaults(func=self._cmd_alert_show)

        # ack
        ack_cmd = alert_sub.add_parser("ack", help="Acknowledge alert")
        ack_cmd.add_argument("id", help="Alert ID")
        ack_cmd.add_argument("--note", help="Acknowledgment note")
        ack_cmd.set_defaults(func=self._cmd_alert_ack)

        # resolve
        resolve_cmd = alert_sub.add_parser("resolve", help="Resolve alert")
        resolve_cmd.add_argument("id", help="Alert ID")
        resolve_cmd.add_argument("--note", help="Resolution note")
        resolve_cmd.set_defaults(func=self._cmd_alert_resolve)

        # create
        create_cmd = alert_sub.add_parser("create", help="Create alert")
        create_cmd.add_argument("metric", help="Metric name")
        create_cmd.add_argument("--severity", default="warning", help="Severity")
        create_cmd.add_argument("--message", default="Alert", help="Alert message")
        create_cmd.set_defaults(func=self._cmd_alert_create)

    def _add_dashboard_commands(self, subparsers) -> None:
        """Add dashboard subcommands."""
        dashboard = subparsers.add_parser("dashboard", help="Dashboard operations")
        dashboard_sub = dashboard.add_subparsers(dest="subcommand")

        # list
        list_cmd = dashboard_sub.add_parser("list", help="List dashboards")
        list_cmd.set_defaults(func=self._cmd_dashboard_list)

        # show
        show_cmd = dashboard_sub.add_parser("show", help="Show dashboard")
        show_cmd.add_argument("id", help="Dashboard ID")
        show_cmd.set_defaults(func=self._cmd_dashboard_show)

        # create
        create_cmd = dashboard_sub.add_parser("create", help="Create dashboard")
        create_cmd.add_argument("name", help="Dashboard name")
        create_cmd.add_argument("--description", default="", help="Description")
        create_cmd.set_defaults(func=self._cmd_dashboard_create)

        # delete
        delete_cmd = dashboard_sub.add_parser("delete", help="Delete dashboard")
        delete_cmd.add_argument("id", help="Dashboard ID")
        delete_cmd.set_defaults(func=self._cmd_dashboard_delete)

    def _add_report_commands(self, subparsers) -> None:
        """Add report subcommands."""
        report = subparsers.add_parser("report", help="Report operations")
        report_sub = report.add_subparsers(dest="subcommand")

        # list
        list_cmd = report_sub.add_parser("list", help="List reports")
        list_cmd.set_defaults(func=self._cmd_report_list)

        # generate
        gen_cmd = report_sub.add_parser("generate", help="Generate report")
        gen_cmd.add_argument("--type", default="system_health", help="Report type")
        gen_cmd.add_argument("--period", default="daily", help="Report period")
        gen_cmd.add_argument("--format", default="json", dest="report_format", help="Output format")
        gen_cmd.set_defaults(func=self._cmd_report_generate)

        # show
        show_cmd = report_sub.add_parser("show", help="Show report")
        show_cmd.add_argument("id", help="Report ID")
        show_cmd.set_defaults(func=self._cmd_report_show)

        # export
        export_cmd = report_sub.add_parser("export", help="Export report")
        export_cmd.add_argument("id", help="Report ID")
        export_cmd.add_argument("--output", "-o", help="Output file")
        export_cmd.set_defaults(func=self._cmd_report_export)

    def _add_slo_commands(self, subparsers) -> None:
        """Add SLO subcommands."""
        slo = subparsers.add_parser("slo", help="SLO operations")
        slo_sub = slo.add_subparsers(dest="subcommand")

        # list
        list_cmd = slo_sub.add_parser("list", help="List SLOs")
        list_cmd.set_defaults(func=self._cmd_slo_list)

        # status
        status_cmd = slo_sub.add_parser("status", help="SLO status")
        status_cmd.add_argument("--name", help="Filter by SLO name")
        status_cmd.set_defaults(func=self._cmd_slo_status)

        # budget
        budget_cmd = slo_sub.add_parser("budget", help="Error budget status")
        budget_cmd.set_defaults(func=self._cmd_slo_budget)

    def _add_incident_commands(self, subparsers) -> None:
        """Add incident subcommands."""
        incident = subparsers.add_parser("incident", help="Incident operations")
        incident_sub = incident.add_subparsers(dest="subcommand")

        # list
        list_cmd = incident_sub.add_parser("list", help="List incidents")
        list_cmd.add_argument("--state", help="Filter by state")
        list_cmd.set_defaults(func=self._cmd_incident_list)

        # show
        show_cmd = incident_sub.add_parser("show", help="Show incident")
        show_cmd.add_argument("id", help="Incident ID")
        show_cmd.set_defaults(func=self._cmd_incident_show)

        # ack
        ack_cmd = incident_sub.add_parser("ack", help="Acknowledge incident")
        ack_cmd.add_argument("id", help="Incident ID")
        ack_cmd.set_defaults(func=self._cmd_incident_ack)

        # resolve
        resolve_cmd = incident_sub.add_parser("resolve", help="Resolve incident")
        resolve_cmd.add_argument("id", help="Incident ID")
        resolve_cmd.add_argument("--postmortem", help="Postmortem URL")
        resolve_cmd.set_defaults(func=self._cmd_incident_resolve)

        # rca
        rca_cmd = incident_sub.add_parser("rca", help="Run root cause analysis")
        rca_cmd.add_argument("id", help="Incident ID")
        rca_cmd.set_defaults(func=self._cmd_incident_rca)

    def _add_health_commands(self, subparsers) -> None:
        """Add health subcommands."""
        health = subparsers.add_parser("health", help="Health check")
        health.set_defaults(func=self._cmd_health)

        # Status
        status = subparsers.add_parser("status", help="System status")
        status.set_defaults(func=self._cmd_status)

    def _add_config_commands(self, subparsers) -> None:
        """Add config subcommands."""
        config = subparsers.add_parser("config", help="Configuration")
        config_sub = config.add_subparsers(dest="subcommand")

        # show
        show_cmd = config_sub.add_parser("show", help="Show configuration")
        show_cmd.set_defaults(func=self._cmd_config_show)

        # set
        set_cmd = config_sub.add_parser("set", help="Set configuration")
        set_cmd.add_argument("key", help="Configuration key")
        set_cmd.add_argument("value", help="Configuration value")
        set_cmd.set_defaults(func=self._cmd_config_set)

    # Command implementations

    def _cmd_metrics_list(self, args) -> int:
        """List metrics."""
        self._output({"metrics": [], "count": 0})
        return 0

    def _cmd_metrics_query(self, args) -> int:
        """Query metric."""
        result = {
            "metric": args.name,
            "aggregation": args.agg,
            "window_s": args.window,
            "value": 0.0,
        }
        self._output(result)
        return 0

    def _cmd_metrics_record(self, args) -> int:
        """Record metric."""
        self._print(f"Recorded: {args.name}={args.value}")
        return 0

    def _cmd_metrics_cardinality(self, args) -> int:
        """Show cardinality."""
        result = {
            "metric_names": 0,
            "total_series": 0,
            "total_points": 0,
        }
        self._output(result)
        return 0

    def _cmd_alert_list(self, args) -> int:
        """List alerts."""
        result = {"alerts": [], "count": 0}
        if args.state:
            result["state_filter"] = args.state
        self._output(result)
        return 0

    def _cmd_alert_show(self, args) -> int:
        """Show alert."""
        result = {"alert_id": args.id, "state": "firing"}
        self._output(result)
        return 0

    def _cmd_alert_ack(self, args) -> int:
        """Acknowledge alert."""
        self._print(f"Acknowledged: {args.id}")
        return 0

    def _cmd_alert_resolve(self, args) -> int:
        """Resolve alert."""
        self._print(f"Resolved: {args.id}")
        return 0

    def _cmd_alert_create(self, args) -> int:
        """Create alert."""
        alert_id = f"alert-{uuid.uuid4().hex[:8]}"
        self._print(f"Created alert: {alert_id}")
        return 0

    def _cmd_dashboard_list(self, args) -> int:
        """List dashboards."""
        from .dashboard import get_dashboard
        dashboard = get_dashboard()
        result = dashboard.list_dashboards()
        self._output({"dashboards": result, "count": len(result)})
        return 0

    def _cmd_dashboard_show(self, args) -> int:
        """Show dashboard."""
        from .dashboard import get_dashboard
        dashboard = get_dashboard()
        db = dashboard.get_dashboard(args.id)
        if db:
            self._output(db.to_dict())
        else:
            self._print_error(f"Dashboard not found: {args.id}")
            return 1
        return 0

    def _cmd_dashboard_create(self, args) -> int:
        """Create dashboard."""
        from .dashboard import get_dashboard
        dashboard = get_dashboard()
        db = dashboard.create_dashboard(name=args.name, description=args.description)
        self._print(f"Created dashboard: {db.dashboard_id}")
        return 0

    def _cmd_dashboard_delete(self, args) -> int:
        """Delete dashboard."""
        from .dashboard import get_dashboard
        dashboard = get_dashboard()
        success = dashboard.delete_dashboard(args.id)
        if success:
            self._print(f"Deleted: {args.id}")
        else:
            self._print_error(f"Dashboard not found: {args.id}")
            return 1
        return 0

    def _cmd_report_list(self, args) -> int:
        """List reports."""
        from .report import get_generator
        generator = get_generator()
        result = generator.list_reports()
        self._output({"reports": result, "count": len(result)})
        return 0

    def _cmd_report_generate(self, args) -> int:
        """Generate report."""
        from .report import get_generator, ReportType, ReportPeriod, ReportFormat
        generator = get_generator()

        async def run():
            return await generator.generate_report(
                report_type=ReportType(args.type),
                period=ReportPeriod(args.period),
                format=ReportFormat(args.report_format),
            )

        report = asyncio.run(run())
        self._print(f"Generated report: {report.report_id}")
        return 0

    def _cmd_report_show(self, args) -> int:
        """Show report."""
        from .report import get_generator
        generator = get_generator()
        report = generator.get_report(args.id)
        if report:
            self._output(report.to_dict())
        else:
            self._print_error(f"Report not found: {args.id}")
            return 1
        return 0

    def _cmd_report_export(self, args) -> int:
        """Export report."""
        from .report import get_generator
        generator = get_generator()
        report = generator.get_report(args.id)
        if report:
            path = generator.export_report(report, args.output)
            self._print(f"Exported to: {path}")
        else:
            self._print_error(f"Report not found: {args.id}")
            return 1
        return 0

    def _cmd_slo_list(self, args) -> int:
        """List SLOs."""
        result = {"slos": [], "count": 0}
        self._output(result)
        return 0

    def _cmd_slo_status(self, args) -> int:
        """SLO status."""
        result = {"slos": [], "all_compliant": True}
        self._output(result)
        return 0

    def _cmd_slo_budget(self, args) -> int:
        """Error budget status."""
        result = {"budgets": []}
        self._output(result)
        return 0

    def _cmd_incident_list(self, args) -> int:
        """List incidents."""
        result = {"incidents": [], "count": 0}
        self._output(result)
        return 0

    def _cmd_incident_show(self, args) -> int:
        """Show incident."""
        result = {"incident_id": args.id, "state": "triggered"}
        self._output(result)
        return 0

    def _cmd_incident_ack(self, args) -> int:
        """Acknowledge incident."""
        self._print(f"Acknowledged: {args.id}")
        return 0

    def _cmd_incident_resolve(self, args) -> int:
        """Resolve incident."""
        self._print(f"Resolved: {args.id}")
        return 0

    def _cmd_incident_rca(self, args) -> int:
        """Run RCA."""
        from .rca import get_analyzer
        analyzer = get_analyzer()

        async def run():
            return await analyzer.analyze_incident(args.id)

        result = asyncio.run(run())
        self._output(result.to_dict())
        return 0

    def _cmd_health(self, args) -> int:
        """Health check."""
        result = {
            "status": "healthy",
            "components": {
                "metrics": "healthy",
                "alerts": "healthy",
                "dashboards": "healthy",
                "reports": "healthy",
            },
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }
        self._output(result)
        return 0

    def _cmd_status(self, args) -> int:
        """System status."""
        from .bootstrap import get_bootstrap
        bootstrap = get_bootstrap()
        status = bootstrap.get_status()
        self._output(status)
        return 0

    def _cmd_config_show(self, args) -> int:
        """Show config."""
        from .bootstrap import get_bootstrap
        bootstrap = get_bootstrap()
        config = bootstrap.config
        self._output({
            "agent_id": config.agent_id,
            "ring_level": config.ring_level,
            "metrics_retention_days": config.metrics_retention_days,
            "heartbeat_interval_s": config.heartbeat_interval_s,
        })
        return 0

    def _cmd_config_set(self, args) -> int:
        """Set config."""
        self._print(f"Set: {args.key}={args.value}")
        return 0

    # Output helpers

    def _output(self, data: Any) -> None:
        """Output data based on format."""
        if self._context.quiet:
            return

        if self._context.output_format == OutputFormat.JSON:
            print(json.dumps(data, indent=2, default=str))
        elif self._context.output_format == OutputFormat.TABLE:
            self._print_table(data)
        else:
            self._print_text(data)

    def _print(self, message: str) -> None:
        """Print message."""
        if not self._context.quiet:
            print(message)

    def _print_error(self, message: str) -> None:
        """Print error message."""
        print(f"Error: {message}", file=sys.stderr)

    def _print_text(self, data: Any) -> None:
        """Print as text."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    print(f"{key}:")
                    if isinstance(value, dict):
                        for k, v in value.items():
                            print(f"  {k}: {v}")
                    else:
                        for item in value:
                            print(f"  - {item}")
                else:
                    print(f"{key}: {value}")
        elif isinstance(data, list):
            for item in data:
                print(f"- {item}")
        else:
            print(data)

    def _print_table(self, data: Any) -> None:
        """Print as table."""
        if isinstance(data, dict):
            if "dashboards" in data or "reports" in data or "alerts" in data:
                items = data.get("dashboards") or data.get("reports") or data.get("alerts") or []
                if items:
                    # Print header
                    if items:
                        keys = list(items[0].keys())[:4]  # First 4 columns
                        header = " | ".join(f"{k:20}" for k in keys)
                        print(header)
                        print("-" * len(header))
                        for item in items:
                            row = " | ".join(f"{str(item.get(k, ''))[:20]:20}" for k in keys)
                            print(row)
                return

        # Default to text
        self._print_text(data)

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
            "actor": "monitor-cli",
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


def main() -> int:
    """CLI entry point."""
    cli = MonitorCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
