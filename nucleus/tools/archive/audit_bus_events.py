#!/usr/bin/env python3
"""Audit Bus Events Emitter.

DKIN v28 Remediation: Step 97

Emits standardized bus events for audit lifecycle:
- audit.scheduled - When an audit is scheduled
- audit.completed - When an audit finishes
- audit.findings - Batch findings report

Usage:
    # Schedule an audit
    python3 nucleus/tools/audit_bus_events.py schedule --type quick --trigger cron

    # Emit completion event
    python3 nucleus/tools/audit_bus_events.py complete --audit-id UUID --report-path PATH

    # Emit findings batch
    python3 nucleus/tools/audit_bus_events.py findings --audit-id UUID --findings-json PATH
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

# Add tools directory to path
TOOLS_DIR = Path(__file__).parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


@dataclass
class AuditBusEmitter:
    """Emits audit-related bus events."""

    actor: str = "pbaudit"
    durable: bool = True
    _emit_fn: Any = None
    _bus_paths: Any = None

    def __post_init__(self):
        self._init_bus()

    def _init_bus(self) -> None:
        """Initialize bus connection."""
        try:
            from agent_bus import emit_event, resolve_bus_paths
            self._bus_paths = resolve_bus_paths(None)
            self._emit_fn = emit_event
        except ImportError:
            self._emit_fn = None
            self._bus_paths = None

    def _emit(self, topic: str, data: dict, level: str = "info") -> str | None:
        """Emit a bus event."""
        if not self._emit_fn or not self._bus_paths:
            print(f"[DRY RUN] Would emit {topic}: {json.dumps(data, default=str)}")
            return None

        try:
            event_id = self._emit_fn(
                self._bus_paths,
                topic=topic,
                kind="log",
                level=level,
                actor=self.actor,
                data=data,
                trace_id=None,
                run_id=None,
                durable=self.durable,
            )
            return event_id
        except Exception as e:
            print(f"[ERROR] Failed to emit {topic}: {e}", file=sys.stderr)
            return None

    def emit_scheduled(
        self,
        audit_type: str = "quick",
        trigger: str = "manual",
        scheduled_time: str | None = None,
    ) -> dict:
        """Emit audit.scheduled event.

        Args:
            audit_type: Type of audit (quick, full, deep)
            trigger: What triggered the scheduling (manual, cron, ci, omega-loop)
            scheduled_time: ISO8601 timestamp (defaults to now)

        Returns:
            Event payload dict with event_id
        """
        audit_id = str(uuid.uuid4())
        if not scheduled_time:
            scheduled_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        payload = {
            "event_type": "audit.scheduled",
            "audit_id": audit_id,
            "scheduled_time": scheduled_time,
            "audit_type": audit_type,
            "trigger": trigger,
        }

        event_id = self._emit("audit.scheduled", payload)
        payload["event_id"] = event_id
        return payload

    def emit_completed(
        self,
        audit_id: str,
        report: dict | None = None,
        report_path: str | None = None,
    ) -> dict:
        """Emit audit.completed event.

        Args:
            audit_id: UUID of the audit
            report: Full audit report dict
            report_path: Path to saved report file

        Returns:
            Event payload dict with event_id
        """
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        payload = {
            "event_type": "audit.completed",
            "audit_id": audit_id,
            "timestamp": timestamp,
        }

        if report:
            payload["summary"] = report.get("summary", {})
            payload["findings_count"] = len(report.get("findings", []))
            payload["audit_type"] = report.get("audit_type", "unknown")

        if report_path:
            payload["report_path"] = report_path

        event_id = self._emit("audit.completed", payload)
        payload["event_id"] = event_id
        return payload

    def emit_findings(
        self,
        audit_id: str,
        findings: list[dict],
    ) -> dict:
        """Emit audit.findings event (batch findings report).

        Args:
            audit_id: UUID of the audit
            findings: List of finding dicts

        Returns:
            Event payload dict with event_id
        """
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Summarize by severity
        by_severity = {}
        for f in findings:
            sev = f.get("severity", "unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1

        payload = {
            "event_type": "audit.findings",
            "audit_id": audit_id,
            "timestamp": timestamp,
            "total": len(findings),
            "by_severity": by_severity,
            "findings": findings,
        }

        # Determine level based on critical findings
        level = "info"
        if by_severity.get("critical", 0) > 0:
            level = "error"
        elif by_severity.get("high", 0) > 0:
            level = "warn"

        event_id = self._emit("audit.findings", payload, level=level)
        payload["event_id"] = event_id
        return payload


def main():
    parser = argparse.ArgumentParser(
        description="Audit Bus Events Emitter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Schedule command
    schedule_parser = subparsers.add_parser("schedule", help="Emit audit.scheduled event")
    schedule_parser.add_argument("--type", choices=["quick", "full", "deep"], default="quick")
    schedule_parser.add_argument("--trigger", choices=["manual", "cron", "ci", "omega-loop"], default="manual")
    schedule_parser.add_argument("--time", help="ISO8601 scheduled time")

    # Complete command
    complete_parser = subparsers.add_parser("complete", help="Emit audit.completed event")
    complete_parser.add_argument("--audit-id", required=True, help="Audit UUID")
    complete_parser.add_argument("--report-path", help="Path to report JSON")

    # Findings command
    findings_parser = subparsers.add_parser("findings", help="Emit audit.findings event")
    findings_parser.add_argument("--audit-id", required=True, help="Audit UUID")
    findings_parser.add_argument("--findings-json", required=True, help="Path to findings JSON")

    args = parser.parse_args()

    emitter = AuditBusEmitter()

    if args.command == "schedule":
        result = emitter.emit_scheduled(
            audit_type=args.type,
            trigger=args.trigger,
            scheduled_time=args.time,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "complete":
        report = None
        if args.report_path and Path(args.report_path).exists():
            with open(args.report_path) as f:
                report = json.load(f)

        result = emitter.emit_completed(
            audit_id=args.audit_id,
            report=report,
            report_path=args.report_path,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "findings":
        with open(args.findings_json) as f:
            findings = json.load(f)

        if isinstance(findings, dict) and "findings" in findings:
            findings = findings["findings"]

        result = emitter.emit_findings(
            audit_id=args.audit_id,
            findings=findings,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
