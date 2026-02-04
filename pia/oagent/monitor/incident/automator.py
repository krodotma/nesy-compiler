#!/usr/bin/env python3
"""
Incident Response Automator - Step 260

Automates incident response based on alerts and playbooks.

PBTSO Phase: ITERATE

Bus Topics:
- monitor.incident.create (subscribed/emitted)
- monitor.incident.escalate (emitted)
- monitor.incident.resolve (subscribed/emitted)

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ..alerts.manager import Alert, AlertManager, AlertState, get_manager as get_alert_manager
from ..anomaly.detector import AnomalySeverity


class IncidentState(Enum):
    """Incident lifecycle states."""
    TRIGGERED = "triggered"     # Initial state
    ACKNOWLEDGED = "acknowledged"  # Someone is working on it
    MITIGATING = "mitigating"   # Mitigation in progress
    RESOLVED = "resolved"       # Issue resolved
    POSTMORTEM = "postmortem"   # In postmortem phase


class IncidentSeverity(Enum):
    """Incident severity levels."""
    SEV1 = "sev1"  # Critical - immediate business impact
    SEV2 = "sev2"  # Major - significant impact
    SEV3 = "sev3"  # Minor - limited impact
    SEV4 = "sev4"  # Low - minimal impact


@dataclass
class ResponseAction:
    """An automated response action.

    Attributes:
        action_id: Unique action ID
        name: Action name
        description: What the action does
        handler: Handler function name or reference
        params: Action parameters
        auto_execute: Whether to execute automatically
        requires_approval: Whether approval is needed
        timeout_s: Action timeout
        executed: Whether action was executed
        result: Execution result
    """
    action_id: str
    name: str
    description: str
    handler: str
    params: Dict[str, Any] = field(default_factory=dict)
    auto_execute: bool = False
    requires_approval: bool = True
    timeout_s: int = 300
    executed: bool = False
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_id": self.action_id,
            "name": self.name,
            "description": self.description,
            "handler": self.handler,
            "params": self.params,
            "auto_execute": self.auto_execute,
            "requires_approval": self.requires_approval,
            "timeout_s": self.timeout_s,
            "executed": self.executed,
            "result": self.result,
        }


@dataclass
class ResponsePlaybook:
    """Playbook for incident response.

    Attributes:
        playbook_id: Unique playbook ID
        name: Playbook name
        description: Playbook description
        trigger_pattern: Pattern to match for triggering
        severity_threshold: Minimum severity to trigger
        actions: Ordered list of response actions
        auto_escalate_after_s: Auto-escalate after this duration
        enabled: Whether playbook is active
    """
    playbook_id: str
    name: str
    description: str
    trigger_pattern: str
    severity_threshold: IncidentSeverity = IncidentSeverity.SEV3
    actions: List[ResponseAction] = field(default_factory=list)
    auto_escalate_after_s: int = 1800  # 30 minutes
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "playbook_id": self.playbook_id,
            "name": self.name,
            "description": self.description,
            "trigger_pattern": self.trigger_pattern,
            "severity_threshold": self.severity_threshold.value,
            "actions": [a.to_dict() for a in self.actions],
            "auto_escalate_after_s": self.auto_escalate_after_s,
            "enabled": self.enabled,
        }


@dataclass
class Incident:
    """A managed incident.

    Attributes:
        incident_id: Unique incident ID
        title: Incident title
        description: Incident description
        severity: Incident severity
        state: Current state
        source_alerts: Source alert IDs
        affected_services: Affected service names
        playbook_id: Associated playbook
        actions_executed: Executed actions
        timeline: Event timeline
        created_at: Creation timestamp
        updated_at: Last update timestamp
        acknowledged_at: Acknowledgment timestamp
        acknowledged_by: Who acknowledged
        resolved_at: Resolution timestamp
        resolved_by: Who resolved
        postmortem_url: Link to postmortem
        escalation_level: Current escalation level
    """
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    state: IncidentState = IncidentState.TRIGGERED
    source_alerts: List[str] = field(default_factory=list)
    affected_services: Set[str] = field(default_factory=set)
    playbook_id: Optional[str] = None
    actions_executed: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None
    postmortem_url: Optional[str] = None
    escalation_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "state": self.state.value,
            "source_alerts": self.source_alerts,
            "affected_services": list(self.affected_services),
            "playbook_id": self.playbook_id,
            "actions_executed": self.actions_executed,
            "timeline": self.timeline,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "postmortem_url": self.postmortem_url,
            "escalation_level": self.escalation_level,
            "age_s": time.time() - self.created_at,
        }

    def add_timeline_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Add event to timeline."""
        self.timeline.append({
            "timestamp": time.time(),
            "type": event_type,
            "details": details,
        })
        self.updated_at = time.time()


class IncidentAutomator:
    """
    Automate incident response.

    The automator:
    - Creates incidents from critical alerts
    - Executes response playbooks
    - Manages incident lifecycle
    - Handles escalations

    Example:
        automator = IncidentAutomator()

        # Register playbook
        playbook = ResponsePlaybook(
            playbook_id="pb-001",
            name="High CPU Response",
            trigger_pattern="cpu.*",
            actions=[
                ResponseAction(
                    action_id="act-001",
                    name="Notify oncall",
                    handler="notify_slack",
                )
            ]
        )
        automator.register_playbook(playbook)

        # Create incident from alert
        incident = automator.create_from_alert(alert)

        # Run playbook
        automator.run_playbook(incident)
    """

    BUS_TOPICS = {
        "create": "monitor.incident.create",
        "escalate": "monitor.incident.escalate",
        "resolve": "monitor.incident.resolve",
        "action": "monitor.incident.action",
    }

    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        bus_dir: Optional[str] = None
    ):
        """Initialize incident automator.

        Args:
            alert_manager: Alert manager to monitor
            bus_dir: Directory for bus events
        """
        self._alert_manager = alert_manager or get_alert_manager()
        self._incidents: Dict[str, Incident] = {}
        self._playbooks: Dict[str, ResponsePlaybook] = {}
        self._action_handlers: Dict[str, Callable[[ResponseAction, Incident], Dict[str, Any]]] = {}
        self._lock = threading.RLock()

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register default handlers
        self._register_default_handlers()

    def register_playbook(self, playbook: ResponsePlaybook) -> None:
        """Register a response playbook.

        Args:
            playbook: Playbook to register
        """
        self._playbooks[playbook.playbook_id] = playbook

    def register_action_handler(
        self,
        handler_name: str,
        handler: Callable[[ResponseAction, Incident], Dict[str, Any]]
    ) -> None:
        """Register an action handler.

        Args:
            handler_name: Handler name
            handler: Handler function
        """
        self._action_handlers[handler_name] = handler

    def create(
        self,
        title: str,
        description: str,
        severity: IncidentSeverity,
        affected_services: Optional[Set[str]] = None
    ) -> Incident:
        """Create a new incident.

        Args:
            title: Incident title
            description: Description
            severity: Severity level
            affected_services: Affected services

        Returns:
            Created incident
        """
        with self._lock:
            incident = Incident(
                incident_id=f"inc-{uuid.uuid4().hex[:8]}",
                title=title,
                description=description,
                severity=severity,
                affected_services=affected_services or set(),
            )

            incident.add_timeline_event("created", {
                "title": title,
                "severity": severity.value,
            })

            self._incidents[incident.incident_id] = incident

        # Emit creation event
        self._emit_incident_event("create", incident)

        # Find and run matching playbook
        playbook = self._find_matching_playbook(incident)
        if playbook:
            incident.playbook_id = playbook.playbook_id
            self.run_playbook(incident, playbook)

        return incident

    def create_from_alert(self, alert: Alert) -> Incident:
        """Create incident from alert.

        Args:
            alert: Source alert

        Returns:
            Created incident
        """
        severity = self._alert_severity_to_incident(alert.severity)

        affected = set()
        if "service" in alert.labels:
            affected.add(alert.labels["service"])
        if "actor" in alert.labels:
            affected.add(alert.labels["actor"])

        incident = self.create(
            title=f"Alert: {alert.metric_name}",
            description=alert.message,
            severity=severity,
            affected_services=affected,
        )

        incident.source_alerts.append(alert.alert_id)
        return incident

    def acknowledge(
        self,
        incident_id: str,
        acknowledged_by: str
    ) -> bool:
        """Acknowledge an incident.

        Args:
            incident_id: Incident ID
            acknowledged_by: Who acknowledged

        Returns:
            True if acknowledged
        """
        with self._lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return False

            if incident.state != IncidentState.TRIGGERED:
                return False

            incident.state = IncidentState.ACKNOWLEDGED
            incident.acknowledged_at = time.time()
            incident.acknowledged_by = acknowledged_by

            incident.add_timeline_event("acknowledged", {
                "by": acknowledged_by,
            })

        self._emit_incident_event("acknowledge", incident)
        return True

    def start_mitigation(self, incident_id: str) -> bool:
        """Start mitigation phase.

        Args:
            incident_id: Incident ID

        Returns:
            True if started
        """
        with self._lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return False

            incident.state = IncidentState.MITIGATING
            incident.add_timeline_event("mitigation_started", {})

        return True

    def resolve(
        self,
        incident_id: str,
        resolved_by: str,
        postmortem_url: Optional[str] = None
    ) -> bool:
        """Resolve an incident.

        Args:
            incident_id: Incident ID
            resolved_by: Who resolved
            postmortem_url: Link to postmortem

        Returns:
            True if resolved
        """
        with self._lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return False

            incident.state = IncidentState.RESOLVED
            incident.resolved_at = time.time()
            incident.resolved_by = resolved_by
            incident.postmortem_url = postmortem_url

            incident.add_timeline_event("resolved", {
                "by": resolved_by,
                "postmortem_url": postmortem_url,
            })

        # Resolve associated alerts
        for alert_id in incident.source_alerts:
            self._alert_manager.resolve(
                alert_id,
                resolved_by=f"incident:{incident_id}"
            )

        self._emit_incident_event("resolve", incident)
        return True

    def escalate(self, incident_id: str, reason: str = "") -> bool:
        """Escalate an incident.

        Args:
            incident_id: Incident ID
            reason: Escalation reason

        Returns:
            True if escalated
        """
        with self._lock:
            incident = self._incidents.get(incident_id)
            if not incident:
                return False

            incident.escalation_level += 1
            incident.add_timeline_event("escalated", {
                "level": incident.escalation_level,
                "reason": reason,
            })

        self._emit_incident_event("escalate", incident, {"reason": reason})
        return True

    def run_playbook(
        self,
        incident: Incident,
        playbook: Optional[ResponsePlaybook] = None
    ) -> List[Dict[str, Any]]:
        """Run playbook actions for incident.

        Args:
            incident: Incident to respond to
            playbook: Playbook to run (uses incident's if not provided)

        Returns:
            List of action results
        """
        if playbook is None:
            playbook = self._playbooks.get(incident.playbook_id or "")
            if not playbook:
                return []

        results = []

        for action in playbook.actions:
            if action.requires_approval and not action.auto_execute:
                # Skip actions requiring approval
                incident.add_timeline_event("action_pending", {
                    "action": action.name,
                    "requires_approval": True,
                })
                continue

            result = self.execute_action(action, incident)
            results.append(result)

        return results

    def execute_action(
        self,
        action: ResponseAction,
        incident: Incident
    ) -> Dict[str, Any]:
        """Execute a response action.

        Args:
            action: Action to execute
            incident: Target incident

        Returns:
            Execution result
        """
        handler = self._action_handlers.get(action.handler)
        if not handler:
            result = {
                "success": False,
                "error": f"Unknown handler: {action.handler}",
            }
        else:
            try:
                result = handler(action, incident)
                result["success"] = True
            except Exception as e:
                result = {
                    "success": False,
                    "error": str(e),
                }

        action.executed = True
        action.result = result
        incident.actions_executed.append(action.action_id)

        incident.add_timeline_event("action_executed", {
            "action": action.name,
            "result": result,
        })

        self._emit_incident_event("action", incident, {
            "action": action.to_dict(),
            "result": result,
        })

        return result

    def get(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID.

        Args:
            incident_id: Incident ID

        Returns:
            Incident or None
        """
        return self._incidents.get(incident_id)

    def get_active(self) -> List[Incident]:
        """Get all active incidents.

        Returns:
            List of active incidents
        """
        with self._lock:
            return [
                i for i in self._incidents.values()
                if i.state not in (IncidentState.RESOLVED, IncidentState.POSTMORTEM)
            ]

    def get_playbooks(self) -> List[ResponsePlaybook]:
        """Get all registered playbooks.

        Returns:
            List of playbooks
        """
        return list(self._playbooks.values())

    def check_auto_escalation(self) -> int:
        """Check for incidents needing auto-escalation.

        Returns:
            Number of incidents escalated
        """
        escalated = 0
        now = time.time()

        with self._lock:
            for incident in self._incidents.values():
                if incident.state == IncidentState.TRIGGERED:
                    playbook = self._playbooks.get(incident.playbook_id or "")
                    if playbook:
                        age = now - incident.created_at
                        if age > playbook.auto_escalate_after_s:
                            self.escalate(incident.incident_id, "Auto-escalation timeout")
                            escalated += 1

        return escalated

    def handle_create_request(self, event: Dict[str, Any]) -> Optional[Incident]:
        """Handle create request from bus.

        Args:
            event: Bus event

        Returns:
            Created incident or None
        """
        data = event.get("data", {})
        return self.create(
            title=data.get("title", "Incident"),
            description=data.get("description", ""),
            severity=IncidentSeverity(data.get("severity", "sev3")),
            affected_services=set(data.get("affected_services", [])),
        )

    def handle_resolve_request(self, event: Dict[str, Any]) -> bool:
        """Handle resolve request from bus.

        Args:
            event: Bus event

        Returns:
            Success
        """
        data = event.get("data", {})
        return self.resolve(
            incident_id=data.get("incident_id", ""),
            resolved_by=data.get("resolved_by", event.get("actor", "unknown")),
            postmortem_url=data.get("postmortem_url"),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get automator statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            by_state = {}
            for state in IncidentState:
                by_state[state.value] = sum(
                    1 for i in self._incidents.values()
                    if i.state == state
                )

            by_severity = {}
            for sev in IncidentSeverity:
                by_severity[sev.value] = sum(
                    1 for i in self._incidents.values()
                    if i.severity == sev
                )

            return {
                "total_incidents": len(self._incidents),
                "by_state": by_state,
                "by_severity": by_severity,
                "registered_playbooks": len(self._playbooks),
                "active_incidents": len(self.get_active()),
            }

    def _find_matching_playbook(
        self,
        incident: Incident
    ) -> Optional[ResponsePlaybook]:
        """Find playbook matching incident."""
        import re

        for playbook in self._playbooks.values():
            if not playbook.enabled:
                continue

            # Check severity threshold
            if self._severity_level(incident.severity) < self._severity_level(playbook.severity_threshold):
                continue

            # Check pattern match
            pattern = re.compile(playbook.trigger_pattern)
            if pattern.search(incident.title) or pattern.search(incident.description):
                return playbook

            # Check affected services
            for service in incident.affected_services:
                if pattern.search(service):
                    return playbook

        return None

    def _alert_severity_to_incident(
        self,
        alert_severity: AnomalySeverity
    ) -> IncidentSeverity:
        """Map alert severity to incident severity."""
        mapping = {
            AnomalySeverity.INFO: IncidentSeverity.SEV4,
            AnomalySeverity.WARNING: IncidentSeverity.SEV3,
            AnomalySeverity.CRITICAL: IncidentSeverity.SEV1,
        }
        return mapping.get(alert_severity, IncidentSeverity.SEV3)

    def _severity_level(self, severity: IncidentSeverity) -> int:
        """Get numeric severity level."""
        levels = {
            IncidentSeverity.SEV4: 1,
            IncidentSeverity.SEV3: 2,
            IncidentSeverity.SEV2: 3,
            IncidentSeverity.SEV1: 4,
        }
        return levels.get(severity, 0)

    def _register_default_handlers(self) -> None:
        """Register default action handlers."""
        # Notify handler
        def notify_handler(action: ResponseAction, incident: Incident) -> Dict[str, Any]:
            return {
                "notified": True,
                "channel": action.params.get("channel", "bus"),
                "message": f"Incident {incident.incident_id}: {incident.title}",
            }

        # Log handler
        def log_handler(action: ResponseAction, incident: Incident) -> Dict[str, Any]:
            log_path = Path(self._bus_dir) / "incident_actions.log"
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "timestamp": time.time(),
                    "incident_id": incident.incident_id,
                    "action": action.name,
                }) + "\n")
            return {"logged": True}

        self._action_handlers["notify"] = notify_handler
        self._action_handlers["notify_slack"] = notify_handler
        self._action_handlers["log"] = log_handler

    def _emit_incident_event(
        self,
        event_type: str,
        incident: Incident,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Emit incident event to bus."""
        event_id = str(uuid.uuid4())
        level = (
            "error" if incident.severity in (IncidentSeverity.SEV1, IncidentSeverity.SEV2)
            else "warn"
        )

        data = {
            "incident_id": incident.incident_id,
            "title": incident.title,
            "severity": incident.severity.value,
            "state": incident.state.value,
        }
        if extra_data:
            data.update(extra_data)

        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": self.BUS_TOPICS.get(event_type, f"monitor.incident.{event_type}"),
            "kind": "incident",
            "level": level,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        with open(self._bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id


# Singleton instance
_automator: Optional[IncidentAutomator] = None


def get_automator() -> IncidentAutomator:
    """Get or create the incident automator singleton.

    Returns:
        IncidentAutomator instance
    """
    global _automator
    if _automator is None:
        _automator = IncidentAutomator()
    return _automator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Incident Automator (Step 260)")
    parser.add_argument("--create", metavar="TITLE", help="Create incident")
    parser.add_argument("--severity", default="sev3", help="Incident severity")
    parser.add_argument("--description", default="", help="Incident description")
    parser.add_argument("--ack", metavar="ID", help="Acknowledge incident")
    parser.add_argument("--resolve", metavar="ID", help="Resolve incident")
    parser.add_argument("--escalate", metavar="ID", help="Escalate incident")
    parser.add_argument("--by", default="cli", help="Actor name")
    parser.add_argument("--list", action="store_true", help="List active incidents")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    automator = get_automator()

    if args.create:
        incident = automator.create(
            title=args.create,
            description=args.description,
            severity=IncidentSeverity(args.severity),
        )
        if args.json:
            print(json.dumps(incident.to_dict(), indent=2))
        else:
            print(f"Created incident: {incident.incident_id}")
            print(f"  Title: {incident.title}")
            print(f"  Severity: {incident.severity.value}")
            print(f"  State: {incident.state.value}")

    if args.ack:
        success = automator.acknowledge(args.ack, args.by)
        print(f"Acknowledged: {success}")

    if args.resolve:
        success = automator.resolve(args.resolve, args.by)
        print(f"Resolved: {success}")

    if args.escalate:
        success = automator.escalate(args.escalate)
        print(f"Escalated: {success}")

    if args.list:
        incidents = automator.get_active()
        if args.json:
            print(json.dumps([i.to_dict() for i in incidents], indent=2))
        else:
            print(f"Active Incidents ({len(incidents)}):")
            for i in incidents:
                print(f"  [{i.severity.value}] {i.incident_id}: {i.title} ({i.state.value})")

    if args.stats:
        stats = automator.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Incident Automator Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
