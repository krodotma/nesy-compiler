#!/usr/bin/env python3
"""
tracker.py - Deployment History Tracker (Step 226)

PBTSO Phase: DISTILL
A2A Integration: Tracks deployment history via deploy.history.*

Provides:
- DeploymentStatus: Deployment status enum
- HistoryEventType: Types of history events
- DeploymentEvent: Individual deployment event
- DeploymentRecord: Complete deployment record
- DeploymentHistoryTracker: Main tracker class

Bus Topics:
- deploy.history.record
- deploy.history.event
- deploy.history.query

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
from typing import Any, Dict, List, Optional, Iterator
import sqlite3


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
    actor: str = "history-tracker"
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

class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class HistoryEventType(Enum):
    """Types of history events."""
    STARTED = "started"
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    PHASE_FAILED = "phase_failed"
    HEALTH_CHECK = "health_check"
    TRAFFIC_SHIFT = "traffic_shift"
    ROLLBACK_STARTED = "rollback_started"
    ROLLBACK_COMPLETED = "rollback_completed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_REJECTED = "approval_rejected"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CUSTOM = "custom"


@dataclass
class DeploymentEvent:
    """
    Individual deployment event.

    Attributes:
        event_id: Unique event identifier
        deployment_id: Parent deployment
        event_type: Type of event
        timestamp: Event timestamp
        phase: Deployment phase
        message: Event message
        details: Event details
        actor: Who/what triggered event
    """
    event_id: str
    deployment_id: str
    event_type: HistoryEventType
    timestamp: float = field(default_factory=time.time)
    phase: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    actor: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "deployment_id": self.deployment_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "phase": self.phase,
            "message": self.message,
            "details": self.details,
            "actor": self.actor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentEvent":
        data = dict(data)
        if "event_type" in data:
            data["event_type"] = HistoryEventType(data["event_type"])
        if "details" in data and isinstance(data["details"], str):
            data["details"] = json.loads(data["details"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DeploymentRecord:
    """
    Complete deployment record.

    Attributes:
        deployment_id: Unique deployment identifier
        service_name: Service deployed
        version: Version deployed
        previous_version: Previous version
        environment: Target environment
        strategy: Deployment strategy used
        status: Final status
        started_at: Start timestamp
        completed_at: Completion timestamp
        duration_ms: Total duration
        events: Deployment events
        initiated_by: Who initiated deployment
        approved_by: Who approved (if applicable)
        rollback_of: ID of deployment this rolled back
        rolled_back_by: ID of deployment that rolled this back
        config: Deployment configuration
        metadata: Additional metadata
    """
    deployment_id: str
    service_name: str
    version: str
    previous_version: str = ""
    environment: str = "prod"
    strategy: str = "blue_green"
    status: DeploymentStatus = DeploymentStatus.PENDING
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    duration_ms: float = 0.0
    events: List[DeploymentEvent] = field(default_factory=list)
    initiated_by: str = ""
    approved_by: str = ""
    rollback_of: str = ""
    rolled_back_by: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "service_name": self.service_name,
            "version": self.version,
            "previous_version": self.previous_version,
            "environment": self.environment,
            "strategy": self.strategy,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "events": [e.to_dict() for e in self.events],
            "initiated_by": self.initiated_by,
            "approved_by": self.approved_by,
            "rollback_of": self.rollback_of,
            "rolled_back_by": self.rolled_back_by,
            "config": self.config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentRecord":
        data = dict(data)
        if "status" in data:
            data["status"] = DeploymentStatus(data["status"])
        if "events" in data:
            data["events"] = [
                DeploymentEvent.from_dict(e) if isinstance(e, dict) else e
                for e in data["events"]
            ]
        if "config" in data and isinstance(data["config"], str):
            data["config"] = json.loads(data["config"])
        if "metadata" in data and isinstance(data["metadata"], str):
            data["metadata"] = json.loads(data["metadata"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==============================================================================
# Deployment History Tracker (Step 226)
# ==============================================================================

class DeploymentHistoryTracker:
    """
    Deployment History Tracker - tracks deployment history and audit trail.

    PBTSO Phase: DISTILL

    Responsibilities:
    - Record complete deployment history
    - Track deployment events and phases
    - Maintain audit trail
    - Query historical deployments
    - Calculate deployment statistics

    Example:
        >>> tracker = DeploymentHistoryTracker()
        >>> record = tracker.start_deployment(
        ...     service_name="api",
        ...     version="v2.0.0",
        ...     environment="prod",
        ... )
        >>> tracker.add_event(record.deployment_id, HistoryEventType.PHASE_COMPLETED, ...)
        >>> tracker.complete_deployment(record.deployment_id, DeploymentStatus.SUCCEEDED)
    """

    BUS_TOPICS = {
        "record": "deploy.history.record",
        "event": "deploy.history.event",
        "query": "deploy.history.query",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "history-tracker",
        use_sqlite: bool = True,
    ):
        """
        Initialize the history tracker.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
            use_sqlite: Use SQLite for storage (recommended for large histories)
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "history"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self.use_sqlite = use_sqlite

        self._deployments: Dict[str, DeploymentRecord] = {}
        self._db_conn: Optional[sqlite3.Connection] = None

        if use_sqlite:
            self._init_sqlite()
        else:
            self._load_state()

    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        db_path = self.state_dir / "history.db"
        self._db_conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db_conn.row_factory = sqlite3.Row

        self._db_conn.executescript("""
            CREATE TABLE IF NOT EXISTS deployments (
                deployment_id TEXT PRIMARY KEY,
                service_name TEXT NOT NULL,
                version TEXT NOT NULL,
                previous_version TEXT,
                environment TEXT NOT NULL,
                strategy TEXT,
                status TEXT NOT NULL,
                started_at REAL NOT NULL,
                completed_at REAL,
                duration_ms REAL,
                initiated_by TEXT,
                approved_by TEXT,
                rollback_of TEXT,
                rolled_back_by TEXT,
                config TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS deployment_events (
                event_id TEXT PRIMARY KEY,
                deployment_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                phase TEXT,
                message TEXT,
                details TEXT,
                actor TEXT,
                FOREIGN KEY (deployment_id) REFERENCES deployments(deployment_id)
            );

            CREATE INDEX IF NOT EXISTS idx_deployments_service
                ON deployments(service_name, environment);
            CREATE INDEX IF NOT EXISTS idx_deployments_started
                ON deployments(started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_events_deployment
                ON deployment_events(deployment_id);
        """)
        self._db_conn.commit()

    def start_deployment(
        self,
        service_name: str,
        version: str,
        environment: str,
        previous_version: str = "",
        strategy: str = "blue_green",
        initiated_by: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> DeploymentRecord:
        """
        Start tracking a new deployment.

        Args:
            service_name: Service being deployed
            version: Version to deploy
            environment: Target environment
            previous_version: Previous version
            strategy: Deployment strategy
            initiated_by: Who initiated deployment
            config: Deployment configuration

        Returns:
            Created DeploymentRecord
        """
        deployment_id = f"deploy-{uuid.uuid4().hex[:12]}"

        record = DeploymentRecord(
            deployment_id=deployment_id,
            service_name=service_name,
            version=version,
            previous_version=previous_version,
            environment=environment,
            strategy=strategy,
            status=DeploymentStatus.RUNNING,
            initiated_by=initiated_by,
            config=config or {},
        )

        # Add started event
        event = DeploymentEvent(
            event_id=f"event-{uuid.uuid4().hex[:12]}",
            deployment_id=deployment_id,
            event_type=HistoryEventType.STARTED,
            message=f"Deployment of {service_name} {version} to {environment} started",
            actor=initiated_by,
        )
        record.events.append(event)

        self._save_deployment(record)

        _emit_bus_event(
            self.BUS_TOPICS["record"],
            {
                "deployment_id": deployment_id,
                "service_name": service_name,
                "version": version,
                "environment": environment,
                "action": "started",
            },
            actor=self.actor_id,
        )

        return record

    def add_event(
        self,
        deployment_id: str,
        event_type: HistoryEventType,
        message: str = "",
        phase: str = "",
        details: Optional[Dict[str, Any]] = None,
        actor: str = "",
    ) -> DeploymentEvent:
        """
        Add an event to a deployment.

        Args:
            deployment_id: Deployment to add event to
            event_type: Type of event
            message: Event message
            phase: Current phase
            details: Event details
            actor: Who/what triggered event

        Returns:
            Created DeploymentEvent
        """
        event = DeploymentEvent(
            event_id=f"event-{uuid.uuid4().hex[:12]}",
            deployment_id=deployment_id,
            event_type=event_type,
            message=message,
            phase=phase,
            details=details or {},
            actor=actor,
        )

        if self.use_sqlite:
            self._save_event_sqlite(event)
        else:
            record = self._deployments.get(deployment_id)
            if record:
                record.events.append(event)
                self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["event"],
            {
                "deployment_id": deployment_id,
                "event_type": event_type.value,
                "phase": phase,
                "message": message,
            },
            actor=self.actor_id,
        )

        return event

    def complete_deployment(
        self,
        deployment_id: str,
        status: DeploymentStatus,
        message: str = "",
    ) -> Optional[DeploymentRecord]:
        """
        Mark a deployment as complete.

        Args:
            deployment_id: Deployment to complete
            status: Final status
            message: Completion message

        Returns:
            Updated DeploymentRecord or None
        """
        record = self.get_deployment(deployment_id)
        if not record:
            return None

        record.status = status
        record.completed_at = time.time()
        record.duration_ms = (record.completed_at - record.started_at) * 1000

        # Add completion event
        event_type = (
            HistoryEventType.COMPLETED if status == DeploymentStatus.SUCCEEDED
            else HistoryEventType.FAILED if status == DeploymentStatus.FAILED
            else HistoryEventType.CANCELLED
        )

        event = DeploymentEvent(
            event_id=f"event-{uuid.uuid4().hex[:12]}",
            deployment_id=deployment_id,
            event_type=event_type,
            message=message or f"Deployment {status.value}",
        )
        record.events.append(event)

        self._save_deployment(record)

        _emit_bus_event(
            self.BUS_TOPICS["record"],
            {
                "deployment_id": deployment_id,
                "service_name": record.service_name,
                "version": record.version,
                "status": status.value,
                "duration_ms": record.duration_ms,
                "action": "completed",
            },
            actor=self.actor_id,
        )

        return record

    def record_rollback(
        self,
        original_deployment_id: str,
        rollback_to_version: str,
        initiated_by: str = "",
    ) -> Optional[DeploymentRecord]:
        """
        Record a rollback deployment.

        Args:
            original_deployment_id: Deployment being rolled back
            rollback_to_version: Version to roll back to
            initiated_by: Who initiated rollback

        Returns:
            Created rollback DeploymentRecord
        """
        original = self.get_deployment(original_deployment_id)
        if not original:
            return None

        rollback = self.start_deployment(
            service_name=original.service_name,
            version=rollback_to_version,
            environment=original.environment,
            previous_version=original.version,
            strategy="rollback",
            initiated_by=initiated_by,
        )

        rollback.rollback_of = original_deployment_id

        # Update original
        original.rolled_back_by = rollback.deployment_id
        self._save_deployment(original)

        self.add_event(
            original_deployment_id,
            HistoryEventType.ROLLBACK_STARTED,
            message=f"Rollback initiated to version {rollback_to_version}",
            actor=initiated_by,
        )

        return rollback

    def get_deployment(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """Get a deployment by ID."""
        if self.use_sqlite:
            return self._get_deployment_sqlite(deployment_id)
        return self._deployments.get(deployment_id)

    def _get_deployment_sqlite(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """Get deployment from SQLite."""
        cursor = self._db_conn.execute(
            "SELECT * FROM deployments WHERE deployment_id = ?",
            (deployment_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        record = DeploymentRecord(
            deployment_id=row["deployment_id"],
            service_name=row["service_name"],
            version=row["version"],
            previous_version=row["previous_version"] or "",
            environment=row["environment"],
            strategy=row["strategy"] or "",
            status=DeploymentStatus(row["status"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"] or 0.0,
            duration_ms=row["duration_ms"] or 0.0,
            initiated_by=row["initiated_by"] or "",
            approved_by=row["approved_by"] or "",
            rollback_of=row["rollback_of"] or "",
            rolled_back_by=row["rolled_back_by"] or "",
            config=json.loads(row["config"]) if row["config"] else {},
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

        # Load events
        events_cursor = self._db_conn.execute(
            "SELECT * FROM deployment_events WHERE deployment_id = ? ORDER BY timestamp",
            (deployment_id,)
        )
        for event_row in events_cursor:
            record.events.append(DeploymentEvent(
                event_id=event_row["event_id"],
                deployment_id=event_row["deployment_id"],
                event_type=HistoryEventType(event_row["event_type"]),
                timestamp=event_row["timestamp"],
                phase=event_row["phase"] or "",
                message=event_row["message"] or "",
                details=json.loads(event_row["details"]) if event_row["details"] else {},
                actor=event_row["actor"] or "",
            ))

        return record

    def query_deployments(
        self,
        service_name: Optional[str] = None,
        environment: Optional[str] = None,
        status: Optional[DeploymentStatus] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DeploymentRecord]:
        """
        Query deployment history.

        Args:
            service_name: Filter by service
            environment: Filter by environment
            status: Filter by status
            start_time: Start timestamp
            end_time: End timestamp
            limit: Maximum results
            offset: Result offset

        Returns:
            List of matching DeploymentRecords
        """
        if self.use_sqlite:
            return self._query_sqlite(
                service_name, environment, status,
                start_time, end_time, limit, offset
            )

        deployments = list(self._deployments.values())

        if service_name:
            deployments = [d for d in deployments if d.service_name == service_name]
        if environment:
            deployments = [d for d in deployments if d.environment == environment]
        if status:
            deployments = [d for d in deployments if d.status == status]
        if start_time:
            deployments = [d for d in deployments if d.started_at >= start_time]
        if end_time:
            deployments = [d for d in deployments if d.started_at <= end_time]

        deployments.sort(key=lambda d: d.started_at, reverse=True)
        return deployments[offset:offset + limit]

    def _query_sqlite(
        self,
        service_name: Optional[str],
        environment: Optional[str],
        status: Optional[DeploymentStatus],
        start_time: Optional[float],
        end_time: Optional[float],
        limit: int,
        offset: int,
    ) -> List[DeploymentRecord]:
        """Query deployments from SQLite."""
        conditions = []
        params: List[Any] = []

        if service_name:
            conditions.append("service_name = ?")
            params.append(service_name)
        if environment:
            conditions.append("environment = ?")
            params.append(environment)
        if status:
            conditions.append("status = ?")
            params.append(status.value)
        if start_time:
            conditions.append("started_at >= ?")
            params.append(start_time)
        if end_time:
            conditions.append("started_at <= ?")
            params.append(end_time)

        query = "SELECT deployment_id FROM deployments"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = self._db_conn.execute(query, params)
        deployment_ids = [row[0] for row in cursor.fetchall()]

        return [
            self._get_deployment_sqlite(did)
            for did in deployment_ids
            if self._get_deployment_sqlite(did)
        ]

    def get_latest(
        self,
        service_name: str,
        environment: str,
    ) -> Optional[DeploymentRecord]:
        """Get the latest deployment for a service/environment."""
        deployments = self.query_deployments(
            service_name=service_name,
            environment=environment,
            limit=1,
        )
        return deployments[0] if deployments else None

    def get_current_version(
        self,
        service_name: str,
        environment: str,
    ) -> Optional[str]:
        """Get the currently deployed version."""
        latest = self.get_latest(service_name, environment)
        if latest and latest.status == DeploymentStatus.SUCCEEDED:
            return latest.version
        return None

    def get_statistics(
        self,
        service_name: Optional[str] = None,
        environment: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get deployment statistics.

        Args:
            service_name: Filter by service
            environment: Filter by environment
            days: Number of days to analyze

        Returns:
            Statistics dictionary
        """
        start_time = time.time() - (days * 86400)
        deployments = self.query_deployments(
            service_name=service_name,
            environment=environment,
            start_time=start_time,
            limit=10000,
        )

        if not deployments:
            return {
                "total": 0,
                "succeeded": 0,
                "failed": 0,
                "rolled_back": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "deployments_per_day": 0.0,
            }

        succeeded = sum(1 for d in deployments if d.status == DeploymentStatus.SUCCEEDED)
        failed = sum(1 for d in deployments if d.status == DeploymentStatus.FAILED)
        rolled_back = sum(1 for d in deployments if d.status == DeploymentStatus.ROLLED_BACK)

        durations = [d.duration_ms for d in deployments if d.duration_ms > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total": len(deployments),
            "succeeded": succeeded,
            "failed": failed,
            "rolled_back": rolled_back,
            "cancelled": sum(1 for d in deployments if d.status == DeploymentStatus.CANCELLED),
            "success_rate": (succeeded / len(deployments)) * 100 if deployments else 0,
            "rollback_rate": (rolled_back / len(deployments)) * 100 if deployments else 0,
            "avg_duration_ms": avg_duration,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "deployments_per_day": len(deployments) / days,
            "period_days": days,
        }

    def get_timeline(
        self,
        deployment_id: str,
    ) -> List[DeploymentEvent]:
        """Get timeline of events for a deployment."""
        record = self.get_deployment(deployment_id)
        if not record:
            return []
        return sorted(record.events, key=lambda e: e.timestamp)

    def _save_deployment(self, record: DeploymentRecord) -> None:
        """Save deployment record."""
        if self.use_sqlite:
            self._save_deployment_sqlite(record)
        else:
            self._deployments[record.deployment_id] = record
            self._save_state()

    def _save_deployment_sqlite(self, record: DeploymentRecord) -> None:
        """Save deployment to SQLite."""
        self._db_conn.execute("""
            INSERT OR REPLACE INTO deployments
            (deployment_id, service_name, version, previous_version, environment,
             strategy, status, started_at, completed_at, duration_ms,
             initiated_by, approved_by, rollback_of, rolled_back_by, config, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.deployment_id,
            record.service_name,
            record.version,
            record.previous_version,
            record.environment,
            record.strategy,
            record.status.value,
            record.started_at,
            record.completed_at,
            record.duration_ms,
            record.initiated_by,
            record.approved_by,
            record.rollback_of,
            record.rolled_back_by,
            json.dumps(record.config),
            json.dumps(record.metadata),
        ))

        # Save events
        for event in record.events:
            self._save_event_sqlite(event)

        self._db_conn.commit()

    def _save_event_sqlite(self, event: DeploymentEvent) -> None:
        """Save event to SQLite."""
        self._db_conn.execute("""
            INSERT OR REPLACE INTO deployment_events
            (event_id, deployment_id, event_type, timestamp, phase, message, details, actor)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.deployment_id,
            event.event_type.value,
            event.timestamp,
            event.phase,
            event.message,
            json.dumps(event.details),
            event.actor,
        ))
        self._db_conn.commit()

    def _save_state(self) -> None:
        """Save state to JSON (non-SQLite mode)."""
        state = {
            "deployments": {
                did: d.to_dict()
                for did, d in list(self._deployments.items())[-1000:]
            }
        }
        state_file = self.state_dir / "history_state.json"
        with open(state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load state from JSON (non-SQLite mode)."""
        state_file = self.state_dir / "history_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            for did, data in state.get("deployments", {}).items():
                self._deployments[did] = DeploymentRecord.from_dict(data)
        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for history tracker."""
    import argparse

    parser = argparse.ArgumentParser(description="Deployment History Tracker (Step 226)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # record command
    record_parser = subparsers.add_parser("record", help="Start recording a deployment")
    record_parser.add_argument("--service", "-s", required=True, help="Service name")
    record_parser.add_argument("--version", "-v", required=True, help="Version")
    record_parser.add_argument("--env", "-e", default="prod", help="Environment")
    record_parser.add_argument("--strategy", default="blue_green", help="Strategy")
    record_parser.add_argument("--by", default="", help="Initiated by")
    record_parser.add_argument("--json", action="store_true", help="JSON output")

    # complete command
    complete_parser = subparsers.add_parser("complete", help="Complete a deployment")
    complete_parser.add_argument("deployment_id", help="Deployment ID")
    complete_parser.add_argument("--status", "-s", required=True,
                                  choices=["succeeded", "failed", "cancelled"])
    complete_parser.add_argument("--message", "-m", default="", help="Message")
    complete_parser.add_argument("--json", action="store_true", help="JSON output")

    # get command
    get_parser = subparsers.add_parser("get", help="Get deployment details")
    get_parser.add_argument("deployment_id", help="Deployment ID")
    get_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List deployments")
    list_parser.add_argument("--service", "-s", help="Filter by service")
    list_parser.add_argument("--env", "-e", help="Filter by environment")
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument("--limit", "-l", type=int, default=10, help="Limit")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Get statistics")
    stats_parser.add_argument("--service", "-s", help="Filter by service")
    stats_parser.add_argument("--env", "-e", help="Filter by environment")
    stats_parser.add_argument("--days", "-d", type=int, default=30, help="Days")
    stats_parser.add_argument("--json", action="store_true", help="JSON output")

    # timeline command
    timeline_parser = subparsers.add_parser("timeline", help="Get deployment timeline")
    timeline_parser.add_argument("deployment_id", help="Deployment ID")
    timeline_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    tracker = DeploymentHistoryTracker()

    if args.command == "record":
        record = tracker.start_deployment(
            service_name=args.service,
            version=args.version,
            environment=args.env,
            strategy=args.strategy,
            initiated_by=args.by,
        )

        if args.json:
            print(json.dumps(record.to_dict(), indent=2))
        else:
            print(f"Started: {record.deployment_id}")
            print(f"  Service: {record.service_name}")
            print(f"  Version: {record.version}")
            print(f"  Environment: {record.environment}")

        return 0

    elif args.command == "complete":
        record = tracker.complete_deployment(
            deployment_id=args.deployment_id,
            status=DeploymentStatus(args.status.upper()),
            message=args.message,
        )

        if not record:
            print(f"Deployment not found: {args.deployment_id}")
            return 1

        if args.json:
            print(json.dumps(record.to_dict(), indent=2))
        else:
            print(f"Completed: {record.deployment_id}")
            print(f"  Status: {record.status.value}")
            print(f"  Duration: {record.duration_ms:.0f}ms")

        return 0

    elif args.command == "get":
        record = tracker.get_deployment(args.deployment_id)

        if not record:
            print(f"Deployment not found: {args.deployment_id}")
            return 1

        if args.json:
            print(json.dumps(record.to_dict(), indent=2))
        else:
            print(f"Deployment: {record.deployment_id}")
            print(f"  Service: {record.service_name}")
            print(f"  Version: {record.version}")
            print(f"  Environment: {record.environment}")
            print(f"  Status: {record.status.value}")
            print(f"  Events: {len(record.events)}")

        return 0

    elif args.command == "list":
        status = DeploymentStatus(args.status.upper()) if args.status else None
        deployments = tracker.query_deployments(
            service_name=args.service,
            environment=args.env,
            status=status,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps([d.to_dict() for d in deployments], indent=2))
        else:
            for d in deployments:
                ts = datetime.fromtimestamp(d.started_at).strftime("%Y-%m-%d %H:%M")
                print(f"{d.deployment_id} ({d.service_name}:{d.version}) - {d.status.value} - {ts}")

        return 0

    elif args.command == "stats":
        stats = tracker.get_statistics(
            service_name=args.service,
            environment=args.env,
            days=args.days,
        )

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Deployment Statistics ({args.days} days)")
            print(f"  Total: {stats['total']}")
            print(f"  Succeeded: {stats['succeeded']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Success Rate: {stats['success_rate']:.1f}%")
            print(f"  Avg Duration: {stats['avg_duration_ms']:.0f}ms")
            print(f"  Per Day: {stats['deployments_per_day']:.1f}")

        return 0

    elif args.command == "timeline":
        events = tracker.get_timeline(args.deployment_id)

        if args.json:
            print(json.dumps([e.to_dict() for e in events], indent=2))
        else:
            if not events:
                print("No events found")
            else:
                for e in events:
                    ts = datetime.fromtimestamp(e.timestamp).strftime("%H:%M:%S")
                    print(f"[{ts}] {e.event_type.value}: {e.message}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
