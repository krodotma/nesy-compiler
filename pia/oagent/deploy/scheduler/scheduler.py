#!/usr/bin/env python3
"""
scheduler.py - Deployment Scheduler (Step 223)

PBTSO Phase: PLAN
A2A Integration: Schedules deployments via deploy.scheduler.*

Provides:
- ScheduleType: Types of schedules
- ScheduleStatus: Schedule status
- DeploymentWindow: Deployment time windows
- ScheduledDeployment: Scheduled deployment configuration
- DeploymentScheduler: Main scheduler class

Bus Topics:
- deploy.scheduler.create
- deploy.scheduler.trigger
- deploy.scheduler.cancel
- deploy.scheduler.update

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import asyncio
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
from typing import Any, Dict, List, Optional, Callable
import re


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
    actor: str = "deployment-scheduler"
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

class ScheduleType(Enum):
    """Types of deployment schedules."""
    ONCE = "once"           # One-time scheduled deployment
    CRON = "cron"           # Cron-based schedule
    INTERVAL = "interval"   # Fixed interval
    WINDOW = "window"       # During maintenance windows only
    MANUAL = "manual"       # Manual trigger only


class ScheduleStatus(Enum):
    """Schedule status."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    PAUSED = "paused"


class DayOfWeek(Enum):
    """Days of the week."""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


@dataclass
class DeploymentWindow:
    """
    Deployment time window.

    Attributes:
        window_id: Unique window identifier
        name: Window name
        start_hour: Start hour (0-23)
        end_hour: End hour (0-23)
        days_of_week: Allowed days
        timezone: Timezone
        environments: Allowed environments
        enabled: Whether window is enabled
    """
    window_id: str
    name: str
    start_hour: int = 9
    end_hour: int = 17
    days_of_week: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri
    timezone: str = "UTC"
    environments: List[str] = field(default_factory=lambda: ["staging"])
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def is_active(self, dt: Optional[datetime] = None) -> bool:
        """Check if window is currently active."""
        if not self.enabled:
            return False

        if dt is None:
            dt = datetime.now(timezone.utc)

        # Check day of week
        if dt.weekday() not in self.days_of_week:
            return False

        # Check hour
        if not (self.start_hour <= dt.hour < self.end_hour):
            return False

        return True


@dataclass
class ScheduledDeployment:
    """
    Scheduled deployment configuration.

    Attributes:
        schedule_id: Unique schedule identifier
        name: Schedule name
        service_name: Service to deploy
        version: Version to deploy
        schedule_type: Type of schedule
        scheduled_time: Scheduled time (for ONCE type)
        cron_expression: Cron expression (for CRON type)
        interval_seconds: Interval in seconds (for INTERVAL type)
        deployment_config: Deployment configuration
        status: Current status
        environment: Target environment
        window_id: Associated deployment window
        next_run: Next scheduled run time
        last_run: Last run time
        run_count: Number of runs
        max_runs: Maximum runs (0 = unlimited)
        created_at: Creation timestamp
        created_by: Creator identifier
    """
    schedule_id: str
    name: str
    service_name: str
    version: str = "latest"
    schedule_type: ScheduleType = ScheduleType.ONCE
    scheduled_time: Optional[float] = None
    cron_expression: str = ""
    interval_seconds: int = 0
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    status: ScheduleStatus = ScheduleStatus.PENDING
    environment: str = "staging"
    window_id: Optional[str] = None
    next_run: Optional[float] = None
    last_run: Optional[float] = None
    run_count: int = 0
    max_runs: int = 0
    created_at: float = field(default_factory=time.time)
    created_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "name": self.name,
            "service_name": self.service_name,
            "version": self.version,
            "schedule_type": self.schedule_type.value,
            "scheduled_time": self.scheduled_time,
            "cron_expression": self.cron_expression,
            "interval_seconds": self.interval_seconds,
            "deployment_config": self.deployment_config,
            "status": self.status.value,
            "environment": self.environment,
            "window_id": self.window_id,
            "next_run": self.next_run,
            "last_run": self.last_run,
            "run_count": self.run_count,
            "max_runs": self.max_runs,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduledDeployment":
        data = dict(data)
        if "schedule_type" in data:
            data["schedule_type"] = ScheduleType(data["schedule_type"])
        if "status" in data:
            data["status"] = ScheduleStatus(data["status"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==============================================================================
# Cron Parser
# ==============================================================================

class CronParser:
    """Simple cron expression parser."""

    @staticmethod
    def parse(expression: str) -> Dict[str, List[int]]:
        """
        Parse a cron expression.

        Format: minute hour day_of_month month day_of_week
        """
        parts = expression.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expression}")

        return {
            "minute": CronParser._parse_field(parts[0], 0, 59),
            "hour": CronParser._parse_field(parts[1], 0, 23),
            "day": CronParser._parse_field(parts[2], 1, 31),
            "month": CronParser._parse_field(parts[3], 1, 12),
            "weekday": CronParser._parse_field(parts[4], 0, 6),
        }

    @staticmethod
    def _parse_field(field: str, min_val: int, max_val: int) -> List[int]:
        """Parse a single cron field."""
        values = []

        for part in field.split(","):
            if part == "*":
                values.extend(range(min_val, max_val + 1))
            elif "/" in part:
                base, step = part.split("/")
                if base == "*":
                    start = min_val
                else:
                    start = int(base)
                values.extend(range(start, max_val + 1, int(step)))
            elif "-" in part:
                start, end = part.split("-")
                values.extend(range(int(start), int(end) + 1))
            else:
                values.append(int(part))

        return sorted(set(values))

    @staticmethod
    def next_run(expression: str, after: Optional[datetime] = None) -> datetime:
        """Calculate next run time for cron expression."""
        if after is None:
            after = datetime.now(timezone.utc)

        parsed = CronParser.parse(expression)

        # Start from next minute
        dt = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Find next matching time (limit iterations)
        for _ in range(366 * 24 * 60):  # Max 1 year
            if (
                dt.minute in parsed["minute"]
                and dt.hour in parsed["hour"]
                and dt.day in parsed["day"]
                and dt.month in parsed["month"]
                and dt.weekday() in parsed["weekday"]
            ):
                return dt

            dt += timedelta(minutes=1)

        raise ValueError(f"Could not find next run time for: {expression}")


# ==============================================================================
# Deployment Scheduler (Step 223)
# ==============================================================================

class DeploymentScheduler:
    """
    Deployment Scheduler - manages scheduled deployments.

    PBTSO Phase: PLAN

    Responsibilities:
    - Schedule one-time and recurring deployments
    - Manage deployment windows
    - Execute scheduled deployments
    - Handle schedule cancellation and updates
    - Support cron-based scheduling

    Example:
        >>> scheduler = DeploymentScheduler()
        >>> schedule = scheduler.schedule_once(
        ...     name="prod-deploy",
        ...     service_name="api",
        ...     version="v2.0.0",
        ...     scheduled_time=time.time() + 3600,  # 1 hour from now
        ... )
        >>> await scheduler.start()
    """

    BUS_TOPICS = {
        "create": "deploy.scheduler.create",
        "trigger": "deploy.scheduler.trigger",
        "cancel": "deploy.scheduler.cancel",
        "update": "deploy.scheduler.update",
        "complete": "deploy.scheduler.complete",
        "failed": "deploy.scheduler.failed",
    }

    # A2A heartbeat configuration
    HEARTBEAT_INTERVAL = 300  # 5 minutes
    HEARTBEAT_TIMEOUT = 900   # 15 minutes

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "deployment-scheduler",
    ):
        """
        Initialize the scheduler.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "scheduler"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        self._schedules: Dict[str, ScheduledDeployment] = {}
        self._windows: Dict[str, DeploymentWindow] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._deploy_callback: Optional[Callable] = None
        self._last_heartbeat = time.time()

        self._load_state()

    def schedule_once(
        self,
        name: str,
        service_name: str,
        version: str,
        scheduled_time: float,
        environment: str = "staging",
        deployment_config: Optional[Dict[str, Any]] = None,
        window_id: Optional[str] = None,
        created_by: str = "",
    ) -> ScheduledDeployment:
        """
        Schedule a one-time deployment.

        Args:
            name: Schedule name
            service_name: Service to deploy
            version: Version to deploy
            scheduled_time: Unix timestamp for deployment
            environment: Target environment
            deployment_config: Additional deployment config
            window_id: Deployment window to respect
            created_by: Creator identifier

        Returns:
            Created ScheduledDeployment
        """
        schedule_id = f"schedule-{uuid.uuid4().hex[:12]}"

        schedule = ScheduledDeployment(
            schedule_id=schedule_id,
            name=name,
            service_name=service_name,
            version=version,
            schedule_type=ScheduleType.ONCE,
            scheduled_time=scheduled_time,
            next_run=scheduled_time,
            environment=environment,
            deployment_config=deployment_config or {},
            window_id=window_id,
            created_by=created_by,
            status=ScheduleStatus.SCHEDULED,
        )

        self._schedules[schedule_id] = schedule
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["create"],
            {
                "schedule_id": schedule_id,
                "name": name,
                "service_name": service_name,
                "version": version,
                "schedule_type": "once",
                "scheduled_time": scheduled_time,
            },
            actor=self.actor_id,
        )

        return schedule

    def schedule_cron(
        self,
        name: str,
        service_name: str,
        version: str,
        cron_expression: str,
        environment: str = "staging",
        deployment_config: Optional[Dict[str, Any]] = None,
        window_id: Optional[str] = None,
        max_runs: int = 0,
        created_by: str = "",
    ) -> ScheduledDeployment:
        """
        Schedule a cron-based recurring deployment.

        Args:
            name: Schedule name
            service_name: Service to deploy
            version: Version to deploy
            cron_expression: Cron expression (minute hour day month weekday)
            environment: Target environment
            deployment_config: Additional deployment config
            window_id: Deployment window to respect
            max_runs: Maximum number of runs (0 = unlimited)
            created_by: Creator identifier

        Returns:
            Created ScheduledDeployment
        """
        # Validate cron expression
        next_run_dt = CronParser.next_run(cron_expression)

        schedule_id = f"schedule-{uuid.uuid4().hex[:12]}"

        schedule = ScheduledDeployment(
            schedule_id=schedule_id,
            name=name,
            service_name=service_name,
            version=version,
            schedule_type=ScheduleType.CRON,
            cron_expression=cron_expression,
            next_run=next_run_dt.timestamp(),
            environment=environment,
            deployment_config=deployment_config or {},
            window_id=window_id,
            max_runs=max_runs,
            created_by=created_by,
            status=ScheduleStatus.SCHEDULED,
        )

        self._schedules[schedule_id] = schedule
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["create"],
            {
                "schedule_id": schedule_id,
                "name": name,
                "service_name": service_name,
                "version": version,
                "schedule_type": "cron",
                "cron_expression": cron_expression,
                "next_run": schedule.next_run,
            },
            actor=self.actor_id,
        )

        return schedule

    def schedule_interval(
        self,
        name: str,
        service_name: str,
        version: str,
        interval_seconds: int,
        environment: str = "staging",
        deployment_config: Optional[Dict[str, Any]] = None,
        window_id: Optional[str] = None,
        max_runs: int = 0,
        created_by: str = "",
    ) -> ScheduledDeployment:
        """
        Schedule an interval-based recurring deployment.

        Args:
            name: Schedule name
            service_name: Service to deploy
            version: Version to deploy
            interval_seconds: Interval between deployments
            environment: Target environment
            deployment_config: Additional deployment config
            window_id: Deployment window to respect
            max_runs: Maximum number of runs (0 = unlimited)
            created_by: Creator identifier

        Returns:
            Created ScheduledDeployment
        """
        schedule_id = f"schedule-{uuid.uuid4().hex[:12]}"

        schedule = ScheduledDeployment(
            schedule_id=schedule_id,
            name=name,
            service_name=service_name,
            version=version,
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=interval_seconds,
            next_run=time.time() + interval_seconds,
            environment=environment,
            deployment_config=deployment_config or {},
            window_id=window_id,
            max_runs=max_runs,
            created_by=created_by,
            status=ScheduleStatus.SCHEDULED,
        )

        self._schedules[schedule_id] = schedule
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["create"],
            {
                "schedule_id": schedule_id,
                "name": name,
                "service_name": service_name,
                "version": version,
                "schedule_type": "interval",
                "interval_seconds": interval_seconds,
            },
            actor=self.actor_id,
        )

        return schedule

    def create_window(
        self,
        name: str,
        start_hour: int,
        end_hour: int,
        days_of_week: Optional[List[int]] = None,
        environments: Optional[List[str]] = None,
    ) -> DeploymentWindow:
        """
        Create a deployment window.

        Args:
            name: Window name
            start_hour: Start hour (0-23)
            end_hour: End hour (0-23)
            days_of_week: Allowed days (0=Monday, 6=Sunday)
            environments: Allowed environments

        Returns:
            Created DeploymentWindow
        """
        window_id = f"window-{uuid.uuid4().hex[:12]}"

        window = DeploymentWindow(
            window_id=window_id,
            name=name,
            start_hour=start_hour,
            end_hour=end_hour,
            days_of_week=days_of_week or [0, 1, 2, 3, 4],
            environments=environments or ["staging"],
        )

        self._windows[window_id] = window
        self._save_state()

        return window

    def cancel_schedule(self, schedule_id: str) -> bool:
        """Cancel a scheduled deployment."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return False

        if schedule.status in (ScheduleStatus.COMPLETED, ScheduleStatus.CANCELLED):
            return False

        schedule.status = ScheduleStatus.CANCELLED
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["cancel"],
            {"schedule_id": schedule_id, "service_name": schedule.service_name},
            actor=self.actor_id,
        )

        return True

    def pause_schedule(self, schedule_id: str) -> bool:
        """Pause a scheduled deployment."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return False

        if schedule.status != ScheduleStatus.SCHEDULED:
            return False

        schedule.status = ScheduleStatus.PAUSED
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["update"],
            {"schedule_id": schedule_id, "status": "paused"},
            actor=self.actor_id,
        )

        return True

    def resume_schedule(self, schedule_id: str) -> bool:
        """Resume a paused schedule."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return False

        if schedule.status != ScheduleStatus.PAUSED:
            return False

        schedule.status = ScheduleStatus.SCHEDULED
        self._update_next_run(schedule)
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["update"],
            {"schedule_id": schedule_id, "status": "scheduled"},
            actor=self.actor_id,
        )

        return True

    def trigger_now(self, schedule_id: str) -> bool:
        """Trigger a scheduled deployment immediately."""
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return False

        asyncio.create_task(self._execute_schedule(schedule))
        return True

    def set_deploy_callback(
        self,
        callback: Callable[[ScheduledDeployment], Any],
    ) -> None:
        """Set callback for deployment execution."""
        self._deploy_callback = callback

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                now = time.time()

                # Check for schedules to execute
                for schedule in list(self._schedules.values()):
                    if schedule.status != ScheduleStatus.SCHEDULED:
                        continue

                    if schedule.next_run and schedule.next_run <= now:
                        # Check deployment window
                        if schedule.window_id:
                            window = self._windows.get(schedule.window_id)
                            if window and not window.is_active():
                                continue

                        await self._execute_schedule(schedule)

                # Send heartbeat
                if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
                    self._send_heartbeat()
                    self._last_heartbeat = now

                await asyncio.sleep(10)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(30)

    async def _execute_schedule(self, schedule: ScheduledDeployment) -> None:
        """Execute a scheduled deployment."""
        schedule.status = ScheduleStatus.RUNNING
        schedule.last_run = time.time()
        schedule.run_count += 1
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["trigger"],
            {
                "schedule_id": schedule.schedule_id,
                "service_name": schedule.service_name,
                "version": schedule.version,
                "environment": schedule.environment,
                "run_count": schedule.run_count,
            },
            actor=self.actor_id,
        )

        try:
            if self._deploy_callback:
                if asyncio.iscoroutinefunction(self._deploy_callback):
                    await self._deploy_callback(schedule)
                else:
                    self._deploy_callback(schedule)

            # Update status based on schedule type
            if schedule.schedule_type == ScheduleType.ONCE:
                schedule.status = ScheduleStatus.COMPLETED
            elif schedule.max_runs > 0 and schedule.run_count >= schedule.max_runs:
                schedule.status = ScheduleStatus.COMPLETED
            else:
                schedule.status = ScheduleStatus.SCHEDULED
                self._update_next_run(schedule)

            _emit_bus_event(
                self.BUS_TOPICS["complete"],
                {
                    "schedule_id": schedule.schedule_id,
                    "service_name": schedule.service_name,
                    "status": schedule.status.value,
                },
                actor=self.actor_id,
            )

        except Exception as e:
            schedule.status = ScheduleStatus.FAILED
            schedule.metadata["last_error"] = str(e)

            _emit_bus_event(
                self.BUS_TOPICS["failed"],
                {
                    "schedule_id": schedule.schedule_id,
                    "service_name": schedule.service_name,
                    "error": str(e),
                },
                level="error",
                actor=self.actor_id,
            )

        self._save_state()

    def _update_next_run(self, schedule: ScheduledDeployment) -> None:
        """Update next run time for a schedule."""
        now = time.time()

        if schedule.schedule_type == ScheduleType.CRON:
            next_run_dt = CronParser.next_run(schedule.cron_expression)
            schedule.next_run = next_run_dt.timestamp()
        elif schedule.schedule_type == ScheduleType.INTERVAL:
            schedule.next_run = now + schedule.interval_seconds
        elif schedule.schedule_type == ScheduleType.ONCE:
            schedule.next_run = None

    def _send_heartbeat(self) -> None:
        """Send A2A heartbeat."""
        _emit_bus_event(
            "a2a.scheduler.heartbeat",
            {
                "actor_id": self.actor_id,
                "active_schedules": len([
                    s for s in self._schedules.values()
                    if s.status == ScheduleStatus.SCHEDULED
                ]),
            },
            kind="heartbeat",
            actor=self.actor_id,
        )

    def get_schedule(self, schedule_id: str) -> Optional[ScheduledDeployment]:
        """Get a schedule by ID."""
        return self._schedules.get(schedule_id)

    def get_window(self, window_id: str) -> Optional[DeploymentWindow]:
        """Get a deployment window by ID."""
        return self._windows.get(window_id)

    def list_schedules(
        self,
        status: Optional[ScheduleStatus] = None,
        service_name: Optional[str] = None,
    ) -> List[ScheduledDeployment]:
        """List schedules."""
        schedules = list(self._schedules.values())

        if status:
            schedules = [s for s in schedules if s.status == status]
        if service_name:
            schedules = [s for s in schedules if s.service_name == service_name]

        return sorted(schedules, key=lambda s: s.next_run or 0)

    def list_windows(self) -> List[DeploymentWindow]:
        """List deployment windows."""
        return list(self._windows.values())

    def get_upcoming(self, hours: int = 24) -> List[ScheduledDeployment]:
        """Get upcoming scheduled deployments."""
        cutoff = time.time() + (hours * 3600)
        return [
            s for s in self._schedules.values()
            if s.status == ScheduleStatus.SCHEDULED
            and s.next_run
            and s.next_run <= cutoff
        ]

    def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule."""
        if schedule_id not in self._schedules:
            return False

        del self._schedules[schedule_id]
        self._save_state()
        return True

    def delete_window(self, window_id: str) -> bool:
        """Delete a deployment window."""
        if window_id not in self._windows:
            return False

        del self._windows[window_id]
        self._save_state()
        return True

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "schedules": {sid: s.to_dict() for sid, s in self._schedules.items()},
            "windows": {wid: w.to_dict() for wid, w in self._windows.items()},
        }
        state_file = self.state_dir / "scheduler_state.json"
        with open(state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "scheduler_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            for sid, data in state.get("schedules", {}).items():
                self._schedules[sid] = ScheduledDeployment.from_dict(data)

            for wid, data in state.get("windows", {}).items():
                self._windows[wid] = DeploymentWindow(**data)
        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for deployment scheduler."""
    import argparse

    parser = argparse.ArgumentParser(description="Deployment Scheduler (Step 223)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # schedule-once command
    once_parser = subparsers.add_parser("schedule-once", help="Schedule one-time deployment")
    once_parser.add_argument("service_name", help="Service name")
    once_parser.add_argument("--name", "-n", required=True, help="Schedule name")
    once_parser.add_argument("--version", "-v", default="latest", help="Version")
    once_parser.add_argument("--at", required=True, help="Time (ISO format or Unix timestamp)")
    once_parser.add_argument("--env", "-e", default="staging", help="Environment")
    once_parser.add_argument("--json", action="store_true", help="JSON output")

    # schedule-cron command
    cron_parser = subparsers.add_parser("schedule-cron", help="Schedule cron deployment")
    cron_parser.add_argument("service_name", help="Service name")
    cron_parser.add_argument("--name", "-n", required=True, help="Schedule name")
    cron_parser.add_argument("--version", "-v", default="latest", help="Version")
    cron_parser.add_argument("--cron", "-c", required=True, help="Cron expression")
    cron_parser.add_argument("--env", "-e", default="staging", help="Environment")
    cron_parser.add_argument("--max-runs", type=int, default=0, help="Max runs (0=unlimited)")
    cron_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List schedules")
    list_parser.add_argument("--status", "-s", help="Filter by status")
    list_parser.add_argument("--service", help="Filter by service")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a schedule")
    cancel_parser.add_argument("schedule_id", help="Schedule ID")
    cancel_parser.add_argument("--json", action="store_true", help="JSON output")

    # upcoming command
    upcoming_parser = subparsers.add_parser("upcoming", help="Show upcoming deployments")
    upcoming_parser.add_argument("--hours", type=int, default=24, help="Hours ahead")
    upcoming_parser.add_argument("--json", action="store_true", help="JSON output")

    # window command
    window_parser = subparsers.add_parser("window", help="Create deployment window")
    window_parser.add_argument("--name", "-n", required=True, help="Window name")
    window_parser.add_argument("--start", type=int, required=True, help="Start hour (0-23)")
    window_parser.add_argument("--end", type=int, required=True, help="End hour (0-23)")
    window_parser.add_argument("--days", default="0,1,2,3,4", help="Days of week (0=Mon)")
    window_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    scheduler = DeploymentScheduler()

    if args.command == "schedule-once":
        # Parse time
        try:
            scheduled_time = float(args.at)
        except ValueError:
            scheduled_time = datetime.fromisoformat(args.at).timestamp()

        schedule = scheduler.schedule_once(
            name=args.name,
            service_name=args.service_name,
            version=args.version,
            scheduled_time=scheduled_time,
            environment=args.env,
        )

        if args.json:
            print(json.dumps(schedule.to_dict(), indent=2))
        else:
            print(f"Scheduled: {schedule.schedule_id}")
            print(f"  Service: {schedule.service_name}")
            print(f"  Version: {schedule.version}")
            print(f"  Time: {datetime.fromtimestamp(scheduled_time).isoformat()}")

        return 0

    elif args.command == "schedule-cron":
        schedule = scheduler.schedule_cron(
            name=args.name,
            service_name=args.service_name,
            version=args.version,
            cron_expression=args.cron,
            environment=args.env,
            max_runs=args.max_runs,
        )

        if args.json:
            print(json.dumps(schedule.to_dict(), indent=2))
        else:
            print(f"Scheduled: {schedule.schedule_id}")
            print(f"  Service: {schedule.service_name}")
            print(f"  Cron: {args.cron}")
            if schedule.next_run:
                print(f"  Next Run: {datetime.fromtimestamp(schedule.next_run).isoformat()}")

        return 0

    elif args.command == "list":
        status = ScheduleStatus(args.status) if args.status else None
        schedules = scheduler.list_schedules(status=status, service_name=args.service)

        if args.json:
            print(json.dumps([s.to_dict() for s in schedules], indent=2))
        else:
            for s in schedules:
                next_run = ""
                if s.next_run:
                    next_run = datetime.fromtimestamp(s.next_run).strftime("%Y-%m-%d %H:%M")
                print(f"{s.schedule_id} ({s.service_name}) - {s.status.value} - {next_run}")

        return 0

    elif args.command == "cancel":
        success = scheduler.cancel_schedule(args.schedule_id)

        if args.json:
            print(json.dumps({"success": success, "schedule_id": args.schedule_id}))
        else:
            if success:
                print(f"Cancelled: {args.schedule_id}")
            else:
                print(f"Failed to cancel: {args.schedule_id}")

        return 0 if success else 1

    elif args.command == "upcoming":
        upcoming = scheduler.get_upcoming(hours=args.hours)

        if args.json:
            print(json.dumps([s.to_dict() for s in upcoming], indent=2))
        else:
            if not upcoming:
                print("No upcoming deployments")
            else:
                for s in upcoming:
                    next_run = datetime.fromtimestamp(s.next_run).strftime("%Y-%m-%d %H:%M")
                    print(f"{next_run} - {s.service_name}:{s.version} ({s.environment})")

        return 0

    elif args.command == "window":
        days = [int(d) for d in args.days.split(",")]
        window = scheduler.create_window(
            name=args.name,
            start_hour=args.start,
            end_hour=args.end,
            days_of_week=days,
        )

        if args.json:
            print(json.dumps(window.to_dict(), indent=2))
        else:
            print(f"Created window: {window.window_id}")
            print(f"  Name: {window.name}")
            print(f"  Hours: {window.start_hour}:00 - {window.end_hour}:00")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
