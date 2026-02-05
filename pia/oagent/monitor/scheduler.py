#!/usr/bin/env python3
"""
Monitor Scheduling System - Step 273

Scheduled monitoring checks and automated task execution.

PBTSO Phase: ITERATE

Bus Topics:
- monitor.schedule.create (subscribed)
- monitor.schedule.execute (emitted)
- monitor.schedule.complete (emitted)

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
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set


class ScheduleType(Enum):
    """Types of schedules."""
    INTERVAL = "interval"      # Run every N seconds
    CRON = "cron"             # Cron expression
    ONCE = "once"             # Run once at time
    ON_DEMAND = "on_demand"   # Manual trigger only


class TaskState(Enum):
    """Task execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Task priorities."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ScheduledTask:
    """A scheduled monitoring task.

    Attributes:
        task_id: Unique task ID
        name: Task name
        handler: Handler function name
        schedule_type: Type of schedule
        schedule_config: Schedule configuration
        enabled: Whether task is enabled
        priority: Task priority
        timeout_s: Task timeout
        retry_count: Number of retries
        last_run: Last run timestamp
        next_run: Next scheduled run
        run_count: Total run count
        error_count: Error count
        config: Task configuration
    """
    task_id: str
    name: str
    handler: str
    schedule_type: ScheduleType
    schedule_config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_s: int = 60
    retry_count: int = 3
    last_run: Optional[float] = None
    next_run: Optional[float] = None
    run_count: int = 0
    error_count: int = 0
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "handler": self.handler,
            "schedule_type": self.schedule_type.value,
            "schedule_config": self.schedule_config,
            "enabled": self.enabled,
            "priority": self.priority.value,
            "timeout_s": self.timeout_s,
            "retry_count": self.retry_count,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "config": self.config,
        }


@dataclass
class TaskExecution:
    """A task execution instance.

    Attributes:
        execution_id: Unique execution ID
        task_id: Task ID
        state: Execution state
        started_at: Start timestamp
        completed_at: Completion timestamp
        result: Execution result
        error: Error message if failed
        retry_attempt: Current retry attempt
    """
    execution_id: str
    task_id: str
    state: TaskState = TaskState.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    retry_attempt: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "execution_id": self.execution_id,
            "task_id": self.task_id,
            "state": self.state.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "result": self.result if isinstance(self.result, (dict, list, str, int, float, bool, type(None))) else str(self.result),
            "error": self.error,
            "retry_attempt": self.retry_attempt,
        }

    @property
    def duration_ms(self) -> float:
        """Execution duration in milliseconds."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000


class MonitorScheduler:
    """
    Schedule and execute monitoring tasks.

    The scheduler:
    - Manages scheduled monitoring checks
    - Supports interval, cron, and one-time schedules
    - Handles task execution with retries
    - Provides task history and statistics

    Example:
        scheduler = MonitorScheduler()
        await scheduler.start()

        # Register a task
        task = scheduler.register_task(
            name="Health Check",
            handler="check_health",
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={"interval_s": 60},
        )

        # Or trigger manually
        execution = await scheduler.run_task(task.task_id)

        await scheduler.stop()
    """

    BUS_TOPICS = {
        "create": "monitor.schedule.create",
        "execute": "monitor.schedule.execute",
        "complete": "monitor.schedule.complete",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        max_concurrent_tasks: int = 10,
        bus_dir: Optional[str] = None,
    ):
        """Initialize scheduler.

        Args:
            max_concurrent_tasks: Maximum concurrent task executions
            bus_dir: Bus directory
        """
        self._max_concurrent = max_concurrent_tasks
        self._tasks: Dict[str, ScheduledTask] = {}
        self._handlers: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._executions: Dict[str, TaskExecution] = {}
        self._execution_history: List[TaskExecution] = []
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._last_heartbeat = time.time()

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Register default handlers
        self._register_default_handlers()

    async def start(self) -> bool:
        """Start the scheduler.

        Returns:
            True if started
        """
        if self._running:
            return False

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        self._emit_bus_event(
            "monitor.scheduler.started",
            {
                "tasks": len(self._tasks),
                "handlers": len(self._handlers),
            }
        )

        return True

    async def stop(self) -> bool:
        """Stop the scheduler.

        Returns:
            True if stopped
        """
        if not self._running:
            return False

        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        self._emit_bus_event(
            "monitor.scheduler.stopped",
            {"tasks_executed": sum(t.run_count for t in self._tasks.values())}
        )

        return True

    def register_handler(
        self,
        name: str,
        handler: Callable[..., Coroutine[Any, Any, Any]]
    ) -> None:
        """Register a task handler.

        Args:
            name: Handler name
            handler: Async handler function
        """
        self._handlers[name] = handler

    def register_task(
        self,
        name: str,
        handler: str,
        schedule_type: ScheduleType,
        schedule_config: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_s: int = 60,
        config: Optional[Dict[str, Any]] = None,
    ) -> ScheduledTask:
        """Register a scheduled task.

        Args:
            name: Task name
            handler: Handler name
            schedule_type: Schedule type
            schedule_config: Schedule configuration
            priority: Task priority
            timeout_s: Task timeout
            config: Task configuration

        Returns:
            Created task
        """
        task_id = f"task-{uuid.uuid4().hex[:8]}"

        task = ScheduledTask(
            task_id=task_id,
            name=name,
            handler=handler,
            schedule_type=schedule_type,
            schedule_config=schedule_config or {},
            priority=priority,
            timeout_s=timeout_s,
            config=config or {},
        )

        # Calculate next run
        task.next_run = self._calculate_next_run(task)

        self._tasks[task_id] = task

        self._emit_bus_event(
            self.BUS_TOPICS["create"],
            {
                "task_id": task_id,
                "name": name,
                "schedule_type": schedule_type.value,
            }
        )

        return task

    def unregister_task(self, task_id: str) -> bool:
        """Unregister a task.

        Args:
            task_id: Task ID

        Returns:
            True if removed
        """
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    def enable_task(self, task_id: str) -> bool:
        """Enable a task.

        Args:
            task_id: Task ID

        Returns:
            True if enabled
        """
        task = self._tasks.get(task_id)
        if task:
            task.enabled = True
            task.next_run = self._calculate_next_run(task)
            return True
        return False

    def disable_task(self, task_id: str) -> bool:
        """Disable a task.

        Args:
            task_id: Task ID

        Returns:
            True if disabled
        """
        task = self._tasks.get(task_id)
        if task:
            task.enabled = False
            return True
        return False

    async def run_task(
        self,
        task_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskExecution:
        """Run a task immediately.

        Args:
            task_id: Task ID
            context: Execution context

        Returns:
            Task execution
        """
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")

        execution = TaskExecution(
            execution_id=f"exec-{uuid.uuid4().hex[:8]}",
            task_id=task_id,
        )

        self._executions[execution.execution_id] = execution

        async with self._semaphore:
            await self._execute_task(task, execution, context or {})

        self._execution_history.append(execution)
        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-1000:]

        return execution

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """Get a task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task or None
        """
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tasks.

        Returns:
            Task summaries
        """
        return [t.to_dict() for t in self._tasks.values()]

    def get_execution(self, execution_id: str) -> Optional[TaskExecution]:
        """Get an execution by ID.

        Args:
            execution_id: Execution ID

        Returns:
            Execution or None
        """
        return self._executions.get(execution_id)

    def get_task_history(
        self,
        task_id: str,
        limit: int = 10
    ) -> List[TaskExecution]:
        """Get execution history for a task.

        Args:
            task_id: Task ID
            limit: Maximum results

        Returns:
            Execution history
        """
        history = [e for e in self._execution_history if e.task_id == task_id]
        return list(reversed(history[-limit:]))

    def get_due_tasks(self) -> List[ScheduledTask]:
        """Get tasks that are due for execution.

        Returns:
            Due tasks
        """
        now = time.time()
        due = []

        for task in self._tasks.values():
            if not task.enabled:
                continue
            if task.next_run and task.next_run <= now:
                due.append(task)

        # Sort by priority (higher first) then by next_run
        due.sort(key=lambda t: (-t.priority.value, t.next_run or 0))
        return due

    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Statistics
        """
        total_runs = sum(t.run_count for t in self._tasks.values())
        total_errors = sum(t.error_count for t in self._tasks.values())

        recent_executions = self._execution_history[-100:]
        success_count = sum(1 for e in recent_executions if e.state == TaskState.COMPLETED)
        success_rate = success_count / len(recent_executions) if recent_executions else 0.0

        return {
            "running": self._running,
            "total_tasks": len(self._tasks),
            "enabled_tasks": sum(1 for t in self._tasks.values() if t.enabled),
            "total_handlers": len(self._handlers),
            "total_runs": total_runs,
            "total_errors": total_errors,
            "recent_success_rate": success_rate,
            "pending_executions": len([e for e in self._executions.values() if e.state == TaskState.PENDING]),
        }

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
                "component": "scheduler",
                "status": "healthy" if self._running else "stopped",
                "tasks": len(self._tasks),
            }
        )

        return True

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Get due tasks
                due_tasks = self.get_due_tasks()

                # Execute due tasks
                for task in due_tasks:
                    asyncio.create_task(self._run_scheduled_task(task))

                # Emit heartbeat
                self.emit_heartbeat()

            except Exception as e:
                self._emit_bus_event(
                    "monitor.scheduler.error",
                    {"error": str(e)},
                    level="error"
                )

            await asyncio.sleep(1)  # Check every second

    async def _run_scheduled_task(self, task: ScheduledTask) -> None:
        """Run a scheduled task.

        Args:
            task: Task to run
        """
        execution = TaskExecution(
            execution_id=f"exec-{uuid.uuid4().hex[:8]}",
            task_id=task.task_id,
        )

        self._executions[execution.execution_id] = execution

        self._emit_bus_event(
            self.BUS_TOPICS["execute"],
            {
                "execution_id": execution.execution_id,
                "task_id": task.task_id,
                "task_name": task.name,
            }
        )

        async with self._semaphore:
            await self._execute_task(task, execution, {})

        # Update task state
        task.last_run = execution.completed_at
        task.next_run = self._calculate_next_run(task)

        self._execution_history.append(execution)
        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-1000:]

    async def _execute_task(
        self,
        task: ScheduledTask,
        execution: TaskExecution,
        context: Dict[str, Any]
    ) -> None:
        """Execute a task.

        Args:
            task: Task to execute
            execution: Execution instance
            context: Execution context
        """
        handler = self._handlers.get(task.handler)
        if not handler:
            execution.state = TaskState.FAILED
            execution.error = f"Handler not found: {task.handler}"
            execution.completed_at = time.time()
            task.error_count += 1
            return

        execution.state = TaskState.RUNNING
        execution.started_at = time.time()

        while execution.retry_attempt <= task.retry_count:
            try:
                # Build context
                full_context = {
                    **context,
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "config": task.config,
                    "retry_attempt": execution.retry_attempt,
                }

                # Execute with timeout
                execution.result = await asyncio.wait_for(
                    handler(full_context),
                    timeout=task.timeout_s
                )

                execution.state = TaskState.COMPLETED
                execution.completed_at = time.time()
                task.run_count += 1

                self._emit_bus_event(
                    self.BUS_TOPICS["complete"],
                    {
                        "execution_id": execution.execution_id,
                        "task_id": task.task_id,
                        "state": "completed",
                        "duration_ms": execution.duration_ms,
                    }
                )

                return

            except asyncio.TimeoutError:
                execution.error = f"Task timed out after {task.timeout_s}s"
            except Exception as e:
                execution.error = str(e)

            execution.retry_attempt += 1
            if execution.retry_attempt <= task.retry_count:
                await asyncio.sleep(min(5 * execution.retry_attempt, 30))

        # All retries exhausted
        execution.state = TaskState.FAILED
        execution.completed_at = time.time()
        task.run_count += 1
        task.error_count += 1

        self._emit_bus_event(
            self.BUS_TOPICS["complete"],
            {
                "execution_id": execution.execution_id,
                "task_id": task.task_id,
                "state": "failed",
                "error": execution.error,
            },
            level="error"
        )

    def _calculate_next_run(self, task: ScheduledTask) -> Optional[float]:
        """Calculate next run time for a task."""
        if not task.enabled:
            return None

        now = time.time()

        if task.schedule_type == ScheduleType.INTERVAL:
            interval = task.schedule_config.get("interval_s", 60)
            if task.last_run:
                return task.last_run + interval
            return now + interval

        elif task.schedule_type == ScheduleType.CRON:
            # Simple cron parsing for common patterns
            cron_expr = task.schedule_config.get("expression", "*/5 * * * *")
            return self._parse_cron_next(cron_expr, now)

        elif task.schedule_type == ScheduleType.ONCE:
            run_at = task.schedule_config.get("run_at")
            if run_at and run_at > now:
                return run_at
            return None

        elif task.schedule_type == ScheduleType.ON_DEMAND:
            return None

        return None

    def _parse_cron_next(self, expression: str, after: float) -> float:
        """Parse cron expression and get next run time."""
        # Simple implementation for common patterns
        parts = expression.split()
        if len(parts) < 5:
            return after + 300  # Default to 5 minutes

        minute_part = parts[0]

        # Handle */N pattern (every N minutes)
        if minute_part.startswith("*/"):
            try:
                interval = int(minute_part[2:])
                current = datetime.fromtimestamp(after, tz=timezone.utc)
                next_minute = ((current.minute // interval) + 1) * interval
                if next_minute >= 60:
                    next_dt = current.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    next_dt = current.replace(minute=next_minute, second=0, microsecond=0)
                return next_dt.timestamp()
            except ValueError:
                pass

        # Default fallback
        return after + 300

    def _register_default_handlers(self) -> None:
        """Register default task handlers."""

        async def health_check(context: Dict[str, Any]) -> Dict[str, Any]:
            """Health check handler."""
            return {"status": "healthy", "timestamp": time.time()}

        async def collect_metrics(context: Dict[str, Any]) -> Dict[str, Any]:
            """Metrics collection handler."""
            return {"collected": True, "timestamp": time.time()}

        async def check_alerts(context: Dict[str, Any]) -> Dict[str, Any]:
            """Alert check handler."""
            return {"checked": True, "alerts_found": 0}

        async def cleanup_old_data(context: Dict[str, Any]) -> Dict[str, Any]:
            """Cleanup old data handler."""
            return {"cleaned": True}

        async def generate_report(context: Dict[str, Any]) -> Dict[str, Any]:
            """Report generation handler."""
            return {"generated": True, "report_type": context.get("config", {}).get("type", "daily")}

        self.register_handler("health_check", health_check)
        self.register_handler("collect_metrics", collect_metrics)
        self.register_handler("check_alerts", check_alerts)
        self.register_handler("cleanup_old_data", cleanup_old_data)
        self.register_handler("generate_report", generate_report)

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
_scheduler: Optional[MonitorScheduler] = None


def get_scheduler() -> MonitorScheduler:
    """Get or create the scheduler singleton.

    Returns:
        MonitorScheduler instance
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = MonitorScheduler()
    return _scheduler


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Scheduling System (Step 273)")
    parser.add_argument("--list", action="store_true", help="List tasks")
    parser.add_argument("--create", metavar="NAME", help="Create task")
    parser.add_argument("--handler", default="health_check", help="Handler name")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds")
    parser.add_argument("--run", metavar="ID", help="Run task")
    parser.add_argument("--enable", metavar="ID", help="Enable task")
    parser.add_argument("--disable", metavar="ID", help="Disable task")
    parser.add_argument("--history", metavar="ID", help="Show task history")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    scheduler = get_scheduler()

    if args.list:
        tasks = scheduler.list_tasks()
        if args.json:
            print(json.dumps(tasks, indent=2))
        else:
            print("Tasks:")
            for t in tasks:
                enabled = "enabled" if t["enabled"] else "disabled"
                print(f"  [{t['task_id']}] {t['name']} ({enabled})")

    if args.create:
        task = scheduler.register_task(
            name=args.create,
            handler=args.handler,
            schedule_type=ScheduleType.INTERVAL,
            schedule_config={"interval_s": args.interval},
        )
        if args.json:
            print(json.dumps(task.to_dict(), indent=2))
        else:
            print(f"Created task: {task.task_id}")

    if args.run:
        async def run():
            await scheduler.start()
            execution = await scheduler.run_task(args.run)
            await scheduler.stop()
            return execution

        execution = asyncio.run(run())
        if args.json:
            print(json.dumps(execution.to_dict(), indent=2))
        else:
            print(f"Execution: {execution.execution_id}")
            print(f"  State: {execution.state.value}")
            print(f"  Duration: {execution.duration_ms:.1f}ms")

    if args.enable:
        success = scheduler.enable_task(args.enable)
        print(f"Enabled: {success}")

    if args.disable:
        success = scheduler.disable_task(args.disable)
        print(f"Disabled: {success}")

    if args.history:
        history = scheduler.get_task_history(args.history)
        if args.json:
            print(json.dumps([e.to_dict() for e in history], indent=2))
        else:
            print(f"History for {args.history}:")
            for e in history:
                print(f"  [{e.state.value}] {e.execution_id} ({e.duration_ms:.1f}ms)")

    if args.stats:
        stats = scheduler.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Scheduler Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")
