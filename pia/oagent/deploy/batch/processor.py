#!/usr/bin/env python3
"""
processor.py - Deploy Batch Processor (Step 239)

PBTSO Phase: PLAN, DISTRIBUTE
A2A Integration: Batch deployment operations via deploy.batch.*

Provides:
- BatchStatus: Batch job status types
- BatchOperation: Batch operation types
- BatchItem: Individual batch item
- BatchJob: Batch job definition
- BatchResult: Batch execution result
- DeployBatchProcessor: Main batch processor

Bus Topics:
- deploy.batch.start
- deploy.batch.progress
- deploy.batch.complete
- deploy.batch.failed

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
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ==============================================================================
# Bus Emission Helper with File Locking (DKIN v30)
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
    actor: str = "batch-processor"
) -> str:
    """Emit an event to the Pluribus bus with fcntl.flock() file locking."""
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

class BatchStatus(Enum):
    """Batch job status types."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class BatchOperation(Enum):
    """Batch operation types."""
    DEPLOY = "deploy"
    ROLLBACK = "rollback"
    RESTART = "restart"
    SCALE = "scale"
    UPDATE_CONFIG = "update_config"
    HEALTH_CHECK = "health_check"
    CUSTOM = "custom"


class BatchStrategy(Enum):
    """Batch execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ROLLING = "rolling"


@dataclass
class BatchItem:
    """
    Individual batch item.

    Attributes:
        item_id: Item identifier
        operation: Operation type
        target: Target (service, deployment, etc.)
        params: Operation parameters
        status: Item status
        started_at: Start timestamp
        completed_at: Completion timestamp
        result: Operation result
        error: Error message if failed
    """
    item_id: str
    operation: BatchOperation
    target: str
    params: Dict[str, Any] = field(default_factory=dict)
    status: BatchStatus = BatchStatus.PENDING
    started_at: float = 0.0
    completed_at: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "operation": self.operation.value,
            "target": self.target,
            "params": self.params,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatchItem":
        data = dict(data)
        if "operation" in data:
            data["operation"] = BatchOperation(data["operation"])
        if "status" in data:
            data["status"] = BatchStatus(data["status"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BatchJob:
    """
    Batch job definition.

    Attributes:
        job_id: Job identifier
        name: Job name
        description: Job description
        items: Batch items
        strategy: Execution strategy
        max_parallel: Max parallel items
        stop_on_error: Stop on first error
        timeout_s: Job timeout
        created_at: Creation timestamp
        started_at: Start timestamp
        completed_at: Completion timestamp
        status: Job status
        progress: Completion progress (0-100)
        created_by: User who created the job
    """
    job_id: str
    name: str
    description: str = ""
    items: List[BatchItem] = field(default_factory=list)
    strategy: BatchStrategy = BatchStrategy.SEQUENTIAL
    max_parallel: int = 5
    stop_on_error: bool = False
    timeout_s: int = 3600
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0
    status: BatchStatus = BatchStatus.PENDING
    progress: float = 0.0
    created_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "description": self.description,
            "items": [i.to_dict() for i in self.items],
            "strategy": self.strategy.value,
            "max_parallel": self.max_parallel,
            "stop_on_error": self.stop_on_error,
            "timeout_s": self.timeout_s,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status.value,
            "progress": self.progress,
            "created_by": self.created_by,
        }


@dataclass
class BatchResult:
    """
    Batch execution result.

    Attributes:
        job_id: Job identifier
        status: Final status
        total_items: Total items
        completed_items: Completed items
        failed_items: Failed items
        skipped_items: Skipped items
        duration_ms: Total duration
        errors: List of errors
    """
    job_id: str
    status: BatchStatus
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    duration_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
        }


# ==============================================================================
# Deploy Batch Processor (Step 239)
# ==============================================================================

class DeployBatchProcessor:
    """
    Deploy Batch Processor - batch deployment operations.

    PBTSO Phase: PLAN, DISTRIBUTE

    Responsibilities:
    - Execute batch deployment operations
    - Support sequential, parallel, and rolling strategies
    - Track job progress and status
    - Handle errors and retries
    - Support job pause/resume/cancel

    Example:
        >>> processor = DeployBatchProcessor()
        >>> job = processor.create_job("multi-deploy", items=[
        ...     BatchItem(item_id="1", operation=BatchOperation.DEPLOY, target="service-a"),
        ...     BatchItem(item_id="2", operation=BatchOperation.DEPLOY, target="service-b"),
        ... ])
        >>> result = await processor.execute(job.job_id)
        >>> print(f"Completed: {result.completed_items}/{result.total_items}")
    """

    BUS_TOPICS = {
        "start": "deploy.batch.start",
        "progress": "deploy.batch.progress",
        "complete": "deploy.batch.complete",
        "failed": "deploy.batch.failed",
    }

    # A2A heartbeat (CITIZEN v2)
    HEARTBEAT_INTERVAL_S = 300
    HEARTBEAT_TIMEOUT_S = 900

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "batch-processor",
        default_timeout_s: int = 3600,
    ):
        """
        Initialize the batch processor.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
            default_timeout_s: Default job timeout
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "batch"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id
        self.default_timeout_s = default_timeout_s

        # Jobs storage
        self._jobs: Dict[str, BatchJob] = {}

        # Operation handlers
        self._handlers: Dict[BatchOperation, Callable] = {}

        # Running jobs
        self._running_tasks: Dict[str, asyncio.Task] = {}

        self._load_state()

    def create_job(
        self,
        name: str,
        items: Optional[List[BatchItem]] = None,
        strategy: BatchStrategy = BatchStrategy.SEQUENTIAL,
        max_parallel: int = 5,
        stop_on_error: bool = False,
        timeout_s: Optional[int] = None,
        description: str = "",
        created_by: str = "",
    ) -> BatchJob:
        """
        Create a new batch job.

        Args:
            name: Job name
            items: Batch items
            strategy: Execution strategy
            max_parallel: Max parallel items
            stop_on_error: Stop on first error
            timeout_s: Job timeout
            description: Job description
            created_by: User creating the job

        Returns:
            BatchJob
        """
        job = BatchJob(
            job_id=f"batch-{uuid.uuid4().hex[:12]}",
            name=name,
            description=description,
            items=items or [],
            strategy=strategy,
            max_parallel=max_parallel,
            stop_on_error=stop_on_error,
            timeout_s=timeout_s or self.default_timeout_s,
            created_by=created_by,
        )

        self._jobs[job.job_id] = job
        self._save_job(job)

        return job

    def add_item(
        self,
        job_id: str,
        operation: BatchOperation,
        target: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[BatchItem]:
        """
        Add an item to a batch job.

        Args:
            job_id: Job identifier
            operation: Operation type
            target: Operation target
            params: Operation parameters

        Returns:
            BatchItem or None if job not found
        """
        job = self._jobs.get(job_id)
        if not job or job.status != BatchStatus.PENDING:
            return None

        item = BatchItem(
            item_id=f"item-{uuid.uuid4().hex[:8]}",
            operation=operation,
            target=target,
            params=params or {},
        )

        job.items.append(item)
        self._save_job(job)

        return item

    def register_handler(
        self,
        operation: BatchOperation,
        handler: Callable[[BatchItem], Any],
    ) -> None:
        """Register a handler for an operation type."""
        self._handlers[operation] = handler

    async def execute(self, job_id: str) -> BatchResult:
        """
        Execute a batch job.

        Args:
            job_id: Job identifier

        Returns:
            BatchResult
        """
        job = self._jobs.get(job_id)
        if not job:
            return BatchResult(
                job_id=job_id,
                status=BatchStatus.FAILED,
                errors=["Job not found"],
            )

        if job.status not in (BatchStatus.PENDING, BatchStatus.PAUSED):
            return BatchResult(
                job_id=job_id,
                status=job.status,
                errors=["Job is not in pending/paused state"],
            )

        job.status = BatchStatus.RUNNING
        job.started_at = time.time()

        _emit_bus_event(
            self.BUS_TOPICS["start"],
            {
                "job_id": job_id,
                "name": job.name,
                "item_count": len(job.items),
                "strategy": job.strategy.value,
            },
            actor=self.actor_id,
        )

        try:
            if job.strategy == BatchStrategy.SEQUENTIAL:
                result = await self._execute_sequential(job)
            elif job.strategy == BatchStrategy.PARALLEL:
                result = await self._execute_parallel(job)
            elif job.strategy == BatchStrategy.ROLLING:
                result = await self._execute_rolling(job)
            else:
                result = await self._execute_sequential(job)

        except asyncio.CancelledError:
            job.status = BatchStatus.CANCELLED
            result = self._build_result(job)

        except Exception as e:
            job.status = BatchStatus.FAILED
            result = self._build_result(job)
            result.errors.append(str(e))

        job.completed_at = time.time()
        result.duration_ms = (job.completed_at - job.started_at) * 1000

        self._save_job(job)

        topic = self.BUS_TOPICS["complete"] if result.status == BatchStatus.COMPLETED else self.BUS_TOPICS["failed"]
        _emit_bus_event(
            topic,
            {
                "job_id": job_id,
                "status": result.status.value,
                "completed_items": result.completed_items,
                "failed_items": result.failed_items,
                "duration_ms": result.duration_ms,
            },
            level="info" if result.status == BatchStatus.COMPLETED else "error",
            actor=self.actor_id,
        )

        return result

    async def _execute_sequential(self, job: BatchJob) -> BatchResult:
        """Execute items sequentially."""
        for i, item in enumerate(job.items):
            if job.status == BatchStatus.CANCELLED:
                break

            if job.status == BatchStatus.PAUSED:
                await self._wait_for_resume(job)

            await self._execute_item(item)

            job.progress = ((i + 1) / len(job.items)) * 100
            self._emit_progress(job)

            if not item.status == BatchStatus.COMPLETED and job.stop_on_error:
                job.status = BatchStatus.FAILED
                break

        return self._build_result(job)

    async def _execute_parallel(self, job: BatchJob) -> BatchResult:
        """Execute items in parallel with max concurrency."""
        semaphore = asyncio.Semaphore(job.max_parallel)
        completed = 0

        async def execute_with_semaphore(item: BatchItem):
            nonlocal completed
            async with semaphore:
                if job.status == BatchStatus.CANCELLED:
                    return
                await self._execute_item(item)
                completed += 1
                job.progress = (completed / len(job.items)) * 100
                self._emit_progress(job)

        tasks = [execute_with_semaphore(item) for item in job.items]
        await asyncio.gather(*tasks, return_exceptions=True)

        return self._build_result(job)

    async def _execute_rolling(self, job: BatchJob) -> BatchResult:
        """Execute items in rolling batches."""
        batch_size = job.max_parallel
        completed = 0

        for i in range(0, len(job.items), batch_size):
            if job.status == BatchStatus.CANCELLED:
                break

            if job.status == BatchStatus.PAUSED:
                await self._wait_for_resume(job)

            batch = job.items[i:i + batch_size]
            tasks = [self._execute_item(item) for item in batch]
            await asyncio.gather(*tasks, return_exceptions=True)

            completed += len(batch)
            job.progress = (completed / len(job.items)) * 100
            self._emit_progress(job)

            # Check for failures in batch
            if job.stop_on_error:
                for item in batch:
                    if item.status == BatchStatus.FAILED:
                        job.status = BatchStatus.FAILED
                        return self._build_result(job)

        return self._build_result(job)

    async def _execute_item(self, item: BatchItem) -> None:
        """Execute a single batch item."""
        item.status = BatchStatus.RUNNING
        item.started_at = time.time()

        try:
            handler = self._handlers.get(item.operation)
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(item)
                else:
                    result = handler(item)
                item.result = result if isinstance(result, dict) else {"result": result}
                item.status = BatchStatus.COMPLETED
            else:
                # Default handler - simulate operation
                await self._default_handler(item)
                item.status = BatchStatus.COMPLETED

        except Exception as e:
            item.status = BatchStatus.FAILED
            item.error = str(e)

        item.completed_at = time.time()

    async def _default_handler(self, item: BatchItem) -> None:
        """Default handler for operations."""
        # Simulate operation
        await asyncio.sleep(0.1)
        item.result = {
            "operation": item.operation.value,
            "target": item.target,
            "status": "simulated",
        }

    async def _wait_for_resume(self, job: BatchJob) -> None:
        """Wait for job to be resumed."""
        while job.status == BatchStatus.PAUSED:
            await asyncio.sleep(1)

    def _emit_progress(self, job: BatchJob) -> None:
        """Emit progress event."""
        completed = sum(1 for i in job.items if i.status == BatchStatus.COMPLETED)
        failed = sum(1 for i in job.items if i.status == BatchStatus.FAILED)

        _emit_bus_event(
            self.BUS_TOPICS["progress"],
            {
                "job_id": job.job_id,
                "progress": job.progress,
                "completed": completed,
                "failed": failed,
                "total": len(job.items),
            },
            kind="metric",
            actor=self.actor_id,
        )

    def _build_result(self, job: BatchJob) -> BatchResult:
        """Build result from job state."""
        completed = sum(1 for i in job.items if i.status == BatchStatus.COMPLETED)
        failed = sum(1 for i in job.items if i.status == BatchStatus.FAILED)
        skipped = sum(1 for i in job.items if i.status == BatchStatus.PENDING)
        errors = [i.error for i in job.items if i.error]

        if job.status == BatchStatus.CANCELLED:
            status = BatchStatus.CANCELLED
        elif failed > 0:
            status = BatchStatus.PARTIAL if completed > 0 else BatchStatus.FAILED
        else:
            status = BatchStatus.COMPLETED

        job.status = status

        return BatchResult(
            job_id=job.job_id,
            status=status,
            total_items=len(job.items),
            completed_items=completed,
            failed_items=failed,
            skipped_items=skipped,
            errors=errors,
        )

    def pause_job(self, job_id: str) -> bool:
        """Pause a running job."""
        job = self._jobs.get(job_id)
        if job and job.status == BatchStatus.RUNNING:
            job.status = BatchStatus.PAUSED
            self._save_job(job)
            return True
        return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        job = self._jobs.get(job_id)
        if job and job.status == BatchStatus.PAUSED:
            job.status = BatchStatus.RUNNING
            self._save_job(job)
            return True
        return False

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self._jobs.get(job_id)
        if job and job.status in (BatchStatus.PENDING, BatchStatus.RUNNING, BatchStatus.PAUSED):
            job.status = BatchStatus.CANCELLED
            # Cancel running task if any
            if job_id in self._running_tasks:
                self._running_tasks[job_id].cancel()
            self._save_job(job)
            return True
        return False

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[BatchStatus] = None,
        limit: int = 100,
    ) -> List[BatchJob]:
        """List jobs with optional filters."""
        jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def get_item(self, job_id: str, item_id: str) -> Optional[BatchItem]:
        """Get a specific item from a job."""
        job = self._jobs.get(job_id)
        if job:
            for item in job.items:
                if item.item_id == item_id:
                    return item
        return None

    def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            job_file = self.state_dir / f"{job_id}.json"
            if job_file.exists():
                job_file.unlink()
            return True
        return False

    def _save_job(self, job: BatchJob) -> None:
        """Save job to disk."""
        job_file = self.state_dir / f"{job.job_id}.json"
        with open(job_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(job.to_dict(), f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load jobs from disk."""
        for job_file in self.state_dir.glob("batch-*.json"):
            try:
                with open(job_file, "r") as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        data = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                job = BatchJob(
                    job_id=data["job_id"],
                    name=data["name"],
                    description=data.get("description", ""),
                    strategy=BatchStrategy(data.get("strategy", "sequential")),
                    max_parallel=data.get("max_parallel", 5),
                    stop_on_error=data.get("stop_on_error", False),
                    timeout_s=data.get("timeout_s", self.default_timeout_s),
                    created_at=data.get("created_at", time.time()),
                    started_at=data.get("started_at", 0),
                    completed_at=data.get("completed_at", 0),
                    status=BatchStatus(data.get("status", "pending")),
                    progress=data.get("progress", 0),
                    created_by=data.get("created_by", ""),
                )

                for item_data in data.get("items", []):
                    job.items.append(BatchItem.from_dict(item_data))

                self._jobs[job.job_id] = job

            except (json.JSONDecodeError, KeyError, IOError):
                continue


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for batch processor."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Batch Processor (Step 239)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a batch job")
    create_parser.add_argument("name", help="Job name")
    create_parser.add_argument("--strategy", "-s", default="sequential",
                               choices=["sequential", "parallel", "rolling"])
    create_parser.add_argument("--parallel", "-p", type=int, default=5, help="Max parallel")
    create_parser.add_argument("--stop-on-error", action="store_true", help="Stop on error")
    create_parser.add_argument("--description", "-d", default="", help="Description")

    # add-item command
    add_parser = subparsers.add_parser("add-item", help="Add item to job")
    add_parser.add_argument("job_id", help="Job ID")
    add_parser.add_argument("target", help="Target")
    add_parser.add_argument("--operation", "-o", default="deploy",
                            choices=["deploy", "rollback", "restart", "scale", "health_check"])
    add_parser.add_argument("--params", "-p", help="JSON params")

    # execute command
    execute_parser = subparsers.add_parser("execute", help="Execute a batch job")
    execute_parser.add_argument("job_id", help="Job ID")
    execute_parser.add_argument("--json", action="store_true", help="JSON output")

    # status command
    status_parser = subparsers.add_parser("status", help="Get job status")
    status_parser.add_argument("job_id", help="Job ID")
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List batch jobs")
    list_parser.add_argument("--status", choices=["pending", "running", "completed", "failed"])
    list_parser.add_argument("--limit", "-l", type=int, default=20, help="Max results")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # pause command
    pause_parser = subparsers.add_parser("pause", help="Pause a job")
    pause_parser.add_argument("job_id", help="Job ID")

    # resume command
    resume_parser = subparsers.add_parser("resume", help="Resume a job")
    resume_parser.add_argument("job_id", help="Job ID")

    # cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a job")
    cancel_parser.add_argument("job_id", help="Job ID")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a job")
    delete_parser.add_argument("job_id", help="Job ID")

    args = parser.parse_args()
    processor = DeployBatchProcessor()

    if args.command == "create":
        job = processor.create_job(
            name=args.name,
            strategy=BatchStrategy(args.strategy),
            max_parallel=args.parallel,
            stop_on_error=args.stop_on_error,
            description=args.description,
        )
        print(f"Created job: {job.job_id}")
        print(f"  Name: {job.name}")
        print(f"  Strategy: {job.strategy.value}")
        return 0

    elif args.command == "add-item":
        params = json.loads(args.params) if args.params else {}
        item = processor.add_item(
            args.job_id,
            BatchOperation(args.operation),
            args.target,
            params,
        )
        if item:
            print(f"Added item: {item.item_id}")
            print(f"  Target: {item.target}")
            print(f"  Operation: {item.operation.value}")
        else:
            print("Failed to add item (job not found or not pending)")
            return 1
        return 0

    elif args.command == "execute":
        result = asyncio.get_event_loop().run_until_complete(
            processor.execute(args.job_id)
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status_icon = "OK" if result.status == BatchStatus.COMPLETED else "FAIL"
            print(f"[{status_icon}] Batch Job: {result.job_id}")
            print(f"  Status: {result.status.value}")
            print(f"  Completed: {result.completed_items}/{result.total_items}")
            print(f"  Failed: {result.failed_items}")
            print(f"  Duration: {result.duration_ms:.1f}ms")
            if result.errors:
                print("  Errors:")
                for e in result.errors[:5]:
                    print(f"    - {e}")

        return 0 if result.status == BatchStatus.COMPLETED else 1

    elif args.command == "status":
        job = processor.get_job(args.job_id)
        if not job:
            print(f"Job not found: {args.job_id}")
            return 1

        if args.json:
            print(json.dumps(job.to_dict(), indent=2))
        else:
            print(f"Job: {job.job_id}")
            print(f"  Name: {job.name}")
            print(f"  Status: {job.status.value}")
            print(f"  Progress: {job.progress:.1f}%")
            print(f"  Items: {len(job.items)}")
            completed = sum(1 for i in job.items if i.status == BatchStatus.COMPLETED)
            failed = sum(1 for i in job.items if i.status == BatchStatus.FAILED)
            print(f"  Completed: {completed}, Failed: {failed}")

        return 0

    elif args.command == "list":
        status = BatchStatus(args.status) if args.status else None
        jobs = processor.list_jobs(status=status, limit=args.limit)

        if args.json:
            print(json.dumps([j.to_dict() for j in jobs], indent=2))
        else:
            if not jobs:
                print("No batch jobs found")
            else:
                for j in jobs:
                    print(f"{j.job_id} ({j.name}) - {j.status.value} [{len(j.items)} items]")

        return 0

    elif args.command == "pause":
        success = processor.pause_job(args.job_id)
        if success:
            print(f"Paused job: {args.job_id}")
        else:
            print(f"Failed to pause job: {args.job_id}")
            return 1
        return 0

    elif args.command == "resume":
        success = processor.resume_job(args.job_id)
        if success:
            print(f"Resumed job: {args.job_id}")
        else:
            print(f"Failed to resume job: {args.job_id}")
            return 1
        return 0

    elif args.command == "cancel":
        success = processor.cancel_job(args.job_id)
        if success:
            print(f"Cancelled job: {args.job_id}")
        else:
            print(f"Failed to cancel job: {args.job_id}")
            return 1
        return 0

    elif args.command == "delete":
        success = processor.delete_job(args.job_id)
        if success:
            print(f"Deleted job: {args.job_id}")
        else:
            print(f"Job not found: {args.job_id}")
            return 1
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
