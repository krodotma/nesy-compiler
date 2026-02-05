#!/usr/bin/env python3
"""
batch_processor.py - Batch Code Operations (Step 89)

PBTSO Phase: ITERATE, VERIFY

Provides:
- Batch job management
- Parallel processing with worker pool
- Progress tracking
- Error handling per item
- Checkpointing and resume

Bus Topics:
- code.batch.started
- code.batch.progress
- code.batch.completed
- code.batch.failed

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, TypeVar

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class BatchStatus(Enum):
    """Status of a batch job."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_workers: int = 4
    batch_size: int = 100
    timeout_per_item_s: float = 30.0
    max_retries: int = 3
    retry_delay_s: float = 1.0
    checkpoint_interval: int = 10
    checkpoint_dir: str = "/pluribus/.pluribus/checkpoints"
    enable_progress: bool = True
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "timeout_per_item_s": self.timeout_per_item_s,
            "max_retries": self.max_retries,
            "checkpoint_interval": self.checkpoint_interval,
        }


# =============================================================================
# Agent Bus with File Locking
# =============================================================================

class LockedAgentBus:
    """Agent bus with file locking for safe concurrent writes."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        line = json.dumps(full_event, ensure_ascii=False, separators=(",", ":")) + "\n"

        fd = os.open(str(self.bus_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        return event_id


# =============================================================================
# Batch Types
# =============================================================================

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchItem(Generic[T]):
    """A single item in a batch."""
    id: str
    data: T
    status: BatchStatus = BatchStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    retries: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "retries": self.retries,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class BatchResult(Generic[R]):
    """Result of a batch operation."""
    job_id: str
    status: BatchStatus
    total: int
    completed: int
    failed: int
    results: List[R]
    errors: List[Dict[str, Any]]
    started_at: float
    completed_at: Optional[float] = None
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "results_count": len(self.results),
            "errors_count": len(self.errors),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
        }


@dataclass
class BatchJob(Generic[T, R]):
    """A batch job containing multiple items."""
    id: str
    name: str
    items: List[BatchItem[T]]
    status: BatchStatus = BatchStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    checkpoint_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        """Get job progress as percentage."""
        if not self.items:
            return 0.0
        completed = sum(1 for i in self.items if i.status in (BatchStatus.COMPLETED, BatchStatus.FAILED))
        return (completed / len(self.items)) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "total_items": len(self.items),
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


# =============================================================================
# Batch Processor
# =============================================================================

class BatchProcessor(Generic[T, R]):
    """
    Batch processing system for code operations.

    PBTSO Phase: ITERATE, VERIFY

    Features:
    - Parallel processing with configurable workers
    - Progress tracking and reporting
    - Error handling with retries
    - Checkpointing and resume
    - Pause/resume capability

    Usage:
        async def process_file(path: str) -> str:
            return transform(path)

        processor = BatchProcessor(process_file, config)
        result = await processor.process(files)
    """

    BUS_TOPICS = {
        "started": "code.batch.started",
        "progress": "code.batch.progress",
        "completed": "code.batch.completed",
        "failed": "code.batch.failed",
    }

    def __init__(
        self,
        process_func: Callable[[T], Coroutine[Any, Any, R]],
        config: Optional[BatchConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.process_func = process_func
        self.config = config or BatchConfig()
        self.bus = bus or LockedAgentBus()

        self._jobs: Dict[str, BatchJob] = {}
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._paused = False
        self._cancelled = False

    # =========================================================================
    # Job Management
    # =========================================================================

    def create_job(
        self,
        items: List[T],
        name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchJob[T, R]:
        """Create a new batch job."""
        job_id = f"batch-{uuid.uuid4().hex[:12]}"

        batch_items = [
            BatchItem(id=f"{job_id}-{i}", data=item)
            for i, item in enumerate(items)
        ]

        job = BatchJob[T, R](
            id=job_id,
            name=name or job_id,
            items=batch_items,
            metadata=metadata or {},
        )

        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self, status: Optional[BatchStatus] = None) -> List[BatchJob]:
        """List all jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs

    # =========================================================================
    # Processing
    # =========================================================================

    async def process(
        self,
        items: List[T],
        name: str = "",
    ) -> BatchResult[R]:
        """
        Process a batch of items.

        Args:
            items: List of items to process
            name: Job name

        Returns:
            BatchResult with all results
        """
        job = self.create_job(items, name)
        return await self.run_job(job.id)

    async def run_job(self, job_id: str) -> BatchResult[R]:
        """Run a created job."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        self._semaphore = asyncio.Semaphore(self.config.max_workers)
        self._paused = False
        self._cancelled = False

        job.status = BatchStatus.RUNNING
        job.started_at = time.time()

        # Emit started event
        self.bus.emit({
            "topic": self.BUS_TOPICS["started"],
            "kind": "batch",
            "actor": "batch-processor",
            "data": job.to_dict(),
        })

        results: List[R] = []
        errors: List[Dict[str, Any]] = []

        try:
            # Process items in batches
            for batch_start in range(0, len(job.items), self.config.batch_size):
                if self._cancelled:
                    job.status = BatchStatus.CANCELLED
                    break

                while self._paused:
                    await asyncio.sleep(0.1)

                batch_end = min(batch_start + self.config.batch_size, len(job.items))
                batch = job.items[batch_start:batch_end]

                # Process batch in parallel
                tasks = [
                    self._process_item(item)
                    for item in batch
                    if item.status == BatchStatus.PENDING
                ]

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for item, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        errors.append({
                            "item_id": item.id,
                            "error": str(result),
                        })
                    elif item.status == BatchStatus.COMPLETED:
                        results.append(item.result)

                # Emit progress
                if self.config.enable_progress:
                    self._emit_progress(job)

                # Checkpoint
                if (batch_end - job.checkpoint_index) >= self.config.checkpoint_interval:
                    self._save_checkpoint(job)

            # Mark completed
            if job.status == BatchStatus.RUNNING:
                job.status = BatchStatus.COMPLETED

            job.completed_at = time.time()

            # Final result
            completed_count = sum(1 for i in job.items if i.status == BatchStatus.COMPLETED)
            failed_count = sum(1 for i in job.items if i.status == BatchStatus.FAILED)

            result = BatchResult[R](
                job_id=job.id,
                status=job.status,
                total=len(job.items),
                completed=completed_count,
                failed=failed_count,
                results=results,
                errors=errors,
                started_at=job.started_at,
                completed_at=job.completed_at,
                duration_s=job.completed_at - job.started_at if job.started_at else 0,
            )

            # Emit completed event
            self.bus.emit({
                "topic": self.BUS_TOPICS["completed"],
                "kind": "batch",
                "actor": "batch-processor",
                "data": result.to_dict(),
            })

            return result

        except Exception as e:
            job.status = BatchStatus.FAILED
            job.completed_at = time.time()

            # Emit failed event
            self.bus.emit({
                "topic": self.BUS_TOPICS["failed"],
                "kind": "batch",
                "level": "error",
                "actor": "batch-processor",
                "data": {"job_id": job.id, "error": str(e)},
            })

            raise

    async def _process_item(self, item: BatchItem[T]) -> Optional[R]:
        """Process a single item with retries."""
        if not self._semaphore:
            return None

        async with self._semaphore:
            item.status = BatchStatus.RUNNING
            item.started_at = time.time()

            for attempt in range(self.config.max_retries + 1):
                try:
                    result = await asyncio.wait_for(
                        self.process_func(item.data),
                        timeout=self.config.timeout_per_item_s,
                    )
                    item.result = result
                    item.status = BatchStatus.COMPLETED
                    item.completed_at = time.time()
                    return result

                except asyncio.TimeoutError:
                    item.error = "Timeout"
                    item.retries = attempt + 1
                except Exception as e:
                    item.error = str(e)
                    item.retries = attempt + 1

                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay_s * (attempt + 1))

            item.status = BatchStatus.FAILED
            item.completed_at = time.time()
            return None

    def _emit_progress(self, job: BatchJob) -> None:
        """Emit progress event."""
        completed = sum(1 for i in job.items if i.status in (BatchStatus.COMPLETED, BatchStatus.FAILED))
        self.bus.emit({
            "topic": self.BUS_TOPICS["progress"],
            "kind": "batch",
            "actor": "batch-processor",
            "data": {
                "job_id": job.id,
                "completed": completed,
                "total": len(job.items),
                "progress": job.progress,
            },
        })

    # =========================================================================
    # Control
    # =========================================================================

    def pause(self, job_id: str) -> bool:
        """Pause a running job."""
        job = self._jobs.get(job_id)
        if job and job.status == BatchStatus.RUNNING:
            self._paused = True
            job.status = BatchStatus.PAUSED
            return True
        return False

    def resume(self, job_id: str) -> bool:
        """Resume a paused job."""
        job = self._jobs.get(job_id)
        if job and job.status == BatchStatus.PAUSED:
            self._paused = False
            job.status = BatchStatus.RUNNING
            return True
        return False

    def cancel(self, job_id: str) -> bool:
        """Cancel a job."""
        job = self._jobs.get(job_id)
        if job and job.status in (BatchStatus.RUNNING, BatchStatus.PAUSED, BatchStatus.PENDING):
            self._cancelled = True
            job.status = BatchStatus.CANCELLED
            return True
        return False

    # =========================================================================
    # Checkpointing
    # =========================================================================

    def _save_checkpoint(self, job: BatchJob) -> None:
        """Save checkpoint for job."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"{job.id}.json"
        checkpoint_data = {
            "job": job.to_dict(),
            "items": [item.to_dict() for item in job.items],
            "checkpoint_index": sum(
                1 for i in job.items if i.status in (BatchStatus.COMPLETED, BatchStatus.FAILED)
            ),
        }

        checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2))
        job.checkpoint_index = checkpoint_data["checkpoint_index"]

    def load_checkpoint(self, job_id: str) -> Optional[BatchJob]:
        """Load job from checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"{job_id}.json"

        if not checkpoint_path.exists():
            return None

        try:
            data = json.loads(checkpoint_path.read_text())
            # Reconstruct job (simplified - actual implementation would need type info)
            return self._jobs.get(job_id)
        except Exception:
            return None

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        total_jobs = len(self._jobs)
        completed_jobs = sum(1 for j in self._jobs.values() if j.status == BatchStatus.COMPLETED)
        failed_jobs = sum(1 for j in self._jobs.values() if j.status == BatchStatus.FAILED)

        return {
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs,
            "running_jobs": sum(1 for j in self._jobs.values() if j.status == BatchStatus.RUNNING),
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Batch Processor."""
    import argparse

    parser = argparse.ArgumentParser(description="Batch Processor (Step 89)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list command
    list_parser = subparsers.add_parser("list", help="List batch jobs")
    list_parser.add_argument("--status", "-s", help="Filter by status")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # stats command
    subparsers.add_parser("stats", help="Show statistics")

    # demo command
    demo_parser = subparsers.add_parser("demo", help="Run batch processing demo")
    demo_parser.add_argument("--count", "-n", type=int, default=20, help="Number of items")
    demo_parser.add_argument("--workers", "-w", type=int, default=4, help="Workers")

    args = parser.parse_args()

    async def demo_process(item: int) -> int:
        """Demo processing function."""
        await asyncio.sleep(0.1 + (item % 3) * 0.1)  # Simulate work
        if item % 10 == 9:
            raise ValueError(f"Simulated error for item {item}")
        return item * 2

    async def run() -> int:
        if args.command == "list":
            processor = BatchProcessor(demo_process)
            jobs = processor.list_jobs()
            if args.json:
                print(json.dumps([j.to_dict() for j in jobs], indent=2))
            else:
                print(f"Jobs: {len(jobs)}")
                for job in jobs:
                    print(f"  {job.id}: {job.status.value} ({job.progress:.1f}%)")
            return 0

        elif args.command == "stats":
            processor = BatchProcessor(demo_process)
            stats = processor.get_stats()
            print(json.dumps(stats, indent=2))
            return 0

        elif args.command == "demo":
            print(f"Running batch demo with {args.count} items and {args.workers} workers...")

            config = BatchConfig(max_workers=args.workers)
            processor = BatchProcessor(demo_process, config)

            items = list(range(args.count))
            result = await processor.process(items, name="demo-batch")

            print(f"\nResult:")
            print(f"  Status: {result.status.value}")
            print(f"  Completed: {result.completed}/{result.total}")
            print(f"  Failed: {result.failed}")
            print(f"  Duration: {result.duration_s:.2f}s")

            if result.errors:
                print(f"\nErrors:")
                for err in result.errors[:5]:
                    print(f"  {err['item_id']}: {err['error']}")

            return 0 if result.status == BatchStatus.COMPLETED else 1

        return 1

    return asyncio.run(run())


if __name__ == "__main__":
    import sys
    sys.exit(main())
