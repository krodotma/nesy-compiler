#!/usr/bin/env python3
"""
Monitor Batch Processor - Step 289

Batch monitoring operations for the Monitor Agent.

PBTSO Phase: SKILL

Bus Topics:
- monitor.batch.started (emitted)
- monitor.batch.completed (emitted)
- monitor.batch.failed (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
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
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, TypeVar


class BatchStatus(Enum):
    """Batch processing status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class BatchPriority(Enum):
    """Batch processing priority."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchItem(Generic[T]):
    """A single item in a batch.

    Attributes:
        item_id: Unique item ID
        data: Item data
        status: Processing status
        result: Processing result
        error: Error message if failed
        processing_time_ms: Processing duration
    """
    item_id: str
    data: T
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class BatchJob(Generic[T]):
    """A batch processing job.

    Attributes:
        job_id: Unique job ID
        name: Job name
        items: Batch items
        status: Job status
        priority: Job priority
        created_at: Creation timestamp
        started_at: Start timestamp
        completed_at: Completion timestamp
        total_items: Total item count
        processed_items: Processed item count
        failed_items: Failed item count
    """
    job_id: str
    name: str
    items: List[BatchItem[T]] = field(default_factory=list)
    status: BatchStatus = BatchStatus.PENDING
    priority: BatchPriority = BatchPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "failed_items": self.failed_items,
            "success_rate": (
                (self.processed_items - self.failed_items) / self.processed_items
                if self.processed_items > 0 else 0.0
            ),
        }


@dataclass
class BatchConfig:
    """Batch processing configuration.

    Attributes:
        batch_size: Items per batch
        max_concurrent: Maximum concurrent batches
        timeout_s: Batch timeout
        retry_count: Retries per item
        fail_fast: Stop on first failure
    """
    batch_size: int = 100
    max_concurrent: int = 5
    timeout_s: int = 300
    retry_count: int = 3
    fail_fast: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_size": self.batch_size,
            "max_concurrent": self.max_concurrent,
            "timeout_s": self.timeout_s,
            "retry_count": self.retry_count,
            "fail_fast": self.fail_fast,
        }


class MonitorBatchProcessor:
    """
    Batch monitoring operations for the Monitor Agent.

    Provides:
    - Batch processing of metrics
    - Batch alert evaluation
    - Batch report generation
    - Priority queuing
    - Progress tracking

    Example:
        processor = MonitorBatchProcessor()

        # Process metrics in batch
        job = await processor.create_job(
            name="metrics_ingest",
            items=metrics,
            processor=process_metric,
        )

        # Wait for completion
        result = await processor.wait_for_job(job.job_id)

        # Get progress
        progress = processor.get_job_progress(job.job_id)
    """

    BUS_TOPICS = {
        "started": "monitor.batch.started",
        "completed": "monitor.batch.completed",
        "failed": "monitor.batch.failed",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        bus_dir: Optional[str] = None,
    ):
        """Initialize batch processor.

        Args:
            config: Batch configuration
            bus_dir: Bus directory
        """
        self._config = config or BatchConfig()
        self._last_heartbeat = time.time()

        # Job tracking
        self._jobs: Dict[str, BatchJob] = {}
        self._job_queue: List[str] = []  # Job IDs sorted by priority
        self._active_jobs: int = 0
        self._lock = threading.RLock()

        # Statistics
        self._total_jobs = 0
        self._completed_jobs = 0
        self._failed_jobs = 0
        self._total_items_processed = 0

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    async def create_job(
        self,
        name: str,
        items: List[T],
        processor: Callable[[T], Coroutine[Any, Any, R]],
        priority: BatchPriority = BatchPriority.NORMAL,
        config: Optional[BatchConfig] = None,
    ) -> BatchJob[T]:
        """Create and start a batch job.

        Args:
            name: Job name
            items: Items to process
            processor: Processing function
            priority: Job priority
            config: Job-specific configuration

        Returns:
            Batch job
        """
        job_config = config or self._config

        # Create job
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        batch_items = [
            BatchItem(
                item_id=f"item-{i}",
                data=item,
            )
            for i, item in enumerate(items)
        ]

        job = BatchJob(
            job_id=job_id,
            name=name,
            items=batch_items,
            priority=priority,
            total_items=len(items),
        )

        with self._lock:
            self._jobs[job_id] = job
            self._total_jobs += 1

            # Add to queue by priority
            self._job_queue.append(job_id)
            self._job_queue.sort(
                key=lambda jid: -self._jobs[jid].priority.value
            )

        # Start processing
        asyncio.create_task(self._process_job(job, processor, job_config))

        return job

    async def wait_for_job(
        self,
        job_id: str,
        timeout_s: Optional[int] = None,
    ) -> Optional[BatchJob]:
        """Wait for a job to complete.

        Args:
            job_id: Job ID
            timeout_s: Timeout in seconds

        Returns:
            Completed job or None if timeout
        """
        timeout = timeout_s or self._config.timeout_s
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self._lock:
                job = self._jobs.get(job_id)
                if job and job.status in (
                    BatchStatus.COMPLETED,
                    BatchStatus.FAILED,
                    BatchStatus.PARTIAL,
                    BatchStatus.CANCELLED,
                ):
                    return job

            await asyncio.sleep(0.1)

        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job.

        Args:
            job_id: Job ID

        Returns:
            True if cancelled
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == BatchStatus.PENDING:
                job.status = BatchStatus.CANCELLED
                if job_id in self._job_queue:
                    self._job_queue.remove(job_id)
                return True
            return False

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get a job by ID.

        Args:
            job_id: Job ID

        Returns:
            Batch job or None
        """
        return self._jobs.get(job_id)

    def get_job_progress(self, job_id: str) -> Dict[str, Any]:
        """Get job progress.

        Args:
            job_id: Job ID

        Returns:
            Progress dictionary
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return {"error": "Job not found"}

            progress = job.processed_items / job.total_items if job.total_items > 0 else 0.0

            return {
                "job_id": job_id,
                "status": job.status.value,
                "progress": progress,
                "processed_items": job.processed_items,
                "total_items": job.total_items,
                "failed_items": job.failed_items,
                "elapsed_s": (
                    (job.completed_at or time.time()) - (job.started_at or job.created_at)
                ),
            }

    def list_jobs(
        self,
        status: Optional[BatchStatus] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List batch jobs.

        Args:
            status: Filter by status
            limit: Maximum results

        Returns:
            List of job summaries
        """
        with self._lock:
            jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        jobs.sort(key=lambda j: -j.created_at)

        return [j.to_dict() for j in jobs[:limit]]

    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            return {
                "total_jobs": self._total_jobs,
                "completed_jobs": self._completed_jobs,
                "failed_jobs": self._failed_jobs,
                "active_jobs": self._active_jobs,
                "pending_jobs": len(self._job_queue),
                "total_items_processed": self._total_items_processed,
                "success_rate": (
                    self._completed_jobs / self._total_jobs
                    if self._total_jobs > 0 else 0.0
                ),
            }

    async def process_batch(
        self,
        items: List[T],
        processor: Callable[[T], Coroutine[Any, Any, R]],
        batch_size: Optional[int] = None,
        max_concurrent: Optional[int] = None,
    ) -> List[R]:
        """Process items in batches (simpler API).

        Args:
            items: Items to process
            processor: Processing function
            batch_size: Batch size
            max_concurrent: Maximum concurrent

        Returns:
            List of results
        """
        size = batch_size or self._config.batch_size
        concurrent = max_concurrent or self._config.max_concurrent

        results: List[R] = []
        semaphore = asyncio.Semaphore(concurrent)

        async def process_with_semaphore(item: T) -> R:
            async with semaphore:
                return await processor(item)

        # Process in batches
        for i in range(0, len(items), size):
            batch = items[i:i + size]
            batch_results = await asyncio.gather(
                *[process_with_semaphore(item) for item in batch],
                return_exceptions=True,
            )

            for result in batch_results:
                if isinstance(result, Exception):
                    results.append(None)
                else:
                    results.append(result)

        return results

    async def _process_job(
        self,
        job: BatchJob,
        processor: Callable[[T], Coroutine[Any, Any, R]],
        config: BatchConfig,
    ) -> None:
        """Process a batch job.

        Args:
            job: Batch job
            processor: Processing function
            config: Job configuration
        """
        with self._lock:
            job.status = BatchStatus.RUNNING
            job.started_at = time.time()
            self._active_jobs += 1

            if job.job_id in self._job_queue:
                self._job_queue.remove(job.job_id)

        self._emit_bus_event(
            self.BUS_TOPICS["started"],
            {
                "job_id": job.job_id,
                "name": job.name,
                "total_items": job.total_items,
            },
        )

        semaphore = asyncio.Semaphore(config.max_concurrent)

        async def process_item(item: BatchItem) -> None:
            async with semaphore:
                item.status = BatchStatus.RUNNING
                start_time = time.time()

                for attempt in range(config.retry_count):
                    try:
                        item.result = await asyncio.wait_for(
                            processor(item.data),
                            timeout=config.timeout_s / config.batch_size,
                        )
                        item.status = BatchStatus.COMPLETED
                        break
                    except Exception as e:
                        if attempt == config.retry_count - 1:
                            item.status = BatchStatus.FAILED
                            item.error = str(e)
                        else:
                            await asyncio.sleep(0.1 * (2 ** attempt))

                item.processing_time_ms = (time.time() - start_time) * 1000

                with self._lock:
                    job.processed_items += 1
                    if item.status == BatchStatus.FAILED:
                        job.failed_items += 1
                        if config.fail_fast:
                            raise RuntimeError("Fail fast triggered")
                    self._total_items_processed += 1

        try:
            # Process items in batches
            for i in range(0, len(job.items), config.batch_size):
                batch = job.items[i:i + config.batch_size]
                await asyncio.gather(
                    *[process_item(item) for item in batch],
                    return_exceptions=not config.fail_fast,
                )

            # Determine final status
            with self._lock:
                if job.failed_items == 0:
                    job.status = BatchStatus.COMPLETED
                    self._completed_jobs += 1
                elif job.failed_items < job.total_items:
                    job.status = BatchStatus.PARTIAL
                    self._completed_jobs += 1
                else:
                    job.status = BatchStatus.FAILED
                    self._failed_jobs += 1

        except Exception as e:
            with self._lock:
                job.status = BatchStatus.FAILED
                self._failed_jobs += 1

            self._emit_bus_event(
                self.BUS_TOPICS["failed"],
                {
                    "job_id": job.job_id,
                    "error": str(e),
                },
                level="error",
            )

        finally:
            with self._lock:
                job.completed_at = time.time()
                self._active_jobs -= 1

        if job.status in (BatchStatus.COMPLETED, BatchStatus.PARTIAL):
            self._emit_bus_event(
                self.BUS_TOPICS["completed"],
                {
                    "job_id": job.job_id,
                    "status": job.status.value,
                    "processed_items": job.processed_items,
                    "failed_items": job.failed_items,
                    "duration_s": job.completed_at - job.started_at,
                },
            )

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now
        stats = self.get_stats()

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "monitor_batch_processor",
                "status": "healthy",
                "active_jobs": stats["active_jobs"],
                "total_processed": stats["total_items_processed"],
            },
        )

        return True

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event",
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
            "actor": "monitor-batch-processor",
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
_processor: Optional[MonitorBatchProcessor] = None


def get_batch_processor() -> MonitorBatchProcessor:
    """Get or create the batch processor singleton.

    Returns:
        MonitorBatchProcessor instance
    """
    global _processor
    if _processor is None:
        _processor = MonitorBatchProcessor()
    return _processor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Batch Processor (Step 289)")
    parser.add_argument("--list", action="store_true", help="List batch jobs")
    parser.add_argument("--status", metavar="STATUS", help="Filter by status")
    parser.add_argument("--progress", metavar="JOB_ID", help="Get job progress")
    parser.add_argument("--cancel", metavar="JOB_ID", help="Cancel a job")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--test", type=int, metavar="N", help="Test with N items")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    processor = get_batch_processor()

    if args.list:
        status_filter = BatchStatus(args.status) if args.status else None
        jobs = processor.list_jobs(status=status_filter)
        if args.json:
            print(json.dumps(jobs, indent=2))
        else:
            print("Batch Jobs:")
            for j in jobs:
                print(f"  {j['job_id']}: {j['name']} [{j['status']}] ({j['processed_items']}/{j['total_items']})")

    if args.progress:
        progress = processor.get_job_progress(args.progress)
        if args.json:
            print(json.dumps(progress, indent=2))
        else:
            print(f"Job Progress: {args.progress}")
            for k, v in progress.items():
                print(f"  {k}: {v}")

    if args.cancel:
        success = processor.cancel_job(args.cancel)
        if args.json:
            print(json.dumps({"cancelled": success}))
        else:
            print(f"Cancel {args.cancel}: {'success' if success else 'failed'}")

    if args.stats:
        stats = processor.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Batch Processor Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    if args.test:
        async def test_processor():
            items = list(range(args.test))

            async def process_item(x: int) -> int:
                await asyncio.sleep(0.01)
                return x * 2

            job = await processor.create_job(
                name="test_job",
                items=items,
                processor=process_item,
            )

            result = await processor.wait_for_job(job.job_id)
            return result

        result = asyncio.run(test_processor())
        if args.json:
            print(json.dumps(result.to_dict() if result else {}, indent=2))
        else:
            if result:
                print(f"Test Job: {result.status.value}")
                print(f"  Processed: {result.processed_items}/{result.total_items}")
                print(f"  Failed: {result.failed_items}")
            else:
                print("Test job timed out")
