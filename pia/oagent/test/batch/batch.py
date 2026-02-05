#!/usr/bin/env python3
"""
Step 139: Test Batch Processor

Batch test operations for the Test Agent.

PBTSO Phase: TEST, DISTRIBUTE
Bus Topics:
- test.batch.start (emits)
- test.batch.progress (emits)
- test.batch.complete (emits)

Dependencies: Steps 101-138 (Test Components)
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import fcntl
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

T = TypeVar('T')


# ============================================================================
# Constants
# ============================================================================

class BatchStatus(Enum):
    """Batch job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class BatchPriority(Enum):
    """Batch job priority."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class BatchJob:
    """
    A batch job definition.

    Attributes:
        job_id: Unique job ID
        name: Job name
        items: Items to process
        processor: Processing function name
        priority: Job priority
        status: Current status
        created_at: Creation timestamp
        started_at: Start timestamp
        completed_at: Completion timestamp
        metadata: Additional metadata
    """
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    items: List[Any] = field(default_factory=list)
    processor: str = "default"
    priority: BatchPriority = BatchPriority.NORMAL
    status: BatchStatus = BatchStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3
    timeout_s: int = 3600

    @property
    def duration_s(self) -> float:
        """Get job duration."""
        if self.started_at is None:
            return 0
        end = self.completed_at or time.time()
        return end - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "items_count": len(self.items),
            "processor": self.processor,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_s": self.duration_s,
            "metadata": self.metadata,
        }


@dataclass
class BatchProgress:
    """
    Progress of a batch job.

    Attributes:
        job_id: Job ID
        total_items: Total items to process
        processed_items: Items processed
        successful_items: Successful items
        failed_items: Failed items
        current_item: Current item being processed
        eta_s: Estimated time remaining
    """
    job_id: str
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    current_item: Optional[str] = None
    eta_s: float = 0

    @property
    def percent_complete(self) -> float:
        """Get completion percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.processed_items == 0:
            return 0.0
        return (self.successful_items / self.processed_items) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "percent_complete": self.percent_complete,
            "success_rate": self.success_rate,
            "current_item": self.current_item,
            "eta_s": self.eta_s,
        }


@dataclass
class BatchResult:
    """
    Result of a batch job.

    Attributes:
        job_id: Job ID
        status: Final status
        total_items: Total items
        successful_items: Successful items
        failed_items: Failed items
        results: Individual results
        errors: Error messages
        duration_s: Total duration
    """
    job_id: str
    status: BatchStatus = BatchStatus.COMPLETED
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    results: List[Any] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.total_items == 0:
            return 0.0
        return (self.successful_items / self.total_items) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "success_rate": self.success_rate,
            "duration_s": self.duration_s,
            "errors": self.errors[:10],  # Limit errors in output
        }


@dataclass
class BatchConfig:
    """
    Configuration for batch processing.

    Attributes:
        max_workers: Maximum parallel workers
        batch_size: Items per batch
        output_dir: Output directory
        checkpoint_interval: Checkpoint interval
        retry_failed: Retry failed items
        max_retries: Maximum retries
        timeout_s: Job timeout
    """
    max_workers: int = 4
    batch_size: int = 100
    output_dir: str = ".pluribus/test-agent/batch"
    checkpoint_interval: int = 10
    retry_failed: bool = True
    max_retries: int = 3
    timeout_s: int = 3600
    progress_callback_interval_s: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "checkpoint_interval": self.checkpoint_interval,
            "retry_failed": self.retry_failed,
            "max_retries": self.max_retries,
            "timeout_s": self.timeout_s,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class BatchBus:
    """Bus interface for batch with file locking."""

    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def heartbeat(self, agent_id: str) -> None:
        """Send A2A heartbeat."""
        now = time.time()
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.emit({
                "topic": "a2a.heartbeat",
                "kind": "heartbeat",
                "actor": agent_id,
                "data": {"status": "alive"},
            })
            self._last_heartbeat = now


# ============================================================================
# Test Batch Processor
# ============================================================================

class TestBatchProcessor:
    """
    Batch test operations for the Test Agent.

    Features:
    - Parallel batch processing
    - Progress tracking
    - Checkpointing
    - Retry support
    - Job queue management

    PBTSO Phase: TEST, DISTRIBUTE
    Bus Topics: test.batch.start, test.batch.progress, test.batch.complete
    """

    BUS_TOPICS = {
        "start": "test.batch.start",
        "progress": "test.batch.progress",
        "complete": "test.batch.complete",
        "error": "test.batch.error",
    }

    def __init__(self, bus=None, config: Optional[BatchConfig] = None):
        """
        Initialize the batch processor.

        Args:
            bus: Optional bus instance
            config: Batch configuration
        """
        self.bus = bus or BatchBus()
        self.config = config or BatchConfig()
        self._jobs: Dict[str, BatchJob] = {}
        self._progress: Dict[str, BatchProgress] = {}
        self._processors: Dict[str, Callable] = {}
        self._job_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._running = False
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._lock = threading.Lock()

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Register default processors
        self._register_default_processors()

    def _register_default_processors(self) -> None:
        """Register default processors."""
        self.register_processor("default", self._default_processor)
        self.register_processor("test_run", self._test_run_processor)
        self.register_processor("report_generate", self._report_generate_processor)

    def register_processor(self, name: str, processor: Callable[[Any], Any]) -> None:
        """Register a batch processor."""
        self._processors[name] = processor

    def _default_processor(self, item: Any) -> Any:
        """Default processor - returns item as-is."""
        return item

    def _test_run_processor(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a test run item."""
        # Simulate test execution
        test_path = item.get("test_path", "")
        time.sleep(0.1)  # Simulate work
        return {
            "test_path": test_path,
            "status": "passed",
            "duration_ms": 100,
        }

    def _report_generate_processor(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a report generation item."""
        report_type = item.get("type", "json")
        return {
            "type": report_type,
            "generated": True,
        }

    def submit(self, job: BatchJob) -> str:
        """
        Submit a batch job.

        Args:
            job: Batch job to submit

        Returns:
            Job ID
        """
        with self._lock:
            self._jobs[job.job_id] = job
            self._progress[job.job_id] = BatchProgress(
                job_id=job.job_id,
                total_items=len(job.items),
            )

            # Add to queue with priority
            self._job_queue.put((-job.priority.value, job.created_at, job.job_id))

            self._emit_event("start", {
                "job_id": job.job_id,
                "name": job.name,
                "items_count": len(job.items),
                "processor": job.processor,
            })

        return job.job_id

    def process(self, job: BatchJob) -> BatchResult:
        """
        Process a batch job synchronously.

        Args:
            job: Batch job to process

        Returns:
            BatchResult with processing results
        """
        job.status = BatchStatus.RUNNING
        job.started_at = time.time()

        progress = self._progress.get(job.job_id) or BatchProgress(
            job_id=job.job_id,
            total_items=len(job.items),
        )

        processor = self._processors.get(job.processor, self._default_processor)
        results = []
        errors = []

        # Process items in batches
        for i in range(0, len(job.items), self.config.batch_size):
            batch = job.items[i:i + self.config.batch_size]

            # Process batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._process_item, processor, item, job.max_retries): item
                    for item in batch
                }

                for future in concurrent.futures.as_completed(futures, timeout=job.timeout_s):
                    item = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        progress.successful_items += 1
                    except Exception as e:
                        errors.append({
                            "item": str(item)[:100],
                            "error": str(e),
                        })
                        progress.failed_items += 1

                    progress.processed_items += 1
                    progress.current_item = str(item)[:50]

                    # Calculate ETA
                    if progress.processed_items > 0:
                        elapsed = time.time() - job.started_at
                        rate = progress.processed_items / elapsed
                        remaining = progress.total_items - progress.processed_items
                        progress.eta_s = remaining / rate if rate > 0 else 0

                    # Emit progress
                    self._emit_event("progress", progress.to_dict())

                    # Checkpoint
                    if progress.processed_items % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(job, progress, results)

        # Complete job
        job.status = BatchStatus.COMPLETED if progress.failed_items == 0 else BatchStatus.FAILED
        job.completed_at = time.time()

        result = BatchResult(
            job_id=job.job_id,
            status=job.status,
            total_items=progress.total_items,
            successful_items=progress.successful_items,
            failed_items=progress.failed_items,
            results=results,
            errors=errors,
            duration_s=job.duration_s,
        )

        self._emit_event("complete", result.to_dict())
        self._save_result(result)

        return result

    def _process_item(self, processor: Callable, item: Any, max_retries: int) -> Any:
        """Process a single item with retry."""
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                return processor(item)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))

        raise last_error

    async def process_async(self, job: BatchJob) -> BatchResult:
        """
        Process a batch job asynchronously.

        Args:
            job: Batch job to process

        Returns:
            BatchResult with processing results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process, job)

    def start_worker(self) -> None:
        """Start the batch worker."""
        self._running = True
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._executor.submit(self._worker_loop)

    def stop_worker(self) -> None:
        """Stop the batch worker."""
        self._running = False
        if self._executor:
            self._executor.shutdown(wait=True)

    def _worker_loop(self) -> None:
        """Worker loop for processing queued jobs."""
        while self._running:
            try:
                # Get next job from queue (with timeout)
                try:
                    _, _, job_id = self._job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                job = self._jobs.get(job_id)
                if job and job.status == BatchStatus.PENDING:
                    self.process(job)

            except Exception as e:
                self._emit_event("error", {"error": str(e)})

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_progress(self, job_id: str) -> Optional[BatchProgress]:
        """Get job progress."""
        return self._progress.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        job = self._jobs.get(job_id)
        if job and job.status == BatchStatus.PENDING:
            job.status = BatchStatus.CANCELLED
            return True
        return False

    def pause_job(self, job_id: str) -> bool:
        """Pause a running job."""
        job = self._jobs.get(job_id)
        if job and job.status == BatchStatus.RUNNING:
            job.status = BatchStatus.PAUSED
            return True
        return False

    def resume_job(self, job_id: str) -> bool:
        """Resume a paused job."""
        job = self._jobs.get(job_id)
        if job and job.status == BatchStatus.PAUSED:
            job.status = BatchStatus.RUNNING
            return True
        return False

    def list_jobs(self, status: Optional[BatchStatus] = None) -> List[BatchJob]:
        """List jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def _save_checkpoint(
        self,
        job: BatchJob,
        progress: BatchProgress,
        results: List[Any],
    ) -> None:
        """Save job checkpoint."""
        checkpoint_file = Path(self.config.output_dir) / f"checkpoint_{job.job_id}.json"

        checkpoint = {
            "job_id": job.job_id,
            "progress": progress.to_dict(),
            "processed_count": len(results),
            "timestamp": time.time(),
        }

        with open(checkpoint_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(checkpoint, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _save_result(self, result: BatchResult) -> None:
        """Save batch result."""
        result_file = Path(self.config.output_dir) / f"result_{result.job_id}.json"

        with open(result_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(result.to_dict(), f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.batch.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "batch",
            "actor": "test-agent",
            "data": data,
        })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Batch Processor."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Batch Processor")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a batch job")
    submit_parser.add_argument("--name", required=True, help="Job name")
    submit_parser.add_argument("--items", required=True, help="Items JSON file")
    submit_parser.add_argument("--processor", default="default", help="Processor name")
    submit_parser.add_argument("--priority", choices=["low", "normal", "high", "urgent"],
                               default="normal")

    # Process command
    process_parser = subparsers.add_parser("process", help="Process a job immediately")
    process_parser.add_argument("job_id", help="Job ID to process")

    # Status command
    status_parser = subparsers.add_parser("status", help="Get job status")
    status_parser.add_argument("job_id", help="Job ID")

    # List command
    list_parser = subparsers.add_parser("list", help="List jobs")
    list_parser.add_argument("--status", choices=["pending", "running", "completed", "failed"])
    list_parser.add_argument("--limit", type=int, default=20)

    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a job")
    cancel_parser.add_argument("job_id", help="Job ID to cancel")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/batch")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = BatchConfig(
        output_dir=args.output,
        max_workers=args.workers,
    )
    processor = TestBatchProcessor(config=config)

    if args.command == "submit":
        # Load items
        with open(args.items) as f:
            items = json.load(f)

        priority_map = {
            "low": BatchPriority.LOW,
            "normal": BatchPriority.NORMAL,
            "high": BatchPriority.HIGH,
            "urgent": BatchPriority.URGENT,
        }

        job = BatchJob(
            name=args.name,
            items=items,
            processor=args.processor,
            priority=priority_map.get(args.priority, BatchPriority.NORMAL),
        )

        job_id = processor.submit(job)

        if args.json:
            print(json.dumps({"job_id": job_id}))
        else:
            print(f"Submitted job: {job_id}")
            print(f"  Name: {args.name}")
            print(f"  Items: {len(items)}")
            print(f"  Processor: {args.processor}")

    elif args.command == "process":
        job = processor.get_job(args.job_id)
        if job is None:
            print(f"Job not found: {args.job_id}")
            exit(1)

        print(f"Processing job: {args.job_id}")
        result = processor.process(job)

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"\nResult: {result.status.value}")
            print(f"  Processed: {result.total_items}")
            print(f"  Successful: {result.successful_items}")
            print(f"  Failed: {result.failed_items}")
            print(f"  Duration: {result.duration_s:.2f}s")

    elif args.command == "status":
        job = processor.get_job(args.job_id)
        progress = processor.get_progress(args.job_id)

        if job is None:
            print(f"Job not found: {args.job_id}")
            exit(1)

        if args.json:
            print(json.dumps({
                "job": job.to_dict(),
                "progress": progress.to_dict() if progress else None,
            }, indent=2))
        else:
            print(f"Job: {job.job_id}")
            print(f"  Name: {job.name}")
            print(f"  Status: {job.status.value}")
            if progress:
                print(f"  Progress: {progress.percent_complete:.1f}%")
                print(f"  Processed: {progress.processed_items}/{progress.total_items}")
                if progress.eta_s > 0:
                    print(f"  ETA: {progress.eta_s:.0f}s")

    elif args.command == "list":
        status = BatchStatus(args.status) if args.status else None
        jobs = processor.list_jobs(status)[:args.limit]

        if args.json:
            print(json.dumps([j.to_dict() for j in jobs], indent=2))
        else:
            print(f"\nJobs ({len(jobs)}):")
            for job in jobs:
                dt = datetime.fromtimestamp(job.created_at)
                print(f"  [{job.status.value:10}] {job.job_id[:8]}... {job.name}")
                print(f"              Items: {len(job.items)}, Created: {dt.strftime('%Y-%m-%d %H:%M')}")

    elif args.command == "cancel":
        if processor.cancel_job(args.job_id):
            print(f"Cancelled: {args.job_id}")
        else:
            print(f"Cannot cancel job: {args.job_id}")
            exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
