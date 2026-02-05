#!/usr/bin/env python3
"""
batch_processor.py - Batch Query Processing (Step 39)

Batch processing for multiple research queries with
parallel execution, progress tracking, and result aggregation.

PBTSO Phase: OPTIMIZE

Bus Topics:
- a2a.research.batch.start
- a2a.research.batch.progress
- a2a.research.batch.complete
- research.batch.error

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import asyncio
import concurrent.futures
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
from typing import Any, Callable, Coroutine, Dict, Generic, List, Optional, TypeVar, Union

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class BatchStatus(Enum):
    """Batch job statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"  # Some items succeeded


class ExecutionStrategy(Enum):
    """Batch execution strategies."""
    SEQUENTIAL = "sequential"  # One at a time
    PARALLEL = "parallel"      # All at once
    CHUNKED = "chunked"        # In chunks


@dataclass
class BatchConfig:
    """Configuration for batch processor."""

    max_concurrent: int = 10
    chunk_size: int = 5
    timeout_per_item_seconds: int = 30
    total_timeout_seconds: int = 300
    retry_failed: bool = True
    max_retries: int = 2
    continue_on_error: bool = True
    emit_progress: bool = True
    progress_interval: int = 1
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchItem(Generic[T]):
    """A single item in a batch."""

    id: str
    data: T
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    attempts: int = 0
    duration_ms: float = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "status": self.status.value,
            "error": self.error,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
        }


@dataclass
class BatchJob(Generic[T, R]):
    """A batch job containing multiple items."""

    id: str
    items: List[BatchItem[T]]
    status: BatchStatus = BatchStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    total_duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_items(self) -> int:
        """Total number of items."""
        return len(self.items)

    @property
    def completed_items(self) -> int:
        """Number of completed items."""
        return sum(
            1 for item in self.items
            if item.status in (BatchStatus.COMPLETED, BatchStatus.FAILED)
        )

    @property
    def successful_items(self) -> int:
        """Number of successful items."""
        return sum(1 for item in self.items if item.status == BatchStatus.COMPLETED)

    @property
    def failed_items(self) -> int:
        """Number of failed items."""
        return sum(1 for item in self.items if item.status == BatchStatus.FAILED)

    @property
    def progress(self) -> float:
        """Progress as percentage."""
        if not self.items:
            return 0.0
        return self.completed_items / self.total_items

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "status": self.status.value,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "progress": self.progress,
            "total_duration_ms": self.total_duration_ms,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class BatchResult(Generic[R]):
    """Result of batch processing."""

    job_id: str
    status: BatchStatus
    results: List[R]
    errors: List[Dict[str, Any]]
    total_items: int
    successful_items: int
    failed_items: int
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
        }


# ============================================================================
# Batch Processor
# ============================================================================


class BatchProcessor:
    """
    Batch processing for multiple research queries.

    Features:
    - Sequential, parallel, and chunked execution
    - Progress tracking and reporting
    - Error handling and retry
    - Cancellation support
    - Result aggregation

    PBTSO Phase: OPTIMIZE

    Example:
        processor = BatchProcessor()

        # Process queries
        queries = ["query1", "query2", "query3"]
        result = await processor.process(
            queries,
            process_fn=research_fn,
        )

        # With progress callback
        result = await processor.process(
            queries,
            process_fn=research_fn,
            on_progress=lambda p: print(f"Progress: {p*100:.0f}%"),
        )
    """

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the batch processor.

        Args:
            config: Batch processor configuration
            bus: AgentBus for event emission
        """
        self.config = config or BatchConfig()
        self.bus = bus or AgentBus()

        # Active jobs
        self._jobs: Dict[str, BatchJob] = {}
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "total_items": 0,
            "successful_items": 0,
            "failed_items": 0,
        }

    async def process(
        self,
        items: List[T],
        process_fn: Callable[[T], Union[R, Coroutine[Any, Any, R]]],
        strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL,
        on_progress: Optional[Callable[[float], None]] = None,
        on_item_complete: Optional[Callable[[BatchItem[T]], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchResult[R]:
        """
        Process a batch of items.

        Args:
            items: Items to process
            process_fn: Function to process each item
            strategy: Execution strategy
            on_progress: Progress callback
            on_item_complete: Item completion callback
            metadata: Optional job metadata

        Returns:
            BatchResult with all results
        """
        # Create job
        job_id = str(uuid.uuid4())[:8]
        batch_items = [
            BatchItem(id=f"{job_id}-{i}", data=item)
            for i, item in enumerate(items)
        ]
        job = BatchJob[T, R](
            id=job_id,
            items=batch_items,
            metadata=metadata or {},
        )

        with self._lock:
            self._jobs[job_id] = job

        self._stats["total_jobs"] += 1
        self._stats["total_items"] += len(items)

        # Emit start event
        self._emit_event("a2a.research.batch.start", {
            "job_id": job_id,
            "total_items": len(items),
            "strategy": strategy.value,
        })

        try:
            job.status = BatchStatus.RUNNING
            job.started_at = time.time()

            # Execute based on strategy
            if strategy == ExecutionStrategy.SEQUENTIAL:
                await self._process_sequential(job, process_fn, on_progress, on_item_complete)
            elif strategy == ExecutionStrategy.PARALLEL:
                await self._process_parallel(job, process_fn, on_progress, on_item_complete)
            elif strategy == ExecutionStrategy.CHUNKED:
                await self._process_chunked(job, process_fn, on_progress, on_item_complete)

            # Determine final status
            job.completed_at = time.time()
            job.total_duration_ms = (job.completed_at - job.started_at) * 1000

            if job.failed_items == 0:
                job.status = BatchStatus.COMPLETED
                self._stats["completed_jobs"] += 1
            elif job.successful_items == 0:
                job.status = BatchStatus.FAILED
                self._stats["failed_jobs"] += 1
            else:
                job.status = BatchStatus.PARTIAL

            self._stats["successful_items"] += job.successful_items
            self._stats["failed_items"] += job.failed_items

            # Build result
            results = [
                item.result for item in job.items
                if item.status == BatchStatus.COMPLETED and item.result is not None
            ]
            errors = [
                {"id": item.id, "error": item.error}
                for item in job.items
                if item.status == BatchStatus.FAILED
            ]

            result = BatchResult(
                job_id=job_id,
                status=job.status,
                results=results,
                errors=errors,
                total_items=job.total_items,
                successful_items=job.successful_items,
                failed_items=job.failed_items,
                duration_ms=job.total_duration_ms,
            )

            # Emit complete event
            self._emit_event("a2a.research.batch.complete", result.to_dict())

            return result

        except Exception as e:
            job.status = BatchStatus.FAILED
            job.completed_at = time.time()

            self._emit_event("research.batch.error", {
                "job_id": job_id,
                "error": str(e),
            })

            raise

    async def _process_sequential(
        self,
        job: BatchJob,
        process_fn: Callable,
        on_progress: Optional[Callable],
        on_item_complete: Optional[Callable],
    ) -> None:
        """Process items sequentially."""
        for item in job.items:
            await self._process_item(item, process_fn)

            if on_item_complete:
                on_item_complete(item)

            if on_progress:
                on_progress(job.progress)

            self._emit_progress(job)

    async def _process_parallel(
        self,
        job: BatchJob,
        process_fn: Callable,
        on_progress: Optional[Callable],
        on_item_complete: Optional[Callable],
    ) -> None:
        """Process items in parallel."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def process_with_semaphore(item: BatchItem) -> None:
            async with semaphore:
                await self._process_item(item, process_fn)

                if on_item_complete:
                    on_item_complete(item)

                if on_progress:
                    on_progress(job.progress)

        # Create tasks
        tasks = [
            asyncio.create_task(process_with_semaphore(item))
            for item in job.items
        ]

        # Progress reporting
        if self.config.emit_progress:
            progress_task = asyncio.create_task(
                self._emit_progress_loop(job, on_progress)
            )

        # Wait for all
        await asyncio.gather(*tasks, return_exceptions=True)

        if self.config.emit_progress:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

    async def _process_chunked(
        self,
        job: BatchJob,
        process_fn: Callable,
        on_progress: Optional[Callable],
        on_item_complete: Optional[Callable],
    ) -> None:
        """Process items in chunks."""
        chunks = [
            job.items[i:i + self.config.chunk_size]
            for i in range(0, len(job.items), self.config.chunk_size)
        ]

        for chunk in chunks:
            # Process chunk in parallel
            tasks = [
                asyncio.create_task(self._process_item(item, process_fn))
                for item in chunk
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            for item in chunk:
                if on_item_complete:
                    on_item_complete(item)

            if on_progress:
                on_progress(job.progress)

            self._emit_progress(job)

    async def _process_item(
        self,
        item: BatchItem,
        process_fn: Callable,
    ) -> None:
        """Process a single item."""
        item.status = BatchStatus.RUNNING
        item.started_at = time.time()
        item.attempts += 1

        try:
            # Run with timeout
            result = await asyncio.wait_for(
                self._run_process_fn(process_fn, item.data),
                timeout=self.config.timeout_per_item_seconds,
            )

            item.result = result
            item.status = BatchStatus.COMPLETED
            item.error = None

        except asyncio.TimeoutError:
            item.error = f"Timeout after {self.config.timeout_per_item_seconds}s"
            item.status = BatchStatus.FAILED

        except Exception as e:
            item.error = str(e)
            item.status = BatchStatus.FAILED

            # Retry if configured
            if self.config.retry_failed and item.attempts < self.config.max_retries:
                item.status = BatchStatus.PENDING
                await self._process_item(item, process_fn)

        finally:
            item.completed_at = time.time()
            item.duration_ms = (item.completed_at - item.started_at) * 1000

    async def _run_process_fn(
        self,
        process_fn: Callable,
        data: Any,
    ) -> Any:
        """Run the process function, handling both sync and async."""
        result = process_fn(data)
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _emit_progress_loop(
        self,
        job: BatchJob,
        on_progress: Optional[Callable],
    ) -> None:
        """Emit progress at intervals."""
        while True:
            await asyncio.sleep(self.config.progress_interval)
            self._emit_progress(job)
            if on_progress:
                on_progress(job.progress)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == BatchStatus.RUNNING:
                job.status = BatchStatus.CANCELLED
                return True
            return False

    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            **self._stats,
            "active_jobs": sum(
                1 for job in self._jobs.values()
                if job.status == BatchStatus.RUNNING
            ),
        }

    def _emit_progress(self, job: BatchJob) -> None:
        """Emit progress event."""
        if not self.config.emit_progress:
            return

        self._emit_event("a2a.research.batch.progress", {
            "job_id": job.id,
            "progress": job.progress,
            "completed_items": job.completed_items,
            "total_items": job.total_items,
        })

    def _emit_event(self, topic: str, data: Dict[str, Any]) -> str:
        """Emit event with file locking."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "batch",
            "level": "info",
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id


# ============================================================================
# Synchronous Wrapper
# ============================================================================


class SyncBatchProcessor:
    """Synchronous wrapper for BatchProcessor."""

    def __init__(self, config: Optional[BatchConfig] = None):
        self._processor = BatchProcessor(config)

    def process(
        self,
        items: List[T],
        process_fn: Callable[[T], R],
        strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> BatchResult[R]:
        """Process a batch synchronously."""
        return asyncio.run(
            self._processor.process(
                items,
                process_fn,
                strategy,
                on_progress,
            )
        )


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Batch Processor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch Processor (Step 39)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run batch processing demo")
    demo_parser.add_argument("--items", type=int, default=10, help="Number of items")
    demo_parser.add_argument("--strategy", choices=["sequential", "parallel", "chunked"], default="parallel")

    args = parser.parse_args()

    processor = BatchProcessor()

    if args.command == "stats":
        stats = processor.get_stats()

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Batch Processor Statistics:")
            print(f"  Total Jobs: {stats['total_jobs']}")
            print(f"  Completed: {stats['completed_jobs']}")
            print(f"  Failed: {stats['failed_jobs']}")
            print(f"  Total Items: {stats['total_items']}")
            print(f"  Successful Items: {stats['successful_items']}")
            print(f"  Failed Items: {stats['failed_items']}")

    elif args.command == "demo":
        print(f"Running batch processing demo ({args.strategy})...\n")

        # Create demo items
        items = [f"item-{i}" for i in range(args.items)]

        # Demo process function
        async def demo_process(item: str) -> str:
            import random
            await asyncio.sleep(random.uniform(0.1, 0.5))

            # Randomly fail some
            if random.random() < 0.1:
                raise ValueError(f"Random failure for {item}")

            return f"Processed: {item}"

        # Process with progress
        def on_progress(p: float):
            bar_width = 40
            filled = int(bar_width * p)
            bar = "=" * filled + "-" * (bar_width - filled)
            print(f"\rProgress: [{bar}] {p*100:.0f}%", end="", flush=True)

        strategy = ExecutionStrategy(args.strategy)

        async def run():
            result = await processor.process(
                items,
                demo_process,
                strategy=strategy,
                on_progress=on_progress,
            )
            return result

        result = asyncio.run(run())

        print("\n\nResults:")
        print(f"  Status: {result.status.value}")
        print(f"  Total: {result.total_items}")
        print(f"  Successful: {result.successful_items}")
        print(f"  Failed: {result.failed_items}")
        print(f"  Duration: {result.duration_ms:.1f}ms")

        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for err in result.errors[:5]:
                print(f"  - {err['id']}: {err['error']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
