#!/usr/bin/env python3
"""
Review Batch Processor (Step 189)

Batch processing system for review operations with parallel execution,
progress tracking, and failure handling.

PBTSO Phase: BUILD, DISTRIBUTE
Bus Topics: review.batch.start, review.batch.progress, review.batch.complete

Batch Features:
- Parallel processing with concurrency control
- Progress tracking
- Partial failure handling
- Retry support
- Result aggregation

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Awaitable

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type


# ============================================================================
# Types
# ============================================================================

class BatchState(Enum):
    """Batch processing states."""
    PENDING = "pending"       # Batch created, not started
    RUNNING = "running"       # Batch is processing
    PAUSED = "paused"         # Batch is paused
    COMPLETED = "completed"   # All items processed
    FAILED = "failed"         # Batch failed
    CANCELLED = "cancelled"   # Batch was cancelled


class ItemState(Enum):
    """Individual item states."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class BatchItem(Generic[T, R]):
    """
    An item in a batch.

    Attributes:
        item_id: Unique item identifier
        input: Input data
        result: Processing result
        state: Item state
        error: Error message if failed
        attempts: Number of processing attempts
        started_at: Processing start time
        completed_at: Processing completion time
        duration_ms: Processing duration
    """
    item_id: str
    input: T
    result: Optional[R] = None
    state: ItemState = ItemState.PENDING
    error: Optional[str] = None
    attempts: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "state": self.state.value,
            "error": self.error,
            "attempts": self.attempts,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class BatchConfig:
    """
    Configuration for batch processing.

    Attributes:
        max_concurrency: Maximum concurrent items
        timeout_per_item_seconds: Timeout per item
        max_retries: Maximum retries per item
        retry_delay_seconds: Delay between retries
        fail_fast: Stop on first failure
        continue_on_error: Continue processing on errors
        progress_interval_seconds: Progress update interval
    """
    max_concurrency: int = 10
    timeout_per_item_seconds: float = 60.0
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    fail_fast: bool = False
    continue_on_error: bool = True
    progress_interval_seconds: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BatchProgress:
    """
    Batch processing progress.

    Attributes:
        total: Total items
        completed: Completed items
        failed: Failed items
        pending: Pending items
        processing: Currently processing
        percent_complete: Completion percentage
        estimated_remaining_seconds: Estimated time remaining
    """
    total: int = 0
    completed: int = 0
    failed: int = 0
    pending: int = 0
    processing: int = 0
    percent_complete: float = 0.0
    estimated_remaining_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "percent_complete": round(self.percent_complete, 1),
            "estimated_remaining_seconds": round(self.estimated_remaining_seconds, 1),
        }


@dataclass
class BatchResult(Generic[R]):
    """
    Result of batch processing.

    Attributes:
        batch_id: Unique batch identifier
        state: Final batch state
        total_items: Total items processed
        successful_items: Successfully completed items
        failed_items: Failed items
        results: List of individual results
        started_at: Batch start time
        completed_at: Batch completion time
        duration_ms: Total duration
        errors: List of errors
    """
    batch_id: str
    state: BatchState
    total_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    results: List[R] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    duration_ms: float = 0
    errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "state": self.state.value,
            "total_items": self.total_items,
            "successful_items": self.successful_items,
            "failed_items": self.failed_items,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": round(self.duration_ms, 2),
            "errors": self.errors,
            "success_rate": round(
                self.successful_items / self.total_items * 100
                if self.total_items > 0 else 0, 1
            ),
        }


# ============================================================================
# Batch Processor
# ============================================================================

class BatchProcessor(Generic[T, R]):
    """
    Batch processing system for review operations.

    Example:
        # Define processor function
        async def process_file(file_path: str) -> ReviewResult:
            # ... process file ...
            return ReviewResult(...)

        # Create processor
        processor = BatchProcessor(process_file)

        # Process batch
        files = ["/path/file1.py", "/path/file2.py", "/path/file3.py"]
        result = await processor.process(files)

        print(f"Completed: {result.successful_items}/{result.total_items}")

        # With progress callback
        def on_progress(progress: BatchProgress):
            print(f"{progress.percent_complete}% complete")

        result = await processor.process(files, on_progress=on_progress)
    """

    BUS_TOPICS = {
        "start": "review.batch.start",
        "progress": "review.batch.progress",
        "complete": "review.batch.complete",
        "item_complete": "review.batch.item.complete",
        "item_failed": "review.batch.item.failed",
    }

    def __init__(
        self,
        processor_fn: Callable[[T], Awaitable[R]],
        config: Optional[BatchConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the batch processor.

        Args:
            processor_fn: Async function to process each item
            config: Batch processing configuration
            bus_path: Path to event bus file
        """
        self.processor_fn = processor_fn
        self.config = config or BatchConfig()
        self.bus_path = bus_path or self._get_bus_path()

        # State
        self._current_batch_id: Optional[str] = None
        self._items: Dict[str, BatchItem[T, R]] = {}
        self._state = BatchState.PENDING
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._cancelled = False
        self._paused = False

        # Progress tracking
        self._start_time: Optional[float] = None
        self._item_durations: List[float] = []
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "batch") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "batch-processor",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _calculate_progress(self) -> BatchProgress:
        """Calculate current progress."""
        total = len(self._items)
        completed = sum(1 for i in self._items.values() if i.state == ItemState.COMPLETED)
        failed = sum(1 for i in self._items.values() if i.state == ItemState.FAILED)
        processing = sum(1 for i in self._items.values() if i.state == ItemState.PROCESSING)
        pending = sum(1 for i in self._items.values() if i.state == ItemState.PENDING)

        percent = (completed + failed) / total * 100 if total > 0 else 0

        # Estimate remaining time
        estimated_remaining = 0.0
        if self._item_durations and (pending + processing) > 0:
            avg_duration = sum(self._item_durations) / len(self._item_durations)
            estimated_remaining = avg_duration * (pending + processing) / 1000  # Convert to seconds

        return BatchProgress(
            total=total,
            completed=completed,
            failed=failed,
            pending=pending,
            processing=processing,
            percent_complete=percent,
            estimated_remaining_seconds=estimated_remaining,
        )

    async def _process_item(self, item: BatchItem[T, R]) -> None:
        """Process a single item."""
        if self._cancelled:
            item.state = ItemState.SKIPPED
            return

        while self._paused:
            await asyncio.sleep(0.5)

        async with self._semaphore:
            item.state = ItemState.PROCESSING
            item.started_at = time.time()
            item.attempts += 1

            try:
                result = await asyncio.wait_for(
                    self.processor_fn(item.input),
                    timeout=self.config.timeout_per_item_seconds,
                )
                item.result = result
                item.state = ItemState.COMPLETED
                item.completed_at = time.time()
                item.duration_ms = (item.completed_at - item.started_at) * 1000
                self._item_durations.append(item.duration_ms)

                self._emit_event(self.BUS_TOPICS["item_complete"], {
                    "batch_id": self._current_batch_id,
                    "item_id": item.item_id,
                    "duration_ms": item.duration_ms,
                })

            except asyncio.TimeoutError:
                item.error = f"Timeout after {self.config.timeout_per_item_seconds}s"
                await self._handle_item_failure(item)

            except Exception as e:
                item.error = str(e)
                await self._handle_item_failure(item)

    async def _handle_item_failure(self, item: BatchItem[T, R]) -> None:
        """Handle item processing failure."""
        if item.attempts < self.config.max_retries and self.config.continue_on_error:
            item.state = ItemState.RETRYING
            await asyncio.sleep(self.config.retry_delay_seconds)
            item.state = ItemState.PENDING
            await self._process_item(item)
        else:
            item.state = ItemState.FAILED
            item.completed_at = time.time()
            if item.started_at:
                item.duration_ms = (item.completed_at - item.started_at) * 1000

            self._emit_event(self.BUS_TOPICS["item_failed"], {
                "batch_id": self._current_batch_id,
                "item_id": item.item_id,
                "error": item.error,
                "attempts": item.attempts,
            })

            if self.config.fail_fast:
                self._cancelled = True

    async def process(
        self,
        inputs: List[T],
        on_progress: Optional[Callable[[BatchProgress], None]] = None,
    ) -> BatchResult[R]:
        """
        Process a batch of items.

        Args:
            inputs: List of input items
            on_progress: Optional progress callback

        Returns:
            BatchResult with processing results

        Emits:
            review.batch.start
            review.batch.progress
            review.batch.complete
        """
        self._current_batch_id = str(uuid.uuid4())[:8]
        self._start_time = time.time()
        self._state = BatchState.RUNNING
        self._cancelled = False
        self._paused = False
        self._items.clear()
        self._item_durations.clear()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

        # Create items
        for i, input_data in enumerate(inputs):
            item_id = f"{self._current_batch_id}-{i:04d}"
            self._items[item_id] = BatchItem(
                item_id=item_id,
                input=input_data,
            )

        # Emit start event
        self._emit_event(self.BUS_TOPICS["start"], {
            "batch_id": self._current_batch_id,
            "total_items": len(self._items),
            "max_concurrency": self.config.max_concurrency,
        })

        # Start progress reporting
        progress_task = asyncio.create_task(
            self._report_progress(on_progress)
        )

        try:
            # Process all items
            await asyncio.gather(*[
                self._process_item(item)
                for item in self._items.values()
            ])

            # Determine final state
            if self._cancelled:
                self._state = BatchState.CANCELLED
            elif any(i.state == ItemState.FAILED for i in self._items.values()):
                if all(i.state in (ItemState.COMPLETED, ItemState.FAILED) for i in self._items.values()):
                    self._state = BatchState.COMPLETED
                else:
                    self._state = BatchState.FAILED
            else:
                self._state = BatchState.COMPLETED

        except Exception as e:
            self._state = BatchState.FAILED

        finally:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

        # Build result
        result = BatchResult[R](
            batch_id=self._current_batch_id,
            state=self._state,
            total_items=len(self._items),
            successful_items=sum(1 for i in self._items.values() if i.state == ItemState.COMPLETED),
            failed_items=sum(1 for i in self._items.values() if i.state == ItemState.FAILED),
            results=[i.result for i in self._items.values() if i.result is not None],
            started_at=datetime.fromtimestamp(self._start_time, tz=timezone.utc).isoformat() + "Z",
            completed_at=datetime.now(timezone.utc).isoformat() + "Z",
            duration_ms=(time.time() - self._start_time) * 1000,
            errors=[
                {"item_id": i.item_id, "error": i.error}
                for i in self._items.values()
                if i.error
            ],
        )

        # Emit complete event
        self._emit_event(self.BUS_TOPICS["complete"], result.to_dict())

        return result

    async def _report_progress(
        self,
        on_progress: Optional[Callable[[BatchProgress], None]],
    ) -> None:
        """Report progress periodically."""
        while True:
            await asyncio.sleep(self.config.progress_interval_seconds)

            progress = self._calculate_progress()

            self._emit_event(self.BUS_TOPICS["progress"], {
                "batch_id": self._current_batch_id,
                **progress.to_dict(),
            })

            if on_progress:
                try:
                    on_progress(progress)
                except Exception:
                    pass

    def pause(self) -> bool:
        """Pause batch processing."""
        if self._state == BatchState.RUNNING:
            self._paused = True
            self._state = BatchState.PAUSED
            return True
        return False

    def resume(self) -> bool:
        """Resume batch processing."""
        if self._state == BatchState.PAUSED:
            self._paused = False
            self._state = BatchState.RUNNING
            return True
        return False

    def cancel(self) -> bool:
        """Cancel batch processing."""
        if self._state in (BatchState.RUNNING, BatchState.PAUSED):
            self._cancelled = True
            self._paused = False
            return True
        return False

    def get_progress(self) -> BatchProgress:
        """Get current progress."""
        return self._calculate_progress()

    def get_items(self) -> List[Dict[str, Any]]:
        """Get all items status."""
        return [i.to_dict() for i in self._items.values()]

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        progress = self._calculate_progress()
        status = {
            "agent": "batch-processor",
            "healthy": True,
            "batch_id": self._current_batch_id,
            "state": self._state.value,
            "progress": progress.to_dict(),
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self._emit_event("a2a.heartbeat", status, kind="heartbeat")
        return status


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Batch Processor."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Batch Processor (Step 189)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo batch")
    demo_parser.add_argument("--items", type=int, default=10, help="Number of items")
    demo_parser.add_argument("--concurrency", type=int, default=3, help="Concurrency")
    demo_parser.add_argument("--fail-rate", type=float, default=0.1, help="Failure rate")

    # Config command
    subparsers.add_parser("config", help="Show configuration")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.command == "demo":
        import random

        # Demo processor function
        async def demo_processor(item: int) -> Dict[str, Any]:
            # Simulate processing
            await asyncio.sleep(random.uniform(0.1, 0.5))

            # Simulate occasional failures
            if random.random() < args.fail_rate:
                raise Exception(f"Simulated failure for item {item}")

            return {"item": item, "result": item * 2}

        config = BatchConfig(max_concurrency=args.concurrency)
        processor = BatchProcessor(demo_processor, config)

        def on_progress(progress: BatchProgress):
            if not args.json:
                print(f"\r  Progress: {progress.percent_complete:.1f}% "
                      f"({progress.completed}/{progress.total})", end="")

        if not args.json:
            print(f"Processing {args.items} items with concurrency {args.concurrency}...")

        result = asyncio.run(processor.process(
            list(range(args.items)),
            on_progress=on_progress,
        ))

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"\n\nBatch Complete: {result.batch_id}")
            print(f"  State: {result.state.value}")
            print(f"  Success: {result.successful_items}/{result.total_items}")
            print(f"  Failed: {result.failed_items}")
            print(f"  Duration: {result.duration_ms:.1f}ms")
            if result.errors:
                print(f"  Errors:")
                for err in result.errors[:5]:
                    print(f"    {err['item_id']}: {err['error']}")

    elif args.command == "config":
        config = BatchConfig()
        if args.json:
            print(json.dumps(config.to_dict(), indent=2))
        else:
            print("Batch Processor Configuration")
            for k, v in config.to_dict().items():
                print(f"  {k}: {v}")

    else:
        # Default: show status
        config = BatchConfig()
        if args.json:
            print(json.dumps({"config": config.to_dict()}, indent=2))
        else:
            print("Batch Processor")
            print(f"  Max Concurrency: {config.max_concurrency}")
            print(f"  Timeout: {config.timeout_per_item_seconds}s")
            print(f"  Max Retries: {config.max_retries}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
