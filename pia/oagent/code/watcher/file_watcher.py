#!/usr/bin/env python3
"""
file_watcher.py - File Watcher (Step 69)

PBTSO Phase: ITERATE, VERIFY

Provides:
- Real-time file change detection
- Debounced event handling
- Pattern-based filtering
- Change batching
- External change detection

Bus Topics:
- code.watcher.event
- code.watcher.batch
- code.watcher.started
- code.watcher.stopped

Protocol: DKIN v30, PAIP v16
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set


# =============================================================================
# Types
# =============================================================================

class EventType(Enum):
    """Type of file system event."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    MOVED = "moved"


@dataclass
class FileEvent:
    """
    Represents a file system event.
    """
    id: str
    event_type: EventType
    path: str
    timestamp: float
    old_path: Optional[str] = None  # For rename/move
    is_directory: bool = False
    size: int = 0
    hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "path": self.path,
            "timestamp": self.timestamp,
            "old_path": self.old_path,
            "is_directory": self.is_directory,
            "size": self.size,
            "hash": self.hash,
        }


@dataclass
class WatchConfig:
    """Configuration for file watching."""
    include_patterns: List[str] = field(default_factory=lambda: ["**/*"])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "**/__pycache__/**", "**/node_modules/**", "**/.git/**",
        "**/.venv/**", "**/dist/**", "**/*.pyc", "**/.DS_Store"
    ])
    debounce_ms: int = 100
    batch_interval_ms: int = 500
    recursive: bool = True
    follow_symlinks: bool = False
    ignore_hidden: bool = True


EventHandler = Callable[[FileEvent], Coroutine[Any, Any, None]]
BatchHandler = Callable[[List[FileEvent]], Coroutine[Any, Any, None]]


# =============================================================================
# File Watcher
# =============================================================================

class FileWatcher:
    """
    Real-time file change detection.

    PBTSO Phase: ITERATE, VERIFY

    Features:
    - Polling-based change detection (cross-platform)
    - Pattern-based filtering
    - Debounced event handling
    - Event batching
    - External change detection

    Usage:
        watcher = FileWatcher(working_dir)
        watcher.on_event(handle_event)
        await watcher.start()
    """

    BUS_TOPICS = {
        "event": "code.watcher.event",
        "batch": "code.watcher.batch",
        "started": "code.watcher.started",
        "stopped": "code.watcher.stopped",
    }

    def __init__(
        self,
        working_dir: Path,
        bus: Optional[Any] = None,
        config: Optional[WatchConfig] = None,
        poll_interval_ms: int = 500,
    ):
        self.working_dir = Path(working_dir)
        self.bus = bus
        self.config = config or WatchConfig()
        self.poll_interval_ms = poll_interval_ms

        self._running = False
        self._watch_task: Optional[asyncio.Task] = None
        self._event_handlers: List[EventHandler] = []
        self._batch_handlers: List[BatchHandler] = []

        # State tracking
        self._file_states: Dict[str, Dict[str, Any]] = {}  # path -> {mtime, size, hash}
        self._pending_events: Dict[str, FileEvent] = {}  # path -> event (for debouncing)
        self._event_batch: List[FileEvent] = []
        self._last_batch_time: float = 0

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def on_event(self, handler: EventHandler) -> None:
        """Register an event handler."""
        self._event_handlers.append(handler)

    def on_batch(self, handler: BatchHandler) -> None:
        """Register a batch event handler."""
        self._batch_handlers.append(handler)

    def remove_handler(self, handler: EventHandler) -> None:
        """Remove an event handler."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    # =========================================================================
    # Watch Control
    # =========================================================================

    async def start(self) -> None:
        """Start watching for file changes."""
        if self._running:
            return

        self._running = True

        # Initialize file states
        await self._scan_directory()

        # Emit started event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["started"],
                "kind": "watcher",
                "actor": "code-agent",
                "data": {
                    "working_dir": str(self.working_dir),
                    "files_tracked": len(self._file_states),
                },
            })

        # Start watch loop
        self._watch_task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop watching for file changes."""
        self._running = False

        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None

        # Emit stopped event
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["stopped"],
                "kind": "watcher",
                "actor": "code-agent",
                "data": {
                    "total_events": len(self._event_batch),
                },
            })

    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    # =========================================================================
    # Watch Loop
    # =========================================================================

    async def _watch_loop(self) -> None:
        """Main watch loop."""
        while self._running:
            try:
                await self._check_changes()
                await self._process_pending_events()
                await self._flush_batch_if_needed()
                await asyncio.sleep(self.poll_interval_ms / 1000)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue watching
                if self.bus:
                    self.bus.emit({
                        "topic": "code.watcher.error",
                        "kind": "error",
                        "actor": "code-agent",
                        "data": {"error": str(e)},
                    })
                await asyncio.sleep(1)

    async def _check_changes(self) -> None:
        """Check for file system changes."""
        current_files: Set[str] = set()

        # Walk directory
        for file_path in self._walk_directory():
            rel_path = str(file_path.relative_to(self.working_dir))
            current_files.add(rel_path)

            try:
                stat = file_path.stat()
                mtime = stat.st_mtime
                size = stat.st_size

                if rel_path not in self._file_states:
                    # New file
                    self._queue_event(FileEvent(
                        id=f"evt-{uuid.uuid4().hex[:8]}",
                        event_type=EventType.CREATED,
                        path=rel_path,
                        timestamp=time.time(),
                        is_directory=file_path.is_dir(),
                        size=size,
                    ))
                    self._file_states[rel_path] = {"mtime": mtime, "size": size}

                else:
                    old_state = self._file_states[rel_path]
                    if mtime != old_state["mtime"] or size != old_state["size"]:
                        # Modified file
                        self._queue_event(FileEvent(
                            id=f"evt-{uuid.uuid4().hex[:8]}",
                            event_type=EventType.MODIFIED,
                            path=rel_path,
                            timestamp=time.time(),
                            is_directory=file_path.is_dir(),
                            size=size,
                        ))
                        self._file_states[rel_path] = {"mtime": mtime, "size": size}

            except (OSError, PermissionError):
                pass

        # Check for deleted files
        deleted = set(self._file_states.keys()) - current_files
        for rel_path in deleted:
            self._queue_event(FileEvent(
                id=f"evt-{uuid.uuid4().hex[:8]}",
                event_type=EventType.DELETED,
                path=rel_path,
                timestamp=time.time(),
            ))
            del self._file_states[rel_path]

    def _walk_directory(self) -> List[Path]:
        """Walk directory with filtering."""
        files: List[Path] = []

        if self.config.recursive:
            iterator = self.working_dir.rglob("*")
        else:
            iterator = self.working_dir.glob("*")

        for file_path in iterator:
            if not file_path.is_file():
                continue

            rel_path = str(file_path.relative_to(self.working_dir))

            # Check hidden files
            if self.config.ignore_hidden:
                if any(part.startswith(".") for part in file_path.parts):
                    continue

            # Check exclude patterns
            if self._matches_patterns(rel_path, self.config.exclude_patterns):
                continue

            # Check include patterns
            if not self._matches_patterns(rel_path, self.config.include_patterns):
                continue

            files.append(file_path)

        return files

    def _matches_patterns(self, path: str, patterns: List[str]) -> bool:
        """Check if path matches any pattern."""
        for pattern in patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False

    # =========================================================================
    # Event Processing
    # =========================================================================

    def _queue_event(self, event: FileEvent) -> None:
        """Queue an event for debounced processing."""
        # Debounce: replace pending event for same path
        self._pending_events[event.path] = event

    async def _process_pending_events(self) -> None:
        """Process pending events after debounce period."""
        if not self._pending_events:
            return

        now = time.time()
        debounce_seconds = self.config.debounce_ms / 1000

        to_process = []
        for path, event in list(self._pending_events.items()):
            if now - event.timestamp >= debounce_seconds:
                to_process.append(event)
                del self._pending_events[path]

        for event in to_process:
            await self._emit_event(event)

    async def _emit_event(self, event: FileEvent) -> None:
        """Emit an event to handlers."""
        # Add to batch
        self._event_batch.append(event)

        # Call individual event handlers
        for handler in self._event_handlers:
            try:
                await handler(event)
            except Exception:
                pass

        # Emit to bus
        if self.bus:
            self.bus.emit({
                "topic": self.BUS_TOPICS["event"],
                "kind": "file_event",
                "actor": "code-agent",
                "data": event.to_dict(),
            })

    async def _flush_batch_if_needed(self) -> None:
        """Flush event batch if interval has passed."""
        if not self._event_batch:
            return

        now = time.time()
        batch_seconds = self.config.batch_interval_ms / 1000

        if now - self._last_batch_time >= batch_seconds:
            batch = self._event_batch.copy()
            self._event_batch = []
            self._last_batch_time = now

            # Call batch handlers
            for handler in self._batch_handlers:
                try:
                    await handler(batch)
                except Exception:
                    pass

            # Emit batch to bus
            if self.bus:
                self.bus.emit({
                    "topic": self.BUS_TOPICS["batch"],
                    "kind": "file_batch",
                    "actor": "code-agent",
                    "data": {
                        "count": len(batch),
                        "events": [e.to_dict() for e in batch],
                    },
                })

    # =========================================================================
    # Scanning
    # =========================================================================

    async def _scan_directory(self) -> None:
        """Initial scan of directory to build state."""
        for file_path in self._walk_directory():
            rel_path = str(file_path.relative_to(self.working_dir))
            try:
                stat = file_path.stat()
                self._file_states[rel_path] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                }
            except (OSError, PermissionError):
                pass

    async def rescan(self) -> List[FileEvent]:
        """Rescan directory and return all detected changes."""
        events: List[FileEvent] = []

        old_states = self._file_states.copy()
        self._file_states = {}

        await self._scan_directory()

        # Find changes
        new_files = set(self._file_states.keys()) - set(old_states.keys())
        deleted_files = set(old_states.keys()) - set(self._file_states.keys())
        common_files = set(self._file_states.keys()) & set(old_states.keys())

        for path in new_files:
            events.append(FileEvent(
                id=f"evt-{uuid.uuid4().hex[:8]}",
                event_type=EventType.CREATED,
                path=path,
                timestamp=time.time(),
                size=self._file_states[path]["size"],
            ))

        for path in deleted_files:
            events.append(FileEvent(
                id=f"evt-{uuid.uuid4().hex[:8]}",
                event_type=EventType.DELETED,
                path=path,
                timestamp=time.time(),
            ))

        for path in common_files:
            old = old_states[path]
            new = self._file_states[path]
            if old["mtime"] != new["mtime"] or old["size"] != new["size"]:
                events.append(FileEvent(
                    id=f"evt-{uuid.uuid4().hex[:8]}",
                    event_type=EventType.MODIFIED,
                    path=path,
                    timestamp=time.time(),
                    size=new["size"],
                ))

        return events

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_tracked_files(self) -> List[str]:
        """Get list of tracked files."""
        return list(self._file_states.keys())

    def get_file_state(self, path: str) -> Optional[Dict[str, Any]]:
        """Get state of a tracked file."""
        return self._file_states.get(path)

    def is_tracking(self, path: str) -> bool:
        """Check if a file is being tracked."""
        return path in self._file_states

    def add_exclude_pattern(self, pattern: str) -> None:
        """Add an exclude pattern."""
        if pattern not in self.config.exclude_patterns:
            self.config.exclude_patterns.append(pattern)

    def remove_exclude_pattern(self, pattern: str) -> None:
        """Remove an exclude pattern."""
        if pattern in self.config.exclude_patterns:
            self.config.exclude_patterns.remove(pattern)

    def get_stats(self) -> Dict[str, Any]:
        """Get watcher statistics."""
        return {
            "running": self._running,
            "files_tracked": len(self._file_states),
            "pending_events": len(self._pending_events),
            "batch_size": len(self._event_batch),
            "handlers": len(self._event_handlers),
            "batch_handlers": len(self._batch_handlers),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

async def watch_directory(
    directory: Path,
    on_change: Callable[[FileEvent], Coroutine[Any, Any, None]],
    patterns: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> FileWatcher:
    """
    Convenience function to start watching a directory.

    Args:
        directory: Directory to watch
        on_change: Callback for file changes
        patterns: Include patterns
        exclude: Exclude patterns

    Returns:
        Running FileWatcher instance
    """
    config = WatchConfig()
    if patterns:
        config.include_patterns = patterns
    if exclude:
        config.exclude_patterns.extend(exclude)

    watcher = FileWatcher(directory, config=config)
    watcher.on_event(on_change)
    await watcher.start()

    return watcher


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for File Watcher."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="File Watcher (Step 69)")
    parser.add_argument("directory", nargs="?", default=".", help="Directory to watch")
    parser.add_argument("--pattern", "-p", nargs="*", help="Include patterns")
    parser.add_argument("--exclude", "-e", nargs="*", help="Exclude patterns")
    parser.add_argument("--debounce", type=int, default=100, help="Debounce ms")
    parser.add_argument("--poll", type=int, default=500, help="Poll interval ms")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    config = WatchConfig(
        debounce_ms=args.debounce,
    )
    if args.pattern:
        config.include_patterns = args.pattern
    if args.exclude:
        config.exclude_patterns.extend(args.exclude)

    watcher = FileWatcher(
        Path(args.directory),
        config=config,
        poll_interval_ms=args.poll,
    )

    async def on_event(event: FileEvent):
        if args.json:
            print(json.dumps(event.to_dict()))
        else:
            print(f"[{event.event_type.value}] {event.path}")

    async def run():
        watcher.on_event(on_event)
        await watcher.start()

        print(f"Watching {args.directory} (Ctrl+C to stop)")
        print(f"Tracking {len(watcher.get_tracked_files())} files")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await watcher.stop()
            print("\nStopped")

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
