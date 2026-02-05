#!/usr/bin/env python3
"""
incremental_updater.py - Incremental Updater (Step 27)

Update indexes incrementally as files change.
Watches filesystem and maintains index freshness.

PBTSO Phase: ITERATE, MAINTAIN

Bus Topics:
- a2a.research.update.start
- a2a.research.update.complete
- research.update.file
- research.update.batch

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class UpdateType(Enum):
    """Type of update operation."""
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RENAME = "rename"


class UpdatePriority(Enum):
    """Priority of update."""
    IMMEDIATE = "immediate"   # Process right away
    HIGH = "high"             # Process soon
    NORMAL = "normal"         # Process in batch
    LOW = "low"               # Process when idle


@dataclass
class UpdaterConfig:
    """Configuration for incremental updater."""

    watch_paths: List[str] = field(default_factory=list)
    extensions: List[str] = field(default_factory=lambda: [".py", ".ts", ".js", ".go", ".rs"])
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "__pycache__", "node_modules", ".git", ".venv", "*.pyc",
        "dist", "build", ".egg-info",
    ])
    batch_size: int = 50
    batch_delay_ms: int = 500
    debounce_ms: int = 100
    max_queue_size: int = 1000
    enable_watch: bool = True
    state_path: Optional[str] = None
    bus_path: Optional[str] = None

    def __post_init__(self):
        if not self.watch_paths:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.watch_paths = [pluribus_root]
        if self.state_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.state_path = f"{pluribus_root}/.pluribus/research/update_state.json"
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class UpdateEvent:
    """An update event for a file."""

    path: str
    update_type: UpdateType
    priority: UpdatePriority = UpdatePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    old_path: Optional[str] = None  # For renames
    content_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "update_type": self.update_type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "old_path": self.old_path,
        }


@dataclass
class FileState:
    """State of a tracked file."""

    path: str
    content_hash: str
    last_indexed: float
    last_modified: float
    symbol_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileState":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class UpdateStats:
    """Statistics on updates."""

    files_updated: int = 0
    files_created: int = 0
    files_deleted: int = 0
    symbols_added: int = 0
    symbols_removed: int = 0
    batches_processed: int = 0
    last_update_time: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Incremental Updater
# ============================================================================


class IncrementalUpdater:
    """
    Update research indexes incrementally as files change.

    Features:
    - Watch filesystem for changes
    - Queue and batch updates
    - Maintain content hashes for change detection
    - Prioritize updates by importance

    PBTSO Phase: ITERATE, MAINTAIN

    Example:
        updater = IncrementalUpdater()
        await updater.start()

        # Or manually process changes
        updater.queue_update(UpdateEvent(path="/src/main.py", update_type=UpdateType.MODIFY))
        await updater.process_pending()
    """

    def __init__(
        self,
        config: Optional[UpdaterConfig] = None,
        bus: Optional[AgentBus] = None,
        on_update: Optional[Callable[[UpdateEvent], Any]] = None,
    ):
        """
        Initialize the incremental updater.

        Args:
            config: Updater configuration
            bus: AgentBus for event emission
            on_update: Callback for processing updates
        """
        self.config = config or UpdaterConfig()
        self.bus = bus or AgentBus()
        self.on_update = on_update

        # State
        self._file_states: Dict[str, FileState] = {}
        self._update_queue: List[UpdateEvent] = []
        self._pending_paths: Set[str] = set()
        self._stats = UpdateStats()

        # Control
        self._running = False
        self._watch_task: Optional[asyncio.Task] = None
        self._process_task: Optional[asyncio.Task] = None

        # Load state
        self._load_state()

    async def start(self) -> None:
        """Start the incremental updater."""
        if self._running:
            return

        self._running = True

        self._emit_with_lock({
            "topic": "a2a.research.update.start",
            "kind": "update",
            "data": {"watch_paths": self.config.watch_paths}
        })

        # Start background tasks
        if self.config.enable_watch:
            self._watch_task = asyncio.create_task(self._watch_loop())

        self._process_task = asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        """Stop the incremental updater."""
        self._running = False

        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        # Save state
        self._save_state()

        self._emit_with_lock({
            "topic": "a2a.research.update.complete",
            "kind": "update",
            "data": self._stats.to_dict()
        })

    def queue_update(self, event: UpdateEvent) -> bool:
        """
        Queue an update for processing.

        Args:
            event: Update event

        Returns:
            True if queued (False if queue full or duplicate)
        """
        if len(self._update_queue) >= self.config.max_queue_size:
            return False

        # Debounce: don't queue if same path is pending
        if event.path in self._pending_paths:
            # Update existing event if higher priority
            for existing in self._update_queue:
                if existing.path == event.path:
                    if event.priority.value < existing.priority.value:
                        existing.priority = event.priority
                    existing.timestamp = event.timestamp
                    return True
            return False

        self._update_queue.append(event)
        self._pending_paths.add(event.path)

        return True

    async def process_pending(self) -> int:
        """
        Process all pending updates.

        Returns:
            Number of updates processed
        """
        if not self._update_queue:
            return 0

        # Sort by priority
        self._update_queue.sort(key=lambda e: (e.priority.value, e.timestamp))

        processed = 0

        while self._update_queue:
            # Take a batch
            batch = self._update_queue[:self.config.batch_size]
            self._update_queue = self._update_queue[self.config.batch_size:]

            # Process batch
            for event in batch:
                self._pending_paths.discard(event.path)

                try:
                    await self._process_event(event)
                    processed += 1
                except Exception as e:
                    self._emit_with_lock({
                        "topic": "research.update.error",
                        "kind": "error",
                        "level": "error",
                        "data": {"path": event.path, "error": str(e)}
                    })

            self._stats.batches_processed += 1

            # Emit batch event
            self._emit_with_lock({
                "topic": "research.update.batch",
                "kind": "update",
                "data": {"batch_size": len(batch), "remaining": len(self._update_queue)}
            })

            # Small delay between batches
            if self._update_queue:
                await asyncio.sleep(self.config.batch_delay_ms / 1000)

        return processed

    def scan_for_changes(self) -> List[UpdateEvent]:
        """
        Scan watched paths for changes since last update.

        Returns:
            List of detected change events
        """
        events = []

        for watch_path in self.config.watch_paths:
            root = Path(watch_path)
            if not root.exists():
                continue

            for ext in self.config.extensions:
                for file_path in root.rglob(f"*{ext}"):
                    # Check ignore patterns
                    if self._should_ignore(file_path):
                        continue

                    path_str = str(file_path)

                    try:
                        stat = file_path.stat()
                        mtime = stat.st_mtime

                        # Check if file changed
                        if path_str in self._file_states:
                            state = self._file_states[path_str]
                            if mtime > state.last_indexed:
                                # Check content hash
                                current_hash = self._hash_file(file_path)
                                if current_hash != state.content_hash:
                                    events.append(UpdateEvent(
                                        path=path_str,
                                        update_type=UpdateType.MODIFY,
                                        content_hash=current_hash,
                                    ))
                        else:
                            # New file
                            events.append(UpdateEvent(
                                path=path_str,
                                update_type=UpdateType.CREATE,
                                content_hash=self._hash_file(file_path),
                            ))

                    except Exception:
                        continue

        # Check for deleted files
        current_paths = set(str(p) for p in events)
        for path_str in list(self._file_states.keys()):
            if not Path(path_str).exists():
                events.append(UpdateEvent(
                    path=path_str,
                    update_type=UpdateType.DELETE,
                ))

        return events

    def get_stats(self) -> UpdateStats:
        """Get update statistics."""
        return self._stats

    def get_file_state(self, path: str) -> Optional[FileState]:
        """Get state for a specific file."""
        return self._file_states.get(path)

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self._update_queue)

    def clear_queue(self) -> int:
        """Clear update queue. Returns number of items cleared."""
        count = len(self._update_queue)
        self._update_queue.clear()
        self._pending_paths.clear()
        return count

    # ========================================================================
    # Internal Methods
    # ========================================================================

    async def _watch_loop(self) -> None:
        """Background loop to watch for file changes."""
        try:
            # Try to use watchdog if available
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class Handler(FileSystemEventHandler):
                def __init__(self, updater: IncrementalUpdater):
                    self.updater = updater

                def on_modified(self, event):
                    if not event.is_directory:
                        self.updater._handle_fs_event(event.src_path, UpdateType.MODIFY)

                def on_created(self, event):
                    if not event.is_directory:
                        self.updater._handle_fs_event(event.src_path, UpdateType.CREATE)

                def on_deleted(self, event):
                    if not event.is_directory:
                        self.updater._handle_fs_event(event.src_path, UpdateType.DELETE)

                def on_moved(self, event):
                    if not event.is_directory:
                        self.updater._handle_fs_event(
                            event.dest_path, UpdateType.RENAME, event.src_path
                        )

            observer = Observer()
            handler = Handler(self)

            for watch_path in self.config.watch_paths:
                if Path(watch_path).exists():
                    observer.schedule(handler, watch_path, recursive=True)

            observer.start()

            while self._running:
                await asyncio.sleep(1)

            observer.stop()
            observer.join()

        except ImportError:
            # Fallback to polling
            while self._running:
                events = self.scan_for_changes()
                for event in events:
                    self.queue_update(event)
                await asyncio.sleep(5)  # Poll every 5 seconds

    async def _process_loop(self) -> None:
        """Background loop to process queued updates."""
        while self._running:
            if self._update_queue:
                await self.process_pending()
            await asyncio.sleep(self.config.batch_delay_ms / 1000)

    def _handle_fs_event(
        self,
        path: str,
        update_type: UpdateType,
        old_path: Optional[str] = None,
    ) -> None:
        """Handle a filesystem event."""
        # Check extension
        ext = Path(path).suffix
        if ext not in self.config.extensions:
            return

        # Check ignore patterns
        if self._should_ignore(Path(path)):
            return

        event = UpdateEvent(
            path=path,
            update_type=update_type,
            old_path=old_path,
        )

        self.queue_update(event)

    async def _process_event(self, event: UpdateEvent) -> None:
        """Process a single update event."""
        self._emit_with_lock({
            "topic": "research.update.file",
            "kind": "update",
            "data": event.to_dict()
        })

        if event.update_type == UpdateType.CREATE:
            self._stats.files_created += 1

            # Update file state
            if event.content_hash:
                self._file_states[event.path] = FileState(
                    path=event.path,
                    content_hash=event.content_hash,
                    last_indexed=time.time(),
                    last_modified=time.time(),
                )

        elif event.update_type == UpdateType.MODIFY:
            self._stats.files_updated += 1

            # Update file state
            if event.content_hash:
                if event.path in self._file_states:
                    self._file_states[event.path].content_hash = event.content_hash
                    self._file_states[event.path].last_indexed = time.time()
                else:
                    self._file_states[event.path] = FileState(
                        path=event.path,
                        content_hash=event.content_hash,
                        last_indexed=time.time(),
                        last_modified=time.time(),
                    )

        elif event.update_type == UpdateType.DELETE:
            self._stats.files_deleted += 1

            # Remove file state
            self._file_states.pop(event.path, None)

        elif event.update_type == UpdateType.RENAME:
            # Remove old, add new
            if event.old_path:
                old_state = self._file_states.pop(event.old_path, None)
                if old_state:
                    old_state.path = event.path
                    self._file_states[event.path] = old_state

        self._stats.last_update_time = time.time()

        # Call external handler
        if self.on_update:
            result = self.on_update(event)
            if asyncio.iscoroutine(result):
                await result

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        path_str = str(path)

        for pattern in self.config.ignore_patterns:
            if "*" in pattern:
                # Glob pattern
                if path.match(pattern):
                    return True
            else:
                # Simple substring match
                if pattern in path_str:
                    return True

        return False

    def _hash_file(self, path: Path) -> str:
        """Compute content hash for a file."""
        try:
            content = path.read_bytes()
            return hashlib.sha256(content).hexdigest()[:16]
        except Exception:
            return ""

    def _load_state(self) -> None:
        """Load state from disk."""
        state_path = Path(self.config.state_path)
        if not state_path.exists():
            return

        try:
            with open(state_path) as f:
                data = json.load(f)

            for path, state_data in data.get("files", {}).items():
                self._file_states[path] = FileState.from_dict(state_data)

            stats_data = data.get("stats", {})
            self._stats = UpdateStats(**stats_data)

        except Exception:
            pass

    def _save_state(self) -> None:
        """Save state to disk."""
        state_path = Path(self.config.state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "files": {path: state.to_dict() for path, state in self._file_states.items()},
            "stats": self._stats.to_dict(),
            "saved_at": time.time(),
        }

        with open(state_path, "w") as f:
            json.dump(data, f, indent=2)

    def _emit_with_lock(self, event: Dict[str, Any]) -> str:
        """Emit event with file locking."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        import socket
        import uuid

        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": event.get("topic", "unknown"),
            "kind": event.get("kind", "event"),
            "level": event.get("level", "info"),
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": event.get("data", {}),
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(full_event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Incremental Updater."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Incremental Updater (Step 27)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan for changes")
    scan_parser.add_argument("--path", default=".", help="Path to scan")
    scan_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Watch command
    watch_parser = subparsers.add_parser("watch", help="Watch for changes")
    watch_parser.add_argument("--path", default=".", help="Path to watch")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show update statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Update command
    update_parser = subparsers.add_parser("update", help="Manually update a file")
    update_parser.add_argument("path", help="File path")
    update_parser.add_argument("--type", choices=[t.value for t in UpdateType],
                              default="modify", help="Update type")

    args = parser.parse_args()

    config = UpdaterConfig(
        watch_paths=[getattr(args, "path", ".")],
    )
    updater = IncrementalUpdater(config)

    if args.command == "scan":
        events = updater.scan_for_changes()
        if args.json:
            print(json.dumps([e.to_dict() for e in events], indent=2))
        else:
            print(f"Found {len(events)} changes:")
            for e in events[:20]:
                print(f"  [{e.update_type.value}] {e.path}")
            if len(events) > 20:
                print(f"  ... and {len(events) - 20} more")

    elif args.command == "watch":
        print(f"Watching {args.path} for changes (Ctrl+C to stop)...")

        async def run():
            await updater.start()
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping...")
            await updater.stop()

        asyncio.run(run())

    elif args.command == "stats":
        stats = updater.get_stats()
        if args.json:
            print(json.dumps(stats.to_dict(), indent=2))
        else:
            print("Update Statistics:")
            print(f"  Files Updated: {stats.files_updated}")
            print(f"  Files Created: {stats.files_created}")
            print(f"  Files Deleted: {stats.files_deleted}")
            print(f"  Batches Processed: {stats.batches_processed}")

    elif args.command == "update":
        event = UpdateEvent(
            path=args.path,
            update_type=UpdateType(args.type),
        )
        updater.queue_update(event)
        asyncio.run(updater.process_pending())
        print(f"Updated: {args.path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
