#!/usr/bin/env python3
"""
Bus Consumer Core Library (v3)
==============================

Pluggable, robust, and scalable consumer for the Pluribus event bus.

Features:
- **Pluggable Backends**: File (NDJSON), WebSocket (bus-bridge), Abstract (future Kafka/NATS)
- **Consumer Groups**: Horizontal scaling with partition assignment
- **Persistence**: Tracks read offset and file identity for reliable resume
- **Rotation Awareness**: Detects file rotation (inode change) and handles it seamlessly
- **Metrics**: Prometheus-compatible metrics for observability
- **Backward Compatible**: Drop-in replacement for BusConsumer v2

Usage:
    # File backend (default, backward compatible)
    consumer = BusConsumer(bus_path, state_path=Path("my_state.json"))

    # WebSocket backend
    consumer = BusConsumer.websocket(ws_url="ws://localhost:9200")

    # With consumer group
    consumer = BusConsumer(bus_path, consumer_group="rag-indexer", instance_id="worker-1")

    for event in consumer.consume(tail=True):
        process(event)
        consumer.ack()
"""
from __future__ import annotations

import abc
import json
import os
import sqlite3
import subprocess
import threading
import time
import logging
import asyncio
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Iterator, Any, Generator, Dict, List, Protocol
from contextlib import contextmanager
from datetime import datetime
import hashlib

# Configure logging
logger = logging.getLogger("BusConsumer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


# =============================================================================
# Metrics (Prometheus-compatible)
# =============================================================================

@dataclass
class ConsumerMetrics:
    """Metrics for observability."""
    events_consumed: int = 0
    events_skipped: int = 0
    errors: int = 0
    lag_seconds: float = 0.0
    last_event_ts: float = 0.0
    reconnects: int = 0
    backend_type: str = "unknown"
    consumer_group: str = ""
    instance_id: str = ""

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = [
            f'bus_consumer_events_total{{backend="{self.backend_type}",group="{self.consumer_group}"}} {self.events_consumed}',
            f'bus_consumer_errors_total{{backend="{self.backend_type}",group="{self.consumer_group}"}} {self.errors}',
            f'bus_consumer_lag_seconds{{backend="{self.backend_type}",group="{self.consumer_group}"}} {self.lag_seconds:.3f}',
            f'bus_consumer_reconnects_total{{backend="{self.backend_type}",group="{self.consumer_group}"}} {self.reconnects}',
        ]
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        """Export metrics as dict for bus emission."""
        return {
            "events_consumed": self.events_consumed,
            "events_skipped": self.events_skipped,
            "errors": self.errors,
            "lag_seconds": round(self.lag_seconds, 3),
            "last_event_ts": self.last_event_ts,
            "reconnects": self.reconnects,
            "backend_type": self.backend_type,
            "consumer_group": self.consumer_group,
            "instance_id": self.instance_id,
        }


# =============================================================================
# State Store (SQLite-backed for atomicity)
# =============================================================================

class StateStore:
    """Atomic state storage using SQLite."""

    def __init__(self, state_path: Optional[Path] = None, use_sqlite: bool = True):
        self.state_path = Path(state_path) if state_path else None
        self.use_sqlite = use_sqlite and state_path is not None
        self._conn: Optional[sqlite3.Connection] = None
        self._memory_state: Dict[str, Any] = {}

        if self.use_sqlite and self.state_path:
            self._init_sqlite()

    def _init_sqlite(self):
        """Initialize SQLite state store."""
        db_path = self.state_path.with_suffix('.db') if self.state_path else ':memory:'
        self.state_path.parent.mkdir(parents=True, exist_ok=True) if self.state_path else None

        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS consumer_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        if self.use_sqlite and self._conn:
            row = self._conn.execute(
                "SELECT value FROM consumer_state WHERE key = ?", (key,)
            ).fetchone()
            if row:
                return json.loads(row[0])
        return self._memory_state.get(key, default)

    def set(self, key: str, value: Any):
        """Set state value atomically."""
        if self.use_sqlite and self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO consumer_state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time())
            )
            self._conn.commit()
        self._memory_state[key] = value

    def get_all(self) -> Dict[str, Any]:
        """Get all state values."""
        if self.use_sqlite and self._conn:
            rows = self._conn.execute("SELECT key, value FROM consumer_state").fetchall()
            return {k: json.loads(v) for k, v in rows}
        return self._memory_state.copy()

    def close(self):
        """Close state store."""
        if self._conn:
            self._conn.close()
            self._conn = None


# =============================================================================
# Backend Interface (Abstract Base Class)
# =============================================================================

class BusBackend(abc.ABC):
    """Abstract interface for bus backends."""

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish connection to the backend."""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close connection to the backend."""
        pass

    @abc.abstractmethod
    def consume(
        self,
        tail: bool = False,
        topic_filter: Optional[str] = None,
        poll_interval: float = 0.5
    ) -> Generator[Optional[dict], None, None]:
        """Consume events from the backend."""
        pass

    @abc.abstractmethod
    def publish(self, event: dict) -> None:
        """Publish an event to the backend."""
        pass

    @abc.abstractmethod
    def get_position(self) -> dict:
        """Get current consumption position."""
        pass

    @abc.abstractmethod
    def set_position(self, position: dict) -> None:
        """Set consumption position (for replay/seek)."""
        pass

    @property
    @abc.abstractmethod
    def backend_type(self) -> str:
        """Return backend type identifier."""
        pass


# =============================================================================
# File Backend (NDJSON - Default)
# =============================================================================

class FileBackend(BusBackend):
    """File-based NDJSON backend (default, backward compatible)."""

    def __init__(self, bus_path: Path):
        self.bus_path = Path(bus_path)
        self._file: Optional[Any] = None
        self._last_inode: int = 0
        self._last_offset: int = 0
        self._last_id: Optional[str] = None

    @property
    def backend_type(self) -> str:
        return "file"

    def connect(self) -> None:
        """No-op for file backend."""
        pass

    def disconnect(self) -> None:
        """Close file if open."""
        if self._file:
            self._file.close()
            self._file = None

    def _open_bus(self):
        """Open the bus file, handling rotation logic."""
        if not self.bus_path.exists():
            raise FileNotFoundError(f"Bus file not found: {self.bus_path}")

        f = self.bus_path.open("r", encoding="utf-8", errors="replace")
        fd = f.fileno()
        stats = os.fstat(fd)
        current_inode = stats.st_ino
        current_size = stats.st_size

        # Rotation/Reset Detection
        if self._last_inode != 0 and current_inode != self._last_inode:
            logger.info(f"File rotation detected (inode {self._last_inode} -> {current_inode}). Resetting offset.")
            self._last_offset = 0
        elif self._last_offset > current_size:
            logger.warning(f"File truncated (offset {self._last_offset} > size {current_size}). Resetting offset.")
            self._last_offset = 0

        f.seek(self._last_offset)
        self._last_inode = current_inode
        self._file = f
        return f

    def consume(
        self,
        tail: bool = False,
        topic_filter: Optional[str] = None,
        poll_interval: float = 0.5
    ) -> Generator[Optional[dict], None, None]:
        """Consume events from NDJSON file."""
        try:
            f = self._open_bus()
        except FileNotFoundError:
            if tail:
                logger.info("Bus file not found, waiting...")
                while not self.bus_path.exists():
                    time.sleep(poll_interval)
                    yield None
                f = self._open_bus()
            else:
                return

        while True:
            line = f.readline()
            if not line:
                if not tail:
                    break

                current_pos = f.tell()

                # Check for rotation
                try:
                    stats = self.bus_path.stat()
                    if stats.st_ino != self._last_inode:
                        logger.info("Rotation detected during tail. Reopening.")
                        f.close()
                        f = self._open_bus()
                        continue
                except FileNotFoundError:
                    pass

                time.sleep(poll_interval)
                f.seek(current_pos)
                yield None
                continue

            self._last_offset = f.tell()

            # Fast string filter
            if topic_filter and topic_filter not in line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            self._last_id = event.get("id")
            yield event

    def publish(self, event: dict) -> None:
        """Append event to NDJSON file."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        with self.bus_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def get_position(self) -> dict:
        """Get current file position."""
        return {
            "inode": self._last_inode,
            "offset": self._last_offset,
            "last_id": self._last_id,
        }

    def set_position(self, position: dict) -> None:
        """Set file position."""
        self._last_inode = position.get("inode", 0)
        self._last_offset = position.get("offset", 0)
        self._last_id = position.get("last_id")


# =============================================================================
# WebSocket Backend (bus-bridge)
# =============================================================================

class WebSocketBackend(BusBackend):
    """WebSocket backend connecting to bus-bridge."""

    def __init__(
        self,
        ws_url: str = "ws://localhost:9200",
        reconnect_interval: float = 5.0,
        subscriptions: Optional[List[str]] = None
    ):
        self.ws_url = ws_url
        self.reconnect_interval = reconnect_interval
        self.subscriptions = subscriptions or ["*"]

        self._ws: Optional[Any] = None
        self._event_queue: queue.Queue = queue.Queue(maxsize=10000)
        self._connected = threading.Event()
        self._stop_flag = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        self._last_event_id: Optional[str] = None
        self._reconnect_count = 0

    @property
    def backend_type(self) -> str:
        return "websocket"

    def connect(self) -> None:
        """Connect to bus-bridge WebSocket."""
        try:
            import websocket
        except ImportError:
            raise ImportError("websocket-client required for WebSocket backend: pip install websocket-client")

        def on_message(ws, message):
            try:
                msg = json.loads(message)
                if msg.get("type") == "event":
                    event = msg.get("event")
                    if event:
                        try:
                            self._event_queue.put_nowait(event)
                        except queue.Full:
                            logger.warning("Event queue full, dropping event")
                elif msg.get("type") == "sync":
                    # Initial sync - queue all historical events
                    for event in msg.get("events", []):
                        try:
                            self._event_queue.put_nowait(event)
                        except queue.Full:
                            break
            except json.JSONDecodeError:
                pass

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.info(f"WebSocket closed: {close_status_code} {close_msg}")
            self._connected.clear()

        def on_open(ws):
            logger.info(f"WebSocket connected to {self.ws_url}")
            self._connected.set()

            # Subscribe to topics
            for topic in self.subscriptions:
                ws.send(json.dumps({"type": "subscribe", "topic": topic}))

            # Request initial sync
            ws.send(json.dumps({"type": "sync"}))

        self._ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # Start reader thread
        self._stop_flag.clear()
        self._reader_thread = threading.Thread(target=self._run_ws, daemon=True)
        self._reader_thread.start()

        # Wait for connection
        if not self._connected.wait(timeout=10.0):
            raise ConnectionError(f"Failed to connect to {self.ws_url}")

    def _run_ws(self):
        """WebSocket event loop with reconnection."""
        while not self._stop_flag.is_set():
            try:
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                logger.error(f"WebSocket run error: {e}")

            if self._stop_flag.is_set():
                break

            self._reconnect_count += 1
            logger.info(f"Reconnecting in {self.reconnect_interval}s (attempt {self._reconnect_count})")
            time.sleep(self.reconnect_interval)

    def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        self._stop_flag.set()
        if self._ws:
            self._ws.close()
        if self._reader_thread:
            self._reader_thread.join(timeout=5.0)

    def consume(
        self,
        tail: bool = False,
        topic_filter: Optional[str] = None,
        poll_interval: float = 0.5
    ) -> Generator[Optional[dict], None, None]:
        """Consume events from WebSocket."""
        while True:
            try:
                event = self._event_queue.get(timeout=poll_interval)

                # Apply topic filter
                if topic_filter and topic_filter not in event.get("topic", ""):
                    continue

                self._last_event_id = event.get("id")
                yield event

            except queue.Empty:
                if not tail:
                    break
                yield None

    def publish(self, event: dict) -> None:
        """Publish event via WebSocket."""
        if self._ws and self._connected.is_set():
            self._ws.send(json.dumps({"type": "publish", "event": event}))
        else:
            raise ConnectionError("WebSocket not connected")

    def get_position(self) -> dict:
        """Get current position (last event ID for WebSocket)."""
        return {
            "last_event_id": self._last_event_id,
            "reconnect_count": self._reconnect_count,
        }

    def set_position(self, position: dict) -> None:
        """Set position (limited for WebSocket - mainly for tracking)."""
        self._last_event_id = position.get("last_event_id")


# =============================================================================
# Consumer Group Support
# =============================================================================

@dataclass
class ConsumerGroupConfig:
    """Configuration for consumer groups."""
    group_id: str
    instance_id: str
    total_partitions: int = 1
    assigned_partitions: List[int] = field(default_factory=list)

    def partition_for_topic(self, topic: str) -> int:
        """Determine partition for a topic (consistent hashing)."""
        hash_val = int(hashlib.md5(topic.encode()).hexdigest(), 16)
        return hash_val % self.total_partitions

    def should_process(self, topic: str) -> bool:
        """Check if this instance should process the topic."""
        if not self.assigned_partitions:
            return True  # No partitioning, process all
        partition = self.partition_for_topic(topic)
        return partition in self.assigned_partitions


# =============================================================================
# Main Consumer Class (v3)
# =============================================================================

class BusConsumer:
    """Unified consumer for the Pluribus event bus (v3)."""

    def __init__(
        self,
        bus_path: Optional[Path] = None,
        state_path: Optional[Path] = None,
        backend: Optional[BusBackend] = None,
        consumer_group: Optional[str] = None,
        instance_id: Optional[str] = None,
        use_sqlite_state: bool = True
    ):
        """
        Initialize bus consumer.

        Args:
            bus_path: Path to bus file (for FileBackend)
            state_path: Path to state file/db
            backend: Custom backend instance (overrides bus_path)
            consumer_group: Consumer group ID for horizontal scaling
            instance_id: Instance ID within consumer group
            use_sqlite_state: Use SQLite for atomic state storage
        """
        # Backend selection
        if backend:
            self._backend = backend
        elif bus_path:
            self._backend = FileBackend(Path(bus_path))
        else:
            # Default to standard bus path
            default_path = Path("/pluribus/.pluribus/bus/events.ndjson")
            self._backend = FileBackend(default_path)

        # State management
        self._state = StateStore(state_path, use_sqlite=use_sqlite_state)

        # Consumer group
        self._group_config: Optional[ConsumerGroupConfig] = None
        if consumer_group:
            self._group_config = ConsumerGroupConfig(
                group_id=consumer_group,
                instance_id=instance_id or f"instance-{os.getpid()}",
            )

        # Metrics
        self.metrics = ConsumerMetrics(
            backend_type=self._backend.backend_type,
            consumer_group=consumer_group or "",
            instance_id=instance_id or "",
        )

        # Load persisted state
        self._load_state()

    @classmethod
    def websocket(
        cls,
        ws_url: str = "ws://localhost:9200",
        state_path: Optional[Path] = None,
        subscriptions: Optional[List[str]] = None,
        consumer_group: Optional[str] = None,
        instance_id: Optional[str] = None,
    ) -> "BusConsumer":
        """Create a WebSocket-backed consumer."""
        backend = WebSocketBackend(ws_url=ws_url, subscriptions=subscriptions)
        return cls(
            backend=backend,
            state_path=state_path,
            consumer_group=consumer_group,
            instance_id=instance_id,
        )

    @classmethod
    def file(
        cls,
        bus_path: Path,
        state_path: Optional[Path] = None,
        consumer_group: Optional[str] = None,
        instance_id: Optional[str] = None,
    ) -> "BusConsumer":
        """Create a file-backed consumer (explicit)."""
        return cls(
            bus_path=bus_path,
            state_path=state_path,
            consumer_group=consumer_group,
            instance_id=instance_id,
        )

    def _load_state(self):
        """Load persisted state into backend."""
        position = self._state.get("position", {})
        if position:
            self._backend.set_position(position)
            logger.debug(f"Loaded state: {position}")

    def _save_state(self):
        """Persist current state."""
        position = self._backend.get_position()
        position["ts"] = time.time()
        self._state.set("position", position)

    def connect(self) -> "BusConsumer":
        """Connect to backend (for backends that require connection)."""
        self._backend.connect()
        return self

    def disconnect(self) -> None:
        """Disconnect from backend."""
        self._backend.disconnect()
        self._state.close()

    def __enter__(self) -> "BusConsumer":
        """Context manager entry."""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def ack(self) -> None:
        """Explicitly acknowledge processing up to the current point."""
        self._save_state()

    def consume(
        self,
        tail: bool = False,
        topic_filter: Optional[str] = None,
        poll_interval: float = 0.5
    ) -> Generator[Optional[dict], None, None]:
        """
        Yields events from the bus.

        Args:
            tail: If True, keep listening for new events indefinitely.
            topic_filter: Optional string for pre-filtering (optimization).
            poll_interval: Seconds to wait when tailing and no data available.

        Yields:
            dict: Event data.
            None: If tailing and no data is available (heartbeat for maintenance tasks).
        """
        for event in self._backend.consume(
            tail=tail,
            topic_filter=topic_filter,
            poll_interval=poll_interval
        ):
            if event is None:
                yield None
                continue

            # Consumer group filtering
            if self._group_config:
                topic = event.get("topic", "")
                if not self._group_config.should_process(topic):
                    self.metrics.events_skipped += 1
                    continue

            # Update metrics
            self.metrics.events_consumed += 1
            event_ts = event.get("ts", 0)
            if event_ts:
                self.metrics.last_event_ts = event_ts
                self.metrics.lag_seconds = time.time() - (event_ts / 1000 if event_ts > 1e12 else event_ts)

            yield event

    def publish(self, event: dict) -> None:
        """Publish an event to the bus."""
        # Ensure required fields
        if "id" not in event:
            event["id"] = f"{int(time.time() * 1000)}-{os.urandom(4).hex()}"
        if "ts" not in event:
            event["ts"] = time.time()
        if "iso" not in event:
            event["iso"] = datetime.utcnow().isoformat() + "Z"

        self._backend.publish(event)

    def grep_history(self, pattern: str, limit: int = 1000) -> Generator[dict, None, None]:
        """
        Efficiently grep historical events using system `grep`.
        Only works with FileBackend. Does NOT update state/offset.
        """
        if not isinstance(self._backend, FileBackend):
            raise NotImplementedError("grep_history only supported for FileBackend")

        cmd = ["grep", "-E", pattern, str(self._backend.bus_path)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
        count = 0
        try:
            if proc.stdout:
                for line in proc.stdout:
                    try:
                        yield json.loads(line)
                        count += 1
                        if count >= limit:
                            break
                    except json.JSONDecodeError:
                        continue
        finally:
            proc.terminate()

    def get_metrics(self) -> ConsumerMetrics:
        """Get current metrics."""
        return self.metrics

    def emit_metrics(self, topic: str = "bus.consumer.metrics") -> None:
        """Emit metrics as a bus event."""
        self.publish({
            "topic": topic,
            "kind": "metrics",
            "level": "info",
            "actor": f"bus-consumer-{self._group_config.instance_id if self._group_config else 'default'}",
            "data": self.metrics.to_dict(),
        })


# =============================================================================
# Backward Compatibility - Legacy API
# =============================================================================

# Keep load_state and save_state methods for backward compatibility
def _legacy_load_state(self):
    """Legacy load_state for backward compatibility."""
    self._load_state()

def _legacy_save_state(self):
    """Legacy save_state for backward compatibility."""
    self._save_state()

BusConsumer.load_state = _legacy_load_state
BusConsumer.save_state = _legacy_save_state


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bus Consumer v3")
    parser.add_argument("--bus-path", type=Path, default=Path("/pluribus/.pluribus/bus/events.ndjson"))
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--backend", choices=["file", "websocket"], default="file")
    parser.add_argument("--ws-url", default="ws://localhost:9200")
    parser.add_argument("--topic-filter", default=None)
    parser.add_argument("--consumer-group", default=None)
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--tail", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--metrics", action="store_true", help="Print metrics periodically")

    args = parser.parse_args()

    # Create consumer
    if args.backend == "websocket":
        consumer = BusConsumer.websocket(
            ws_url=args.ws_url,
            state_path=args.state_path,
            consumer_group=args.consumer_group,
            instance_id=args.instance_id,
        )
    else:
        consumer = BusConsumer(
            bus_path=args.bus_path,
            state_path=args.state_path,
            consumer_group=args.consumer_group,
            instance_id=args.instance_id,
        )

    count = 0
    last_metrics_time = time.time()

    try:
        with consumer:
            for event in consumer.consume(tail=args.tail, topic_filter=args.topic_filter):
                if event:
                    print(json.dumps(event))
                    count += 1
                    consumer.ack()

                    if args.limit and count >= args.limit:
                        break

                # Periodic metrics
                if args.metrics and time.time() - last_metrics_time > 10:
                    print(f"[METRICS] {consumer.metrics.to_dict()}", file=__import__('sys').stderr)
                    last_metrics_time = time.time()

    except KeyboardInterrupt:
        print(f"\n[INFO] Consumed {count} events", file=__import__('sys').stderr)
        print(f"[METRICS] {consumer.metrics.to_dict()}", file=__import__('sys').stderr)
