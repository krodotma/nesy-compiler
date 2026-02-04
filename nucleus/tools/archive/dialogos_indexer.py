#!/usr/bin/env python3
"""
dialogos_indexer.py - Daemon that indexes dialogos trace into IR/KG/Vector stores.

Tails `.pluribus/dialogos/trace.ndjson` for new records and:
- Normalizes events via qa_ir.normalize_event() -> appends to qa_ir.ndjson
- Extracts entities/facts via graphiti_bridge patterns -> appends to KG
- Generates embeddings for chroma_memory (if available)

CLI Interface:
    python3 dialogos_indexer.py daemon --poll 1.0
    python3 dialogos_indexer.py daemon --metrics-port 9090
    python3 dialogos_indexer.py reindex --since 24h
    python3 dialogos_indexer.py status
    python3 dialogos_indexer.py health

Bus Events:
    dialogos.indexer.record - each record indexed
    dialogos.indexer.stats  - periodic stats
    dialogos.indexer.backpressure - queue > 100 records

Effects: R(file), W(file), Bus(emit)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar

sys.dont_write_bytecode = True

# Local imports with fallback
try:
    from .agent_bus import resolve_bus_paths, emit_event, default_actor
    from .qa_ir import normalize_event, QAIrStore
    from .graphiti_bridge import GraphitiService, extract_facts_from_bus_event
except Exception:
    from agent_bus import resolve_bus_paths, emit_event, default_actor
    from qa_ir import normalize_event, QAIrStore
    from graphiti_bridge import GraphitiService, extract_facts_from_bus_event


T = TypeVar("T")


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_duration(duration_str: str) -> float:
    """Parse duration string like '24h', '1d', '30m' to seconds."""
    duration_str = duration_str.strip().lower()
    if not duration_str:
        return 0.0

    multipliers = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
    }

    # Try to parse with suffix
    for suffix, mult in multipliers.items():
        if duration_str.endswith(suffix):
            try:
                val = float(duration_str[:-1])
                return val * mult
            except ValueError:
                pass

    # Try plain number (assume seconds)
    try:
        return float(duration_str)
    except ValueError:
        return 0.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_file(path: Path) -> None:
    ensure_dir(path.parent)
    if not path.exists():
        path.touch()


def iter_ndjson(path: Path) -> Iterator[dict]:
    """Iterate over NDJSON file."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def append_ndjson(path: Path, obj: dict) -> None:
    """Append object to NDJSON file."""
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def compute_content_hash(record: dict) -> str:
    """Compute a content hash for deduplication based on actual content, not ID."""
    # Extract content-significant fields
    data = record.get("data") if isinstance(record.get("data"), dict) else {}
    content_parts = [
        record.get("actor", ""),
        record.get("topic", ""),
        record.get("event_type", ""),
        str(data.get("content", "")),
        str(data.get("text", "")),
        str(data.get("prompt", "")),
        str(data.get("message", "")),
    ]
    content_str = "|".join(content_parts)
    return hashlib.sha256(content_str.encode("utf-8")).hexdigest()[:32]


def retry_with_backoff(
    func: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    on_error: Optional[Callable[[Exception, int], None]] = None,
) -> tuple[bool, Optional[T], Optional[Exception]]:
    """
    Execute function with exponential backoff retry.

    Returns:
        (success, result, last_exception)
    """
    last_exception: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            result = func()
            return (True, result, None)
        except Exception as e:
            last_exception = e
            if on_error:
                on_error(e, attempt)
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s
                time.sleep(delay)
    return (False, None, last_exception)


@dataclass
class IndexerStats:
    """Statistics for the indexer."""
    records_indexed: int = 0
    facts_extracted: int = 0
    embeddings_stored: int = 0
    errors: int = 0
    retries: int = 0
    duplicates_skipped: int = 0
    last_record_ts: float = 0.0
    start_time: float = field(default_factory=time.time)
    queue_high_water: int = 0
    backpressure_events: int = 0

    def to_dict(self) -> dict:
        uptime = time.time() - self.start_time
        return {
            "records_indexed": self.records_indexed,
            "facts_extracted": self.facts_extracted,
            "embeddings_stored": self.embeddings_stored,
            "errors": self.errors,
            "retries": self.retries,
            "duplicates_skipped": self.duplicates_skipped,
            "last_record_ts": self.last_record_ts,
            "last_record_iso": (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.last_record_ts))
                if self.last_record_ts > 0 else None
            ),
            "uptime_s": int(uptime),
            "records_per_minute": round(self.records_indexed / (uptime / 60), 2) if uptime > 0 else 0,
            "queue_high_water": self.queue_high_water,
            "backpressure_events": self.backpressure_events,
        }


class PrometheusMetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""

    indexer: Optional["DialogosIndexer"] = None

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default logging
        pass

    def do_GET(self) -> None:
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()

            if self.indexer:
                metrics = self._format_prometheus_metrics()
                self.wfile.write(metrics.encode("utf-8"))
            else:
                self.wfile.write(b"# No indexer available\n")
        elif self.path == "/health":
            if self.indexer:
                health = self.indexer.get_health()
                status_code = 200 if health.get("status") == "ok" else 503
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(health).encode("utf-8"))
            else:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "error", "reason": "indexer_unavailable"}')
        else:
            self.send_response(404)
            self.end_headers()

    def _format_prometheus_metrics(self) -> str:
        """Format metrics in Prometheus exposition format."""
        if not self.indexer:
            return ""

        stats = self.indexer.stats.to_dict()
        lines = [
            "# HELP dialogos_indexer_records_indexed_total Total records indexed",
            "# TYPE dialogos_indexer_records_indexed_total counter",
            f"dialogos_indexer_records_indexed_total {stats['records_indexed']}",
            "",
            "# HELP dialogos_indexer_facts_extracted_total Total facts extracted",
            "# TYPE dialogos_indexer_facts_extracted_total counter",
            f"dialogos_indexer_facts_extracted_total {stats['facts_extracted']}",
            "",
            "# HELP dialogos_indexer_embeddings_stored_total Total embeddings stored",
            "# TYPE dialogos_indexer_embeddings_stored_total counter",
            f"dialogos_indexer_embeddings_stored_total {stats['embeddings_stored']}",
            "",
            "# HELP dialogos_indexer_errors_total Total errors",
            "# TYPE dialogos_indexer_errors_total counter",
            f"dialogos_indexer_errors_total {stats['errors']}",
            "",
            "# HELP dialogos_indexer_retries_total Total retry attempts",
            "# TYPE dialogos_indexer_retries_total counter",
            f"dialogos_indexer_retries_total {stats['retries']}",
            "",
            "# HELP dialogos_indexer_duplicates_skipped_total Total duplicate records skipped",
            "# TYPE dialogos_indexer_duplicates_skipped_total counter",
            f"dialogos_indexer_duplicates_skipped_total {stats['duplicates_skipped']}",
            "",
            "# HELP dialogos_indexer_uptime_seconds Indexer uptime in seconds",
            "# TYPE dialogos_indexer_uptime_seconds gauge",
            f"dialogos_indexer_uptime_seconds {stats['uptime_s']}",
            "",
            "# HELP dialogos_indexer_last_record_timestamp_seconds Timestamp of last indexed record",
            "# TYPE dialogos_indexer_last_record_timestamp_seconds gauge",
            f"dialogos_indexer_last_record_timestamp_seconds {stats['last_record_ts']}",
            "",
            "# HELP dialogos_indexer_queue_size Current queue size",
            "# TYPE dialogos_indexer_queue_size gauge",
            f"dialogos_indexer_queue_size {len(self.indexer._pending_queue)}",
            "",
            "# HELP dialogos_indexer_queue_high_water Maximum queue size observed",
            "# TYPE dialogos_indexer_queue_high_water gauge",
            f"dialogos_indexer_queue_high_water {stats['queue_high_water']}",
            "",
            "# HELP dialogos_indexer_backpressure_events_total Total backpressure events",
            "# TYPE dialogos_indexer_backpressure_events_total counter",
            f"dialogos_indexer_backpressure_events_total {stats['backpressure_events']}",
            "",
            "# HELP dialogos_indexer_graphiti_available Whether graphiti is available",
            "# TYPE dialogos_indexer_graphiti_available gauge",
            f"dialogos_indexer_graphiti_available {1 if self.indexer.graphiti else 0}",
            "",
            "# HELP dialogos_indexer_chroma_available Whether chroma is available",
            "# TYPE dialogos_indexer_chroma_available gauge",
            f"dialogos_indexer_chroma_available {1 if self.indexer.chroma else 0}",
            "",
        ]
        return "\n".join(lines) + "\n"


class DialogosIndexer:
    """Indexes dialogos trace into IR/KG/Vector stores."""

    # Batch processing constants
    BATCH_SIZE = 50
    BACKPRESSURE_THRESHOLD = 100

    def __init__(
        self,
        *,
        root: Path,
        bus_dir: Optional[Path] = None,
        actor: str = "dialogos-indexer",
        poll_s: float = 1.0,
        stats_interval_s: float = 60.0,
        enable_chroma: bool = True,
        enable_graphiti: bool = True,
        metrics_port: Optional[int] = None,
    ):
        self.root = root
        self.bus_dir = bus_dir or Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
        self.actor = actor
        self.poll_s = poll_s
        self.stats_interval_s = stats_interval_s
        self.enable_chroma = enable_chroma
        self.enable_graphiti = enable_graphiti
        self.metrics_port = metrics_port

        # Paths
        self.trace_path = root / ".pluribus" / "dialogos" / "trace.ndjson"
        self.ir_output_dir = root / ".pluribus" / "index" / "dialogos"
        self.state_path = root / ".pluribus" / "index" / "dialogos" / "indexer_state.json"

        # Components
        self.ir_store = QAIrStore(str(self.ir_output_dir), filename="dialogos_ir.ndjson")
        self.graphiti: Optional[GraphitiService] = None
        self.chroma: Any = None

        # State
        self.stats = IndexerStats()
        self._running = False
        self._last_stats_emit = 0.0
        self._processed_ids: set[str] = set()
        self._content_hashes: set[str] = set()  # For content deduplication
        self._pending_queue: deque[dict] = deque()  # For batch processing
        self._backpressure_warned = False

        # Metrics server
        self._metrics_server: Optional[HTTPServer] = None
        self._metrics_thread: Optional[threading.Thread] = None

    def _init_graphiti(self) -> None:
        """Initialize graphiti service."""
        if self.enable_graphiti and self.graphiti is None:
            try:
                self.graphiti = GraphitiService(self.root)
            except Exception as e:
                self._emit_bus(
                    "dialogos.indexer.error",
                    {"component": "graphiti", "error": str(e)},
                    level="warn",
                )

    def _init_chroma(self) -> None:
        """Initialize chroma memory (optional)."""
        if not self.enable_chroma:
            return
        if self.chroma is not None:
            return

        try:
            # Try to import chroma_memory
            try:
                from .chroma_memory import ChromaMemory
            except Exception:
                from chroma_memory import ChromaMemory

            self.chroma = ChromaMemory()
        except ImportError:
            # ChromaDB not available
            self.chroma = None
        except Exception as e:
            self._emit_bus(
                "dialogos.indexer.error",
                {"component": "chroma", "error": str(e)},
                level="warn",
            )
            self.chroma = None

    def _start_metrics_server(self) -> None:
        """Start the Prometheus metrics HTTP server in a background thread."""
        if self.metrics_port is None:
            return

        try:
            # Set the indexer reference on the handler class
            PrometheusMetricsHandler.indexer = self

            self._metrics_server = HTTPServer(
                ("0.0.0.0", self.metrics_port),
                PrometheusMetricsHandler,
            )
            self._metrics_thread = threading.Thread(
                target=self._metrics_server.serve_forever,
                daemon=True,
                name="metrics-server",
            )
            self._metrics_thread.start()
            print(f"[dialogos-indexer] Metrics server started on port {self.metrics_port}", file=sys.stderr)
        except Exception as e:
            print(f"[dialogos-indexer] Failed to start metrics server: {e}", file=sys.stderr)
            self._emit_bus(
                "dialogos.indexer.error",
                {"component": "metrics_server", "error": str(e)},
                level="warn",
            )

    def _stop_metrics_server(self) -> None:
        """Stop the metrics server."""
        if self._metrics_server:
            self._metrics_server.shutdown()
            self._metrics_server = None

    def _emit_bus(
        self,
        topic: str,
        data: dict,
        *,
        kind: str = "metric",
        level: str = "info",
    ) -> None:
        """Emit event to bus."""
        try:
            paths = resolve_bus_paths(str(self.bus_dir))
            emit_event(
                paths,
                topic=topic,
                kind=kind,
                level=level,
                actor=self.actor,
                data=data,
                trace_id=data.get("trace_id"),
                run_id=data.get("run_id"),
                durable=False,
            )
        except Exception:
            pass  # Best effort

    def _load_state(self) -> dict:
        """Load indexer state."""
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_state(self, state: dict) -> None:
        """Save indexer state."""
        ensure_dir(self.state_path.parent)
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _check_backpressure(self) -> None:
        """Check queue size and emit backpressure warning if needed."""
        queue_size = len(self._pending_queue)

        # Update high water mark
        if queue_size > self.stats.queue_high_water:
            self.stats.queue_high_water = queue_size

        if queue_size > self.BACKPRESSURE_THRESHOLD:
            if not self._backpressure_warned:
                self._backpressure_warned = True
                self.stats.backpressure_events += 1
                print(
                    f"[dialogos-indexer] WARNING: Backpressure - queue size {queue_size} > {self.BACKPRESSURE_THRESHOLD}",
                    file=sys.stderr,
                )
                self._emit_bus(
                    "dialogos.indexer.backpressure",
                    {
                        "queue_size": queue_size,
                        "threshold": self.BACKPRESSURE_THRESHOLD,
                        "records_indexed": self.stats.records_indexed,
                    },
                    level="warn",
                )
        else:
            self._backpressure_warned = False

    def _index_record(self, record: dict) -> bool:
        """Index a single dialogos record.

        Returns True if successfully indexed.
        """
        record_id = record.get("id") or ""
        if not record_id or record_id in self._processed_ids:
            return False

        # Content deduplication check
        content_hash = compute_content_hash(record)
        if content_hash in self._content_hashes:
            self.stats.duplicates_skipped += 1
            self._processed_ids.add(record_id)  # Mark ID as seen even if content is duplicate
            return False

        try:
            # 1. Normalize and store in IR with retry
            def store_ir() -> None:
                ir_event = normalize_event(record, source="dialogos")
                self.ir_store.append(ir_event)

            success, _, error = retry_with_backoff(
                store_ir,
                max_attempts=3,
                base_delay=1.0,
                on_error=lambda e, a: self._on_retry_error("ir_store", e, a),
            )
            if not success:
                raise error or Exception("IR store failed after retries")

            # 2. Extract facts via graphiti patterns with retry
            facts: list[dict] = []
            if self.graphiti is not None:
                facts = extract_facts_from_bus_event(record)
                for fact in facts:
                    def add_fact(f: dict = fact) -> Any:
                        return self.graphiti.add_fact(
                            subject=f["subject"],
                            predicate=f["predicate"],
                            object_value=f["object"],
                            valid_from=f.get("valid_from"),
                            confidence=f.get("confidence", 1.0),
                            source=f.get("source", f"dialogos:{record_id}"),
                            provenance_id=record_id,
                        )

                    success, result, _ = retry_with_backoff(
                        add_fact,
                        max_attempts=3,
                        base_delay=1.0,
                        on_error=lambda e, a: self._on_retry_error("graphiti", e, a),
                    )
                    if success and result and "error" not in result:
                        self.stats.facts_extracted += 1

            # 3. Store in chroma for vector search with retry
            if self.chroma is not None:
                # Extract content for embedding
                data = record.get("data") if isinstance(record.get("data"), dict) else {}
                content = (
                    data.get("content")
                    or data.get("text")
                    or data.get("prompt")
                    or data.get("message")
                    or ""
                )
                if content and len(content) >= 20:
                    def store_chroma() -> None:
                        self.chroma.store(
                            "dialogos_trace",
                            documents=[content[:4000]],
                            ids=[record_id],
                            metadatas=[{
                                "actor": record.get("actor", ""),
                                "iso": record.get("iso", ""),
                                "event_type": record.get("event_type", ""),
                                "session_id": record.get("session_id", ""),
                                "content_hash": content_hash,
                            }],
                        )

                    success, _, _ = retry_with_backoff(
                        store_chroma,
                        max_attempts=3,
                        base_delay=1.0,
                        on_error=lambda e, a: self._on_retry_error("chroma", e, a),
                    )
                    if success:
                        self.stats.embeddings_stored += 1

            # Mark as processed
            self._processed_ids.add(record_id)
            self._content_hashes.add(content_hash)
            self.stats.records_indexed += 1
            self.stats.last_record_ts = record.get("ts", time.time())

            # Emit record indexed event
            self._emit_bus(
                "dialogos.indexer.record",
                {
                    "record_id": record_id,
                    "content_hash": content_hash,
                    "actor": record.get("actor", ""),
                    "event_type": record.get("event_type", ""),
                    "facts_extracted": len(facts) if self.graphiti else 0,
                    "embedded": self.chroma is not None and content and len(content) >= 20,
                },
            )

            return True

        except Exception as e:
            self.stats.errors += 1
            self._emit_bus(
                "dialogos.indexer.error",
                {"record_id": record_id, "error": str(e)},
                level="warn",
            )
            return False

    def _on_retry_error(self, component: str, error: Exception, attempt: int) -> None:
        """Handle retry error logging."""
        self.stats.retries += 1
        if attempt > 0:  # Don't log first attempt
            print(
                f"[dialogos-indexer] Retry {attempt + 1} for {component}: {error}",
                file=sys.stderr,
            )

    def _process_batch(self, records: list[dict]) -> int:
        """Process a batch of records.

        Returns the number of successfully indexed records.
        """
        indexed = 0
        for record in records:
            if self._index_record(record):
                indexed += 1
        return indexed

    def _maybe_emit_stats(self) -> None:
        """Emit stats periodically."""
        now = time.time()
        if now - self._last_stats_emit >= self.stats_interval_s:
            self._last_stats_emit = now
            self._emit_bus("dialogos.indexer.stats", self.stats.to_dict())

    def _tail_trace_file(self) -> Iterator[dict]:
        """Tail the trace file for new records."""
        ensure_file(self.trace_path)

        with self.trace_path.open("r", encoding="utf-8", errors="replace") as f:
            # Seek to end to start tailing
            f.seek(0, os.SEEK_END)
            inode = os.fstat(f.fileno()).st_ino

            while self._running:
                line = f.readline()
                if line:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
                    continue

                # Check for file rotation
                try:
                    stats = os.stat(self.trace_path)
                    if stats.st_ino != inode:
                        # File was rotated, reopen
                        f.close()
                        f = self.trace_path.open("r", encoding="utf-8", errors="replace")
                        inode = os.fstat(f.fileno()).st_ino
                        continue
                    if stats.st_size < f.tell():
                        # File was truncated, seek to start
                        f.seek(0)
                        continue
                except FileNotFoundError:
                    pass

                # Emit stats while waiting
                self._maybe_emit_stats()
                time.sleep(self.poll_s)

    def run_daemon(self) -> int:
        """Run as daemon, tailing trace file."""
        self._running = True
        self._init_graphiti()
        self._init_chroma()
        self._start_metrics_server()

        # Load state
        state = self._load_state()
        self._processed_ids = set(state.get("processed_ids", [])[-10000:])  # Keep last 10k
        self._content_hashes = set(state.get("content_hashes", [])[-10000:])  # Keep last 10k

        # Setup signal handlers
        def handle_shutdown(signum: int, frame: Any) -> None:
            print(f"\n[dialogos-indexer] Received signal {signum}, shutting down...", file=sys.stderr)
            self._running = False

        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)

        print(f"[dialogos-indexer] Starting daemon, tailing {self.trace_path}", file=sys.stderr)
        print(f"[dialogos-indexer] IR output: {self.ir_output_dir}", file=sys.stderr)
        print(f"[dialogos-indexer] Graphiti: {'enabled' if self.graphiti else 'disabled'}", file=sys.stderr)
        print(f"[dialogos-indexer] Chroma: {'enabled' if self.chroma else 'disabled'}", file=sys.stderr)
        print(f"[dialogos-indexer] Batch size: {self.BATCH_SIZE}", file=sys.stderr)
        if self.metrics_port:
            print(f"[dialogos-indexer] Metrics: http://0.0.0.0:{self.metrics_port}/metrics", file=sys.stderr)

        self._emit_bus(
            "dialogos.indexer.started",
            {
                "trace_path": str(self.trace_path),
                "ir_output_dir": str(self.ir_output_dir),
                "graphiti_enabled": self.graphiti is not None,
                "chroma_enabled": self.chroma is not None,
                "batch_size": self.BATCH_SIZE,
                "metrics_port": self.metrics_port,
            },
        )

        try:
            batch: list[dict] = []
            for record in self._tail_trace_file():
                # Add to pending queue for backpressure monitoring
                self._pending_queue.append(record)
                self._check_backpressure()

                # Collect records for batch processing
                batch.append(record)

                if len(batch) >= self.BATCH_SIZE:
                    # Process the batch
                    self._process_batch(batch)
                    # Remove processed items from pending queue
                    for _ in range(len(batch)):
                        if self._pending_queue:
                            self._pending_queue.popleft()
                    batch = []

            # Process any remaining records
            if batch:
                self._process_batch(batch)
                for _ in range(len(batch)):
                    if self._pending_queue:
                        self._pending_queue.popleft()

        except Exception as e:
            print(f"[dialogos-indexer] Error: {e}", file=sys.stderr)
            self._emit_bus(
                "dialogos.indexer.error",
                {"error": str(e)},
                level="error",
            )
            return 1
        finally:
            # Stop metrics server
            self._stop_metrics_server()

            # Save state on shutdown
            self._save_state({
                "processed_ids": list(self._processed_ids)[-10000:],
                "content_hashes": list(self._content_hashes)[-10000:],
                "last_shutdown": now_iso_utc(),
                "stats": self.stats.to_dict(),
            })
            self._emit_bus(
                "dialogos.indexer.stopped",
                self.stats.to_dict(),
            )
            print(f"[dialogos-indexer] Shutdown complete. Stats: {self.stats.to_dict()}", file=sys.stderr)

        return 0

    def reindex(self, since_seconds: float) -> dict:
        """Reindex records from the trace file since a given time."""
        self._init_graphiti()
        self._init_chroma()

        cutoff_ts = time.time() - since_seconds
        reindexed = 0
        skipped = 0
        batch: list[dict] = []

        print(f"[dialogos-indexer] Reindexing records since {since_seconds}s ago...", file=sys.stderr)
        print(f"[dialogos-indexer] Using batch size: {self.BATCH_SIZE}", file=sys.stderr)

        for record in iter_ndjson(self.trace_path):
            record_ts = record.get("ts", 0)
            if record_ts < cutoff_ts:
                skipped += 1
                continue

            batch.append(record)

            if len(batch) >= self.BATCH_SIZE:
                reindexed += self._process_batch(batch)
                batch = []

        # Process remaining batch
        if batch:
            reindexed += self._process_batch(batch)

        result = {
            "reindexed": reindexed,
            "skipped": skipped,
            "since_seconds": since_seconds,
            "facts_extracted": self.stats.facts_extracted,
            "embeddings_stored": self.stats.embeddings_stored,
            "duplicates_skipped": self.stats.duplicates_skipped,
            "errors": self.stats.errors,
            "retries": self.stats.retries,
        }

        self._emit_bus("dialogos.indexer.reindex.complete", result)
        return result

    def get_status(self) -> dict:
        """Get indexer status."""
        state = self._load_state()
        trace_exists = self.trace_path.exists()
        trace_size = self.trace_path.stat().st_size if trace_exists else 0

        # Count records in trace
        record_count = 0
        if trace_exists:
            for _ in iter_ndjson(self.trace_path):
                record_count += 1

        # Count records in IR
        ir_path = self.ir_output_dir / "dialogos_ir.ndjson"
        ir_count = 0
        if ir_path.exists():
            for _ in iter_ndjson(ir_path):
                ir_count += 1

        return {
            "trace_path": str(self.trace_path),
            "trace_exists": trace_exists,
            "trace_size_bytes": trace_size,
            "trace_record_count": record_count,
            "ir_output_dir": str(self.ir_output_dir),
            "ir_record_count": ir_count,
            "state": {
                "processed_count": len(state.get("processed_ids", [])),
                "content_hashes_count": len(state.get("content_hashes", [])),
                "last_shutdown": state.get("last_shutdown"),
                "last_stats": state.get("stats"),
            },
            "chroma_available": self._check_chroma_available(),
            "graphiti_available": self._check_graphiti_available(),
        }

    def get_health(self) -> dict:
        """Get health check status in JSON format."""
        state = self._load_state()
        last_stats = state.get("stats", {})

        # Calculate uptime
        uptime_seconds = int(time.time() - self.stats.start_time)

        # Calculate last record age
        last_record_ts = self.stats.last_record_ts or last_stats.get("last_record_ts", 0)
        if last_record_ts > 0:
            last_record_age_seconds = int(time.time() - last_record_ts)
        else:
            last_record_age_seconds = -1  # No records indexed yet

        # Check component availability
        graphiti_available = self._check_graphiti_available()
        chroma_available = self._check_chroma_available()

        # Determine overall status
        records_indexed = self.stats.records_indexed or last_stats.get("records_indexed", 0)
        error_count = self.stats.errors or last_stats.get("errors", 0)

        if not graphiti_available and not chroma_available:
            status = "degraded"
        elif error_count > 0 and records_indexed == 0:
            status = "error"
        elif error_count > records_indexed * 0.1:  # > 10% error rate
            status = "degraded"
        else:
            status = "ok"

        return {
            "status": status,
            "uptime_seconds": uptime_seconds,
            "records_indexed_total": records_indexed,
            "last_record_age_seconds": last_record_age_seconds,
            "graphiti_available": graphiti_available,
            "chroma_available": chroma_available,
            "errors_total": error_count,
            "retries_total": self.stats.retries or last_stats.get("retries", 0),
            "duplicates_skipped_total": self.stats.duplicates_skipped or last_stats.get("duplicates_skipped", 0),
            "queue_size": len(self._pending_queue),
            "backpressure_events": self.stats.backpressure_events or last_stats.get("backpressure_events", 0),
        }

    def _check_chroma_available(self) -> bool:
        """Check if ChromaDB is available."""
        try:
            import chromadb  # noqa: F401
            return True
        except ImportError:
            return False

    def _check_graphiti_available(self) -> bool:
        """Check if Graphiti service is available."""
        if self.graphiti is not None:
            return True
        # Check if module can be imported
        try:
            from graphiti_bridge import GraphitiService  # noqa: F401
            return True
        except ImportError:
            return False


def cmd_daemon(args: argparse.Namespace) -> int:
    """Run indexer daemon."""
    root = Path(args.root).expanduser().resolve()
    indexer = DialogosIndexer(
        root=root,
        bus_dir=Path(args.bus_dir).expanduser().resolve() if args.bus_dir else None,
        actor=args.actor,
        poll_s=args.poll,
        stats_interval_s=args.stats_interval,
        enable_chroma=not args.no_chroma,
        enable_graphiti=not args.no_graphiti,
        metrics_port=args.metrics_port,
    )
    return indexer.run_daemon()


def cmd_reindex(args: argparse.Namespace) -> int:
    """Reindex from trace file."""
    root = Path(args.root).expanduser().resolve()
    indexer = DialogosIndexer(
        root=root,
        bus_dir=Path(args.bus_dir).expanduser().resolve() if args.bus_dir else None,
        actor=args.actor,
        enable_chroma=not args.no_chroma,
        enable_graphiti=not args.no_graphiti,
    )

    since_seconds = parse_duration(args.since)
    if since_seconds <= 0:
        print(f"Invalid duration: {args.since}", file=sys.stderr)
        return 1

    result = indexer.reindex(since_seconds)
    print(json.dumps(result, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show indexer status."""
    root = Path(args.root).expanduser().resolve()
    indexer = DialogosIndexer(root=root)
    status = indexer.get_status()
    print(json.dumps(status, indent=2))
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Show health check status."""
    root = Path(args.root).expanduser().resolve()
    indexer = DialogosIndexer(root=root)
    health = indexer.get_health()
    print(json.dumps(health, indent=2))
    # Return non-zero exit code if not healthy
    if health.get("status") == "error":
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dialogos_indexer.py",
        description="Indexes dialogos trace into IR/KG/Vector stores.",
    )
    p.add_argument(
        "--root",
        default=os.environ.get("PLURIBUS_ROOT", "/pluribus"),
        help="Pluribus root directory",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # daemon command
    daemon = sub.add_parser("daemon", help="Run as daemon, tailing trace file")
    daemon.add_argument("--bus-dir", default=None, help="Bus directory")
    daemon.add_argument("--actor", default="dialogos-indexer", help="Actor name")
    daemon.add_argument("--poll", type=float, default=_env_float("DIALOGOS_INDEXER_POLL_S", 1.0), help="Poll interval seconds")
    daemon.add_argument("--stats-interval", type=float, default=_env_float("DIALOGOS_INDEXER_STATS_S", 60.0), help="Stats emit interval seconds")
    daemon.add_argument("--no-chroma", action="store_true", help="Disable chroma embedding")
    daemon.add_argument("--no-graphiti", action="store_true", help="Disable graphiti fact extraction")
    daemon.add_argument("--metrics-port", type=int, default=None, help="Port for Prometheus metrics endpoint (e.g., 9090)")
    daemon.set_defaults(func=cmd_daemon)

    # reindex command
    reindex = sub.add_parser("reindex", help="Reindex records from trace file")
    reindex.add_argument("--bus-dir", default=None, help="Bus directory")
    reindex.add_argument("--actor", default="dialogos-indexer", help="Actor name")
    reindex.add_argument("--since", default="24h", help="Reindex records since (e.g., 24h, 1d, 30m)")
    reindex.add_argument("--no-chroma", action="store_true", help="Disable chroma embedding")
    reindex.add_argument("--no-graphiti", action="store_true", help="Disable graphiti fact extraction")
    reindex.set_defaults(func=cmd_reindex)

    # status command
    status = sub.add_parser("status", help="Show indexer status")
    status.set_defaults(func=cmd_status)

    # health command
    health = sub.add_parser("health", help="Show health check status (JSON)")
    health.set_defaults(func=cmd_health)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
