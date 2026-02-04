#!/usr/bin/env python3
"""
OMEGA HEARTBEAT - Meta-Observer for Pluribus Liveness (v2 Elite Edition)
=========================================================================

A sophisticated ω-automaton implementation for continuous system health
monitoring. Implements Büchi acceptance conditions for liveness properties
and emits rich telemetry to the Pluribus bus.

Architecture:
- Incremental file tailing with position tracking (no full re-reads)
- Direct NDJSON append (no subprocess overhead)
- Sliding window metrics with O(1) updates
- Shannon entropy for topic/actor distribution health
- Self-monitoring with cycle time tracking
- Graceful signal handling with state persistence
- Anomaly detection via statistical deviation

Bus Topics Emitted:
- omega.heartbeat          - Core liveness signal (every cycle)
- omega.metrics.velocity   - Events/second with sliding window
- omega.metrics.latency    - Request/response latency percentiles
- omega.metrics.entropy    - Topic/actor distribution entropy
- omega.metrics.topology   - Actor-topic relationship graph
- omega.pending.pairs      - ω-automaton backlog (Büchi non-acceptance)
- omega.providers.health   - Provider availability and latency
- omega.anomaly.detected   - Statistical anomalies found
- omega.health.composite   - Aggregate health score [0.0, 1.0]
- omega.meta.self          - Self-monitoring metrics

Reference: nucleus/specs/OMEGA_WORKER_PROTOCOL.md
"""
from __future__ import annotations

import atexit
import collections
import fcntl
import hashlib
import json
import math
import os
import signal
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Iterator, Optional

sys.dont_write_bytecode = True

# =============================================================================
# Constants
# =============================================================================

VERSION = "2.0.0"
PROTOCOL_VERSION = "omega-v2"

# Default configuration
DEFAULT_INTERVAL_S = 60.0
DEFAULT_BUS_DIR = "/pluribus/.pluribus/bus"
DEFAULT_WINDOW_SIZE = 100  # Events for sliding window
DEFAULT_LATENCY_SAMPLES = 50  # Samples for percentile calculation

# Health score weights
WEIGHT_VELOCITY = 0.20
WEIGHT_ERROR_RATE = 0.25
WEIGHT_LATENCY = 0.15
WEIGHT_PENDING = 0.20
WEIGHT_ENTROPY = 0.10
WEIGHT_PROVIDERS = 0.10

# Anomaly detection thresholds (sigma)
ANOMALY_SIGMA_THRESHOLD = 2.5

# Provider check timeout
PROVIDER_CHECK_TIMEOUT_S = 2.0


# =============================================================================
# Time Utilities
# =============================================================================

def now_ts() -> float:
    """Current Unix timestamp."""
    return time.time()


def now_iso() -> str:
    """Current UTC time in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}μs"
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# =============================================================================
# Statistical Utilities
# =============================================================================

def shannon_entropy(counts: dict[str, int]) -> float:
    """
    Calculate Shannon entropy of a distribution.

    H(X) = -Σ p(x) * log2(p(x))

    Returns entropy in bits. Higher = more uniform distribution.
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def normalized_entropy(counts: dict[str, int]) -> float:
    """
    Calculate normalized Shannon entropy [0.0, 1.0].

    1.0 = perfectly uniform distribution
    0.0 = single category dominates
    """
    n = len(counts)
    if n <= 1:
        return 1.0

    h = shannon_entropy(counts)
    h_max = math.log2(n)
    return h / h_max if h_max > 0 else 1.0


def percentile(values: list[float], p: float) -> float:
    """Calculate p-th percentile of sorted values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * p / 100
    lower = int(idx)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = idx - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


def mean(values: list[float]) -> float:
    """Calculate mean of values."""
    return sum(values) / len(values) if values else 0.0


def stddev(values: list[float]) -> float:
    """Calculate standard deviation."""
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


# =============================================================================
# Sliding Window
# =============================================================================

@dataclass
class SlidingWindow:
    """
    O(1) sliding window for streaming statistics.

    Maintains running sum and count for efficient mean calculation.
    Uses deque for automatic oldest-element eviction.
    """
    max_size: int
    _values: Deque[float] = field(default_factory=collections.deque)
    _sum: float = 0.0

    def push(self, value: float) -> None:
        """Add value to window."""
        if len(self._values) >= self.max_size:
            evicted = self._values.popleft()
            self._sum -= evicted
        self._values.append(value)
        self._sum += value

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def mean(self) -> float:
        return self._sum / self.count if self.count > 0 else 0.0

    @property
    def sum(self) -> float:
        return self._sum

    def values(self) -> list[float]:
        return list(self._values)

    def percentile(self, p: float) -> float:
        return percentile(self.values(), p)

    def stddev(self) -> float:
        return stddev(self.values())


# =============================================================================
# Incremental File Tailer
# =============================================================================

@dataclass
class FileTailer:
    """
    Efficient incremental file tailer with position tracking.

    Only reads new data since last poll, avoiding O(n) full file scans.
    Handles file truncation (log rotation) gracefully.
    """
    path: Path
    _position: int = 0
    _inode: int = 0

    def __post_init__(self):
        self._update_inode()

    def _update_inode(self) -> None:
        """Track inode to detect file replacement."""
        try:
            stat = self.path.stat()
            self._inode = stat.st_ino
        except FileNotFoundError:
            self._inode = 0

    def _check_rotation(self) -> bool:
        """Check if file was rotated/replaced."""
        try:
            stat = self.path.stat()
            if stat.st_ino != self._inode:
                self._position = 0
                self._inode = stat.st_ino
                return True
            if stat.st_size < self._position:
                # File was truncated
                self._position = 0
                return True
        except FileNotFoundError:
            self._position = 0
            self._inode = 0
        return False

    def read_new_lines(self) -> Iterator[str]:
        """
        Yield new lines since last read.

        Uses seek() to jump to last position - O(new_bytes) not O(file_size).
        """
        if not self.path.exists():
            return

        self._check_rotation()

        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(self._position)

                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line.endswith("\n"):
                        yield line.rstrip("\n")
                    else:
                        # Incomplete line - don't advance position
                        break

                self._position = f.tell()
        except Exception:
            pass

    def read_new_events(self) -> Iterator[dict]:
        """Yield new parsed NDJSON events."""
        for line in self.read_new_lines():
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError:
                continue


# =============================================================================
# Direct Bus Emission
# =============================================================================

class BusEmitter:
    """
    Direct NDJSON bus emission with file locking.

    No subprocess overhead - direct append with atomic locking.
    """

    def __init__(self, bus_dir: str | Path, actor: str = "omega"):
        self.bus_dir = Path(bus_dir)
        self.events_path = self.bus_dir / "events.ndjson"
        self.actor = actor
        self.host = socket.gethostname()
        self.pid = os.getpid()
        self._emit_count = 0

    def _ensure_dir(self) -> None:
        self.bus_dir.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        topic: str,
        kind: str,
        level: str,
        data: dict,
        *,
        trace_id: Optional[str] = None,
    ) -> str:
        """
        Emit event to bus with atomic file locking.

        Returns event ID.
        """
        self._ensure_dir()

        event_id = str(uuid.uuid4())
        ts = now_ts()

        event = {
            "id": event_id,
            "ts": ts,
            "iso": now_iso(),
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": self.actor,
            "host": self.host,
            "pid": self.pid,
            "data": data,
        }

        if trace_id:
            event["trace_id"] = trace_id

        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"

        try:
            with self.events_path.open("a", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            self._emit_count += 1
        except Exception as e:
            # Best-effort - don't crash on emit failure
            print(f"[omega] emit error: {e}", file=sys.stderr)

        return event_id

    @property
    def emit_count(self) -> int:
        return self._emit_count


# =============================================================================
# Incremental State Tracker
# =============================================================================

@dataclass
class OmegaState:
    """
    Incremental state machine for ω-automaton monitoring.

    Maintains pending request/response pairs, velocity metrics,
    and topic/actor distributions without full file re-reads.
    """
    # Pending pairs: (topic, req_id) -> request_ts
    pending_pairs: dict[tuple[str, str], float] = field(default_factory=dict)

    # Metrics
    event_count: int = 0
    error_count: int = 0

    # Topic/actor distributions for entropy
    topic_counts: dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))
    actor_counts: dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))
    level_counts: dict[str, int] = field(default_factory=lambda: collections.defaultdict(int))

    # Latency tracking (request->response pairs)
    latencies: SlidingWindow = field(default_factory=lambda: SlidingWindow(DEFAULT_LATENCY_SAMPLES))

    # Velocity tracking
    velocity_window: SlidingWindow = field(default_factory=lambda: SlidingWindow(DEFAULT_WINDOW_SIZE))
    last_velocity_ts: float = 0.0

    # Request timestamps for latency calculation
    request_timestamps: dict[str, float] = field(default_factory=dict)

    # Anomaly detection history
    velocity_history: SlidingWindow = field(default_factory=lambda: SlidingWindow(30))
    error_rate_history: SlidingWindow = field(default_factory=lambda: SlidingWindow(30))

    def process_event(self, event: dict) -> None:
        """
        Process a single event incrementally.

        O(1) operation - no file re-reads.
        """
        self.event_count += 1

        topic = event.get("topic", "")
        actor = event.get("actor", "")
        kind = event.get("kind", "")
        level = event.get("level", "")
        data = event.get("data", {})

        # Coerce timestamp to float
        raw_ts = event.get("ts")
        try:
            ts = float(raw_ts) if raw_ts is not None else now_ts()
        except (TypeError, ValueError):
            ts = now_ts()

        # Track distributions
        if topic:
            self.topic_counts[topic] += 1
        if actor:
            self.actor_counts[actor] += 1
        if level:
            self.level_counts[level] += 1

        # Track errors
        if level == "error":
            self.error_count += 1

        # Track request/response pairs for latency and pending detection
        req_id = self._extract_req_id(data)

        if kind == "request" and req_id:
            key = (topic, req_id)
            self.pending_pairs[key] = ts
            self.request_timestamps[req_id] = ts

        elif kind == "response" and req_id:
            # Remove from pending
            key = (topic, req_id)
            self.pending_pairs.pop(key, None)

            # Also check for topic-agnostic match
            for pending_key in list(self.pending_pairs.keys()):
                if pending_key[1] == req_id:
                    self.pending_pairs.pop(pending_key, None)
                    break

            # Calculate latency if we have the request timestamp
            if req_id in self.request_timestamps:
                req_ts = self.request_timestamps.pop(req_id)
                latency_s = ts - req_ts
                if 0 < latency_s < 3600:  # Sanity check: < 1 hour
                    self.latencies.push(latency_s)

        # Update velocity window
        self.velocity_window.push(ts)

    def _extract_req_id(self, data: object) -> Optional[str]:
        """Extract request ID from event data."""
        if not isinstance(data, dict):
            return None
        for key in ("req_id", "request_id", "id", "correlation_id"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None

    def compute_velocity(self) -> float:
        """Compute events per second from velocity window."""
        values = self.velocity_window.values()
        if len(values) < 2:
            return 0.0
        duration = values[-1] - values[0]
        if duration <= 0:
            return 0.0
        return len(values) / duration

    def compute_error_rate(self) -> float:
        """Compute error rate."""
        if self.event_count == 0:
            return 0.0
        return self.error_count / self.event_count

    def compute_topic_entropy(self) -> float:
        """Compute normalized topic distribution entropy."""
        return normalized_entropy(dict(self.topic_counts))

    def compute_actor_entropy(self) -> float:
        """Compute normalized actor distribution entropy."""
        return normalized_entropy(dict(self.actor_counts))

    def get_pending_summary(self) -> dict:
        """Get pending pairs summary with ages."""
        now = now_ts()
        by_topic: dict[str, int] = collections.defaultdict(int)
        ages: list[float] = []

        for (topic, _), req_ts in self.pending_pairs.items():
            by_topic[topic] += 1
            ages.append(now - req_ts)

        return {
            "total": len(self.pending_pairs),
            "by_topic": dict(by_topic),
            "oldest_age_s": max(ages) if ages else 0.0,
            "mean_age_s": mean(ages) if ages else 0.0,
        }

    def get_latency_summary(self) -> dict:
        """Get latency percentile summary."""
        if self.latencies.count == 0:
            return {"p50_s": 0, "p95_s": 0, "p99_s": 0, "mean_s": 0}

        return {
            "p50_s": self.latencies.percentile(50),
            "p95_s": self.latencies.percentile(95),
            "p99_s": self.latencies.percentile(99),
            "mean_s": self.latencies.mean,
            "samples": self.latencies.count,
        }

    def detect_anomalies(self) -> list[dict]:
        """
        Detect statistical anomalies via sigma deviation.

        Returns list of anomaly descriptors.
        """
        anomalies = []

        # Velocity anomaly
        velocity = self.compute_velocity()
        self.velocity_history.push(velocity)

        if self.velocity_history.count >= 10:
            v_mean = self.velocity_history.mean
            v_std = self.velocity_history.stddev()
            if v_std > 0:
                v_sigma = abs(velocity - v_mean) / v_std
                if v_sigma > ANOMALY_SIGMA_THRESHOLD:
                    anomalies.append({
                        "type": "velocity_deviation",
                        "severity": "warn" if v_sigma < 3.5 else "error",
                        "current": velocity,
                        "mean": v_mean,
                        "sigma": v_sigma,
                        "direction": "drop" if velocity < v_mean else "spike",
                    })

        # Error rate anomaly
        error_rate = self.compute_error_rate()
        self.error_rate_history.push(error_rate)

        if error_rate > 0.1:  # >10% error rate
            anomalies.append({
                "type": "high_error_rate",
                "severity": "error" if error_rate > 0.25 else "warn",
                "current": error_rate,
                "threshold": 0.1,
            })

        # Pending backlog anomaly
        pending = len(self.pending_pairs)
        if pending > 50:
            anomalies.append({
                "type": "pending_backlog",
                "severity": "warn" if pending < 100 else "error",
                "pending_count": pending,
                "threshold": 50,
            })

        return anomalies


# =============================================================================
# Provider Health Checker
# =============================================================================

class ProviderHealthChecker:
    """
    Provider availability and health checking.

    Checks for CLI availability and optionally tests connectivity.
    """

    PROVIDERS = [
        ("claude", "claude", ["--version"]),
        ("codex", "codex", ["--version"]),
        ("gemini", "gemini", ["--version"]),
        ("ollama", "ollama", ["list"]),
        ("aider", "aider", ["--version"]),
    ]

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._last_check: float = 0
        self._cache_ttl: float = 60.0  # 1 minute cache

    def check_all(self, force: bool = False) -> dict:
        """
        Check all providers.

        Uses cached results if within TTL.
        """
        now = now_ts()
        if not force and (now - self._last_check) < self._cache_ttl:
            return self._cache

        import shutil
        import subprocess

        results = {}

        for name, binary, test_args in self.PROVIDERS:
            path = shutil.which(binary)
            available = path is not None
            latency_ms = None
            healthy = False

            if available and test_args:
                try:
                    start = time.perf_counter()
                    result = subprocess.run(
                        [path] + test_args,
                        capture_output=True,
                        timeout=PROVIDER_CHECK_TIMEOUT_S,
                    )
                    latency_ms = (time.perf_counter() - start) * 1000
                    healthy = result.returncode == 0
                except Exception:
                    healthy = False

            results[name] = {
                "available": available,
                "healthy": healthy,
                "path": path,
                "latency_ms": latency_ms,
            }

        self._cache = results
        self._last_check = now
        return results

    def compute_score(self) -> float:
        """Compute provider health score [0.0, 1.0]."""
        results = self.check_all()
        if not results:
            return 0.0

        healthy_count = sum(1 for r in results.values() if r.get("healthy"))
        return healthy_count / len(results)


# =============================================================================
# Composite Health Score
# =============================================================================

def compute_health_score(
    velocity: float,
    error_rate: float,
    latency_p95: float,
    pending_count: int,
    topic_entropy: float,
    provider_score: float,
) -> tuple[float, dict]:
    """
    Compute composite health score [0.0, 1.0].

    Uses weighted combination of individual health factors.
    Returns (score, breakdown).
    """
    breakdown = {}

    # Velocity score (0 = dead, 1 = healthy)
    # Assume healthy velocity is 0.1 - 10 events/sec
    if velocity <= 0:
        v_score = 0.0
    elif velocity < 0.01:
        v_score = velocity / 0.01 * 0.5  # Scale up to 0.5
    elif velocity > 100:
        v_score = max(0.5, 1.0 - (velocity - 100) / 900)  # Penalize extreme velocity
    else:
        v_score = 1.0
    breakdown["velocity"] = v_score

    # Error rate score (0% = 1.0, 50%+ = 0.0)
    e_score = max(0.0, 1.0 - error_rate * 2)
    breakdown["error_rate"] = e_score

    # Latency score (< 1s = 1.0, > 10s = 0.0)
    if latency_p95 <= 0:
        l_score = 1.0  # No data = assume OK
    elif latency_p95 < 1.0:
        l_score = 1.0
    elif latency_p95 > 10.0:
        l_score = 0.0
    else:
        l_score = 1.0 - (latency_p95 - 1.0) / 9.0
    breakdown["latency"] = l_score

    # Pending score (0 = 1.0, 100+ = 0.0)
    if pending_count <= 0:
        p_score = 1.0
    elif pending_count >= 100:
        p_score = 0.0
    else:
        p_score = 1.0 - pending_count / 100
    breakdown["pending"] = p_score

    # Entropy score (already normalized 0-1)
    breakdown["entropy"] = topic_entropy

    # Provider score (already 0-1)
    breakdown["providers"] = provider_score

    # Weighted combination
    score = (
        WEIGHT_VELOCITY * breakdown["velocity"] +
        WEIGHT_ERROR_RATE * breakdown["error_rate"] +
        WEIGHT_LATENCY * breakdown["latency"] +
        WEIGHT_PENDING * breakdown["pending"] +
        WEIGHT_ENTROPY * breakdown["entropy"] +
        WEIGHT_PROVIDERS * breakdown["providers"]
    )

    return score, breakdown


# =============================================================================
# Omega Heartbeat Daemon
# =============================================================================

class OmegaHeartbeat:
    """
    Main Omega Heartbeat daemon.

    Implements the ω-automaton monitoring loop with:
    - Incremental file tailing
    - Rich metric emission
    - Anomaly detection
    - Self-monitoring
    """

    def __init__(
        self,
        bus_dir: str | Path,
        interval_s: float = DEFAULT_INTERVAL_S,
        quiet: bool = False,
    ):
        self.bus_dir = Path(bus_dir)
        self.interval_s = interval_s
        self.quiet = quiet

        # Core components
        self.emitter = BusEmitter(bus_dir, actor="omega")
        self.tailer = FileTailer(self.bus_dir / "events.ndjson")
        self.state = OmegaState()
        self.provider_checker = ProviderHealthChecker()

        # Self-monitoring
        self.start_ts = now_ts()
        self.cycle_count = 0
        self.cycle_times: SlidingWindow = SlidingWindow(20)

        # Shutdown handling
        self._shutdown = threading.Event()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown handlers."""
        def handle_shutdown(signum, frame):
            if not self.quiet:
                print(f"\n[omega] Received signal {signum}, shutting down...")
            self._shutdown.set()

        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)

        # Register atexit for cleanup
        atexit.register(self._on_exit)

    def _on_exit(self) -> None:
        """Emit shutdown event."""
        self.emitter.emit(
            topic="omega.shutdown",
            kind="log",
            level="info",
            data={
                "message": "Omega heartbeat stopping",
                "uptime_s": now_ts() - self.start_ts,
                "total_cycles": self.cycle_count,
                "total_events_processed": self.state.event_count,
            },
        )

    def _log(self, msg: str) -> None:
        """Log message if not quiet."""
        if not self.quiet:
            print(f"[omega] {msg}")

    def _process_new_events(self) -> int:
        """Process new events from tailer."""
        count = 0
        for event in self.tailer.read_new_events():
            # Skip our own events to avoid feedback loops
            if event.get("actor") == "omega":
                continue
            self.state.process_event(event)
            count += 1
        return count

    def _emit_heartbeat(self) -> None:
        """Emit core heartbeat signal."""
        self.emitter.emit(
            topic="omega.heartbeat",
            kind="metric",
            level="info",
            data={
                "cycle": self.cycle_count,
                "uptime_s": now_ts() - self.start_ts,
                "version": VERSION,
                "protocol": PROTOCOL_VERSION,
            },
        )

    def _emit_velocity_metrics(self) -> None:
        """Emit velocity metrics."""
        velocity = self.state.compute_velocity()

        self.emitter.emit(
            topic="omega.metrics.velocity",
            kind="metric",
            level="info",
            data={
                "events_per_second": velocity,
                "total_events": self.state.event_count,
                "window_size": self.state.velocity_window.count,
            },
        )

    def _emit_latency_metrics(self) -> None:
        """Emit latency percentile metrics."""
        summary = self.state.get_latency_summary()

        self.emitter.emit(
            topic="omega.metrics.latency",
            kind="metric",
            level="info",
            data=summary,
        )

    def _emit_entropy_metrics(self) -> None:
        """Emit entropy metrics."""
        self.emitter.emit(
            topic="omega.metrics.entropy",
            kind="metric",
            level="info",
            data={
                "topic_entropy": self.state.compute_topic_entropy(),
                "actor_entropy": self.state.compute_actor_entropy(),
                "unique_topics": len(self.state.topic_counts),
                "unique_actors": len(self.state.actor_counts),
                "top_topics": dict(
                    sorted(self.state.topic_counts.items(), key=lambda x: -x[1])[:10]
                ),
                "top_actors": dict(
                    sorted(self.state.actor_counts.items(), key=lambda x: -x[1])[:10]
                ),
            },
        )

    def _emit_pending_pairs(self) -> None:
        """Emit pending pairs (ω-automaton backlog)."""
        summary = self.state.get_pending_summary()

        level = "info"
        if summary["total"] > 50:
            level = "warn"
        if summary["total"] > 100:
            level = "error"

        self.emitter.emit(
            topic="omega.pending.pairs",
            kind="metric",
            level=level,
            data=summary,
        )

    def _emit_provider_health(self) -> None:
        """Emit provider health status."""
        results = self.provider_checker.check_all()
        score = self.provider_checker.compute_score()

        self.emitter.emit(
            topic="omega.providers.health",
            kind="metric",
            level="info" if score > 0.5 else "warn",
            data={
                "providers": results,
                "health_score": score,
            },
        )

    def _emit_anomalies(self, anomalies: list[dict]) -> None:
        """Emit detected anomalies."""
        if not anomalies:
            return

        for anomaly in anomalies:
            self.emitter.emit(
                topic="omega.anomaly.detected",
                kind="alert",
                level=anomaly.get("severity", "warn"),
                data=anomaly,
            )

    def _emit_health_composite(self) -> None:
        """Emit composite health score."""
        velocity = self.state.compute_velocity()
        error_rate = self.state.compute_error_rate()
        latency_summary = self.state.get_latency_summary()
        pending_summary = self.state.get_pending_summary()
        topic_entropy = self.state.compute_topic_entropy()
        provider_score = self.provider_checker.compute_score()

        score, breakdown = compute_health_score(
            velocity=velocity,
            error_rate=error_rate,
            latency_p95=latency_summary["p95_s"],
            pending_count=pending_summary["total"],
            topic_entropy=topic_entropy,
            provider_score=provider_score,
        )

        # Determine level based on score
        if score >= 0.8:
            level = "info"
            status = "healthy"
        elif score >= 0.5:
            level = "warn"
            status = "degraded"
        else:
            level = "error"
            status = "unhealthy"

        self.emitter.emit(
            topic="omega.health.composite",
            kind="metric",
            level=level,
            data={
                "score": score,
                "status": status,
                "breakdown": breakdown,
                "weights": {
                    "velocity": WEIGHT_VELOCITY,
                    "error_rate": WEIGHT_ERROR_RATE,
                    "latency": WEIGHT_LATENCY,
                    "pending": WEIGHT_PENDING,
                    "entropy": WEIGHT_ENTROPY,
                    "providers": WEIGHT_PROVIDERS,
                },
            },
        )

    def _emit_self_metrics(self) -> None:
        """Emit self-monitoring metrics."""
        self.emitter.emit(
            topic="omega.meta.self",
            kind="metric",
            level="info",
            data={
                "cycle": self.cycle_count,
                "uptime_s": now_ts() - self.start_ts,
                "avg_cycle_time_ms": self.cycle_times.mean * 1000,
                "p95_cycle_time_ms": self.cycle_times.percentile(95) * 1000,
                "events_emitted": self.emitter.emit_count,
                "events_processed": self.state.event_count,
                "memory_pending_pairs": len(self.state.pending_pairs),
                "memory_request_timestamps": len(self.state.request_timestamps),
            },
        )

    def run_cycle(self) -> None:
        """Run a single monitoring cycle."""
        cycle_start = time.perf_counter()
        self.cycle_count += 1

        # Process new events (incremental)
        new_events = self._process_new_events()

        # Always emit heartbeat
        self._emit_heartbeat()

        # Emit velocity every cycle
        self._emit_velocity_metrics()

        # Emit pending pairs every cycle
        self._emit_pending_pairs()

        # Detect and emit anomalies
        anomalies = self.state.detect_anomalies()
        self._emit_anomalies(anomalies)

        # Every 3 cycles: latency and entropy
        if self.cycle_count % 3 == 0:
            self._emit_latency_metrics()
            self._emit_entropy_metrics()

        # Every 6 cycles: provider health
        if self.cycle_count % 6 == 0:
            self._emit_provider_health()

        # Every 10 cycles: composite health and self-metrics
        if self.cycle_count % 10 == 0:
            self._emit_health_composite()
            self._emit_self_metrics()

        # Track cycle time
        cycle_time = time.perf_counter() - cycle_start
        self.cycle_times.push(cycle_time)

        self._log(
            f"cycle={self.cycle_count} "
            f"new={new_events} "
            f"total={self.state.event_count} "
            f"pending={len(self.state.pending_pairs)} "
            f"time={format_duration(cycle_time)}"
        )

    def run(self) -> int:
        """
        Main run loop.

        Returns exit code.
        """
        self._log(f"Omega Heartbeat v{VERSION} starting")
        self._log(f"  Bus: {self.bus_dir}")
        self._log(f"  Interval: {self.interval_s}s")
        self._log(f"  Protocol: {PROTOCOL_VERSION}")

        # Emit startup event
        self.emitter.emit(
            topic="omega.startup",
            kind="log",
            level="info",
            data={
                "message": "Omega meta-observer online",
                "version": VERSION,
                "protocol": PROTOCOL_VERSION,
                "interval_s": self.interval_s,
                "pid": os.getpid(),
                "host": socket.gethostname(),
            },
        )

        # Main loop
        while not self._shutdown.is_set():
            try:
                self.run_cycle()
            except Exception as e:
                self._log(f"Cycle error: {e}")
                self.emitter.emit(
                    topic="omega.error",
                    kind="log",
                    level="error",
                    data={"error": str(e), "cycle": self.cycle_count},
                )

            # Sleep with shutdown check
            self._shutdown.wait(timeout=self.interval_s)

        self._log("Omega Heartbeat stopped")
        return 0


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Omega Heartbeat - Meta-observer for Pluribus liveness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Version: {VERSION}
Protocol: {PROTOCOL_VERSION}

Environment Variables:
  PLURIBUS_BUS_DIR    Bus directory (default: {DEFAULT_BUS_DIR})
  OMEGA_INTERVAL_S    Heartbeat interval (default: {DEFAULT_INTERVAL_S})

Examples:
  # Run with defaults
  python3 omega_heartbeat.py

  # Custom interval
  python3 omega_heartbeat.py --interval 5

  # Quiet mode (no stdout logging)
  python3 omega_heartbeat.py --quiet
""",
    )

    parser.add_argument(
        "--bus-dir",
        default=os.environ.get("PLURIBUS_BUS_DIR", DEFAULT_BUS_DIR),
        help=f"Bus directory (default: {DEFAULT_BUS_DIR})",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=float(os.environ.get("OMEGA_INTERVAL_S", DEFAULT_INTERVAL_S)),
        help=f"Heartbeat interval in seconds (default: {DEFAULT_INTERVAL_S})",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - no stdout logging",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"omega-heartbeat {VERSION}",
    )

    args = parser.parse_args()

    daemon = OmegaHeartbeat(
        bus_dir=args.bus_dir,
        interval_s=args.interval,
        quiet=args.quiet,
    )

    return daemon.run()


if __name__ == "__main__":
    sys.exit(main())
