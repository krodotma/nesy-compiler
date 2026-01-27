#!/usr/bin/env python3
"""
agent_bus.py - Bus event emission (FalkorDB primary, NDJSON DR-only)

Provides:
- Append-only writes with fallback support
- Optional partitioning and rotation
- CLI for publish/tail/resolve
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import io
import gzip
import json
import os
import socket
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore

# Performance: Cache hostname and PID at module load (avoid syscalls per emit)
_CACHED_HOSTNAME: str = ""
_CACHED_PID: int = 0

def _get_cached_hostname() -> str:
    global _CACHED_HOSTNAME
    if not _CACHED_HOSTNAME:
        _CACHED_HOSTNAME = socket.gethostname()
    return _CACHED_HOSTNAME

def _get_cached_pid() -> int:
    global _CACHED_PID
    if not _CACHED_PID:
        _CACHED_PID = os.getpid()
    return _CACHED_PID

# FalkorDB integration for real-time bus (NDJSON remains for DR)
_FALKORDB_SERVICE = None
_FALKORDB_CHECKED = False

# Performance: Event batching for high-throughput scenarios
_BATCH_BUFFER: list[tuple[str, str, bool, bool]] = []  # [(path, line, durable, rotate), ...]
_BATCH_SIZE = int(os.environ.get("PLURIBUS_BUS_BATCH_SIZE", "0"))  # 0 = disabled
_BATCH_FLUSH_MS = int(os.environ.get("PLURIBUS_BUS_BATCH_FLUSH_MS", "50"))
_BATCH_LOCK = None  # Lazy init threading.Lock
_BATCH_TIMER = None  # Background flush timer

# Non-blocking ring buffer + spooler for NDJSON DR writes
_SPOOLER = None  # Lazy init BusSpooler
_SPOOLER_LOCK = threading.Lock()


@dataclass(frozen=True)
class _SpoolItem:
    path: str
    line: str
    rotate: bool
    size_bytes: int


class _BusSpooler:
    def __init__(self, max_entries: int, max_bytes: int, flush_s: float, drop_policy: str):
        self._max_entries = max(0, max_entries)
        self._max_bytes = max(0, max_bytes)
        self._flush_s = max(0.0, flush_s)
        self._drop_policy = drop_policy
        self._queue: deque[_SpoolItem] = deque()
        self._bytes = 0
        self._dropped = 0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bus-spooler", daemon=True)
        self._thread.start()

    def enqueue(self, item: _SpoolItem) -> None:
        with self._lock:
            if not self._accept_item(item):
                return
            self._queue.append(item)
            self._bytes += item.size_bytes
        self._wake.set()

    def flush(self) -> None:
        self._drain()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"queued": len(self._queue), "bytes": self._bytes, "dropped": self._dropped}

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout_s)
        self._drain()

    def _accept_item(self, item: _SpoolItem) -> bool:
        if self._max_entries <= 0 and self._max_bytes <= 0:
            return True

        if self._drop_policy == "newest":
            if self._max_entries > 0 and len(self._queue) >= self._max_entries:
                self._dropped += 1
                return False
            if self._max_bytes > 0 and (self._bytes + item.size_bytes) > self._max_bytes:
                self._dropped += 1
                return False
            return True

        # Default: drop oldest to retain newest
        if self._max_entries > 0:
            while len(self._queue) >= self._max_entries:
                self._drop_oldest()
        if self._max_bytes > 0:
            while self._queue and (self._bytes + item.size_bytes) > self._max_bytes:
                self._drop_oldest()
            if not self._queue and (self._bytes + item.size_bytes) > self._max_bytes:
                self._dropped += 1
                return False
        return True

    def _drop_oldest(self) -> None:
        if not self._queue:
            return
        dropped = self._queue.popleft()
        self._bytes -= dropped.size_bytes
        self._dropped += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self._flush_s if self._flush_s > 0 else 0.05)
            self._wake.clear()
            self._drain()

    def _drain(self) -> None:
        with self._lock:
            if not self._queue:
                return
            items = list(self._queue)
            self._queue.clear()
            self._bytes = 0

        by_path: dict[str, list[str]] = {}
        rotate_paths: set[str] = set()
        for item in items:
            by_path.setdefault(item.path, []).append(item.line)
            if item.rotate:
                rotate_paths.add(item.path)

        for path, lines in by_path.items():
            _append_line_direct(path, "".join(lines), durable=False)

        for path in rotate_paths:
            _apply_rotation_policy(path, durable=False)


def _spooler_enabled() -> bool:
    return _truthy_env(os.environ.get("PLURIBUS_BUS_SPOOLER"), True)


def _spooler_config() -> tuple[int, int, float, str]:
    def _int_env(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _float_env(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    max_entries = _int_env("PLURIBUS_BUS_SPOOLER_SIZE", 4096)
    max_bytes = _int_env("PLURIBUS_BUS_SPOOLER_MAX_BYTES", 8 * 1024 * 1024)
    flush_ms = _float_env("PLURIBUS_BUS_SPOOLER_FLUSH_MS", 25.0)
    drop_policy = (os.environ.get("PLURIBUS_BUS_SPOOLER_DROP") or "oldest").strip().lower()
    if drop_policy not in {"oldest", "newest"}:
        drop_policy = "oldest"
    return max_entries, max_bytes, max(0.0, flush_ms / 1000.0), drop_policy


def _get_spooler() -> _BusSpooler:
    global _SPOOLER
    if _SPOOLER is not None:
        return _SPOOLER
    with _SPOOLER_LOCK:
        if _SPOOLER is None:
            max_entries, max_bytes, flush_s, drop_policy = _spooler_config()
            _SPOOLER = _BusSpooler(max_entries, max_bytes, flush_s, drop_policy)
            atexit.register(_SPOOLER.stop)
    return _SPOOLER


def flush_spooler() -> None:
    if _SPOOLER is None:
        return
    _SPOOLER.flush()


def _reset_spooler_for_tests() -> None:
    global _SPOOLER
    if _SPOOLER is None:
        return
    _SPOOLER.stop()
    _SPOOLER = None

def _get_falkordb_service():
    """Get FalkorDB service, lazy-loaded. Returns None if unavailable."""
    global _FALKORDB_SERVICE, _FALKORDB_CHECKED
    if _FALKORDB_CHECKED:
        return _FALKORDB_SERVICE
    _FALKORDB_CHECKED = True

    backend = os.environ.get("PLURIBUS_BUS_BACKEND", "both").lower()
    if backend == "ndjson":
        return None  # NDJSON-only mode

    try:
        from falkordb_bus_events import BusEventService
        _FALKORDB_SERVICE = BusEventService()
        if _FALKORDB_SERVICE.connect():
            return _FALKORDB_SERVICE
        else:
            _FALKORDB_SERVICE = None
    except Exception:
        pass
    return None

EVENTS_FILE = "events.ndjson"
BUS_FILE_MODE = 0o666
_PARTITION_CONFIG_CACHE: Optional[dict[str, Any]] = None


class EventKind:
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    ERROR = "error"


class EventLevel:
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass
class BusEvent:
    """Canonical bus event structure with optional lineage metadata."""
    id: str
    ts: float
    iso: str
    topic: str
    kind: str
    level: str
    actor: str
    data: dict
    host: Optional[str] = None
    pid: Optional[int] = None
    trace_id: Optional[str] = None
    run_id: Optional[str] = None
    lineage_id: Optional[str] = None
    parent_lineage_id: Optional[str] = None
    mutation_op: Optional[str] = None

    @classmethod
    def create(
        cls,
        topic: str,
        actor: str,
        data: dict,
        kind: str = EventKind.EVENT,
        level: str = EventLevel.INFO,
        lineage_id: Optional[str] = None,
        parent_lineage_id: Optional[str] = None,
        mutation_op: Optional[str] = None,
    ) -> "BusEvent":
        now = time.time()
        return cls(
            id=str(uuid.uuid4()),
            ts=now,
            iso=datetime.utcfromtimestamp(now).isoformat() + "Z",
            topic=topic,
            kind=kind,
            level=level,
            actor=actor,
            data=data,
            host=socket.gethostname(),
            pid=os.getpid(),
            lineage_id=lineage_id,
            parent_lineage_id=parent_lineage_id,
            mutation_op=mutation_op,
        )


@dataclass(frozen=True)
class BusPaths:
    """Resolved bus directories."""
    active_dir: str
    events_path: str
    primary_dir: str
    fallback_dir: Optional[str]

    @property
    def bus_dir(self) -> str:
        return self.active_dir


def _truthy_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _protocol_is_v1() -> bool:
    protocol = (os.environ.get("PLURIBUS_PROTOCOL") or "").strip().lower()
    if not protocol:
        return False
    return "v1" in protocol


def _parse_since_arg(raw: str | None) -> Optional[float]:
    if not raw:
        return None
    value = raw.strip().lower()
    try:
        if value.endswith("h"):
            return time.time() - (float(value[:-1]) * 3600)
        if value.endswith("d"):
            return time.time() - (float(value[:-1]) * 86400)
        if value.endswith("m"):
            return time.time() - (float(value[:-1]) * 60)
        return float(value)
    except ValueError:
        return None


def _ndjson_read_allowed() -> bool:
    explicit = os.environ.get("PLURIBUS_NDJSON_READ")
    if explicit is not None:
        return _truthy_env(explicit, False)
    mode = (os.environ.get("PLURIBUS_NDJSON_MODE") or "").strip().lower()
    if not mode or mode in {"allow", "enabled", "on"}:
        if _protocol_is_v1():
            return _truthy_env(os.environ.get("PLURIBUS_DR_MODE"), False)
        return True
    if mode in {"dr", "disaster", "recovery"}:
        return _truthy_env(os.environ.get("PLURIBUS_DR_MODE"), False)
    if mode in {"off", "disabled", "deny", "no"}:
        return False
    return True


def _ndjson_write_allowed() -> bool:
    explicit = os.environ.get("PLURIBUS_NDJSON_WRITE")
    if explicit is not None:
        return _truthy_env(explicit, False)
    mode = (os.environ.get("PLURIBUS_NDJSON_MODE") or "").strip().lower()
    if not mode or mode in {"allow", "enabled", "on"}:
        if _protocol_is_v1():
            return _truthy_env(os.environ.get("PLURIBUS_DR_MODE"), False)
        return True
    if mode in {"dr", "disaster", "recovery"}:
        return _truthy_env(os.environ.get("PLURIBUS_DR_MODE"), False)
    if mode in {"off", "disabled", "deny", "no"}:
        return False
    return True


def _require_ndjson_read_allowed() -> None:
    if _ndjson_read_allowed():
        return
    raise RuntimeError(
        "NDJSON reads are disabled (set PLURIBUS_DR_MODE=1 or PLURIBUS_NDJSON_MODE=allow)."
    )


def _tail_irk_fallback(paths: BusPaths, limit: int, since_ts: Optional[float], as_json: bool) -> int:
    if not _ndjson_read_allowed():
        sys.stderr.write(
            "NDJSON reads are disabled (set PLURIBUS_DR_MODE=1 or PLURIBUS_NDJSON_MODE=allow).\n"
        )
        return 3
    lines = _tail_lines(Path(paths.events_path), limit)
    timeline: list[dict[str, Any]] = []
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = obj.get("ts")
        if since_ts is not None and isinstance(ts, (int, float)) and ts < since_ts:
            continue
        timeline.append({
            "id": obj.get("id"),
            "topic": obj.get("topic"),
            "actor": obj.get("actor"),
            "ts": ts,
            "trace_id": obj.get("trace_id"),
        })
    timeline.sort(key=lambda item: item.get("ts") or 0, reverse=True)
    timeline = timeline[:limit]
    if as_json:
        print(json.dumps(timeline, ensure_ascii=False))
    else:
        for item in timeline:
            print(json.dumps(item, ensure_ascii=False))
    return 0


def _default_bus_dir() -> Path:
    root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
    return Path(os.environ.get("PLURIBUS_BUS_DIR") or (Path(root) / ".pluribus" / "bus"))


def _default_fallback_bus_dir() -> Optional[Path]:
    env = (os.environ.get("PLURIBUS_FALLBACK_BUS_DIR") or "").strip()
    if env:
        return Path(env)
    root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
    return Path(root) / ".pluribus_local" / "bus"


def _ensure_mode(path: Path, mode: int = BUS_FILE_MODE) -> None:
    try:
        os.chmod(path, mode)
    except (PermissionError, OSError):
        return


def append_line(path: str, line: str, durable: bool, rotate: bool = False) -> None:
    """Append a line to a file with optional batching/spooling for performance."""
    global _BATCH_LOCK, _BATCH_TIMER

    if _spooler_enabled() and not durable:
        spooler = _get_spooler()
        spooler.enqueue(_SpoolItem(path=path, line=line, rotate=rotate, size_bytes=len(line.encode("utf-8"))))
        return

    if durable and _spooler_enabled():
        flush_spooler()

    # Fast path: batching disabled or durable write requested
    if _BATCH_SIZE <= 0 or durable:
        _append_line_direct(path, line, durable)
        if rotate:
            _apply_rotation_policy(path, durable)
        return

    # Batching path: accumulate writes
    if _BATCH_LOCK is None:
        _BATCH_LOCK = threading.Lock()

    with _BATCH_LOCK:
        _BATCH_BUFFER.append((path, line, durable, rotate))

        # Flush if batch is full
        if len(_BATCH_BUFFER) >= _BATCH_SIZE:
            _flush_batch()
        elif _BATCH_TIMER is None and _BATCH_FLUSH_MS > 0:
            # Schedule timed flush
            def flush_timer():
                global _BATCH_TIMER
                with _BATCH_LOCK:
                    _flush_batch()
                    _BATCH_TIMER = None

            _BATCH_TIMER = threading.Timer(_BATCH_FLUSH_MS / 1000.0, flush_timer)
            _BATCH_TIMER.daemon = True
            _BATCH_TIMER.start()


def _flush_batch() -> None:
    """Flush accumulated batch writes. Must be called with _BATCH_LOCK held."""
    global _BATCH_BUFFER

    if not _BATCH_BUFFER:
        return

    # Group by path for efficient multi-line writes
    by_path: dict[str, list[str]] = {}
    rotate_paths: set[str] = set()

    for path, line, durable, rotate in _BATCH_BUFFER:
        if path not in by_path:
            by_path[path] = []
        by_path[path].append(line)
        if rotate:
            rotate_paths.add(path)

    _BATCH_BUFFER = []

    # Write each path's accumulated lines
    for path, lines in by_path.items():
        combined = "".join(lines)
        _append_line_direct(path, combined, durable=False)
        if path in rotate_paths:
            _apply_rotation_policy(path, durable=False)


def _append_line_direct(path: str, line: str, durable: bool) -> None:
    """Direct file append without batching."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_WRONLY | os.O_APPEND | os.O_CREAT, BUS_FILE_MODE)
    try:
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError:
                pass
        os.write(fd, line.encode("utf-8"))
        if durable:
            os.fsync(fd)
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)
    _ensure_mode(p)


def _can_append_events(path: Path) -> bool:
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, BUS_FILE_MODE)
    except OSError:
        return False
    try:
        os.close(fd)
    except OSError:
        pass
    _ensure_mode(path)
    return True


def _touch_events(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, BUS_FILE_MODE)
        try:
            os.close(fd)
        except OSError:
            pass
    _ensure_mode(path)


def resolve_bus_paths(bus_dir: Optional[str] = None) -> BusPaths:
    primary = Path(bus_dir) if bus_dir else _default_bus_dir()
    fallback = _default_fallback_bus_dir()

    primary_events = primary / EVENTS_FILE
    active = primary
    fallback_dir: Optional[str] = None

    if not _can_append_events(primary_events):
        if fallback is not None:
            fallback_events = fallback / EVENTS_FILE
            _touch_events(fallback_events)
            active = fallback
            fallback_dir = str(fallback)
        else:
            _touch_events(primary_events)
    else:
        _touch_events(primary_events)

    events_path = str(Path(active) / EVENTS_FILE)
    return BusPaths(
        active_dir=str(active),
        events_path=events_path,
        primary_dir=str(primary),
        fallback_dir=fallback_dir,
    )


def _load_partition_config() -> dict[str, Any]:
    global _PARTITION_CONFIG_CACHE
    if _PARTITION_CONFIG_CACHE is not None:
        return _PARTITION_CONFIG_CACHE
    config_path = (os.environ.get("PLURIBUS_BUS_PARTITION_CONFIG") or "").strip()
    if not config_path:
        _PARTITION_CONFIG_CACHE = {}
        return _PARTITION_CONFIG_CACHE
    try:
        data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    _PARTITION_CONFIG_CACHE = data
    return _PARTITION_CONFIG_CACHE


def _topic_bucket(topic: str) -> str:
    overrides = _load_partition_config().get("bucket_overrides") or {}
    if isinstance(overrides, dict):
        ordered = sorted(overrides.items(), key=lambda kv: len(kv[0]), reverse=True)
        for prefix, bucket in ordered:
            if topic.startswith(prefix):
                return str(bucket)

    t = (topic or "").lower()
    if t.startswith("omega."):
        return "omega"
    if t.startswith("qa."):
        return "qa"
    if t.startswith("telemetry."):
        return "telemetry"
    if t.startswith("browser."):
        return "browser"
    if t.startswith("dashboard."):
        return "dashboard"
    if t.startswith("agent."):
        return "agent"
    if t.startswith("operator."):
        return "operator"
    if t.startswith("rd.tasks.") or t.startswith("task."):
        return "task"
    if t.startswith("a2a."):
        return "a2a"
    if t.startswith("lens."):
        return "lens"
    if t.startswith("dialogos."):
        return "dialogos"
    if t.startswith("infer_sync."):
        return "infer_sync"
    if t.startswith("providers."):
        return "providers"
    return "other"


def _partition_enabled() -> bool:
    return _truthy_env(os.environ.get("PLURIBUS_BUS_PARTITION"), True)


def _partition_fanout() -> list[str]:
    raw = (os.environ.get("PLURIBUS_BUS_PARTITION_FANOUT") or "topic,type").strip()
    if not raw:
        return []
    parts = [p.strip().lower() for p in raw.replace(" ", ",").split(",") if p.strip()]
    allowed = {"topic", "type", "eventtypes", "actor", "frequency"}
    out: list[str] = []
    for p in parts:
        if p == "eventtypes":
            p = "type"
        if p in allowed and p not in out:
            out.append(p)
    return out


def _partition_shards() -> int:
    raw = (os.environ.get("PLURIBUS_BUS_PARTITION_SHARDS") or "1").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 1
    return max(1, value)


def _frequency_bucket(_: dict) -> str:
    return "hot"


def _topic_segments(topic: str, bucket: str) -> list[str]:
    parts = [p for p in (topic or "").split(".") if p]
    if parts and parts[0] == bucket:
        parts = parts[1:]
    return parts or ["root"]


def _write_partition(base: Path, parts: list[str], line: str, durable: bool) -> None:
    path = base
    for part in parts:
        path = path / part
    path = path / EVENTS_FILE
    append_line(str(path), line, durable)


def _emit_partitions(paths: BusPaths, payload: dict, line: str, durable: bool) -> None:
    if not _partition_enabled():
        return

    fanout = _partition_fanout()
    if not fanout:
        return

    topic = str(payload.get("topic") or "")
    kind = str(payload.get("kind") or "log")
    level = str(payload.get("level") or "info")
    actor = str(payload.get("actor") or "unknown")
    bucket = _topic_bucket(topic)
    segments = _topic_segments(topic, bucket)

    shard_count = _partition_shards()
    shard_suffix: list[str] = []
    if shard_count > 1:
        shard = abs(hash(payload.get("id") or topic)) % shard_count
        shard_suffix = [f"shard-{shard}"]

    base_root = Path(paths.active_dir) / "topics"
    for mode in fanout:
        if mode == "topic":
            base = base_root / "topic" / Path(*shard_suffix)
            _write_partition(base / bucket, segments, line, durable)
        elif mode == "type":
            base = base_root / "eventtypes" / Path(*shard_suffix)
            _write_partition(base / kind / level / bucket, segments, line, durable)
        elif mode == "actor":
            base = base_root / "actors" / Path(*shard_suffix)
            _write_partition(base / actor / bucket, segments, line, durable)
        elif mode == "frequency":
            base = base_root / "frequency" / Path(*shard_suffix)
            freq = _frequency_bucket(payload)
            _write_partition(base / freq / bucket, segments, line, durable)


def rotate_log_tail(
    path: str,
    *,
    retain_bytes: int,
    archive_dir: str,
    durable: bool = False,
) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    size = p.stat().st_size
    if size <= retain_bytes:
        return None

    archive_root = Path(archive_dir)
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = archive_root / f"{p.stem}-{stamp}{p.suffix}.gz"

    cutoff = max(0, size - retain_bytes)
    tail_bytes = b""
    with p.open("rb") as src:
        if cutoff > 0:
            with gzip.open(archive_path, "wb") as gz:
                remaining = cutoff
                while remaining > 0:
                    chunk = src.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    gz.write(chunk)
                    remaining -= len(chunk)
        src.seek(cutoff)
        tail_bytes = src.read()

    tmp_path = p.with_suffix(p.suffix + ".tmp")
    with tmp_path.open("wb") as out:
        out.write(tail_bytes)
        if durable:
            out.flush()
            os.fsync(out.fileno())
    os.replace(tmp_path, p)
    _ensure_mode(p)
    return str(archive_path)


def _rotation_limits() -> tuple[int, int, int, int]:
    # Defaults: rotate at 90MB, cap at 100MB, retain tail at rotate threshold.
    def _float_env(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    rotate_mb = _float_env("PLURIBUS_BUS_ROTATE_MB", 90.0)
    cap_mb = _float_env("PLURIBUS_BUS_CAP_MB", 100.0)
    retain_raw = os.environ.get("PLURIBUS_BUS_RETAIN_MB")
    if retain_raw is None or retain_raw == "":
        retain_mb = rotate_mb
    else:
        retain_mb = _float_env("PLURIBUS_BUS_RETAIN_MB", rotate_mb)
    archive_retain_mb = _float_env("PLURIBUS_BUS_ARCHIVE_RETAIN_MB", 0.0)

    rotate_mb = max(0.0, rotate_mb)
    retain_mb = max(0.0, retain_mb)
    cap_mb = max(0.0, cap_mb)
    if cap_mb > 0:
        retain_mb = min(retain_mb, cap_mb)

    return (
        int(rotate_mb * 1024 * 1024),
        int(retain_mb * 1024 * 1024),
        int(cap_mb * 1024 * 1024),
        int(archive_retain_mb * 1024 * 1024),
    )


def _prune_archives(archive_dir: Path, retain_bytes: int) -> None:
    if retain_bytes <= 0 or not archive_dir.exists():
        return
    entries = [p for p in archive_dir.glob("*.gz") if p.is_file()]
    if not entries:
        return
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    kept = 0
    for entry in entries:
        size = entry.stat().st_size
        if kept < retain_bytes:
            kept += size
            continue
        try:
            entry.unlink()
        except OSError:
            continue


def _apply_rotation_policy(events_path: str, durable: bool) -> None:
    if not _truthy_env(os.environ.get("PLURIBUS_BUS_ROTATE"), True):
        return
    rotate_bytes, retain_bytes, cap_bytes, archive_retain_bytes = _rotation_limits()
    if rotate_bytes <= 0 and cap_bytes <= 0:
        return
    target = Path(events_path)
    if not target.exists():
        return
    size = target.stat().st_size
    if size <= 0:
        return
    should_rotate = rotate_bytes > 0 and size >= rotate_bytes
    should_cap = cap_bytes > 0 and size > cap_bytes
    if not (should_rotate or should_cap):
        return
    if retain_bytes <= 0 and cap_bytes > 0:
        retain_bytes = cap_bytes
    if retain_bytes <= 0:
        return
    archive_dir = target.parent / "archive"
    rotate_log_tail(
        str(target),
        retain_bytes=retain_bytes,
        archive_dir=str(archive_dir),
        durable=durable,
    )
    if archive_retain_bytes > 0:
        _prune_archives(archive_dir, archive_retain_bytes)


def _maybe_rotate(paths: BusPaths, durable: bool) -> None:
    _apply_rotation_policy(paths.events_path, durable)


def _emit_payload(paths: BusPaths, payload: dict, durable: bool) -> str:
    """Emit event to FalkorDB (primary) and/or NDJSON (DR).

    Backend modes (PLURIBUS_BUS_BACKEND env var):
    - 'both' (default): Write to FalkorDB + NDJSON
    - 'falkordb': Write to FalkorDB only (fastest, no DR)
    - 'ndjson': Write to NDJSON only (legacy mode)
    """
    backend = os.environ.get("PLURIBUS_BUS_BACKEND", "both").lower()

    # Write to FalkorDB if enabled
    if backend in ("both", "falkordb"):
        service = _get_falkordb_service()
        if service:
            try:
                service.record_event_dict(payload)
            except Exception:
                pass  # Graceful degradation - NDJSON still available

    # Write to NDJSON for DR unless falkordb-only mode
    ndjson_allowed = _ndjson_write_allowed() or backend == "ndjson"
    if backend != "falkordb" and ndjson_allowed:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        append_line(paths.events_path, line, durable, rotate=True)

        if paths.primary_dir and paths.primary_dir != paths.active_dir:
            primary_events = Path(paths.primary_dir) / EVENTS_FILE
            if _can_append_events(primary_events):
                append_line(str(primary_events), line, durable, rotate=True)

        _emit_partitions(paths, payload, line, durable)
    elif backend != "falkordb" and not ndjson_allowed:
        # DR-only policy: avoid NDJSON writes unless explicitly enabled.
        pass

    return str(payload.get("id"))


def emit_event(
    paths: BusPaths,
    *,
    topic: str,
    kind: str,
    level: str,
    actor: str,
    data: dict,
    trace_id: Optional[str] = None,
    run_id: Optional[str] = None,
    durable: bool = False,
) -> str:
    """Emit an event to the bus.

    Performance optimizations:
    - Cached hostname/PID to avoid syscalls
    - Compact JSON serialization
    - Optional batching via PLURIBUS_BUS_BATCH_SIZE env var
    """
    now = time.time()
    payload = {
        "id": str(uuid.uuid4()),
        "ts": now,
        "iso": datetime.now(timezone.utc).isoformat() + "Z",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": _get_cached_hostname(),  # Performance: cached
        "pid": _get_cached_pid(),  # Performance: cached
        "data": data,
    }
    if trace_id:
        payload["trace_id"] = trace_id
    if run_id:
        payload["run_id"] = run_id
    return _emit_payload(paths, payload, durable)


def emit_bus_event(
    topic: str,
    actor: str,
    data: dict,
    kind: str = EventKind.EVENT,
    level: str = EventLevel.INFO,
    lineage_id: Optional[str] = None,
    parent_lineage_id: Optional[str] = None,
    mutation_op: Optional[str] = None,
) -> str:
    paths = resolve_bus_paths(None)
    payload = BusEvent.create(
        topic,
        actor,
        data,
        kind=kind,
        level=level,
        lineage_id=lineage_id,
        parent_lineage_id=parent_lineage_id,
        mutation_op=mutation_op,
    )
    return _emit_payload(paths, asdict(payload), durable=False)


def iter_lines_follow(path: str, sleep_s: float = 0.25) -> Generator[str, None, None]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+", encoding="utf-8") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if line:
                yield line.rstrip("\n")
            else:
                time.sleep(sleep_s)


def _tail_lines(path: Path, limit: int) -> list[str]:
    if limit <= 0 or not path.exists():
        return []
    buf: deque[str] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line:
                buf.append(line.rstrip("\n"))
    return list(buf)


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("CAGENT_ID") or os.environ.get("USER") or "unknown"


def get_agent_id() -> str:
    return os.environ.get("CAGENT_ID", "unknown")


class AgentBus:
    """Append-only NDJSON bus wrapper for older integrations."""

    def __init__(self, bus_dir: Optional[Path] = None):
        self.paths = resolve_bus_paths(str(bus_dir) if bus_dir else None)
        self.events_file = Path(self.paths.events_path)

    def emit(self, event: BusEvent) -> str:
        payload = asdict(event)
        payload.setdefault("host", socket.gethostname())
        payload.setdefault("pid", os.getpid())
        return _emit_payload(self.paths, payload, durable=False)

    def tail(self, n: int = 10) -> list[BusEvent]:
        _require_ndjson_read_allowed()
        lines = _tail_lines(self.events_file, n)
        events: list[BusEvent] = []
        for line in lines:
            try:
                data = json.loads(line)
                events.append(BusEvent(**data))
            except (json.JSONDecodeError, TypeError):
                continue
        return events

    def watch(self, topic_filter: Optional[str] = None) -> Generator[BusEvent, None, None]:
        _require_ndjson_read_allowed()
        for line in iter_lines_follow(str(self.events_file)):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if topic_filter and not str(data.get("topic", "")).startswith(topic_filter):
                continue
            try:
                yield BusEvent(**data)
            except TypeError:
                continue

    def query(self, topic_filter: Optional[str] = None, limit: int = 100) -> list[BusEvent]:
        _require_ndjson_read_allowed()
        events: list[BusEvent] = []
        if not self.events_file.exists():
            return events
        with self.events_file.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if topic_filter and not str(data.get("topic", "")).startswith(topic_filter):
                    continue
                try:
                    events.append(BusEvent(**data))
                except TypeError:
                    continue
                if len(events) >= limit:
                    break
        return events


# Topic constants
class Topics:
    CAGENT_BOOTSTRAP_START = "cagent.bootstrap.start"
    CAGENT_BOOTSTRAP_COMPLETE = "cagent.bootstrap.complete"
    CAGENT_TASK_START = "cagent.task.start"
    CAGENT_TASK_UPDATE = "cagent.task.update"
    CAGENT_TASK_COMPLETE = "cagent.task.complete"
    CAGENT_TASK_RESUME = "cagent.task.resume"
    CAGENT_DELEGATE_REQUEST = "cagent.delegate.request"
    CAGENT_ESCALATE_REQUEST = "cagent.escalate.request"

    DIALOGOS_SUBMIT = "dialogos.submit"
    DIALOGOS_CELL = "dialogos.cell"
    DIALOGOS_ERROR = "dialogos.error"
    DIALOGOS_TIMEOUT = "dialogos.timeout"

    PAIP_CLONE_CREATED = "paip.clone.created"
    PAIP_CLONE_SYNCED = "paip.clone.synced"
    PAIP_CLONE_DELETED = "paip.clone.deleted"
    PAIP_ORPHAN_DETECTED = "paip.orphan.detected"

    AMBER_PRESERVATION_START = "amber.preservation.start"
    AMBER_PRESERVATION_COMPLETE = "amber.preservation.complete"

    OPERATOR_COMMAND = "operator.command"
    OPERATOR_RESPONSE = "operator.response"


def _parse_json_arg(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _filter_event(obj: dict, args: argparse.Namespace) -> bool:
    if args.topic and not str(obj.get("topic", "")).startswith(args.topic):
        return False
    if args.actor and obj.get("actor") != args.actor:
        return False
    if args.kind and obj.get("kind") != args.kind:
        return False
    if args.level and obj.get("level") != args.level:
        return False
    return True


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="agent_bus.py", description="Append-only bus")
    ap.add_argument("--bus-dir", help="Override bus directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pub = sub.add_parser("pub", help="Publish event")
    pub.add_argument("--topic", required=True)
    pub.add_argument("--kind", required=True)
    pub.add_argument("--level", required=True)
    pub.add_argument("--actor", default=default_actor())
    pub.add_argument("--data", default="{}")
    pub.add_argument("--trace-id")
    pub.add_argument("--run-id")
    pub.add_argument("--durable", action="store_true")

    tail = sub.add_parser("tail", help="Tail events")
    tail.add_argument("--limit", type=int, default=50)
    tail.add_argument("--topic")
    tail.add_argument("--actor")
    tail.add_argument("--kind")
    tail.add_argument("--level")
    tail.add_argument("--raw", action="store_true")
    tail.add_argument("--follow", action="store_true")

    tail_irk = sub.add_parser("tail-irk", help="Tail IRKG event timeline (no NDJSON)")
    tail_irk.add_argument("--limit", type=int, default=50)
    tail_irk.add_argument("--since", help="Time window (e.g., 1h, 24h, 7d)")
    tail_irk.add_argument("--json", action="store_true", help="Output JSON array (default JSON lines)")
    tail_irk.add_argument("--fallback-ndjson", action="store_true", help="Allow NDJSON fallback when IRKG is unavailable")

    resolve = sub.add_parser("resolve", help="Resolve bus directory")
    resolve.add_argument("--events-path", action="store_true")
    resolve.add_argument("--json", action="store_true")

    sub.add_parser("mk-run-id", help="Generate run id")

    args = ap.parse_args(argv)

    paths = resolve_bus_paths(args.bus_dir)

    if args.cmd == "pub":
        data = _parse_json_arg(args.data)
        event_id = emit_event(
            paths,
            topic=args.topic,
            kind=args.kind,
            level=args.level,
            actor=args.actor,
            data=data if isinstance(data, dict) else {"message": data},
            trace_id=args.trace_id,
            run_id=args.run_id,
            durable=args.durable,
        )
        print(event_id)
        return 0

    if args.cmd == "tail":
        if not _ndjson_read_allowed():
            sys.stderr.write(
                "NDJSON reads are disabled (set PLURIBUS_DR_MODE=1 or PLURIBUS_NDJSON_MODE=allow).\n"
            )
            return 3
        lines = iter_lines_follow(paths.events_path) if args.follow else _tail_lines(Path(paths.events_path), args.limit)
        for line in lines:
            if args.raw:
                print(line)
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _filter_event(obj, args):
                continue
            print(json.dumps(obj, ensure_ascii=False))
        return 0

    if args.cmd == "tail-irk":
        fallback_enabled = args.fallback_ndjson or _truthy_env(os.environ.get("PLURIBUS_IRKG_FALLBACK"), False)
        try:
            from nucleus.tools.falkordb_bus_events import BusEventService
        except Exception as exc:
            if fallback_enabled:
                since_ts = _parse_since_arg(args.since)
                return _tail_irk_fallback(paths, args.limit, since_ts, args.json)
            sys.stderr.write(f"IRKG tail unavailable: {exc}\n")
            return 2

        since_ts = _parse_since_arg(args.since)
        if args.since and since_ts is None:
            sys.stderr.write("Invalid --since value (use 1h, 24h, 7d, or epoch seconds).\n")
            return 2

        service = BusEventService()
        err_sink = io.StringIO() if fallback_enabled else None
        with contextlib.redirect_stderr(err_sink) if err_sink else contextlib.nullcontext():
            timeline = service.get_event_timeline(since_ts=since_ts, limit=args.limit)
        if not service._connected:
            if fallback_enabled:
                return _tail_irk_fallback(paths, args.limit, since_ts, args.json)
            sys.stderr.write("IRKG unavailable and NDJSON fallback disabled.\n")
            return 3
        if args.json:
            print(json.dumps(timeline, ensure_ascii=False))
        else:
            for item in timeline:
                print(json.dumps(item, ensure_ascii=False))
        return 0

    if args.cmd == "resolve":
        if args.json:
            print(json.dumps({
                "active_dir": paths.active_dir,
                "primary_dir": paths.primary_dir,
                "fallback_dir": paths.fallback_dir,
                "events_path": paths.events_path,
            }, ensure_ascii=False))
        elif args.events_path:
            print(paths.events_path)
        else:
            print(paths.active_dir)
        return 0

    if args.cmd == "mk-run-id":
        print(str(uuid.uuid4()))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
