#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import random
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

sys.dont_write_bytecode = True

try:
    from .agent_bus import resolve_bus_paths, emit_event, default_actor
    from .qa_ir import normalize_event, QAIrStore
    from .core.bus_consumer import BusConsumer
except Exception:  # pragma: no cover
    from agent_bus import resolve_bus_paths, emit_event, default_actor
    from qa_ir import normalize_event, QAIrStore
    try:
        from core.bus_consumer import BusConsumer
    except ImportError:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from core.bus_consumer import BusConsumer


def _default_output_dir() -> str:
    root = os.environ.get("PLURIBUS_ROOT") or "/pluribus"
    return os.path.join(root, ".pluribus", "index", "qa")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_list(name: str) -> List[str]:
    raw = os.environ.get(name, "")
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_meminfo(text: str) -> Dict[str, int]:
    info: Dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0].rstrip(":")
        try:
            value = int(parts[1])
        except ValueError:
            continue
        info[key] = value
    return info


def _read_meminfo(path: str = "/proc/meminfo") -> Dict[str, int]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return _parse_meminfo(handle.read())
    except OSError:
        return {}


def _mem_available_kb(info: Dict[str, int]) -> int:
    available = info.get("MemAvailable")
    if available is None:
        available = info.get("MemFree", 0)
    return int(available or 0)


def _mem_total_kb(info: Dict[str, int]) -> int:
    return int(info.get("MemTotal", 0) or 0)


def _memory_pressure(info: Dict[str, int], *, min_available_kb: int, min_available_ratio: float) -> bool:
    if not info:
        return False
    available_kb = _mem_available_kb(info)
    total_kb = _mem_total_kb(info)
    if min_available_kb > 0 and available_kb and available_kb < min_available_kb:
        return True
    if min_available_ratio > 0 and total_kb > 0:
        return (available_kb / total_kb) < min_available_ratio
    return False


def _ensure_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "a", encoding="utf-8"):
            pass


def _open_tail(path: str) -> Tuple[Any, int]:
    handle = open(path, "r", encoding="utf-8", errors="replace")
    handle.seek(0, os.SEEK_END)
    inode = os.fstat(handle.fileno()).st_ino
    return handle, inode


def should_rotate(stats: os.stat_result, *, now: float, max_bytes: int, max_age_s: int) -> bool:
    if max_bytes > 0 and stats.st_size >= max_bytes:
        return True
    if max_age_s > 0 and (now - stats.st_mtime) >= max_age_s:
        return True
    return False


def rotate_bus_file(events_path: str, archive_dir: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(events_path):
        return None
    stats = os.stat(events_path)
    if stats.st_size <= 0:
        return None
    os.makedirs(archive_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive_path = os.path.join(archive_dir, f"events-{ts}-{os.getpid()}.ndjson")
    os.rename(events_path, archive_path)
    with open(events_path, "a", encoding="utf-8"):
        pass
    return {"archive_path": archive_path, "bytes": stats.st_size}


class RateTracker:
    def __init__(self, window_s: int):
        self.window_s = window_s
        self.samples: Dict[str, List[float]] = {}

    def add(self, topic: str, ts: float) -> float:
        bucket = self.samples.setdefault(topic, [])
        bucket.append(ts)
        cutoff = float(ts) - self.window_s
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        return len(bucket) / float(self.window_s)


class RecurrenceTracker:
    def __init__(self, window_s: int):
        self.window_s = window_s
        self.samples: Dict[str, List[float]] = {}

    def add(self, key: str, ts: float) -> Tuple[int, float]:
        bucket = self.samples.setdefault(key, [])
        bucket.append(ts)
        cutoff = float(ts) - self.window_s
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        if len(bucket) < 2:
            return len(bucket), 0.0
        return len(bucket), max(0.0, bucket[-1] - bucket[0])


def _should_ignore(topic: str, prefixes: List[str]) -> bool:
    for prefix in prefixes:
        if topic.startswith(prefix):
            return True
    return False


def _in_scope(topic: str, include_prefixes: List[str], ignore_prefixes: List[str]) -> bool:
    if include_prefixes and not any(topic.startswith(p) for p in include_prefixes):
        return False
    if _should_ignore(topic, ignore_prefixes):
        return False
    return True


def _emit(bus_dir: str, topic: str, data: Dict[str, Any], *, kind: str = "metric", level: str = "info") -> None:
    paths = resolve_bus_paths(bus_dir)
    emit_event(
        paths,
        topic=topic,
        kind=kind,
        level=level,
        actor=default_actor(),
        data=data,
        trace_id=data.get("trace_id"),
        run_id=data.get("run_id"),
        durable=False,
    )


def _maybe_emit_mem_pressure(
    args: argparse.Namespace,
    *,
    last_mem_check: float,
    last_mem_emit: float,
    mem_min_kb: float,
) -> Tuple[float, float]:
    now = time.time()
    if not args.emit_bus or args.mem_check_s <= 0:
        return last_mem_check, last_mem_emit
    if (now - last_mem_check) < args.mem_check_s:
        return last_mem_check, last_mem_emit
    last_mem_check = now

    meminfo = _read_meminfo()
    if not _memory_pressure(
        meminfo,
        min_available_kb=int(mem_min_kb),
        min_available_ratio=max(0.0, args.mem_available_min_ratio),
    ):
        return last_mem_check, last_mem_emit
    if (now - last_mem_emit) < args.mem_emit_cooldown_s:
        return last_mem_check, last_mem_emit
    last_mem_emit = now

    available_kb = _mem_available_kb(meminfo)
    total_kb = _mem_total_kb(meminfo)
    ratio = (available_kb / total_kb) if total_kb > 0 else 0.0
    _emit(
        args.bus_dir,
        "qa.anomaly.detected",
        {
            "topic": "system.memory.pressure",
            "actor": "system",
            "severity": 4,
            "entropy_score": 0.0,
            "rate": 0.0,
            "fingerprint": "system.memory.pressure",
            "reasons": ["memory_pressure"],
            "available_kb": available_kb,
            "total_kb": total_kb,
            "available_ratio": round(ratio, 4),
            "min_available_mb": args.mem_available_min_mb,
            "min_available_ratio": args.mem_available_min_ratio,
        },
        kind="metric",
        level="warn",
    )
    return last_mem_check, last_mem_emit


def _action_plan(topic: str, actor: str, *, count: int, span_s: float, window_s: int) -> Dict[str, Any]:
    action_id = f"qa-action-{uuid.uuid4().hex[:10]}"
    summary = f"Recurring {topic} ({count}x/{window_s}s, span {int(span_s)}s)"
    actions: List[Dict[str, Any]] = []

    if topic.startswith("telemetry.client.") or actor in ("dashboard-telemetry", "telemetry"):
        actions.append({
            "id": "telemetry.reduce",
            "label": "Reduce client telemetry noise",
            "summary": "Lower telemetry sample rate or tighten burst caps in error-collector.",
            "risk": "low",
        })
    elif "bus-bridge" in actor or "bus-bridge" in topic:
        actions.append({
            "id": "restart.bus-bridge",
            "label": "Restart bus-bridge service",
            "command": "systemctl restart pluribus-bus-bridge",
            "requires_root": True,
            "risk": "medium",
        })
    elif "dashboard" in actor or topic.startswith("dashboard."):
        actions.append({
            "id": "restart.dashboard",
            "label": "Restart dashboard service",
            "command": "systemctl restart pluribus-dashboard",
            "requires_root": True,
            "risk": "medium",
        })
    elif topic.startswith("provider.") or "vps" in topic:
        actions.append({
            "id": "restart.vps-session-daemon",
            "label": "Restart VPS session daemon",
            "command": "systemctl restart pluribus-vps-session-daemon",
            "requires_root": True,
            "risk": "medium",
        })
    elif topic.startswith("browser.") or "browser" in actor:
        actions.append({
            "id": "restart.browser-session-daemon",
            "label": "Restart browser session daemon",
            "command": "systemctl restart pluribus-browser-session-daemon",
            "requires_root": True,
            "risk": "medium",
        })

    if not actions:
        actions.append({
            "id": "inspect.logs",
            "label": "Inspect recent events + logs for fingerprint",
            "summary": "Review recent bus events and system logs to pinpoint the repeating error.",
            "risk": "low",
        })

    return {
        "id": action_id,
        "summary": summary,
        "requires_approval": True,
        "actions": actions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="QA observer daemon")
    parser.add_argument("--bus-dir", default="/pluribus/.pluribus/bus")
    parser.add_argument("--output-dir", default=_default_output_dir())
    parser.add_argument("--poll", type=float, default=0.25)
    parser.add_argument("--sample-rate", type=float, default=_env_float("QA_SAMPLE_RATE", 1.0))
    parser.add_argument("--rate-window-s", type=int, default=60)
    parser.add_argument("--rate-threshold", type=float, default=5.0)
    parser.add_argument("--entropy-threshold", type=float, default=0.85)
    parser.add_argument("--emit-bus", action="store_true")
    parser.add_argument("--ignore-topic", action="append", default=["qa."])
    parser.add_argument("--scope-prefix", action="append", default=[])
    parser.add_argument("--action-window-s", type=int, default=_env_int("QA_ACTION_WINDOW_S", 900))
    parser.add_argument("--action-min-count", type=int, default=_env_int("QA_ACTION_MIN_COUNT", 3))
    parser.add_argument("--action-min-span-s", type=int, default=_env_int("QA_ACTION_MIN_SPAN_S", 120))
    parser.add_argument("--action-min-severity", type=int, default=_env_int("QA_ACTION_MIN_SEVERITY", 3))
    parser.add_argument("--action-cooldown-s", type=int, default=_env_int("QA_ACTION_COOLDOWN_S", 600))
    parser.add_argument("--action-review", dest="action_review", action="store_true", default=True)
    parser.add_argument("--no-action-review", dest="action_review", action="store_false")
    parser.add_argument("--max-bus-bytes", type=int, default=_env_int("QA_MAX_BUS_BYTES", 524288000))
    parser.add_argument("--rotate-seconds", type=int, default=_env_int("QA_ROTATE_SECONDS", 86400))
    parser.add_argument("--rotate-check-s", type=float, default=_env_float("QA_ROTATE_CHECK_S", 30.0))
    parser.add_argument("--archive-dir", default=os.environ.get("QA_ARCHIVE_DIR"))
    parser.add_argument("--mem-check-s", type=float, default=_env_float("QA_MEM_CHECK_S", 0.0))
    parser.add_argument("--mem-available-min-mb", type=float, default=_env_float("QA_MEM_AVAILABLE_MIN_MB", 0.0))
    parser.add_argument("--mem-available-min-ratio", type=float, default=_env_float("QA_MEM_AVAILABLE_MIN_RATIO", 0.0))
    parser.add_argument("--mem-emit-cooldown-s", type=float, default=_env_float("QA_MEM_EMIT_COOLDOWN_S", 60.0))
    args = parser.parse_args()

    store = QAIrStore(args.output_dir)
    tracker = RateTracker(args.rate_window_s)
    recurrent = RecurrenceTracker(args.action_window_s)
    last_emit: Dict[str, float] = {}
    last_action_emit: Dict[str, float] = {}
    last_hygiene_emit: Dict[str, float] = {}  # Cooldown for hygiene recommendations
    include_prefixes = [p for p in (args.scope_prefix + _env_list("QA_SCOPE_PREFIXES")) if p]
    ignore_prefixes = [p for p in (args.ignore_topic + _env_list("QA_IGNORE_TOPICS")) if p]
    action_include = [p for p in _env_list("QA_ACTION_SCOPE_PREFIXES") if p] or include_prefixes
    action_ignore = [p for p in _env_list("QA_ACTION_IGNORE_TOPICS") if p]
    if not action_ignore:
        action_ignore = ["telemetry.client."]

    paths = resolve_bus_paths(args.bus_dir)
    archive_dir = args.archive_dir or os.path.join(paths.bus_dir, "archive")
    _ensure_file(paths.events_path)
    
    last_rotate_check = 0.0
    last_mem_check = 0.0
    last_mem_emit = 0.0
    mem_min_kb = max(0.0, args.mem_available_min_mb) * 1024.0

    consumer = BusConsumer(Path(paths.events_path))

    def process_event(raw):
        nonlocal last_mem_check, last_mem_emit
        
        topic = raw.get("topic") or "unknown"
        if args.sample_rate < 1.0 and random.random() > args.sample_rate:
            return

        ir = normalize_event(raw, source="bus")
        store.append(ir)
        if not _in_scope(topic, include_prefixes, ignore_prefixes):
            return
        
        try:
            event_ts = float(ir["ts"])
        except (ValueError, TypeError):
            event_ts = time.time()

        rate = tracker.add(topic, event_ts)
        recur_count, recur_span = recurrent.add(ir["fingerprint"], event_ts)

        reasons = []
        if ir["severity"] >= 4:
            reasons.append("error_level")
        if rate >= args.rate_threshold:
            reasons.append("rate_spike")
        if ir["severity"] >= 4 and ir["entropy_score"] >= args.entropy_threshold: # Only entropy spike on errors
            reasons.append("entropy_spike")

        now = time.time()
        last = last_emit.get(topic, 0)
        if reasons and (now - last) > args.rate_window_s:
            last_emit[topic] = now
            if args.emit_bus:
                _emit(
                    args.bus_dir,
                    "qa.anomaly.detected",
                    {
                        "topic": topic,
                        "actor": ir["actor"],
                        "severity": ir["severity"],
                        "entropy_score": ir["entropy_score"],
                        "rate": round(rate, 3),
                        "fingerprint": ir["fingerprint"],
                        "reasons": reasons,
                    },
                )

        # Hygiene recommendations with cooldown (prevent spam)
        if rate >= args.rate_threshold and args.emit_bus:
            hygiene_last = last_hygiene_emit.get(topic, 0)
            hygiene_cooldown = args.rate_window_s * 5  # 5x window = 5 min default
            if (now - hygiene_last) >= hygiene_cooldown:
                last_hygiene_emit[topic] = now
                _emit(
                    args.bus_dir,
                    "qa.hygiene.recommendation",
                    {
                        "topic": topic,
                        "action": "sample",
                        "rate": round(rate, 3),
                        "window_s": args.rate_window_s,
                        "cooldown_s": hygiene_cooldown,
                    },
                )

        if (
            args.emit_bus
            and args.action_review
            and ir["severity"] >= args.action_min_severity
            and recur_count >= args.action_min_count
            and recur_span >= args.action_min_span_s
            and _in_scope(topic, action_include, action_ignore)
        ):
            last_action = last_action_emit.get(ir["fingerprint"], 0)
            if now - last_action >= args.action_cooldown_s:
                last_action_emit[ir["fingerprint"]] = now
                plan = _action_plan(
                    topic,
                    ir["actor"],
                    count=recur_count,
                    span_s=recur_span,
                    window_s=args.action_window_s,
                )
                _emit(
                    args.bus_dir,
                    "qa.action.review.request",
                    {
                        "topic": topic,
                        "actor": ir["actor"],
                        "severity": ir["severity"],
                        "entropy_score": ir["entropy_score"],
                        "rate": round(rate, 3),
                        "fingerprint": ir["fingerprint"],
                        "criteria": {
                            "window_s": args.action_window_s,
                            "min_count": args.action_min_count,
                            "min_span_s": args.action_min_span_s,
                            "min_severity": args.action_min_severity,
                            "count": recur_count,
                            "span_s": round(recur_span, 2),
                        },
                        "analysis": "Recurring in-scope event exceeded action thresholds (non-outlier).",
                        "action_plan": plan,
                        "sample": {
                            "level": ir["level"],
                            "kind": ir["kind"],
                        },
                    },
                    kind="request",
                    level="warn" if ir["severity"] >= 4 else "info",
                )
        
        last_mem_check, last_mem_emit = _maybe_emit_mem_pressure(
            args,
            last_mem_check=last_mem_check,
            last_mem_emit=last_mem_emit,
            mem_min_kb=mem_min_kb,
        )

    def idle_callback():
        nonlocal last_rotate_check, last_mem_check, last_mem_emit
        now = time.time()
        
        # Periodic memory check
        last_mem_check, last_mem_emit = _maybe_emit_mem_pressure(
            args,
            last_mem_check=last_mem_check,
            last_mem_emit=last_mem_emit,
            mem_min_kb=mem_min_kb,
        )

        # Periodic rotation check
        if args.rotate_check_s > 0 and (now - last_rotate_check) >= args.rotate_check_s:
            last_rotate_check = now
            try:
                stats = os.stat(paths.events_path)
                if should_rotate(stats, now=now, max_bytes=args.max_bus_bytes, max_age_s=args.rotate_seconds):
                    result = rotate_bus_file(paths.events_path, archive_dir)
                    if result and args.emit_bus:
                        _emit(
                            args.bus_dir,
                            "qa.hygiene.rotated",
                            {
                                "events_path": paths.events_path,
                                "archive_path": result["archive_path"],
                                "bytes": result["bytes"],
                                "max_bytes": args.max_bus_bytes,
                                "rotate_seconds": args.rotate_seconds,
                            },
                        )
                    # Note: rotation (inode change) is handled by consumer.tail internally
            except Exception as exc:
                if args.emit_bus:
                    _emit(
                        args.bus_dir,
                        "qa.hygiene.rotation_failed",
                        {
                            "events_path": paths.events_path,
                            "error": str(exc),
                        },
                    )

    consumer.tail(callback=process_event, idle_callback=idle_callback, poll_s=args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
