#!/usr/bin/env python3
from __future__ import annotations

"""
PBHYGIENE — System Hygiene Operator (DKIN v17)

Hygiene audit and cleanup coordination to prevent:
- Context window saturation
- Unbounded log growth
- Repository bloat
- Agent crashes from large file ingestion

Usage:
    python3 nucleus/tools/pbhygiene_operator.py --audit
    python3 nucleus/tools/pbhygiene_operator.py --rotate-bus
    python3 nucleus/tools/pbhygiene_operator.py --prune-logs --dry-run
    python3 nucleus/tools/pbhygiene_operator.py --clean --confirm
    python3 nucleus/tools/pbhygiene_operator.py --pre-pblock-check
"""

import argparse
import gzip
import json
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

# === Configuration ===
HYGIENE_PROTOCOL_VERSION = 17

# Size thresholds
MAX_BUS_SIZE_MB = 100
MAX_LOG_SIZE_MB = 50
MAX_FILE_SIZE_KB = 500
MAX_CONTEXT_MB = 10
LARGE_FILE_THRESHOLD_MB = 1
BUS_MIN_RETAIN_MB_DEFAULT = 100.0
BUS_RETAIN_MB_DEFAULT = 100.0
BUS_ROTATE_MB_DEFAULT = 100.0
BUS_ROTATE_HEADROOM_MB_DEFAULT = 0.0
BUS_PARTITION_DEFAULT = "topics"

# Age thresholds
MAX_BUS_AGE_DAYS = 7
MAX_LOG_AGE_DAYS = 3

# Load average thresholds
LOAD_CRITICAL_THRESHOLD = 20.0  # Any load avg > 20 triggers kill
LOAD_TREND_ACCELERATION = 1.5   # 1m > 5m*1.5 AND 5m > 15m*1.5 = accelerating
LOAD_HIGH_THRESHOLD = 10.0      # Warning level

# Killzombies script path
KILLZOMBIES_PATH = Path("/pluribus/killzombies.zsh")

# Paths
DEFAULT_BUS_DIR = Path("/var/lib/pluribus/.pluribus/bus")
DEFAULT_REPO_ROOT = Path("/pluribus")
HYGIENE_STATE_FILE = Path("/var/lib/pluribus/.pluribus/hygiene_state.json")

# Skip patterns for context safety
SKIP_PATTERNS = [
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    "dist",
    "build",
    "target",
    "coverage",
    "LOST_FOUND",
    ".pluribus",
    ".pluribus_local",
]

SKIP_EXTENSIONS = [
    ".ndjson", ".log", ".lock", ".min.js", ".min.css",
    ".wasm", ".node", ".so", ".dylib", ".pyc", ".pyo",
    ".zip", ".tar.gz", ".7z", ".png", ".jpg", ".jpeg",
    ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot",
]


@dataclass
class LoadMetrics:
    """System load average metrics."""
    load_1m: float = 0.0
    load_5m: float = 0.0
    load_15m: float = 0.0
    above_threshold: bool = False
    trend_critical: bool = False
    trend_reason: str = ""


@dataclass
class HygieneReport:
    """Hygiene audit report."""
    timestamp_iso: str = ""
    protocol_version: int = HYGIENE_PROTOCOL_VERSION

    # System metrics
    uptime: str = ""
    load: LoadMetrics = field(default_factory=LoadMetrics)

    # Bus metrics
    bus_size_mb: float = 0.0
    bus_event_count: int = 0
    bus_oldest_event_hours: float = 0.0
    bus_needs_rotation: bool = False
    bus_partition_mb: float = 0.0
    bus_partition_files: int = 0
    bus_partition_needs_rotation: bool = False

    # Log metrics
    log_files_count: int = 0
    log_files_total_mb: float = 0.0
    stale_logs_count: int = 0
    stale_logs: list[str] = field(default_factory=list)

    # Large file metrics
    large_files_count: int = 0
    large_files: list[dict] = field(default_factory=list)

    # Skip pattern violations
    violations: list[str] = field(default_factory=list)

    # Overall status
    status: str = "unknown"  # clean, warning, critical
    recommendations: list[str] = field(default_factory=list)


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def get_uptime() -> str:
    """Get system uptime from /proc/uptime or uptime command."""
    import subprocess
    try:
        result = subprocess.run(["uptime"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    # Fallback to /proc/uptime
    try:
        with open("/proc/uptime", "r") as f:
            uptime_seconds = float(f.read().split()[0])
            days = int(uptime_seconds // 86400)
            hours = int((uptime_seconds % 86400) // 3600)
            mins = int((uptime_seconds % 3600) // 60)
            return f"up {days} days, {hours}:{mins:02d}"
    except Exception:
        return "unknown"


def get_load_averages() -> tuple[float, float, float]:
    """Get 1, 5, 15 minute load averages from /proc/loadavg."""
    try:
        with open("/proc/loadavg", "r") as f:
            parts = f.read().strip().split()
            return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return 0.0, 0.0, 0.0


def detect_load_trend() -> LoadMetrics:
    """Detect load average trends and determine if action needed.

    Returns LoadMetrics with:
    - above_threshold: True if any load avg > LOAD_CRITICAL_THRESHOLD (20)
    - trend_critical: True if load is accelerating rapidly
    - trend_reason: Human-readable explanation of the issue
    """
    load_1m, load_5m, load_15m = get_load_averages()

    metrics = LoadMetrics(
        load_1m=load_1m,
        load_5m=load_5m,
        load_15m=load_15m,
    )

    reasons = []

    # Check absolute threshold (>20 is critical)
    if load_1m > LOAD_CRITICAL_THRESHOLD:
        metrics.above_threshold = True
        reasons.append(f"load_1m={load_1m:.1f} > {LOAD_CRITICAL_THRESHOLD}")
    if load_5m > LOAD_CRITICAL_THRESHOLD:
        metrics.above_threshold = True
        reasons.append(f"load_5m={load_5m:.1f} > {LOAD_CRITICAL_THRESHOLD}")
    if load_15m > LOAD_CRITICAL_THRESHOLD:
        metrics.above_threshold = True
        reasons.append(f"load_15m={load_15m:.1f} > {LOAD_CRITICAL_THRESHOLD}")

    # Check for accelerating trend (rapid increase)
    # If 1m > 5m * 1.5 AND 5m > 15m * 1.5, load is accelerating dangerously
    if load_5m > 0 and load_15m > 0:
        accel_1_to_5 = load_1m > load_5m * LOAD_TREND_ACCELERATION
        accel_5_to_15 = load_5m > load_15m * LOAD_TREND_ACCELERATION

        if accel_1_to_5 and accel_5_to_15 and load_1m > LOAD_HIGH_THRESHOLD:
            metrics.trend_critical = True
            reasons.append(
                f"accelerating trend: {load_15m:.1f} → {load_5m:.1f} → {load_1m:.1f}"
            )

    # Also flag if 1m is dramatically higher than 15m (sudden spike)
    if load_15m > 0 and load_1m > load_15m * 2.0 and load_1m > LOAD_HIGH_THRESHOLD:
        metrics.trend_critical = True
        if "accelerating" not in " ".join(reasons):
            reasons.append(f"sudden spike: {load_1m:.1f} vs 15m avg {load_15m:.1f}")

    metrics.trend_reason = "; ".join(reasons) if reasons else "stable"

    return metrics


def run_killzombies(kill: bool = False, emit_bus_flag: bool = False) -> dict:
    """Run killzombies.zsh script.

    Args:
        kill: If True, actually kill processes (otherwise dry-run)
        emit_bus_flag: If True, emit bus events

    Returns:
        dict with results from killzombies.zsh
    """
    import subprocess

    if not KILLZOMBIES_PATH.exists():
        return {"success": False, "error": "killzombies.zsh not found"}

    cmd = ["/usr/bin/zsh", str(KILLZOMBIES_PATH), "--json"]
    if kill:
        cmd.append("--kill")
    if emit_bus_flag:
        cmd.append("--emit-bus")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(DEFAULT_REPO_ROOT),
        )

        output = result.stdout.strip()
        if output:
            # Parse last JSON line (may have multiple outputs)
            for line in reversed(output.split("\n")):
                line = line.strip()
                if line.startswith("{"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        pass

        return {
            "success": result.returncode == 0,
            "stdout": output,
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rotation_limits() -> tuple[int, int]:
    try:
        retain_mb = float(os.environ.get("PLURIBUS_BUS_RETAIN_MB") or BUS_RETAIN_MB_DEFAULT)
    except (TypeError, ValueError):
        retain_mb = BUS_RETAIN_MB_DEFAULT
    try:
        min_retain_mb = float(os.environ.get("PLURIBUS_BUS_MIN_RETAIN_MB") or BUS_MIN_RETAIN_MB_DEFAULT)
    except (TypeError, ValueError):
        min_retain_mb = BUS_MIN_RETAIN_MB_DEFAULT
    try:
        rotate_mb = float(os.environ.get("PLURIBUS_BUS_ROTATE_MB") or BUS_ROTATE_MB_DEFAULT)
    except (TypeError, ValueError):
        rotate_mb = BUS_ROTATE_MB_DEFAULT
    try:
        headroom_mb = float(os.environ.get("PLURIBUS_BUS_HEADROOM_MB") or BUS_ROTATE_HEADROOM_MB_DEFAULT)
    except (TypeError, ValueError):
        headroom_mb = BUS_ROTATE_HEADROOM_MB_DEFAULT

    retain_mb = max(retain_mb, min_retain_mb)
    rotate_floor = retain_mb + max(0.0, headroom_mb)
    rotate_mb = max(rotate_mb, rotate_floor)

    return int(rotate_mb * 1024 * 1024), int(retain_mb * 1024 * 1024)


def _partition_dir(bus_dir: Path) -> Path:
    name = os.environ.get("PLURIBUS_BUS_PARTITION_DIR", BUS_PARTITION_DEFAULT)
    return bus_dir / name


def rotate_log_tail(path: Path, *, retain_bytes: int, archive_dir: Path) -> dict:
    if retain_bytes <= 0:
        return {"rotated": False, "reason": "retain_bytes<=0"}
    if not path.exists():
        return {"rotated": False, "reason": "missing"}
    size_bytes = path.stat().st_size
    if size_bytes <= retain_bytes:
        return {"rotated": False, "reason": "below_retain"}

    ensure_dir(archive_dir)
    archive_name = f"{path.stem}-{now_iso_utc().replace(':', '')}.ndjson.gz"
    archive_path = archive_dir / archive_name

    with path.open("rb") as f_in:
        with gzip.open(archive_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    with path.open("rb") as f_in:
        if size_bytes > retain_bytes:
            f_in.seek(size_bytes - retain_bytes)
        tail = f_in.read()
    path.write_bytes(tail)

    return {"rotated": True, "archived_to": str(archive_path), "archived_bytes": size_bytes}


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def emit_bus(bus_dir: Path, *, topic: str, kind: str, level: str, actor: str, data: dict) -> None:
    evt = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    append_ndjson(bus_dir / "events.ndjson", evt)


def get_file_age_hours(path: Path) -> float:
    """Get file age in hours."""
    try:
        mtime = path.stat().st_mtime
        return (time.time() - mtime) / 3600
    except OSError:
        return 0.0


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    try:
        return path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def count_ndjson_lines(path: Path) -> int:
    """Count lines in NDJSON file."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def should_skip_path(path: Path) -> bool:
    """Check if path matches skip patterns."""
    path_str = str(path)
    for pattern in SKIP_PATTERNS:
        if pattern in path_str:
            return True
    return path.suffix.lower() in SKIP_EXTENSIONS


def audit_bus(bus_dir: Path) -> dict:
    """Audit bus events file."""
    events_path = bus_dir / "events.ndjson"
    partition_dir = _partition_dir(bus_dir)
    rotate_bytes, _retain_bytes = _rotation_limits()
    result = {
        "exists": events_path.exists(),
        "size_mb": 0.0,
        "event_count": 0,
        "oldest_event_hours": 0.0,
        "needs_rotation": False,
        "partition_size_mb": 0.0,
        "partition_files": 0,
        "partition_needs_rotation": False,
    }

    if not events_path.exists():
        return result

    result["size_mb"] = get_file_size_mb(events_path)
    result["event_count"] = count_ndjson_lines(events_path)
    result["oldest_event_hours"] = get_file_age_hours(events_path)

    # Check rotation thresholds
    if result["size_mb"] > MAX_BUS_SIZE_MB:
        result["needs_rotation"] = True
    if result["oldest_event_hours"] > MAX_BUS_AGE_DAYS * 24:
        result["needs_rotation"] = True

    if partition_dir.exists():
        total_bytes = 0
        files = 0
        for path in partition_dir.rglob("*.ndjson"):
            if not path.is_file():
                continue
            files += 1
            size_bytes = path.stat().st_size
            total_bytes += size_bytes
            if size_bytes > rotate_bytes:
                result["partition_needs_rotation"] = True
        result["partition_files"] = files
        result["partition_size_mb"] = round(total_bytes / (1024 * 1024), 2)

    return result


def audit_logs(repo_root: Path) -> dict:
    """Audit log files in repository."""
    log_files = []
    stale_logs = []
    total_size_mb = 0.0

    for pattern in ["**/*.log", "**/*_log*"]:
        for log_path in repo_root.glob(pattern):
            if not log_path.is_file():
                continue
            if should_skip_path(log_path):
                continue

            size_mb = get_file_size_mb(log_path)
            age_hours = get_file_age_hours(log_path)
            total_size_mb += size_mb

            log_files.append({
                "path": str(log_path.relative_to(repo_root)),
                "size_mb": round(size_mb, 2),
                "age_hours": round(age_hours, 1),
            })

            if age_hours > MAX_LOG_AGE_DAYS * 24:
                stale_logs.append(str(log_path.relative_to(repo_root)))

    return {
        "count": len(log_files),
        "total_size_mb": round(total_size_mb, 2),
        "stale_count": len(stale_logs),
        "stale_logs": stale_logs[:20],  # Limit output
    }


def audit_large_files(repo_root: Path) -> dict:
    """Find large files that might bloat context."""
    large_files = []

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if should_skip_path(path):
            continue
        if ".git" in str(path):
            continue

        size_mb = get_file_size_mb(path)
        if size_mb > LARGE_FILE_THRESHOLD_MB:
            large_files.append({
                "path": str(path.relative_to(repo_root)),
                "size_mb": round(size_mb, 2),
            })

    # Sort by size descending
    large_files.sort(key=lambda x: x["size_mb"], reverse=True)

    return {
        "count": len(large_files),
        "files": large_files[:30],  # Top 30
    }


def run_audit(bus_dir: Path, repo_root: Path, auto_kill_on_load: bool = False) -> HygieneReport:
    """Run full hygiene audit.

    Args:
        bus_dir: Path to bus directory
        repo_root: Path to repository root
        auto_kill_on_load: If True, automatically run killzombies when load is critical
    """
    report = HygieneReport(timestamp_iso=now_iso_utc())

    # System uptime
    report.uptime = get_uptime()

    # System load averages and trend detection
    report.load = detect_load_trend()

    # Audit bus
    bus_result = audit_bus(bus_dir)
    report.bus_size_mb = bus_result["size_mb"]
    report.bus_event_count = bus_result["event_count"]
    report.bus_oldest_event_hours = bus_result["oldest_event_hours"]
    report.bus_needs_rotation = bus_result["needs_rotation"]
    report.bus_partition_mb = bus_result["partition_size_mb"]
    report.bus_partition_files = bus_result["partition_files"]
    report.bus_partition_needs_rotation = bus_result["partition_needs_rotation"]

    # Audit logs
    log_result = audit_logs(repo_root)
    report.log_files_count = log_result["count"]
    report.log_files_total_mb = log_result["total_size_mb"]
    report.stale_logs_count = log_result["stale_count"]
    report.stale_logs = log_result["stale_logs"]

    # Audit large files
    large_result = audit_large_files(repo_root)
    report.large_files_count = large_result["count"]
    report.large_files = large_result["files"]

    # Determine status and recommendations
    issues = []

    if report.bus_needs_rotation:
        issues.append("critical")
        report.recommendations.append(f"Bus rotation needed: {report.bus_size_mb:.1f}MB / {report.bus_event_count} events")
    if report.bus_partition_needs_rotation:
        issues.append("warning")
        report.recommendations.append(
            f"Bus partition rotation needed: {report.bus_partition_mb:.1f}MB across {report.bus_partition_files} files"
        )

    if report.stale_logs_count > 0:
        issues.append("warning")
        report.recommendations.append(f"Prune {report.stale_logs_count} stale log files")

    if report.log_files_total_mb > MAX_LOG_SIZE_MB:
        issues.append("warning")
        report.recommendations.append(f"Log files total {report.log_files_total_mb:.1f}MB exceeds threshold")

    if report.large_files_count > 10:
        issues.append("warning")
        report.recommendations.append(f"Found {report.large_files_count} large files (>1MB)")

    # Load average checks
    load_needs_action = False
    if report.load.above_threshold:
        issues.append("critical")
        report.recommendations.append(
            f"LOAD CRITICAL: {report.load.trend_reason}"
        )
        load_needs_action = True
    elif report.load.trend_critical:
        issues.append("critical")
        report.recommendations.append(
            f"LOAD TREND CRITICAL: {report.load.trend_reason}"
        )
        load_needs_action = True
    elif report.load.load_1m > LOAD_HIGH_THRESHOLD:
        issues.append("warning")
        report.recommendations.append(
            f"Load elevated: {report.load.load_1m:.1f}/{report.load.load_5m:.1f}/{report.load.load_15m:.1f}"
        )

    # Auto-kill runaway processes if load is critical
    if load_needs_action and auto_kill_on_load:
        kill_result = run_killzombies(kill=True, emit_bus_flag=True)
        if kill_result.get("killed_count", 0) > 0:
            report.recommendations.append(
                f"AUTO-KILLED {kill_result['killed_count']} runaway processes"
            )
        elif kill_result.get("candidates", 0) == 0:
            report.recommendations.append(
                "No killable candidates found despite high load"
            )

    if "critical" in issues:
        report.status = "critical"
    elif "warning" in issues:
        report.status = "warning"
    else:
        report.status = "clean"
        report.recommendations.append("No hygiene issues detected")

    return report


def rotate_bus(bus_dir: Path, actor: str, dry_run: bool = False) -> dict:
    """Rotate bus events file."""
    events_path = bus_dir / "events.ndjson"
    archive_dir = bus_dir / "archive"
    partition_dir = _partition_dir(bus_dir)
    rotate_bytes, retain_bytes = _rotation_limits()

    if not events_path.exists():
        return {"success": False, "error": "events.ndjson not found"}

    size_mb = get_file_size_mb(events_path)
    event_count = count_ndjson_lines(events_path)

    archive_name = f"events-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.ndjson.gz"
    archive_path = archive_dir / archive_name

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "would_archive": str(archive_path),
            "size_mb": size_mb,
            "event_count": event_count,
            "retain_mb": round(retain_bytes / (1024 * 1024), 2),
            "rotate_mb": round(rotate_bytes / (1024 * 1024), 2),
        }

    ensure_dir(archive_dir)

    rotated = False
    archived_to = None
    if events_path.stat().st_size > rotate_bytes:
        result = rotate_log_tail(events_path, retain_bytes=retain_bytes, archive_dir=archive_dir)
        if result.get("rotated"):
            rotated = True
            archived_to = result.get("archived_to")

    # Emit rotation event
    emit_bus(bus_dir, topic="operator.pbhygiene.rotate", kind="request", level="info", actor=actor, data={
        "status": "rotated" if rotated else "skipped",
        "archived_to": archived_to,
        "archived_size_mb": size_mb,
        "archived_event_count": event_count,
        "retain_mb": round(retain_bytes / (1024 * 1024), 2),
        "rotate_mb": round(rotate_bytes / (1024 * 1024), 2),
        "protocol_version": HYGIENE_PROTOCOL_VERSION,
    })

    partition_results = []
    if partition_dir.exists():
        for part in partition_dir.rglob("*.ndjson"):
            if not part.is_file():
                continue
            if part.stat().st_size <= rotate_bytes:
                continue
            try:
                rel_dir = part.parent.relative_to(partition_dir)
            except ValueError:
                rel_dir = Path()
            part_archive = archive_dir / "partitions" / rel_dir
            result = rotate_log_tail(part, retain_bytes=retain_bytes, archive_dir=part_archive)
            if result.get("rotated"):
                partition_results.append({"path": str(part), "archived_to": result.get("archived_to")})

    return {
        "success": True,
        "archived_to": archived_to,
        "size_mb": size_mb,
        "event_count": event_count,
        "rotated": rotated,
        "partition_rotations": partition_results,
    }


def prune_logs(repo_root: Path, bus_dir: Path, actor: str, dry_run: bool = False) -> dict:
    """Prune stale log files."""
    pruned = []
    skipped = []

    log_result = audit_logs(repo_root)

    for log_rel in log_result["stale_logs"]:
        log_path = repo_root / log_rel
        if not log_path.exists():
            continue

        if dry_run:
            pruned.append({"path": log_rel, "action": "would_delete"})
        else:
            try:
                log_path.unlink()
                pruned.append({"path": log_rel, "action": "deleted"})
            except OSError as e:
                skipped.append({"path": log_rel, "error": str(e)})

    if not dry_run and pruned:
        emit_bus(bus_dir, topic="operator.pbhygiene.prune", kind="request", level="info", actor=actor, data={
            "pruned_count": len(pruned),
            "skipped_count": len(skipped),
            "protocol_version": HYGIENE_PROTOCOL_VERSION,
        })

    return {
        "success": True,
        "dry_run": dry_run,
        "pruned": pruned,
        "skipped": skipped,
    }


def pre_pblock_check(bus_dir: Path, repo_root: Path) -> dict:
    """Check hygiene before entering PBLOCK."""
    report = run_audit(bus_dir, repo_root)

    blockers = []
    warnings = []

    if report.bus_size_mb > MAX_BUS_SIZE_MB:
        blockers.append(f"Bus size {report.bus_size_mb:.1f}MB exceeds {MAX_BUS_SIZE_MB}MB limit")

    if report.bus_partition_needs_rotation:
        warnings.append("Bus partition logs exceed rotation threshold")

    if report.stale_logs_count > 10:
        warnings.append(f"{report.stale_logs_count} stale log files need pruning")

    if report.large_files_count > 20:
        warnings.append(f"{report.large_files_count} large files may bloat context")

    can_enter_pblock = len(blockers) == 0

    return {
        "can_enter_pblock": can_enter_pblock,
        "blockers": blockers,
        "warnings": warnings,
        "report_status": report.status,
        "recommendations": report.recommendations,
    }


def format_report(report: HygieneReport) -> str:
    """Format hygiene report for display."""
    status_icons = {"clean": "✅", "warning": "⚠️", "critical": "🚨"}
    icon = status_icons.get(report.status, "❓")

    uptime_short = report.uptime[:58] if report.uptime else "unknown"

    # Load status indicator
    load_icon = "✅"
    if report.load.above_threshold or report.load.trend_critical:
        load_icon = "🔥"
    elif report.load.load_1m > LOAD_HIGH_THRESHOLD:
        load_icon = "⚠️"

    lines = [
        "╔══════════════════════════════════════════════════════════════════════╗",
        f"║  {icon} PBHYGIENE AUDIT — DKIN v{report.protocol_version}                                   ║",
        "╠══════════════════════════════════════════════════════════════════════╣",
        f"║  Timestamp: {report.timestamp_iso:40s}         ║",
        f"║  Uptime: {uptime_short:59s}║",
        f"║  Status: {report.status.upper():10s}                                              ║",
        "╠══════════════════════════════════════════════════════════════════════╣",
        f"║  {load_icon} LOAD AVERAGE:                                                     ║",
        f"║    1min: {report.load.load_1m:6.2f}  5min: {report.load.load_5m:6.2f}  15min: {report.load.load_15m:6.2f}              ║",
        f"║    Trend: {report.load.trend_reason[:56]:56s}     ║",
        "╠══════════════════════════════════════════════════════════════════════╣",
        "║  BUS METRICS:                                                        ║",
        f"║    Size: {report.bus_size_mb:8.1f} MB ({report.bus_event_count} events)                          ║",
        f"║    Age: {report.bus_oldest_event_hours:8.1f} hours                                        ║",
        f"║    Partitions: {report.bus_partition_files:5d} ({report.bus_partition_mb:6.1f} MB)                             ║",
        f"║    Needs rotation: {'YES' if report.bus_needs_rotation else 'NO':5s}                                         ║",
        "╠══════════════════════════════════════════════════════════════════════╣",
        "║  LOG METRICS:                                                        ║",
        f"║    Log files: {report.log_files_count:5d} ({report.log_files_total_mb:.1f} MB total)                            ║",
        f"║    Stale logs: {report.stale_logs_count:5d}                                                ║",
        "╠══════════════════════════════════════════════════════════════════════╣",
        "║  LARGE FILES:                                                        ║",
        f"║    Count (>1MB): {report.large_files_count:5d}                                             ║",
    ]

    if report.large_files[:5]:
        for lf in report.large_files[:5]:
            path_short = lf["path"][:50]
            lines.append(f"║      {lf['size_mb']:6.1f}MB  {path_short:50s} ║")

    lines.append("╠══════════════════════════════════════════════════════════════════════╣")
    lines.append("║  RECOMMENDATIONS:                                                    ║")

    for rec in report.recommendations[:5]:
        rec_short = rec[:62]
        lines.append(f"║    • {rec_short:62s} ║")

    lines.append("╚══════════════════════════════════════════════════════════════════════╝")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pbhygiene_operator.py",
        description="PBHYGIENE — System Hygiene Operator (DKIN v17)",
    )
    p.add_argument("--bus-dir", default=None, help="Bus directory")
    p.add_argument("--repo-root", default=None, help="Repository root")
    p.add_argument("--actor", default=None, help="Actor identity")

    action = p.add_mutually_exclusive_group()
    action.add_argument("--audit", action="store_true", help="Run hygiene audit")
    action.add_argument("--rotate-bus", action="store_true", help="Rotate bus events")
    action.add_argument("--prune-logs", action="store_true", help="Prune stale logs")
    action.add_argument("--clean", action="store_true", help="Full cleanup (audit + rotate + prune)")
    action.add_argument("--pre-pblock-check", action="store_true", help="Check hygiene before PBLOCK")

    p.add_argument("--dry-run", action="store_true", help="Dry run (no changes)")
    p.add_argument("--confirm", action="store_true", help="Confirm destructive operations")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.add_argument("--auto-kill", action="store_true",
                   help="Auto-kill runaway processes when load >20 or critical trend")

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    actor = args.actor or os.environ.get("PLURIBUS_ACTOR") or "pbhygiene"
    bus_dir = Path(args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or DEFAULT_BUS_DIR)
    repo_root = Path(args.repo_root or DEFAULT_REPO_ROOT)

    ensure_dir(bus_dir)

    result: dict[str, Any] = {}

    auto_kill = getattr(args, "auto_kill", False)

    if args.audit or (not any([args.rotate_bus, args.prune_logs, args.clean, args.pre_pblock_check])):
        report = run_audit(bus_dir, repo_root, auto_kill_on_load=auto_kill)
        if args.json:
            result = {
                "timestamp_iso": report.timestamp_iso,
                "uptime": report.uptime,
                "status": report.status,
                "load": {
                    "load_1m": report.load.load_1m,
                    "load_5m": report.load.load_5m,
                    "load_15m": report.load.load_15m,
                    "above_threshold": report.load.above_threshold,
                    "trend_critical": report.load.trend_critical,
                    "trend_reason": report.load.trend_reason,
                },
                "bus": {
                    "size_mb": report.bus_size_mb,
                    "event_count": report.bus_event_count,
                    "needs_rotation": report.bus_needs_rotation,
                    "partition_mb": report.bus_partition_mb,
                    "partition_files": report.bus_partition_files,
                    "partition_needs_rotation": report.bus_partition_needs_rotation,
                },
                "logs": {
                    "count": report.log_files_count,
                    "total_mb": report.log_files_total_mb,
                    "stale_count": report.stale_logs_count,
                },
                "large_files": {
                    "count": report.large_files_count,
                    "files": report.large_files[:10],
                },
                "recommendations": report.recommendations,
            }
            sys.stdout.write(json.dumps(result, indent=2) + "\n")
        else:
            sys.stdout.write(format_report(report) + "\n")

        # Emit audit event (include load metrics)
        emit_bus(bus_dir, topic="operator.pbhygiene.audit", kind="metric", level="info", actor=actor, data={
            "status": report.status,
            "bus_size_mb": report.bus_size_mb,
            "log_count": report.log_files_count,
            "large_files_count": report.large_files_count,
            "load_1m": report.load.load_1m,
            "load_5m": report.load.load_5m,
            "load_15m": report.load.load_15m,
            "load_critical": report.load.above_threshold or report.load.trend_critical,
            "protocol_version": HYGIENE_PROTOCOL_VERSION,
        })

        return 0 if report.status != "critical" else 1

    elif args.rotate_bus:
        result = rotate_bus(bus_dir, actor, dry_run=args.dry_run)

    elif args.prune_logs:
        if not args.dry_run and not args.confirm:
            sys.stderr.write("ERROR: --prune-logs requires --dry-run or --confirm\n")
            return 1
        result = prune_logs(repo_root, bus_dir, actor, dry_run=args.dry_run)

    elif args.clean:
        if not args.confirm:
            sys.stderr.write("ERROR: --clean requires --confirm\n")
            return 1
        # Run all cleanup operations
        result = {
            "audit": run_audit(bus_dir, repo_root).__dict__,
            "rotate": rotate_bus(bus_dir, actor),
            "prune": prune_logs(repo_root, bus_dir, actor),
        }

    elif args.pre_pblock_check:
        result = pre_pblock_check(bus_dir, repo_root)
        if args.json:
            sys.stdout.write(json.dumps(result, indent=2) + "\n")
        else:
            if result["can_enter_pblock"]:
                sys.stdout.write("✅ Ready for PBLOCK\n")
            else:
                sys.stdout.write("🚨 PBLOCK blocked:\n")
                for b in result["blockers"]:
                    sys.stdout.write(f"  ❌ {b}\n")
            if result["warnings"]:
                sys.stdout.write("Warnings:\n")
                for w in result["warnings"]:
                    sys.stdout.write(f"  ⚠️ {w}\n")
        return 0 if result["can_enter_pblock"] else 1

    if args.json:
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    else:
        if "error" in result:
            sys.stderr.write(f"ERROR: {result['error']}\n")
            return 1
        sys.stdout.write(f"Success: {json.dumps(result, indent=2)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
