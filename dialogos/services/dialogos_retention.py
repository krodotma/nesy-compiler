#!/usr/bin/env python3
"""
dialogos_retention.py - Retention and compression policies for dialogos trace
==============================================================================

Operations:
  - rotate: Split trace.ndjson when it exceeds max_size_mb
  - compress: Gzip traces older than compress_after_days
  - archive: Move compressed traces to archive_path
  - prune: Delete archives older than max_age_days
  - status: Show retention stats

Bus Events:
  - dialogos.retention.rotate - when trace is rotated
  - dialogos.retention.compress - when files are compressed
  - dialogos.retention.prune - when archives are deleted

Safety:
  - Never delete active trace file
  - Emit warning if trace hasn't been written in 24h
  - Keep at least 7 days of data always
"""
from __future__ import annotations

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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TRACE_DIR = Path("/pluribus/.pluribus/dialogos")
DEFAULT_TRACE_FILE = DEFAULT_TRACE_DIR / "trace.ndjson"
DEFAULT_ARCHIVE_DIR = Path("/pluribus/.pluribus/dialogos/archive")
DEFAULT_BUS_DIR = Path("/pluribus/.pluribus/bus")

# Safety: always keep at least 7 days of data
MIN_RETENTION_DAYS = 7


# ---------------------------------------------------------------------------
# RetentionPolicy dataclass
# ---------------------------------------------------------------------------
@dataclass
class RetentionPolicy:
    """Retention and compression policy for dialogos traces."""
    max_age_days: int = 30  # Archive after 30 days
    max_size_mb: int = 500  # Rotate at 500MB
    compress_after_days: int = 7  # Compress after 7 days
    archive_path: Path = field(default_factory=lambda: DEFAULT_ARCHIVE_DIR)

    def validate(self) -> list[str]:
        """Validate policy constraints, return list of warnings."""
        warnings = []
        if self.max_age_days < MIN_RETENTION_DAYS:
            warnings.append(f"max_age_days={self.max_age_days} is below minimum {MIN_RETENTION_DAYS}")
        if self.compress_after_days < 1:
            warnings.append(f"compress_after_days={self.compress_after_days} is below minimum 1")
        if self.compress_after_days > self.max_age_days:
            warnings.append(f"compress_after_days={self.compress_after_days} exceeds max_age_days={self.max_age_days}")
        if self.max_size_mb < 1:
            warnings.append(f"max_size_mb={self.max_size_mb} is below minimum 1")
        return warnings


def load_policy(policy_path: Path | None = None) -> RetentionPolicy:
    """Load policy from JSON file or return defaults."""
    if policy_path and policy_path.exists():
        try:
            data = json.loads(policy_path.read_text(encoding="utf-8"))
            archive_path = Path(data.get("archive_path", str(DEFAULT_ARCHIVE_DIR)))
            return RetentionPolicy(
                max_age_days=int(data.get("max_age_days", 30)),
                max_size_mb=int(data.get("max_size_mb", 500)),
                compress_after_days=int(data.get("compress_after_days", 7)),
                archive_path=archive_path,
            )
        except Exception:
            pass
    return RetentionPolicy()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def now_ts() -> float:
    return time.time()


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "dialogos-retention"


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0  # type: ignore
    return f"{n:.1f}PB"


def days_ago(ts: float) -> float:
    return (now_ts() - ts) / 86400.0


def file_age_days(path: Path) -> float:
    """Return file age in days based on mtime."""
    if not path.exists():
        return 0.0
    return days_ago(path.stat().st_mtime)


def file_size_mb(path: Path) -> float:
    """Return file size in MB."""
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Bus event emission
# ---------------------------------------------------------------------------
def emit_bus_event(
    bus_dir: Path,
    *,
    topic: str,
    kind: str,
    level: str,
    actor: str,
    data: dict[str, Any],
) -> str:
    """Emit event to bus (best-effort)."""
    try:
        from nucleus.tools import agent_bus
        paths = agent_bus.resolve_bus_paths(str(bus_dir))
        return agent_bus.emit_event(
            paths,
            topic=topic,
            kind=kind,
            level=level,
            actor=actor,
            data=data,
            trace_id=None,
            run_id=None,
            durable=False,
        )
    except Exception as e:
        sys.stderr.write(f"WARN: failed to emit bus event: {e}\n")
        return ""


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------
def get_rotated_files(trace_dir: Path) -> list[Path]:
    """Get list of rotated trace files (trace.ndjson.YYYYMMDD_HHMMSS)."""
    rotated = []
    if not trace_dir.exists():
        return rotated
    for f in trace_dir.iterdir():
        if f.is_file() and f.name.startswith("trace.ndjson.") and not f.name.endswith(".gz"):
            rotated.append(f)
    return sorted(rotated, key=lambda p: p.stat().st_mtime)


def get_compressed_files(trace_dir: Path) -> list[Path]:
    """Get list of compressed trace files (*.ndjson.*.gz)."""
    compressed = []
    if not trace_dir.exists():
        return compressed
    for f in trace_dir.iterdir():
        if f.is_file() and f.name.endswith(".gz") and ".ndjson" in f.name:
            compressed.append(f)
    return sorted(compressed, key=lambda p: p.stat().st_mtime)


def get_archived_files(archive_dir: Path) -> list[Path]:
    """Get list of archived files in archive directory."""
    archived = []
    if not archive_dir.exists():
        return archived
    for f in archive_dir.iterdir():
        if f.is_file():
            archived.append(f)
    return sorted(archived, key=lambda p: p.stat().st_mtime)


def rotate_trace(
    trace_path: Path,
    policy: RetentionPolicy,
    *,
    dry_run: bool = False,
    bus_dir: Path | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """Rotate trace file if it exceeds max_size_mb."""
    result: dict[str, Any] = {
        "action": "rotate",
        "dry_run": dry_run,
        "rotated": False,
        "trace_path": str(trace_path),
        "size_mb": 0.0,
        "threshold_mb": policy.max_size_mb,
        "new_file": None,
    }

    if not trace_path.exists():
        result["error"] = "trace file does not exist"
        return result

    size_mb = file_size_mb(trace_path)
    result["size_mb"] = round(size_mb, 2)

    if size_mb < policy.max_size_mb:
        result["message"] = f"size {size_mb:.1f}MB < threshold {policy.max_size_mb}MB, no rotation needed"
        return result

    # Generate rotated filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    new_name = f"trace.ndjson.{timestamp}"
    new_path = trace_path.parent / new_name
    result["new_file"] = str(new_path)

    if dry_run:
        result["message"] = f"would rotate {trace_path.name} to {new_name}"
        result["rotated"] = True
        return result

    # Perform rotation
    try:
        shutil.move(str(trace_path), str(new_path))
        # Create new empty trace file
        trace_path.touch()
        result["rotated"] = True
        result["message"] = f"rotated to {new_name}"

        # Emit bus event
        if bus_dir:
            emit_bus_event(
                bus_dir,
                topic="dialogos.retention.rotate",
                kind="metric",
                level="info",
                actor=actor or default_actor(),
                data={
                    "rotated_from": str(trace_path),
                    "rotated_to": str(new_path),
                    "size_mb": round(size_mb, 2),
                    "threshold_mb": policy.max_size_mb,
                    "timestamp": now_iso_utc(),
                },
            )
    except Exception as e:
        result["error"] = str(e)

    return result


def compress_traces(
    trace_dir: Path,
    policy: RetentionPolicy,
    *,
    dry_run: bool = False,
    bus_dir: Path | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """Compress rotated traces older than compress_after_days."""
    result: dict[str, Any] = {
        "action": "compress",
        "dry_run": dry_run,
        "compressed": [],
        "skipped": [],
        "errors": [],
        "threshold_days": policy.compress_after_days,
    }

    rotated_files = get_rotated_files(trace_dir)

    for f in rotated_files:
        age_days = file_age_days(f)
        if age_days < policy.compress_after_days:
            result["skipped"].append({
                "file": f.name,
                "age_days": round(age_days, 1),
            })
            continue

        gz_path = f.with_suffix(f.suffix + ".gz")

        if dry_run:
            result["compressed"].append({
                "file": f.name,
                "gz_file": gz_path.name,
                "age_days": round(age_days, 1),
                "size_bytes": f.stat().st_size,
            })
            continue

        # Compress the file
        try:
            with f.open("rb") as f_in:
                with gzip.open(gz_path, "wb", compresslevel=9) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            original_size = f.stat().st_size
            compressed_size = gz_path.stat().st_size
            f.unlink()  # Remove original after successful compression
            result["compressed"].append({
                "file": f.name,
                "gz_file": gz_path.name,
                "age_days": round(age_days, 1),
                "original_bytes": original_size,
                "compressed_bytes": compressed_size,
                "ratio": round(compressed_size / original_size, 2) if original_size > 0 else 0,
            })
        except Exception as e:
            result["errors"].append({
                "file": f.name,
                "error": str(e),
            })

    # Emit bus event if any files were compressed
    if result["compressed"] and bus_dir and not dry_run:
        emit_bus_event(
            bus_dir,
            topic="dialogos.retention.compress",
            kind="metric",
            level="info",
            actor=actor or default_actor(),
            data={
                "files_compressed": len(result["compressed"]),
                "total_original_bytes": sum(c.get("original_bytes", 0) for c in result["compressed"]),
                "total_compressed_bytes": sum(c.get("compressed_bytes", 0) for c in result["compressed"]),
                "threshold_days": policy.compress_after_days,
                "timestamp": now_iso_utc(),
            },
        )

    return result


def archive_traces(
    trace_dir: Path,
    policy: RetentionPolicy,
    *,
    dry_run: bool = False,
    bus_dir: Path | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """Move compressed traces to archive_path."""
    result: dict[str, Any] = {
        "action": "archive",
        "dry_run": dry_run,
        "archived": [],
        "errors": [],
        "archive_path": str(policy.archive_path),
    }

    compressed_files = get_compressed_files(trace_dir)

    if not compressed_files:
        result["message"] = "no compressed files to archive"
        return result

    if not dry_run:
        ensure_dir(policy.archive_path)

    for f in compressed_files:
        dest = policy.archive_path / f.name

        if dry_run:
            result["archived"].append({
                "file": f.name,
                "dest": str(dest),
                "size_bytes": f.stat().st_size,
            })
            continue

        try:
            shutil.move(str(f), str(dest))
            result["archived"].append({
                "file": f.name,
                "dest": str(dest),
                "size_bytes": dest.stat().st_size,
            })
        except Exception as e:
            result["errors"].append({
                "file": f.name,
                "error": str(e),
            })

    return result


def prune_archives(
    policy: RetentionPolicy,
    *,
    older_than_days: int | None = None,
    dry_run: bool = False,
    bus_dir: Path | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """Delete archives older than max_age_days (or older_than_days if specified)."""
    threshold_days = older_than_days if older_than_days is not None else policy.max_age_days

    # Safety: never prune files younger than MIN_RETENTION_DAYS
    if threshold_days < MIN_RETENTION_DAYS:
        threshold_days = MIN_RETENTION_DAYS

    result: dict[str, Any] = {
        "action": "prune",
        "dry_run": dry_run,
        "pruned": [],
        "skipped": [],
        "errors": [],
        "threshold_days": threshold_days,
        "archive_path": str(policy.archive_path),
    }

    archived_files = get_archived_files(policy.archive_path)

    for f in archived_files:
        age_days = file_age_days(f)
        if age_days < threshold_days:
            result["skipped"].append({
                "file": f.name,
                "age_days": round(age_days, 1),
            })
            continue

        if dry_run:
            result["pruned"].append({
                "file": f.name,
                "age_days": round(age_days, 1),
                "size_bytes": f.stat().st_size,
            })
            continue

        try:
            size_bytes = f.stat().st_size
            f.unlink()
            result["pruned"].append({
                "file": f.name,
                "age_days": round(age_days, 1),
                "size_bytes": size_bytes,
            })
        except Exception as e:
            result["errors"].append({
                "file": f.name,
                "error": str(e),
            })

    # Emit bus event if any files were pruned
    if result["pruned"] and bus_dir and not dry_run:
        emit_bus_event(
            bus_dir,
            topic="dialogos.retention.prune",
            kind="metric",
            level="info",
            actor=actor or default_actor(),
            data={
                "files_pruned": len(result["pruned"]),
                "total_bytes_freed": sum(p.get("size_bytes", 0) for p in result["pruned"]),
                "threshold_days": threshold_days,
                "timestamp": now_iso_utc(),
            },
        )

    return result


def get_status(
    trace_dir: Path,
    policy: RetentionPolicy,
) -> dict[str, Any]:
    """Get retention status and statistics."""
    trace_path = trace_dir / "trace.ndjson"

    status: dict[str, Any] = {
        "trace_dir": str(trace_dir),
        "archive_path": str(policy.archive_path),
        "policy": {
            "max_age_days": policy.max_age_days,
            "max_size_mb": policy.max_size_mb,
            "compress_after_days": policy.compress_after_days,
        },
        "active_trace": None,
        "rotated_files": [],
        "compressed_files": [],
        "archived_files": [],
        "totals": {
            "active_bytes": 0,
            "rotated_bytes": 0,
            "compressed_bytes": 0,
            "archived_bytes": 0,
            "total_bytes": 0,
        },
        "warnings": [],
    }

    # Check active trace file
    if trace_path.exists():
        stat = trace_path.stat()
        age_days = file_age_days(trace_path)
        size_mb = file_size_mb(trace_path)
        hours_since_write = (now_ts() - stat.st_mtime) / 3600.0

        status["active_trace"] = {
            "path": str(trace_path),
            "size_bytes": stat.st_size,
            "size_human": human_bytes(stat.st_size),
            "age_days": round(age_days, 2),
            "hours_since_write": round(hours_since_write, 2),
            "needs_rotation": size_mb >= policy.max_size_mb,
        }
        status["totals"]["active_bytes"] = stat.st_size

        # Warning if trace hasn't been written in 24h
        if hours_since_write > 24:
            status["warnings"].append({
                "type": "stale_trace",
                "message": f"trace.ndjson hasn't been written in {hours_since_write:.1f} hours",
                "hours": round(hours_since_write, 2),
            })
    else:
        status["warnings"].append({
            "type": "missing_trace",
            "message": "active trace file does not exist",
        })

    # Rotated files
    for f in get_rotated_files(trace_dir):
        stat = f.stat()
        status["rotated_files"].append({
            "name": f.name,
            "size_bytes": stat.st_size,
            "size_human": human_bytes(stat.st_size),
            "age_days": round(file_age_days(f), 2),
            "needs_compression": file_age_days(f) >= policy.compress_after_days,
        })
        status["totals"]["rotated_bytes"] += stat.st_size

    # Compressed files
    for f in get_compressed_files(trace_dir):
        stat = f.stat()
        status["compressed_files"].append({
            "name": f.name,
            "size_bytes": stat.st_size,
            "size_human": human_bytes(stat.st_size),
            "age_days": round(file_age_days(f), 2),
        })
        status["totals"]["compressed_bytes"] += stat.st_size

    # Archived files
    for f in get_archived_files(policy.archive_path):
        stat = f.stat()
        status["archived_files"].append({
            "name": f.name,
            "size_bytes": stat.st_size,
            "size_human": human_bytes(stat.st_size),
            "age_days": round(file_age_days(f), 2),
            "needs_pruning": file_age_days(f) >= policy.max_age_days,
        })
        status["totals"]["archived_bytes"] += stat.st_size

    # Calculate total
    status["totals"]["total_bytes"] = (
        status["totals"]["active_bytes"]
        + status["totals"]["rotated_bytes"]
        + status["totals"]["compressed_bytes"]
        + status["totals"]["archived_bytes"]
    )
    status["totals"]["total_human"] = human_bytes(status["totals"]["total_bytes"])

    # Validate policy
    policy_warnings = policy.validate()
    for w in policy_warnings:
        status["warnings"].append({
            "type": "policy_validation",
            "message": w,
        })

    return status


def run_all(
    trace_dir: Path,
    policy: RetentionPolicy,
    *,
    dry_run: bool = False,
    bus_dir: Path | None = None,
    actor: str | None = None,
) -> dict[str, Any]:
    """Run all retention operations: rotate, compress, archive, prune."""
    trace_path = trace_dir / "trace.ndjson"

    results: dict[str, Any] = {
        "dry_run": dry_run,
        "timestamp": now_iso_utc(),
        "operations": {},
    }

    # 1. Rotate
    results["operations"]["rotate"] = rotate_trace(
        trace_path, policy, dry_run=dry_run, bus_dir=bus_dir, actor=actor
    )

    # 2. Compress
    results["operations"]["compress"] = compress_traces(
        trace_dir, policy, dry_run=dry_run, bus_dir=bus_dir, actor=actor
    )

    # 3. Archive
    results["operations"]["archive"] = archive_traces(
        trace_dir, policy, dry_run=dry_run, bus_dir=bus_dir, actor=actor
    )

    # 4. Prune
    results["operations"]["prune"] = prune_archives(
        policy, dry_run=dry_run, bus_dir=bus_dir, actor=actor
    )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_duration(s: str) -> int:
    """Parse duration string like '90d' or '30' to days."""
    s = s.strip().lower()
    if s.endswith("d"):
        return int(s[:-1])
    return int(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dialogos_retention.py",
        description="Retention and compression policies for dialogos trace.",
    )
    p.add_argument(
        "--trace-dir",
        default=str(DEFAULT_TRACE_DIR),
        help=f"Dialogos trace directory (default: {DEFAULT_TRACE_DIR})",
    )
    p.add_argument(
        "--policy",
        default=None,
        help="Path to retention policy JSON file",
    )
    p.add_argument(
        "--bus-dir",
        default=str(DEFAULT_BUS_DIR),
        help=f"Bus directory for event emission (default: {DEFAULT_BUS_DIR})",
    )
    p.add_argument(
        "--actor",
        default=None,
        help="Actor name for bus events",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # status
    status_p = sub.add_parser("status", help="Show retention status and statistics")
    status_p.set_defaults(func=cmd_status)

    # rotate
    rotate_p = sub.add_parser("rotate", help="Rotate trace file if it exceeds max_size_mb")
    rotate_p.add_argument("--dry-run", action="store_true", help="Show what would be done")
    rotate_p.set_defaults(func=cmd_rotate)

    # compress
    compress_p = sub.add_parser("compress", help="Compress traces older than compress_after_days")
    compress_p.add_argument("--dry-run", action="store_true", help="Show what would be done")
    compress_p.set_defaults(func=cmd_compress)

    # archive
    archive_p = sub.add_parser("archive", help="Move compressed traces to archive_path")
    archive_p.add_argument("--dry-run", action="store_true", help="Show what would be done")
    archive_p.set_defaults(func=cmd_archive)

    # prune
    prune_p = sub.add_parser("prune", help="Delete archives older than max_age_days")
    prune_p.add_argument("--older-than", default=None, help="Override threshold (e.g., '90d' or '90')")
    prune_p.add_argument("--dry-run", action="store_true", help="Show what would be done")
    prune_p.set_defaults(func=cmd_prune)

    # all
    all_p = sub.add_parser("all", help="Run all retention operations")
    all_p.add_argument("--dry-run", action="store_true", help="Show what would be done")
    all_p.set_defaults(func=cmd_all)

    return p


def cmd_status(args: argparse.Namespace) -> int:
    trace_dir = Path(args.trace_dir)
    policy = load_policy(Path(args.policy) if args.policy else None)
    status = get_status(trace_dir, policy)

    if args.json:
        sys.stdout.write(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
        return 0

    # Human-readable output
    print(f"=== Dialogos Retention Status ===")
    print(f"Trace dir:    {status['trace_dir']}")
    print(f"Archive path: {status['archive_path']}")
    print()
    print(f"Policy:")
    print(f"  max_age_days:        {status['policy']['max_age_days']}")
    print(f"  max_size_mb:         {status['policy']['max_size_mb']}")
    print(f"  compress_after_days: {status['policy']['compress_after_days']}")
    print()

    if status["active_trace"]:
        at = status["active_trace"]
        print(f"Active trace:")
        print(f"  Size: {at['size_human']} ({at['size_bytes']} bytes)")
        print(f"  Age:  {at['age_days']} days")
        print(f"  Last write: {at['hours_since_write']:.1f} hours ago")
        if at["needs_rotation"]:
            print(f"  [!] Needs rotation (exceeds {status['policy']['max_size_mb']}MB)")
    else:
        print("Active trace: NOT FOUND")
    print()

    print(f"Rotated files: {len(status['rotated_files'])}")
    for f in status["rotated_files"]:
        flag = " [compress]" if f["needs_compression"] else ""
        print(f"  - {f['name']} ({f['size_human']}, {f['age_days']}d){flag}")

    print(f"\nCompressed files: {len(status['compressed_files'])}")
    for f in status["compressed_files"]:
        print(f"  - {f['name']} ({f['size_human']}, {f['age_days']}d)")

    print(f"\nArchived files: {len(status['archived_files'])}")
    for f in status["archived_files"]:
        flag = " [prune]" if f["needs_pruning"] else ""
        print(f"  - {f['name']} ({f['size_human']}, {f['age_days']}d){flag}")

    print(f"\nTotals:")
    print(f"  Active:     {human_bytes(status['totals']['active_bytes'])}")
    print(f"  Rotated:    {human_bytes(status['totals']['rotated_bytes'])}")
    print(f"  Compressed: {human_bytes(status['totals']['compressed_bytes'])}")
    print(f"  Archived:   {human_bytes(status['totals']['archived_bytes'])}")
    print(f"  Total:      {status['totals']['total_human']}")

    if status["warnings"]:
        print(f"\nWarnings:")
        for w in status["warnings"]:
            print(f"  [!] {w['type']}: {w['message']}")

    return 0


def cmd_rotate(args: argparse.Namespace) -> int:
    trace_dir = Path(args.trace_dir)
    trace_path = trace_dir / "trace.ndjson"
    policy = load_policy(Path(args.policy) if args.policy else None)
    bus_dir = Path(args.bus_dir) if not args.dry_run else None

    result = rotate_trace(
        trace_path, policy, dry_run=args.dry_run, bus_dir=bus_dir, actor=args.actor
    )

    if args.json:
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    else:
        mode = "[DRY-RUN] " if args.dry_run else ""
        if result.get("rotated"):
            print(f"{mode}Rotated: {result.get('message', '')}")
        elif result.get("error"):
            print(f"{mode}Error: {result['error']}")
        else:
            print(f"{mode}{result.get('message', 'No rotation needed')}")

    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    trace_dir = Path(args.trace_dir)
    policy = load_policy(Path(args.policy) if args.policy else None)
    bus_dir = Path(args.bus_dir) if not args.dry_run else None

    result = compress_traces(
        trace_dir, policy, dry_run=args.dry_run, bus_dir=bus_dir, actor=args.actor
    )

    if args.json:
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    else:
        mode = "[DRY-RUN] " if args.dry_run else ""
        print(f"{mode}Compressed: {len(result['compressed'])} files")
        for f in result["compressed"]:
            if "ratio" in f:
                print(f"  - {f['file']} -> {f['gz_file']} (ratio: {f['ratio']})")
            else:
                print(f"  - {f['file']} -> {f['gz_file']}")
        if result["skipped"]:
            print(f"Skipped: {len(result['skipped'])} files (not old enough)")
        if result["errors"]:
            print(f"Errors: {len(result['errors'])}")
            for e in result["errors"]:
                print(f"  - {e['file']}: {e['error']}")

    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    trace_dir = Path(args.trace_dir)
    policy = load_policy(Path(args.policy) if args.policy else None)
    bus_dir = Path(args.bus_dir) if not args.dry_run else None

    result = archive_traces(
        trace_dir, policy, dry_run=args.dry_run, bus_dir=bus_dir, actor=args.actor
    )

    if args.json:
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    else:
        mode = "[DRY-RUN] " if args.dry_run else ""
        print(f"{mode}Archived: {len(result['archived'])} files to {result['archive_path']}")
        for f in result["archived"]:
            print(f"  - {f['file']}")
        if result["errors"]:
            print(f"Errors: {len(result['errors'])}")
            for e in result["errors"]:
                print(f"  - {e['file']}: {e['error']}")

    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    policy = load_policy(Path(args.policy) if args.policy else None)
    bus_dir = Path(args.bus_dir) if not args.dry_run else None

    older_than = None
    if args.older_than:
        older_than = parse_duration(args.older_than)

    result = prune_archives(
        policy, older_than_days=older_than, dry_run=args.dry_run, bus_dir=bus_dir, actor=args.actor
    )

    if args.json:
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    else:
        mode = "[DRY-RUN] " if args.dry_run else ""
        print(f"{mode}Pruned: {len(result['pruned'])} files (older than {result['threshold_days']} days)")
        total_freed = sum(p.get("size_bytes", 0) for p in result["pruned"])
        for f in result["pruned"]:
            print(f"  - {f['file']} ({f['age_days']}d)")
        if total_freed > 0:
            print(f"Total freed: {human_bytes(total_freed)}")
        if result["skipped"]:
            print(f"Skipped: {len(result['skipped'])} files (not old enough)")
        if result["errors"]:
            print(f"Errors: {len(result['errors'])}")
            for e in result["errors"]:
                print(f"  - {e['file']}: {e['error']}")

    return 0


def cmd_all(args: argparse.Namespace) -> int:
    trace_dir = Path(args.trace_dir)
    policy = load_policy(Path(args.policy) if args.policy else None)
    bus_dir = Path(args.bus_dir) if not args.dry_run else None

    result = run_all(
        trace_dir, policy, dry_run=args.dry_run, bus_dir=bus_dir, actor=args.actor
    )

    if args.json:
        sys.stdout.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    else:
        mode = "[DRY-RUN] " if args.dry_run else ""
        print(f"{mode}=== Retention All Operations ===")
        print(f"Timestamp: {result['timestamp']}")
        print()

        ops = result["operations"]

        # Rotate
        rot = ops.get("rotate", {})
        if rot.get("rotated"):
            print(f"Rotate: {rot.get('message', 'rotated')}")
        else:
            print(f"Rotate: {rot.get('message', 'no action')}")

        # Compress
        comp = ops.get("compress", {})
        print(f"Compress: {len(comp.get('compressed', []))} files")

        # Archive
        arch = ops.get("archive", {})
        print(f"Archive: {len(arch.get('archived', []))} files")

        # Prune
        pru = ops.get("prune", {})
        print(f"Prune: {len(pru.get('pruned', []))} files")

    return 0


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
