#!/usr/bin/env python3
from __future__ import annotations

"""
PBRECONCILE — Lossless Reconciliation Operator (DKIN v21)

Prevents erasure chaos from parallel agent operations by:
- Scanning for orphaned work in /tmp
- Extracting uncommitted changes as patches
- Detecting collisions when multiple agents modify same files
- Archiving orphan work for 1-week retention
- Applying non-conflicting patches to main

Usage:
    python3 nucleus/tools/pbreconcile_operator.py --scan-orphans
    python3 nucleus/tools/pbreconcile_operator.py --extract-diffs
    python3 nucleus/tools/pbreconcile_operator.py --detect-collisions
    python3 nucleus/tools/pbreconcile_operator.py --reconcile --dry-run
    python3 nucleus/tools/pbreconcile_operator.py --archive-orphan /tmp/pluribus_codex_xyz
    python3 nucleus/tools/pbreconcile_operator.py --show-dag
    python3 nucleus/tools/pbreconcile_operator.py --prune-archives
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

# === Configuration ===
RECONCILE_PROTOCOL_VERSION = 21

# Retention policy
ORPHAN_SCAN_INTERVAL_HOURS = 1
GRACE_PERIOD_HOURS = 24
ARCHIVE_RETENTION_DAYS = 7
COLLISION_FREEZE_HOURS = 4

# Paths
DEFAULT_BUS_DIR = Path("/var/lib/pluribus/.pluribus/bus")
DEFAULT_RECONCILE_DIR = Path("/var/lib/pluribus/.pluribus/reconcile")
DEFAULT_REPO_ROOT = Path("/pluribus")
TMP_DIR = Path("/tmp")

# Orphan detection patterns
ORPHAN_PATTERNS = ["pluribus_codex_*", "pluribus_claude_*", "pluribus_gemini_*", "pluribus_qwen_*"]


@dataclass
class OrphanInfo:
    """Information about an orphaned work directory."""
    path: str
    agent_type: str
    session_id: str
    branch: str = ""
    uncommitted_count: int = 0
    modified_files: list[str] = field(default_factory=list)
    untracked_files: list[str] = field(default_factory=list)
    age_hours: float = 0.0
    has_manifest: bool = False
    collision_risk: bool = False


@dataclass
class CollisionInfo:
    """Information about a file collision."""
    path: str
    agents: list[str]
    hashes: list[str]
    resolution_status: str = "pending"  # pending, merged, frozen, resolved


@dataclass
class ReconcileState:
    """Current reconciliation state."""
    version: str = f"v{RECONCILE_PROTOCOL_VERSION}"
    last_scan: str = ""
    active_orphans: int = 0
    pending_patches: int = 0
    collisions_frozen: int = 0
    archives_count: int = 0
    total_files_preserved: int = 0


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
        "host": os.uname().nodename,
        "pid": os.getpid(),
        "data": data,
    }
    append_ndjson(bus_dir / "events.ndjson", evt)


def get_file_age_hours(path: Path) -> float:
    """Get file/directory age in hours."""
    try:
        mtime = path.stat().st_mtime
        return (time.time() - mtime) / 3600
    except OSError:
        return 0.0


def collision_hash(path: str, content: str) -> str:
    """Generate collision detection hash."""
    return hashlib.sha256(f"{path}:{content[:1000]}".encode()).hexdigest()[:16]


def semantic_key(path: str) -> str:
    """Generate semantic key for a file."""
    p = Path(path)
    ext = p.suffix.lower()
    name = p.stem

    # Determine file type
    if ext in [".tsx", ".jsx"]:
        ftype = "component"
    elif ext in [".ts", ".js"]:
        ftype = "module"
    elif ext == ".py":
        ftype = "python"
    elif ext == ".md":
        ftype = "doc"
    elif ext in [".json", ".yaml", ".yml"]:
        ftype = "config"
    elif ext in [".css", ".scss"]:
        ftype = "style"
    else:
        ftype = "file"

    return f"{ftype}:{name}"


def run_git_cmd(repo_path: Path, *args: str) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), *args],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def scan_orphans(tmp_dir: Path = TMP_DIR) -> list[OrphanInfo]:
    """Scan for orphaned work directories in /tmp."""
    orphans = []

    for pattern in ORPHAN_PATTERNS:
        for orphan_path in tmp_dir.glob(pattern):
            if not orphan_path.is_dir():
                continue

            # Check if it's a git repo
            git_dir = orphan_path / ".git"
            if not git_dir.exists():
                continue

            # Parse agent info from directory name
            name_parts = orphan_path.name.split("_")
            agent_type = name_parts[1] if len(name_parts) > 1 else "unknown"
            session_id = name_parts[-1] if len(name_parts) > 2 else "unknown"

            orphan = OrphanInfo(
                path=str(orphan_path),
                agent_type=agent_type,
                session_id=session_id,
                age_hours=get_file_age_hours(orphan_path),
            )

            # Get git status
            rc, stdout, _ = run_git_cmd(orphan_path, "status", "--porcelain")
            if rc == 0 and stdout.strip():
                lines = stdout.strip().split("\n")
                for line in lines:
                    if line.startswith(" M ") or line.startswith("M "):
                        orphan.modified_files.append(line[3:].strip())
                    elif line.startswith("??"):
                        orphan.untracked_files.append(line[3:].strip())
                orphan.uncommitted_count = len(orphan.modified_files) + len(orphan.untracked_files)

            # Get current branch
            rc, stdout, _ = run_git_cmd(orphan_path, "branch", "--show-current")
            if rc == 0:
                orphan.branch = stdout.strip()

            # Check for manifest
            manifest_path = orphan_path / ".pluribus" / "work_manifest.json"
            orphan.has_manifest = manifest_path.exists()

            orphans.append(orphan)

    return orphans


def extract_diff(orphan_path: Path, output_dir: Path) -> dict:
    """Extract uncommitted changes from orphan as patch file."""
    ensure_dir(output_dir)

    # Get diff for modified files
    rc, diff_output, stderr = run_git_cmd(orphan_path, "diff")
    if rc != 0:
        return {"success": False, "error": stderr}

    # Get diff for staged files
    rc, staged_diff, _ = run_git_cmd(orphan_path, "diff", "--cached")
    if rc == 0 and staged_diff:
        diff_output += "\n" + staged_diff

    if not diff_output.strip():
        return {"success": True, "no_changes": True}

    # Generate patch filename
    name_parts = orphan_path.name.split("_")
    agent_type = name_parts[1] if len(name_parts) > 1 else "unknown"
    session_id = name_parts[-1] if len(name_parts) > 2 else orphan_path.name
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    patch_filename = f"{agent_type}-{session_id}-{timestamp}.patch"
    patch_path = output_dir / patch_filename

    patch_path.write_text(diff_output, encoding="utf-8")

    return {
        "success": True,
        "patch_path": str(patch_path),
        "patch_size_bytes": len(diff_output),
        "lines_changed": len(diff_output.split("\n")),
    }


def detect_collisions(orphans: list[OrphanInfo], repo_root: Path) -> list[CollisionInfo]:
    """Detect files modified by multiple agents."""
    file_agents: dict[str, list[tuple[str, str]]] = {}  # path -> [(agent_id, hash)]

    for orphan in orphans:
        agent_id = f"{orphan.agent_type}-{orphan.session_id}"
        orphan_path = Path(orphan.path)

        for modified_file in orphan.modified_files:
            full_path = orphan_path / modified_file
            if full_path.exists():
                try:
                    content = full_path.read_text(encoding="utf-8", errors="replace")[:2000]
                    file_hash = collision_hash(modified_file, content)
                except Exception:
                    file_hash = "error"

                if modified_file not in file_agents:
                    file_agents[modified_file] = []
                file_agents[modified_file].append((agent_id, file_hash))

    collisions = []
    for path, agents_hashes in file_agents.items():
        if len(agents_hashes) > 1:
            # Check if hashes differ (true collision) vs same change
            hashes = [h for _, h in agents_hashes]
            if len(set(hashes)) > 1:  # Different changes
                collisions.append(CollisionInfo(
                    path=path,
                    agents=[a for a, _ in agents_hashes],
                    hashes=hashes,
                    resolution_status="pending",
                ))

    return collisions


def archive_orphan(orphan_path: Path, archive_dir: Path) -> dict:
    """Archive an orphan directory for retention."""
    ensure_dir(archive_dir)

    if not orphan_path.exists():
        return {"success": False, "error": "Orphan path does not exist"}

    # Generate archive name
    name_parts = orphan_path.name.split("_")
    agent_type = name_parts[1] if len(name_parts) > 1 else "unknown"
    session_id = name_parts[-1] if len(name_parts) > 2 else orphan_path.name
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    archive_name = f"orphan-{agent_type}-{session_id}-{timestamp}.tar.gz"
    archive_path = archive_dir / archive_name

    # Create manifest for archive
    manifest = {
        "manifest_version": "v21",
        "agent_type": agent_type,
        "session_id": session_id,
        "archived_at": now_iso_utc(),
        "original_path": str(orphan_path),
        "ttl_expires": (datetime.now(timezone.utc) + timedelta(days=ARCHIVE_RETENTION_DAYS)).isoformat().replace("+00:00", "Z"),
    }

    # Get file list
    files_preserved = []
    for f in orphan_path.rglob("*"):
        if f.is_file() and ".git" not in str(f):
            files_preserved.append(str(f.relative_to(orphan_path)))
    manifest["files_preserved"] = files_preserved
    manifest["file_count"] = len(files_preserved)

    try:
        # Create tarball
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(orphan_path, arcname=orphan_path.name)

        # Write manifest alongside archive
        manifest_path = archive_dir / f"orphan-{agent_type}-{session_id}-{timestamp}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        archive_size = archive_path.stat().st_size

        return {
            "success": True,
            "archive_path": str(archive_path),
            "manifest_path": str(manifest_path),
            "archive_size_bytes": archive_size,
            "files_preserved": len(files_preserved),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def apply_patch(patch_path: Path, repo_root: Path, dry_run: bool = False) -> dict:
    """Apply a patch to the repository."""
    if not patch_path.exists():
        return {"success": False, "error": "Patch file does not exist"}

    cmd_args = ["apply", "--check"] if dry_run else ["apply"]
    cmd_args.append(str(patch_path))

    rc, stdout, stderr = run_git_cmd(repo_root, *cmd_args)

    if rc != 0:
        return {
            "success": False,
            "error": stderr,
            "conflicts": "patch does not apply cleanly",
        }

    return {
        "success": True,
        "dry_run": dry_run,
        "patch_applied": str(patch_path),
    }


def prune_archives(archive_dir: Path, max_age_days: int = ARCHIVE_RETENTION_DAYS) -> dict:
    """Remove archives older than retention period."""
    if not archive_dir.exists():
        return {"success": True, "pruned": 0}

    pruned = []
    max_age_hours = max_age_days * 24

    for archive_file in archive_dir.glob("orphan-*.tar.gz"):
        age_hours = get_file_age_hours(archive_file)
        if age_hours > max_age_hours:
            try:
                archive_file.unlink()
                # Also remove manifest
                manifest_file = archive_file.with_suffix(".manifest.json")
                if manifest_file.exists():
                    manifest_file.unlink()
                pruned.append(str(archive_file.name))
            except OSError:
                pass

    return {"success": True, "pruned": len(pruned), "files": pruned}


def load_state(reconcile_dir: Path) -> ReconcileState:
    """Load reconciliation state."""
    state_file = reconcile_dir / "state.json"
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            state = ReconcileState()
            for k, v in data.items():
                if hasattr(state, k):
                    setattr(state, k, v)
            return state
        except Exception:
            pass
    return ReconcileState()


def save_state(reconcile_dir: Path, state: ReconcileState) -> None:
    """Save reconciliation state."""
    ensure_dir(reconcile_dir)
    state_file = reconcile_dir / "state.json"
    state_file.write_text(json.dumps(state.__dict__, indent=2), encoding="utf-8")


def show_dag(reconcile_dir: Path) -> dict:
    """Show reconciliation DAG events."""
    dag_file = reconcile_dir / "dag.ndjson"
    if not dag_file.exists():
        return {"events": [], "count": 0}

    events = []
    try:
        with dag_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
    except Exception:
        pass

    return {"events": events[-50:], "count": len(events)}  # Last 50 events


def format_scan_report(orphans: list[OrphanInfo], collisions: list[CollisionInfo]) -> str:
    """Format orphan scan report for display."""
    lines = [
        "=" * 72,
        f"  PBRECONCILE SCAN — DKIN v{RECONCILE_PROTOCOL_VERSION}",
        f"  Timestamp: {now_iso_utc()}",
        "=" * 72,
        "",
        f"  ORPHANED WORK DIRECTORIES: {len(orphans)}",
        "-" * 72,
    ]

    if not orphans:
        lines.append("  No orphaned work found.")
    else:
        for orphan in orphans:
            status = "UNCOMMITTED" if orphan.uncommitted_count > 0 else "clean"
            lines.extend([
                f"  {orphan.path}",
                f"    Agent: {orphan.agent_type} | Branch: {orphan.branch}",
                f"    Age: {orphan.age_hours:.1f}h | Files: {orphan.uncommitted_count} | Status: {status}",
                "",
            ])

    lines.extend([
        "-" * 72,
        f"  COLLISIONS DETECTED: {len(collisions)}",
        "-" * 72,
    ])

    if not collisions:
        lines.append("  No collisions detected.")
    else:
        for collision in collisions:
            lines.extend([
                f"  {collision.path}",
                f"    Agents: {', '.join(collision.agents)}",
                f"    Status: {collision.resolution_status}",
                "",
            ])

    lines.append("=" * 72)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pbreconcile_operator.py",
        description="PBRECONCILE — Lossless Reconciliation Operator (DKIN v21)",
    )
    p.add_argument("--bus-dir", default=None, help="Bus directory")
    p.add_argument("--reconcile-dir", default=None, help="Reconciliation data directory")
    p.add_argument("--repo-root", default=None, help="Repository root")
    p.add_argument("--actor", default=None, help="Actor identity")

    action = p.add_mutually_exclusive_group()
    action.add_argument("--scan-orphans", action="store_true", help="Scan for orphaned work")
    action.add_argument("--extract-diffs", action="store_true", help="Extract patches from orphans")
    action.add_argument("--detect-collisions", action="store_true", help="Detect file collisions")
    action.add_argument("--reconcile", action="store_true", help="Apply non-conflicting patches")
    action.add_argument("--archive-orphan", metavar="PATH", help="Archive specific orphan")
    action.add_argument("--show-dag", action="store_true", help="Show reconciliation DAG")
    action.add_argument("--prune-archives", action="store_true", help="Remove expired archives")
    action.add_argument("--full-reconcile", action="store_true", help="Full reconciliation cycle")

    p.add_argument("--dry-run", action="store_true", help="Dry run (no changes)")
    p.add_argument("--confirm", action="store_true", help="Confirm destructive operations")
    p.add_argument("--json", action="store_true", help="JSON output")

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    actor = args.actor or os.environ.get("PLURIBUS_ACTOR") or "pbreconcile"
    bus_dir = Path(args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or DEFAULT_BUS_DIR)
    reconcile_dir = Path(args.reconcile_dir or DEFAULT_RECONCILE_DIR)
    repo_root = Path(args.repo_root or DEFAULT_REPO_ROOT)

    ensure_dir(bus_dir)
    ensure_dir(reconcile_dir)
    ensure_dir(reconcile_dir / "archive")
    ensure_dir(reconcile_dir / "patches")
    ensure_dir(reconcile_dir / "manifests")

    result: dict[str, Any] = {}

    # Default action: scan orphans
    if args.scan_orphans or (not any([
        args.extract_diffs, args.detect_collisions, args.reconcile,
        args.archive_orphan, args.show_dag, args.prune_archives, args.full_reconcile
    ])):
        orphans = scan_orphans()
        collisions = detect_collisions(orphans, repo_root)

        # Update state
        state = load_state(reconcile_dir)
        state.last_scan = now_iso_utc()
        state.active_orphans = len(orphans)
        state.collisions_frozen = sum(1 for c in collisions if c.resolution_status == "frozen")
        save_state(reconcile_dir, state)

        if args.json:
            result = {
                "timestamp": now_iso_utc(),
                "orphans": [o.__dict__ for o in orphans],
                "collisions": [c.__dict__ for c in collisions],
                "total_uncommitted_files": sum(o.uncommitted_count for o in orphans),
            }
            sys.stdout.write(json.dumps(result, indent=2) + "\n")
        else:
            sys.stdout.write(format_scan_report(orphans, collisions) + "\n")

        # Emit bus event
        emit_bus(bus_dir, topic="reconcile.orphan.scanned", kind="metric", level="info", actor=actor, data={
            "orphan_count": len(orphans),
            "collision_count": len(collisions),
            "uncommitted_files": sum(o.uncommitted_count for o in orphans),
            "protocol_version": RECONCILE_PROTOCOL_VERSION,
        })

        return 0

    elif args.extract_diffs:
        orphans = scan_orphans()
        patches_dir = reconcile_dir / "patches"

        extracted = []
        for orphan in orphans:
            if orphan.uncommitted_count > 0:
                result = extract_diff(Path(orphan.path), patches_dir)
                if result.get("success") and not result.get("no_changes"):
                    extracted.append({
                        "orphan": orphan.path,
                        "patch": result.get("patch_path"),
                        "lines": result.get("lines_changed"),
                    })

                    # Emit bus event
                    emit_bus(bus_dir, topic="reconcile.diff.extracted", kind="artifact", level="info", actor=actor, data={
                        "orphan_path": orphan.path,
                        "patch_path": result.get("patch_path"),
                        "lines_changed": result.get("lines_changed"),
                    })

        # Update state
        state = load_state(reconcile_dir)
        state.pending_patches = len(list(patches_dir.glob("*.patch")))
        save_state(reconcile_dir, state)

        result = {"success": True, "extracted": extracted, "count": len(extracted)}

    elif args.detect_collisions:
        orphans = scan_orphans()
        collisions = detect_collisions(orphans, repo_root)

        if collisions:
            for collision in collisions:
                emit_bus(bus_dir, topic="reconcile.collision.detected", kind="warn", level="warn", actor=actor, data={
                    "file_path": collision.path,
                    "agents": collision.agents,
                    "hashes": collision.hashes,
                })

        result = {
            "success": True,
            "collisions": [c.__dict__ for c in collisions],
            "count": len(collisions),
        }

    elif args.reconcile:
        if not args.dry_run and not args.confirm:
            sys.stderr.write("ERROR: --reconcile requires --dry-run or --confirm\n")
            return 1

        patches_dir = reconcile_dir / "patches"
        applied = []
        failed = []

        for patch_file in patches_dir.glob("*.patch"):
            result = apply_patch(patch_file, repo_root, dry_run=args.dry_run)
            if result.get("success"):
                applied.append(str(patch_file.name))
                if not args.dry_run:
                    emit_bus(bus_dir, topic="reconcile.patch.applied", kind="request", level="info", actor=actor, data={
                        "patch_file": str(patch_file.name),
                    })
            else:
                failed.append({"patch": str(patch_file.name), "error": result.get("error")})

        result = {
            "success": len(failed) == 0,
            "dry_run": args.dry_run,
            "applied": applied,
            "failed": failed,
        }

    elif args.archive_orphan:
        orphan_path = Path(args.archive_orphan)
        archive_dir = reconcile_dir / "archive"

        result = archive_orphan(orphan_path, archive_dir)

        if result.get("success"):
            emit_bus(bus_dir, topic="reconcile.archive.created", kind="artifact", level="info", actor=actor, data={
                "original_path": str(orphan_path),
                "archive_path": result.get("archive_path"),
                "files_preserved": result.get("files_preserved"),
            })

            # Update state
            state = load_state(reconcile_dir)
            state.archives_count += 1
            state.total_files_preserved += result.get("files_preserved", 0)
            save_state(reconcile_dir, state)

    elif args.show_dag:
        result = show_dag(reconcile_dir)

    elif args.prune_archives:
        archive_dir = reconcile_dir / "archive"
        result = prune_archives(archive_dir)

        if result.get("pruned", 0) > 0:
            emit_bus(bus_dir, topic="reconcile.archive.pruned", kind="request", level="info", actor=actor, data={
                "pruned_count": result.get("pruned"),
            })

    elif args.full_reconcile:
        if not args.confirm:
            sys.stderr.write("ERROR: --full-reconcile requires --confirm\n")
            return 1

        # 1. Scan orphans
        orphans = scan_orphans()
        collisions = detect_collisions(orphans, repo_root)

        # 2. Extract diffs from orphans with uncommitted work
        patches_dir = reconcile_dir / "patches"
        extracted = []
        for orphan in orphans:
            if orphan.uncommitted_count > 0:
                extract_result = extract_diff(Path(orphan.path), patches_dir)
                if extract_result.get("success") and not extract_result.get("no_changes"):
                    extracted.append(extract_result.get("patch_path"))

        # 3. Archive orphans
        archive_dir = reconcile_dir / "archive"
        archived = []
        for orphan in orphans:
            if orphan.uncommitted_count > 0:
                archive_result = archive_orphan(Path(orphan.path), archive_dir)
                if archive_result.get("success"):
                    archived.append(archive_result.get("archive_path"))

        # 4. Apply non-conflicting patches (skip if collisions)
        applied = []
        collision_files = {c.path for c in collisions}
        for patch_file in patches_dir.glob("*.patch"):
            # Check if patch touches collision files
            patch_content = patch_file.read_text(encoding="utf-8", errors="replace")
            touches_collision = any(cf in patch_content for cf in collision_files)

            if not touches_collision:
                apply_result = apply_patch(patch_file, repo_root, dry_run=False)
                if apply_result.get("success"):
                    applied.append(str(patch_file.name))

        result = {
            "success": True,
            "orphans_found": len(orphans),
            "collisions_detected": len(collisions),
            "patches_extracted": len(extracted),
            "orphans_archived": len(archived),
            "patches_applied": len(applied),
        }

        emit_bus(bus_dir, topic="reconcile.full.completed", kind="request", level="info", actor=actor, data=result)

    # Output result
    if args.json:
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    else:
        if "error" in result:
            sys.stderr.write(f"ERROR: {result['error']}\n")
            return 1
        sys.stdout.write(f"Result: {json.dumps(result, indent=2)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
