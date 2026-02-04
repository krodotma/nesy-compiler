#!/usr/bin/env python3
"""
dashboard_guard.py - CI/CD Enforcement Gate for Pluribus Dashboard

PURPOSE:
  Ensures the dashboard NEVER serves broken code by running typecheck
  before any change goes live. Auto-reverts breaking changes.

PHILOSOPHY:
  "Always up, always live, always evolving" requires gatekeeping.
  This daemon is the guardian that enforces build integrity.

PROTOCOL:
  1. Watch dashboard/src for file changes
  2. On change: run `npm run typecheck`
  3. If PASS: allow change, emit dashboard.guard.pass
  4. If FAIL:
     - If baseline is clean: revert (strict)
     - If baseline already failing: revert only on new error signatures
     - If failure is unparseable: revert conservatively
     - If file is untracked and cannot be reverted: quarantine it
  5. Continuous monitoring with debounce

USAGE:
  python3 dashboard_guard.py [--dry-run] [--no-revert]

BUS EVENTS:
  - dashboard.guard.pass   {file, duration_ms}
  - dashboard.guard.fail   {file, error, reverted}
  - dashboard.guard.revert {file, reason}
  - dashboard.guard.baseline {success, error_count, error_instances}
  - dashboard.guard.degraded {files, reason}
  - dashboard.guard.quarantine {file, target, reason}
  - dashboard.guard.start  {pid, mode}

Author: Claude Opus (Pluribus Core Team)
"""

import os
import sys
import json
import time
import subprocess
import argparse
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Iterable
import hashlib

# Best-effort import for structured bus emission (avoids schema drift).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from nucleus.tools import agent_bus  # type: ignore
except Exception:  # pragma: no cover
    agent_bus = None  # type: ignore

# Optional watchdog - fallback to polling if not available
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    print("[dashboard_guard] watchdog not installed, using polling mode")

# Paths
DASHBOARD_DIR = Path("/pluribus/nucleus/dashboard")
SRC_DIR = DASHBOARD_DIR / "src"
QUARANTINE_DIR = DASHBOARD_DIR / ".guard_quarantine"
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
EVENTS_FILE = BUS_DIR / "events.ndjson"

# Config
DEBOUNCE_MS = 500  # Wait for rapid changes to settle
# NOTE: Dashboard typecheck can exceed 30s under load; keep this comfortably above typical tsc runtimes to avoid false reverts.
TYPECHECK_TIMEOUT = 120  # seconds
WATCHED_EXTENSIONS = {".tsx", ".ts", ".jsx", ".js"}
TS_ERROR_RE = re.compile(
    r"^(?P<path>.+?)\((?P<line>\d+),(?P<col>\d+)\): error TS(?P<code>\d+): (?P<message>.*)$"
)

ErrorKey = tuple[str, str]
ErrorCount = dict[ErrorKey, int]


def normalize_error_path(path_str: str) -> str:
    cleaned = path_str.strip().replace("\\", "/")
    if not cleaned:
        return cleaned
    path = Path(cleaned)
    if path.is_absolute():
        try:
            return path.relative_to(DASHBOARD_DIR).as_posix()
        except Exception:
            return cleaned
    return cleaned


def extract_ts_errors(output: str) -> list[dict]:
    errors: list[dict] = []
    for line in output.splitlines():
        match = TS_ERROR_RE.match(line.strip())
        if not match:
            continue
        errors.append({
            "path": normalize_error_path(match.group("path")),
            "code": f"TS{match.group('code')}",
            "message": match.group("message").strip(),
        })
    return errors


def error_counts(errors: Iterable[dict]) -> ErrorCount:
    counts: ErrorCount = {}
    for err in errors:
        key = (err.get("path", ""), err.get("code", ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def count_error_instances(counts: ErrorCount) -> int:
    return sum(counts.values())


def diff_new_errors(previous: ErrorCount, current: ErrorCount) -> ErrorCount:
    new_only: ErrorCount = {}
    for key, count in current.items():
        prev_count = previous.get(key, 0)
        if count > prev_count:
            new_only[key] = count - prev_count
    return new_only


def summarize_error_counts(counts: ErrorCount, limit: int = 5) -> list[str]:
    out: list[str] = []
    for (path, code), count in sorted(counts.items())[:limit]:
        suffix = f"(+{count})" if count > 0 else ""
        out.append(f"{path}:{code}{suffix}")
    return out


def decide_revert(
    previous_success: Optional[bool],
    previous_errors: ErrorCount,
    new_errors: ErrorCount,
) -> tuple[bool, str, ErrorCount]:
    new_only = diff_new_errors(previous_errors, new_errors)
    if previous_success is True:
        return True, "regression_from_clean", new_only
    if previous_success is False:
        if new_only:
            return True, "baseline_worsened", new_only
        if count_error_instances(new_errors) < count_error_instances(previous_errors):
            return False, "baseline_improved", {}
        return False, "baseline_unchanged", {}
    # Unknown baseline -> be conservative
    return True, "no_baseline", new_only or new_errors


def emit_event(topic: str, payload: dict) -> None:
    """Emit event to Pluribus bus."""
    try:
        if agent_bus is not None:
            paths = agent_bus.resolve_bus_paths(str(BUS_DIR))
            agent_bus.emit_event(
                paths,
                topic=topic,
                kind="metric",
                level="info",
                actor="dashboard-guard",
                data=payload,
                trace_id=None,
                run_id=None,
                durable=False,
            )
            return

        BUS_DIR.mkdir(parents=True, exist_ok=True)
        event = {
            "id": hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8", errors="replace")).hexdigest()[:16],
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "topic": topic,
            "kind": "metric",
            "level": "info",
            "actor": "dashboard-guard",
            "data": payload,
        }
        with open(EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as e:
        print(f"[bus] Failed to emit {topic}: {e}")


def run_typecheck() -> tuple[bool, str, float]:
    """
    Run npm typecheck and return (success, output, duration_ms).
    """
    start = time.time()
    try:
        result = subprocess.run(
            ["npm", "run", "typecheck"],
            cwd=DASHBOARD_DIR,
            capture_output=True,
            text=True,
            timeout=TYPECHECK_TIMEOUT,
        )
        duration_ms = (time.time() - start) * 1000
        output = result.stdout + result.stderr
        success = result.returncode == 0
        return success, output, duration_ms
    except subprocess.TimeoutExpired:
        return False, "Typecheck timed out", (time.time() - start) * 1000
    except Exception as e:
        return False, str(e), (time.time() - start) * 1000


def revert_file(filepath: Path) -> bool:
    """Revert a file using git checkout."""
    try:
        result = subprocess.run(
            ["git", "checkout", "--", str(filepath)],
            cwd=DASHBOARD_DIR,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def is_untracked(filepath: Path) -> bool:
    """Check if a file is untracked in git."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", str(filepath)],
            cwd=DASHBOARD_DIR,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().startswith("??")
    except Exception:
        return False


def quarantine_file(filepath: Path, reason: str) -> Optional[Path]:
    """Move a file aside to preserve evidence when revert is impossible."""
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            rel = filepath.relative_to(DASHBOARD_DIR)
        except Exception:
            rel = Path(filepath.name)
        target = QUARANTINE_DIR / stamp / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        filepath.replace(target)
        emit_event("dashboard.guard.quarantine", {
            "file": str(filepath),
            "target": str(target),
            "reason": reason,
        })
        return target
    except Exception:
        return None

def get_file_hash(filepath: Path) -> Optional[str]:
    """Get MD5 hash of file contents."""
    try:
        return hashlib.md5(filepath.read_bytes()).hexdigest()
    except Exception:
        return None


class DashboardGuard:
    """Main guard class that watches and validates changes."""

    def __init__(self, dry_run: bool = False, no_revert: bool = False, baseline: bool = True):
        self.dry_run = dry_run
        self.no_revert = no_revert
        self.last_check = 0
        self.file_hashes: dict[str, str] = {}
        self.pending_files: set[str] = set()
        self.last_success: Optional[bool] = None
        self.last_error_counts: ErrorCount = {}
        self._init_hashes()
        if baseline:
            self._init_baseline()

    def _init_hashes(self) -> None:
        """Initialize file hashes for change detection."""
        for ext in WATCHED_EXTENSIONS:
            for filepath in SRC_DIR.rglob(f"*{ext}"):
                h = get_file_hash(filepath)
                if h:
                    self.file_hashes[str(filepath)] = h

    def _init_baseline(self) -> None:
        """Capture baseline typecheck result to avoid reverting on pre-existing failures."""
        success, output, duration_ms = run_typecheck()
        errors = extract_ts_errors(output)
        self.last_success = success
        self.last_error_counts = error_counts(errors)
        emit_event("dashboard.guard.baseline", {
            "success": success,
            "error_count": len(self.last_error_counts),
            "error_instances": count_error_instances(self.last_error_counts),
            "duration_ms": round(duration_ms),
        })

    def on_file_change(self, filepath: str) -> None:
        """Handle a file change event."""
        path = Path(filepath)

        # Only watch relevant files
        if path.suffix not in WATCHED_EXTENSIONS:
            return
        if not str(path).startswith(str(SRC_DIR)):
            return

        # Check if file actually changed (avoid duplicate events)
        new_hash = get_file_hash(path)
        old_hash = self.file_hashes.get(filepath)
        if new_hash == old_hash:
            return

        self.file_hashes[filepath] = new_hash
        self.pending_files.add(filepath)

        # Debounce - wait for rapid changes to settle
        current_time = time.time() * 1000
        if current_time - self.last_check < DEBOUNCE_MS:
            return
        self.last_check = current_time

        # Process pending files
        self._process_pending()

    def _process_pending(self) -> None:
        """Process all pending file changes."""
        if not self.pending_files:
            return

        files = list(self.pending_files)
        self.pending_files.clear()

        print(f"\n[guard] Checking {len(files)} file(s): {', '.join(Path(f).name for f in files)}")

        # Run typecheck
        success, output, duration_ms = run_typecheck()
        previous_success = self.last_success
        previous_counts = dict(self.last_error_counts)
        errors = extract_ts_errors(output)
        new_error_counts = error_counts(errors)
        unknown_failure = False
        if not success and not new_error_counts:
            unknown_failure = True
            new_error_counts = {("guard", "TS0000"): 1}
        new_error_instances = count_error_instances(new_error_counts)
        baseline_error_instances = count_error_instances(previous_counts)

        if success:
            print(f"[guard] ✓ Typecheck PASSED ({duration_ms:.0f}ms)")
            emit_event("dashboard.guard.pass", {
                "files": files,
                "duration_ms": round(duration_ms),
                "baseline_success": previous_success,
                "baseline_error_count": len(previous_counts),
                "baseline_error_instances": baseline_error_instances,
            })
            self.last_success = True
            self.last_error_counts = {}
        else:
            print(f"[guard] ✗ Typecheck FAILED ({duration_ms:.0f}ms)")

            # Extract error summary
            error_lines = [l for l in output.split("\n") if "error TS" in l][:5]
            error_summary = "\n".join(error_lines) if error_lines else output[:500]

            print(f"[guard] Error: {error_summary}")

            should_revert, reason, new_only = decide_revert(
                previous_success, previous_counts, new_error_counts
            )
            if unknown_failure:
                should_revert = True
                reason = "unknown_failure"
                new_only = new_error_counts

            if self.dry_run:
                print("[guard] DRY RUN - would revert files")
                emit_event("dashboard.guard.fail", {
                    "files": files,
                    "error": error_summary,
                    "reverted": False,
                    "dry_run": True,
                    "baseline_success": previous_success,
                    "baseline_error_count": len(previous_counts),
                    "baseline_error_instances": baseline_error_instances,
                    "new_error_count": len(new_error_counts),
                    "new_error_instances": new_error_instances,
                    "new_error_keys": summarize_error_counts(new_only),
                    "reason": reason,
                })
            elif self.no_revert:
                print("[guard] NO REVERT mode - leaving broken files")
                emit_event("dashboard.guard.fail", {
                    "files": files,
                    "error": error_summary,
                    "reverted": False,
                    "baseline_success": previous_success,
                    "baseline_error_count": len(previous_counts),
                    "baseline_error_instances": baseline_error_instances,
                    "new_error_count": len(new_error_counts),
                    "new_error_instances": new_error_instances,
                    "new_error_keys": summarize_error_counts(new_only),
                    "reason": reason,
                })
                self.last_success = False
                self.last_error_counts = new_error_counts
            elif not should_revert:
                print(f"[guard] Baseline failure {reason} - allowing edits")
                emit_event("dashboard.guard.degraded", {
                    "files": files,
                    "error": error_summary,
                    "baseline_error_count": len(previous_counts),
                    "baseline_error_instances": baseline_error_instances,
                    "new_error_count": len(new_error_counts),
                    "new_error_instances": new_error_instances,
                    "reason": reason,
                })
                emit_event("dashboard.guard.fail", {
                    "files": files,
                    "error": error_summary,
                    "reverted": False,
                    "baseline_success": previous_success,
                    "baseline_error_count": len(previous_counts),
                    "baseline_error_instances": baseline_error_instances,
                    "new_error_count": len(new_error_counts),
                    "new_error_instances": new_error_instances,
                    "new_error_keys": summarize_error_counts(new_only),
                    "reason": reason,
                })
                self.last_success = False
                self.last_error_counts = new_error_counts
            else:
                # Revert the breaking changes
                reverted = []
                quarantined = []
                for filepath in files:
                    path = Path(filepath)
                    if revert_file(path):
                        reverted.append(filepath)
                        print(f"[guard] ↩ Reverted: {Path(filepath).name}")
                        # Update hash to reverted version
                        h = get_file_hash(path)
                        if h:
                            self.file_hashes[filepath] = h
                        continue
                    if path.exists() and is_untracked(path):
                        target = quarantine_file(path, reason)
                        if target is not None:
                            quarantined.append(str(target))
                            self.file_hashes.pop(filepath, None)
                revert_ok = len(reverted) + len(quarantined) == len(files)

                emit_event("dashboard.guard.revert", {
                    "files": reverted,
                    "quarantined": quarantined,
                    "reason": error_summary,
                    "revert_complete": revert_ok,
                    "baseline_success": previous_success,
                })
                emit_event("dashboard.guard.fail", {
                    "files": files,
                    "error": error_summary,
                    "reverted": True,
                    "baseline_success": previous_success,
                    "baseline_error_count": len(previous_counts),
                    "baseline_error_instances": baseline_error_instances,
                    "new_error_count": len(new_error_counts),
                    "new_error_instances": new_error_instances,
                    "new_error_keys": summarize_error_counts(new_only),
                    "reason": reason,
                })
                if revert_ok:
                    self.last_success = previous_success
                    self.last_error_counts = previous_counts
                else:
                    self.last_success = False
                    self.last_error_counts = new_error_counts


if HAS_WATCHDOG:
    class DashboardEventHandler(FileSystemEventHandler):
        """Watchdog event handler."""

        def __init__(self, guard: DashboardGuard):
            self.guard = guard

        def on_modified(self, event):
            if not event.is_directory:
                self.guard.on_file_change(event.src_path)

        def on_created(self, event):
            if not event.is_directory:
                self.guard.on_file_change(event.src_path)


def run_with_watchdog(guard: DashboardGuard) -> None:
    """Run using watchdog file observer."""
    handler = DashboardEventHandler(guard)
    observer = Observer()
    observer.schedule(handler, str(SRC_DIR), recursive=True)
    observer.start()
    print(f"[guard] Watching {SRC_DIR} with watchdog...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def run_with_polling(guard: DashboardGuard, interval: float = 2.0) -> None:
    """Run using polling fallback."""
    print(f"[guard] Watching {SRC_DIR} with polling (every {interval}s)...")

    try:
        while True:
            for ext in WATCHED_EXTENSIONS:
                for filepath in SRC_DIR.rglob(f"*{ext}"):
                    guard.on_file_change(str(filepath))
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Dashboard CI/CD Guard - Enforces typecheck before changes go live"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually revert files, just report",
    )
    parser.add_argument(
        "--no-revert",
        action="store_true",
        help="Don't revert files on failure (just emit events)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run typecheck once and exit",
    )
    args = parser.parse_args()

    guard = DashboardGuard(dry_run=args.dry_run, no_revert=args.no_revert, baseline=not args.once)

    # Emit start event
    emit_event("dashboard.guard.start", {
        "pid": os.getpid(),
        "mode": "dry-run" if args.dry_run else ("no-revert" if args.no_revert else "enforce"),
        "watchdog": HAS_WATCHDOG,
    })

    print("=" * 60)
    print("  DASHBOARD GUARD - CI/CD Enforcement Gate")
    print("=" * 60)
    print(f"  Mode: {'DRY RUN' if args.dry_run else ('NO REVERT' if args.no_revert else 'ENFORCE')}")
    print(f"  Watching: {SRC_DIR}")
    print(f"  Method: {'watchdog' if HAS_WATCHDOG else 'polling'}")
    print("=" * 60)

    if args.once:
        # Just run typecheck once
        success, output, duration = run_typecheck()
        print(f"\nTypecheck: {'PASS' if success else 'FAIL'} ({duration:.0f}ms)")
        if not success:
            print(output)
        sys.exit(0 if success else 1)

    # Run continuous watch
    if HAS_WATCHDOG:
        run_with_watchdog(guard)
    else:
        run_with_polling(guard)


if __name__ == "__main__":
    main()
