#!/usr/bin/env python3
"""
Pluribus /tmp Janitorial Service - tmpjanitor.py

Aggressive cleanup of /tmp to prevent VPS boot hangs.
Enforces 100MB limit on Pluribus temp files.

Run via:
- systemd timer (every 5 min)
- cron fallback (every 10 min)
- Manual: python3 tmpjanitor.py

PLURIBUS v1 compliant - emits bus events for observability.
"""

import os
import sys
import time
import json
import uuid
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

# === CONFIGURATION ===

CONFIG = {
    # Maximum total size for Pluribus temp files (100MB)
    "max_total_size_mb": 100,
    
    # Maximum age for any temp file (1 hour)
    "max_file_age_hours": 1,
    
    # Directories to clean (beyond /tmp)
    "cleanup_paths": [
        "/tmp",
        "/tmp/pluribus_*",
        "/var/tmp",
        "/pluribus/.pluribus/tmp",
    ],
    
    # Patterns to aggressively delete (no age check)
    "aggressive_patterns": [
        "*.log.*.gz",      # Rotated logs
        "*.ndjson.bak",    # Bus backups
        "core.*",          # Core dumps
        "npm-*",           # NPM cache
        "vite-*",          # Vite temp
        ".vite-*",
        "playwright-*",    # Test artifacts
    ],
    
    # Patterns to preserve (never delete)
    "preserve_patterns": [
        ".X*-lock",        # X11 locks
        ".ICE-*",          # ICE connections
        "ssh-*",           # SSH agent
        "systemd-*",       # Systemd runtime
    ],
    
    # Bus event path
    "bus_path": "/pluribus/.pluribus/bus/events.ndjson",
    
    # Log path
    "log_path": "/var/log/pluribus-tmpjanitor.log",
}

# === LOGGING ===

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [tmpjanitor] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CONFIG["log_path"], mode="a") if os.path.exists(os.path.dirname(CONFIG["log_path"])) else logging.NullHandler(),
    ]
)
logger = logging.getLogger("tmpjanitor")


def emit_bus_event(topic: str, level: str, data: Dict[str, Any]) -> None:
    """Emit event to Pluribus bus."""
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": datetime.utcnow().isoformat() + "Z",
        "topic": topic,
        "kind": "metric",
        "level": level,
        "actor": "tmpjanitor",
        "proto": "PLURIBUS v1",
        "data": data
    }
    try:
        bus_path = Path(CONFIG["bus_path"])
        if bus_path.parent.exists():
            with open(bus_path, "a") as f:
                f.write(json.dumps(event) + "\n")
    except Exception:
        pass  # Silent fail on bus emit


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError):
        pass
    return total


def get_files_by_age(path: Path) -> List[Tuple[Path, float, int]]:
    """Get files sorted by modification time (oldest first).
    Returns: [(path, mtime, size), ...]
    """
    files = []
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    stat = entry.stat()
                    files.append((entry, stat.st_mtime, stat.st_size))
                except (OSError, PermissionError):
                    pass
    except (OSError, PermissionError):
        pass
    
    # Sort by mtime (oldest first)
    files.sort(key=lambda x: x[1])
    return files


def should_preserve(path: Path) -> bool:
    """Check if file matches preserve patterns."""
    name = path.name
    for pattern in CONFIG["preserve_patterns"]:
        if pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif pattern.endswith("*"):
            if name.startswith(pattern[:-1]):
                return True
        elif name == pattern:
            return True
    return False


def is_aggressive_target(path: Path) -> bool:
    """Check if file matches aggressive cleanup patterns."""
    name = path.name
    for pattern in CONFIG["aggressive_patterns"]:
        if pattern.startswith("*") and pattern.endswith("*"):
            if pattern[1:-1] in name:
                return True
        elif pattern.startswith("*"):
            if name.endswith(pattern[1:]):
                return True
        elif pattern.endswith("*"):
            if name.startswith(pattern[:-1]):
                return True
        elif name == pattern:
            return True
    return False


def cleanup_path(base_path: str) -> Dict[str, Any]:
    """Clean up a single path, enforcing size and age limits.
    Returns stats dict.
    """
    stats = {
        "path": base_path,
        "files_deleted": 0,
        "bytes_freed": 0,
        "files_preserved": 0,
        "errors": 0,
    }
    
    # Handle glob patterns
    if "*" in base_path:
        from glob import glob
        paths = [Path(p) for p in glob(base_path)]
    else:
        paths = [Path(base_path)]
    
    for path in paths:
        if not path.exists():
            continue
        
        max_age = time.time() - (CONFIG["max_file_age_hours"] * 3600)
        max_size_bytes = CONFIG["max_total_size_mb"] * 1024 * 1024
        
        current_size = get_dir_size(path)
        files = get_files_by_age(path)
        
        for file_path, mtime, size in files:
            if should_preserve(file_path):
                stats["files_preserved"] += 1
                continue
            
            should_delete = False
            reason = ""
            
            # Aggressive patterns - delete immediately
            if is_aggressive_target(file_path):
                should_delete = True
                reason = "aggressive_pattern"
            
            # Age check
            elif mtime < max_age:
                should_delete = True
                reason = "age_exceeded"
            
            # Size check (delete oldest first to get under limit)
            elif current_size > max_size_bytes:
                should_delete = True
                reason = "size_limit"
            
            if should_delete:
                try:
                    if file_path.is_dir():
                        shutil.rmtree(file_path)
                    else:
                        file_path.unlink()
                    
                    stats["files_deleted"] += 1
                    stats["bytes_freed"] += size
                    current_size -= size
                    
                    logger.debug(f"Deleted: {file_path} ({reason})")
                except Exception as e:
                    stats["errors"] += 1
                    logger.warning(f"Failed to delete {file_path}: {e}")
    
    return stats


def cleanup_all() -> Dict[str, Any]:
    """Run cleanup on all configured paths."""
    start_time = time.time()
    
    total_stats = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "files_deleted": 0,
        "bytes_freed": 0,
        "files_preserved": 0,
        "errors": 0,
        "paths_cleaned": [],
    }
    
    for path in CONFIG["cleanup_paths"]:
        stats = cleanup_path(path)
        total_stats["files_deleted"] += stats["files_deleted"]
        total_stats["bytes_freed"] += stats["bytes_freed"]
        total_stats["files_preserved"] += stats["files_preserved"]
        total_stats["errors"] += stats["errors"]
        total_stats["paths_cleaned"].append(stats)
    
    total_stats["duration_seconds"] = round(time.time() - start_time, 2)
    total_stats["bytes_freed_mb"] = round(total_stats["bytes_freed"] / (1024 * 1024), 2)
    
    # Log summary
    logger.info(
        f"Cleanup complete: {total_stats['files_deleted']} files deleted, "
        f"{total_stats['bytes_freed_mb']}MB freed, "
        f"{total_stats['errors']} errors"
    )
    
    # Emit bus event
    emit_bus_event(
        topic="system.tmpjanitor.cleanup",
        level="info" if total_stats["errors"] == 0 else "warn",
        data=total_stats
    )
    
    return total_stats


def check_tmp_health() -> Dict[str, Any]:
    """Check /tmp health without cleaning."""
    tmp_path = Path("/tmp")
    size_bytes = get_dir_size(tmp_path)
    size_mb = size_bytes / (1024 * 1024)
    
    file_count = sum(1 for _ in tmp_path.rglob("*") if _.is_file())
    
    health = {
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 2),
        "file_count": file_count,
        "limit_mb": CONFIG["max_total_size_mb"],
        "over_limit": size_mb > CONFIG["max_total_size_mb"],
        "status": "critical" if size_mb > 500 else "warning" if size_mb > 100 else "healthy",
    }
    
    return health


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pluribus /tmp Janitorial Service")
    parser.add_argument("--check", action="store_true", help="Check /tmp health without cleaning")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument("--force", action="store_true", help="Force cleanup even if under limit")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    
    args = parser.parse_args()
    
    if args.check:
        health = check_tmp_health()
        if args.json:
            print(json.dumps(health, indent=2))
        else:
            status_icon = {"healthy": "✓", "warning": "⚠", "critical": "✗"}[health["status"]]
            print(f"/tmp Health: {status_icon} {health['status'].upper()}")
            print(f"  Size: {health['size_mb']} MB / {health['limit_mb']} MB limit")
            print(f"  Files: {health['file_count']}")
        sys.exit(0 if health["status"] == "healthy" else 1)
    
    stats = cleanup_all()
    
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(f"Cleanup complete:")
        print(f"  Files deleted: {stats['files_deleted']}")
        print(f"  Space freed: {stats['bytes_freed_mb']} MB")
        print(f"  Duration: {stats['duration_seconds']}s")
        if stats["errors"]:
            print(f"  Errors: {stats['errors']}")


if __name__ == "__main__":
    main()
