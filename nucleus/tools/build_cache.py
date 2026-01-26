#!/usr/bin/env python3
"""
build_cache.py - Nx-Equivalent Build Artifact Cache

First-principles caching implementation for Pluribus.
Provides build artifact caching without external dependencies.

DKIN v29 | PAIP v15 | CITIZEN v1

Principle: "Same hash = same output" (Nx caching philosophy)

Hash includes:
- Project source files (content hash)
- Dependency versions (lockfile hash)
- Configuration files
- Runtime environment (NODE_VERSION, PYTHON_VERSION)
- Command-line arguments

Usage:
    python3 build_cache.py hash <project>                # Compute cache key
    python3 build_cache.py check <project> <target>      # Check if cached
    python3 build_cache.py store <project> <target> <path>  # Store artifact
    python3 build_cache.py restore <project> <target> <dest> # Restore cached
    python3 build_cache.py status                        # Show cache status
    python3 build_cache.py clean --older-than 7d         # Clean old entries

Reference: nucleus/specs/projects.json
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.dont_write_bytecode = True

# =============================================================================
# Constants
# =============================================================================

PROTOCOL_VERSION = "29"
OPERATOR_NAME = "PBCACHE"
CACHE_VERSION = "1"

# Paths
TOOLS_DIR = Path(__file__).parent.resolve()
NUCLEUS_DIR = TOOLS_DIR.parent
REPO_ROOT = NUCLEUS_DIR.parent
PROJECTS_JSON = NUCLEUS_DIR / "specs" / "projects.json"
DEFAULT_CACHE_DIR = REPO_ROOT / ".pluribus" / "cache" / "builds"

# Hash computation
HASH_ALGORITHM = "sha256"
CHUNK_SIZE = 65536  # 64KB chunks for file hashing


# =============================================================================
# Helpers
# =============================================================================

def now_ts() -> float:
    return time.time()


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def format_size(bytes_: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_ < 1024:
            return f"{bytes_:.1f}{unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f}TB"


def resolve_bus_dir() -> Path:
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "").strip()
    if bus_dir:
        return Path(bus_dir).expanduser().resolve()
    root = os.environ.get("PLURIBUS_ROOT") or str(REPO_ROOT)
    return Path(root) / ".pluribus" / "bus"


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "pbcache"


# =============================================================================
# File I/O
# =============================================================================

try:
    import fcntl
    def lock_file(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    def unlock_file(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
except ImportError:
    def lock_file(handle) -> None: pass
    def unlock_file(handle) -> None: pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        lock_file(f)
        try:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        finally:
            unlock_file(f)


def emit_bus_event(topic: str, kind: str, level: str, actor: str, data: dict) -> str:
    evt_id = str(uuid.uuid4())
    evt = {
        "id": evt_id,
        "ts": now_ts(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    bus_path = resolve_bus_dir() / "events.ndjson"
    append_ndjson(bus_path, evt)
    return evt_id


# =============================================================================
# Hash Computation
# =============================================================================

def hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def hash_string(s: str) -> str:
    """Compute SHA256 hash of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def hash_dict(d: dict) -> str:
    """Compute deterministic hash of a dictionary."""
    # Sort keys for deterministic ordering
    s = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hash_string(s)


# =============================================================================
# Project Loading
# =============================================================================

@dataclass
class ProjectConfig:
    """Project configuration for cache computation."""
    id: str
    root: str
    inputs: list[str]
    targets: dict[str, dict]
    dependencies: list[str]


def load_project_config(project_id: str) -> Optional[ProjectConfig]:
    """Load project configuration from projects.json."""
    if not PROJECTS_JSON.exists():
        return None

    data = json.loads(PROJECTS_JSON.read_text(encoding="utf-8"))
    projects = data.get("projects", {})

    if project_id not in projects:
        return None

    proj = projects[project_id]
    return ProjectConfig(
        id=proj.get("id", project_id),
        root=proj.get("root", project_id),
        inputs=proj.get("inputs", []),
        targets=proj.get("targets", {}),
        dependencies=proj.get("dependencies", []),
    )


def get_project_files(project: ProjectConfig, repo_root: Path) -> list[Path]:
    """Get all files matching project input patterns."""
    files = []
    project_dir = repo_root / project.root

    for pattern in project.inputs:
        # Handle glob patterns
        if "**" in pattern or "*" in pattern:
            # Convert to path pattern
            base_dir = repo_root
            parts = pattern.split("/")
            # Find the fixed prefix
            fixed_parts = []
            for p in parts:
                if "*" in p:
                    break
                fixed_parts.append(p)
            if fixed_parts:
                base_dir = repo_root / "/".join(fixed_parts)

            # Glob from base
            if base_dir.exists():
                glob_pattern = "/".join(parts[len(fixed_parts):]) or "*"
                for f in base_dir.rglob(glob_pattern.replace("**/*", "*").replace("**", "*")):
                    if f.is_file():
                        files.append(f)
        else:
            # Exact path
            p = repo_root / pattern
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file():
                        files.append(f)

    # Also include files directly in project root
    if project_dir.exists():
        for f in project_dir.rglob("*"):
            if f.is_file() and f not in files:
                # Skip common non-source files
                if not any(p in str(f) for p in ["node_modules", "__pycache__", ".git", "dist", "coverage"]):
                    files.append(f)

    return sorted(set(files))


# =============================================================================
# Cache Key Computation
# =============================================================================

@dataclass
class CacheKey:
    """Cache key with all contributing factors."""
    project_id: str
    target: str
    source_hash: str
    deps_hash: str
    config_hash: str
    env_hash: str
    full_hash: str

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "target": self.target,
            "source_hash": self.source_hash[:16],
            "deps_hash": self.deps_hash[:16],
            "config_hash": self.config_hash[:16],
            "env_hash": self.env_hash[:16],
            "full_hash": self.full_hash,
        }


def compute_source_hash(project: ProjectConfig, repo_root: Path) -> str:
    """Compute hash of all source files."""
    files = get_project_files(project, repo_root)

    # Create deterministic file list with hashes
    file_hashes = []
    for f in sorted(files):
        rel_path = f.relative_to(repo_root)
        content_hash = hash_file(f)
        file_hashes.append(f"{rel_path}:{content_hash}")

    return hash_string("\n".join(file_hashes))


def compute_deps_hash(project: ProjectConfig, repo_root: Path) -> str:
    """Compute hash of dependency lockfiles."""
    lockfiles = [
        repo_root / "package-lock.json",
        repo_root / "yarn.lock",
        repo_root / "pnpm-lock.yaml",
        repo_root / "requirements.txt",
        repo_root / "requirements-dev.txt",
        repo_root / "pyproject.toml",
        repo_root / "poetry.lock",
        repo_root / project.root / "package-lock.json",
        repo_root / project.root / "requirements.txt",
    ]

    lockfile_hashes = []
    for lf in lockfiles:
        if lf.exists():
            h = hash_file(lf)
            lockfile_hashes.append(f"{lf.name}:{h}")

    return hash_string("\n".join(sorted(lockfile_hashes))) if lockfile_hashes else hash_string("no-deps")


def compute_config_hash(project: ProjectConfig, target: str, repo_root: Path) -> str:
    """Compute hash of configuration files."""
    config_files = [
        repo_root / "nx.json",
        repo_root / "tsconfig.json",
        repo_root / "pyproject.toml",
        repo_root / project.root / "tsconfig.json",
        repo_root / project.root / "package.json",
        repo_root / project.root / "project.json",
        PROJECTS_JSON,
    ]

    config_hashes = []
    for cf in config_files:
        if cf.exists():
            h = hash_file(cf)
            config_hashes.append(f"{cf.name}:{h}")

    # Include target config
    target_config = project.targets.get(target, {})
    config_hashes.append(f"target:{hash_dict(target_config)}")

    return hash_string("\n".join(sorted(config_hashes)))


def compute_env_hash() -> str:
    """Compute hash of runtime environment."""
    env_vars = {
        "NODE_VERSION": os.environ.get("NODE_VERSION", ""),
        "PYTHON_VERSION": f"{sys.version_info.major}.{sys.version_info.minor}",
        "CACHE_VERSION": CACHE_VERSION,
    }

    # Check actual node version
    try:
        import subprocess
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            env_vars["NODE_VERSION"] = result.stdout.strip()
    except Exception:
        pass

    return hash_dict(env_vars)


def compute_cache_key(project_id: str, target: str, repo_root: Path) -> Optional[CacheKey]:
    """Compute full cache key for a project target."""
    project = load_project_config(project_id)
    if not project:
        return None

    if target not in project.targets:
        return None

    source_hash = compute_source_hash(project, repo_root)
    deps_hash = compute_deps_hash(project, repo_root)
    config_hash = compute_config_hash(project, target, repo_root)
    env_hash = compute_env_hash()

    # Combine all hashes
    combined = f"{source_hash}:{deps_hash}:{config_hash}:{env_hash}"
    full_hash = hash_string(combined)

    return CacheKey(
        project_id=project_id,
        target=target,
        source_hash=source_hash,
        deps_hash=deps_hash,
        config_hash=config_hash,
        env_hash=env_hash,
        full_hash=full_hash,
    )


# =============================================================================
# Cache Storage
# =============================================================================

@dataclass
class CacheEntry:
    """Cached build artifact entry."""
    hash: str
    project_id: str
    target: str
    created_at: float
    size_bytes: int
    artifact_path: Path

    @property
    def age(self) -> float:
        return now_ts() - self.created_at

    def to_dict(self) -> dict:
        return {
            "hash": self.hash,
            "project_id": self.project_id,
            "target": self.target,
            "created_at": self.created_at,
            "created_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.created_at)),
            "size_bytes": self.size_bytes,
            "age_s": self.age,
        }


class BuildCache:
    """Build artifact cache manager."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        ensure_dir(self.cache_dir)
        self.index_file = self.cache_dir / "index.json"

    def _load_index(self) -> dict:
        """Load cache index."""
        if self.index_file.exists():
            try:
                return json.loads(self.index_file.read_text())
            except Exception:
                pass
        return {"version": CACHE_VERSION, "entries": {}}

    def _save_index(self, index: dict) -> None:
        """Save cache index."""
        self.index_file.write_text(json.dumps(index, indent=2))

    def _entry_path(self, cache_hash: str) -> Path:
        """Get path for cache entry."""
        # Use first 2 chars as directory for sharding
        return self.cache_dir / cache_hash[:2] / f"{cache_hash}.tar.gz"

    def _meta_path(self, cache_hash: str) -> Path:
        """Get path for cache metadata."""
        return self.cache_dir / cache_hash[:2] / f"{cache_hash}.meta.json"

    def check(self, cache_key: CacheKey) -> Optional[CacheEntry]:
        """Check if cache exists for key."""
        entry_path = self._entry_path(cache_key.full_hash)
        meta_path = self._meta_path(cache_key.full_hash)

        if not entry_path.exists() or not meta_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text())
            return CacheEntry(
                hash=cache_key.full_hash,
                project_id=meta["project_id"],
                target=meta["target"],
                created_at=meta["created_at"],
                size_bytes=entry_path.stat().st_size,
                artifact_path=entry_path,
            )
        except Exception:
            return None

    def store(self, cache_key: CacheKey, artifact_path: Path) -> CacheEntry:
        """Store artifact in cache."""
        entry_path = self._entry_path(cache_key.full_hash)
        meta_path = self._meta_path(cache_key.full_hash)

        ensure_dir(entry_path.parent)

        # Create tarball
        with tarfile.open(entry_path, "w:gz") as tar:
            if artifact_path.is_dir():
                for f in artifact_path.rglob("*"):
                    if f.is_file():
                        arcname = f.relative_to(artifact_path)
                        tar.add(f, arcname=str(arcname))
            else:
                tar.add(artifact_path, arcname=artifact_path.name)

        # Save metadata
        meta = {
            "hash": cache_key.full_hash,
            "project_id": cache_key.project_id,
            "target": cache_key.target,
            "created_at": now_ts(),
            "key": cache_key.to_dict(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return CacheEntry(
            hash=cache_key.full_hash,
            project_id=cache_key.project_id,
            target=cache_key.target,
            created_at=meta["created_at"],
            size_bytes=entry_path.stat().st_size,
            artifact_path=entry_path,
        )

    def restore(self, cache_key: CacheKey, dest_path: Path) -> bool:
        """Restore cached artifact to destination."""
        entry = self.check(cache_key)
        if not entry:
            return False

        ensure_dir(dest_path)

        try:
            with tarfile.open(entry.artifact_path, "r:gz") as tar:
                tar.extractall(dest_path)
            return True
        except Exception:
            return False

    def list_entries(self) -> list[CacheEntry]:
        """List all cache entries."""
        entries = []
        for meta_file in self.cache_dir.rglob("*.meta.json"):
            try:
                meta = json.loads(meta_file.read_text())
                entry_path = meta_file.with_suffix("").with_suffix(".tar.gz")
                if entry_path.exists():
                    entries.append(CacheEntry(
                        hash=meta["hash"],
                        project_id=meta["project_id"],
                        target=meta["target"],
                        created_at=meta["created_at"],
                        size_bytes=entry_path.stat().st_size,
                        artifact_path=entry_path,
                    ))
            except Exception:
                pass
        return sorted(entries, key=lambda e: e.created_at, reverse=True)

    def clean(self, older_than_s: float) -> list[str]:
        """Clean entries older than specified age."""
        cutoff = now_ts() - older_than_s
        cleaned = []

        for meta_file in self.cache_dir.rglob("*.meta.json"):
            try:
                meta = json.loads(meta_file.read_text())
                if meta.get("created_at", 0) < cutoff:
                    entry_path = meta_file.with_suffix("").with_suffix(".tar.gz")
                    if entry_path.exists():
                        entry_path.unlink()
                    meta_file.unlink()
                    cleaned.append(meta["hash"])
            except Exception:
                pass

        return cleaned

    def get_stats(self) -> dict:
        """Get cache statistics."""
        entries = self.list_entries()
        total_size = sum(e.size_bytes for e in entries)

        return {
            "total_entries": len(entries),
            "total_size_bytes": total_size,
            "total_size_human": format_size(total_size),
            "oldest_entry_age": format_duration(entries[-1].age) if entries else "N/A",
            "newest_entry_age": format_duration(entries[0].age) if entries else "N/A",
            "cache_dir": str(self.cache_dir),
        }


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_cache.py",
        description="PBCACHE - Nx-equivalent build artifact cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute cache key for a project target
  python3 build_cache.py hash nucleus/dashboard build

  # Check if cached
  python3 build_cache.py check nucleus/dashboard build

  # Store build artifact
  python3 build_cache.py store nucleus/dashboard build ./dist

  # Restore cached artifact
  python3 build_cache.py restore nucleus/dashboard build ./dist

  # Show cache status
  python3 build_cache.py status

  # Clean old entries
  python3 build_cache.py clean --older-than 7d
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # hash
    p_hash = subparsers.add_parser("hash", help="Compute cache key")
    p_hash.add_argument("project", help="Project ID (e.g., nucleus/dashboard)")
    p_hash.add_argument("target", help="Target name (e.g., build)")
    p_hash.add_argument("--json", action="store_true", help="JSON output")

    # check
    p_check = subparsers.add_parser("check", help="Check if cached")
    p_check.add_argument("project", help="Project ID")
    p_check.add_argument("target", help="Target name")

    # store
    p_store = subparsers.add_parser("store", help="Store artifact")
    p_store.add_argument("project", help="Project ID")
    p_store.add_argument("target", help="Target name")
    p_store.add_argument("path", help="Artifact path to store")
    p_store.add_argument("--emit-bus", action="store_true", help="Emit to bus")

    # restore
    p_restore = subparsers.add_parser("restore", help="Restore artifact")
    p_restore.add_argument("project", help="Project ID")
    p_restore.add_argument("target", help="Target name")
    p_restore.add_argument("dest", help="Destination path")
    p_restore.add_argument("--emit-bus", action="store_true", help="Emit to bus")

    # status
    p_status = subparsers.add_parser("status", help="Show cache status")
    p_status.add_argument("--json", action="store_true", help="JSON output")

    # clean
    p_clean = subparsers.add_parser("clean", help="Clean old entries")
    p_clean.add_argument("--older-than", default="7d", help="Age threshold (e.g., 7d, 24h)")
    p_clean.add_argument("--dry-run", action="store_true", help="Preview without deleting")

    return parser


def parse_duration(s: str) -> float:
    """Parse duration string to seconds."""
    s = s.strip().lower()
    if s.endswith("d"):
        return float(s[:-1]) * 86400
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s) * 86400  # Default to days


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cache = BuildCache()

    if args.command == "hash":
        key = compute_cache_key(args.project, args.target, REPO_ROOT)
        if not key:
            print(f"Error: Project or target not found: {args.project}:{args.target}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(key.to_dict(), indent=2))
        else:
            print(f"Cache Key: {key.full_hash}")
            print(f"  Source:  {key.source_hash[:16]}...")
            print(f"  Deps:    {key.deps_hash[:16]}...")
            print(f"  Config:  {key.config_hash[:16]}...")
            print(f"  Env:     {key.env_hash[:16]}...")

    elif args.command == "check":
        key = compute_cache_key(args.project, args.target, REPO_ROOT)
        if not key:
            print(f"Error: Project or target not found", file=sys.stderr)
            return 1
        entry = cache.check(key)
        if entry:
            print(f"CACHE HIT: {key.full_hash[:16]}")
            print(f"  Created: {format_duration(entry.age)} ago")
            print(f"  Size: {format_size(entry.size_bytes)}")
            return 0
        else:
            print(f"CACHE MISS: {key.full_hash[:16]}")
            return 1

    elif args.command == "store":
        key = compute_cache_key(args.project, args.target, REPO_ROOT)
        if not key:
            print(f"Error: Project or target not found", file=sys.stderr)
            return 1
        artifact_path = Path(args.path)
        if not artifact_path.exists():
            print(f"Error: Artifact path not found: {artifact_path}", file=sys.stderr)
            return 1
        entry = cache.store(key, artifact_path)
        print(f"STORED: {key.full_hash[:16]} ({format_size(entry.size_bytes)})")
        if args.emit_bus:
            emit_bus_event(
                topic="operator.pbcache.store",
                kind="log",
                level="info",
                actor=default_actor(),
                data=entry.to_dict(),
            )

    elif args.command == "restore":
        key = compute_cache_key(args.project, args.target, REPO_ROOT)
        if not key:
            print(f"Error: Project or target not found", file=sys.stderr)
            return 1
        dest_path = Path(args.dest)
        if cache.restore(key, dest_path):
            print(f"RESTORED: {key.full_hash[:16]} -> {dest_path}")
            if args.emit_bus:
                emit_bus_event(
                    topic="operator.pbcache.restore",
                    kind="log",
                    level="info",
                    actor=default_actor(),
                    data={"hash": key.full_hash, "dest": str(dest_path)},
                )
            return 0
        else:
            print(f"RESTORE FAILED: Cache miss or error")
            return 1

    elif args.command == "status":
        stats = cache.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("PBCACHE Status")
            print("=" * 40)
            print(f"  Entries:  {stats['total_entries']}")
            print(f"  Size:     {stats['total_size_human']}")
            print(f"  Oldest:   {stats['oldest_entry_age']}")
            print(f"  Newest:   {stats['newest_entry_age']}")
            print(f"  Location: {stats['cache_dir']}")

    elif args.command == "clean":
        older_than_s = parse_duration(args.older_than)
        if args.dry_run:
            entries = cache.list_entries()
            to_clean = [e for e in entries if e.age > older_than_s]
            print(f"Would clean {len(to_clean)} entries older than {args.older_than}")
            for e in to_clean[:10]:
                print(f"  {e.hash[:16]} ({format_duration(e.age)} old)")
            if len(to_clean) > 10:
                print(f"  ... and {len(to_clean) - 10} more")
        else:
            cleaned = cache.clean(older_than_s)
            print(f"Cleaned {len(cleaned)} entries older than {args.older_than}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
