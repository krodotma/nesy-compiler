#!/usr/bin/env python3
"""
multi_repo_manager.py - Multi-Repo Manager (Step 28)

Handle multiple repositories for cross-repo research.
Manages repository registry, federated search, and cross-repo references.

PBTSO Phase: RESEARCH, PLAN

Bus Topics:
- a2a.research.repo.register
- a2a.research.repo.search
- research.repo.sync
- research.repo.stats

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class RepoType(Enum):
    """Type of repository."""
    LOCAL = "local"       # Local filesystem
    GIT = "git"           # Git repository
    REMOTE = "remote"     # Remote API-accessible


class RepoStatus(Enum):
    """Repository status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SYNCING = "syncing"
    ERROR = "error"


@dataclass
class RepoConfig:
    """Configuration for multi-repo manager."""

    registry_path: Optional[str] = None
    default_branch: str = "main"
    auto_sync: bool = True
    sync_interval_seconds: int = 3600  # 1 hour
    max_repos: int = 50
    parallel_search: bool = True
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.registry_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.registry_path = f"{pluribus_root}/.pluribus/research/repos.json"
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Repository:
    """A registered repository."""

    id: str
    name: str
    path: str
    repo_type: RepoType
    status: RepoStatus = RepoStatus.ACTIVE
    remote_url: Optional[str] = None
    branch: str = "main"
    last_sync: Optional[float] = None
    indexed_at: Optional[float] = None
    symbol_count: int = 0
    file_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "repo_type": self.repo_type.value,
            "status": self.status.value,
            "remote_url": self.remote_url,
            "branch": self.branch,
            "last_sync": self.last_sync,
            "indexed_at": self.indexed_at,
            "symbol_count": self.symbol_count,
            "file_count": self.file_count,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Repository":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            path=data["path"],
            repo_type=RepoType(data.get("repo_type", "local")),
            status=RepoStatus(data.get("status", "active")),
            remote_url=data.get("remote_url"),
            branch=data.get("branch", "main"),
            last_sync=data.get("last_sync"),
            indexed_at=data.get("indexed_at"),
            symbol_count=data.get("symbol_count", 0),
            file_count=data.get("file_count", 0),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CrossRepoResult:
    """Search result with repository context."""

    repo_id: str
    repo_name: str
    result: Dict[str, Any]
    full_path: str
    relevance_boost: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **self.result,
            "repo_id": self.repo_id,
            "repo_name": self.repo_name,
            "full_path": self.full_path,
        }


@dataclass
class FederatedSearchResult:
    """Result of a federated search across repos."""

    query: str
    results: List[CrossRepoResult]
    repos_searched: int
    total_time_ms: float
    by_repo: Dict[str, int]  # Results per repo

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "total_results": len(self.results),
            "repos_searched": self.repos_searched,
            "total_time_ms": self.total_time_ms,
            "by_repo": self.by_repo,
        }


# ============================================================================
# Multi-Repo Manager
# ============================================================================


class MultiRepoManager:
    """
    Manage multiple repositories for cross-repo research.

    Features:
    - Repository registry
    - Federated search across repos
    - Cross-repo reference resolution
    - Repository sync and indexing

    PBTSO Phase: RESEARCH, PLAN

    Example:
        manager = MultiRepoManager()

        # Register repositories
        manager.register_repo("/project1", name="project1")
        manager.register_repo("/project2", name="project2")

        # Federated search
        results = await manager.federated_search("UserService")
    """

    def __init__(
        self,
        config: Optional[RepoConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the multi-repo manager.

        Args:
            config: Manager configuration
            bus: AgentBus for event emission
        """
        self.config = config or RepoConfig()
        self.bus = bus or AgentBus()

        # Repository registry
        self._repos: Dict[str, Repository] = {}

        # Search handlers per repo (would connect to repo-specific indexes)
        self._search_handlers: Dict[str, Callable] = {}

        # Load existing registry
        self._load_registry()

    def register_repo(
        self,
        path: str,
        name: Optional[str] = None,
        repo_type: RepoType = RepoType.LOCAL,
        tags: Optional[List[str]] = None,
        remote_url: Optional[str] = None,
    ) -> Repository:
        """
        Register a repository.

        Args:
            path: Path to repository
            name: Display name (default: directory name)
            repo_type: Type of repository
            tags: Tags for filtering
            remote_url: Remote URL (for git repos)

        Returns:
            Registered Repository
        """
        import hashlib

        path = str(Path(path).resolve())
        name = name or Path(path).name

        # Generate ID
        repo_id = hashlib.sha256(path.encode()).hexdigest()[:12]

        # Detect git info
        branch = self.config.default_branch
        if repo_type == RepoType.GIT or (Path(path) / ".git").exists():
            repo_type = RepoType.GIT
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=path, capture_output=True, text=True
                )
                if result.returncode == 0:
                    branch = result.stdout.strip()

                if not remote_url:
                    result = subprocess.run(
                        ["git", "config", "--get", "remote.origin.url"],
                        cwd=path, capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        remote_url = result.stdout.strip()
            except Exception:
                pass

        repo = Repository(
            id=repo_id,
            name=name,
            path=path,
            repo_type=repo_type,
            remote_url=remote_url,
            branch=branch,
            tags=tags or [],
        )

        self._repos[repo_id] = repo
        self._save_registry()

        self._emit_with_lock({
            "topic": "a2a.research.repo.register",
            "kind": "repo",
            "data": repo.to_dict()
        })

        return repo

    def unregister_repo(self, repo_id: str) -> bool:
        """
        Unregister a repository.

        Args:
            repo_id: Repository ID

        Returns:
            True if removed
        """
        if repo_id in self._repos:
            del self._repos[repo_id]
            self._save_registry()
            return True
        return False

    def get_repo(self, repo_id: str) -> Optional[Repository]:
        """Get repository by ID."""
        return self._repos.get(repo_id)

    def get_repo_by_path(self, path: str) -> Optional[Repository]:
        """Get repository by path."""
        path = str(Path(path).resolve())
        for repo in self._repos.values():
            if repo.path == path:
                return repo
        return None

    def list_repos(
        self,
        tags: Optional[List[str]] = None,
        status: Optional[RepoStatus] = None,
    ) -> List[Repository]:
        """
        List registered repositories.

        Args:
            tags: Filter by tags
            status: Filter by status

        Returns:
            List of matching repositories
        """
        repos = list(self._repos.values())

        if tags:
            repos = [r for r in repos if any(t in r.tags for t in tags)]

        if status:
            repos = [r for r in repos if r.status == status]

        return repos

    async def federated_search(
        self,
        query: str,
        repo_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        max_per_repo: int = 20,
        search_func: Optional[Callable] = None,
    ) -> FederatedSearchResult:
        """
        Search across multiple repositories.

        Args:
            query: Search query
            repo_ids: Specific repos to search (None = all)
            tags: Filter repos by tags
            max_per_repo: Maximum results per repo
            search_func: Search function to use (async (repo, query) -> List[Dict])

        Returns:
            FederatedSearchResult
        """
        start_time = time.time()

        # Select repos to search
        if repo_ids:
            repos = [self._repos[rid] for rid in repo_ids if rid in self._repos]
        else:
            repos = self.list_repos(tags=tags, status=RepoStatus.ACTIVE)

        self._emit_with_lock({
            "topic": "a2a.research.repo.search",
            "kind": "search",
            "data": {"query": query, "repos": len(repos)}
        })

        all_results: List[CrossRepoResult] = []
        by_repo: Dict[str, int] = {}

        # Search function
        async def search_repo(repo: Repository) -> List[CrossRepoResult]:
            results = []

            if search_func:
                # Use provided search function
                raw_results = await search_func(repo, query)
            elif repo.id in self._search_handlers:
                # Use registered handler
                handler = self._search_handlers[repo.id]
                raw_results = await handler(query, max_per_repo)
            else:
                # Default: simple file search
                raw_results = await self._default_search(repo, query, max_per_repo)

            for result in raw_results[:max_per_repo]:
                # Build full path
                rel_path = result.get("path", "")
                full_path = str(Path(repo.path) / rel_path) if rel_path else repo.path

                results.append(CrossRepoResult(
                    repo_id=repo.id,
                    repo_name=repo.name,
                    result=result,
                    full_path=full_path,
                ))

            return results

        # Execute searches
        if self.config.parallel_search:
            tasks = [search_repo(repo) for repo in repos]
            results_lists = await asyncio.gather(*tasks, return_exceptions=True)

            for repo, results in zip(repos, results_lists):
                if isinstance(results, Exception):
                    by_repo[repo.id] = 0
                else:
                    all_results.extend(results)
                    by_repo[repo.id] = len(results)
        else:
            for repo in repos:
                try:
                    results = await search_repo(repo)
                    all_results.extend(results)
                    by_repo[repo.id] = len(results)
                except Exception:
                    by_repo[repo.id] = 0

        # Sort by score
        all_results.sort(
            key=lambda r: r.result.get("score", 0) * r.relevance_boost,
            reverse=True
        )

        total_time = (time.time() - start_time) * 1000

        return FederatedSearchResult(
            query=query,
            results=all_results,
            repos_searched=len(repos),
            total_time_ms=total_time,
            by_repo=by_repo,
        )

    async def sync_repo(self, repo_id: str) -> bool:
        """
        Sync a repository (git pull for git repos).

        Args:
            repo_id: Repository ID

        Returns:
            True if successful
        """
        repo = self._repos.get(repo_id)
        if not repo:
            return False

        repo.status = RepoStatus.SYNCING

        try:
            if repo.repo_type == RepoType.GIT:
                result = subprocess.run(
                    ["git", "pull", "--rebase"],
                    cwd=repo.path,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode != 0:
                    repo.status = RepoStatus.ERROR
                    return False

            repo.last_sync = time.time()
            repo.status = RepoStatus.ACTIVE
            self._save_registry()

            self._emit_with_lock({
                "topic": "research.repo.sync",
                "kind": "repo",
                "data": {"repo_id": repo_id, "success": True}
            })

            return True

        except Exception as e:
            repo.status = RepoStatus.ERROR
            self._emit_with_lock({
                "topic": "research.repo.sync",
                "kind": "error",
                "level": "error",
                "data": {"repo_id": repo_id, "error": str(e)}
            })
            return False

    async def sync_all(self) -> Dict[str, bool]:
        """
        Sync all active repositories.

        Returns:
            Dict mapping repo IDs to success status
        """
        results = {}
        for repo_id in self._repos:
            results[repo_id] = await self.sync_repo(repo_id)
        return results

    def register_search_handler(
        self,
        repo_id: str,
        handler: Callable[[str, int], List[Dict[str, Any]]],
    ) -> None:
        """
        Register a custom search handler for a repository.

        Args:
            repo_id: Repository ID
            handler: Async function (query, limit) -> List[results]
        """
        self._search_handlers[repo_id] = handler

    def update_repo_stats(
        self,
        repo_id: str,
        symbol_count: int,
        file_count: int,
    ) -> None:
        """
        Update repository statistics after indexing.

        Args:
            repo_id: Repository ID
            symbol_count: Number of symbols indexed
            file_count: Number of files indexed
        """
        if repo_id in self._repos:
            self._repos[repo_id].symbol_count = symbol_count
            self._repos[repo_id].file_count = file_count
            self._repos[repo_id].indexed_at = time.time()
            self._save_registry()

    def get_cross_repo_references(
        self,
        symbol_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Find cross-repo references to a symbol.

        Args:
            symbol_name: Symbol to find

        Returns:
            List of references with repo context
        """
        # This would integrate with reference resolver
        # For now, return empty list
        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        active = sum(1 for r in self._repos.values() if r.status == RepoStatus.ACTIVE)
        total_symbols = sum(r.symbol_count for r in self._repos.values())
        total_files = sum(r.file_count for r in self._repos.values())

        self._emit_with_lock({
            "topic": "research.repo.stats",
            "kind": "stats",
            "data": {
                "total_repos": len(self._repos),
                "active_repos": active,
            }
        })

        return {
            "total_repos": len(self._repos),
            "active_repos": active,
            "total_symbols": total_symbols,
            "total_files": total_files,
            "repos": [r.to_dict() for r in self._repos.values()],
        }

    # ========================================================================
    # Internal Methods
    # ========================================================================

    async def _default_search(
        self,
        repo: Repository,
        query: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Default search implementation using grep."""
        results = []

        try:
            # Use grep for simple text search
            process = await asyncio.create_subprocess_exec(
                "grep", "-r", "-n", "-l", "--include=*.py", "--include=*.ts",
                "--include=*.js", query, repo.path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=10)

            for line in stdout.decode().strip().split("\n")[:limit]:
                if line:
                    rel_path = str(Path(line).relative_to(repo.path))
                    results.append({
                        "path": rel_path,
                        "type": "file",
                        "score": 0.5,
                    })

        except Exception:
            pass

        return results

    def _load_registry(self) -> None:
        """Load repository registry from disk."""
        registry_path = Path(self.config.registry_path)
        if not registry_path.exists():
            return

        try:
            with open(registry_path) as f:
                data = json.load(f)

            for repo_data in data.get("repos", []):
                repo = Repository.from_dict(repo_data)
                self._repos[repo.id] = repo

        except Exception:
            pass

    def _save_registry(self) -> None:
        """Save repository registry to disk."""
        registry_path = Path(self.config.registry_path)
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "repos": [r.to_dict() for r in self._repos.values()],
            "saved_at": time.time(),
        }

        with open(registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def _emit_with_lock(self, event: Dict[str, Any]) -> str:
        """Emit event with file locking."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        import socket
        import uuid

        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": event.get("topic", "unknown"),
            "kind": event.get("kind", "event"),
            "level": event.get("level", "info"),
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": event.get("data", {}),
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(full_event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Multi-Repo Manager."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Repo Manager (Step 28)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register command
    reg_parser = subparsers.add_parser("register", help="Register a repository")
    reg_parser.add_argument("path", help="Path to repository")
    reg_parser.add_argument("--name", help="Display name")
    reg_parser.add_argument("--tags", nargs="+", help="Tags")

    # List command
    list_parser = subparsers.add_parser("list", help="List repositories")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Search command
    search_parser = subparsers.add_parser("search", help="Federated search")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--repos", nargs="+", help="Specific repo IDs")
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync repositories")
    sync_parser.add_argument("--repo", help="Specific repo ID (or all)")

    # Unregister command
    unreg_parser = subparsers.add_parser("unregister", help="Unregister a repository")
    unreg_parser.add_argument("repo_id", help="Repository ID")

    args = parser.parse_args()

    manager = MultiRepoManager()

    if args.command == "register":
        repo = manager.register_repo(
            args.path,
            name=args.name,
            tags=args.tags,
        )
        print(f"Registered: {repo.name} ({repo.id})")
        print(f"  Path: {repo.path}")
        print(f"  Type: {repo.repo_type.value}")

    elif args.command == "list":
        repos = manager.list_repos()
        if args.json:
            print(json.dumps([r.to_dict() for r in repos], indent=2))
        else:
            print(f"Registered Repositories ({len(repos)}):")
            for r in repos:
                print(f"  [{r.id[:8]}] {r.name}")
                print(f"    Path: {r.path}")
                print(f"    Status: {r.status.value}")

    elif args.command == "search":
        async def run_search():
            result = await manager.federated_search(
                args.query,
                repo_ids=args.repos,
            )
            return result

        result = asyncio.run(run_search())

        if args.json:
            print(json.dumps({
                **result.to_dict(),
                "results": [r.to_dict() for r in result.results[:20]],
            }, indent=2))
        else:
            print(f"Federated Search: '{args.query}'")
            print(f"Searched {result.repos_searched} repos in {result.total_time_ms:.1f}ms")
            print(f"\nResults ({len(result.results)}):")
            for r in result.results[:20]:
                print(f"  [{r.repo_name}] {r.result.get('path', 'unknown')}")

    elif args.command == "sync":
        async def run_sync():
            if args.repo:
                return {args.repo: await manager.sync_repo(args.repo)}
            else:
                return await manager.sync_all()

        results = asyncio.run(run_sync())
        for repo_id, success in results.items():
            status = "OK" if success else "FAILED"
            print(f"  {repo_id}: {status}")

    elif args.command == "unregister":
        if manager.unregister_repo(args.repo_id):
            print(f"Unregistered: {args.repo_id}")
        else:
            print(f"Repository not found: {args.repo_id}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
