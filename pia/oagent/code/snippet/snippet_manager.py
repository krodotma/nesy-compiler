#!/usr/bin/env python3
"""
snippet_manager.py - Code Snippet Manager (Step 73)

PBTSO Phase: SKILL

Provides:
- Snippet storage and retrieval
- Semantic snippet search
- Snippet versioning
- Snippet sharing across agents
- Context-aware snippet suggestions

Bus Topics:
- code.snippet.add
- code.snippet.search
- code.snippet.used

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class SnippetCategory(Enum):
    """Categories for code snippets."""
    ALGORITHM = "algorithm"
    DATA_STRUCTURE = "data_structure"
    UTILITY = "utility"
    PATTERN = "pattern"
    BOILERPLATE = "boilerplate"
    TEST = "test"
    CONFIG = "config"
    API = "api"
    DATABASE = "database"
    SECURITY = "security"
    LOGGING = "logging"
    ERROR_HANDLING = "error_handling"
    ASYNC = "async"
    CUSTOM = "custom"


@dataclass
class SnippetConfig:
    """Configuration for the snippet manager."""
    storage_path: str = "/pluribus/.pluribus/snippets"
    enable_versioning: bool = True
    max_versions: int = 10
    enable_sharing: bool = True
    enable_semantic_search: bool = True
    index_on_add: bool = True
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "storage_path": self.storage_path,
            "enable_versioning": self.enable_versioning,
            "max_versions": self.max_versions,
            "enable_sharing": self.enable_sharing,
            "enable_semantic_search": self.enable_semantic_search,
        }


# =============================================================================
# Types
# =============================================================================

@dataclass
class CodeSnippet:
    """A reusable code snippet."""
    id: str
    name: str
    code: str
    language: str
    category: SnippetCategory = SnippetCategory.CUSTOM
    description: str = ""
    tags: List[str] = field(default_factory=list)
    author: str = ""
    version: int = 1
    dependencies: List[str] = field(default_factory=list)
    usage_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        """Get content hash for deduplication."""
        return hashlib.sha256(self.code.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "language": self.language,
            "category": self.category.value,
            "description": self.description,
            "tags": self.tags,
            "author": self.author,
            "version": self.version,
            "dependencies": self.dependencies,
            "usage_count": self.usage_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "hash": self.hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeSnippet":
        """Create snippet from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            code=data["code"],
            language=data["language"],
            category=SnippetCategory(data.get("category", "custom")),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            author=data.get("author", ""),
            version=data.get("version", 1),
            dependencies=data.get("dependencies", []),
            usage_count=data.get("usage_count", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SnippetSearchResult:
    """Result of a snippet search."""
    snippet: CodeSnippet
    score: float
    match_type: str  # "exact", "tag", "semantic", "fuzzy"
    highlights: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snippet": self.snippet.to_dict(),
            "score": self.score,
            "match_type": self.match_type,
            "highlights": self.highlights,
        }


@dataclass
class SnippetVersion:
    """A version of a snippet."""
    version: int
    code: str
    updated_at: float
    changelog: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "code": self.code,
            "updated_at": self.updated_at,
            "changelog": self.changelog,
        }


# =============================================================================
# Agent Bus with File Locking
# =============================================================================

class LockedAgentBus:
    """Agent bus with file locking for safe concurrent writes."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        line = json.dumps(full_event, ensure_ascii=False, separators=(",", ":")) + "\n"

        fd = os.open(str(self.bus_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        return event_id


# =============================================================================
# Snippet Manager
# =============================================================================

class SnippetManager:
    """
    Code snippet manager.

    PBTSO Phase: SKILL

    Responsibilities:
    - Store and organize code snippets
    - Provide fast snippet search
    - Track snippet versions
    - Share snippets between agents
    - Suggest relevant snippets

    Usage:
        manager = SnippetManager(config)
        snippet_id = manager.add(snippet)
        results = manager.search("async function")
    """

    BUS_TOPICS = {
        "add": "code.snippet.add",
        "search": "code.snippet.search",
        "used": "code.snippet.used",
        "heartbeat": "code.snippet.heartbeat",
    }

    # Built-in snippets
    BUILTIN_SNIPPETS: List[CodeSnippet] = []

    def __init__(
        self,
        config: Optional[SnippetConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or SnippetConfig()
        self.bus = bus or LockedAgentBus()
        self._snippets: Dict[str, CodeSnippet] = {}
        self._versions: Dict[str, List[SnippetVersion]] = {}
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> snippet IDs
        self._language_index: Dict[str, Set[str]] = {}  # language -> snippet IDs
        self._category_index: Dict[str, Set[str]] = {}  # category -> snippet IDs

        self._load_builtins()
        self._load_from_storage()

    def _load_builtins(self) -> None:
        """Load built-in snippets."""
        builtins = [
            CodeSnippet(
                id="builtin-python-singleton",
                name="Python Singleton",
                code='''class Singleton:
    """Thread-safe singleton pattern."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
''',
                language="python",
                category=SnippetCategory.PATTERN,
                description="Thread-safe singleton implementation",
                tags=["singleton", "pattern", "thread-safe"],
            ),
            CodeSnippet(
                id="builtin-python-retry",
                name="Python Retry Decorator",
                code='''def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator
''',
                language="python",
                category=SnippetCategory.UTILITY,
                description="Retry decorator with exponential backoff",
                tags=["retry", "decorator", "error-handling", "backoff"],
                dependencies=["functools", "time"],
            ),
            CodeSnippet(
                id="builtin-python-context-timer",
                name="Python Timer Context Manager",
                code='''@contextmanager
def timer(name: str = "operation"):
    """Context manager for timing operations."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"{name} took {elapsed:.3f}s")
''',
                language="python",
                category=SnippetCategory.UTILITY,
                description="Context manager for timing code blocks",
                tags=["timer", "context-manager", "profiling"],
                dependencies=["contextlib", "time"],
            ),
            CodeSnippet(
                id="builtin-python-async-gather",
                name="Python Async Gather with Limits",
                code='''async def gather_with_concurrency(
    n: int,
    *coros,
) -> List[Any]:
    """Run coroutines with a concurrency limit."""
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(*(sem_coro(c) for c in coros))
''',
                language="python",
                category=SnippetCategory.ASYNC,
                description="Async gather with concurrency limit",
                tags=["async", "concurrency", "semaphore", "gather"],
                dependencies=["asyncio"],
            ),
            CodeSnippet(
                id="builtin-python-dataclass-json",
                name="Python Dataclass JSON Serialization",
                code='''@dataclass
class JsonSerializable:
    """Base class for JSON-serializable dataclasses."""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "JsonSerializable":
        return cls(**json.loads(json_str))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
''',
                language="python",
                category=SnippetCategory.DATA_STRUCTURE,
                description="Base dataclass with JSON serialization",
                tags=["dataclass", "json", "serialization"],
                dependencies=["dataclasses", "json"],
            ),
            CodeSnippet(
                id="builtin-typescript-fetch-wrapper",
                name="TypeScript Fetch Wrapper",
                code='''async function fetchJson<T>(
    url: string,
    options: RequestInit = {}
): Promise<T> {
    const response = await fetch(url, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...options.headers,
        },
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json() as Promise<T>;
}
''',
                language="typescript",
                category=SnippetCategory.API,
                description="Type-safe fetch wrapper for JSON APIs",
                tags=["fetch", "api", "typescript", "async"],
            ),
        ]

        for snippet in builtins:
            self._add_to_index(snippet)

    def _load_from_storage(self) -> None:
        """Load snippets from storage."""
        storage_path = Path(self.config.storage_path)
        if not storage_path.exists():
            return

        snippets_file = storage_path / "snippets.json"
        if snippets_file.exists():
            try:
                with open(snippets_file) as f:
                    data = json.load(f)
                    for snippet_data in data.get("snippets", []):
                        snippet = CodeSnippet.from_dict(snippet_data)
                        self._add_to_index(snippet)
            except (json.JSONDecodeError, OSError):
                pass

    def _save_to_storage(self) -> None:
        """Save snippets to storage."""
        storage_path = Path(self.config.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)

        snippets_file = storage_path / "snippets.json"
        data = {
            "snippets": [s.to_dict() for s in self._snippets.values()],
            "saved_at": time.time(),
        }

        with open(snippets_file, "w") as f:
            json.dump(data, f, indent=2)

    def _add_to_index(self, snippet: CodeSnippet) -> None:
        """Add snippet to all indexes."""
        self._snippets[snippet.id] = snippet

        # Tag index
        for tag in snippet.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(snippet.id)

        # Language index
        if snippet.language not in self._language_index:
            self._language_index[snippet.language] = set()
        self._language_index[snippet.language].add(snippet.id)

        # Category index
        if snippet.category.value not in self._category_index:
            self._category_index[snippet.category.value] = set()
        self._category_index[snippet.category.value].add(snippet.id)

    def _remove_from_index(self, snippet: CodeSnippet) -> None:
        """Remove snippet from all indexes."""
        for tag in snippet.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(snippet.id)

        if snippet.language in self._language_index:
            self._language_index[snippet.language].discard(snippet.id)

        if snippet.category.value in self._category_index:
            self._category_index[snippet.category.value].discard(snippet.id)

    def add(self, snippet: CodeSnippet, changelog: str = "") -> str:
        """
        Add or update a snippet.

        Args:
            snippet: Snippet to add
            changelog: Description of changes (for versioning)

        Returns:
            Snippet ID
        """
        existing = self._snippets.get(snippet.id)

        if existing and self.config.enable_versioning:
            # Save version history
            if snippet.id not in self._versions:
                self._versions[snippet.id] = []

            self._versions[snippet.id].append(SnippetVersion(
                version=existing.version,
                code=existing.code,
                updated_at=existing.updated_at,
                changelog=changelog,
            ))

            # Trim old versions
            if len(self._versions[snippet.id]) > self.config.max_versions:
                self._versions[snippet.id] = self._versions[snippet.id][-self.config.max_versions:]

            snippet.version = existing.version + 1
            self._remove_from_index(existing)

        snippet.updated_at = time.time()
        self._add_to_index(snippet)

        # Emit add event
        self.bus.emit({
            "topic": self.BUS_TOPICS["add"],
            "kind": "snippet",
            "actor": "snippet-manager",
            "data": {
                "snippet_id": snippet.id,
                "name": snippet.name,
                "language": snippet.language,
                "version": snippet.version,
            },
        })

        self._save_to_storage()
        return snippet.id

    def get(self, snippet_id: str) -> Optional[CodeSnippet]:
        """Get snippet by ID."""
        return self._snippets.get(snippet_id)

    def delete(self, snippet_id: str) -> bool:
        """Delete a snippet."""
        snippet = self._snippets.get(snippet_id)
        if not snippet:
            return False

        self._remove_from_index(snippet)
        del self._snippets[snippet_id]
        self._versions.pop(snippet_id, None)

        self._save_to_storage()
        return True

    def search(
        self,
        query: str,
        language: Optional[str] = None,
        category: Optional[SnippetCategory] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[SnippetSearchResult]:
        """
        Search for snippets.

        Args:
            query: Search query
            language: Filter by language
            category: Filter by category
            tags: Filter by tags
            limit: Maximum results

        Returns:
            List of search results
        """
        candidates: Set[str] = set(self._snippets.keys())

        # Apply filters
        if language and language in self._language_index:
            candidates &= self._language_index[language]

        if category and category.value in self._category_index:
            candidates &= self._category_index[category.value]

        if tags:
            tag_matches = set()
            for tag in tags:
                if tag in self._tag_index:
                    tag_matches |= self._tag_index[tag]
            candidates &= tag_matches

        results: List[SnippetSearchResult] = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for snippet_id in candidates:
            snippet = self._snippets[snippet_id]
            score = 0.0
            match_type = "fuzzy"
            highlights = []

            # Exact name match
            if query_lower == snippet.name.lower():
                score = 1.0
                match_type = "exact"
                highlights.append(f"Exact name match: {snippet.name}")

            # Name contains query
            elif query_lower in snippet.name.lower():
                score = 0.8
                match_type = "exact"
                highlights.append(f"Name contains: {query}")

            # Tag match
            tag_matches = query_words & set(t.lower() for t in snippet.tags)
            if tag_matches:
                score = max(score, 0.7 * len(tag_matches) / len(query_words))
                match_type = "tag"
                highlights.extend([f"Tag: {t}" for t in tag_matches])

            # Description match
            if query_lower in snippet.description.lower():
                score = max(score, 0.6)
                highlights.append("Description match")

            # Code contains query
            if query_lower in snippet.code.lower():
                score = max(score, 0.5)
                highlights.append("Code contains query")

            # Word overlap in description
            desc_words = set(snippet.description.lower().split())
            overlap = query_words & desc_words
            if overlap:
                overlap_score = 0.4 * len(overlap) / len(query_words)
                score = max(score, overlap_score)
                if not highlights:
                    highlights.append(f"Word matches: {', '.join(overlap)}")

            if score > 0:
                results.append(SnippetSearchResult(
                    snippet=snippet,
                    score=score,
                    match_type=match_type,
                    highlights=highlights,
                ))

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)

        # Emit search event
        self.bus.emit({
            "topic": self.BUS_TOPICS["search"],
            "kind": "snippet",
            "actor": "snippet-manager",
            "data": {
                "query": query,
                "language": language,
                "results_count": len(results),
            },
        })

        return results[:limit]

    def use(self, snippet_id: str) -> Optional[str]:
        """
        Mark a snippet as used and return its code.

        Args:
            snippet_id: ID of snippet to use

        Returns:
            Snippet code or None if not found
        """
        snippet = self._snippets.get(snippet_id)
        if not snippet:
            return None

        snippet.usage_count += 1
        snippet.updated_at = time.time()

        # Emit used event
        self.bus.emit({
            "topic": self.BUS_TOPICS["used"],
            "kind": "snippet",
            "actor": "snippet-manager",
            "data": {
                "snippet_id": snippet_id,
                "usage_count": snippet.usage_count,
            },
        })

        return snippet.code

    def get_versions(self, snippet_id: str) -> List[SnippetVersion]:
        """Get version history for a snippet."""
        return self._versions.get(snippet_id, [])

    def restore_version(self, snippet_id: str, version: int) -> bool:
        """Restore a snippet to a previous version."""
        versions = self._versions.get(snippet_id, [])
        for v in versions:
            if v.version == version:
                snippet = self._snippets.get(snippet_id)
                if snippet:
                    snippet.code = v.code
                    snippet.updated_at = time.time()
                    self._save_to_storage()
                    return True
        return False

    def suggest(
        self,
        context: str,
        language: str,
        limit: int = 5,
    ) -> List[SnippetSearchResult]:
        """
        Suggest snippets based on context.

        Args:
            context: Code context
            language: Target language
            limit: Maximum suggestions

        Returns:
            List of suggested snippets
        """
        # Extract keywords from context
        keywords = self._extract_keywords(context)

        # Search with keywords
        results = self.search(
            " ".join(keywords),
            language=language,
            limit=limit,
        )

        return results

    def _extract_keywords(self, context: str) -> List[str]:
        """Extract keywords from code context."""
        # Remove common programming keywords
        stop_words = {
            "def", "class", "import", "from", "return", "if", "else", "for",
            "while", "try", "except", "with", "as", "in", "is", "not", "and",
            "or", "None", "True", "False", "self", "async", "await", "yield",
            "function", "const", "let", "var", "export", "default",
        }

        # Extract words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', context)

        # Filter and deduplicate
        keywords = []
        seen = set()
        for word in words:
            word_lower = word.lower()
            if word_lower not in stop_words and word_lower not in seen and len(word) > 2:
                keywords.append(word_lower)
                seen.add(word_lower)

        return keywords[:10]

    def list_by_category(self, category: SnippetCategory) -> List[CodeSnippet]:
        """List snippets by category."""
        snippet_ids = self._category_index.get(category.value, set())
        return [self._snippets[sid] for sid in snippet_ids if sid in self._snippets]

    def list_by_language(self, language: str) -> List[CodeSnippet]:
        """List snippets by language."""
        snippet_ids = self._language_index.get(language, set())
        return [self._snippets[sid] for sid in snippet_ids if sid in self._snippets]

    def get_popular(self, limit: int = 10) -> List[CodeSnippet]:
        """Get most used snippets."""
        snippets = list(self._snippets.values())
        snippets.sort(key=lambda s: s.usage_count, reverse=True)
        return snippets[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get snippet manager statistics."""
        return {
            "total_snippets": len(self._snippets),
            "languages": list(self._language_index.keys()),
            "categories": list(self._category_index.keys()),
            "tags": list(self._tag_index.keys()),
            "total_uses": sum(s.usage_count for s in self._snippets.values()),
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Snippet Manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Code Snippet Manager (Step 73)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # add command
    add_parser = subparsers.add_parser("add", help="Add a snippet")
    add_parser.add_argument("name", help="Snippet name")
    add_parser.add_argument("--code", "-c", help="Snippet code (or read from stdin)")
    add_parser.add_argument("--language", "-l", default="python", help="Language")
    add_parser.add_argument("--category", default="custom",
                           choices=[c.value for c in SnippetCategory])
    add_parser.add_argument("--tags", "-t", nargs="+", default=[], help="Tags")
    add_parser.add_argument("--description", "-d", default="", help="Description")

    # search command
    search_parser = subparsers.add_parser("search", help="Search snippets")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--language", "-l", help="Filter by language")
    search_parser.add_argument("--limit", "-n", type=int, default=10, help="Max results")
    search_parser.add_argument("--json", action="store_true", help="JSON output")

    # get command
    get_parser = subparsers.add_parser("get", help="Get a snippet")
    get_parser.add_argument("snippet_id", help="Snippet ID")
    get_parser.add_argument("--json", action="store_true", help="JSON output")

    # use command
    use_parser = subparsers.add_parser("use", help="Use a snippet (returns code)")
    use_parser.add_argument("snippet_id", help="Snippet ID")

    # list command
    list_parser = subparsers.add_parser("list", help="List snippets")
    list_parser.add_argument("--language", "-l", help="Filter by language")
    list_parser.add_argument("--category", "-c", help="Filter by category")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # popular command
    pop_parser = subparsers.add_parser("popular", help="Show popular snippets")
    pop_parser.add_argument("--limit", "-n", type=int, default=10, help="Limit")

    # stats command
    subparsers.add_parser("stats", help="Show manager stats")

    args = parser.parse_args()

    manager = SnippetManager()

    if args.command == "add":
        code = args.code
        if code is None:
            import sys
            code = sys.stdin.read()

        snippet = CodeSnippet(
            id=f"snippet-{uuid.uuid4().hex[:8]}",
            name=args.name,
            code=code,
            language=args.language,
            category=SnippetCategory(args.category),
            description=args.description,
            tags=args.tags,
        )

        snippet_id = manager.add(snippet)
        print(f"Added snippet: {snippet_id}")
        return 0

    elif args.command == "search":
        results = manager.search(
            args.query,
            language=args.language,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            for r in results:
                print(f"{r.snippet.id}: {r.snippet.name} ({r.score:.2f}) - {r.match_type}")
                if r.highlights:
                    print(f"  {', '.join(r.highlights)}")
        return 0

    elif args.command == "get":
        snippet = manager.get(args.snippet_id)

        if not snippet:
            print(f"Snippet not found: {args.snippet_id}")
            return 1

        if args.json:
            print(json.dumps(snippet.to_dict(), indent=2))
        else:
            print(f"ID: {snippet.id}")
            print(f"Name: {snippet.name}")
            print(f"Language: {snippet.language}")
            print(f"Category: {snippet.category.value}")
            print(f"Tags: {', '.join(snippet.tags)}")
            print(f"Uses: {snippet.usage_count}")
            print(f"\nCode:\n{snippet.code}")
        return 0

    elif args.command == "use":
        code = manager.use(args.snippet_id)

        if code is None:
            print(f"Snippet not found: {args.snippet_id}")
            return 1

        print(code)
        return 0

    elif args.command == "list":
        if args.language:
            snippets = manager.list_by_language(args.language)
        elif args.category:
            snippets = manager.list_by_category(SnippetCategory(args.category))
        else:
            snippets = list(manager._snippets.values())

        if args.json:
            print(json.dumps([s.to_dict() for s in snippets], indent=2))
        else:
            for s in snippets:
                print(f"{s.id}: {s.name} ({s.language})")
        return 0

    elif args.command == "popular":
        snippets = manager.get_popular(args.limit)

        for s in snippets:
            print(f"{s.id}: {s.name} ({s.usage_count} uses)")
        return 0

    elif args.command == "stats":
        stats = manager.get_stats()
        print(json.dumps(stats, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
