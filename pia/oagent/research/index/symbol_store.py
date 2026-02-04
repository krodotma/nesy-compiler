#!/usr/bin/env python3
"""
symbol_store.py - Symbol Index Store (Step 6)

Stores and indexes code symbols for fast querying.
Supports both in-memory and FalkorDB-backed storage.

PBTSO Phase: RESEARCH, DISTILL

Bus Topics:
- research.index.symbol
- research.query.symbols

Protocol: DKIN v30, PAIP v16
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator

from ..bootstrap import AgentBus


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Symbol:
    """Represents a code symbol in the index."""

    name: str
    kind: str  # class, function, method, variable, import, constant, interface, type, enum
    path: str  # File path
    line: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent: Optional[str] = None  # Parent class/module
    language: str = "unknown"
    visibility: str = "public"
    is_async: bool = False
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Symbol":
        """Create Symbol from dictionary."""
        return cls(**data)


# ============================================================================
# Symbol Index Store
# ============================================================================


class SymbolIndexStore:
    """
    Index store for code symbols.

    Provides fast symbol lookup by name, kind, path, and other attributes.
    Uses SQLite for persistent storage.

    Example:
        store = SymbolIndexStore()
        store.index_symbol(Symbol(name="MyClass", kind="class", path="src/main.py", line=10))
        results = store.query(name="MyClass")
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        bus: Optional[AgentBus] = None,
        falkordb_service: Optional[Any] = None,
    ):
        """
        Initialize the symbol index store.

        Args:
            db_path: Path to SQLite database (default: in-memory)
            bus: AgentBus for event emission
            falkordb_service: Optional FalkorDB service for graph queries
        """
        self.bus = bus or AgentBus()
        self.fdb = falkordb_service

        if db_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            db_path = Path(pluribus_root) / ".pluribus" / "research" / "symbols.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database schema."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                path TEXT NOT NULL,
                line INTEGER NOT NULL,
                signature TEXT,
                docstring TEXT,
                parent TEXT,
                language TEXT DEFAULT 'unknown',
                visibility TEXT DEFAULT 'public',
                is_async INTEGER DEFAULT 0,
                return_type TEXT,
                decorators TEXT,
                indexed_at REAL DEFAULT (strftime('%s', 'now')),
                UNIQUE(name, kind, path, line)
            )
        """)

        # Create indexes for fast queries
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_path ON symbols(path)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_parent ON symbols(parent)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_language ON symbols(language)")

        self._conn.commit()

    def index_symbol(self, symbol: Symbol) -> int:
        """
        Index a symbol to the store.

        Args:
            symbol: Symbol to index

        Returns:
            Symbol ID
        """
        decorators_json = json.dumps(symbol.decorators) if symbol.decorators else None

        cursor = self._conn.execute("""
            INSERT OR REPLACE INTO symbols
            (name, kind, path, line, signature, docstring, parent, language,
             visibility, is_async, return_type, decorators)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol.name,
            symbol.kind,
            symbol.path,
            symbol.line,
            symbol.signature,
            symbol.docstring,
            symbol.parent,
            symbol.language,
            symbol.visibility,
            1 if symbol.is_async else 0,
            symbol.return_type,
            decorators_json,
        ))

        self._conn.commit()
        symbol_id = cursor.lastrowid

        # Emit event
        self.bus.emit({
            "topic": "research.index.symbol",
            "kind": "index",
            "data": {
                "id": symbol_id,
                "name": symbol.name,
                "kind": symbol.kind,
                "path": symbol.path,
            }
        })

        # Also index to FalkorDB if available
        if self.fdb:
            self._index_to_falkordb(symbol)

        return symbol_id

    def index_symbols(self, symbols: List[Symbol]) -> int:
        """
        Index multiple symbols efficiently.

        Args:
            symbols: List of symbols to index

        Returns:
            Number of symbols indexed
        """
        count = 0
        for symbol in symbols:
            try:
                self.index_symbol(symbol)
                count += 1
            except Exception:
                pass

        return count

    def query(
        self,
        name: Optional[str] = None,
        kind: Optional[str] = None,
        path: Optional[str] = None,
        parent: Optional[str] = None,
        language: Optional[str] = None,
        name_pattern: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Symbol]:
        """
        Query symbols from the index.

        Args:
            name: Exact symbol name
            kind: Symbol kind (class, function, etc.)
            path: File path
            parent: Parent symbol name
            language: Programming language
            name_pattern: SQL LIKE pattern for name
            limit: Maximum results
            offset: Result offset

        Returns:
            List of matching symbols
        """
        conditions = []
        params = []

        if name:
            conditions.append("name = ?")
            params.append(name)
        if kind:
            conditions.append("kind = ?")
            params.append(kind)
        if path:
            conditions.append("path = ?")
            params.append(path)
        if parent:
            conditions.append("parent = ?")
            params.append(parent)
        if language:
            conditions.append("language = ?")
            params.append(language)
        if name_pattern:
            conditions.append("name LIKE ?")
            params.append(name_pattern)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT * FROM symbols
            WHERE {where_clause}
            ORDER BY path, line
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        cursor = self._conn.execute(query, params)
        results = []

        for row in cursor:
            decorators = json.loads(row["decorators"]) if row["decorators"] else []
            results.append(Symbol(
                name=row["name"],
                kind=row["kind"],
                path=row["path"],
                line=row["line"],
                signature=row["signature"],
                docstring=row["docstring"],
                parent=row["parent"],
                language=row["language"],
                visibility=row["visibility"],
                is_async=bool(row["is_async"]),
                return_type=row["return_type"],
                decorators=decorators,
            ))

        # Emit query event
        self.bus.emit({
            "topic": "research.query.symbols",
            "kind": "query",
            "data": {
                "filters": {
                    "name": name,
                    "kind": kind,
                    "path": path,
                    "name_pattern": name_pattern,
                },
                "results": len(results),
            }
        })

        return results

    def search(self, query: str, limit: int = 50) -> List[Symbol]:
        """
        Full-text search for symbols.

        Args:
            query: Search query (searches name and docstring)
            limit: Maximum results

        Returns:
            List of matching symbols
        """
        pattern = f"%{query}%"

        cursor = self._conn.execute("""
            SELECT * FROM symbols
            WHERE name LIKE ? OR docstring LIKE ?
            ORDER BY
                CASE WHEN name LIKE ? THEN 0 ELSE 1 END,
                name
            LIMIT ?
        """, (pattern, pattern, f"{query}%", limit))

        results = []
        for row in cursor:
            decorators = json.loads(row["decorators"]) if row["decorators"] else []
            results.append(Symbol(
                name=row["name"],
                kind=row["kind"],
                path=row["path"],
                line=row["line"],
                signature=row["signature"],
                docstring=row["docstring"],
                parent=row["parent"],
                language=row["language"],
                visibility=row["visibility"],
                is_async=bool(row["is_async"]),
                return_type=row["return_type"],
                decorators=decorators,
            ))

        return results

    def get_by_path(self, path: str) -> List[Symbol]:
        """Get all symbols in a file."""
        return self.query(path=path, limit=10000)

    def get_classes(self, path: Optional[str] = None) -> List[Symbol]:
        """Get all classes, optionally filtered by path."""
        return self.query(kind="class", path=path, limit=10000)

    def get_functions(self, path: Optional[str] = None) -> List[Symbol]:
        """Get all functions, optionally filtered by path."""
        return self.query(kind="function", path=path, limit=10000)

    def get_methods(self, class_name: str) -> List[Symbol]:
        """Get all methods of a class."""
        return self.query(kind="method", parent=class_name, limit=10000)

    def count(self, kind: Optional[str] = None) -> int:
        """Count symbols in the index."""
        if kind:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM symbols WHERE kind = ?",
                (kind,)
            )
        else:
            cursor = self._conn.execute("SELECT COUNT(*) FROM symbols")

        return cursor.fetchone()[0]

    def stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        cursor = self._conn.execute("""
            SELECT kind, COUNT(*) as count
            FROM symbols
            GROUP BY kind
        """)

        by_kind = {row["kind"]: row["count"] for row in cursor}

        cursor = self._conn.execute("""
            SELECT language, COUNT(*) as count
            FROM symbols
            GROUP BY language
        """)

        by_language = {row["language"]: row["count"] for row in cursor}

        cursor = self._conn.execute("SELECT COUNT(DISTINCT path) FROM symbols")
        file_count = cursor.fetchone()[0]

        return {
            "total": self.count(),
            "by_kind": by_kind,
            "by_language": by_language,
            "file_count": file_count,
        }

    def clear(self) -> None:
        """Clear all symbols from the index."""
        self._conn.execute("DELETE FROM symbols")
        self._conn.commit()

    def delete_path(self, path: str) -> int:
        """Delete all symbols for a path."""
        cursor = self._conn.execute(
            "DELETE FROM symbols WHERE path = ?",
            (path,)
        )
        self._conn.commit()
        return cursor.rowcount

    def _index_to_falkordb(self, symbol: Symbol) -> None:
        """Index symbol to FalkorDB knowledge graph."""
        if not self.fdb:
            return

        try:
            # Create symbol node
            query = """
            MERGE (s:Symbol {name: $name, path: $path, line: $line})
            SET s.kind = $kind,
                s.language = $language,
                s.signature = $signature,
                s.visibility = $visibility
            RETURN s.name
            """
            self.fdb.query(query, {
                "name": symbol.name,
                "path": symbol.path,
                "line": symbol.line,
                "kind": symbol.kind,
                "language": symbol.language,
                "signature": symbol.signature,
                "visibility": symbol.visibility,
            })

            # Create parent relationship if exists
            if symbol.parent:
                query = """
                MATCH (s:Symbol {name: $name, path: $path})
                MATCH (p:Symbol {name: $parent, path: $path})
                MERGE (s)-[:BELONGS_TO]->(p)
                """
                self.fdb.query(query, {
                    "name": symbol.name,
                    "path": symbol.path,
                    "parent": symbol.parent,
                })

        except Exception:
            pass  # FalkorDB errors are non-fatal

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Symbol Index Store."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Symbol Index Store (Step 6)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query symbols")
    query_parser.add_argument("--name", help="Symbol name")
    query_parser.add_argument("--kind", help="Symbol kind")
    query_parser.add_argument("--path", help="File path")
    query_parser.add_argument("--pattern", help="Name pattern (SQL LIKE)")
    query_parser.add_argument("--limit", type=int, default=50, help="Max results")
    query_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search symbols")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=50, help="Max results")
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show index statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    store = SymbolIndexStore()

    if args.command == "query":
        results = store.query(
            name=args.name,
            kind=args.kind,
            path=args.path,
            name_pattern=args.pattern,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps([s.to_dict() for s in results], indent=2))
        else:
            print(f"Found {len(results)} symbols:")
            for s in results:
                print(f"  {s.kind:10} {s.name:30} {s.path}:{s.line}")

    elif args.command == "search":
        results = store.search(args.query, limit=args.limit)

        if args.json:
            print(json.dumps([s.to_dict() for s in results], indent=2))
        else:
            print(f"Found {len(results)} symbols matching '{args.query}':")
            for s in results:
                print(f"  {s.kind:10} {s.name:30} {s.path}:{s.line}")

    elif args.command == "stats":
        stats = store.stats()

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Symbol Index Statistics:")
            print(f"  Total Symbols: {stats['total']}")
            print(f"  Files Indexed: {stats['file_count']}")
            print(f"  By Kind:")
            for kind, count in sorted(stats['by_kind'].items()):
                print(f"    {kind}: {count}")
            print(f"  By Language:")
            for lang, count in sorted(stats['by_language'].items()):
                print(f"    {lang}: {count}")

    store.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
