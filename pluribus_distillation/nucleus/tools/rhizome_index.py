#!/usr/bin/env python3
"""
rhizome_index.py - Semantic Search Index for Rhizome DAG

Provides SQLite FTS5 text search and embedding-based semantic search
for the rhizome semantic layer.

Ring: 1 (Infrastructure)
Protocol: DKIN v30 | PAIP v16 | Citizen v1

Usage:
    python3 rhizome_index.py embed <file>           # Generate embedding
    python3 rhizome_index.py index <file>           # Add to search index
    python3 rhizome_index.py search <query>         # Search (lexical + semantic)
    python3 rhizome_index.py status                 # Show index stats
    python3 rhizome_index.py vacuum                 # Optimize index

Bus Topics:
    rhizome.index.updated
    rhizome.search.executed
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# Configuration
INDEX_DIR = Path(os.environ.get("PLURIBUS_INDEX_DIR", ".pluribus/index"))
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", ".pluribus/bus"))


class RhizomeIndex:
    """
    Semantic search index for rhizome DAG.
    
    Features:
    - SQLite FTS5 for full-text search
    - Content hash deduplication
    - Tag-based filtering
    - Embedding storage (for future HNSW integration)
    """

    def __init__(self, index_dir: Path = None, bus_dir: Path = None):
        self.index_dir = index_dir or INDEX_DIR
        self.bus_dir = bus_dir or BUS_DIR
        self.db_path = self.index_dir / "rhizome_index.db"
        self.bus_path = self.bus_dir / "events.ndjson"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with FTS5."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main content table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content_hash TEXT UNIQUE,
                source_path TEXT,
                content TEXT,
                tags TEXT,
                embedding BLOB,
                created_ts REAL,
                indexed_ts REAL
            )
        """)
        
        # FTS5 virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                id,
                source_path,
                content,
                tags,
                content='documents',
                content_rowid='rowid'
            )
        """)
        
        # Triggers to keep FTS in sync
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, id, source_path, content, tags)
                VALUES (new.rowid, new.id, new.source_path, new.content, new.tags);
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, id, source_path, content, tags)
                VALUES('delete', old.rowid, old.id, old.source_path, old.content, old.tags);
            END
        """)
        
        conn.commit()
        conn.close()

    def _emit_bus_event(self, topic: str, data: Dict[str, Any], level: str = "info"):
        """Emit event to the Pluribus bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        
        event = {
            "id": uuid.uuid4().hex,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "kind": "event",
            "level": level,
            "actor": "rhizome_index",
            "data": data,
        }
        
        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _generate_embedding(self, text: str) -> Optional[bytes]:
        """Generate embedding for text.
        
        TODO: Integrate with sentence-transformers.
        Placeholder returns hash-derived bytes.
        """
        h = hashlib.sha256(text.encode()).digest()
        return h

    def index_file(
        self,
        file_path: Path,
        tags: List[str] = None,
        emit_bus: bool = True,
    ) -> str:
        """Index a file for search."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content = file_path.read_text(errors='replace')
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        doc_id = uuid.uuid4().hex[:16]
        tags = tags or []
        
        # Generate embedding
        embedding = self._generate_embedding(content)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check for duplicate
        cursor.execute("SELECT id FROM documents WHERE content_hash = ?", (content_hash,))
        existing = cursor.fetchone()
        if existing:
            conn.close()
            return existing[0]  # Return existing doc ID
        
        # Insert
        cursor.execute("""
            INSERT INTO documents (id, content_hash, source_path, content, tags, embedding, created_ts, indexed_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id,
            content_hash,
            str(file_path),
            content[:50000],  # Limit content size
            ",".join(tags),
            embedding,
            file_path.stat().st_mtime,
            time.time(),
        ))
        
        conn.commit()
        conn.close()
        
        if emit_bus:
            self._emit_bus_event("rhizome.index.updated", {
                "doc_id": doc_id,
                "content_hash": content_hash[:16],
                "source_path": str(file_path),
                "tags": tags,
            })
        
        return doc_id

    def search(
        self,
        query: str,
        limit: int = 10,
        semantic: bool = False,
        emit_bus: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search the index.
        
        Args:
            query: Search query
            limit: Maximum results
            semantic: Use embedding search (TODO)
            emit_bus: Emit search event
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # FTS5 search
        cursor.execute("""
            SELECT d.id, d.source_path, d.tags, d.created_ts,
                   snippet(documents_fts, 2, '>>>', '<<<', '...', 32) as snippet
            FROM documents_fts f
            JOIN documents d ON f.rowid = d.rowid
            WHERE documents_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "source_path": row[1],
                "tags": row[2].split(",") if row[2] else [],
                "created_ts": row[3],
                "snippet": row[4],
            })
        
        conn.close()
        
        if emit_bus:
            self._emit_bus_event("rhizome.search.executed", {
                "query": query[:100],
                "results_count": len(results),
                "semantic": semantic,
            })
        
        return results

    def status(self) -> Dict[str, Any]:
        """Get index statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(LENGTH(content)) FROM documents")
        total_size = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT tags FROM documents WHERE tags != ''")
        all_tags = set()
        for row in cursor.fetchall():
            all_tags.update(row[0].split(","))
        
        conn.close()
        
        return {
            "document_count": doc_count,
            "total_content_bytes": total_size,
            "unique_tags": len(all_tags),
            "db_path": str(self.db_path),
            "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
        }

    def vacuum(self) -> Dict[str, Any]:
        """Optimize the index."""
        conn = sqlite3.connect(self.db_path)
        
        # FTS5 optimize
        conn.execute("INSERT INTO documents_fts(documents_fts) VALUES('optimize')")
        
        # SQLite vacuum
        conn.execute("VACUUM")
        
        conn.close()
        
        return self.status()


def cmd_index(args):
    """Index a file."""
    idx = RhizomeIndex()
    doc_id = idx.index_file(Path(args.file), tags=args.tags or [])
    print(f"Indexed: {doc_id}")
    return 0


def cmd_search(args):
    """Search the index."""
    idx = RhizomeIndex()
    results = idx.search(args.query, limit=args.limit or 10, semantic=args.semantic)
    
    if not results:
        print("No results found")
        return 0
    
    for r in results:
        tags = ", ".join(r["tags"][:3]) if r["tags"] else "(no tags)"
        print(f"  {r['id']}: {r['source_path']}")
        print(f"    Tags: {tags}")
        print(f"    {r['snippet']}")
        print()
    
    print(f"{len(results)} result(s)")
    return 0


def cmd_status(args):
    """Show index status."""
    idx = RhizomeIndex()
    status = idx.status()
    
    print("Rhizome Search Index")
    print(f"  Documents: {status['document_count']}")
    print(f"  Content: {status['total_content_bytes'] / 1024:.1f} KB")
    print(f"  Tags: {status['unique_tags']}")
    print(f"  DB Size: {status['db_size_bytes'] / 1024:.1f} KB")
    return 0


def cmd_vacuum(args):
    """Optimize index."""
    idx = RhizomeIndex()
    status = idx.vacuum()
    print(f"Optimized. {status['document_count']} documents, {status['db_size_bytes'] / 1024:.1f} KB")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Rhizome Search Index")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # index
    p_index = subparsers.add_parser("index", help="Index a file")
    p_index.add_argument("file", help="File to index")
    p_index.add_argument("--tags", action="append", help="Add tag (repeatable)")
    
    # search
    p_search = subparsers.add_parser("search", help="Search the index")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", type=int, help="Max results")
    p_search.add_argument("--semantic", action="store_true", help="Use semantic search")
    
    # status
    subparsers.add_parser("status", help="Show status")
    
    # vacuum
    subparsers.add_parser("vacuum", help="Optimize index")
    
    args = parser.parse_args()
    
    if args.command == "index":
        return cmd_index(args)
    elif args.command == "search":
        return cmd_search(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "vacuum":
        return cmd_vacuum(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
