#!/usr/bin/env python3
"""
RAG Vector Store with sqlite-vec (Hybrid BM25 + Semantic)
=========================================================

Per comprehensive_implementation_matrix.md:
1. Derived Index: Rebuildable from events.ndjson (SoR)
2. Isomorphic: Works in browser WASM + TUI + native
3. Zero Network Effects: Pure embedded, no server
4. Hybrid Search: BM25 (FTS5) + semantic (vec0)

Sextet Compliance:
- Source: Bus / Input Text
- Transducer: Embedding Model (sentence-transformers) or FTS5 fallback
- Regulator: Local file access only (Sextet 'L')
- Memory: .pluribus/index/rag.sqlite3
- Feedback: Bus Events

Usage:
  python3 rag_vector.py init                           # Initialize database
  python3 rag_vector.py index-event '{"data": {...}}'  # Index a bus event
  python3 rag_vector.py rebuild                        # Rebuild from events.ndjson
  python3 rag_vector.py search "query" --limit 10      # Hybrid search
  python3 rag_vector.py stats                          # Show index stats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Optional, Literal

sys.dont_write_bytecode = True

try:
    from .embeddings import DEFAULT_DIM, embed_text as _embed_text
except (ImportError, ValueError):
    from embeddings import DEFAULT_DIM, embed_text as _embed_text

try:
    from .core.bus_consumer import BusConsumer
except (ImportError, ValueError):
    try:
        from core.bus_consumer import BusConsumer
    except ImportError:
        # Fallback for direct execution
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from core.bus_consumer import BusConsumer

# Try to import sqlite-vec
_SQLITE_VEC_AVAILABLE = False
try:
    import sqlite_vec
    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    # Try venv path
    venv_site = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".pluribus/venv/lib/python3.12/site-packages"
    )
    if os.path.exists(venv_site):
        sys.path.insert(0, venv_site)
        try:
            import sqlite_vec
            _SQLITE_VEC_AVAILABLE = True
        except ImportError:
            pass

DB_PATH = os.path.expanduser("/pluribus/.pluribus/index/rag.sqlite3")
EMBEDDING_DIM = int(DEFAULT_DIM)  # all-MiniLM-L6-v2 dimension (default)

# Topics that contain valuable long-term knowledge
TARGET_TOPICS = {
    "dialogos.cell.output",
    "git.commit",
    "strp.response",
    "plurichat.response",
    "ingest.text" # For explicit manual ingest
}


class VectorRAG:
    """Hybrid BM25 + semantic search with sqlite-vec."""

    def __init__(self, db_path: Path, load_model: bool = True):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._vec_available = False

        # Try to load sqlite-vec extension
        self._load_vec_extension()

    def _load_vec_extension(self):
        """Try to load sqlite-vec extension."""
        if not _SQLITE_VEC_AVAILABLE:
            return

        try:
            self.conn.enable_load_extension(True)
            import sqlite_vec
            sqlite_vec.load(self.conn)
            self._vec_available = True
        except Exception:
            self._vec_available = False

    def init_schema(self):
        """Initialize FTS5 and optionally vec0 tables."""
        # FTS5 for BM25 text search (always available)
        self.conn.executescript("""
            -- Full-text search table
            CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
                id,
                content,
                source_type,
                source_id,
                tokenize='porter unicode61'
            );

            -- Metadata table for additional info
            CREATE TABLE IF NOT EXISTS embeddings_meta (
                id TEXT PRIMARY KEY,
                source_type TEXT,
                source_id TEXT,
                source_event_id TEXT,
                created_iso TEXT,
                lineage_id TEXT,
                embedding_dims INTEGER
            );
        """)

        # Vec0 for semantic search (if available)
        if self._vec_available:
            try:
                self.conn.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
                        id TEXT PRIMARY KEY,
                        embedding FLOAT[{EMBEDDING_DIM}]
                    );
                """)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    self._vec_available = False

        self.conn.commit()
        return self._vec_available

    def embed(self, text: str) -> Optional[list[float]]:
        """Generate embedding for text. Returns None if model not available."""
        emb, _meta = _embed_text(text, dim=EMBEDDING_DIM)
        return emb

    def _extract_text(self, event: dict) -> Optional[str]:
        """Extract indexable text from an event based on its topic."""
        topic = event.get("topic")
        data = event.get("data", {})
        
        if topic not in TARGET_TOPICS:
            return None

        text = None
        if topic == "git.commit":
            text = f"Commit: {data.get('message')} (SHA: {data.get('sha')})"
        elif topic in ("strp.response", "plurichat.response"):
            text = data.get("output") or data.get("text")
        elif topic == "dialogos.cell.output":
            text = data.get("content")
        elif topic == "ingest.text":
             text = data.get("text")
        
        # Min length filter to avoid noise
        if text and len(text) > 20:
            return text
        return None

    def index_event(self, event: dict, commit: bool = True) -> Optional[str]:
        """Index a bus event for hybrid search."""
        text_content = self._extract_text(event)
        if not text_content:
            return None

        event_id = event.get("id") or str(uuid.uuid4())
        source_type = event.get("kind", "event")
        
        # Insert into FTS5
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO content_fts (id, content, source_type, source_id)
                VALUES (?, ?, ?, ?)
            """, (event_id, text_content, source_type, event_id))
        except Exception:
            pass

        # Insert metadata
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO embeddings_meta
                (id, source_type, source_id, source_event_id, created_iso, lineage_id, embedding_dims)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                source_type,
                event_id,
                event_id,
                event.get("iso"),
                event.get("lineage_id"),
                EMBEDDING_DIM if self._vec_available else 0
            ))
        except Exception:
            pass

        # Insert embedding if vec0 available AND model is loaded
        if self._vec_available:
            embedding = self.embed(text_content)
            if embedding:
                try:
                    embedding_json = json.dumps(embedding)
                    self.conn.execute("""
                        INSERT OR REPLACE INTO embeddings (id, embedding)
                        VALUES (?, ?)
                    """, (event_id, embedding_json))
                except Exception:
                    pass

        if commit:
            self.conn.commit()
        return event_id

    def search_bm25(self, query: str, limit: int = 10) -> list[dict]:
        """BM25 text search using FTS5."""
        try:
            results = self.conn.execute("""
                SELECT id, source_type, source_id, rank, content
                FROM content_fts
                WHERE content_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()

            return [
                {
                    "id": r[0], 
                    "source_type": r[1], 
                    "source_id": r[2], 
                    "bm25_score": -r[3], 
                    "content_snippet": r[4][:200], # Preview
                    "mode": "bm25"
                }
                for r in results
            ]
        except Exception:
            return []

    def search_semantic(self, query: str, limit: int = 10) -> list[dict]:
        """Semantic search using vec0."""
        if not self._vec_available:
            return []

        try:
            query_embedding = self.embed(query)
            if not query_embedding:
                return []
                
            embedding_json = json.dumps(query_embedding)

            results = self.conn.execute("""
                SELECT id, distance
                FROM embeddings
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
            """, (embedding_json, limit)).fetchall()

            # Retrieve content for results
            enriched_results = []
            for r in results:
                id_ = r[0]
                distance = r[1]
                
                # Fetch content
                meta = self.conn.execute(
                    "SELECT content FROM content_fts WHERE id = ?", (id_,)
                ).fetchone()
                content = meta[0] if meta else ""
                
                enriched_results.append({
                    "id": id_,
                    "semantic_score": 1.0 / (1.0 + distance),
                    "distance": distance,
                    "content_snippet": content[:200],
                    "mode": "semantic"
                })

            return enriched_results
        except Exception:
            return []

    def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        bm25_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> list[dict]:
        """Combined BM25 + semantic search using Reciprocal Rank Fusion."""
        
        # Adjust weights if semantic search is unavailable
        if not self._vec_available:
            return self.search_bm25(query, limit)

        # Get results from both
        bm25_results = self.search_bm25(query, limit * 2)
        semantic_results = self.search_semantic(query, limit * 2)

        # If no semantic results, fall back to pure BM25
        if not semantic_results:
            return bm25_results[:limit]

        # Reciprocal Rank Fusion (RRF)
        scores: dict[str, float] = {}
        details: dict[str, dict] = {}

        for rank, r in enumerate(semantic_results):
            id_ = r["id"]
            scores[id_] = scores.get(id_, 0) + semantic_weight / (rank + 1)
            details[id_] = r

        for rank, r in enumerate(bm25_results):
            id_ = r["id"]
            scores[id_] = scores.get(id_, 0) + bm25_weight / (rank + 1)
            if id_ not in details:
                details[id_] = r
            else:
                # Prefer content from BM25 if available as it might be fresher or just because
                pass

        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:limit]

        return [
            {
                "id": id_,
                "score": score,
                "mode": "hybrid",
                **{k: v for k, v in details.get(id_, {}).items() if k != "mode"}
            }
            for id_, score in ranked
        ]

    def rebuild_from_bus(self, events_path: Path):
        """Rebuild entire index from events.ndjson (idempotent)."""
        import subprocess
        # Clear existing data
        try:
            self.conn.execute("DELETE FROM content_fts")
            self.conn.execute("DELETE FROM embeddings_meta")
            if self._vec_available:
                self.conn.execute("DELETE FROM embeddings")
        except Exception:
            pass

        indexed = 0
        skipped = 0
        errors = 0

        # Use unified BusConsumer
        consumer = BusConsumer(events_path)
        topic_pattern = "|".join(TARGET_TOPICS)
        BATCH_SIZE = 1000

        def process_event(event):
            nonlocal indexed, skipped, errors
            try:
                if self.index_event(event, commit=False):
                    indexed += 1
                else:
                    skipped += 1
                
                if (indexed + skipped) % BATCH_SIZE == 0:
                    self.conn.commit()
            except Exception:
                errors += 1

        consumer.scan(topic_filter=topic_pattern, callback=process_event)

        self.conn.commit()
        return {"indexed": indexed, "skipped": skipped, "errors": errors}

    def stats(self) -> dict:
        """Get index statistics."""
        fts_count = 0
        meta_count = 0
        vec_count = 0
        last_id = None

        try:
            fts_count = self.conn.execute("SELECT COUNT(*) FROM content_fts").fetchone()[0]
        except Exception:
            pass

        try:
            meta = self.conn.execute("SELECT id FROM embeddings_meta ORDER BY created_iso DESC LIMIT 1").fetchone()
            last_id = meta[0] if meta else None
            meta_count = self.conn.execute("SELECT COUNT(*) FROM embeddings_meta").fetchone()[0]
        except Exception:
            pass

        if self._vec_available:
            try:
                vec_count = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            except Exception:
                pass

        _probe_vec, probe_meta = _embed_text("probe", dim=EMBEDDING_DIM)
        return {
            "fts_entries": fts_count,
            "meta_entries": meta_count,
            "vec_entries": vec_count,
            "last_indexed_id": last_id,
            "vec_available": self._vec_available,
            "embedding_dim": EMBEDDING_DIM,
            "embedder": probe_meta,
            "db_path": str(self.db_path)
        }

    def sync_from_bus(self, events_path: Path):
        """Incrementally sync index from events.ndjson using last_indexed_id."""
        stats = self.stats()
        last_id = stats.get("last_indexed_id")
        
        indexed = 0
        skipped = 0
        
        # Use unified BusConsumer with state management
        state_path = self.db_path.with_suffix(".consumer_state.json")
        consumer = BusConsumer(events_path, state_path=state_path)
        
        # Override last_id from DB stats if not in consumer state
        if not consumer._last_id:
            consumer._last_id = last_id

        topic_pattern = "|".join(TARGET_TOPICS)
        BATCH_SIZE = 1000

        def process_event(event):
            nonlocal indexed, skipped
            if self.index_event(event, commit=False):
                indexed += 1
            else:
                skipped += 1
            
            if (indexed + skipped) % BATCH_SIZE == 0:
                self.conn.commit()

        consumer.scan(topic_filter=topic_pattern, callback=process_event)
            
        self.conn.commit()
        return {"indexed": indexed, "skipped": skipped, "mode": "incremental"}

    def search_symbolic(self, query: str, limit: int = 10) -> list[dict]:
        """Strict entity match search in Graphiti Knowledge Graph."""
        symbolic_results = []
        try:
            from graphiti_bridge import GraphitiService
            # Resolve root from DB path robustly
            db_res = self.db_path.resolve()
            # Try current, parent, grandparent, or great-grandparent
            potential_roots = [
                db_res.parent, 
                db_res.parents[1] if len(db_res.parents) > 1 else None,
                db_res.parents[2] if len(db_res.parents) > 2 else None
            ]
            service = None
            for r in potential_roots:
                if r and (r / ".pluribus" / "kg").exists():
                    service = GraphitiService(r)
                    break
            
            if not service:
                service = GraphitiService(db_res.parent)

            kg_res = service.query_facts(subject=query, limit=limit)
            for f in kg_res.get("facts", []):
                symbolic_results.append({
                    "id": f["id"],
                    "content_snippet": f"Fact: {f['subject']['name']} {f['predicate']} {f['object']['name']}",
                    "mode": "symbolic",
                    "provenance_id": f.get("provenance_id")
                })
        except Exception:
            pass
        return symbolic_results

    def search_unified(self, query: str, limit: int = 10) -> list[dict]:
        """Unified retrieval: RRF across BM25, Semantic, and Knowledge Graph (Symbolic)."""
        # 1. Get results from Neural layers
        bm25_results = self.search_bm25(query, limit * 2)
        semantic_results = self.search_semantic(query, limit * 2)
        
        # 2. Get results from Symbolic layer (Graphiti)
        symbolic_results = self.search_symbolic(query, limit * 2)

        # 3. Reciprocal Rank Fusion (RRF)
        # Weights: Vector(0.5), KG(0.3), Lexical(0.2)
        scores: dict[str, float] = {}
        details: dict[str, dict] = {}

        for rank, r in enumerate(semantic_results):
            id_ = r["id"]
            scores[id_] = scores.get(id_, 0) + 0.5 / (rank + 1)
            details[id_] = r

        for rank, r in enumerate(symbolic_results):
            id_ = r.get("provenance_id") or r["id"]
            scores[id_] = scores.get(id_, 0) + 0.3 / (rank + 1)
            if id_ not in details: details[id_] = r

        for rank, r in enumerate(bm25_results):
            id_ = r["id"]
            scores[id_] = scores.get(id_, 0) + 0.2 / (rank + 1)
            if id_ not in details: details[id_] = r

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:limit]
        return [{"id": k, "score": v, **details[k]} for k, v in ranked]

    def close(self):
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="RAG Vector Store with sqlite-vec (Hybrid BM25 + Semantic)")
    parser.add_argument("--db", default=DB_PATH, help="Database path")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    subparsers.add_parser("init", help="Initialize database schema")

    # index-event
    idx_p = subparsers.add_parser("index-event", help="Index a bus event")
    idx_p.add_argument("event_json", help="Event JSON string")

    # sync-from-bus
    sync_p = subparsers.add_parser("sync-from-bus", help="Incrementally sync from events.ndjson")
    sync_p.add_argument("--events", default="/pluribus/.pluribus/bus/events.ndjson", help="Events file")

    # rebuild
    rebuild_p = subparsers.add_parser("rebuild", help="Rebuild from events.ndjson")
    rebuild_p.add_argument("--events", default="/pluribus/.pluribus/bus/events.ndjson", help="Events file")

    # search
    search_p = subparsers.add_parser("search", help="Search the index")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--limit", type=int, default=10, help="Max results")
    search_p.add_argument("--mode", choices=["hybrid", "bm25", "semantic", "unified", "symbolic"], default="hybrid")
    search_p.add_argument("--bm25-weight", type=float, default=0.3)
    search_p.add_argument("--semantic-weight", type=float, default=0.7)

    # stats
    subparsers.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()

    db_path = Path(args.db)
    
    # Instantiate
    rag = VectorRAG(db_path, load_model=args.command in ("index-event", "rebuild", "search"))

    if args.command == "init":
        vec_ok = rag.init_schema()
        print(json.dumps({"status": "initialized", "db": str(db_path), "vec_available": vec_ok}))

    elif args.command == "index-event":
        rag.init_schema()
        event = json.loads(args.event_json)
        event_id = rag.index_event(event)
        if event_id:
            print(f"Indexed: {event_id}")
        else:
            print("Skipped (no content or wrong topic)")

    elif args.command == "sync-from-bus":
        rag.init_schema()
        result = rag.sync_from_bus(Path(args.events))
        print(json.dumps(result, indent=2))

    elif args.command == "rebuild":
        rag.init_schema()
        result = rag.rebuild_from_bus(Path(args.events))
        print(json.dumps(result, indent=2))

    elif args.command == "search":
        rag.init_schema()
        if args.mode == "hybrid":
            results = rag.hybrid_search(
                args.query,
                args.limit,
                args.bm25_weight,
                args.semantic_weight
            )
        elif args.mode == "bm25":
            results = rag.search_bm25(args.query, args.limit)
        elif args.mode == "semantic":
            results = rag.search_semantic(args.query, args.limit)
        elif args.mode == "unified":
            results = rag.search_unified(args.query, args.limit)
        elif args.mode == "symbolic":
            results = rag.search_symbolic(args.query, args.limit)
        else:
            results = []

        for r in results:
            score = r.get("score", r.get("bm25_score", r.get("semantic_score", 0)))
            print(f"[{score:.4f}] {r['id']} - {r.get('content_snippet', '')[:50]}...")

        if not results:
            print("No results found")

    elif args.command == "stats":
        rag.init_schema()
        stats = rag.stats()
        print(json.dumps(stats, indent=2))

    rag.close()


if __name__ == "__main__":
    main()
