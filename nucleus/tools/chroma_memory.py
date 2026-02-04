#!/usr/bin/env python3
"""
chroma_memory.py - ChromaDB-based Agentic Memory for Pluribus

Provides persistent vector storage for:
- Agent context/memory across sessions
- Knowledge graph embeddings
- RAG augmentation (alongside existing sqlite-vec)

Features:
- HNSW indexing for fast similarity search
- Sub-10ms query times
- Integration with Pluribus bus events

Usage:
    python -m nucleus.tools.chroma_memory status
    python -m nucleus.tools.chroma_memory store "document" --collection agent_memory
    python -m nucleus.tools.chroma_memory query "search query" --top_k 5
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

# Paths
CHROMA_PATH = Path(os.environ.get("PLURIBUS_CHROMA_DIR", "/pluribus/.pluribus/index/chroma"))
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
EVENTS_PATH = BUS_DIR / "events.ndjson"

# Collection names
COLLECTIONS = {
    "agent_memory": "Agent session context and working memory",
    "knowledge_graph": "Knowledge graph node embeddings",
    "bus_events": "Curated bus event embeddings for RAG",
    "sota_artifacts": "SOTA distillation artifacts",
}

# Default embedding model (sentence-transformers compatible)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def emit_bus_event(topic: str, kind: str, data: dict, level: str = "info"):
    """Emit event to Pluribus bus."""
    if not EVENTS_PATH.parent.exists():
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "id": f"{int(time.time() * 1000)}-{os.urandom(4).hex()}",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": "chroma-memory",
        "ts": time.time(),
        "iso": datetime.utcnow().isoformat() + "Z",
        "data": data,
    }

    with open(EVENTS_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


class ChromaMemory:
    """ChromaDB-based agentic memory manager."""

    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = persist_path or CHROMA_PATH
        self._client = None
        self._collections = {}

    @property
    def client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                self.persist_path.mkdir(parents=True, exist_ok=True)

                self._client = chromadb.PersistentClient(
                    path=str(self.persist_path),
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    )
                )
            except ImportError:
                raise RuntimeError(
                    "ChromaDB not installed. Run: pip install chromadb"
                )

        return self._client

    def get_collection(self, name: str):
        """Get or create a collection."""
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    def store(
        self,
        collection_name: str,
        documents: list[str],
        ids: Optional[list[str]] = None,
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> dict:
        """Store documents in collection.

        If embeddings not provided, uses ChromaDB's default embedding function.
        """
        collection = self.get_collection(collection_name)

        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc-{int(time.time() * 1000)}-{i}" for i in range(len(documents))]

        # Add default metadata
        if metadatas is None:
            metadatas = [{}] * len(documents)

        for i, meta in enumerate(metadatas):
            meta["stored_at"] = datetime.utcnow().isoformat() + "Z"
            meta["doc_id"] = ids[i]

        # Store
        start = time.time()

        if embeddings:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        latency = (time.time() - start) * 1000

        # Emit bus event
        emit_bus_event(
            "memory.chroma.upsert",
            "log",
            {
                "collection": collection_name,
                "count": len(documents),
                "ids": ids,
                "latency_ms": latency,
            },
        )

        return {
            "stored": len(documents),
            "ids": ids,
            "latency_ms": latency,
        }

    def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        where: Optional[dict] = None,
        include: Optional[list[str]] = None,
    ) -> dict:
        """Query collection for similar documents."""
        collection = self.get_collection(collection_name)

        start = time.time()

        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=include or ["documents", "metadatas", "distances"],
        )

        latency = (time.time() - start) * 1000

        # Emit bus event
        emit_bus_event(
            "memory.chroma.query",
            "log",
            {
                "collection": collection_name,
                "query_len": len(query_text),
                "n_results": n_results,
                "found": len(results["ids"][0]) if results["ids"] else 0,
                "latency_ms": latency,
            },
        )

        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results.get("documents") else [],
            "metadatas": results["metadatas"][0] if results.get("metadatas") else [],
            "distances": results["distances"][0] if results.get("distances") else [],
            "latency_ms": latency,
        }

    def recall_context(
        self,
        actor: str,
        query: str,
        session_id: Optional[str] = None,
        max_items: int = 10,
    ) -> dict:
        """Recall relevant context for an agent.

        Searches agent_memory collection with optional session filtering.
        """
        where_filter = {"actor": actor}
        if session_id:
            where_filter["session_id"] = session_id

        results = self.query(
            "agent_memory",
            query,
            n_results=max_items,
            where=where_filter,
        )

        # Emit context recall event
        emit_bus_event(
            "memory.context.recall",
            "log",
            {
                "actor": actor,
                "session_id": session_id,
                "query_len": len(query),
                "recalled": len(results["ids"]),
            },
        )

        return results

    def store_agent_context(
        self,
        actor: str,
        content: str,
        session_id: Optional[str] = None,
        topic: Optional[str] = None,
        req_id: Optional[str] = None,
    ) -> dict:
        """Store agent working context."""
        doc_id = f"ctx-{actor}-{int(time.time() * 1000)}"

        metadata = {
            "actor": actor,
            "session_id": session_id or "default",
            "topic": topic or "general",
        }

        if req_id:
            metadata["req_id"] = req_id

        return self.store(
            "agent_memory",
            documents=[content],
            ids=[doc_id],
            metadatas=[metadata],
        )

    def get_stats(self) -> dict:
        """Get memory statistics."""
        stats = {
            "path": str(self.persist_path),
            "collections": {},
        }

        for name in COLLECTIONS:
            try:
                collection = self.get_collection(name)
                count = collection.count()
                stats["collections"][name] = {
                    "count": count,
                    "description": COLLECTIONS[name],
                }
            except Exception as e:
                stats["collections"][name] = {
                    "error": str(e),
                }

        return stats

    def ingest_bus_events(
        self,
        topics: Optional[list[str]] = None,
        since_iso: Optional[str] = None,
        limit: int = 1000,
    ) -> dict:
        """Ingest bus events into chroma for RAG.

        Filters by topics of interest and stores with embeddings.
        """
        if not EVENTS_PATH.exists():
            return {"ingested": 0, "error": "No events file"}

        # Default topics of interest
        if topics is None:
            topics = [
                "dialogos.cell.end",
                "plurichat.response",
                "strp.response",
                "infer_sync.response",
                "a2a.negotiate.response",
            ]

        documents = []
        ids = []
        metadatas = []

        with open(EVENTS_PATH) as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Filter by topic
                if not any(event.get("topic", "").startswith(t.rstrip("*")) for t in topics):
                    continue

                # Filter by time
                if since_iso and event.get("iso", "") < since_iso:
                    continue

                # Extract content
                data = event.get("data", {})
                content = data.get("content") or data.get("text") or data.get("message") or ""
                if not content or len(content) < 20:
                    continue

                doc_id = event.get("id", f"evt-{len(documents)}")
                documents.append(content[:4000])  # Limit length
                ids.append(doc_id)
                metadatas.append({
                    "topic": event.get("topic", ""),
                    "actor": event.get("actor", ""),
                    "iso": event.get("iso", ""),
                    "req_id": data.get("req_id") or data.get("request_id") or "",
                })

                if len(documents) >= limit:
                    break

        if not documents:
            return {"ingested": 0}

        result = self.store(
            "bus_events",
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )

        return {
            "ingested": len(documents),
            "latency_ms": result["latency_ms"],
        }


def cmd_status(args):
    """Show ChromaDB memory status."""
    try:
        memory = ChromaMemory()
        stats = memory.get_stats()

        print("=" * 60)
        print("CHROMADB AGENTIC MEMORY STATUS")
        print("=" * 60)
        print()
        print(f"Path: {stats['path']}")
        print()
        print("Collections:")
        print("-" * 40)

        for name, info in stats["collections"].items():
            if "error" in info:
                print(f"  {name}: ERROR - {info['error']}")
            else:
                print(f"  {name}: {info['count']} documents")
                print(f"    {info['description']}")
        print()

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_store(args):
    """Store document in collection."""
    try:
        memory = ChromaMemory()

        metadata = {}
        if args.actor:
            metadata["actor"] = args.actor
        if args.topic:
            metadata["topic"] = args.topic

        result = memory.store(
            args.collection,
            documents=[args.document],
            metadatas=[metadata] if metadata else None,
        )

        print(f"Stored document: {result['ids'][0]}")
        print(f"Latency: {result['latency_ms']:.1f}ms")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_query(args):
    """Query collection for similar documents."""
    try:
        memory = ChromaMemory()

        results = memory.query(
            args.collection,
            args.query,
            n_results=args.top_k,
        )

        print(f"Found {len(results['ids'])} results ({results['latency_ms']:.1f}ms):")
        print()

        for i, (doc_id, doc, meta, dist) in enumerate(zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
            results["distances"],
        )):
            print(f"[{i+1}] {doc_id} (distance: {dist:.4f})")
            print(f"    {doc[:200]}..." if len(doc) > 200 else f"    {doc}")
            if meta:
                print(f"    Metadata: {meta}")
            print()

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_ingest(args):
    """Ingest bus events into ChromaDB."""
    try:
        memory = ChromaMemory()

        result = memory.ingest_bus_events(
            topics=args.topics.split(",") if args.topics else None,
            since_iso=args.since,
            limit=args.limit,
        )

        print(f"Ingested {result['ingested']} events")
        if result.get("latency_ms"):
            print(f"Latency: {result['latency_ms']:.1f}ms")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="ChromaDB Agentic Memory for Pluribus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status
    status_parser = subparsers.add_parser("status", help="Show memory status")
    status_parser.set_defaults(func=cmd_status)

    # store
    store_parser = subparsers.add_parser("store", help="Store document")
    store_parser.add_argument("document", help="Document text to store")
    store_parser.add_argument("--collection", "-c", default="agent_memory",
                              choices=list(COLLECTIONS.keys()))
    store_parser.add_argument("--actor", "-a", help="Actor name")
    store_parser.add_argument("--topic", "-t", help="Topic tag")
    store_parser.set_defaults(func=cmd_store)

    # query
    query_parser = subparsers.add_parser("query", help="Query for similar documents")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--collection", "-c", default="agent_memory",
                              choices=list(COLLECTIONS.keys()))
    query_parser.add_argument("--top-k", "-k", type=int, default=5)
    query_parser.set_defaults(func=cmd_query)

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest bus events")
    ingest_parser.add_argument("--topics", help="Comma-separated topic prefixes")
    ingest_parser.add_argument("--since", help="ISO timestamp to start from")
    ingest_parser.add_argument("--limit", type=int, default=1000)
    ingest_parser.set_defaults(func=cmd_ingest)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
