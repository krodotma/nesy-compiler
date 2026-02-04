#!/usr/bin/env python3
"""
dialogos_search.py - Semantic search API for the Dialogos ground truth system.

Provides multi-backend search across dialogos traces with:
- ChromaDB vector search (primary, if available)
- QA IR fingerprint matching (fallback)
- Direct trace file grep (fallback)

Usage as module:
    from dialogos_search import DialogosSearch
    searcher = DialogosSearch()
    results = searcher.search("user login flow", top_k=5)

Usage as CLI:
    python3 dialogos_search.py search "query text" --top-k 10
    python3 dialogos_search.py session <session_id>
    python3 dialogos_search.py actor <actor_name>
    python3 dialogos_search.py similar <prompt_sha256>

Author: claude-codex
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

sys.dont_write_bytecode = True

# ============================================================================
# Constants & Configuration
# ============================================================================

DEFAULT_BUS_DIR = "/pluribus/.pluribus/bus"
DEFAULT_EVENTS_FILE = "events.ndjson"
DEFAULT_TOP_K = 10
CHROMA_COLLECTION_NAME = "dialogos_traces"

# Topics that are part of the dialogos ground truth system
DIALOGOS_TOPICS = frozenset({
    "dialogos.submit",
    "dialogos.cell.start",
    "dialogos.cell.end",
    "dialogos.response",
    "dialogos.error",
    "dialogos.trace",
    "dialogos.feedback",
})


def now_iso_utc() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_iso_timestamp(iso_str: str | None) -> float | None:
    """Parse ISO timestamp to epoch float, or None if invalid."""
    if not iso_str:
        return None
    try:
        # Handle various ISO formats
        iso_str = iso_str.rstrip("Z").replace("+00:00", "")
        if "T" in iso_str:
            dt = datetime.strptime(iso_str[:19], "%Y-%m-%dT%H:%M:%S")
            return dt.replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, TypeError):
        pass
    return None


# ============================================================================
# Search Result Types
# ============================================================================

@dataclass
class SearchResult:
    """A single search result entry."""
    id: str
    score: float
    content: str
    actor: str
    ts: str
    topic: str = ""
    session_id: str | None = None
    trace_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "id": self.id,
            "score": round(self.score, 4),
            "content": self.content,
            "actor": self.actor,
            "ts": self.ts,
            "topic": self.topic,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
        }


@dataclass
class SearchResponse:
    """Response wrapper for search operations."""
    query: str
    results: list[SearchResult]
    backend: Literal["chroma", "qa_ir", "trace"]
    latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "backend": self.backend,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


# ============================================================================
# Backend: ChromaDB Vector Search
# ============================================================================

class ChromaBackend:
    """ChromaDB vector search backend (primary)."""

    def __init__(self, persist_dir: str | None = None) -> None:
        self.available = False
        self.client = None
        self.collection = None
        self._init_chroma(persist_dir)

    def _init_chroma(self, persist_dir: str | None) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings

            if persist_dir:
                settings = Settings(
                    persist_directory=persist_dir,
                    anonymized_telemetry=False,
                )
                self.client = chromadb.Client(settings)
            else:
                # Ephemeral in-memory client
                self.client = chromadb.Client()

            self.collection = self.client.get_or_create_collection(
                name=CHROMA_COLLECTION_NAME,
                metadata={"description": "Dialogos ground truth traces"}
            )
            self.available = True
        except ImportError:
            self.available = False
        except Exception as e:
            sys.stderr.write(f"WARN: ChromaDB init failed: {e}\n")
            self.available = False

    def index_event(self, event: dict[str, Any]) -> bool:
        """Index a single event into ChromaDB."""
        if not self.available or not self.collection:
            return False

        try:
            event_id = event.get("id", sha256_text(json.dumps(event)))
            content = self._extract_content(event)
            if not content:
                return False

            metadata = {
                "actor": event.get("actor", "unknown"),
                "topic": event.get("topic", ""),
                "ts": event.get("iso", ""),
                "trace_id": event.get("trace_id") or "",
                "session_id": event.get("data", {}).get("session_id") or "",
            }

            self.collection.upsert(
                ids=[event_id],
                documents=[content],
                metadatas=[metadata],
            )
            return True
        except Exception as e:
            sys.stderr.write(f"WARN: ChromaDB index failed: {e}\n")
            return False

    def _extract_content(self, event: dict[str, Any]) -> str:
        """Extract searchable content from event."""
        parts = []

        # Add topic
        topic = event.get("topic", "")
        if topic:
            parts.append(f"[{topic}]")

        # Add actor
        actor = event.get("actor", "")
        if actor:
            parts.append(f"actor:{actor}")

        # Extract text from data
        data = event.get("data", {})
        if isinstance(data, dict):
            # Common text fields
            for key in ("prompt", "response", "content", "message", "error", "text"):
                if key in data and isinstance(data[key], str):
                    parts.append(data[key])
            # req_id for correlation
            if "req_id" in data:
                parts.append(f"req:{data['req_id']}")
        elif isinstance(data, str):
            parts.append(data)

        return " ".join(parts)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search using semantic similarity."""
        if not self.available or not self.collection:
            return []

        try:
            kwargs: dict[str, Any] = {
                "query_texts": [query],
                "n_results": top_k,
            }
            if where:
                kwargs["where"] = where

            results = self.collection.query(**kwargs)

            output: list[SearchResult] = []
            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, doc_id in enumerate(ids):
                # Convert distance to similarity score (0-1)
                distance = distances[i] if i < len(distances) else 1.0
                score = max(0.0, 1.0 - (distance / 2.0))

                meta = metadatas[i] if i < len(metadatas) else {}
                content = documents[i] if i < len(documents) else ""

                output.append(SearchResult(
                    id=doc_id,
                    score=score,
                    content=content[:500],  # Truncate for response
                    actor=meta.get("actor", "unknown"),
                    ts=meta.get("ts", ""),
                    topic=meta.get("topic", ""),
                    session_id=meta.get("session_id") or None,
                    trace_id=meta.get("trace_id") or None,
                ))

            return output
        except Exception as e:
            sys.stderr.write(f"WARN: ChromaDB search failed: {e}\n")
            return []


# ============================================================================
# Backend: QA IR Fingerprint Matching
# ============================================================================

class QAIRBackend:
    """QA IR fingerprint-based fallback search."""

    def __init__(self, events_path: str) -> None:
        self.events_path = events_path

    def _compute_fingerprint(self, event: dict[str, Any]) -> str:
        """Compute fingerprint for an event (from qa_ir.py logic)."""
        topic = event.get("topic", "")
        actor = event.get("actor", "")
        data = event.get("data")
        payload = f"{topic}\n{actor}\n{json.dumps(data, sort_keys=True, separators=(',', ':'))}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _tokenize(self, text: str) -> set[str]:
        """Simple tokenization for matching."""
        text = text.lower()
        # Remove punctuation and split
        tokens = re.findall(r"[a-z0-9_]+", text)
        return set(tokens)

    def _score_match(self, query_tokens: set[str], event: dict[str, Any]) -> float:
        """Score how well event matches query tokens."""
        # Build event text
        parts = [
            event.get("topic", ""),
            event.get("actor", ""),
        ]
        data = event.get("data", {})
        if isinstance(data, dict):
            for key in ("prompt", "response", "content", "message", "error", "req_id"):
                if key in data:
                    parts.append(str(data[key]))
        elif isinstance(data, str):
            parts.append(data)

        event_text = " ".join(parts)
        event_tokens = self._tokenize(event_text)

        if not query_tokens or not event_tokens:
            return 0.0

        # Jaccard-like overlap
        intersection = len(query_tokens & event_tokens)
        union = len(query_tokens | event_tokens)

        if union == 0:
            return 0.0

        return intersection / union

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        actor: str | None = None,
        session_id: str | None = None,
        since: str | None = None,
    ) -> list[SearchResult]:
        """Search using token matching."""
        if not os.path.exists(self.events_path):
            return []

        query_tokens = self._tokenize(query)
        since_ts = parse_iso_timestamp(since) if since else None

        scored: list[tuple[float, dict[str, Any]]] = []

        try:
            with open(self.events_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Filter by topic (dialogos-related)
                    topic = event.get("topic", "")
                    if not topic.startswith("dialogos"):
                        continue

                    # Filter by actor
                    if actor and event.get("actor") != actor:
                        continue

                    # Filter by session_id
                    if session_id:
                        data = event.get("data", {})
                        if isinstance(data, dict):
                            if data.get("session_id") != session_id:
                                continue
                        else:
                            continue

                    # Filter by timestamp
                    if since_ts:
                        event_ts = event.get("ts")
                        if event_ts and event_ts < since_ts:
                            continue

                    score = self._score_match(query_tokens, event)
                    if score > 0:
                        scored.append((score, event))
        except Exception as e:
            sys.stderr.write(f"WARN: QA IR search failed: {e}\n")
            return []

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[SearchResult] = []
        for score, event in scored[:top_k]:
            data = event.get("data", {})
            content = ""
            if isinstance(data, dict):
                content = data.get("prompt") or data.get("response") or data.get("message") or json.dumps(data)[:200]
            elif isinstance(data, str):
                content = data[:200]

            results.append(SearchResult(
                id=event.get("id", self._compute_fingerprint(event)),
                score=score,
                content=content[:500],
                actor=event.get("actor", "unknown"),
                ts=event.get("iso", ""),
                topic=event.get("topic", ""),
                session_id=data.get("session_id") if isinstance(data, dict) else None,
                trace_id=event.get("trace_id"),
                data=data if isinstance(data, dict) else {},
            ))

        return results

    def find_by_fingerprint(self, fingerprint: str, top_k: int = 5) -> list[SearchResult]:
        """Find events with similar fingerprints (prefix match)."""
        if not os.path.exists(self.events_path):
            return []

        results: list[SearchResult] = []
        prefix = fingerprint[:16]  # Use first 16 chars for matching

        try:
            with open(self.events_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if len(results) >= top_k * 10:  # Read more than needed, score later
                        break

                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    topic = event.get("topic", "")
                    if not topic.startswith("dialogos"):
                        continue

                    event_fp = self._compute_fingerprint(event)

                    # Score based on fingerprint similarity
                    common = sum(1 for a, b in zip(fingerprint, event_fp) if a == b)
                    score = common / len(fingerprint) if fingerprint else 0.0

                    if score > 0.1:  # Threshold
                        data = event.get("data", {})
                        content = ""
                        if isinstance(data, dict):
                            content = data.get("prompt") or data.get("response") or json.dumps(data)
                            content = str(content)[:200] if content else ""

                        results.append(SearchResult(
                            id=event.get("id", event_fp),
                            score=score,
                            content=str(content)[:500] if content else "",
                            actor=event.get("actor", "unknown"),
                            ts=event.get("iso", ""),
                            topic=topic,
                            session_id=data.get("session_id") if isinstance(data, dict) else None,
                            trace_id=event.get("trace_id"),
                            data=data if isinstance(data, dict) else {},
                        ))
        except Exception as e:
            sys.stderr.write(f"WARN: Fingerprint search failed: {e}\n")

        # Sort and truncate
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ============================================================================
# Backend: Direct Trace Grep
# ============================================================================

class TraceGrepBackend:
    """Direct grep-based trace search (last resort fallback)."""

    def __init__(self, events_path: str) -> None:
        self.events_path = events_path

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        actor: str | None = None,
    ) -> list[SearchResult]:
        """Search using grep."""
        if not os.path.exists(self.events_path):
            return []

        # Build grep pattern
        pattern = re.escape(query)

        try:
            cmd = ["grep", "-i", "-E", pattern, self.events_path]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            lines = result.stdout.strip().split("\n")
            results: list[SearchResult] = []

            for line in lines[:top_k * 2]:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Filter by actor if specified
                if actor and event.get("actor") != actor:
                    continue

                topic = event.get("topic", "")
                data = event.get("data", {})
                content = ""
                if isinstance(data, dict):
                    content = data.get("prompt") or data.get("response") or json.dumps(data)[:200]
                elif isinstance(data, str):
                    content = data[:200]

                results.append(SearchResult(
                    id=event.get("id", "unknown"),
                    score=0.5,  # Grep matches get fixed score
                    content=content[:500],
                    actor=event.get("actor", "unknown"),
                    ts=event.get("iso", ""),
                    topic=topic,
                    session_id=data.get("session_id") if isinstance(data, dict) else None,
                    trace_id=event.get("trace_id"),
                    data=data if isinstance(data, dict) else {},
                ))

            return results[:top_k]
        except subprocess.TimeoutExpired:
            sys.stderr.write("WARN: Grep search timed out\n")
            return []
        except Exception as e:
            sys.stderr.write(f"WARN: Grep search failed: {e}\n")
            return []


# ============================================================================
# Main Search API
# ============================================================================

class DialogosSearch:
    """
    Semantic search API for the Dialogos ground truth system.

    Supports multiple backends with automatic fallback:
    1. ChromaDB vector search (primary)
    2. QA IR fingerprint matching (fallback)
    3. Direct trace grep (last resort)

    Example:
        searcher = DialogosSearch()
        results = searcher.search("user authentication flow", top_k=10)
        print(results.to_dict())
    """

    def __init__(
        self,
        bus_dir: str | None = None,
        chroma_persist_dir: str | None = None,
    ) -> None:
        """
        Initialize the search API.

        Args:
            bus_dir: Path to bus directory (default: /pluribus/.pluribus/bus)
            chroma_persist_dir: Path to ChromaDB persistence (optional)
        """
        self.bus_dir = bus_dir or os.environ.get("PLURIBUS_BUS_DIR") or DEFAULT_BUS_DIR
        self.events_path = os.path.join(self.bus_dir, DEFAULT_EVENTS_FILE)

        # Initialize backends
        self.chroma = ChromaBackend(chroma_persist_dir)
        self.qa_ir = QAIRBackend(self.events_path)
        self.trace_grep = TraceGrepBackend(self.events_path)

    def _emit_bus_event(self, topic: str, data: dict[str, Any]) -> None:
        """Emit a bus event for observability."""
        try:
            from agent_bus import resolve_bus_paths, emit_event
            paths = resolve_bus_paths(self.bus_dir)
            emit_event(
                paths,
                topic=topic,
                kind="metric",
                level="info",
                actor="dialogos-search",
                data=data,
                trace_id=None,
                run_id=None,
                durable=False,
            )
        except Exception:
            # Best-effort, non-fatal
            pass

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        actor: str | None = None,
        session_id: str | None = None,
        since: str | None = None,
    ) -> SearchResponse:
        """
        Search dialogos traces using semantic similarity.

        Args:
            query: Search query text
            top_k: Maximum number of results (default: 10)
            actor: Filter by actor name
            session_id: Filter by session ID
            since: ISO timestamp - only return events after this time

        Returns:
            SearchResponse with results and metadata
        """
        start_time = time.perf_counter()
        backend_used: Literal["chroma", "qa_ir", "trace"] = "trace"
        results: list[SearchResult] = []
        error: str | None = None

        # Try ChromaDB first
        if self.chroma.available:
            where_filter = {}
            if actor:
                where_filter["actor"] = actor
            if session_id:
                where_filter["session_id"] = session_id

            results = self.chroma.search(
                query,
                top_k=top_k,
                where=where_filter if where_filter else None,
            )
            if results:
                backend_used = "chroma"

        # Fallback to QA IR
        if not results:
            results = self.qa_ir.search(
                query,
                top_k=top_k,
                actor=actor,
                session_id=session_id,
                since=since,
            )
            if results:
                backend_used = "qa_ir"

        # Last resort: grep
        if not results:
            results = self.trace_grep.search(query, top_k=top_k, actor=actor)
            backend_used = "trace"

        latency_ms = (time.perf_counter() - start_time) * 1000

        response = SearchResponse(
            query=query,
            results=results,
            backend=backend_used,
            latency_ms=latency_ms,
            error=error,
        )

        # Emit bus events
        self._emit_bus_event("dialogos.search.query", {
            "query": query[:100],
            "top_k": top_k,
            "actor": actor,
            "session_id": session_id,
        })
        self._emit_bus_event("dialogos.search.result", {
            "backend": backend_used,
            "result_count": len(results),
            "latency_ms": round(latency_ms, 2),
        })

        return response

    def find_similar_prompts(
        self,
        prompt_sha256: str,
        top_k: int = 5,
    ) -> SearchResponse:
        """
        Find prompts similar to the given prompt hash.

        Args:
            prompt_sha256: SHA256 hash of the prompt to find similar ones for
            top_k: Maximum number of results

        Returns:
            SearchResponse with similar prompts
        """
        start_time = time.perf_counter()

        # Use fingerprint matching
        results = self.qa_ir.find_by_fingerprint(prompt_sha256, top_k=top_k)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return SearchResponse(
            query=f"similar:{prompt_sha256[:16]}...",
            results=results,
            backend="qa_ir",
            latency_ms=latency_ms,
        )

    def get_session_context(
        self,
        session_id: str,
        limit: int = 20,
    ) -> SearchResponse:
        """
        Get all events for a specific session.

        Args:
            session_id: Session ID to retrieve
            limit: Maximum number of events

        Returns:
            SearchResponse with session events ordered chronologically
        """
        start_time = time.perf_counter()
        results: list[SearchResult] = []

        if not os.path.exists(self.events_path):
            return SearchResponse(
                query=f"session:{session_id}",
                results=[],
                backend="trace",
                latency_ms=0,
                error="Events file not found",
            )

        try:
            with open(self.events_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if len(results) >= limit * 2:  # Read extra, sort later
                        break

                    line = line.strip()
                    if not line:
                        continue

                    if session_id not in line:  # Quick pre-filter
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    data = event.get("data", {})
                    if isinstance(data, dict) and data.get("session_id") == session_id:
                        content = data.get("prompt") or data.get("response") or json.dumps(data)[:200]
                        results.append(SearchResult(
                            id=event.get("id", "unknown"),
                            score=1.0,
                            content=content[:500],
                            actor=event.get("actor", "unknown"),
                            ts=event.get("iso", ""),
                            topic=event.get("topic", ""),
                            session_id=session_id,
                            trace_id=event.get("trace_id"),
                            data=data,
                        ))
        except Exception as e:
            return SearchResponse(
                query=f"session:{session_id}",
                results=[],
                backend="trace",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

        # Sort by timestamp
        results.sort(key=lambda r: r.ts)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return SearchResponse(
            query=f"session:{session_id}",
            results=results[:limit],
            backend="trace",
            latency_ms=latency_ms,
        )

    def get_actor_history(
        self,
        actor: str,
        limit: int = 50,
    ) -> SearchResponse:
        """
        Get interaction history for a specific actor.

        Args:
            actor: Actor name to retrieve history for
            limit: Maximum number of events

        Returns:
            SearchResponse with actor's events ordered by recency
        """
        start_time = time.perf_counter()
        results: list[SearchResult] = []

        if not os.path.exists(self.events_path):
            return SearchResponse(
                query=f"actor:{actor}",
                results=[],
                backend="trace",
                latency_ms=0,
                error="Events file not found",
            )

        try:
            # Read from end for most recent first
            with open(self.events_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            # Process in reverse for recency
            for line in reversed(lines):
                if len(results) >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                if f'"actor":"{actor}"' not in line:  # Quick pre-filter
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("actor") != actor:
                    continue

                topic = event.get("topic", "")
                if not topic.startswith("dialogos"):
                    continue

                data = event.get("data", {})
                content = ""
                if isinstance(data, dict):
                    content = data.get("prompt") or data.get("response") or json.dumps(data)[:200]
                elif isinstance(data, str):
                    content = data[:200]

                results.append(SearchResult(
                    id=event.get("id", "unknown"),
                    score=1.0,
                    content=content[:500],
                    actor=actor,
                    ts=event.get("iso", ""),
                    topic=topic,
                    session_id=data.get("session_id") if isinstance(data, dict) else None,
                    trace_id=event.get("trace_id"),
                    data=data if isinstance(data, dict) else {},
                ))
        except Exception as e:
            return SearchResponse(
                query=f"actor:{actor}",
                results=[],
                backend="trace",
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e),
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        return SearchResponse(
            query=f"actor:{actor}",
            results=results,
            backend="trace",
            latency_ms=latency_ms,
        )

    def index_all(self, max_events: int = 100000) -> dict[str, Any]:
        """
        Index all dialogos events into ChromaDB.

        Args:
            max_events: Maximum events to index

        Returns:
            Indexing statistics
        """
        if not self.chroma.available:
            return {"error": "ChromaDB not available", "indexed": 0}

        if not os.path.exists(self.events_path):
            return {"error": "Events file not found", "indexed": 0}

        indexed = 0
        errors = 0
        start_time = time.perf_counter()

        try:
            with open(self.events_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if indexed >= max_events:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        errors += 1
                        continue

                    topic = event.get("topic", "")
                    if not topic.startswith("dialogos"):
                        continue

                    if self.chroma.index_event(event):
                        indexed += 1
                    else:
                        errors += 1
        except Exception as e:
            return {
                "error": str(e),
                "indexed": indexed,
                "errors": errors,
                "elapsed_s": time.perf_counter() - start_time,
            }

        return {
            "indexed": indexed,
            "errors": errors,
            "elapsed_s": round(time.perf_counter() - start_time, 2),
        }


# ============================================================================
# CLI Interface
# ============================================================================

def cmd_search(args: argparse.Namespace) -> int:
    """Handle search command."""
    searcher = DialogosSearch(bus_dir=args.bus_dir)
    response = searcher.search(
        args.query,
        top_k=args.top_k,
        actor=args.actor,
        session_id=args.session_id,
        since=args.since,
    )

    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
    else:
        print(f"Query: {response.query}")
        print(f"Backend: {response.backend} | Latency: {response.latency_ms:.1f}ms")
        print(f"Results: {len(response.results)}")
        print("-" * 60)
        for i, r in enumerate(response.results, 1):
            print(f"{i}. [{r.score:.3f}] {r.actor} @ {r.ts}")
            print(f"   Topic: {r.topic}")
            print(f"   {r.content[:100]}...")
            print()

    return 0


def cmd_session(args: argparse.Namespace) -> int:
    """Handle session command."""
    searcher = DialogosSearch(bus_dir=args.bus_dir)
    response = searcher.get_session_context(args.session_id, limit=args.limit)

    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
    else:
        print(f"Session: {args.session_id}")
        print(f"Events: {len(response.results)} | Latency: {response.latency_ms:.1f}ms")
        print("-" * 60)
        for r in response.results:
            print(f"[{r.ts}] {r.actor} - {r.topic}")
            print(f"  {r.content[:80]}...")
            print()

    return 0


def cmd_actor(args: argparse.Namespace) -> int:
    """Handle actor command."""
    searcher = DialogosSearch(bus_dir=args.bus_dir)
    response = searcher.get_actor_history(args.actor_name, limit=args.limit)

    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
    else:
        print(f"Actor: {args.actor_name}")
        print(f"Events: {len(response.results)} | Latency: {response.latency_ms:.1f}ms")
        print("-" * 60)
        for r in response.results:
            print(f"[{r.ts}] {r.topic}")
            print(f"  {r.content[:80]}...")
            print()

    return 0


def cmd_similar(args: argparse.Namespace) -> int:
    """Handle similar command."""
    searcher = DialogosSearch(bus_dir=args.bus_dir)
    response = searcher.find_similar_prompts(args.prompt_sha256, top_k=args.top_k)

    if args.json:
        print(json.dumps(response.to_dict(), indent=2))
    else:
        print(f"Similar to: {args.prompt_sha256[:32]}...")
        print(f"Results: {len(response.results)} | Latency: {response.latency_ms:.1f}ms")
        print("-" * 60)
        for i, r in enumerate(response.results, 1):
            print(f"{i}. [{r.score:.3f}] {r.actor} @ {r.ts}")
            print(f"   {r.content[:100]}...")
            print()

    return 0


def cmd_index(args: argparse.Namespace) -> int:
    """Handle index command."""
    searcher = DialogosSearch(bus_dir=args.bus_dir)
    stats = searcher.index_all(max_events=args.max_events)

    print(json.dumps(stats, indent=2))
    return 0 if "error" not in stats else 1


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="dialogos_search.py",
        description="Semantic search for Dialogos ground truth system",
    )
    parser.add_argument(
        "--bus-dir",
        default=None,
        help=f"Bus directory (default: {DEFAULT_BUS_DIR})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # search command
    search_parser = subparsers.add_parser("search", help="Search dialogos traces")
    search_parser.add_argument("query", help="Search query text")
    search_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Max results")
    search_parser.add_argument("--actor", help="Filter by actor")
    search_parser.add_argument("--session-id", help="Filter by session ID")
    search_parser.add_argument("--since", help="ISO timestamp - events after this time")
    search_parser.set_defaults(func=cmd_search)

    # session command
    session_parser = subparsers.add_parser("session", help="Get session context")
    session_parser.add_argument("session_id", help="Session ID")
    session_parser.add_argument("--limit", type=int, default=20, help="Max events")
    session_parser.set_defaults(func=cmd_session)

    # actor command
    actor_parser = subparsers.add_parser("actor", help="Get actor history")
    actor_parser.add_argument("actor_name", help="Actor name")
    actor_parser.add_argument("--limit", type=int, default=50, help="Max events")
    actor_parser.set_defaults(func=cmd_actor)

    # similar command
    similar_parser = subparsers.add_parser("similar", help="Find similar prompts")
    similar_parser.add_argument("prompt_sha256", help="SHA256 hash of prompt")
    similar_parser.add_argument("--top-k", type=int, default=5, help="Max results")
    similar_parser.set_defaults(func=cmd_similar)

    # index command
    index_parser = subparsers.add_parser("index", help="Index events into ChromaDB")
    index_parser.add_argument("--max-events", type=int, default=100000, help="Max events to index")
    index_parser.set_defaults(func=cmd_index)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
