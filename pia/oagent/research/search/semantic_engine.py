#!/usr/bin/env python3
"""
semantic_engine.py - Semantic Search Engine (Step 11)

Embedding-based code search with vector similarity.
Supports multiple embedding providers and efficient nearest-neighbor search.

PBTSO Phase: RESEARCH

Bus Topics:
- a2a.research.search.semantic
- research.search.index
- research.search.results

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class EmbeddingModel(Enum):
    """Supported embedding models."""
    OPENAI_ADA = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_3_LARGE = "text-embedding-3-large"
    LOCAL_MINILM = "all-MiniLM-L6-v2"
    LOCAL_CODEBERT = "microsoft/codebert-base"
    MOCK = "mock"  # For testing


@dataclass
class SemanticSearchConfig:
    """Configuration for semantic search engine."""

    embedding_model: str = EmbeddingModel.LOCAL_MINILM.value
    embedding_dim: int = 384  # MiniLM default
    index_path: Optional[str] = None
    max_results: int = 50
    min_similarity: float = 0.3
    chunk_size: int = 512  # Characters per chunk
    chunk_overlap: int = 64
    batch_size: int = 32
    use_gpu: bool = False

    def __post_init__(self):
        if self.index_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.index_path = f"{pluribus_root}/.pluribus/research/semantic_index"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class SearchResult:
    """Result from semantic search."""

    doc_id: str
    content: str
    path: str
    score: float
    chunk_index: int = 0
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class IndexedDocument:
    """Document indexed for semantic search."""

    doc_id: str
    path: str
    content: str
    embedding: List[float]
    chunk_index: int = 0
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    language: str = "unknown"
    indexed_at: float = field(default_factory=time.time)
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]


# ============================================================================
# Embedding Providers
# ============================================================================


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dim: int = 384):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        """Generate deterministic mock embedding based on text hash."""
        import random
        # Use text hash as seed for reproducible embeddings
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        embedding = [random.gauss(0, 1) for _ in range(self._dim)]
        # Normalize
        norm = math.sqrt(sum(x*x for x in embedding))
        return [x / norm for x in embedding]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_gpu: bool = False):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._model = None
        self._dim: Optional[int] = None

    def _ensure_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                device = "cuda" if self.use_gpu else "cpu"
                self._model = SentenceTransformer(self.model_name, device=device)
                self._dim = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._ensure_model()
        return self._dim

    def embed(self, text: str) -> List[float]:
        self._ensure_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenAI API."""

    MODEL_DIMS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._dim = self.MODEL_DIMS.get(model, 1536)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [data.embedding for data in response.data]
        except ImportError:
            raise ImportError("openai not installed. Run: pip install openai")


def get_embedding_provider(config: SemanticSearchConfig) -> EmbeddingProvider:
    """Factory function to get embedding provider based on config."""
    model = config.embedding_model

    if model == EmbeddingModel.MOCK.value:
        return MockEmbeddingProvider(config.embedding_dim)
    elif model.startswith("text-embedding"):
        return OpenAIEmbeddingProvider(model)
    else:
        return SentenceTransformerProvider(model, config.use_gpu)


# ============================================================================
# Semantic Search Engine
# ============================================================================


class SemanticSearchEngine:
    """
    Semantic search engine using embeddings and vector similarity.

    Provides embedding-based code search with support for:
    - Multiple embedding providers (OpenAI, local models)
    - Efficient nearest-neighbor search
    - Document chunking and indexing
    - Persistent index storage

    PBTSO Phase: RESEARCH

    Example:
        engine = SemanticSearchEngine()
        engine.index_document("src/main.py", content)
        results = engine.search("function that handles authentication")
    """

    def __init__(
        self,
        config: Optional[SemanticSearchConfig] = None,
        bus: Optional[AgentBus] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        """
        Initialize the semantic search engine.

        Args:
            config: Search configuration
            bus: AgentBus for event emission
            embedding_provider: Custom embedding provider (optional)
        """
        self.config = config or SemanticSearchConfig()
        self.bus = bus or AgentBus()

        # Initialize embedding provider
        if embedding_provider:
            self.embedder = embedding_provider
        else:
            self.embedder = get_embedding_provider(self.config)

        # Document storage
        self.documents: Dict[str, IndexedDocument] = {}
        self._embeddings_matrix: Optional[List[List[float]]] = None
        self._doc_ids: List[str] = []

        # Load existing index
        self._load_index()

    def index_document(
        self,
        path: str,
        content: str,
        language: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Index a document for semantic search.

        Args:
            path: File path
            content: Document content
            language: Programming language
            metadata: Additional metadata

        Returns:
            Number of chunks indexed
        """
        # Remove existing chunks for this path
        self._remove_path_chunks(path)

        # Split into chunks
        chunks = self._chunk_content(content)

        # Generate embeddings in batch
        embeddings = self.embedder.embed_batch([c["text"] for c in chunks])

        # Index each chunk
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = f"{path}::{i}"
            doc = IndexedDocument(
                doc_id=doc_id,
                path=path,
                content=chunk["text"],
                embedding=embedding,
                chunk_index=i,
                line_start=chunk["line_start"],
                line_end=chunk["line_end"],
                language=language,
            )
            self.documents[doc_id] = doc

        # Rebuild search matrix
        self._rebuild_matrix()

        # Emit event
        self.bus.emit({
            "topic": "research.search.index",
            "kind": "index",
            "data": {
                "path": path,
                "chunks": len(chunks),
                "language": language,
            }
        })

        return len(chunks)

    def index_documents_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Index multiple documents efficiently.

        Args:
            documents: List of dicts with path, content, language keys

        Returns:
            Total chunks indexed
        """
        total_chunks = 0
        all_chunks = []
        chunk_metadata = []

        # Collect all chunks
        for doc in documents:
            path = doc["path"]
            content = doc["content"]
            language = doc.get("language", "unknown")

            # Remove existing
            self._remove_path_chunks(path)

            # Split into chunks
            chunks = self._chunk_content(content)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk["text"])
                chunk_metadata.append({
                    "doc_id": f"{path}::{i}",
                    "path": path,
                    "chunk_index": i,
                    "line_start": chunk["line_start"],
                    "line_end": chunk["line_end"],
                    "language": language,
                })
            total_chunks += len(chunks)

        # Batch embed all chunks
        if all_chunks:
            batch_size = self.config.batch_size
            all_embeddings = []

            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                embeddings = self.embedder.embed_batch(batch)
                all_embeddings.extend(embeddings)

            # Store documents
            for meta, embedding, text in zip(chunk_metadata, all_embeddings, all_chunks):
                doc = IndexedDocument(
                    doc_id=meta["doc_id"],
                    path=meta["path"],
                    content=text,
                    embedding=embedding,
                    chunk_index=meta["chunk_index"],
                    line_start=meta["line_start"],
                    line_end=meta["line_end"],
                    language=meta["language"],
                )
                self.documents[doc.doc_id] = doc

            # Rebuild matrix
            self._rebuild_matrix()

        return total_chunks

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        language_filter: Optional[str] = None,
        path_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search for documents similar to query.

        Args:
            query: Search query text
            top_k: Maximum results (default from config)
            min_score: Minimum similarity score (default from config)
            language_filter: Filter by language
            path_filter: Filter by path prefix

        Returns:
            List of search results sorted by relevance
        """
        if not self.documents:
            return []

        top_k = top_k or self.config.max_results
        min_score = min_score if min_score is not None else self.config.min_similarity

        # Generate query embedding
        query_embedding = self.embedder.embed(query)

        # Calculate similarities
        scores = self._compute_similarities(query_embedding)

        # Build results with filtering
        results = []
        for doc_id, score in zip(self._doc_ids, scores):
            if score < min_score:
                continue

            doc = self.documents[doc_id]

            # Apply filters
            if language_filter and doc.language != language_filter:
                continue
            if path_filter and not doc.path.startswith(path_filter):
                continue

            results.append(SearchResult(
                doc_id=doc_id,
                content=doc.content,
                path=doc.path,
                score=score,
                chunk_index=doc.chunk_index,
                line_start=doc.line_start,
                line_end=doc.line_end,
                metadata={
                    "language": doc.language,
                    "content_hash": doc.content_hash,
                }
            ))

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_k]

        # Emit event
        self.bus.emit({
            "topic": "a2a.research.search.semantic",
            "kind": "query",
            "data": {
                "query_length": len(query),
                "results": len(results),
                "top_score": results[0].score if results else 0,
            }
        })

        return results

    def search_by_embedding(
        self,
        embedding: List[float],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search using a pre-computed embedding.

        Args:
            embedding: Query embedding vector
            top_k: Maximum results
            min_score: Minimum similarity score

        Returns:
            List of search results
        """
        if not self.documents:
            return []

        top_k = top_k or self.config.max_results
        min_score = min_score if min_score is not None else self.config.min_similarity

        # Calculate similarities
        scores = self._compute_similarities(embedding)

        # Build results
        results = []
        for doc_id, score in zip(self._doc_ids, scores):
            if score >= min_score:
                doc = self.documents[doc_id]
                results.append(SearchResult(
                    doc_id=doc_id,
                    content=doc.content,
                    path=doc.path,
                    score=score,
                    chunk_index=doc.chunk_index,
                    line_start=doc.line_start,
                    line_end=doc.line_end,
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def find_similar(
        self,
        path: str,
        chunk_index: int = 0,
        top_k: int = 10,
        exclude_same_file: bool = True,
    ) -> List[SearchResult]:
        """
        Find documents similar to an indexed document.

        Args:
            path: File path
            chunk_index: Chunk index within file
            top_k: Maximum results
            exclude_same_file: Exclude results from same file

        Returns:
            List of similar documents
        """
        doc_id = f"{path}::{chunk_index}"
        if doc_id not in self.documents:
            return []

        doc = self.documents[doc_id]
        results = self.search_by_embedding(doc.embedding, top_k=top_k + 10)

        # Filter
        filtered = []
        for r in results:
            if exclude_same_file and r.path == path:
                continue
            filtered.append(r)
            if len(filtered) >= top_k:
                break

        return filtered

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text (for external use)."""
        return self.embedder.embed(text)

    def remove_document(self, path: str) -> int:
        """
        Remove all chunks for a document.

        Args:
            path: File path

        Returns:
            Number of chunks removed
        """
        count = self._remove_path_chunks(path)
        if count > 0:
            self._rebuild_matrix()
        return count

    def clear(self) -> None:
        """Clear all indexed documents."""
        self.documents.clear()
        self._embeddings_matrix = None
        self._doc_ids = []

    def save_index(self, path: Optional[str] = None) -> bool:
        """
        Save index to disk.

        Args:
            path: Index directory path (default from config)

        Returns:
            True if successful
        """
        index_path = Path(path or self.config.index_path)
        index_path.mkdir(parents=True, exist_ok=True)

        try:
            # Save documents
            docs_data = {}
            for doc_id, doc in self.documents.items():
                docs_data[doc_id] = {
                    "doc_id": doc.doc_id,
                    "path": doc.path,
                    "content": doc.content,
                    "embedding": doc.embedding,
                    "chunk_index": doc.chunk_index,
                    "line_start": doc.line_start,
                    "line_end": doc.line_end,
                    "language": doc.language,
                    "indexed_at": doc.indexed_at,
                    "content_hash": doc.content_hash,
                }

            with open(index_path / "documents.json", "w") as f:
                json.dump(docs_data, f)

            # Save config
            config_data = {
                "embedding_model": self.config.embedding_model,
                "embedding_dim": self.config.embedding_dim,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
            }
            with open(index_path / "config.json", "w") as f:
                json.dump(config_data, f)

            return True

        except Exception as e:
            self.bus.emit({
                "topic": "research.search.error",
                "kind": "error",
                "level": "error",
                "data": {"error": str(e), "operation": "save_index"}
            })
            return False

    def stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        paths = set(doc.path for doc in self.documents.values())
        languages = {}
        for doc in self.documents.values():
            languages[doc.language] = languages.get(doc.language, 0) + 1

        return {
            "total_chunks": len(self.documents),
            "total_files": len(paths),
            "embedding_dim": self.embedder.dimension,
            "embedding_model": self.config.embedding_model,
            "by_language": languages,
        }

    def _chunk_content(self, content: str) -> List[Dict[str, Any]]:
        """Split content into overlapping chunks."""
        lines = content.split("\n")
        chunks = []

        current_chunk = []
        current_size = 0
        chunk_start_line = 0

        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > self.config.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "line_start": chunk_start_line,
                    "line_end": i - 1,
                })

                # Keep overlap
                overlap_lines = []
                overlap_size = 0
                for j in range(len(current_chunk) - 1, -1, -1):
                    if overlap_size + len(current_chunk[j]) > self.config.chunk_overlap:
                        break
                    overlap_lines.insert(0, current_chunk[j])
                    overlap_size += len(current_chunk[j]) + 1

                current_chunk = overlap_lines
                current_size = overlap_size
                chunk_start_line = i - len(overlap_lines)

            current_chunk.append(line)
            current_size += line_size

        # Add final chunk
        if current_chunk:
            chunks.append({
                "text": "\n".join(current_chunk),
                "line_start": chunk_start_line,
                "line_end": len(lines) - 1,
            })

        return chunks

    def _remove_path_chunks(self, path: str) -> int:
        """Remove all chunks for a path."""
        to_remove = [
            doc_id for doc_id in self.documents
            if self.documents[doc_id].path == path
        ]
        for doc_id in to_remove:
            del self.documents[doc_id]
        return len(to_remove)

    def _rebuild_matrix(self) -> None:
        """Rebuild the embeddings matrix for efficient search."""
        self._doc_ids = list(self.documents.keys())
        if self._doc_ids:
            self._embeddings_matrix = [
                self.documents[doc_id].embedding for doc_id in self._doc_ids
            ]
        else:
            self._embeddings_matrix = None

    def _compute_similarities(self, query_embedding: List[float]) -> List[float]:
        """Compute cosine similarities between query and all documents."""
        if not self._embeddings_matrix:
            return []

        # Cosine similarity
        scores = []
        query_norm = math.sqrt(sum(x*x for x in query_embedding))

        for doc_embedding in self._embeddings_matrix:
            dot_product = sum(a*b for a, b in zip(query_embedding, doc_embedding))
            doc_norm = math.sqrt(sum(x*x for x in doc_embedding))
            if query_norm > 0 and doc_norm > 0:
                similarity = dot_product / (query_norm * doc_norm)
            else:
                similarity = 0.0
            scores.append(similarity)

        return scores

    def _load_index(self) -> bool:
        """Load index from disk if exists."""
        index_path = Path(self.config.index_path)
        docs_file = index_path / "documents.json"

        if not docs_file.exists():
            return False

        try:
            with open(docs_file) as f:
                docs_data = json.load(f)

            for doc_id, data in docs_data.items():
                self.documents[doc_id] = IndexedDocument(
                    doc_id=data["doc_id"],
                    path=data["path"],
                    content=data["content"],
                    embedding=data["embedding"],
                    chunk_index=data.get("chunk_index", 0),
                    line_start=data.get("line_start"),
                    line_end=data.get("line_end"),
                    language=data.get("language", "unknown"),
                    indexed_at=data.get("indexed_at", 0),
                    content_hash=data.get("content_hash", ""),
                )

            self._rebuild_matrix()
            return True

        except Exception:
            return False


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Semantic Search Engine."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Semantic Search Engine (Step 11)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a file or directory")
    index_parser.add_argument("path", help="File or directory to index")
    index_parser.add_argument("--model", default="mock", help="Embedding model")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    search_parser.add_argument("--model", default="mock", help="Embedding model")
    search_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show index statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = SemanticSearchConfig(embedding_model=getattr(args, "model", "mock"))
    engine = SemanticSearchEngine(config)

    if args.command == "index":
        path = Path(args.path)
        if path.is_file():
            content = path.read_text(errors="ignore")
            chunks = engine.index_document(str(path), content)
            print(f"Indexed {chunks} chunks from {path}")
        elif path.is_dir():
            total = 0
            for py_file in path.rglob("*.py"):
                content = py_file.read_text(errors="ignore")
                chunks = engine.index_document(str(py_file), content, language="python")
                total += chunks
            print(f"Indexed {total} chunks from {path}")
        engine.save_index()

    elif args.command == "search":
        results = engine.search(args.query, top_k=args.top_k)
        if args.json:
            print(json.dumps([r.to_dict() for r in results], indent=2))
        else:
            print(f"Found {len(results)} results for '{args.query}':")
            for r in results:
                print(f"\n  Score: {r.score:.4f}")
                print(f"  Path: {r.path} (lines {r.line_start}-{r.line_end})")
                print(f"  Content: {r.content[:200]}...")

    elif args.command == "stats":
        stats = engine.stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Semantic Search Index Statistics:")
            print(f"  Total Chunks: {stats['total_chunks']}")
            print(f"  Total Files: {stats['total_files']}")
            print(f"  Embedding Dim: {stats['embedding_dim']}")
            print(f"  Model: {stats['embedding_model']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
