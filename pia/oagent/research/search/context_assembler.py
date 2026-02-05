#!/usr/bin/env python3
"""
context_assembler.py - Context Assembler (Step 12)

Intelligent context window packing for LLM queries.
Optimizes relevance, deduplication, and token budgets.

PBTSO Phase: PLAN, ITERATE

Bus Topics:
- a2a.context.request
- a2a.context.assembled
- research.context.optimized

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class ContextPriority(Enum):
    """Priority levels for context chunks."""
    CRITICAL = 1  # Must include (e.g., user query, error messages)
    HIGH = 2      # Very important (e.g., directly referenced code)
    MEDIUM = 3    # Useful context (e.g., related functions)
    LOW = 4       # Nice to have (e.g., documentation)
    FILLER = 5    # Fill remaining space (e.g., examples)


@dataclass
class ContextAssemblerConfig:
    """Configuration for context assembler."""

    max_tokens: int = 100000
    reserved_tokens: int = 4000  # Reserve for response
    dedup_threshold: float = 0.85  # Similarity threshold for deduplication
    min_chunk_tokens: int = 10
    include_line_numbers: bool = True
    include_file_headers: bool = True
    truncate_long_lines: int = 200  # Max chars per line
    encoding: str = "cl100k_base"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class ContextChunk:
    """A chunk of content for context assembly."""

    content: str
    source: str  # File path or source identifier
    priority: ContextPriority = ContextPriority.MEDIUM
    relevance: float = 0.5  # 0.0 to 1.0
    tokens: int = 0
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    language: str = "unknown"
    chunk_type: str = "code"  # code, doc, error, query, system
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        if self.tokens == 0:
            # Rough estimate: ~4 chars per token
            self.tokens = max(1, len(self.content) // 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "source": self.source,
            "priority": self.priority.name,
            "relevance": self.relevance,
            "tokens": self.tokens,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "language": self.language,
            "chunk_type": self.chunk_type,
        }


@dataclass
class AssembledContext:
    """Result of context assembly."""

    chunks: List[ContextChunk]
    total_tokens: int
    formatted_content: str
    sources: List[str]
    assembly_time: float
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "num_chunks": len(self.chunks),
            "sources": self.sources,
            "assembly_time": self.assembly_time,
            "stats": self.stats,
        }


# ============================================================================
# Token Counter
# ============================================================================


class TokenCounter:
    """Counts tokens using tiktoken or fallback."""

    def __init__(self, encoding: str = "cl100k_base"):
        self.encoding_name = encoding
        self._encoder = None
        self._use_tiktoken = True

    def _ensure_encoder(self):
        """Lazy load encoder."""
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding(self.encoding_name)
            except ImportError:
                self._use_tiktoken = False

    def count(self, text: str) -> int:
        """Count tokens in text."""
        self._ensure_encoder()

        if self._use_tiktoken and self._encoder:
            return len(self._encoder.encode(text))
        else:
            # Fallback: approximate 4 chars per token
            return max(1, len(text) // 4)

    def count_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for multiple texts."""
        return [self.count(text) for text in texts]


# ============================================================================
# Context Assembler
# ============================================================================


class ContextAssembler:
    """
    Intelligent context window packer for LLM queries.

    Assembles context from multiple sources while:
    - Respecting token budgets
    - Prioritizing by relevance and importance
    - Deduplicating similar content
    - Formatting for readability

    PBTSO Phase: PLAN, ITERATE

    Example:
        assembler = ContextAssembler()
        assembler.add_chunk(code_content, "src/main.py", priority=ContextPriority.HIGH)
        assembler.add_chunk(docs, "README.md", priority=ContextPriority.LOW)
        context = assembler.assemble()
    """

    def __init__(
        self,
        config: Optional[ContextAssemblerConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the context assembler.

        Args:
            config: Assembly configuration
            bus: AgentBus for event emission
        """
        self.config = config or ContextAssemblerConfig()
        self.bus = bus or AgentBus()
        self.token_counter = TokenCounter(self.config.encoding)

        self.chunks: List[ContextChunk] = []
        self._content_hashes: Set[str] = set()

    def add_chunk(
        self,
        content: str,
        source: str,
        priority: ContextPriority = ContextPriority.MEDIUM,
        relevance: float = 0.5,
        line_start: Optional[int] = None,
        line_end: Optional[int] = None,
        language: str = "unknown",
        chunk_type: str = "code",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a chunk to the context pool.

        Args:
            content: Chunk content
            source: Source file or identifier
            priority: Priority level
            relevance: Relevance score (0-1)
            line_start: Starting line number
            line_end: Ending line number
            language: Programming language
            chunk_type: Type of content
            metadata: Additional metadata

        Returns:
            True if chunk was added (not a duplicate)
        """
        # Preprocess content
        content = self._preprocess_content(content)

        if not content or len(content.strip()) < 10:
            return False

        # Count tokens
        tokens = self.token_counter.count(content)

        if tokens < self.config.min_chunk_tokens:
            return False

        # Create chunk
        chunk = ContextChunk(
            content=content,
            source=source,
            priority=priority,
            relevance=relevance,
            tokens=tokens,
            line_start=line_start,
            line_end=line_end,
            language=language,
            chunk_type=chunk_type,
            metadata=metadata or {},
        )

        # Check for duplicates
        if chunk.content_hash in self._content_hashes:
            return False

        # Check for near-duplicates
        if self._is_near_duplicate(content):
            return False

        self.chunks.append(chunk)
        self._content_hashes.add(chunk.content_hash)

        return True

    def add_file(
        self,
        path: str,
        content: Optional[str] = None,
        priority: ContextPriority = ContextPriority.MEDIUM,
        relevance: float = 0.5,
        max_lines: Optional[int] = None,
    ) -> bool:
        """
        Add a file to the context pool.

        Args:
            path: File path
            content: File content (read from disk if not provided)
            priority: Priority level
            relevance: Relevance score
            max_lines: Maximum lines to include

        Returns:
            True if file was added
        """
        if content is None:
            try:
                content = Path(path).read_text(errors="ignore")
            except Exception:
                return False

        lines = content.split("\n")
        if max_lines and len(lines) > max_lines:
            content = "\n".join(lines[:max_lines])
            content += f"\n... ({len(lines) - max_lines} more lines)"

        # Detect language
        language = self._detect_language(path)

        return self.add_chunk(
            content=content,
            source=path,
            priority=priority,
            relevance=relevance,
            line_start=1,
            line_end=len(lines),
            language=language,
            chunk_type="code" if language != "unknown" else "doc",
        )

    def add_search_results(
        self,
        results: List[Dict[str, Any]],
        base_priority: ContextPriority = ContextPriority.MEDIUM,
    ) -> int:
        """
        Add search results to context pool.

        Args:
            results: List of search result dicts with content, path, score keys
            base_priority: Base priority for results

        Returns:
            Number of results added
        """
        count = 0
        for result in results:
            # Scale priority by position
            priority = base_priority

            added = self.add_chunk(
                content=result.get("content", ""),
                source=result.get("path", "unknown"),
                priority=priority,
                relevance=result.get("score", 0.5),
                line_start=result.get("line_start"),
                line_end=result.get("line_end"),
                language=result.get("language", "unknown"),
                chunk_type="code",
            )
            if added:
                count += 1

        return count

    def assemble(
        self,
        max_tokens: Optional[int] = None,
        format_style: str = "markdown",
    ) -> AssembledContext:
        """
        Assemble context from added chunks.

        Args:
            max_tokens: Token budget (default from config)
            format_style: Output format (markdown, plain, xml)

        Returns:
            AssembledContext with optimized content
        """
        start_time = time.time()

        max_tokens = max_tokens or (self.config.max_tokens - self.config.reserved_tokens)

        # Sort chunks by priority and relevance
        sorted_chunks = self._rank_chunks()

        # Select chunks within budget
        selected_chunks = []
        total_tokens = 0

        for chunk in sorted_chunks:
            # Account for formatting overhead
            overhead = self._estimate_format_overhead(chunk, format_style)
            chunk_total = chunk.tokens + overhead

            if total_tokens + chunk_total <= max_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_total
            elif chunk.priority == ContextPriority.CRITICAL:
                # Always include critical chunks, truncate if needed
                truncated = self._truncate_chunk(chunk, max_tokens - total_tokens - overhead)
                if truncated:
                    selected_chunks.append(truncated)
                    total_tokens += truncated.tokens + overhead

        # Format output
        formatted = self._format_context(selected_chunks, format_style)
        sources = list(set(c.source for c in selected_chunks))

        # Calculate stats
        stats = {
            "total_chunks_available": len(self.chunks),
            "chunks_selected": len(selected_chunks),
            "chunks_by_priority": {
                p.name: sum(1 for c in selected_chunks if c.priority == p)
                for p in ContextPriority
            },
            "avg_relevance": (
                sum(c.relevance for c in selected_chunks) / len(selected_chunks)
                if selected_chunks else 0
            ),
        }

        assembly_time = time.time() - start_time

        result = AssembledContext(
            chunks=selected_chunks,
            total_tokens=total_tokens,
            formatted_content=formatted,
            sources=sources,
            assembly_time=assembly_time,
            stats=stats,
        )

        # Emit event
        self.bus.emit({
            "topic": "a2a.context.assembled",
            "kind": "context",
            "data": result.to_dict(),
        })

        return result

    def get_optimized_context(self) -> str:
        """Convenience method to get just the formatted context string."""
        return self.assemble().formatted_content

    def clear(self) -> None:
        """Clear all chunks."""
        self.chunks.clear()
        self._content_hashes.clear()

    def stats(self) -> Dict[str, Any]:
        """Get current stats."""
        total_tokens = sum(c.tokens for c in self.chunks)
        return {
            "num_chunks": len(self.chunks),
            "total_tokens": total_tokens,
            "by_priority": {
                p.name: sum(1 for c in self.chunks if c.priority == p)
                for p in ContextPriority
            },
            "by_type": self._count_by_field("chunk_type"),
            "by_language": self._count_by_field("language"),
        }

    def _rank_chunks(self) -> List[ContextChunk]:
        """Rank chunks by priority and relevance."""
        def score(chunk: ContextChunk) -> float:
            # Lower priority value = higher priority
            priority_score = (6 - chunk.priority.value) * 10
            relevance_score = chunk.relevance * 5
            # Slight preference for code over docs
            type_score = 1 if chunk.chunk_type == "code" else 0
            return priority_score + relevance_score + type_score

        return sorted(self.chunks, key=score, reverse=True)

    def _is_near_duplicate(self, content: str) -> bool:
        """Check if content is similar to existing chunks."""
        if not self.chunks:
            return False

        # Simple word overlap check
        content_words = set(content.lower().split())

        for chunk in self.chunks:
            chunk_words = set(chunk.content.lower().split())

            if not content_words or not chunk_words:
                continue

            overlap = len(content_words & chunk_words)
            union = len(content_words | chunk_words)

            if union > 0:
                jaccard = overlap / union
                if jaccard >= self.config.dedup_threshold:
                    return True

        return False

    def _preprocess_content(self, content: str) -> str:
        """Preprocess content for context."""
        # Truncate very long lines
        if self.config.truncate_long_lines:
            lines = content.split("\n")
            processed_lines = []
            for line in lines:
                if len(line) > self.config.truncate_long_lines:
                    line = line[:self.config.truncate_long_lines] + "..."
                processed_lines.append(line)
            content = "\n".join(processed_lines)

        # Remove excessive whitespace
        content = re.sub(r'\n{4,}', '\n\n\n', content)

        return content.strip()

    def _estimate_format_overhead(self, chunk: ContextChunk, style: str) -> int:
        """Estimate tokens for formatting overhead."""
        if style == "plain":
            return 2  # Just newlines

        # For markdown/xml, estimate header/footer
        header_estimate = len(chunk.source) // 4 + 10
        return header_estimate

    def _truncate_chunk(self, chunk: ContextChunk, max_tokens: int) -> Optional[ContextChunk]:
        """Truncate a chunk to fit token budget."""
        if max_tokens <= 0:
            return None

        # Estimate chars from tokens
        max_chars = max_tokens * 4

        if len(chunk.content) <= max_chars:
            return chunk

        truncated_content = chunk.content[:max_chars] + "\n... (truncated)"

        return ContextChunk(
            content=truncated_content,
            source=chunk.source,
            priority=chunk.priority,
            relevance=chunk.relevance,
            tokens=self.token_counter.count(truncated_content),
            line_start=chunk.line_start,
            line_end=None,  # Unknown after truncation
            language=chunk.language,
            chunk_type=chunk.chunk_type,
            metadata=chunk.metadata,
        )

    def _format_context(self, chunks: List[ContextChunk], style: str) -> str:
        """Format chunks into final context string."""
        if style == "plain":
            return self._format_plain(chunks)
        elif style == "xml":
            return self._format_xml(chunks)
        else:  # markdown
            return self._format_markdown(chunks)

    def _format_markdown(self, chunks: List[ContextChunk]) -> str:
        """Format as markdown."""
        parts = []

        for chunk in chunks:
            if self.config.include_file_headers:
                header = f"## {chunk.source}"
                if chunk.line_start is not None:
                    header += f" (lines {chunk.line_start}"
                    if chunk.line_end is not None:
                        header += f"-{chunk.line_end}"
                    header += ")"
                parts.append(header)

            # Add code fence for code chunks
            if chunk.chunk_type == "code" and chunk.language != "unknown":
                parts.append(f"```{chunk.language}")
                parts.append(chunk.content)
                parts.append("```")
            else:
                parts.append(chunk.content)

            parts.append("")  # Blank line between chunks

        return "\n".join(parts)

    def _format_xml(self, chunks: List[ContextChunk]) -> str:
        """Format as XML."""
        parts = ["<context>"]

        for chunk in chunks:
            attrs = f'source="{chunk.source}"'
            if chunk.language != "unknown":
                attrs += f' language="{chunk.language}"'
            if chunk.line_start is not None:
                attrs += f' lines="{chunk.line_start}-{chunk.line_end or "?"}"'

            parts.append(f"  <chunk {attrs}>")
            # Escape XML special chars
            content = chunk.content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            parts.append(f"    {content}")
            parts.append("  </chunk>")

        parts.append("</context>")
        return "\n".join(parts)

    def _format_plain(self, chunks: List[ContextChunk]) -> str:
        """Format as plain text."""
        parts = []

        for chunk in chunks:
            if self.config.include_file_headers:
                parts.append(f"=== {chunk.source} ===")
            parts.append(chunk.content)
            parts.append("")

        return "\n".join(parts)

    def _detect_language(self, path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(path).suffix.lower()
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
        }
        return lang_map.get(ext, "unknown")

    def _count_by_field(self, field: str) -> Dict[str, int]:
        """Count chunks by a field value."""
        counts: Dict[str, int] = {}
        for chunk in self.chunks:
            value = getattr(chunk, field, "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Context Assembler."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Context Assembler (Step 12)"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Files to include in context"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum tokens in output"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "xml", "plain"],
        default="markdown",
        help="Output format"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics"
    )

    args = parser.parse_args()

    assembler = ContextAssembler()

    for i, file_path in enumerate(args.files):
        # Higher priority for earlier files
        priority = ContextPriority.HIGH if i == 0 else ContextPriority.MEDIUM
        relevance = 1.0 - (i * 0.1)  # Decrease relevance for later files

        added = assembler.add_file(
            file_path,
            priority=priority,
            relevance=max(0.1, relevance),
        )
        if added:
            print(f"Added: {file_path}", file=__import__("sys").stderr)
        else:
            print(f"Skipped: {file_path}", file=__import__("sys").stderr)

    if args.stats:
        stats = assembler.stats()
        print(json.dumps(stats, indent=2), file=__import__("sys").stderr)

    context = assembler.assemble(
        max_tokens=args.max_tokens,
        format_style=args.format,
    )

    print(context.formatted_content)
    print(f"\n--- Total tokens: {context.total_tokens} ---", file=__import__("sys").stderr)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
