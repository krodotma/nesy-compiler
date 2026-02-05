#!/usr/bin/env python3
"""
citation_generator.py - Citation Generator (Step 24)

Track and format citations for research results.
Supports multiple citation formats and reference management.

PBTSO Phase: DISTILL

Bus Topics:
- a2a.research.cite.generate
- a2a.research.cite.format
- research.cite.added

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class CitationFormat(Enum):
    """Supported citation formats."""
    INLINE = "inline"         # [path:line] inline
    FOOTNOTE = "footnote"     # Numbered footnotes
    MARKDOWN = "markdown"     # Markdown links
    REFERENCE = "reference"   # Reference list at end
    HTML = "html"             # HTML anchor links
    GITHUB = "github"         # GitHub permalink format


class CitationStyle(Enum):
    """Citation display styles."""
    FULL_PATH = "full_path"   # Full file path
    SHORT_PATH = "short_path" # Just filename
    RELATIVE = "relative"     # Relative to project root
    SYMBOL = "symbol"         # Symbol-based (function/class name)


@dataclass
class CitationConfig:
    """Configuration for citation generator."""

    default_format: CitationFormat = CitationFormat.MARKDOWN
    default_style: CitationStyle = CitationStyle.RELATIVE
    include_line_numbers: bool = True
    include_context: bool = False
    max_context_chars: int = 80
    project_root: Optional[str] = None
    github_repo: Optional[str] = None  # For GitHub permalinks
    github_branch: str = "main"
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.project_root is None:
            self.project_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        if self.bus_path is None:
            self.bus_path = f"{self.project_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Citation:
    """A citation to source code or documentation."""

    id: str                           # Unique citation ID
    path: str                         # File path
    line_start: int                   # Starting line number
    line_end: Optional[int] = None    # Ending line number
    symbol: Optional[str] = None      # Symbol name (function, class)
    context: Optional[str] = None     # Surrounding code context
    commit_hash: Optional[str] = None # Git commit for versioning
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            # Generate ID from path and line
            content = f"{self.path}:{self.line_start}"
            self.id = hashlib.sha256(content.encode()).hexdigest()[:8]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "symbol": self.symbol,
            "context": self.context,
            "commit_hash": self.commit_hash,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_result(cls, result: Dict[str, Any]) -> "Citation":
        """Create Citation from a search result."""
        return cls(
            id="",  # Will be generated
            path=result.get("path", ""),
            line_start=result.get("line", result.get("line_start", 1)),
            line_end=result.get("line_end"),
            symbol=result.get("name", result.get("symbol")),
            context=result.get("context", result.get("content", ""))[:100],
        )


@dataclass
class CitationIndex:
    """Index of all citations in a document."""

    citations: Dict[str, Citation] = field(default_factory=dict)
    usage_count: Dict[str, int] = field(default_factory=dict)
    order: List[str] = field(default_factory=list)  # Order of first appearance

    def add(self, citation: Citation) -> str:
        """Add a citation, returns citation ID."""
        if citation.id not in self.citations:
            self.citations[citation.id] = citation
            self.order.append(citation.id)
            self.usage_count[citation.id] = 0

        self.usage_count[citation.id] += 1
        return citation.id

    def get(self, citation_id: str) -> Optional[Citation]:
        """Get citation by ID."""
        return self.citations.get(citation_id)

    def all(self) -> List[Citation]:
        """Get all citations in order."""
        return [self.citations[cid] for cid in self.order if cid in self.citations]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "citations": {k: v.to_dict() for k, v in self.citations.items()},
            "usage_count": self.usage_count,
            "order": self.order,
        }


@dataclass
class FormattedCitation:
    """A formatted citation ready for output."""

    citation_id: str
    inline_text: str      # Text for inline citation
    reference_text: str   # Text for reference list
    link: Optional[str]   # Clickable link (if applicable)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Citation Generator
# ============================================================================


class CitationGenerator:
    """
    Generate and format citations for research results.

    Tracks all referenced sources and formats them consistently
    for use in answers and documentation.

    PBTSO Phase: DISTILL

    Example:
        generator = CitationGenerator()

        # Add citations from results
        for result in search_results:
            cid = generator.cite(result)
            print(f"Result cited as [{cid}]")

        # Format all citations
        references = generator.format_references()
        print(references)
    """

    def __init__(
        self,
        config: Optional[CitationConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the citation generator.

        Args:
            config: Citation configuration
            bus: AgentBus for event emission
        """
        self.config = config or CitationConfig()
        self.bus = bus or AgentBus()

        self._index = CitationIndex()
        self._formatted_cache: Dict[str, FormattedCitation] = {}

    def cite(
        self,
        source: Any,
        symbol: Optional[str] = None,
    ) -> str:
        """
        Create a citation from a source.

        Args:
            source: Result dict, Citation, or path string
            symbol: Optional symbol name to include

        Returns:
            Citation ID
        """
        # Convert source to Citation
        if isinstance(source, Citation):
            citation = source
        elif isinstance(source, dict):
            citation = Citation.from_result(source)
        elif isinstance(source, str):
            # Assume it's a path
            citation = Citation(id="", path=source, line_start=1)
        else:
            raise ValueError(f"Cannot create citation from {type(source)}")

        if symbol:
            citation.symbol = symbol

        # Add context if configured and not present
        if self.config.include_context and not citation.context:
            citation.context = self._extract_context(citation)

        # Add to index
        citation_id = self._index.add(citation)

        self._emit_with_lock({
            "topic": "research.cite.added",
            "kind": "citation",
            "data": citation.to_dict()
        })

        return citation_id

    def cite_multiple(self, sources: List[Any]) -> List[str]:
        """
        Create citations from multiple sources.

        Args:
            sources: List of result dicts, Citations, or paths

        Returns:
            List of citation IDs
        """
        return [self.cite(source) for source in sources]

    def format_inline(
        self,
        citation_id: str,
        format: Optional[CitationFormat] = None,
        style: Optional[CitationStyle] = None,
    ) -> str:
        """
        Format a citation for inline use.

        Args:
            citation_id: Citation ID
            format: Output format
            style: Display style

        Returns:
            Formatted citation string
        """
        format = format or self.config.default_format
        style = style or self.config.default_style

        citation = self._index.get(citation_id)
        if not citation:
            return f"[{citation_id}]"

        formatted = self._format_citation(citation, format, style)
        return formatted.inline_text

    def format_references(
        self,
        format: Optional[CitationFormat] = None,
        style: Optional[CitationStyle] = None,
        title: str = "References",
    ) -> str:
        """
        Format all citations as a reference list.

        Args:
            format: Output format
            style: Display style
            title: Section title

        Returns:
            Formatted reference list
        """
        format = format or self.config.default_format
        style = style or self.config.default_style

        citations = self._index.all()
        if not citations:
            return ""

        self._emit_with_lock({
            "topic": "a2a.research.cite.format",
            "kind": "citation",
            "data": {"count": len(citations), "format": format.value}
        })

        lines = [f"## {title}\n"] if format == CitationFormat.MARKDOWN else [f"{title}:\n"]

        for i, citation in enumerate(citations, 1):
            formatted = self._format_citation(citation, format, style)

            if format == CitationFormat.FOOTNOTE:
                lines.append(f"[{i}]: {formatted.reference_text}")
            elif format == CitationFormat.MARKDOWN:
                lines.append(f"- [{citation.id}]: {formatted.reference_text}")
            elif format == CitationFormat.REFERENCE:
                lines.append(f"{i}. {formatted.reference_text}")
            elif format == CitationFormat.HTML:
                link = formatted.link or "#"
                lines.append(f'<li id="cite-{citation.id}"><a href="{link}">{formatted.reference_text}</a></li>')
            else:
                lines.append(f"  {formatted.reference_text}")

        return "\n".join(lines)

    def format_footnotes(self) -> str:
        """Format citations as footnotes (numbered)."""
        return self.format_references(CitationFormat.FOOTNOTE)

    def get_citation(self, citation_id: str) -> Optional[Citation]:
        """Get a citation by ID."""
        return self._index.get(citation_id)

    def get_all_citations(self) -> List[Citation]:
        """Get all citations in order."""
        return self._index.all()

    def get_github_link(self, citation: Citation) -> Optional[str]:
        """
        Generate GitHub permalink for a citation.

        Args:
            citation: Citation to link

        Returns:
            GitHub URL or None
        """
        if not self.config.github_repo:
            return None

        # Make path relative to project root
        try:
            rel_path = Path(citation.path).relative_to(self.config.project_root)
        except ValueError:
            rel_path = Path(citation.path)

        base_url = f"https://github.com/{self.config.github_repo}/blob/{self.config.github_branch}"
        url = f"{base_url}/{rel_path}"

        if citation.line_start:
            url += f"#L{citation.line_start}"
            if citation.line_end and citation.line_end != citation.line_start:
                url += f"-L{citation.line_end}"

        return url

    def clear(self) -> None:
        """Clear all citations."""
        self._index = CitationIndex()
        self._formatted_cache.clear()

    def export_bibtex(self) -> str:
        """Export citations in BibTeX format (for documentation)."""
        entries = []

        for citation in self._index.all():
            key = f"code:{citation.id}"
            entry = f"""@misc{{{key},
  title = {{{citation.symbol or citation.path}}},
  note = {{File: {citation.path}, Line: {citation.line_start}}},
  howpublished = {{Source code}}
}}"""
            entries.append(entry)

        return "\n\n".join(entries)

    def export_json(self) -> str:
        """Export citations as JSON."""
        return json.dumps(self._index.to_dict(), indent=2)

    def stats(self) -> Dict[str, Any]:
        """Get citation statistics."""
        citations = self._index.all()
        paths = set(c.path for c in citations)

        return {
            "total_citations": len(citations),
            "unique_files": len(paths),
            "most_cited": max(
                self._index.usage_count.items(),
                key=lambda x: x[1],
                default=("none", 0)
            ),
        }

    # ========================================================================
    # Formatting
    # ========================================================================

    def _format_citation(
        self,
        citation: Citation,
        format: CitationFormat,
        style: CitationStyle,
    ) -> FormattedCitation:
        """Format a single citation."""
        cache_key = f"{citation.id}:{format.value}:{style.value}"
        if cache_key in self._formatted_cache:
            return self._formatted_cache[cache_key]

        # Get display path
        display_path = self._get_display_path(citation.path, style)

        # Build inline text
        if format == CitationFormat.INLINE:
            if self.config.include_line_numbers:
                inline = f"[{display_path}:{citation.line_start}]"
            else:
                inline = f"[{display_path}]"

        elif format == CitationFormat.FOOTNOTE:
            # Footnote number will be assigned later
            inline = f"[^{citation.id}]"

        elif format == CitationFormat.MARKDOWN:
            link = self.get_github_link(citation)
            if link:
                inline = f"[{display_path}:{citation.line_start}]({link})"
            else:
                inline = f"`{display_path}:{citation.line_start}`"

        elif format == CitationFormat.GITHUB:
            link = self.get_github_link(citation)
            inline = link if link else f"{display_path}:{citation.line_start}"

        elif format == CitationFormat.HTML:
            inline = f'<a href="#cite-{citation.id}">[{citation.id}]</a>'

        else:  # REFERENCE
            inline = f"[{citation.id}]"

        # Build reference text
        ref_parts = [display_path]
        if citation.line_start:
            ref_parts.append(f"line {citation.line_start}")
            if citation.line_end and citation.line_end != citation.line_start:
                ref_parts[-1] = f"lines {citation.line_start}-{citation.line_end}"

        if citation.symbol:
            ref_parts.append(f"({citation.symbol})")

        reference = ", ".join(ref_parts)

        # Get link
        link = self.get_github_link(citation)

        formatted = FormattedCitation(
            citation_id=citation.id,
            inline_text=inline,
            reference_text=reference,
            link=link,
        )

        self._formatted_cache[cache_key] = formatted
        return formatted

    def _get_display_path(self, path: str, style: CitationStyle) -> str:
        """Get display path based on style."""
        if style == CitationStyle.FULL_PATH:
            return path

        elif style == CitationStyle.SHORT_PATH:
            return Path(path).name

        elif style == CitationStyle.RELATIVE:
            try:
                return str(Path(path).relative_to(self.config.project_root))
            except ValueError:
                return path

        elif style == CitationStyle.SYMBOL:
            # Would need symbol info
            return Path(path).name

        return path

    def _extract_context(self, citation: Citation) -> str:
        """Extract code context for a citation."""
        try:
            content = Path(citation.path).read_text(errors="ignore")
            lines = content.split("\n")

            if citation.line_start > 0 and citation.line_start <= len(lines):
                line = lines[citation.line_start - 1]
                return line.strip()[:self.config.max_context_chars]

        except Exception:
            pass

        return ""

    def _emit_with_lock(self, event: Dict[str, Any]) -> str:
        """Emit event with file locking."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        import socket
        from datetime import datetime, timezone
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
    """CLI entry point for Citation Generator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Citation Generator (Step 24)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate citations from results")
    gen_parser.add_argument("--input", "-i", required=True, help="Input JSON file with results")
    gen_parser.add_argument("--format", choices=[f.value for f in CitationFormat],
                          default="markdown", help="Citation format")
    gen_parser.add_argument("--style", choices=[s.value for s in CitationStyle],
                          default="relative", help="Path style")
    gen_parser.add_argument("--github", help="GitHub repo (owner/repo)")
    gen_parser.add_argument("--output", "-o", help="Output file")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export citations")
    export_parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    export_parser.add_argument("--format", choices=["json", "bibtex"],
                              default="json", help="Export format")

    args = parser.parse_args()

    config = CitationConfig(
        github_repo=getattr(args, "github", None),
    )
    generator = CitationGenerator(config)

    if args.command == "generate":
        with open(args.input) as f:
            results = json.load(f)

        # Generate citations
        for result in results:
            generator.cite(result)

        # Format output
        format = CitationFormat(args.format)
        style = CitationStyle(args.style)

        output = generator.format_references(format, style)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Citations written to {args.output}")
        else:
            print(output)

    elif args.command == "export":
        with open(args.input) as f:
            results = json.load(f)

        for result in results:
            generator.cite(result)

        if args.format == "json":
            print(generator.export_json())
        elif args.format == "bibtex":
            print(generator.export_bibtex())

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
