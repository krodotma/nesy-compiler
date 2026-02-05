#!/usr/bin/env python3
"""
answer_synthesizer.py - Answer Synthesizer (Step 23)

Generate coherent natural language answers from research results.
Supports multiple output formats and LLM integration.

PBTSO Phase: DISTILL, PLAN

Bus Topics:
- a2a.research.synthesize.start
- a2a.research.synthesize.complete
- research.synthesize.llm

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..bootstrap import AgentBus
from ..search.query_planner import QueryIntent


# ============================================================================
# Configuration
# ============================================================================


class OutputFormat(Enum):
    """Output format for synthesized answers."""
    NATURAL = "natural"       # Natural language prose
    STRUCTURED = "structured" # Structured with headers
    CONCISE = "concise"       # Brief, to the point
    DETAILED = "detailed"     # Comprehensive with examples
    CODE_FOCUSED = "code"     # Code-centric with snippets


class SynthesisStrategy(Enum):
    """Strategy for answer synthesis."""
    TEMPLATE = "template"     # Use predefined templates
    LLM = "llm"               # Use language model
    HYBRID = "hybrid"         # Combine both approaches


@dataclass
class SynthesizerConfig:
    """Configuration for answer synthesizer."""

    strategy: SynthesisStrategy = SynthesisStrategy.TEMPLATE
    default_format: OutputFormat = OutputFormat.STRUCTURED
    max_code_lines: int = 30
    max_results_shown: int = 5
    include_confidence: bool = True
    include_citations: bool = True
    llm_provider: Optional[str] = None  # "openai", "anthropic", etc.
    llm_model: Optional[str] = None
    bus_path: Optional[str] = None

    def __post_init__(self):
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class CodeSnippet:
    """A code snippet in the answer."""

    code: str
    language: str
    path: str
    line_start: int
    line_end: Optional[int] = None
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AnswerSection:
    """A section of the synthesized answer."""

    title: str
    content: str
    snippets: List[CodeSnippet] = field(default_factory=list)
    references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "snippets": [s.to_dict() for s in self.snippets],
            "references": self.references,
        }


@dataclass
class SynthesizedAnswer:
    """A synthesized answer to a research query."""

    query: str
    intent: QueryIntent
    summary: str
    sections: List[AnswerSection]
    confidence: float
    format: OutputFormat
    synthesis_time_ms: float
    result_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "intent": self.intent.value,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
            "confidence": self.confidence,
            "format": self.format.value,
            "synthesis_time_ms": self.synthesis_time_ms,
            "result_count": self.result_count,
        }

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = []

        # Summary
        lines.append(f"## Summary\n")
        lines.append(f"{self.summary}\n")

        # Sections
        for section in self.sections:
            lines.append(f"### {section.title}\n")
            lines.append(f"{section.content}\n")

            # Code snippets
            for snippet in section.snippets:
                lines.append(f"\n**{snippet.path}** (line {snippet.line_start}):\n")
                lines.append(f"```{snippet.language}")
                lines.append(snippet.code)
                lines.append("```\n")
                if snippet.explanation:
                    lines.append(f"*{snippet.explanation}*\n")

            # References
            if section.references:
                lines.append("\n**References:**")
                for ref in section.references:
                    lines.append(f"- {ref}")
                lines.append("")

        # Confidence
        lines.append(f"\n---\n*Confidence: {self.confidence:.0%}*")

        return "\n".join(lines)


# ============================================================================
# Answer Templates
# ============================================================================


INTENT_TEMPLATES = {
    QueryIntent.FIND_SYMBOL: {
        "summary": "Found {count} definition(s) for `{symbol}`.",
        "sections": [
            ("Definition", "The {kind} `{name}` is defined in `{path}` at line {line}."),
            ("Signature", "```{language}\n{signature}\n```"),
            ("Documentation", "{docstring}"),
        ],
    },
    QueryIntent.FIND_USAGE: {
        "summary": "Found {count} usage(s) of `{symbol}` across the codebase.",
        "sections": [
            ("Usages", "The symbol is used in the following locations:"),
            ("Usage List", "- `{path}` line {line}: {context}"),
        ],
    },
    QueryIntent.EXPLAIN_CODE: {
        "summary": "Explanation for the code in question.",
        "sections": [
            ("Overview", "This code {purpose}."),
            ("Key Components", "{components}"),
            ("How It Works", "{explanation}"),
        ],
    },
    QueryIntent.TRACE_DEPENDENCY: {
        "summary": "Dependency trace for `{path}`.",
        "sections": [
            ("Imports", "This file imports from: {imports}"),
            ("Imported By", "This file is imported by: {dependents}"),
        ],
    },
    QueryIntent.SEARCH_CONCEPT: {
        "summary": "Found {count} result(s) related to your query.",
        "sections": [
            ("Relevant Code", "{relevant_description}"),
            ("Key Findings", "{findings}"),
        ],
    },
}


# ============================================================================
# Answer Synthesizer
# ============================================================================


class AnswerSynthesizer:
    """
    Synthesize coherent answers from research results.

    Transforms raw search results into human-readable explanations
    using templates, heuristics, and optionally LLMs.

    PBTSO Phase: DISTILL, PLAN

    Example:
        synthesizer = AnswerSynthesizer()
        answer = synthesizer.synthesize(
            results=ranked_results,
            query="where is UserService defined?",
            intent=QueryIntent.FIND_SYMBOL
        )
        print(answer.to_markdown())
    """

    def __init__(
        self,
        config: Optional[SynthesizerConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the answer synthesizer.

        Args:
            config: Synthesizer configuration
            bus: AgentBus for event emission
        """
        self.config = config or SynthesizerConfig()
        self.bus = bus or AgentBus()

        # Custom template handlers by intent
        self._template_handlers: Dict[QueryIntent, Callable] = {}
        self._register_default_handlers()

    def synthesize(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent,
        format: Optional[OutputFormat] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SynthesizedAnswer:
        """
        Synthesize an answer from research results.

        Args:
            results: Ranked search results
            query: Original query
            intent: Detected query intent
            format: Desired output format
            context: Additional context

        Returns:
            SynthesizedAnswer
        """
        start_time = time.time()
        format = format or self.config.default_format

        self._emit_with_lock({
            "topic": "a2a.research.synthesize.start",
            "kind": "synthesize",
            "data": {
                "query": query,
                "intent": intent.value,
                "result_count": len(results),
            }
        })

        # Choose synthesis strategy
        if self.config.strategy == SynthesisStrategy.LLM and self.config.llm_provider:
            answer = self._synthesize_with_llm(results, query, intent, format, context)
        elif self.config.strategy == SynthesisStrategy.HYBRID:
            answer = self._synthesize_hybrid(results, query, intent, format, context)
        else:
            answer = self._synthesize_with_template(results, query, intent, format, context)

        answer.synthesis_time_ms = (time.time() - start_time) * 1000

        self._emit_with_lock({
            "topic": "a2a.research.synthesize.complete",
            "kind": "synthesize",
            "data": answer.to_dict()
        })

        return answer

    def synthesize_quick(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> str:
        """
        Quick synthesis returning just text.

        Args:
            results: Search results
            query: Original query

        Returns:
            Answer text
        """
        intent = self._detect_intent_from_query(query)
        answer = self.synthesize(results, query, intent, OutputFormat.CONCISE)
        return answer.summary

    def format_result_as_snippet(
        self,
        result: Dict[str, Any],
        max_lines: Optional[int] = None,
    ) -> Optional[CodeSnippet]:
        """
        Format a result as a code snippet.

        Args:
            result: Search result
            max_lines: Maximum lines to include

        Returns:
            CodeSnippet or None
        """
        max_lines = max_lines or self.config.max_code_lines

        path = result.get("path")
        if not path:
            return None

        # Get code content
        content = result.get("content", result.get("context"))
        if not content:
            try:
                full_content = Path(path).read_text(errors="ignore")
                line = result.get("line", 1)
                lines = full_content.split("\n")
                start = max(0, line - 1 - 3)
                end = min(len(lines), line - 1 + max_lines - 3)
                content = "\n".join(lines[start:end])
            except Exception:
                return None

        # Detect language
        language = result.get("language", self._detect_language(path))

        return CodeSnippet(
            code=content,
            language=language,
            path=path,
            line_start=result.get("line_start", result.get("line", 1)),
            line_end=result.get("line_end"),
        )

    def register_template_handler(
        self,
        intent: QueryIntent,
        handler: Callable[[List[Dict], str, OutputFormat, Optional[Dict]], SynthesizedAnswer],
    ) -> None:
        """Register a custom template handler for an intent."""
        self._template_handlers[intent] = handler

    # ========================================================================
    # Synthesis Methods
    # ========================================================================

    def _synthesize_with_template(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent,
        format: OutputFormat,
        context: Optional[Dict[str, Any]],
    ) -> SynthesizedAnswer:
        """Synthesize using templates."""
        # Check for custom handler
        if intent in self._template_handlers:
            return self._template_handlers[intent](results, query, format, context)

        # Use default template synthesis
        sections = []
        snippets = []

        # Extract key information based on intent
        if intent == QueryIntent.FIND_SYMBOL:
            sections, snippets = self._synthesize_find_symbol(results, query)
        elif intent == QueryIntent.FIND_USAGE:
            sections, snippets = self._synthesize_find_usage(results, query)
        elif intent == QueryIntent.EXPLAIN_CODE:
            sections, snippets = self._synthesize_explain(results, query)
        elif intent == QueryIntent.TRACE_DEPENDENCY:
            sections, snippets = self._synthesize_dependencies(results, query)
        else:
            sections, snippets = self._synthesize_generic(results, query)

        # Generate summary
        summary = self._generate_summary(results, query, intent)

        # Calculate confidence
        confidence = self._calculate_confidence(results)

        return SynthesizedAnswer(
            query=query,
            intent=intent,
            summary=summary,
            sections=sections,
            confidence=confidence,
            format=format,
            synthesis_time_ms=0,  # Will be set by caller
            result_count=len(results),
        )

    def _synthesize_with_llm(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent,
        format: OutputFormat,
        context: Optional[Dict[str, Any]],
    ) -> SynthesizedAnswer:
        """Synthesize using LLM."""
        self._emit_with_lock({
            "topic": "research.synthesize.llm",
            "kind": "llm",
            "data": {"provider": self.config.llm_provider}
        })

        # Build prompt
        prompt = self._build_llm_prompt(results, query, intent, format)

        # Call LLM (stub - would need actual integration)
        llm_response = self._call_llm(prompt)

        # Parse response into sections
        sections = self._parse_llm_response(llm_response)

        return SynthesizedAnswer(
            query=query,
            intent=intent,
            summary=llm_response[:200] if llm_response else "No response from LLM.",
            sections=sections,
            confidence=0.8,
            format=format,
            synthesis_time_ms=0,
            result_count=len(results),
            metadata={"llm_used": True, "provider": self.config.llm_provider},
        )

    def _synthesize_hybrid(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent,
        format: OutputFormat,
        context: Optional[Dict[str, Any]],
    ) -> SynthesizedAnswer:
        """Synthesize using both templates and LLM."""
        # First get template-based answer
        template_answer = self._synthesize_with_template(
            results, query, intent, format, context
        )

        # If LLM available and results are complex, enhance with LLM
        if self.config.llm_provider and len(results) > 3:
            try:
                llm_answer = self._synthesize_with_llm(
                    results, query, intent, format, context
                )
                # Merge: use template structure but LLM summary
                template_answer.summary = llm_answer.summary
                template_answer.metadata["hybrid"] = True
            except Exception:
                pass

        return template_answer

    # ========================================================================
    # Intent-Specific Synthesis
    # ========================================================================

    def _synthesize_find_symbol(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> Tuple[List[AnswerSection], List[CodeSnippet]]:
        """Synthesize answer for FIND_SYMBOL intent."""
        sections = []
        snippets = []

        if not results:
            sections.append(AnswerSection(
                title="No Results",
                content="No symbols matching your query were found.",
            ))
            return sections, snippets

        # Primary result
        primary = results[0]
        name = primary.get("name", "Unknown")
        kind = primary.get("kind", "symbol")
        path = primary.get("path", "unknown")
        line = primary.get("line", 1)

        # Definition section
        definition_content = f"The {kind} `{name}` is defined in:\n\n"
        definition_content += f"**File:** `{path}`\n"
        definition_content += f"**Line:** {line}\n"

        if primary.get("signature"):
            definition_content += f"\n**Signature:**\n```\n{primary['signature']}\n```"

        sections.append(AnswerSection(
            title="Definition",
            content=definition_content,
            references=[f"{path}:{line}"],
        ))

        # Add code snippet
        snippet = self.format_result_as_snippet(primary)
        if snippet:
            snippets.append(snippet)
            sections.append(AnswerSection(
                title="Code",
                content="",
                snippets=[snippet],
            ))

        # Documentation section
        if primary.get("docstring"):
            sections.append(AnswerSection(
                title="Documentation",
                content=primary["docstring"],
            ))

        # Other matches
        if len(results) > 1:
            other_content = "Other matches found:\n"
            for r in results[1:self.config.max_results_shown]:
                other_content += f"- `{r.get('name', 'unknown')}` in `{r.get('path', 'unknown')}:{r.get('line', '?')}`\n"
            if len(results) > self.config.max_results_shown:
                other_content += f"\n*...and {len(results) - self.config.max_results_shown} more*"

            sections.append(AnswerSection(
                title="Other Matches",
                content=other_content,
            ))

        return sections, snippets

    def _synthesize_find_usage(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> Tuple[List[AnswerSection], List[CodeSnippet]]:
        """Synthesize answer for FIND_USAGE intent."""
        sections = []
        snippets = []

        if not results:
            sections.append(AnswerSection(
                title="No Usages",
                content="No usages of this symbol were found.",
            ))
            return sections, snippets

        # Group by file
        by_file: Dict[str, List[Dict]] = {}
        for r in results:
            path = r.get("path", "unknown")
            by_file.setdefault(path, []).append(r)

        # Summary
        usage_content = f"Found {len(results)} usage(s) across {len(by_file)} file(s):\n\n"

        for path, usages in list(by_file.items())[:self.config.max_results_shown]:
            usage_content += f"**{path}** ({len(usages)} usage(s))\n"
            for u in usages[:3]:
                line = u.get("line", "?")
                ctx = u.get("context", "")[:60]
                usage_content += f"  - Line {line}: `{ctx}...`\n"
            if len(usages) > 3:
                usage_content += f"  - *...and {len(usages) - 3} more*\n"
            usage_content += "\n"

        sections.append(AnswerSection(
            title="Usages",
            content=usage_content,
            references=[f"{p}:{u[0].get('line', 1)}" for p, u in by_file.items()],
        ))

        return sections, snippets

    def _synthesize_explain(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> Tuple[List[AnswerSection], List[CodeSnippet]]:
        """Synthesize answer for EXPLAIN_CODE intent."""
        sections = []
        snippets = []

        if not results:
            sections.append(AnswerSection(
                title="No Context",
                content="Unable to find relevant code to explain.",
            ))
            return sections, snippets

        # Overview
        primary = results[0]
        kind = primary.get("kind", "code")
        name = primary.get("name", "this")

        overview = f"This {kind} `{name}` "
        if primary.get("docstring"):
            overview += f"is documented as: *{primary['docstring'][:200]}*"
        else:
            overview += "handles specific functionality in the codebase."

        sections.append(AnswerSection(
            title="Overview",
            content=overview,
        ))

        # Code snippet
        snippet = self.format_result_as_snippet(primary)
        if snippet:
            sections.append(AnswerSection(
                title="Code",
                content="",
                snippets=[snippet],
            ))

        # Related code
        if len(results) > 1:
            related = "Related code:\n"
            for r in results[1:4]:
                related += f"- `{r.get('name', 'unknown')}` ({r.get('kind', 'code')}) in `{r.get('path', 'unknown')}`\n"
            sections.append(AnswerSection(
                title="Related",
                content=related,
            ))

        return sections, snippets

    def _synthesize_dependencies(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> Tuple[List[AnswerSection], List[CodeSnippet]]:
        """Synthesize answer for TRACE_DEPENDENCY intent."""
        sections = []

        imports = [r for r in results if r.get("direction") == "imports"]
        imported_by = [r for r in results if r.get("direction") == "imported_by"]

        if imports:
            import_content = "This file imports from:\n"
            for r in imports[:10]:
                import_content += f"- `{r.get('path', 'unknown')}`\n"
            sections.append(AnswerSection(
                title="Imports",
                content=import_content,
            ))

        if imported_by:
            dependent_content = "This file is imported by:\n"
            for r in imported_by[:10]:
                dependent_content += f"- `{r.get('path', 'unknown')}`\n"
            sections.append(AnswerSection(
                title="Dependents",
                content=dependent_content,
            ))

        if not sections:
            sections.append(AnswerSection(
                title="No Dependencies",
                content="No dependency information found.",
            ))

        return sections, []

    def _synthesize_generic(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> Tuple[List[AnswerSection], List[CodeSnippet]]:
        """Synthesize generic answer."""
        sections = []
        snippets = []

        if not results:
            sections.append(AnswerSection(
                title="No Results",
                content="No results found matching your query.",
            ))
            return sections, snippets

        # Results list
        content = f"Found {len(results)} result(s):\n\n"
        for r in results[:self.config.max_results_shown]:
            name = r.get("name", r.get("path", "unknown"))
            kind = r.get("kind", r.get("type", "result"))
            path = r.get("path", "")
            line = r.get("line", "")

            content += f"- **{name}** ({kind})"
            if path:
                content += f" - `{path}`"
                if line:
                    content += f":{line}"
            content += "\n"

        if len(results) > self.config.max_results_shown:
            content += f"\n*...and {len(results) - self.config.max_results_shown} more results*"

        sections.append(AnswerSection(
            title="Results",
            content=content,
        ))

        # Add top result as snippet
        if results:
            snippet = self.format_result_as_snippet(results[0])
            if snippet:
                sections.append(AnswerSection(
                    title="Top Result",
                    content="",
                    snippets=[snippet],
                ))

        return sections, snippets

    # ========================================================================
    # Helpers
    # ========================================================================

    def _register_default_handlers(self) -> None:
        """Register default template handlers."""
        pass  # Using inline methods for default handling

    def _generate_summary(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent,
    ) -> str:
        """Generate summary based on intent and results."""
        count = len(results)

        if intent == QueryIntent.FIND_SYMBOL:
            if count == 0:
                return f"No definition found for the symbol in '{query}'."
            elif count == 1:
                r = results[0]
                return f"Found `{r.get('name', 'symbol')}` ({r.get('kind', 'symbol')}) in `{r.get('path', 'unknown')}`."
            else:
                return f"Found {count} possible definitions matching your query."

        elif intent == QueryIntent.FIND_USAGE:
            if count == 0:
                return "No usages found for this symbol."
            else:
                return f"Found {count} usage(s) across the codebase."

        elif intent == QueryIntent.EXPLAIN_CODE:
            return "Here's an explanation of the requested code."

        elif intent == QueryIntent.TRACE_DEPENDENCY:
            return "Dependency analysis complete."

        else:
            if count == 0:
                return "No relevant results found."
            else:
                return f"Found {count} result(s) related to your query."

    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the answer."""
        if not results:
            return 0.2

        # Average of top result scores
        top_scores = [r.get("score", r.get("final_score", 0.5)) for r in results[:3]]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.5

        # Adjust based on result count
        if len(results) == 1 and avg_score > 0.8:
            return min(0.95, avg_score)
        elif len(results) > 10:
            return min(0.85, avg_score * 0.9)

        return avg_score

    def _detect_intent_from_query(self, query: str) -> QueryIntent:
        """Simple intent detection from query text."""
        query_lower = query.lower()

        if "where is" in query_lower or "find" in query_lower or "definition" in query_lower:
            return QueryIntent.FIND_SYMBOL
        elif "usage" in query_lower or "who uses" in query_lower or "references" in query_lower:
            return QueryIntent.FIND_USAGE
        elif "explain" in query_lower or "how does" in query_lower:
            return QueryIntent.EXPLAIN_CODE
        elif "depends" in query_lower or "import" in query_lower:
            return QueryIntent.TRACE_DEPENDENCY

        return QueryIntent.SEARCH_CONCEPT

    def _detect_language(self, path: str) -> str:
        """Detect programming language from file path."""
        ext = Path(path).suffix.lower()
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
        }
        return lang_map.get(ext, "")

    def _build_llm_prompt(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: QueryIntent,
        format: OutputFormat,
    ) -> str:
        """Build prompt for LLM synthesis."""
        # Format results for prompt
        results_text = ""
        for i, r in enumerate(results[:10]):
            results_text += f"\nResult {i+1}:\n"
            for k, v in r.items():
                if k not in ["embedding", "content_hash"]:
                    results_text += f"  {k}: {str(v)[:200]}\n"

        prompt = f"""Given these search results for the query "{query}":

{results_text}

Generate a {format.value} answer that:
1. Directly addresses the query
2. References specific files and line numbers
3. Includes relevant code snippets
4. Is confident where appropriate

Intent: {intent.value}
"""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call LLM provider (stub implementation)."""
        # This would integrate with actual LLM providers
        return f"[LLM response would be generated here for prompt: {prompt[:100]}...]"

    def _parse_llm_response(self, response: str) -> List[AnswerSection]:
        """Parse LLM response into sections."""
        # Simple parsing - would be more sophisticated with real LLM
        return [AnswerSection(
            title="Answer",
            content=response,
        )]

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
    """CLI entry point for Answer Synthesizer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Answer Synthesizer (Step 23)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Synthesize command
    synth_parser = subparsers.add_parser("synthesize", help="Synthesize answer from results")
    synth_parser.add_argument("--input", "-i", required=True, help="Input JSON file with results")
    synth_parser.add_argument("--query", "-q", required=True, help="Original query")
    synth_parser.add_argument("--intent", choices=[i.value for i in QueryIntent],
                             default="search_concept", help="Query intent")
    synth_parser.add_argument("--format", choices=[f.value for f in OutputFormat],
                             default="structured", help="Output format")
    synth_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Quick command
    quick_parser = subparsers.add_parser("quick", help="Quick synthesis")
    quick_parser.add_argument("--input", "-i", required=True, help="Input JSON file")
    quick_parser.add_argument("--query", "-q", required=True, help="Query")

    args = parser.parse_args()

    synthesizer = AnswerSynthesizer()

    if args.command == "synthesize":
        with open(args.input) as f:
            results = json.load(f)

        intent = QueryIntent(args.intent)
        format = OutputFormat(args.format)

        answer = synthesizer.synthesize(results, args.query, intent, format)

        if args.json:
            print(json.dumps(answer.to_dict(), indent=2))
        else:
            print(answer.to_markdown())

    elif args.command == "quick":
        with open(args.input) as f:
            results = json.load(f)

        text = synthesizer.synthesize_quick(results, args.query)
        print(text)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
