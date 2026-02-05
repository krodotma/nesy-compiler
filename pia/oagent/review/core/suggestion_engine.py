#!/usr/bin/env python3
"""
Review Suggestion Engine (Step 166)

Generates intelligent improvement suggestions based on code analysis,
patterns, and best practices.

PBTSO Phase: DISTILL
Bus Topics: review.suggestion.generate, review.suggestion.apply

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Types
# ============================================================================

class SuggestionType(Enum):
    """Types of suggestions."""
    REFACTOR = "refactor"
    PERFORMANCE = "performance"
    SECURITY = "security"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TEST = "test"
    ERROR_HANDLING = "error_handling"
    NAMING = "naming"
    SIMPLIFICATION = "simplification"
    BEST_PRACTICE = "best_practice"


class SuggestionPriority(Enum):
    """Priority levels for suggestions."""
    CRITICAL = "critical"  # Must fix
    HIGH = "high"          # Should fix
    MEDIUM = "medium"      # Consider fixing
    LOW = "low"            # Nice to have


class ApplicabilityStatus(Enum):
    """Whether a suggestion can be auto-applied."""
    AUTO_APPLICABLE = "auto_applicable"
    MANUAL_REQUIRED = "manual_required"
    REVIEW_REQUIRED = "review_required"


@dataclass
class CodeSnippet:
    """A code snippet with context."""
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str = "python"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Suggestion:
    """
    A code improvement suggestion.

    Attributes:
        suggestion_id: Unique identifier
        suggestion_type: Category of suggestion
        priority: Priority level
        title: Short title
        description: Detailed description
        original: Original code snippet
        suggested: Suggested replacement
        rationale: Why this change is recommended
        applicability: Whether it can be auto-applied
        impact: Expected impact description
        references: Links to documentation/standards
        confidence: Confidence score (0-1)
    """
    suggestion_id: str
    suggestion_type: SuggestionType
    priority: SuggestionPriority
    title: str
    description: str
    original: CodeSnippet
    suggested: Optional[CodeSnippet] = None
    rationale: str = ""
    applicability: ApplicabilityStatus = ApplicabilityStatus.MANUAL_REQUIRED
    impact: str = ""
    references: List[str] = field(default_factory=list)
    confidence: float = 0.8

    def __post_init__(self):
        if not self.suggestion_id:
            self.suggestion_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "suggestion_id": self.suggestion_id,
            "suggestion_type": self.suggestion_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "original": self.original.to_dict(),
            "suggested": self.suggested.to_dict() if self.suggested else None,
            "rationale": self.rationale,
            "applicability": self.applicability.value,
            "impact": self.impact,
            "references": self.references,
            "confidence": self.confidence,
        }
        return result

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"### {self.title}",
            "",
            f"**Type:** {self.suggestion_type.value.replace('_', ' ').title()}",
            f"**Priority:** {self.priority.value.title()}",
            f"**Location:** `{self.original.file_path}:{self.original.start_line}`",
            "",
            self.description,
            "",
        ]

        if self.rationale:
            lines.extend([
                "**Rationale:**",
                self.rationale,
                "",
            ])

        lines.extend([
            "**Original:**",
            f"```{self.original.language}",
            self.original.content,
            "```",
            "",
        ])

        if self.suggested:
            lines.extend([
                "**Suggested:**",
                f"```{self.suggested.language}",
                self.suggested.content,
                "```",
                "",
            ])

        if self.impact:
            lines.extend([f"**Impact:** {self.impact}", ""])

        if self.references:
            lines.extend([
                "**References:**",
                *[f"- {ref}" for ref in self.references],
            ])

        return "\n".join(lines)


@dataclass
class SuggestionBatch:
    """
    A batch of suggestions.

    Attributes:
        batch_id: Unique batch ID
        suggestions: List of suggestions
        files_analyzed: Number of files analyzed
        total_suggestions: Total suggestion count
        by_type: Count by suggestion type
        by_priority: Count by priority
        auto_applicable_count: Number that can be auto-applied
    """
    batch_id: str
    suggestions: List[Suggestion] = field(default_factory=list)
    files_analyzed: int = 0
    total_suggestions: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_priority: Dict[str, int] = field(default_factory=dict)
    auto_applicable_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "files_analyzed": self.files_analyzed,
            "total_suggestions": self.total_suggestions,
            "by_type": self.by_type,
            "by_priority": self.by_priority,
            "auto_applicable_count": self.auto_applicable_count,
        }


# ============================================================================
# Suggestion Rules
# ============================================================================

@dataclass
class SuggestionRule:
    """A rule for generating suggestions."""
    name: str
    pattern: str
    suggestion_type: SuggestionType
    priority: SuggestionPriority
    title: str
    description: str
    replacement: Optional[str] = None
    rationale: str = ""
    applicability: ApplicabilityStatus = ApplicabilityStatus.MANUAL_REQUIRED
    languages: List[str] = field(default_factory=lambda: ["python"])

    def compile(self) -> re.Pattern:
        """Compile the pattern."""
        return re.compile(self.pattern, re.MULTILINE)


# Default suggestion rules
DEFAULT_RULES: List[SuggestionRule] = [
    # Python specific
    SuggestionRule(
        name="use_pathlib",
        pattern=r"os\.path\.(join|exists|isfile|isdir|dirname|basename)\(",
        suggestion_type=SuggestionType.BEST_PRACTICE,
        priority=SuggestionPriority.LOW,
        title="Consider using pathlib",
        description="The os.path module can be replaced with pathlib for more idiomatic Python.",
        rationale="pathlib provides an object-oriented interface to filesystem paths.",
        references=["https://docs.python.org/3/library/pathlib.html"],
    ),
    SuggestionRule(
        name="use_context_manager",
        pattern=r"(\w+)\s*=\s*open\([^)]+\)\n(?!.*\1\.close\(\))",
        suggestion_type=SuggestionType.BEST_PRACTICE,
        priority=SuggestionPriority.MEDIUM,
        title="Use context manager for file operations",
        description="Files should be opened using a context manager (with statement).",
        rationale="Context managers ensure proper resource cleanup even if exceptions occur.",
    ),
    SuggestionRule(
        name="avoid_bare_except",
        pattern=r"except\s*:",
        suggestion_type=SuggestionType.ERROR_HANDLING,
        priority=SuggestionPriority.HIGH,
        title="Avoid bare except clauses",
        description="Bare except clauses catch all exceptions including KeyboardInterrupt and SystemExit.",
        rationale="Always specify the exception type to avoid masking bugs.",
        applicability=ApplicabilityStatus.MANUAL_REQUIRED,
    ),
    SuggestionRule(
        name="use_f_strings",
        pattern=r'"\s*%\s*\(' + r"|" + r"\.format\(",
        suggestion_type=SuggestionType.STYLE,
        priority=SuggestionPriority.LOW,
        title="Consider using f-strings",
        description="f-strings are more readable and efficient than % formatting or .format().",
        rationale="f-strings were introduced in Python 3.6 and are the preferred string formatting method.",
    ),
    SuggestionRule(
        name="simplify_boolean_return",
        pattern=r"if\s+.+:\s*\n\s*return\s+True\s*\n\s*else:\s*\n\s*return\s+False",
        suggestion_type=SuggestionType.SIMPLIFICATION,
        priority=SuggestionPriority.LOW,
        title="Simplify boolean return",
        description="This if/else can be simplified to return the condition directly.",
        rationale="Simplifying code improves readability.",
        applicability=ApplicabilityStatus.REVIEW_REQUIRED,
    ),
    SuggestionRule(
        name="avoid_mutable_default_arg",
        pattern=r"def\s+\w+\([^)]*(?::\s*(?:list|dict|set)\s*)?=\s*(?:\[\]|\{\}|set\(\))",
        suggestion_type=SuggestionType.BEST_PRACTICE,
        priority=SuggestionPriority.HIGH,
        title="Avoid mutable default arguments",
        description="Mutable default arguments are shared between calls and can cause unexpected behavior.",
        rationale="Use None as default and create new instance inside the function.",
    ),
    SuggestionRule(
        name="use_enumerate",
        pattern=r"for\s+\w+\s+in\s+range\(len\(",
        suggestion_type=SuggestionType.STYLE,
        priority=SuggestionPriority.LOW,
        title="Use enumerate instead of range(len())",
        description="enumerate() is more Pythonic when you need both index and value.",
        rationale="enumerate() is clearer and more efficient.",
    ),
    SuggestionRule(
        name="long_function",
        pattern=r"def\s+\w+\([^)]*\):[^\n]*\n(?:[^\n]*\n){50,}",
        suggestion_type=SuggestionType.REFACTOR,
        priority=SuggestionPriority.MEDIUM,
        title="Consider breaking up long function",
        description="This function is quite long. Consider extracting logical sections into separate functions.",
        rationale="Smaller functions are easier to test, understand, and maintain.",
    ),
    # Documentation
    SuggestionRule(
        name="missing_docstring_function",
        pattern=r'def\s+(?!_)\w+\([^)]*\):\s*\n\s+(?!"""|\'\'\')[\w#]',
        suggestion_type=SuggestionType.DOCUMENTATION,
        priority=SuggestionPriority.LOW,
        title="Add docstring to function",
        description="Public functions should have docstrings explaining their purpose.",
        rationale="Docstrings improve code documentation and enable IDE support.",
    ),
    SuggestionRule(
        name="missing_type_hints",
        pattern=r"def\s+\w+\((?:\w+,?\s*)+\)(?!.*->):",
        suggestion_type=SuggestionType.DOCUMENTATION,
        priority=SuggestionPriority.LOW,
        title="Consider adding type hints",
        description="Adding type hints improves code documentation and enables static analysis.",
        rationale="Type hints help catch bugs early and improve IDE support.",
    ),
]


# ============================================================================
# Suggestion Engine
# ============================================================================

class SuggestionEngine:
    """
    Generates code improvement suggestions.

    Analyzes code against rules and patterns to suggest improvements.

    Example:
        engine = SuggestionEngine()

        # Analyze code
        suggestions = engine.analyze_file("/path/to/file.py")

        # Or analyze content directly
        suggestions = engine.analyze_content(code_string, "example.py")

        for s in suggestions:
            print(s.to_markdown())
    """

    BUS_TOPICS = {
        "generate": "review.suggestion.generate",
        "apply": "review.suggestion.apply",
        "feedback": "review.suggestion.feedback",
    }

    def __init__(
        self,
        rules: Optional[List[SuggestionRule]] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the suggestion engine.

        Args:
            rules: Custom suggestion rules (defaults to DEFAULT_RULES)
            bus_path: Path to event bus file
        """
        self.rules = rules or DEFAULT_RULES
        self.bus_path = bus_path or self._get_bus_path()
        self._compiled_rules: List[Tuple[SuggestionRule, re.Pattern]] = []
        self._compile_rules()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _compile_rules(self) -> None:
        """Compile all rules."""
        self._compiled_rules = [
            (rule, rule.compile())
            for rule in self.rules
        ]

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "suggestion") -> str:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "suggestion-engine",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        return event_id

    def _detect_language(self, file_path: str) -> str:
        """Detect language from file path."""
        ext_map = {
            ".py": "python",
            ".pyi": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
        }
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, "unknown")

    def analyze_file(self, file_path: str) -> List[Suggestion]:
        """
        Analyze a file and generate suggestions.

        Args:
            file_path: Path to file

        Returns:
            List of suggestions
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return self.analyze_content(content, file_path)
        except IOError:
            return []

    def analyze_content(
        self,
        content: str,
        file_path: str = "<stdin>",
    ) -> List[Suggestion]:
        """
        Analyze content and generate suggestions.

        Args:
            content: Code content
            file_path: File path for context

        Returns:
            List of suggestions
        """
        language = self._detect_language(file_path)
        suggestions = []
        lines = content.split("\n")

        for rule, pattern in self._compiled_rules:
            # Skip rules not applicable to this language
            if language not in rule.languages:
                continue

            for match in pattern.finditer(content):
                # Calculate line numbers
                start_pos = match.start()
                end_pos = match.end()
                start_line = content[:start_pos].count("\n") + 1
                end_line = content[:end_pos].count("\n") + 1

                # Extract matched content
                matched_content = match.group(0)

                # Create snippet
                original = CodeSnippet(
                    content=matched_content[:500],  # Limit size
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                )

                # Create suggested replacement if available
                suggested = None
                if rule.replacement:
                    try:
                        replaced = pattern.sub(rule.replacement, matched_content)
                        suggested = CodeSnippet(
                            content=replaced[:500],
                            file_path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                            language=language,
                        )
                    except re.error:
                        pass

                suggestions.append(Suggestion(
                    suggestion_id=str(uuid.uuid4())[:8],
                    suggestion_type=rule.suggestion_type,
                    priority=rule.priority,
                    title=rule.title,
                    description=rule.description,
                    original=original,
                    suggested=suggested,
                    rationale=rule.rationale,
                    applicability=rule.applicability,
                    references=getattr(rule, "references", []),
                ))

        return suggestions

    def analyze_files(
        self,
        files: List[str],
    ) -> SuggestionBatch:
        """
        Analyze multiple files.

        Args:
            files: List of file paths

        Returns:
            SuggestionBatch with all suggestions

        Emits:
            review.suggestion.generate
        """
        batch_id = str(uuid.uuid4())[:8]

        self._emit_event(self.BUS_TOPICS["generate"], {
            "batch_id": batch_id,
            "files_count": len(files),
            "status": "started",
        })

        all_suggestions = []
        for file_path in files:
            suggestions = self.analyze_file(file_path)
            all_suggestions.extend(suggestions)

        # Count by type and priority
        by_type: Dict[str, int] = {}
        by_priority: Dict[str, int] = {}
        auto_applicable = 0

        for s in all_suggestions:
            by_type[s.suggestion_type.value] = by_type.get(s.suggestion_type.value, 0) + 1
            by_priority[s.priority.value] = by_priority.get(s.priority.value, 0) + 1
            if s.applicability == ApplicabilityStatus.AUTO_APPLICABLE:
                auto_applicable += 1

        batch = SuggestionBatch(
            batch_id=batch_id,
            suggestions=all_suggestions,
            files_analyzed=len(files),
            total_suggestions=len(all_suggestions),
            by_type=by_type,
            by_priority=by_priority,
            auto_applicable_count=auto_applicable,
        )

        self._emit_event(self.BUS_TOPICS["generate"], {
            "batch_id": batch_id,
            "total_suggestions": len(all_suggestions),
            "by_type": by_type,
            "status": "completed",
        })

        return batch

    def add_rule(self, rule: SuggestionRule) -> None:
        """Add a custom rule."""
        self.rules.append(rule)
        self._compiled_rules.append((rule, rule.compile()))


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Suggestion Engine."""
    import argparse

    parser = argparse.ArgumentParser(description="Review Suggestion Engine (Step 166)")
    parser.add_argument("files", nargs="*", help="Files to analyze")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown")

    args = parser.parse_args()

    engine = SuggestionEngine()

    if args.files:
        batch = engine.analyze_files(args.files)

        if args.json:
            print(json.dumps(batch.to_dict(), indent=2))
        elif args.markdown:
            for s in batch.suggestions:
                print(s.to_markdown())
                print("\n---\n")
        else:
            print(f"Suggestions ({batch.total_suggestions} total)")
            print(f"  Files analyzed: {batch.files_analyzed}")
            print(f"  Auto-applicable: {batch.auto_applicable_count}")
            print("\nBy Type:")
            for t, count in batch.by_type.items():
                print(f"  {t}: {count}")
            print("\nBy Priority:")
            for p, count in batch.by_priority.items():
                print(f"  {p}: {count}")

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
