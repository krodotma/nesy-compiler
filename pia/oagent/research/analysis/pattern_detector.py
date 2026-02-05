#!/usr/bin/env python3
"""
pattern_detector.py - Pattern Detector (Step 15)

Code pattern recognition for design patterns, anti-patterns, and idioms.
Detects common patterns and provides insights for code understanding.

PBTSO Phase: RESEARCH, DISTILL

Bus Topics:
- a2a.research.pattern.detect
- research.pattern.found
- research.pattern.antipattern

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from ..bootstrap import AgentBus
from ..parsers.base import ParseResult


# ============================================================================
# Data Models
# ============================================================================


class PatternCategory(Enum):
    """Category of patterns."""
    DESIGN_PATTERN = "design_pattern"     # GoF patterns
    IDIOM = "idiom"                        # Language-specific idioms
    ANTI_PATTERN = "anti_pattern"         # Bad practices
    ARCHITECTURAL = "architectural"        # Architecture patterns
    TESTING = "testing"                    # Testing patterns
    CONCURRENCY = "concurrency"            # Concurrency patterns


class PatternConfidence(Enum):
    """Confidence level of pattern detection."""
    HIGH = "high"       # Strong evidence
    MEDIUM = "medium"   # Likely match
    LOW = "low"         # Possible match


@dataclass
class Pattern:
    """Definition of a code pattern."""

    name: str
    category: PatternCategory
    description: str
    indicators: List[str]  # What to look for
    examples: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "indicators": self.indicators,
        }


@dataclass
class PatternMatch:
    """A detected pattern instance."""

    pattern: Pattern
    path: str
    line_start: int
    line_end: Optional[int] = None
    confidence: PatternConfidence = PatternConfidence.MEDIUM
    evidence: List[str] = field(default_factory=list)
    context: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_name": self.pattern.name,
            "category": self.pattern.category.value,
            "path": self.path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "confidence": self.confidence.value,
            "evidence": self.evidence,
            "suggestions": self.suggestions,
        }


# ============================================================================
# Pattern Definitions
# ============================================================================


# Design Patterns
SINGLETON_PATTERN = Pattern(
    name="Singleton",
    category=PatternCategory.DESIGN_PATTERN,
    description="Ensures a class has only one instance",
    indicators=["_instance class attribute", "__new__ override", "classmethod getInstance"],
    recommendations=["Consider using module-level instance instead", "Use dependency injection"],
)

FACTORY_PATTERN = Pattern(
    name="Factory",
    category=PatternCategory.DESIGN_PATTERN,
    description="Creates objects without specifying exact class",
    indicators=["create_* method", "factory method", "returns different subclasses"],
    recommendations=["Document supported types", "Consider abstract factory for families"],
)

STRATEGY_PATTERN = Pattern(
    name="Strategy",
    category=PatternCategory.DESIGN_PATTERN,
    description="Defines family of algorithms",
    indicators=["protocol/abstract base", "interchangeable implementations", "context holds strategy"],
    recommendations=["Use Protocol for type safety", "Consider enum for strategy selection"],
)

DECORATOR_PATTERN = Pattern(
    name="Decorator",
    category=PatternCategory.DESIGN_PATTERN,
    description="Adds behavior to objects dynamically",
    indicators=["@decorator syntax", "wrapper function", "functools.wraps"],
    recommendations=["Use functools.wraps to preserve metadata", "Document side effects"],
)

OBSERVER_PATTERN = Pattern(
    name="Observer",
    category=PatternCategory.DESIGN_PATTERN,
    description="Notifies dependents of state changes",
    indicators=["subscribe/unsubscribe methods", "notify/emit methods", "listeners list"],
    recommendations=["Use weak references for listeners", "Consider async notification"],
)

# Idioms
CONTEXT_MANAGER_IDIOM = Pattern(
    name="Context Manager",
    category=PatternCategory.IDIOM,
    description="Resource management with with statement",
    indicators=["__enter__", "__exit__", "@contextmanager"],
    recommendations=["Ensure proper cleanup in __exit__", "Handle exceptions correctly"],
)

DATACLASS_IDIOM = Pattern(
    name="Dataclass",
    category=PatternCategory.IDIOM,
    description="Structured data container",
    indicators=["@dataclass decorator", "field() calls", "frozen=True"],
    recommendations=["Use frozen=True for immutability", "Add __post_init__ for validation"],
)

BUILDER_IDIOM = Pattern(
    name="Builder/Fluent Interface",
    category=PatternCategory.IDIOM,
    description="Chainable method calls for object construction",
    indicators=["methods returning self", "chained calls", "build() final method"],
    recommendations=["Return self for chaining", "Validate in build()"],
)

# Anti-patterns
GOD_CLASS_ANTIPATTERN = Pattern(
    name="God Class",
    category=PatternCategory.ANTI_PATTERN,
    description="Class that does too much",
    indicators=["many methods (>20)", "many attributes", "multiple responsibilities"],
    recommendations=["Split into smaller classes", "Apply Single Responsibility Principle"],
)

LONG_METHOD_ANTIPATTERN = Pattern(
    name="Long Method",
    category=PatternCategory.ANTI_PATTERN,
    description="Method that is too long",
    indicators=["many lines (>50)", "deep nesting", "multiple abstractions"],
    recommendations=["Extract helper methods", "Use early returns"],
)

CIRCULAR_IMPORT_ANTIPATTERN = Pattern(
    name="Circular Import",
    category=PatternCategory.ANTI_PATTERN,
    description="Modules importing each other",
    indicators=["import inside function", "TYPE_CHECKING block"],
    recommendations=["Restructure modules", "Use dependency injection"],
)

MAGIC_NUMBER_ANTIPATTERN = Pattern(
    name="Magic Numbers",
    category=PatternCategory.ANTI_PATTERN,
    description="Unexplained numeric literals",
    indicators=["hardcoded numbers", "non-obvious constants"],
    recommendations=["Extract to named constants", "Add documentation"],
)

# All patterns
ALL_PATTERNS = [
    SINGLETON_PATTERN,
    FACTORY_PATTERN,
    STRATEGY_PATTERN,
    DECORATOR_PATTERN,
    OBSERVER_PATTERN,
    CONTEXT_MANAGER_IDIOM,
    DATACLASS_IDIOM,
    BUILDER_IDIOM,
    GOD_CLASS_ANTIPATTERN,
    LONG_METHOD_ANTIPATTERN,
    CIRCULAR_IMPORT_ANTIPATTERN,
    MAGIC_NUMBER_ANTIPATTERN,
]


# ============================================================================
# Pattern Detector
# ============================================================================


class PatternDetector:
    """
    Detect code patterns, idioms, and anti-patterns.

    Analyzes source code to identify:
    - Design patterns (Singleton, Factory, etc.)
    - Language idioms (context managers, dataclasses)
    - Anti-patterns (god class, long methods)

    PBTSO Phase: RESEARCH, DISTILL

    Example:
        detector = PatternDetector()
        matches = detector.detect_patterns("src/main.py")
        for match in matches:
            print(f"{match.pattern.name} at line {match.line_start}")
    """

    # Thresholds for anti-patterns
    GOD_CLASS_METHOD_THRESHOLD = 20
    LONG_METHOD_LINE_THRESHOLD = 50
    DEEP_NESTING_THRESHOLD = 4

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the pattern detector.

        Args:
            patterns: List of patterns to detect (default: all built-in)
            bus: AgentBus for event emission
        """
        self.patterns = patterns or ALL_PATTERNS
        self.bus = bus or AgentBus()

        # Build pattern lookup
        self._pattern_by_name = {p.name: p for p in self.patterns}

    def detect_patterns(
        self,
        file_path: str,
        content: Optional[str] = None,
        categories: Optional[List[PatternCategory]] = None,
    ) -> List[PatternMatch]:
        """
        Detect patterns in a file.

        Args:
            file_path: Path to the file
            content: File content (read from disk if not provided)
            categories: Filter by categories (None = all)

        Returns:
            List of pattern matches
        """
        if content is None:
            try:
                content = Path(file_path).read_text(errors="ignore")
            except Exception as e:
                return []

        # Filter patterns by category
        patterns_to_check = self.patterns
        if categories:
            patterns_to_check = [p for p in self.patterns if p.category in categories]

        matches = []

        # Parse AST for structural analysis
        try:
            tree = ast.parse(content)
        except SyntaxError:
            tree = None

        # Run detectors
        if tree:
            matches.extend(self._detect_singleton(tree, file_path, content))
            matches.extend(self._detect_factory(tree, file_path, content))
            matches.extend(self._detect_strategy(tree, file_path, content))
            matches.extend(self._detect_decorator_pattern(tree, file_path, content))
            matches.extend(self._detect_observer(tree, file_path, content))
            matches.extend(self._detect_context_manager(tree, file_path, content))
            matches.extend(self._detect_dataclass(tree, file_path, content))
            matches.extend(self._detect_builder(tree, file_path, content))
            matches.extend(self._detect_god_class(tree, file_path, content))
            matches.extend(self._detect_long_method(tree, file_path, content))
            matches.extend(self._detect_magic_numbers(tree, file_path, content))

        # Filter by requested categories
        if categories:
            matches = [m for m in matches if m.pattern.category in categories]

        # Emit events
        for match in matches:
            topic = "research.pattern.antipattern" if match.pattern.category == PatternCategory.ANTI_PATTERN else "research.pattern.found"
            self.bus.emit({
                "topic": topic,
                "kind": "pattern",
                "data": match.to_dict()
            })

        return matches

    def detect_in_codebase(
        self,
        root: Path,
        categories: Optional[List[PatternCategory]] = None,
    ) -> Dict[str, List[PatternMatch]]:
        """
        Detect patterns across a codebase.

        Args:
            root: Root directory
            categories: Filter by categories

        Returns:
            Dict mapping file paths to pattern matches
        """
        results = {}

        for py_file in root.rglob("*.py"):
            matches = self.detect_patterns(str(py_file), categories=categories)
            if matches:
                results[str(py_file)] = matches

        return results

    def get_anti_patterns(self, file_path: str) -> List[PatternMatch]:
        """Get only anti-patterns in a file."""
        return self.detect_patterns(
            file_path,
            categories=[PatternCategory.ANTI_PATTERN],
        )

    def get_design_patterns(self, file_path: str) -> List[PatternMatch]:
        """Get only design patterns in a file."""
        return self.detect_patterns(
            file_path,
            categories=[PatternCategory.DESIGN_PATTERN],
        )

    def _detect_singleton(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[PatternMatch]:
        """Detect Singleton pattern."""
        matches = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                evidence = []

                # Check for _instance attribute
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and target.id == "_instance":
                                evidence.append("Has _instance class attribute")

                    # Check for __new__ override
                    if isinstance(item, ast.FunctionDef) and item.name == "__new__":
                        evidence.append("Overrides __new__")

                    # Check for getInstance classmethod
                    if isinstance(item, ast.FunctionDef):
                        if any(isinstance(d, ast.Name) and d.id == "classmethod"
                               for d in item.decorator_list):
                            if "instance" in item.name.lower():
                                evidence.append(f"Has {item.name} classmethod")

                if len(evidence) >= 2:
                    matches.append(PatternMatch(
                        pattern=SINGLETON_PATTERN,
                        path=file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        confidence=PatternConfidence.HIGH,
                        evidence=evidence,
                        suggestions=["Consider using module-level instance instead"],
                    ))

        return matches

    def _detect_factory(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[PatternMatch]:
        """Detect Factory pattern."""
        matches = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                evidence = []

                # Check for factory naming convention
                if node.name.startswith(("create_", "make_", "build_")):
                    evidence.append(f"Factory method name: {node.name}")

                # Check for multiple return statements with different types
                returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
                if len(returns) > 1:
                    evidence.append("Multiple return paths")

                if evidence:
                    matches.append(PatternMatch(
                        pattern=FACTORY_PATTERN,
                        path=file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        confidence=PatternConfidence.MEDIUM,
                        evidence=evidence,
                    ))

            elif isinstance(node, ast.ClassDef):
                # Check for Factory class
                if "factory" in node.name.lower():
                    matches.append(PatternMatch(
                        pattern=FACTORY_PATTERN,
                        path=file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        confidence=PatternConfidence.HIGH,
                        evidence=["Class named Factory"],
                    ))

        return matches

    def _detect_strategy(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[PatternMatch]:
        """Detect Strategy pattern."""
        matches = []

        # Look for abstract base classes or protocols
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check bases for ABC
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id in ("ABC", "Protocol"):
                        # Check for abstract methods
                        abstract_methods = []
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                for decorator in item.decorator_list:
                                    if isinstance(decorator, ast.Name) and decorator.id == "abstractmethod":
                                        abstract_methods.append(item.name)

                        if abstract_methods:
                            matches.append(PatternMatch(
                                pattern=STRATEGY_PATTERN,
                                path=file_path,
                                line_start=node.lineno,
                                line_end=node.end_lineno,
                                confidence=PatternConfidence.MEDIUM,
                                evidence=[f"Abstract strategy interface with methods: {', '.join(abstract_methods)}"],
                            ))

        return matches

    def _detect_decorator_pattern(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[PatternMatch]:
        """Detect Decorator pattern (Python decorators)."""
        matches = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if it's a decorator (takes and returns function)
                has_wrapper = False
                uses_wraps = False

                for item in ast.walk(node):
                    if isinstance(item, ast.FunctionDef) and item.name == "wrapper":
                        has_wrapper = True
                    if isinstance(item, ast.Name) and item.id == "wraps":
                        uses_wraps = True

                if has_wrapper:
                    evidence = ["Contains wrapper function"]
                    if uses_wraps:
                        evidence.append("Uses functools.wraps")

                    matches.append(PatternMatch(
                        pattern=DECORATOR_PATTERN,
                        path=file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        confidence=PatternConfidence.HIGH if uses_wraps else PatternConfidence.MEDIUM,
                        evidence=evidence,
                    ))

        return matches

    def _detect_observer(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[PatternMatch]:
        """Detect Observer pattern."""
        matches = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}

                evidence = []

                # Check for observer-related methods
                if "subscribe" in methods or "add_listener" in methods:
                    evidence.append("Has subscribe/add_listener method")
                if "unsubscribe" in methods or "remove_listener" in methods:
                    evidence.append("Has unsubscribe/remove_listener method")
                if "notify" in methods or "emit" in methods:
                    evidence.append("Has notify/emit method")

                if len(evidence) >= 2:
                    matches.append(PatternMatch(
                        pattern=OBSERVER_PATTERN,
                        path=file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        confidence=PatternConfidence.HIGH,
                        evidence=evidence,
                    ))

        return matches

    def _detect_context_manager(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[PatternMatch]:
        """Detect Context Manager idiom."""
        matches = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}

                if "__enter__" in methods and "__exit__" in methods:
                    matches.append(PatternMatch(
                        pattern=CONTEXT_MANAGER_IDIOM,
                        path=file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        confidence=PatternConfidence.HIGH,
                        evidence=["Implements __enter__ and __exit__"],
                    ))

            elif isinstance(node, ast.FunctionDef):
                # Check for @contextmanager decorator
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "contextmanager":
                        matches.append(PatternMatch(
                            pattern=CONTEXT_MANAGER_IDIOM,
                            path=file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno,
                            confidence=PatternConfidence.HIGH,
                            evidence=["Uses @contextmanager decorator"],
                        ))

        return matches

    def _detect_dataclass(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[PatternMatch]:
        """Detect Dataclass idiom."""
        matches = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                        evidence = ["Uses @dataclass decorator"]

                        # Check for frozen
                        matches.append(PatternMatch(
                            pattern=DATACLASS_IDIOM,
                            path=file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno,
                            confidence=PatternConfidence.HIGH,
                            evidence=evidence,
                        ))

                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name) and decorator.func.id == "dataclass":
                            matches.append(PatternMatch(
                                pattern=DATACLASS_IDIOM,
                                path=file_path,
                                line_start=node.lineno,
                                line_end=node.end_lineno,
                                confidence=PatternConfidence.HIGH,
                                evidence=["Uses @dataclass decorator"],
                            ))

        return matches

    def _detect_builder(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[PatternMatch]:
        """Detect Builder/Fluent Interface idiom."""
        matches = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods_returning_self = []

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        # Check if method returns self
                        for stmt in ast.walk(item):
                            if isinstance(stmt, ast.Return):
                                if isinstance(stmt.value, ast.Name) and stmt.value.id == "self":
                                    methods_returning_self.append(item.name)
                                    break

                # Builder pattern typically has multiple self-returning methods
                if len(methods_returning_self) >= 3:
                    matches.append(PatternMatch(
                        pattern=BUILDER_IDIOM,
                        path=file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        confidence=PatternConfidence.MEDIUM,
                        evidence=[f"Methods returning self: {', '.join(methods_returning_self[:5])}"],
                    ))

        return matches

    def _detect_god_class(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[PatternMatch]:
        """Detect God Class anti-pattern."""
        matches = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [item for item in node.body if isinstance(item, ast.FunctionDef)]
                attributes = [item for item in node.body if isinstance(item, ast.Assign)]

                evidence = []

                if len(methods) > self.GOD_CLASS_METHOD_THRESHOLD:
                    evidence.append(f"{len(methods)} methods (threshold: {self.GOD_CLASS_METHOD_THRESHOLD})")

                if len(attributes) > 10:
                    evidence.append(f"{len(attributes)} class attributes")

                if evidence:
                    matches.append(PatternMatch(
                        pattern=GOD_CLASS_ANTIPATTERN,
                        path=file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        confidence=PatternConfidence.HIGH if len(methods) > 30 else PatternConfidence.MEDIUM,
                        evidence=evidence,
                        suggestions=["Consider splitting into smaller, focused classes"],
                    ))

        return matches

    def _detect_long_method(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[PatternMatch]:
        """Detect Long Method anti-pattern."""
        matches = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.end_lineno and node.lineno:
                    length = node.end_lineno - node.lineno

                    if length > self.LONG_METHOD_LINE_THRESHOLD:
                        matches.append(PatternMatch(
                            pattern=LONG_METHOD_ANTIPATTERN,
                            path=file_path,
                            line_start=node.lineno,
                            line_end=node.end_lineno,
                            confidence=PatternConfidence.HIGH,
                            evidence=[f"{length} lines (threshold: {self.LONG_METHOD_LINE_THRESHOLD})"],
                            suggestions=["Extract helper methods", "Consider decomposing"],
                        ))

        return matches

    def _detect_magic_numbers(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[PatternMatch]:
        """Detect Magic Number anti-pattern."""
        matches = []

        # Ignore common acceptable numbers
        acceptable = {0, 1, 2, -1, 100, 1000}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                magic_numbers = []

                for child in ast.walk(node):
                    if isinstance(child, ast.Constant):
                        if isinstance(child.value, (int, float)):
                            if child.value not in acceptable and abs(child.value) > 2:
                                magic_numbers.append((child.lineno, child.value))

                # Only report if multiple magic numbers
                if len(magic_numbers) >= 3:
                    matches.append(PatternMatch(
                        pattern=MAGIC_NUMBER_ANTIPATTERN,
                        path=file_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        confidence=PatternConfidence.MEDIUM,
                        evidence=[f"Magic numbers found: {[n[1] for n in magic_numbers[:5]]}"],
                        suggestions=["Extract to named constants"],
                    ))

        return matches


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Pattern Detector."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pattern Detector (Step 15)"
    )
    parser.add_argument(
        "path",
        help="File or directory to analyze"
    )
    parser.add_argument(
        "--category",
        choices=["design_pattern", "idiom", "anti_pattern", "all"],
        default="all",
        help="Pattern category to detect"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    detector = PatternDetector()
    path = Path(args.path)

    # Determine categories
    categories = None
    if args.category != "all":
        categories = [PatternCategory(args.category)]

    if path.is_file():
        matches = detector.detect_patterns(str(path), categories=categories)
    else:
        results = detector.detect_in_codebase(path, categories=categories)
        matches = []
        for file_matches in results.values():
            matches.extend(file_matches)

    if args.json:
        print(json.dumps([m.to_dict() for m in matches], indent=2))
    else:
        print(f"Found {len(matches)} pattern(s):")
        for match in matches:
            icon = "!" if match.pattern.category == PatternCategory.ANTI_PATTERN else "+"
            print(f"\n{icon} {match.pattern.name} [{match.confidence.value}]")
            print(f"  File: {match.path}:{match.line_start}")
            print(f"  Category: {match.pattern.category.value}")
            for ev in match.evidence:
                print(f"  - {ev}")
            if match.suggestions:
                print(f"  Suggestions:")
                for sug in match.suggestions:
                    print(f"    -> {sug}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
