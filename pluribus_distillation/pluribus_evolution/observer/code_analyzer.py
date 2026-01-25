#!/usr/bin/env python3
"""
code_analyzer.py - Analyzes primary trunk code for patterns and optimization opportunities.

Part of the pluribus_evolution observer subsystem.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CodePattern:
    """A detected code pattern."""
    pattern_type: str  # function, class, import, antipattern, etc.
    location: str      # file:line
    description: str
    confidence: float  # 0.0 - 1.0
    suggestion: str | None = None


@dataclass
class AnalysisResult:
    """Result of code analysis."""
    file_path: str
    patterns: list[CodePattern] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


class CodeAnalyzer:
    """
    Analyzes Python code in the primary trunk.

    Retroactive analysis: Finds patterns, antipatterns, and optimization opportunities
    in existing code to guide evolution.
    """

    def __init__(self, primary_root: str = "/pluribus"):
        self.primary_root = Path(primary_root)
        self.patterns_detected: list[CodePattern] = []

    def analyze_file(self, file_path: str) -> AnalysisResult:
        """Analyze a single Python file."""
        path = Path(file_path)
        if not path.exists():
            return AnalysisResult(file_path=file_path, summary="File not found")

        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
        except Exception as e:
            return AnalysisResult(file_path=file_path, summary=f"Parse error: {e}")

        result = AnalysisResult(file_path=file_path)

        # Count metrics
        result.metrics = {
            "lines": len(source.splitlines()),
            "functions": 0,
            "classes": 0,
            "imports": 0,
            "complexity_estimate": 0,
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                result.metrics["functions"] += 1
                # Check for large functions
                if hasattr(node, "end_lineno") and node.end_lineno:
                    func_lines = node.end_lineno - node.lineno
                    if func_lines > 50:
                        result.patterns.append(CodePattern(
                            pattern_type="large_function",
                            location=f"{file_path}:{node.lineno}",
                            description=f"Function '{node.name}' is {func_lines} lines",
                            confidence=0.9,
                            suggestion="Consider breaking into smaller functions"
                        ))
            elif isinstance(node, ast.ClassDef):
                result.metrics["classes"] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                result.metrics["imports"] += 1

        # Estimate complexity
        result.metrics["complexity_estimate"] = (
            result.metrics["functions"] * 2 +
            result.metrics["classes"] * 3 +
            result.metrics["lines"] // 100
        )

        result.summary = (
            f"{result.metrics['functions']} functions, "
            f"{result.metrics['classes']} classes, "
            f"{len(result.patterns)} patterns detected"
        )

        return result

    def analyze_directory(self, dir_path: str, pattern: str = "**/*.py") -> list[AnalysisResult]:
        """Analyze all Python files in a directory."""
        path = Path(dir_path)
        results = []

        for file_path in path.glob(pattern):
            if "__pycache__" in str(file_path) or ".venv" in str(file_path):
                continue
            results.append(self.analyze_file(str(file_path)))

        return results

    def find_antipatterns(self, results: list[AnalysisResult]) -> list[CodePattern]:
        """Extract antipatterns from analysis results."""
        antipatterns = []
        for result in results:
            for pattern in result.patterns:
                if pattern.pattern_type in ("large_function", "deep_nesting", "god_class"):
                    antipatterns.append(pattern)
        return antipatterns

    def to_bus_event(self, results: list[AnalysisResult]) -> dict:
        """Convert analysis results to a bus event payload."""
        total_patterns = sum(len(r.patterns) for r in results)
        total_files = len(results)

        return {
            "topic": "evolution.observer.analysis",
            "kind": "artifact",
            "level": "info",
            "data": {
                "files_analyzed": total_files,
                "patterns_detected": total_patterns,
                "antipatterns": len(self.find_antipatterns(results)),
                "summaries": [r.summary for r in results[:10]],  # First 10
            }
        }


if __name__ == "__main__":
    import sys

    analyzer = CodeAnalyzer()

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "/pluribus/nucleus/tools"

    print(f"Analyzing: {target}")
    results = analyzer.analyze_directory(target)

    for r in results[:5]:
        print(f"  {r.file_path}: {r.summary}")

    print(f"\nTotal: {len(results)} files analyzed")
    antipatterns = analyzer.find_antipatterns(results)
    print(f"Antipatterns found: {len(antipatterns)}")
