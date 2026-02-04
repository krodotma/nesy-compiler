#!/usr/bin/env python3
"""
Code Smell Detector (Step 154)

Detects code smells and anti-patterns using AST analysis and metrics.

PBTSO Phase: VERIFY
Bus Topics: review.smells.detect, review.smells.found

Common code smells detected:
- Long Method: Methods exceeding line threshold
- Large Class: Classes with too many methods/attributes
- Duplicate Code: Similar code blocks
- Dead Code: Unreachable or unused code
- Magic Numbers: Unexplained numeric literals
- God Class: Classes doing too much
- Feature Envy: Methods using other classes more than their own
- Data Clumps: Groups of data that appear together

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# Types
# ============================================================================

class SmellType(Enum):
    """Types of code smells."""
    LONG_METHOD = "long_method"
    LARGE_CLASS = "large_class"
    DUPLICATE_CODE = "duplicate_code"
    DEAD_CODE = "dead_code"
    MAGIC_NUMBER = "magic_number"
    GOD_CLASS = "god_class"
    FEATURE_ENVY = "feature_envy"
    DATA_CLUMPS = "data_clumps"
    LONG_PARAMETER_LIST = "long_parameter_list"
    COMPLEX_CONDITIONAL = "complex_conditional"
    DEEP_NESTING = "deep_nesting"
    PRIMITIVE_OBSESSION = "primitive_obsession"
    SPECULATIVE_GENERALITY = "speculative_generality"
    REFUSED_BEQUEST = "refused_bequest"
    COMMENTS_SMELL = "comments_smell"


class SmellSeverity(Enum):
    """Severity levels for code smells."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def priority(self) -> int:
        """Get numeric priority."""
        return {"high": 3, "medium": 2, "low": 1}.get(self.value, 0)


@dataclass
class CodeSmell:
    """
    Represents a detected code smell.

    Attributes:
        smell_type: Type of the smell
        file: File path where smell was found
        location: Location description (class name, method name, line)
        line: Starting line number
        end_line: Ending line number
        severity: Smell severity
        description: Human-readable description
        suggestion: Refactoring suggestion
        metrics: Related metrics (lines, complexity, etc.)
    """
    smell_type: SmellType
    file: str
    location: str
    line: int
    end_line: int
    severity: SmellSeverity
    description: str
    suggestion: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["smell_type"] = self.smell_type.value
        result["severity"] = self.severity.value
        return result


@dataclass
class SmellDetectionResult:
    """Result from smell detection."""
    files_analyzed: int = 0
    smells: List[CodeSmell] = field(default_factory=list)
    duration_ms: float = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    smell_type_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "files_analyzed": self.files_analyzed,
            "smells": [s.to_dict() for s in self.smells],
            "duration_ms": self.duration_ms,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "smell_type_counts": self.smell_type_counts,
        }


# ============================================================================
# Code Metrics
# ============================================================================

@dataclass
class FunctionMetrics:
    """Metrics for a single function/method."""
    name: str
    file: str
    line: int
    end_line: int
    lines_of_code: int
    parameter_count: int
    cyclomatic_complexity: int
    nesting_depth: int
    return_count: int
    variable_count: int


@dataclass
class ClassMetrics:
    """Metrics for a single class."""
    name: str
    file: str
    line: int
    end_line: int
    lines_of_code: int
    method_count: int
    attribute_count: int
    public_method_count: int
    private_method_count: int
    inheritance_depth: int
    weighted_methods: int  # Sum of method complexities


class MetricsCollector(ast.NodeVisitor):
    """AST visitor to collect code metrics."""

    def __init__(self, file_path: str, content: str):
        self.file_path = file_path
        self.content = content
        self.lines = content.split("\n")
        self.functions: List[FunctionMetrics] = []
        self.classes: List[ClassMetrics] = []
        self._current_class: Optional[str] = None
        self._current_function: Optional[str] = None
        self._nesting_depth = 0
        self._max_nesting = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        prev_class = self._current_class
        self._current_class = node.name

        # Count methods and attributes
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        attributes = []

        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                attributes.append(item.target.id)

        public_methods = [m for m in methods if not m.name.startswith("_")]
        private_methods = [m for m in methods if m.name.startswith("_")]

        # Calculate weighted methods (sum of complexities)
        weighted = sum(self._calculate_complexity(m) for m in methods)

        self.classes.append(ClassMetrics(
            name=node.name,
            file=self.file_path,
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            lines_of_code=(node.end_lineno or node.lineno) - node.lineno + 1,
            method_count=len(methods),
            attribute_count=len(attributes),
            public_method_count=len(public_methods),
            private_method_count=len(private_methods),
            inheritance_depth=len(node.bases),
            weighted_methods=weighted,
        ))

        self.generic_visit(node)
        self._current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Common function visit logic."""
        prev_function = self._current_function
        self._current_function = node.name

        # Reset nesting tracking
        prev_max = self._max_nesting
        self._max_nesting = 0
        self._nesting_depth = 0

        # Count parameters
        params = node.args
        param_count = (
            len(params.args) +
            len(params.posonlyargs) +
            len(params.kwonlyargs) +
            (1 if params.vararg else 0) +
            (1 if params.kwarg else 0)
        )

        # Don't count 'self' or 'cls'
        if param_count > 0 and params.args:
            first_arg = params.args[0].arg
            if first_arg in ("self", "cls"):
                param_count -= 1

        # Calculate complexity
        complexity = self._calculate_complexity(node)

        # Count returns
        return_count = sum(1 for _ in ast.walk(node) if isinstance(_, ast.Return))

        # Count variables
        variables: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                variables.add(child.id)

        # Visit body to calculate nesting
        self.generic_visit(node)

        self.functions.append(FunctionMetrics(
            name=f"{self._current_class}.{node.name}" if self._current_class else node.name,
            file=self.file_path,
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            lines_of_code=(node.end_lineno or node.lineno) - node.lineno + 1,
            parameter_count=param_count,
            cyclomatic_complexity=complexity,
            nesting_depth=self._max_nesting,
            return_count=return_count,
            variable_count=len(variables),
        ))

        self._current_function = prev_function
        self._max_nesting = prev_max

    def visit_If(self, node: ast.If) -> None:
        self._track_nesting(node)

    def visit_For(self, node: ast.For) -> None:
        self._track_nesting(node)

    def visit_While(self, node: ast.While) -> None:
        self._track_nesting(node)

    def visit_With(self, node: ast.With) -> None:
        self._track_nesting(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._track_nesting(node)

    def _track_nesting(self, node: ast.AST) -> None:
        """Track nesting depth."""
        self._nesting_depth += 1
        self._max_nesting = max(self._max_nesting, self._nesting_depth)
        self.generic_visit(node)
        self._nesting_depth -= 1

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a node."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
                if child.ifs:
                    complexity += len(child.ifs)

        return complexity


# ============================================================================
# Code Smell Detector
# ============================================================================

class CodeSmellDetector:
    """
    Detects code smells and anti-patterns.

    Uses AST analysis and code metrics to identify potential
    code quality issues.

    Example:
        detector = CodeSmellDetector()
        result = detector.detect(["/path/to/file.py"])
        for smell in result.smells:
            print(f"[{smell.severity.value}] {smell.location}: {smell.description}")
    """

    # Default thresholds for smell detection
    THRESHOLDS = {
        "method_lines": 50,
        "class_lines": 500,
        "method_params": 5,
        "complexity": 10,
        "nesting_depth": 4,
        "class_methods": 20,
        "class_attributes": 15,
        "weighted_methods": 100,
        "magic_number_threshold": 2,  # Numbers > 2 are suspicious
    }

    def __init__(
        self,
        thresholds: Optional[Dict[str, int]] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the smell detector.

        Args:
            thresholds: Custom thresholds for smell detection
            bus_path: Path to event bus file
        """
        self.thresholds = {**self.THRESHOLDS, **(thresholds or {})}
        self.bus_path = bus_path or self._get_bus_path()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any]) -> None:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "smell",
            "actor": "smell-detector",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def detect(
        self,
        files: List[str],
        content_map: Optional[Dict[str, str]] = None,
    ) -> SmellDetectionResult:
        """
        Detect code smells in files.

        Args:
            files: List of file paths to analyze
            content_map: Optional pre-loaded file contents

        Returns:
            SmellDetectionResult with all smells found

        Emits:
            review.smells.detect (start)
            review.smells.found (per smell batch)
        """
        start_time = time.time()

        # Emit start event
        self._emit_event("review.smells.detect", {
            "files": files[:20],
            "file_count": len(files),
            "status": "started",
        })

        result = SmellDetectionResult(
            files_analyzed=len(files),
        )

        # Only analyze Python files for now
        python_files = [f for f in files if f.endswith(".py")]

        for file_path in python_files:
            # Get file content
            if content_map and file_path in content_map:
                content = content_map[file_path]
            else:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except (IOError, OSError):
                    continue

            # Detect smells
            smells = self._analyze_file(file_path, content)
            result.smells.extend(smells)

        # Calculate counts
        smell_counts: Dict[str, int] = defaultdict(int)
        for smell in result.smells:
            smell_counts[smell.smell_type.value] += 1
            if smell.severity == SmellSeverity.HIGH:
                result.high_count += 1
            elif smell.severity == SmellSeverity.MEDIUM:
                result.medium_count += 1
            else:
                result.low_count += 1

        result.smell_type_counts = dict(smell_counts)
        result.duration_ms = (time.time() - start_time) * 1000

        # Emit smells found
        if result.smells:
            self._emit_event("review.smells.found", {
                "smell_count": len(result.smells),
                "smell_types": result.smell_type_counts,
                "high_count": result.high_count,
                "smells": [s.to_dict() for s in result.smells[:10]],
            })

        # Emit completion
        self._emit_event("review.smells.detect", {
            "status": "completed",
            "files_analyzed": result.files_analyzed,
            "smell_count": len(result.smells),
            "duration_ms": result.duration_ms,
        })

        return result

    def _analyze_file(self, file_path: str, content: str) -> List[CodeSmell]:
        """Analyze a single file for smells."""
        smells: List[CodeSmell] = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return smells

        # Collect metrics
        collector = MetricsCollector(file_path, content)
        collector.visit(tree)

        # Detect smells from function metrics
        for func in collector.functions:
            smells.extend(self._check_function_smells(func))

        # Detect smells from class metrics
        for cls in collector.classes:
            smells.extend(self._check_class_smells(cls))

        # Detect magic numbers
        smells.extend(self._check_magic_numbers(file_path, content, tree))

        # Detect complex conditionals
        smells.extend(self._check_complex_conditionals(file_path, tree))

        # Detect duplicate code (basic hashing approach)
        smells.extend(self._check_duplicate_blocks(file_path, content))

        return smells

    def _check_function_smells(self, func: FunctionMetrics) -> List[CodeSmell]:
        """Check function for smells."""
        smells = []

        # Long Method
        if func.lines_of_code > self.thresholds["method_lines"]:
            severity = SmellSeverity.HIGH if func.lines_of_code > self.thresholds["method_lines"] * 2 else SmellSeverity.MEDIUM
            smells.append(CodeSmell(
                smell_type=SmellType.LONG_METHOD,
                file=func.file,
                location=func.name,
                line=func.line,
                end_line=func.end_line,
                severity=severity,
                description=f"Method has {func.lines_of_code} lines (threshold: {self.thresholds['method_lines']})",
                suggestion="Consider breaking this method into smaller, focused functions",
                metrics={"lines_of_code": func.lines_of_code},
            ))

        # Long Parameter List
        if func.parameter_count > self.thresholds["method_params"]:
            severity = SmellSeverity.MEDIUM if func.parameter_count <= self.thresholds["method_params"] + 2 else SmellSeverity.HIGH
            smells.append(CodeSmell(
                smell_type=SmellType.LONG_PARAMETER_LIST,
                file=func.file,
                location=func.name,
                line=func.line,
                end_line=func.end_line,
                severity=severity,
                description=f"Method has {func.parameter_count} parameters (threshold: {self.thresholds['method_params']})",
                suggestion="Consider using a parameter object or splitting the method",
                metrics={"parameter_count": func.parameter_count},
            ))

        # High Complexity
        if func.cyclomatic_complexity > self.thresholds["complexity"]:
            severity = SmellSeverity.HIGH if func.cyclomatic_complexity > self.thresholds["complexity"] * 2 else SmellSeverity.MEDIUM
            smells.append(CodeSmell(
                smell_type=SmellType.COMPLEX_CONDITIONAL,
                file=func.file,
                location=func.name,
                line=func.line,
                end_line=func.end_line,
                severity=severity,
                description=f"Method has cyclomatic complexity of {func.cyclomatic_complexity} (threshold: {self.thresholds['complexity']})",
                suggestion="Simplify conditionals, extract methods, or use polymorphism",
                metrics={"complexity": func.cyclomatic_complexity},
            ))

        # Deep Nesting
        if func.nesting_depth > self.thresholds["nesting_depth"]:
            smells.append(CodeSmell(
                smell_type=SmellType.DEEP_NESTING,
                file=func.file,
                location=func.name,
                line=func.line,
                end_line=func.end_line,
                severity=SmellSeverity.MEDIUM,
                description=f"Method has nesting depth of {func.nesting_depth} (threshold: {self.thresholds['nesting_depth']})",
                suggestion="Use early returns, extract conditions to methods, or flatten logic",
                metrics={"nesting_depth": func.nesting_depth},
            ))

        return smells

    def _check_class_smells(self, cls: ClassMetrics) -> List[CodeSmell]:
        """Check class for smells."""
        smells = []

        # Large Class
        if cls.lines_of_code > self.thresholds["class_lines"]:
            severity = SmellSeverity.HIGH if cls.lines_of_code > self.thresholds["class_lines"] * 2 else SmellSeverity.MEDIUM
            smells.append(CodeSmell(
                smell_type=SmellType.LARGE_CLASS,
                file=cls.file,
                location=cls.name,
                line=cls.line,
                end_line=cls.end_line,
                severity=severity,
                description=f"Class has {cls.lines_of_code} lines (threshold: {self.thresholds['class_lines']})",
                suggestion="Consider splitting into smaller, focused classes",
                metrics={"lines_of_code": cls.lines_of_code},
            ))

        # God Class (too many methods + high weighted complexity)
        if cls.method_count > self.thresholds["class_methods"] and cls.weighted_methods > self.thresholds["weighted_methods"]:
            smells.append(CodeSmell(
                smell_type=SmellType.GOD_CLASS,
                file=cls.file,
                location=cls.name,
                line=cls.line,
                end_line=cls.end_line,
                severity=SmellSeverity.HIGH,
                description=f"Class has {cls.method_count} methods with total complexity {cls.weighted_methods}",
                suggestion="Extract related functionality into separate classes",
                metrics={
                    "method_count": cls.method_count,
                    "weighted_methods": cls.weighted_methods,
                },
            ))

        # Too many attributes
        if cls.attribute_count > self.thresholds["class_attributes"]:
            smells.append(CodeSmell(
                smell_type=SmellType.DATA_CLUMPS,
                file=cls.file,
                location=cls.name,
                line=cls.line,
                end_line=cls.end_line,
                severity=SmellSeverity.MEDIUM,
                description=f"Class has {cls.attribute_count} attributes (threshold: {self.thresholds['class_attributes']})",
                suggestion="Group related attributes into separate data classes",
                metrics={"attribute_count": cls.attribute_count},
            ))

        return smells

    def _check_magic_numbers(self, file_path: str, content: str, tree: ast.AST) -> List[CodeSmell]:
        """Check for magic numbers."""
        smells = []
        number_occurrences: Dict[int | float, List[int]] = defaultdict(list)

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                # Ignore common acceptable numbers
                if node.value in (0, 1, -1, 2, 10, 100, 1000):
                    continue
                number_occurrences[node.value].append(node.lineno)

        # Report numbers that appear multiple times
        for number, lines in number_occurrences.items():
            if len(lines) >= 2:
                smells.append(CodeSmell(
                    smell_type=SmellType.MAGIC_NUMBER,
                    file=file_path,
                    location=f"value {number}",
                    line=lines[0],
                    end_line=lines[-1],
                    severity=SmellSeverity.LOW,
                    description=f"Magic number {number} appears {len(lines)} times at lines {lines[:5]}",
                    suggestion="Extract to a named constant with a descriptive name",
                    metrics={"occurrences": len(lines), "value": number},
                ))

        return smells

    def _check_complex_conditionals(self, file_path: str, tree: ast.AST) -> List[CodeSmell]:
        """Check for overly complex conditionals."""
        smells = []

        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Count boolean operations in condition
                bool_ops = sum(1 for _ in ast.walk(node.test) if isinstance(_, ast.BoolOp))
                comparisons = sum(1 for _ in ast.walk(node.test) if isinstance(_, ast.Compare))

                if bool_ops + comparisons > 3:
                    smells.append(CodeSmell(
                        smell_type=SmellType.COMPLEX_CONDITIONAL,
                        file=file_path,
                        location=f"line {node.lineno}",
                        line=node.lineno,
                        end_line=node.lineno,
                        severity=SmellSeverity.MEDIUM,
                        description=f"Complex conditional with {bool_ops + comparisons} conditions",
                        suggestion="Extract condition to a well-named method or use guard clauses",
                        metrics={"condition_count": bool_ops + comparisons},
                    ))

        return smells

    def _check_duplicate_blocks(self, file_path: str, content: str) -> List[CodeSmell]:
        """Check for duplicate code blocks using simple hashing."""
        smells = []
        lines = content.split("\n")

        # Skip if file is too small
        if len(lines) < 10:
            return smells

        # Hash 5-line blocks
        block_size = 5
        block_hashes: Dict[str, List[int]] = defaultdict(list)

        for i in range(len(lines) - block_size + 1):
            block = "\n".join(lines[i:i + block_size])
            # Normalize whitespace
            normalized = re.sub(r"\s+", " ", block.strip())
            if len(normalized) < 20:  # Skip very short blocks
                continue
            block_hash = hashlib.md5(normalized.encode()).hexdigest()
            block_hashes[block_hash].append(i + 1)

        # Report duplicates
        for hash_val, line_nums in block_hashes.items():
            if len(line_nums) >= 2:
                smells.append(CodeSmell(
                    smell_type=SmellType.DUPLICATE_CODE,
                    file=file_path,
                    location=f"lines {line_nums[:3]}",
                    line=line_nums[0],
                    end_line=line_nums[0] + block_size,
                    severity=SmellSeverity.MEDIUM,
                    description=f"Duplicate code block found at {len(line_nums)} locations",
                    suggestion="Extract duplicated code to a shared method or function",
                    metrics={"duplicate_count": len(line_nums), "lines": line_nums[:5]},
                ))

        return smells

    def detect_from_metrics(self, metrics: Dict[str, Any]) -> List[CodeSmell]:
        """
        Detect smells from pre-computed metrics.

        Args:
            metrics: Dictionary with file, name, lines_of_code, etc.

        Returns:
            List of detected smells
        """
        smells = []
        file_path = metrics.get("file", "<unknown>")
        name = metrics.get("name", "<unknown>")
        line = metrics.get("line", 1)
        end_line = metrics.get("end_line", line)

        # Check lines of code
        loc = metrics.get("lines_of_code", 0)
        if loc > self.thresholds["method_lines"]:
            smells.append(CodeSmell(
                smell_type=SmellType.LONG_METHOD,
                file=file_path,
                location=name,
                line=line,
                end_line=end_line,
                severity=SmellSeverity.MEDIUM,
                description=f"Method has {loc} lines (threshold: {self.thresholds['method_lines']})",
                suggestion="Consider breaking this method into smaller functions",
                metrics={"lines_of_code": loc},
            ))

        return smells


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Code Smell Detector."""
    import argparse

    parser = argparse.ArgumentParser(description="Code Smell Detector (Step 154)")
    parser.add_argument("files", nargs="+", help="Files to analyze")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument("--severity", choices=["high", "medium", "low"],
                        default="low", help="Minimum severity to report")

    args = parser.parse_args()

    detector = CodeSmellDetector()
    result = detector.detect(args.files)

    # Filter by severity
    min_severity = SmellSeverity[args.severity.upper()]
    result.smells = [
        s for s in result.smells
        if s.severity.priority >= min_severity.priority
    ]

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.summary:
        print(f"Code Smell Summary:")
        print(f"  Files analyzed: {result.files_analyzed}")
        print(f"  Total smells: {len(result.smells)}")
        print(f"  High: {result.high_count}")
        print(f"  Medium: {result.medium_count}")
        print(f"  Low: {result.low_count}")
        print(f"  Duration: {result.duration_ms:.1f}ms")
        if result.smell_type_counts:
            print(f"  By type:")
            for smell_type, count in sorted(result.smell_type_counts.items()):
                print(f"    {smell_type}: {count}")
    else:
        for smell in result.smells:
            severity_color = {
                SmellSeverity.HIGH: "\033[91m",
                SmellSeverity.MEDIUM: "\033[93m",
                SmellSeverity.LOW: "\033[90m",
            }.get(smell.severity, "")
            reset = "\033[0m"

            print(f"{severity_color}[{smell.severity.value.upper()}]{reset} {smell.file}:{smell.line}")
            print(f"  {smell.smell_type.value}: {smell.description}")
            print(f"  Suggestion: {smell.suggestion}")
            print()

    return 1 if result.high_count > 0 else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
