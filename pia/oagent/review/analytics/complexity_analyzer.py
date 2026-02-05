#!/usr/bin/env python3
"""
Code Complexity Analyzer (Step 173)

Analyzes code complexity using multiple metrics.

PBTSO Phase: VERIFY, OBSERVE
Bus Topics: review.complexity.analyze, review.complexity.metrics

Metrics calculated:
- Cyclomatic Complexity
- Cognitive Complexity
- Halstead Metrics
- Maintainability Index
- Lines of Code metrics
- Nesting depth

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import ast
import asyncio
import fcntl
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ============================================================================
# Types
# ============================================================================

class ComplexityLevel(Enum):
    """Complexity rating levels."""
    LOW = "low"           # Easy to maintain
    MODERATE = "moderate" # Acceptable
    HIGH = "high"         # Consider refactoring
    VERY_HIGH = "very_high"  # Needs refactoring


@dataclass
class ComplexityConfig:
    """Configuration for complexity analysis."""
    cyclomatic_threshold_low: int = 5
    cyclomatic_threshold_moderate: int = 10
    cyclomatic_threshold_high: int = 20
    cognitive_threshold_low: int = 8
    cognitive_threshold_moderate: int = 15
    cognitive_threshold_high: int = 25
    max_nesting_depth: int = 4
    max_method_lines: int = 50
    maintainability_threshold: float = 65.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HalsteadMetrics:
    """
    Halstead complexity metrics.

    Attributes:
        n1: Number of distinct operators
        n2: Number of distinct operands
        N1: Total number of operators
        N2: Total number of operands
        vocabulary: n1 + n2
        length: N1 + N2
        calculated_length: n1 * log2(n1) + n2 * log2(n2)
        volume: length * log2(vocabulary)
        difficulty: (n1/2) * (N2/n2)
        effort: difficulty * volume
        time_to_program: effort / 18 (seconds)
        bugs_delivered: volume / 3000
    """
    n1: int = 0
    n2: int = 0
    N1: int = 0
    N2: int = 0

    @property
    def vocabulary(self) -> int:
        return self.n1 + self.n2

    @property
    def length(self) -> int:
        return self.N1 + self.N2

    @property
    def calculated_length(self) -> float:
        if self.n1 == 0 or self.n2 == 0:
            return 0.0
        return self.n1 * math.log2(self.n1) + self.n2 * math.log2(self.n2)

    @property
    def volume(self) -> float:
        if self.vocabulary == 0:
            return 0.0
        return self.length * math.log2(self.vocabulary)

    @property
    def difficulty(self) -> float:
        if self.n2 == 0:
            return 0.0
        return (self.n1 / 2) * (self.N2 / self.n2)

    @property
    def effort(self) -> float:
        return self.difficulty * self.volume

    @property
    def time_to_program(self) -> float:
        return self.effort / 18

    @property
    def bugs_delivered(self) -> float:
        return self.volume / 3000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n1": self.n1,
            "n2": self.n2,
            "N1": self.N1,
            "N2": self.N2,
            "vocabulary": self.vocabulary,
            "length": self.length,
            "calculated_length": round(self.calculated_length, 2),
            "volume": round(self.volume, 2),
            "difficulty": round(self.difficulty, 2),
            "effort": round(self.effort, 2),
            "time_to_program": round(self.time_to_program, 2),
            "bugs_delivered": round(self.bugs_delivered, 4),
        }


@dataclass
class FunctionComplexity:
    """Complexity metrics for a function."""
    name: str
    file: str
    line: int
    end_line: int
    cyclomatic: int
    cognitive: int
    nesting_depth: int
    lines_of_code: int
    parameter_count: int
    halstead: HalsteadMetrics
    level: ComplexityLevel

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "file": self.file,
            "line": self.line,
            "end_line": self.end_line,
            "cyclomatic": self.cyclomatic,
            "cognitive": self.cognitive,
            "nesting_depth": self.nesting_depth,
            "lines_of_code": self.lines_of_code,
            "parameter_count": self.parameter_count,
            "halstead": self.halstead.to_dict(),
            "level": self.level.value,
        }


@dataclass
class ModuleComplexity:
    """Complexity metrics for a module/file."""
    file: str
    lines_of_code: int
    lines_of_comments: int
    blank_lines: int
    functions: List[FunctionComplexity]
    average_cyclomatic: float
    average_cognitive: float
    max_cyclomatic: int
    max_cognitive: int
    maintainability_index: float
    level: ComplexityLevel

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file": self.file,
            "lines_of_code": self.lines_of_code,
            "lines_of_comments": self.lines_of_comments,
            "blank_lines": self.blank_lines,
            "functions": [f.to_dict() for f in self.functions],
            "average_cyclomatic": round(self.average_cyclomatic, 2),
            "average_cognitive": round(self.average_cognitive, 2),
            "max_cyclomatic": self.max_cyclomatic,
            "max_cognitive": self.max_cognitive,
            "maintainability_index": round(self.maintainability_index, 2),
            "level": self.level.value,
        }


@dataclass
class ComplexityMetrics:
    """Aggregated complexity metrics."""
    total_files: int
    total_functions: int
    total_lines: int
    average_cyclomatic: float
    average_cognitive: float
    average_maintainability: float
    high_complexity_functions: int
    very_high_complexity_functions: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "average_cyclomatic": round(self.average_cyclomatic, 2),
            "average_cognitive": round(self.average_cognitive, 2),
            "average_maintainability": round(self.average_maintainability, 2),
        }


@dataclass
class ComplexityResult:
    """Result from complexity analysis."""
    analysis_id: str
    modules: List[ModuleComplexity]
    metrics: ComplexityMetrics
    hotspots: List[FunctionComplexity]
    duration_ms: float
    analyzed_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analysis_id": self.analysis_id,
            "modules": [m.to_dict() for m in self.modules],
            "metrics": self.metrics.to_dict(),
            "hotspots": [h.to_dict() for h in self.hotspots],
            "duration_ms": round(self.duration_ms, 2),
            "analyzed_at": self.analyzed_at,
        }

    def to_markdown(self) -> str:
        """Convert to markdown report."""
        lines = [
            "# Complexity Analysis Report",
            "",
            f"**Analysis ID:** {self.analysis_id}",
            f"**Analyzed:** {self.analyzed_at}",
            "",
            "## Summary",
            "",
            f"- **Files Analyzed:** {self.metrics.total_files}",
            f"- **Functions Analyzed:** {self.metrics.total_functions}",
            f"- **Total Lines:** {self.metrics.total_lines:,}",
            f"- **Avg Cyclomatic Complexity:** {self.metrics.average_cyclomatic:.2f}",
            f"- **Avg Cognitive Complexity:** {self.metrics.average_cognitive:.2f}",
            f"- **Avg Maintainability Index:** {self.metrics.average_maintainability:.1f}",
            "",
        ]

        if self.metrics.high_complexity_functions > 0:
            lines.extend([
                "## Complexity Warnings",
                "",
                f"- High complexity functions: {self.metrics.high_complexity_functions}",
                f"- Very high complexity functions: {self.metrics.very_high_complexity_functions}",
                "",
            ])

        if self.hotspots:
            lines.extend([
                "## Complexity Hotspots",
                "",
                "| Function | File | CC | Cognitive | Level |",
                "|----------|------|---:|----------:|-------|",
            ])
            for h in self.hotspots[:10]:
                lines.append(
                    f"| `{h.name}` | `{h.file}:{h.line}` | {h.cyclomatic} | "
                    f"{h.cognitive} | {h.level.value} |"
                )
            lines.append("")

        lines.extend([
            "",
            "_Generated by Complexity Analyzer_",
        ])

        return "\n".join(lines)


# ============================================================================
# AST Visitors
# ============================================================================

class CyclomaticComplexityVisitor(ast.NodeVisitor):
    """Calculates cyclomatic complexity."""

    def __init__(self):
        self.complexity = 1  # Base complexity

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.complexity += 1
        self.complexity += len(node.ifs)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.complexity += len(node.values) - 1
        self.generic_visit(node)


class CognitiveComplexityVisitor(ast.NodeVisitor):
    """Calculates cognitive complexity."""

    def __init__(self):
        self.complexity = 0
        self.nesting = 0

    def _increment(self, increment: int = 1) -> None:
        self.complexity += increment + self.nesting

    def visit_If(self, node: ast.If) -> None:
        self._increment()
        self.nesting += 1
        self.generic_visit(node)
        self.nesting -= 1

    def visit_For(self, node: ast.For) -> None:
        self._increment()
        self.nesting += 1
        self.generic_visit(node)
        self.nesting -= 1

    def visit_While(self, node: ast.While) -> None:
        self._increment()
        self.nesting += 1
        self.generic_visit(node)
        self.nesting -= 1

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self._increment()
        self.nesting += 1
        self.generic_visit(node)
        self.nesting -= 1

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self._increment()
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Nested functions add complexity
        if self.nesting > 0:
            self._increment()
        self.nesting += 1
        self.generic_visit(node)
        self.nesting -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if self.nesting > 0:
            self._increment()
        self.nesting += 1
        self.generic_visit(node)
        self.nesting -= 1


class HalsteadVisitor(ast.NodeVisitor):
    """Collects Halstead metrics."""

    OPERATORS = {
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
        ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
        ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
        ast.In, ast.NotIn,
    }

    def __init__(self):
        self.operators: Set[str] = set()
        self.operands: Set[str] = set()
        self.operator_count = 0
        self.operand_count = 0

    def visit_BinOp(self, node: ast.BinOp) -> None:
        op_name = type(node.op).__name__
        self.operators.add(op_name)
        self.operator_count += 1
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        op_name = type(node.op).__name__
        self.operators.add(op_name)
        self.operator_count += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        op_name = type(node.op).__name__
        self.operators.add(op_name)
        self.operator_count += len(node.values) - 1
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            op_name = type(op).__name__
            self.operators.add(op_name)
            self.operator_count += 1
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.operands.add(node.id)
        self.operand_count += 1
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        self.operands.add(repr(node.value))
        self.operand_count += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.operators.add("call")
        self.operator_count += 1
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.operators.add("assign")
        self.operator_count += 1
        self.generic_visit(node)

    def get_metrics(self) -> HalsteadMetrics:
        return HalsteadMetrics(
            n1=len(self.operators),
            n2=len(self.operands),
            N1=self.operator_count,
            N2=self.operand_count,
        )


class NestingDepthVisitor(ast.NodeVisitor):
    """Calculates maximum nesting depth."""

    def __init__(self):
        self.max_depth = 0
        self.current_depth = 0

    def _enter_block(self, node: ast.AST) -> None:
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        self._enter_block(node)

    def visit_For(self, node: ast.For) -> None:
        self._enter_block(node)

    def visit_While(self, node: ast.While) -> None:
        self._enter_block(node)

    def visit_With(self, node: ast.With) -> None:
        self._enter_block(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._enter_block(node)


# ============================================================================
# Complexity Analyzer
# ============================================================================

class ComplexityAnalyzer:
    """
    Analyzes code complexity using multiple metrics.

    Example:
        analyzer = ComplexityAnalyzer()
        result = await analyzer.analyze(["/path/to/file.py"])
        print(result.to_markdown())
    """

    BUS_TOPICS = {
        "analyze": "review.complexity.analyze",
        "metrics": "review.complexity.metrics",
    }

    def __init__(
        self,
        config: Optional[ComplexityConfig] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the complexity analyzer.

        Args:
            config: Complexity configuration
            bus_path: Path to event bus file
        """
        self.config = config or ComplexityConfig()
        self.bus_path = bus_path or self._get_bus_path()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "complexity") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "complexity-analyzer",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _get_complexity_level(
        self,
        cyclomatic: int,
        cognitive: int,
    ) -> ComplexityLevel:
        """Determine complexity level."""
        if (cyclomatic >= self.config.cyclomatic_threshold_high or
                cognitive >= self.config.cognitive_threshold_high):
            return ComplexityLevel.VERY_HIGH
        elif (cyclomatic >= self.config.cyclomatic_threshold_moderate or
              cognitive >= self.config.cognitive_threshold_moderate):
            return ComplexityLevel.HIGH
        elif (cyclomatic >= self.config.cyclomatic_threshold_low or
              cognitive >= self.config.cognitive_threshold_low):
            return ComplexityLevel.MODERATE
        return ComplexityLevel.LOW

    def _calculate_maintainability_index(
        self,
        halstead_volume: float,
        cyclomatic: float,
        loc: int,
    ) -> float:
        """
        Calculate Maintainability Index.

        MI = 171 - 5.2 * ln(V) - 0.23 * G - 16.2 * ln(LOC)
        Normalized to 0-100 scale.
        """
        if halstead_volume <= 0 or loc <= 0:
            return 100.0

        mi = 171 - 5.2 * math.log(halstead_volume) - 0.23 * cyclomatic - 16.2 * math.log(loc)
        # Normalize to 0-100
        mi = max(0, (mi * 100) / 171)
        return mi

    async def analyze(
        self,
        files: List[str],
        content_map: Optional[Dict[str, str]] = None,
    ) -> ComplexityResult:
        """
        Analyze code complexity.

        Args:
            files: List of file paths to analyze
            content_map: Optional pre-loaded file contents

        Returns:
            ComplexityResult with all metrics

        Emits:
            review.complexity.analyze
            review.complexity.metrics
        """
        analysis_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        self._emit_event(self.BUS_TOPICS["analyze"], {
            "analysis_id": analysis_id,
            "file_count": len(files),
            "status": "started",
        })

        modules: List[ModuleComplexity] = []
        all_functions: List[FunctionComplexity] = []

        for file_path in files:
            if not file_path.endswith(".py"):
                continue

            # Get content
            if content_map and file_path in content_map:
                content = content_map[file_path]
            else:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except (IOError, OSError):
                    continue

            module = self._analyze_module(file_path, content)
            if module:
                modules.append(module)
                all_functions.extend(module.functions)

        # Calculate aggregated metrics
        total_cyclomatic = sum(f.cyclomatic for f in all_functions)
        total_cognitive = sum(f.cognitive for f in all_functions)
        total_maintainability = sum(m.maintainability_index for m in modules)

        metrics = ComplexityMetrics(
            total_files=len(modules),
            total_functions=len(all_functions),
            total_lines=sum(m.lines_of_code for m in modules),
            average_cyclomatic=total_cyclomatic / len(all_functions) if all_functions else 0,
            average_cognitive=total_cognitive / len(all_functions) if all_functions else 0,
            average_maintainability=total_maintainability / len(modules) if modules else 100,
            high_complexity_functions=sum(
                1 for f in all_functions if f.level == ComplexityLevel.HIGH
            ),
            very_high_complexity_functions=sum(
                1 for f in all_functions if f.level == ComplexityLevel.VERY_HIGH
            ),
        )

        # Find hotspots (highest complexity functions)
        hotspots = sorted(
            all_functions,
            key=lambda f: (f.cyclomatic + f.cognitive),
            reverse=True,
        )[:20]

        result = ComplexityResult(
            analysis_id=analysis_id,
            modules=modules,
            metrics=metrics,
            hotspots=hotspots,
            duration_ms=(time.time() - start_time) * 1000,
            analyzed_at=datetime.now(timezone.utc).isoformat() + "Z",
        )

        self._emit_event(self.BUS_TOPICS["metrics"], {
            "analysis_id": analysis_id,
            "metrics": metrics.to_dict(),
        })

        self._emit_event(self.BUS_TOPICS["analyze"], {
            "analysis_id": analysis_id,
            "status": "completed",
            "duration_ms": result.duration_ms,
        })

        return result

    def _analyze_module(self, file_path: str, content: str) -> Optional[ModuleComplexity]:
        """Analyze a single module."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None

        lines = content.split("\n")
        loc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
        comment_lines = len([l for l in lines if l.strip().startswith("#")])
        blank_lines = len([l for l in lines if not l.strip()])

        functions: List[FunctionComplexity] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = self._analyze_function(node, file_path)
                functions.append(func)

        avg_cyclomatic = sum(f.cyclomatic for f in functions) / len(functions) if functions else 0
        avg_cognitive = sum(f.cognitive for f in functions) / len(functions) if functions else 0
        max_cyclomatic = max((f.cyclomatic for f in functions), default=0)
        max_cognitive = max((f.cognitive for f in functions), default=0)

        # Calculate module-level Halstead and MI
        halstead_visitor = HalsteadVisitor()
        halstead_visitor.visit(tree)
        halstead = halstead_visitor.get_metrics()

        mi = self._calculate_maintainability_index(halstead.volume, avg_cyclomatic, loc)

        level = self._get_complexity_level(int(avg_cyclomatic), int(avg_cognitive))
        if mi < self.config.maintainability_threshold:
            level = ComplexityLevel.HIGH

        return ModuleComplexity(
            file=file_path,
            lines_of_code=loc,
            lines_of_comments=comment_lines,
            blank_lines=blank_lines,
            functions=functions,
            average_cyclomatic=avg_cyclomatic,
            average_cognitive=avg_cognitive,
            max_cyclomatic=max_cyclomatic,
            max_cognitive=max_cognitive,
            maintainability_index=mi,
            level=level,
        )

    def _analyze_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
    ) -> FunctionComplexity:
        """Analyze a single function."""
        # Cyclomatic complexity
        cc_visitor = CyclomaticComplexityVisitor()
        cc_visitor.visit(node)

        # Cognitive complexity
        cog_visitor = CognitiveComplexityVisitor()
        cog_visitor.visit(node)

        # Halstead metrics
        hal_visitor = HalsteadVisitor()
        hal_visitor.visit(node)

        # Nesting depth
        nest_visitor = NestingDepthVisitor()
        nest_visitor.visit(node)

        # Parameter count
        params = node.args
        param_count = (
            len(params.args) +
            len(params.posonlyargs) +
            len(params.kwonlyargs) +
            (1 if params.vararg else 0) +
            (1 if params.kwarg else 0)
        )
        # Don't count self/cls
        if param_count > 0 and params.args and params.args[0].arg in ("self", "cls"):
            param_count -= 1

        loc = (node.end_lineno or node.lineno) - node.lineno + 1
        level = self._get_complexity_level(cc_visitor.complexity, cog_visitor.complexity)

        return FunctionComplexity(
            name=node.name,
            file=file_path,
            line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            cyclomatic=cc_visitor.complexity,
            cognitive=cog_visitor.complexity,
            nesting_depth=nest_visitor.max_depth,
            lines_of_code=loc,
            parameter_count=param_count,
            halstead=hal_visitor.get_metrics(),
            level=level,
        )


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Complexity Analyzer."""
    import argparse

    parser = argparse.ArgumentParser(description="Code Complexity Analyzer (Step 173)")
    parser.add_argument("files", nargs="+", help="Files to analyze")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown")
    parser.add_argument("--threshold", type=int, default=10,
                        help="Cyclomatic complexity threshold")

    args = parser.parse_args()

    config = ComplexityConfig(
        cyclomatic_threshold_moderate=args.threshold,
    )

    analyzer = ComplexityAnalyzer(config)
    result = asyncio.run(analyzer.analyze(args.files))

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.markdown:
        print(result.to_markdown())
    else:
        print(f"Complexity Analysis: {result.analysis_id}")
        print(f"  Files: {result.metrics.total_files}")
        print(f"  Functions: {result.metrics.total_functions}")
        print(f"  Avg Cyclomatic: {result.metrics.average_cyclomatic:.2f}")
        print(f"  Avg Cognitive: {result.metrics.average_cognitive:.2f}")
        print(f"  Avg Maintainability: {result.metrics.average_maintainability:.1f}")
        print(f"\nHotspots:")
        for h in result.hotspots[:5]:
            print(f"  {h.name} ({h.file}:{h.line}): CC={h.cyclomatic}, Cog={h.cognitive}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
