#!/usr/bin/env python3
"""
Step 109: Coverage Analyzer

Analyzes code coverage data from various sources.

PBTSO Phase: OBSERVE, VERIFY
Bus Topics:
- test.coverage.analyze (subscribes)
- telemetry.test.coverage (emits)

Dependencies: Step 106 (Test Runner Orchestrator)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree


# ============================================================================
# Data Types
# ============================================================================

class CoverageType(Enum):
    """Types of coverage metrics."""
    LINE = "line"
    BRANCH = "branch"
    FUNCTION = "function"
    STATEMENT = "statement"


@dataclass
class LineCoverage:
    """Coverage data for a single line."""
    line_number: int
    hit_count: int
    is_covered: bool
    is_partial: bool = False  # For branch coverage
    branch_info: Optional[Dict[str, Any]] = None


@dataclass
class FunctionCoverage:
    """Coverage data for a function."""
    name: str
    start_line: int
    end_line: int
    hit_count: int
    is_covered: bool


@dataclass
class BranchCoverage:
    """Coverage data for a branch point."""
    line_number: int
    branch_id: str
    type: str  # if, switch, ternary, etc.
    taken: int
    not_taken: int
    is_covered: bool


@dataclass
class FileCoverage:
    """Coverage data for a single file."""
    file_path: str
    lines: List[LineCoverage] = field(default_factory=list)
    functions: List[FunctionCoverage] = field(default_factory=list)
    branches: List[BranchCoverage] = field(default_factory=list)

    @property
    def total_lines(self) -> int:
        return len(self.lines)

    @property
    def covered_lines(self) -> int:
        return sum(1 for line in self.lines if line.is_covered)

    @property
    def line_coverage_percent(self) -> float:
        if self.total_lines == 0:
            return 0.0
        return (self.covered_lines / self.total_lines) * 100

    @property
    def total_functions(self) -> int:
        return len(self.functions)

    @property
    def covered_functions(self) -> int:
        return sum(1 for f in self.functions if f.is_covered)

    @property
    def function_coverage_percent(self) -> float:
        if self.total_functions == 0:
            return 0.0
        return (self.covered_functions / self.total_functions) * 100

    @property
    def total_branches(self) -> int:
        return len(self.branches)

    @property
    def covered_branches(self) -> int:
        return sum(1 for b in self.branches if b.is_covered)

    @property
    def branch_coverage_percent(self) -> float:
        if self.total_branches == 0:
            return 0.0
        return (self.covered_branches / self.total_branches) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "total_lines": self.total_lines,
            "covered_lines": self.covered_lines,
            "line_coverage_percent": self.line_coverage_percent,
            "total_functions": self.total_functions,
            "covered_functions": self.covered_functions,
            "function_coverage_percent": self.function_coverage_percent,
            "total_branches": self.total_branches,
            "covered_branches": self.covered_branches,
            "branch_coverage_percent": self.branch_coverage_percent,
        }


@dataclass
class CoverageData:
    """Complete coverage data for a project."""
    id: str
    timestamp: float
    source: str  # pytest-cov, istanbul, c8, etc.
    files: List[FileCoverage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_lines(self) -> int:
        return sum(f.total_lines for f in self.files)

    @property
    def covered_lines(self) -> int:
        return sum(f.covered_lines for f in self.files)

    @property
    def line_coverage_percent(self) -> float:
        if self.total_lines == 0:
            return 0.0
        return (self.covered_lines / self.total_lines) * 100

    @property
    def total_functions(self) -> int:
        return sum(f.total_functions for f in self.files)

    @property
    def covered_functions(self) -> int:
        return sum(f.covered_functions for f in self.files)

    @property
    def function_coverage_percent(self) -> float:
        if self.total_functions == 0:
            return 0.0
        return (self.covered_functions / self.total_functions) * 100

    @property
    def total_branches(self) -> int:
        return sum(f.total_branches for f in self.files)

    @property
    def covered_branches(self) -> int:
        return sum(f.covered_branches for f in self.files)

    @property
    def branch_coverage_percent(self) -> float:
        if self.total_branches == 0:
            return 0.0
        return (self.covered_branches / self.total_branches) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "source": self.source,
            "files": [f.to_dict() for f in self.files],
            "total_lines": self.total_lines,
            "covered_lines": self.covered_lines,
            "line_coverage_percent": self.line_coverage_percent,
            "total_functions": self.total_functions,
            "covered_functions": self.covered_functions,
            "function_coverage_percent": self.function_coverage_percent,
            "total_branches": self.total_branches,
            "covered_branches": self.covered_branches,
            "branch_coverage_percent": self.branch_coverage_percent,
            "metadata": self.metadata,
        }

    def get_uncovered_files(self, threshold: float = 80.0) -> List[FileCoverage]:
        """Get files below coverage threshold."""
        return [f for f in self.files if f.line_coverage_percent < threshold]

    def get_uncovered_lines(self, file_path: str) -> List[int]:
        """Get uncovered line numbers for a file."""
        for f in self.files:
            if f.file_path == file_path:
                return [line.line_number for line in f.lines if not line.is_covered]
        return []


# ============================================================================
# Coverage Parsers
# ============================================================================

class CoberturaParser:
    """Parses Cobertura XML coverage format (used by pytest-cov)."""

    def parse(self, xml_path: Path) -> CoverageData:
        """Parse Cobertura XML file."""
        files = []

        try:
            tree = ElementTree.parse(xml_path)
            root = tree.getroot()

            # Get timestamp
            timestamp = float(root.get("timestamp", time.time() * 1000)) / 1000

            # Parse packages/classes/files
            for package in root.findall(".//package"):
                for cls in package.findall("classes/class"):
                    file_coverage = self._parse_class(cls)
                    if file_coverage:
                        files.append(file_coverage)

        except Exception as e:
            pass

        return CoverageData(
            id=str(uuid.uuid4()),
            timestamp=timestamp if "timestamp" in dir() else time.time(),
            source="cobertura",
            files=files,
        )

    def _parse_class(self, cls: ElementTree.Element) -> Optional[FileCoverage]:
        """Parse a class element."""
        file_path = cls.get("filename", "")
        if not file_path:
            return None

        lines = []
        for line in cls.findall("lines/line"):
            line_num = int(line.get("number", 0))
            hits = int(line.get("hits", 0))
            branch = line.get("branch", "false") == "true"

            lines.append(LineCoverage(
                line_number=line_num,
                hit_count=hits,
                is_covered=hits > 0,
                is_partial=branch and hits > 0,
            ))

        # Parse methods as functions
        functions = []
        for method in cls.findall("methods/method"):
            name = method.get("name", "")
            line_rate = float(method.get("line-rate", 0))

            # Find line range from method's lines
            method_lines = method.findall("lines/line")
            if method_lines:
                start_line = int(method_lines[0].get("number", 0))
                end_line = int(method_lines[-1].get("number", 0))
            else:
                start_line = end_line = 0

            functions.append(FunctionCoverage(
                name=name,
                start_line=start_line,
                end_line=end_line,
                hit_count=1 if line_rate > 0 else 0,
                is_covered=line_rate > 0,
            ))

        return FileCoverage(
            file_path=file_path,
            lines=lines,
            functions=functions,
        )


class IstanbulParser:
    """Parses Istanbul/NYC JSON coverage format (used by Vitest, Jest)."""

    def parse(self, json_path: Path) -> CoverageData:
        """Parse Istanbul JSON file."""
        files = []

        try:
            with open(json_path) as f:
                data = json.load(f)

            for file_path, file_data in data.items():
                file_coverage = self._parse_file(file_path, file_data)
                if file_coverage:
                    files.append(file_coverage)

        except Exception as e:
            pass

        return CoverageData(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            source="istanbul",
            files=files,
        )

    def _parse_file(self, file_path: str, data: Dict[str, Any]) -> Optional[FileCoverage]:
        """Parse file coverage data."""
        lines = []
        functions = []
        branches = []

        # Parse statement map (lines)
        statement_map = data.get("statementMap", {})
        statement_hits = data.get("s", {})

        for stmt_id, stmt_info in statement_map.items():
            start = stmt_info.get("start", {})
            line_num = start.get("line", 0)
            hits = statement_hits.get(stmt_id, 0)

            lines.append(LineCoverage(
                line_number=line_num,
                hit_count=hits,
                is_covered=hits > 0,
            ))

        # Parse function map
        fn_map = data.get("fnMap", {})
        fn_hits = data.get("f", {})

        for fn_id, fn_info in fn_map.items():
            name = fn_info.get("name", f"fn_{fn_id}")
            loc = fn_info.get("loc", {})
            start_line = loc.get("start", {}).get("line", 0)
            end_line = loc.get("end", {}).get("line", 0)
            hits = fn_hits.get(fn_id, 0)

            functions.append(FunctionCoverage(
                name=name,
                start_line=start_line,
                end_line=end_line,
                hit_count=hits,
                is_covered=hits > 0,
            ))

        # Parse branch map
        branch_map = data.get("branchMap", {})
        branch_hits = data.get("b", {})

        for branch_id, branch_info in branch_map.items():
            branch_type = branch_info.get("type", "unknown")
            loc = branch_info.get("loc", {})
            line_num = loc.get("start", {}).get("line", 0)
            hits = branch_hits.get(branch_id, [0, 0])

            branches.append(BranchCoverage(
                line_number=line_num,
                branch_id=branch_id,
                type=branch_type,
                taken=hits[0] if len(hits) > 0 else 0,
                not_taken=hits[1] if len(hits) > 1 else 0,
                is_covered=sum(hits) > 0,
            ))

        return FileCoverage(
            file_path=file_path,
            lines=lines,
            functions=functions,
            branches=branches,
        )


class CoveragePyParser:
    """Parses coverage.py SQLite database."""

    def parse(self, db_path: Path) -> CoverageData:
        """Parse coverage.py database."""
        files = []

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get files
            cursor.execute("SELECT id, path FROM file")
            file_rows = cursor.fetchall()

            for file_id, file_path in file_rows:
                # Get line data
                cursor.execute(
                    "SELECT numbits FROM line_bits WHERE file_id = ?",
                    (file_id,)
                )
                result = cursor.fetchone()

                if result:
                    # numbits is a bit array of covered lines
                    # Would need proper parsing of the binary data
                    pass

                # For now, create placeholder
                files.append(FileCoverage(file_path=file_path))

            conn.close()

        except Exception as e:
            pass

        return CoverageData(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            source="coverage.py",
            files=files,
        )


# ============================================================================
# Coverage Analyzer
# ============================================================================

class CoverageAnalyzer:
    """
    Analyzes code coverage data from various sources.

    PBTSO Phase: OBSERVE, VERIFY
    Bus Topics: test.coverage.analyze, telemetry.test.coverage
    """

    BUS_TOPICS = {
        "analyze": "test.coverage.analyze",
        "telemetry": "telemetry.test.coverage",
    }

    def __init__(self, bus=None):
        """
        Initialize the coverage analyzer.

        Args:
            bus: Optional bus instance for event emission
        """
        self.bus = bus
        self._parsers = {
            "cobertura": CoberturaParser(),
            "istanbul": IstanbulParser(),
            "coverage.py": CoveragePyParser(),
        }

    def analyze(
        self,
        coverage_path: str,
        source_type: Optional[str] = None,
        threshold: float = 80.0,
    ) -> CoverageData:
        """
        Analyze coverage data from a file.

        Args:
            coverage_path: Path to coverage file
            source_type: Type of coverage data (auto-detected if not provided)
            threshold: Coverage threshold for analysis

        Returns:
            CoverageData with parsed results
        """
        path = Path(coverage_path)

        # Emit analyze start event
        self._emit_event("analyze", {
            "coverage_path": coverage_path,
            "status": "started",
        })

        # Auto-detect source type
        if source_type is None:
            source_type = self._detect_source_type(path)

        # Parse coverage data
        parser = self._parsers.get(source_type)
        if parser is None:
            raise ValueError(f"Unknown coverage source type: {source_type}")

        data = parser.parse(path)

        # Add analysis metadata
        data.metadata["threshold"] = threshold
        data.metadata["files_below_threshold"] = len(data.get_uncovered_files(threshold))
        data.metadata["analysis_timestamp"] = time.time()

        # Emit telemetry
        self._emit_event("telemetry", {
            "coverage_id": data.id,
            "line_coverage": data.line_coverage_percent,
            "function_coverage": data.function_coverage_percent,
            "branch_coverage": data.branch_coverage_percent,
            "total_files": len(data.files),
            "files_below_threshold": data.metadata["files_below_threshold"],
        })

        return data

    def _detect_source_type(self, path: Path) -> str:
        """Auto-detect coverage source type from file."""
        suffix = path.suffix.lower()
        name = path.name.lower()

        if suffix == ".xml":
            # Check if Cobertura
            try:
                tree = ElementTree.parse(path)
                root = tree.getroot()
                if root.tag == "coverage" and root.get("version"):
                    return "cobertura"
            except:
                pass
            return "cobertura"  # Default for XML

        elif suffix == ".json":
            return "istanbul"

        elif name == ".coverage" or suffix == ".coverage":
            return "coverage.py"

        return "cobertura"  # Default

    def compare(
        self,
        current: CoverageData,
        baseline: CoverageData,
    ) -> Dict[str, Any]:
        """
        Compare two coverage datasets.

        Args:
            current: Current coverage data
            baseline: Baseline coverage data to compare against

        Returns:
            Dictionary with comparison results
        """
        # Calculate deltas
        line_delta = current.line_coverage_percent - baseline.line_coverage_percent
        function_delta = current.function_coverage_percent - baseline.function_coverage_percent
        branch_delta = current.branch_coverage_percent - baseline.branch_coverage_percent

        # Find new uncovered files
        baseline_files = {f.file_path for f in baseline.files}
        current_files = {f.file_path for f in current.files}

        new_files = current_files - baseline_files
        removed_files = baseline_files - current_files

        # Find regressions
        regressions = []
        improvements = []

        for curr_file in current.files:
            for base_file in baseline.files:
                if curr_file.file_path == base_file.file_path:
                    delta = curr_file.line_coverage_percent - base_file.line_coverage_percent
                    if delta < -5:  # 5% threshold
                        regressions.append({
                            "file": curr_file.file_path,
                            "current": curr_file.line_coverage_percent,
                            "baseline": base_file.line_coverage_percent,
                            "delta": delta,
                        })
                    elif delta > 5:
                        improvements.append({
                            "file": curr_file.file_path,
                            "current": curr_file.line_coverage_percent,
                            "baseline": base_file.line_coverage_percent,
                            "delta": delta,
                        })
                    break

        return {
            "line_coverage_delta": line_delta,
            "function_coverage_delta": function_delta,
            "branch_coverage_delta": branch_delta,
            "new_files": list(new_files),
            "removed_files": list(removed_files),
            "regressions": regressions,
            "improvements": improvements,
            "is_regression": line_delta < -1 or len(regressions) > 0,
        }

    def find_gaps(
        self,
        data: CoverageData,
        min_gap_size: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find coverage gaps (consecutive uncovered lines).

        Args:
            data: Coverage data
            min_gap_size: Minimum number of consecutive uncovered lines

        Returns:
            List of coverage gaps
        """
        gaps = []

        for file_cov in data.files:
            uncovered_lines = sorted(
                line.line_number for line in file_cov.lines if not line.is_covered
            )

            if not uncovered_lines:
                continue

            # Find consecutive sequences
            current_gap_start = uncovered_lines[0]
            current_gap_end = uncovered_lines[0]

            for line in uncovered_lines[1:]:
                if line == current_gap_end + 1:
                    current_gap_end = line
                else:
                    # Check if gap is big enough
                    gap_size = current_gap_end - current_gap_start + 1
                    if gap_size >= min_gap_size:
                        gaps.append({
                            "file": file_cov.file_path,
                            "start_line": current_gap_start,
                            "end_line": current_gap_end,
                            "size": gap_size,
                        })
                    current_gap_start = line
                    current_gap_end = line

            # Don't forget the last gap
            gap_size = current_gap_end - current_gap_start + 1
            if gap_size >= min_gap_size:
                gaps.append({
                    "file": file_cov.file_path,
                    "start_line": current_gap_start,
                    "end_line": current_gap_end,
                    "size": gap_size,
                })

        return gaps

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        if self.bus:
            topic = self.BUS_TOPICS.get(event_type, f"test.coverage.{event_type}")
            self.bus.emit({
                "topic": topic,
                "kind": "coverage_analysis",
                "actor": "test-agent",
                "data": data,
            })


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Coverage Analyzer."""
    import argparse

    parser = argparse.ArgumentParser(description="Coverage Analyzer")
    parser.add_argument("path", help="Path to coverage file")
    parser.add_argument("--type", choices=["cobertura", "istanbul", "coverage.py"])
    parser.add_argument("--threshold", type=float, default=80.0, help="Coverage threshold")
    parser.add_argument("--compare", help="Path to baseline coverage for comparison")
    parser.add_argument("--gaps", action="store_true", help="Find coverage gaps")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    analyzer = CoverageAnalyzer()

    # Analyze current coverage
    data = analyzer.analyze(args.path, source_type=args.type, threshold=args.threshold)

    if args.json:
        print(json.dumps(data.to_dict(), indent=2))
        return

    # Print summary
    print(f"\nCoverage Analysis: {args.path}")
    print(f"{'='*60}")
    print(f"Files analyzed: {len(data.files)}")
    print(f"Line coverage: {data.line_coverage_percent:.1f}%")
    print(f"Function coverage: {data.function_coverage_percent:.1f}%")
    print(f"Branch coverage: {data.branch_coverage_percent:.1f}%")

    # List files below threshold
    uncovered = data.get_uncovered_files(args.threshold)
    if uncovered:
        print(f"\nFiles below {args.threshold}% threshold:")
        for f in sorted(uncovered, key=lambda x: x.line_coverage_percent):
            print(f"  {f.line_coverage_percent:.1f}% - {f.file_path}")

    # Compare if baseline provided
    if args.compare:
        baseline = analyzer.analyze(args.compare, source_type=args.type)
        comparison = analyzer.compare(data, baseline)

        print(f"\nComparison with baseline:")
        print(f"  Line coverage delta: {comparison['line_coverage_delta']:+.1f}%")
        if comparison["regressions"]:
            print(f"  Regressions: {len(comparison['regressions'])} files")
        if comparison["improvements"]:
            print(f"  Improvements: {len(comparison['improvements'])} files")

    # Find gaps if requested
    if args.gaps:
        gaps = analyzer.find_gaps(data)
        if gaps:
            print(f"\nCoverage gaps (>= 3 lines):")
            for gap in gaps[:10]:  # Limit output
                print(f"  {gap['file']}:{gap['start_line']}-{gap['end_line']} ({gap['size']} lines)")


if __name__ == "__main__":
    main()
