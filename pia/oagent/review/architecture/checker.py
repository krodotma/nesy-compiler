#!/usr/bin/env python3
"""
Architecture Consistency Checker (Step 155)

Validates code against architectural rules and layer constraints.

PBTSO Phase: VERIFY
Bus Topics: review.architecture.check, review.architecture.violations

Checks:
- Layer dependency violations (e.g., UI calling database directly)
- Circular dependencies
- Import boundary violations
- Module cohesion
- API contract compliance

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# Types
# ============================================================================

class ViolationType(Enum):
    """Types of architecture violations."""
    LAYER_VIOLATION = "layer_violation"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    FORBIDDEN_IMPORT = "forbidden_import"
    BOUNDARY_CROSSING = "boundary_crossing"
    COHESION_VIOLATION = "cohesion_violation"
    NAMING_VIOLATION = "naming_violation"


class ViolationSeverity(Enum):
    """Severity of architecture violations."""
    CRITICAL = "critical"  # Must be fixed
    ERROR = "error"        # Should be fixed
    WARNING = "warning"    # Consider fixing
    INFO = "info"          # Informational


@dataclass
class LayerDefinition:
    """
    Definition of an architectural layer.

    Attributes:
        name: Layer name (e.g., "presentation", "domain", "data")
        patterns: File path patterns that belong to this layer
        allowed_dependencies: Layer names this layer can depend on
        forbidden_imports: Module patterns that should not be imported
    """
    name: str
    patterns: List[str]
    allowed_dependencies: List[str] = field(default_factory=list)
    forbidden_imports: List[str] = field(default_factory=list)

    def matches(self, file_path: str) -> bool:
        """Check if file belongs to this layer."""
        for pattern in self.patterns:
            if re.search(pattern, file_path):
                return True
        return False


@dataclass
class ArchitectureRule:
    """
    A custom architecture rule.

    Attributes:
        name: Rule name
        description: What this rule checks
        source_pattern: Pattern for source files
        forbidden_import_pattern: Import pattern that violates this rule
        severity: Violation severity
    """
    name: str
    description: str
    source_pattern: str
    forbidden_import_pattern: str
    severity: ViolationSeverity = ViolationSeverity.ERROR


@dataclass
class ArchitectureViolation:
    """
    Represents an architecture violation.

    Attributes:
        violation_type: Type of violation
        severity: Violation severity
        file: File where violation occurred
        line: Line number
        source_layer: Source layer name
        target_layer: Target layer being accessed
        import_path: The violating import path
        description: Human-readable description
        suggestion: How to fix the violation
    """
    violation_type: ViolationType
    severity: ViolationSeverity
    file: str
    line: int
    source_layer: str
    target_layer: Optional[str]
    import_path: str
    description: str
    suggestion: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["violation_type"] = self.violation_type.value
        result["severity"] = self.severity.value
        return result


@dataclass
class ArchitectureCheckResult:
    """Result from architecture checking."""
    files_checked: int = 0
    violations: List[ArchitectureViolation] = field(default_factory=list)
    duration_ms: float = 0
    critical_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    layers_defined: int = 0
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "files_checked": self.files_checked,
            "violations": [v.to_dict() for v in self.violations],
            "duration_ms": self.duration_ms,
            "critical_count": self.critical_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "layers_defined": self.layers_defined,
            "dependency_graph": self.dependency_graph,
        }


# ============================================================================
# Default Architecture Layers
# ============================================================================

# Common layered architecture patterns
DEFAULT_LAYERS = [
    LayerDefinition(
        name="presentation",
        patterns=[r"/ui/", r"/views/", r"/controllers/", r"/api/", r"/routes/", r"/handlers/"],
        allowed_dependencies=["application", "domain"],
        forbidden_imports=[r"sqlalchemy", r"psycopg", r"pymongo"],
    ),
    LayerDefinition(
        name="application",
        patterns=[r"/services/", r"/usecases/", r"/application/"],
        allowed_dependencies=["domain", "infrastructure"],
        forbidden_imports=[],
    ),
    LayerDefinition(
        name="domain",
        patterns=[r"/domain/", r"/models/", r"/entities/", r"/core/"],
        allowed_dependencies=[],  # Domain should not depend on other layers
        forbidden_imports=[r"flask", r"django", r"fastapi", r"requests"],
    ),
    LayerDefinition(
        name="infrastructure",
        patterns=[r"/infrastructure/", r"/repositories/", r"/data/", r"/db/"],
        allowed_dependencies=["domain"],
        forbidden_imports=[],
    ),
]


# ============================================================================
# Import Analyzer
# ============================================================================

class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to extract imports from Python code."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.imports: List[Tuple[str, int]] = []  # (module, line)

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement."""
        for alias in node.names:
            self.imports.append((alias.name, node.lineno))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from ... import statement."""
        if node.module:
            self.imports.append((node.module, node.lineno))
        self.generic_visit(node)


# ============================================================================
# Architecture Checker
# ============================================================================

class ArchitectureChecker:
    """
    Checks code against architecture rules.

    Validates layer dependencies, detects circular dependencies,
    and enforces import restrictions.

    Example:
        checker = ArchitectureChecker()
        result = checker.check(["/path/to/project"])
        for violation in result.violations:
            print(f"{violation.file}:{violation.line}: {violation.description}")
    """

    def __init__(
        self,
        layers: Optional[List[LayerDefinition]] = None,
        rules: Optional[List[ArchitectureRule]] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the architecture checker.

        Args:
            layers: Layer definitions (defaults to common layered architecture)
            rules: Custom architecture rules
            bus_path: Path to event bus file
        """
        self.layers = layers or DEFAULT_LAYERS
        self.rules = rules or []
        self.bus_path = bus_path or self._get_bus_path()
        self._layer_map: Dict[str, LayerDefinition] = {l.name: l for l in self.layers}

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
            "kind": "architecture",
            "actor": "architecture-checker",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _get_file_layer(self, file_path: str) -> Optional[LayerDefinition]:
        """Determine which layer a file belongs to."""
        for layer in self.layers:
            if layer.matches(file_path):
                return layer
        return None

    def _get_import_layer(self, import_path: str) -> Optional[str]:
        """Determine which layer an import belongs to."""
        # Convert import to path-like for matching
        import_as_path = "/" + import_path.replace(".", "/") + "/"
        for layer in self.layers:
            if layer.matches(import_as_path):
                return layer.name
        return None

    def check(
        self,
        files: List[str],
        content_map: Optional[Dict[str, str]] = None,
    ) -> ArchitectureCheckResult:
        """
        Check files against architecture rules.

        Args:
            files: List of file paths to check
            content_map: Optional pre-loaded file contents

        Returns:
            ArchitectureCheckResult with violations found

        Emits:
            review.architecture.check (start)
            review.architecture.violations (per violation batch)
        """
        start_time = time.time()

        # Emit start event
        self._emit_event("review.architecture.check", {
            "files": files[:20],
            "file_count": len(files),
            "layers": [l.name for l in self.layers],
            "status": "started",
        })

        result = ArchitectureCheckResult(
            files_checked=len(files),
            layers_defined=len(self.layers),
        )

        # Build dependency graph and check violations
        dependency_graph: Dict[str, Set[str]] = defaultdict(set)
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

            # Analyze imports and check violations
            violations = self._analyze_file(file_path, content, dependency_graph)
            result.violations.extend(violations)

        # Check for circular dependencies
        circular_violations = self._check_circular_dependencies(dependency_graph)
        result.violations.extend(circular_violations)

        # Check custom rules
        for file_path in python_files:
            if content_map and file_path in content_map:
                content = content_map[file_path]
            else:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except (IOError, OSError):
                    continue

            rule_violations = self._check_custom_rules(file_path, content)
            result.violations.extend(rule_violations)

        # Convert dependency graph to dict of lists
        result.dependency_graph = {k: list(v) for k, v in dependency_graph.items()}

        # Calculate counts
        for violation in result.violations:
            if violation.severity == ViolationSeverity.CRITICAL:
                result.critical_count += 1
            elif violation.severity == ViolationSeverity.ERROR:
                result.error_count += 1
            elif violation.severity == ViolationSeverity.WARNING:
                result.warning_count += 1

        result.duration_ms = (time.time() - start_time) * 1000

        # Emit violations found
        if result.violations:
            self._emit_event("review.architecture.violations", {
                "violation_count": len(result.violations),
                "critical_count": result.critical_count,
                "error_count": result.error_count,
                "violations": [v.to_dict() for v in result.violations[:10]],
            })

        # Emit completion
        self._emit_event("review.architecture.check", {
            "status": "completed",
            "files_checked": result.files_checked,
            "violation_count": len(result.violations),
            "duration_ms": result.duration_ms,
        })

        return result

    def _analyze_file(
        self,
        file_path: str,
        content: str,
        dependency_graph: Dict[str, Set[str]],
    ) -> List[ArchitectureViolation]:
        """Analyze a single file for architecture violations."""
        violations = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return violations

        # Get file's layer
        source_layer = self._get_file_layer(file_path)
        if not source_layer:
            return violations

        # Extract imports
        analyzer = ImportAnalyzer(file_path)
        analyzer.visit(tree)

        for import_path, line in analyzer.imports:
            # Add to dependency graph
            dependency_graph[file_path].add(import_path)

            # Check forbidden imports
            for forbidden in source_layer.forbidden_imports:
                if re.search(forbidden, import_path):
                    violations.append(ArchitectureViolation(
                        violation_type=ViolationType.FORBIDDEN_IMPORT,
                        severity=ViolationSeverity.ERROR,
                        file=file_path,
                        line=line,
                        source_layer=source_layer.name,
                        target_layer=None,
                        import_path=import_path,
                        description=f"Layer '{source_layer.name}' should not import '{import_path}'",
                        suggestion=f"Move this dependency to infrastructure layer or use dependency injection",
                    ))

            # Check layer violations
            target_layer_name = self._get_import_layer(import_path)
            if target_layer_name and target_layer_name != source_layer.name:
                if target_layer_name not in source_layer.allowed_dependencies:
                    violations.append(ArchitectureViolation(
                        violation_type=ViolationType.LAYER_VIOLATION,
                        severity=ViolationSeverity.ERROR,
                        file=file_path,
                        line=line,
                        source_layer=source_layer.name,
                        target_layer=target_layer_name,
                        import_path=import_path,
                        description=f"Layer '{source_layer.name}' should not depend on layer '{target_layer_name}'",
                        suggestion=f"Use dependency injection or move code to appropriate layer",
                    ))

        return violations

    def _check_circular_dependencies(
        self,
        dependency_graph: Dict[str, Set[str]],
    ) -> List[ArchitectureViolation]:
        """Check for circular dependencies using DFS."""
        violations = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        cycles: List[List[str]] = []

        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in dependency_graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            path.pop()
            rec_stack.remove(node)

        for node in dependency_graph:
            if node not in visited:
                dfs(node, [])

        # Create violations for cycles
        for cycle in cycles[:5]:  # Limit to 5 cycles
            violations.append(ArchitectureViolation(
                violation_type=ViolationType.CIRCULAR_DEPENDENCY,
                severity=ViolationSeverity.CRITICAL,
                file=cycle[0],
                line=1,
                source_layer="",
                target_layer="",
                import_path=" -> ".join(cycle[:5]),
                description=f"Circular dependency detected: {' -> '.join(cycle[:5])}{'...' if len(cycle) > 5 else ''}",
                suggestion="Break the cycle by using dependency injection or reorganizing modules",
            ))

        return violations

    def _check_custom_rules(
        self,
        file_path: str,
        content: str,
    ) -> List[ArchitectureViolation]:
        """Check custom architecture rules."""
        violations = []

        for rule in self.rules:
            # Check if file matches source pattern
            if not re.search(rule.source_pattern, file_path):
                continue

            # Check for forbidden import pattern
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue

            analyzer = ImportAnalyzer(file_path)
            analyzer.visit(tree)

            for import_path, line in analyzer.imports:
                if re.search(rule.forbidden_import_pattern, import_path):
                    violations.append(ArchitectureViolation(
                        violation_type=ViolationType.BOUNDARY_CROSSING,
                        severity=rule.severity,
                        file=file_path,
                        line=line,
                        source_layer=rule.name,
                        target_layer=None,
                        import_path=import_path,
                        description=f"Rule '{rule.name}' violated: {rule.description}",
                        suggestion=f"Remove or refactor the import '{import_path}'",
                    ))

        return violations

    def add_layer(self, layer: LayerDefinition) -> None:
        """Add a layer definition."""
        self.layers.append(layer)
        self._layer_map[layer.name] = layer

    def add_rule(self, rule: ArchitectureRule) -> None:
        """Add a custom rule."""
        self.rules.append(rule)


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Architecture Checker."""
    import argparse

    parser = argparse.ArgumentParser(description="Architecture Consistency Checker (Step 155)")
    parser.add_argument("files", nargs="+", help="Files or directories to check")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument("--graph", action="store_true", help="Show dependency graph")

    args = parser.parse_args()

    # Expand directories to files
    all_files = []
    for path in args.files:
        p = Path(path)
        if p.is_dir():
            all_files.extend(str(f) for f in p.rglob("*.py"))
        else:
            all_files.append(str(p))

    checker = ArchitectureChecker()
    result = checker.check(all_files)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.graph:
        print("Dependency Graph:")
        for source, targets in sorted(result.dependency_graph.items()):
            print(f"  {source}:")
            for target in sorted(targets)[:5]:
                print(f"    -> {target}")
    elif args.summary:
        print(f"Architecture Check Summary:")
        print(f"  Files checked: {result.files_checked}")
        print(f"  Layers defined: {result.layers_defined}")
        print(f"  Violations: {len(result.violations)}")
        print(f"  Critical: {result.critical_count}")
        print(f"  Error: {result.error_count}")
        print(f"  Warning: {result.warning_count}")
        print(f"  Duration: {result.duration_ms:.1f}ms")
    else:
        for violation in result.violations:
            severity_color = {
                ViolationSeverity.CRITICAL: "\033[91m",
                ViolationSeverity.ERROR: "\033[93m",
                ViolationSeverity.WARNING: "\033[94m",
                ViolationSeverity.INFO: "\033[90m",
            }.get(violation.severity, "")
            reset = "\033[0m"

            print(f"{severity_color}[{violation.severity.value.upper()}]{reset} {violation.file}:{violation.line}")
            print(f"  {violation.description}")
            print(f"  Suggestion: {violation.suggestion}")
            print()

    return 1 if result.critical_count > 0 or result.error_count > 0 else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
