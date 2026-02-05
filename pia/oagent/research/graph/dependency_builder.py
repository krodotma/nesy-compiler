#!/usr/bin/env python3
"""
dependency_builder.py - Dependency Graph Builder (Step 9)

Builds a graph of module dependencies from parsed imports.

PBTSO Phase: RESEARCH, PLAN

Bus Topics:
- research.graph.dependency
- research.imports.resolved

Protocol: DKIN v30, PAIP v16
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from ..bootstrap import AgentBus
from ..parsers.base import ParserRegistry, ImportInfo


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class DependencyNode:
    """Represents a module in the dependency graph."""

    name: str
    path: str
    imports: Set[str] = field(default_factory=set)  # Modules this imports
    imported_by: Set[str] = field(default_factory=set)  # Modules that import this
    is_external: bool = False  # Whether it's an external package
    language: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "path": self.path,
            "imports": list(self.imports),
            "imported_by": list(self.imported_by),
            "is_external": self.is_external,
            "language": self.language,
        }


@dataclass
class DependencyEdge:
    """Represents an import relationship."""

    source: str  # Importing module path
    target: str  # Imported module path
    import_names: List[str]  # Specific names imported
    is_relative: bool = False
    line: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source": self.source,
            "target": self.target,
            "import_names": self.import_names,
            "is_relative": self.is_relative,
            "line": self.line,
        }


# ============================================================================
# Dependency Graph Builder
# ============================================================================


class DependencyGraphBuilder:
    """
    Build a dependency graph from parsed imports.

    Tracks which modules import which other modules and resolves
    import paths to actual file paths where possible.

    Example:
        builder = DependencyGraphBuilder(root="/project")
        builder.add_imports_from_file("/project/src/main.py")

        for node in builder.get_nodes():
            print(f"{node.name}: imports {len(node.imports)} modules")
    """

    # Standard library modules (partial list for Python)
    PYTHON_STDLIB = {
        "abc", "argparse", "ast", "asyncio", "base64", "collections",
        "concurrent", "contextlib", "copy", "csv", "dataclasses", "datetime",
        "decimal", "enum", "functools", "glob", "hashlib", "heapq", "html",
        "http", "importlib", "inspect", "io", "itertools", "json", "logging",
        "math", "mmap", "multiprocessing", "operator", "os", "pathlib",
        "pickle", "platform", "pprint", "queue", "random", "re", "shutil",
        "signal", "socket", "sqlite3", "ssl", "statistics", "string",
        "struct", "subprocess", "sys", "tempfile", "threading", "time",
        "timeit", "traceback", "typing", "unittest", "urllib", "uuid",
        "warnings", "weakref", "xml", "zipfile",
    }

    def __init__(
        self,
        root: Optional[Path] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the dependency graph builder.

        Args:
            root: Root directory of the project
            bus: AgentBus for event emission
        """
        self.root = Path(root) if root else Path.cwd()
        self.bus = bus or AgentBus()

        self.nodes: Dict[str, DependencyNode] = {}
        self.edges: List[DependencyEdge] = []
        self._resolved_cache: Dict[str, Optional[str]] = {}

    def add_import(
        self,
        source_path: str,
        import_info: ImportInfo,
    ) -> None:
        """
        Add an import relationship to the graph.

        Args:
            source_path: Path of the importing file
            import_info: Import information from parser
        """
        source_rel = self._relative_path(source_path)

        # Ensure source node exists
        if source_rel not in self.nodes:
            self.nodes[source_rel] = DependencyNode(
                name=self._path_to_module(source_rel),
                path=source_rel,
                language=self._detect_language(source_rel),
            )

        # Resolve import target
        target_path = self._resolve_import(
            import_info.module,
            source_path,
            import_info.is_relative,
            import_info.level,
        )

        if target_path:
            target_rel = self._relative_path(target_path)
        else:
            # External or unresolved import
            target_rel = import_info.module

        # Check if external
        is_external = self._is_external_module(import_info.module)

        # Ensure target node exists
        if target_rel not in self.nodes:
            self.nodes[target_rel] = DependencyNode(
                name=import_info.module,
                path=target_rel,
                is_external=is_external,
                language="python" if not is_external else "external",
            )

        # Add edges
        self.nodes[source_rel].imports.add(target_rel)
        self.nodes[target_rel].imported_by.add(source_rel)

        # Record edge
        self.edges.append(DependencyEdge(
            source=source_rel,
            target=target_rel,
            import_names=import_info.names,
            is_relative=import_info.is_relative,
            line=import_info.line,
        ))

    def add_imports_from_file(self, file_path: str) -> int:
        """
        Add all imports from a file to the graph.

        Args:
            file_path: Path to the source file

        Returns:
            Number of imports added
        """
        result = ParserRegistry.parse_file(file_path)
        if result is None or not result.success:
            return 0

        count = 0
        for import_info in result.imports:
            self.add_import(file_path, import_info)
            count += 1

        return count

    def add_imports_from_parse_result(
        self,
        file_path: str,
        imports: List[ImportInfo],
    ) -> int:
        """
        Add imports from a parse result.

        Args:
            file_path: Path to the source file
            imports: List of ImportInfo from parser

        Returns:
            Number of imports added
        """
        count = 0
        for import_info in imports:
            self.add_import(file_path, import_info)
            count += 1
        return count

    def get_nodes(self) -> List[DependencyNode]:
        """Get all nodes in the graph."""
        return list(self.nodes.values())

    def get_edges(self) -> List[DependencyEdge]:
        """Get all edges in the graph."""
        return self.edges

    def get_node(self, path: str) -> Optional[DependencyNode]:
        """Get a node by path."""
        rel_path = self._relative_path(path)
        return self.nodes.get(rel_path)

    def get_dependencies(self, path: str) -> List[str]:
        """Get all dependencies of a module (what it imports)."""
        node = self.get_node(path)
        return list(node.imports) if node else []

    def get_dependents(self, path: str) -> List[str]:
        """Get all dependents of a module (what imports it)."""
        node = self.get_node(path)
        return list(node.imported_by) if node else []

    def get_import_chain(
        self,
        source: str,
        target: str,
        max_depth: int = 10,
    ) -> List[List[str]]:
        """
        Find all import chains from source to target.

        Args:
            source: Source module path
            target: Target module path
            max_depth: Maximum search depth

        Returns:
            List of import chains (each chain is a list of paths)
        """
        source_rel = self._relative_path(source)
        target_rel = self._relative_path(target)

        chains = []
        visited = set()

        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return

            if current == target_rel:
                chains.append(path.copy())
                return

            if current in visited:
                return

            visited.add(current)
            node = self.nodes.get(current)
            if node:
                for next_path in node.imports:
                    path.append(next_path)
                    dfs(next_path, path, depth + 1)
                    path.pop()
            visited.remove(current)

        dfs(source_rel, [source_rel], 0)
        return chains

    def get_circular_dependencies(self) -> List[List[str]]:
        """Find all circular dependencies in the graph."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> bool:
            visited.add(node)
            rec_stack.add(node)

            dep_node = self.nodes.get(node)
            if dep_node:
                for neighbor in dep_node.imports:
                    if neighbor not in visited:
                        path.append(neighbor)
                        if dfs(neighbor, path):
                            return True
                        path.pop()
                    elif neighbor in rec_stack:
                        # Found cycle
                        cycle_start = path.index(neighbor) if neighbor in path else 0
                        cycles.append(path[cycle_start:] + [neighbor])

            rec_stack.remove(node)
            return False

        for node in self.nodes:
            if node not in visited:
                dfs(node, [node])

        return cycles

    def get_external_dependencies(self) -> List[DependencyNode]:
        """Get all external (third-party) dependencies."""
        return [n for n in self.nodes.values() if n.is_external]

    def get_internal_modules(self) -> List[DependencyNode]:
        """Get all internal modules."""
        return [n for n in self.nodes.values() if not n.is_external]

    def to_cypher(self) -> List[str]:
        """
        Generate Cypher queries for FalkorDB.

        Returns:
            List of Cypher query strings
        """
        queries = []

        # Create module nodes
        for node in self.nodes.values():
            queries.append(
                f"MERGE (n:Module {{path: '{node.path}', name: '{node.name}', "
                f"is_external: {str(node.is_external).lower()}, language: '{node.language}'}})"
            )

        # Create import relationships
        for edge in self.edges:
            queries.append(
                f"MATCH (a:Module {{path: '{edge.source}'}}), (b:Module {{path: '{edge.target}'}}) "
                f"MERGE (a)-[:IMPORTS {{line: {edge.line}, is_relative: {str(edge.is_relative).lower()}}}]->(b)"
            )

        return queries

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "stats": {
                "total_modules": len(self.nodes),
                "internal_modules": len(self.get_internal_modules()),
                "external_dependencies": len(self.get_external_dependencies()),
                "total_edges": len(self.edges),
            }
        }

    def emit_graph_event(self) -> None:
        """Emit graph update event to bus."""
        self.bus.emit({
            "topic": "research.graph.dependency",
            "kind": "graph",
            "data": {
                "modules": len(self.nodes),
                "edges": len(self.edges),
                "external": len(self.get_external_dependencies()),
            }
        })

    def _relative_path(self, path: str) -> str:
        """Convert path to relative path from root."""
        try:
            return str(Path(path).relative_to(self.root))
        except ValueError:
            return path

    def _path_to_module(self, path: str) -> str:
        """Convert file path to module name."""
        # Remove extension and convert to dotted notation
        module = str(Path(path).with_suffix(""))
        return module.replace("/", ".").replace("\\", ".")

    def _detect_language(self, path: str) -> str:
        """Detect language from file extension."""
        ext = Path(path).suffix.lower()
        lang_map = {
            ".py": "python",
            ".pyi": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
        }
        return lang_map.get(ext, "unknown")

    def _is_external_module(self, module: str) -> bool:
        """Check if a module is external (not part of project)."""
        # Check stdlib
        base_module = module.split(".")[0]
        if base_module in self.PYTHON_STDLIB:
            return True

        # Check if resolvable in project
        cache_key = f"external:{module}"
        if cache_key in self._resolved_cache:
            return self._resolved_cache[cache_key] is None

        # Try to resolve
        resolved = self._resolve_import(module, str(self.root / "__dummy__.py"), False, 0)
        self._resolved_cache[cache_key] = resolved
        return resolved is None

    def _resolve_import(
        self,
        module: str,
        source_path: str,
        is_relative: bool,
        level: int,
    ) -> Optional[str]:
        """
        Resolve an import to an actual file path.

        Args:
            module: Module name (e.g., "foo.bar")
            source_path: Path of the importing file
            is_relative: Whether it's a relative import
            level: Number of dots for relative imports

        Returns:
            Resolved file path or None
        """
        cache_key = f"{source_path}:{module}:{is_relative}:{level}"
        if cache_key in self._resolved_cache:
            return self._resolved_cache[cache_key]

        source = Path(source_path)

        if is_relative:
            # Relative import
            base = source.parent
            for _ in range(level - 1):
                base = base.parent

            if module:
                parts = module.split(".")
                candidate = base / "/".join(parts)
            else:
                candidate = base
        else:
            # Absolute import
            parts = module.split(".")
            candidate = self.root / "/".join(parts)

        # Try different extensions
        for ext in [".py", ".pyi", "/__init__.py", "/__init__.pyi"]:
            full_path = str(candidate) + ext
            if Path(full_path).exists():
                self._resolved_cache[cache_key] = full_path
                return full_path

        self._resolved_cache[cache_key] = None
        return None


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Dependency Graph Builder."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Dependency Graph Builder (Step 9)"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to analyze"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--cypher",
        action="store_true",
        help="Output as Cypher queries"
    )
    parser.add_argument(
        "--circular",
        action="store_true",
        help="Find circular dependencies"
    )

    args = parser.parse_args()

    from ..scanner import CodebaseScanner

    root = Path(args.root).resolve()
    builder = DependencyGraphBuilder(root=root)

    # Scan and parse Python files
    scanner = CodebaseScanner(root)
    for file_info in scanner.scan():
        if file_info.ext == ".py":
            builder.add_imports_from_file(file_info.abs_path)

    if args.cypher:
        for query in builder.to_cypher():
            print(query)
    elif args.json:
        print(json.dumps(builder.to_dict(), indent=2))
    elif args.circular:
        cycles = builder.get_circular_dependencies()
        if cycles:
            print(f"Found {len(cycles)} circular dependencies:")
            for cycle in cycles:
                print(f"  {' -> '.join(cycle)}")
        else:
            print("No circular dependencies found")
    else:
        print(f"Dependency Graph for: {root}")
        print(f"\n  Total Modules: {len(builder.nodes)}")
        print(f"  Internal Modules: {len(builder.get_internal_modules())}")
        print(f"  External Dependencies: {len(builder.get_external_dependencies())}")
        print(f"  Total Edges: {len(builder.edges)}")

        ext_deps = builder.get_external_dependencies()
        if ext_deps:
            print(f"\n  External Dependencies:")
            for dep in sorted(ext_deps, key=lambda d: d.name)[:20]:
                print(f"    - {dep.name}")
            if len(ext_deps) > 20:
                print(f"    ... and {len(ext_deps) - 20} more")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
