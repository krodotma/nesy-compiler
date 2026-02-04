#!/usr/bin/env python3
"""
call_graph.py - Call Graph Analyzer (Step 10)

Analyzes function/method call relationships in source code.

PBTSO Phase: RESEARCH

Bus Topics:
- research.graph.calls
- research.functions.analyzed

Protocol: DKIN v30, PAIP v16
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..bootstrap import AgentBus


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class CallNode:
    """Represents a function/method in the call graph."""

    name: str
    qualified_name: str  # Full qualified name (module.class.method)
    path: str  # File path
    line: int
    calls: Set[str] = field(default_factory=set)  # Functions this calls
    called_by: Set[str] = field(default_factory=set)  # Functions that call this
    is_method: bool = False
    class_name: Optional[str] = None
    is_async: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "path": self.path,
            "line": self.line,
            "calls": list(self.calls),
            "called_by": list(self.called_by),
            "is_method": self.is_method,
            "class_name": self.class_name,
            "is_async": self.is_async,
        }


@dataclass
class CallEdge:
    """Represents a call relationship."""

    caller: str  # Qualified name of caller
    callee: str  # Name being called (may be unqualified)
    line: int  # Line number of call
    is_method_call: bool = False  # obj.method() style call
    receiver: Optional[str] = None  # Variable being called on

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "caller": self.caller,
            "callee": self.callee,
            "line": self.line,
            "is_method_call": self.is_method_call,
            "receiver": self.receiver,
        }


# ============================================================================
# Call Graph Analyzer
# ============================================================================


class CallGraphAnalyzer:
    """
    Analyze function/method call relationships.

    Builds a call graph showing which functions call which other functions.
    Supports Python AST analysis.

    Example:
        analyzer = CallGraphAnalyzer()
        result = analyzer.analyze_file("/path/to/module.py")

        for func, calls in result.items():
            print(f"{func} calls: {calls}")
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the call graph analyzer.

        Args:
            root: Root directory of the project
            bus: AgentBus for event emission
        """
        self.root = Path(root) if root else Path.cwd()
        self.bus = bus or AgentBus()

        self.nodes: Dict[str, CallNode] = {}
        self.edges: List[CallEdge] = []

    def analyze(self, tree: ast.Module, module_path: str) -> Dict[str, Set[str]]:
        """
        Analyze an AST module for call relationships.

        Args:
            tree: Parsed AST module
            module_path: Path of the source file

        Returns:
            Dict mapping function names to sets of called functions
        """
        calls: Dict[str, Set[str]] = {}
        module_name = self._path_to_module(module_path)

        # Visit all functions and methods
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._analyze_function(node, module_name, module_path, calls)

            elif isinstance(node, ast.ClassDef):
                self._analyze_class(node, module_name, module_path, calls)

        # Emit event
        self.bus.emit({
            "topic": "research.graph.calls",
            "kind": "graph",
            "data": {
                "path": module_path,
                "functions": len(calls),
                "total_calls": sum(len(c) for c in calls.values()),
            }
        })

        return calls

    def analyze_file(self, file_path: str) -> Dict[str, Set[str]]:
        """
        Analyze a file for call relationships.

        Args:
            file_path: Path to the Python file

        Returns:
            Dict mapping function names to sets of called functions
        """
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)
            return self.analyze(tree, file_path)
        except SyntaxError:
            return {}
        except Exception:
            return {}

    def analyze_directory(self, directory: str) -> Dict[str, Set[str]]:
        """
        Analyze all Python files in a directory.

        Args:
            directory: Directory path

        Returns:
            Combined call graph from all files
        """
        all_calls: Dict[str, Set[str]] = {}
        dir_path = Path(directory)

        for py_file in dir_path.rglob("*.py"):
            calls = self.analyze_file(str(py_file))
            all_calls.update(calls)

        return all_calls

    def get_nodes(self) -> List[CallNode]:
        """Get all nodes in the call graph."""
        return list(self.nodes.values())

    def get_edges(self) -> List[CallEdge]:
        """Get all edges in the call graph."""
        return self.edges

    def get_callers(self, function_name: str) -> List[str]:
        """Get all functions that call a given function."""
        node = self.nodes.get(function_name)
        return list(node.called_by) if node else []

    def get_callees(self, function_name: str) -> List[str]:
        """Get all functions that a given function calls."""
        node = self.nodes.get(function_name)
        return list(node.calls) if node else []

    def get_call_chain(
        self,
        source: str,
        target: str,
        max_depth: int = 10,
    ) -> List[List[str]]:
        """
        Find all call chains from source to target.

        Args:
            source: Source function name
            target: Target function name
            max_depth: Maximum search depth

        Returns:
            List of call chains
        """
        chains = []
        visited = set()

        def dfs(current: str, path: List[str], depth: int):
            if depth > max_depth:
                return

            if current == target:
                chains.append(path.copy())
                return

            if current in visited:
                return

            visited.add(current)
            node = self.nodes.get(current)
            if node:
                for callee in node.calls:
                    path.append(callee)
                    dfs(callee, path, depth + 1)
                    path.pop()
            visited.remove(current)

        dfs(source, [source], 0)
        return chains

    def get_recursive_functions(self) -> List[str]:
        """Find all recursive functions."""
        recursive = []
        for name, node in self.nodes.items():
            if name in node.calls:
                recursive.append(name)
        return recursive

    def get_entry_points(self) -> List[CallNode]:
        """Get functions that are never called (potential entry points)."""
        return [n for n in self.nodes.values() if not n.called_by]

    def get_leaf_functions(self) -> List[CallNode]:
        """Get functions that don't call other project functions."""
        return [n for n in self.nodes.values() if not n.calls]

    def to_cypher(self) -> List[str]:
        """Generate Cypher queries for FalkorDB."""
        queries = []

        # Create function nodes
        for node in self.nodes.values():
            queries.append(
                f"MERGE (f:Function {{name: '{node.name}', qualified_name: '{node.qualified_name}', "
                f"path: '{node.path}', line: {node.line}, is_async: {str(node.is_async).lower()}}})"
            )

        # Create call relationships
        for edge in self.edges:
            queries.append(
                f"MATCH (a:Function {{qualified_name: '{edge.caller}'}}), "
                f"(b:Function {{name: '{edge.callee}'}}) "
                f"MERGE (a)-[:CALLS {{line: {edge.line}}}]->(b)"
            )

        return queries

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "stats": {
                "total_functions": len(self.nodes),
                "total_calls": len(self.edges),
                "recursive": len(self.get_recursive_functions()),
                "entry_points": len(self.get_entry_points()),
                "leaf_functions": len(self.get_leaf_functions()),
            }
        }

    def _analyze_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        module_name: str,
        module_path: str,
        calls: Dict[str, Set[str]],
        class_name: Optional[str] = None,
    ) -> None:
        """Analyze a function definition."""
        if class_name:
            qualified_name = f"{module_name}.{class_name}.{node.name}"
        else:
            qualified_name = f"{module_name}.{node.name}"

        calls[qualified_name] = set()

        # Create node
        call_node = CallNode(
            name=node.name,
            qualified_name=qualified_name,
            path=module_path,
            line=node.lineno,
            is_method=class_name is not None,
            class_name=class_name,
            is_async=isinstance(node, ast.AsyncFunctionDef),
        )

        # Visit the function body for calls
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_info = self._extract_call_info(child)
                if call_info:
                    callee_name, is_method_call, receiver = call_info
                    calls[qualified_name].add(callee_name)
                    call_node.calls.add(callee_name)

                    # Create edge
                    self.edges.append(CallEdge(
                        caller=qualified_name,
                        callee=callee_name,
                        line=child.lineno,
                        is_method_call=is_method_call,
                        receiver=receiver,
                    ))

        self.nodes[qualified_name] = call_node

        # Update called_by for callees
        for callee in call_node.calls:
            if callee in self.nodes:
                self.nodes[callee].called_by.add(qualified_name)

    def _analyze_class(
        self,
        node: ast.ClassDef,
        module_name: str,
        module_path: str,
        calls: Dict[str, Set[str]],
    ) -> None:
        """Analyze a class definition for methods."""
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._analyze_function(
                    item,
                    module_name,
                    module_path,
                    calls,
                    class_name=node.name,
                )

    def _extract_call_info(
        self,
        node: ast.Call,
    ) -> Optional[tuple[str, bool, Optional[str]]]:
        """
        Extract call information from a Call node.

        Returns:
            Tuple of (callee_name, is_method_call, receiver) or None
        """
        func = node.func

        if isinstance(func, ast.Name):
            # Simple function call: func()
            return (func.id, False, None)

        elif isinstance(func, ast.Attribute):
            # Method call: obj.method()
            if isinstance(func.value, ast.Name):
                receiver = func.value.id
            else:
                receiver = None
            return (func.attr, True, receiver)

        elif isinstance(func, ast.Call):
            # Chained call: func()()
            # Recursively get the base
            inner = self._extract_call_info(func)
            return inner

        return None

    def _path_to_module(self, path: str) -> str:
        """Convert file path to module name."""
        try:
            rel_path = str(Path(path).relative_to(self.root))
        except ValueError:
            rel_path = path

        module = str(Path(rel_path).with_suffix(""))
        return module.replace("/", ".").replace("\\", ".")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Call Graph Analyzer."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Call Graph Analyzer (Step 10)"
    )
    parser.add_argument(
        "path",
        help="File or directory to analyze"
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
        "--recursive",
        action="store_true",
        help="Find recursive functions"
    )
    parser.add_argument(
        "--entry-points",
        action="store_true",
        help="Find entry points"
    )

    args = parser.parse_args()

    path = Path(args.path)
    analyzer = CallGraphAnalyzer(root=path.parent if path.is_file() else path)

    if path.is_file():
        analyzer.analyze_file(str(path))
    else:
        analyzer.analyze_directory(str(path))

    if args.cypher:
        for query in analyzer.to_cypher():
            print(query)
    elif args.json:
        print(json.dumps(analyzer.to_dict(), indent=2))
    elif args.recursive:
        recursive = analyzer.get_recursive_functions()
        if recursive:
            print(f"Found {len(recursive)} recursive functions:")
            for func in recursive:
                print(f"  - {func}")
        else:
            print("No recursive functions found")
    elif args.entry_points:
        entry_points = analyzer.get_entry_points()
        print(f"Found {len(entry_points)} entry points:")
        for ep in entry_points[:30]:
            print(f"  - {ep.qualified_name} ({ep.path}:{ep.line})")
        if len(entry_points) > 30:
            print(f"  ... and {len(entry_points) - 30} more")
    else:
        print(f"Call Graph Analysis for: {path}")
        stats = analyzer.to_dict()["stats"]
        print(f"\n  Total Functions: {stats['total_functions']}")
        print(f"  Total Call Edges: {stats['total_calls']}")
        print(f"  Recursive Functions: {stats['recursive']}")
        print(f"  Entry Points: {stats['entry_points']}")
        print(f"  Leaf Functions: {stats['leaf_functions']}")

        # Show some sample calls
        nodes = analyzer.get_nodes()
        if nodes:
            print(f"\n  Sample Call Relationships:")
            for node in nodes[:10]:
                if node.calls:
                    calls_preview = list(node.calls)[:3]
                    more = f" (+{len(node.calls)-3} more)" if len(node.calls) > 3 else ""
                    print(f"    {node.name} -> {', '.join(calls_preview)}{more}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
