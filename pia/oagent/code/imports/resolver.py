#!/usr/bin/env python3
"""
resolver.py - Import Dependency Resolver (Step 58)

PBTSO Phase: PLAN

Provides:
- Import path resolution within project
- Required imports detection based on symbol usage
- Circular dependency detection
- Import suggestions from Research Agent

Bus Topics:
- code.imports.resolve
- code.imports.add
- a2a.research.query

Protocol: DKIN v30
"""

from __future__ import annotations

import ast
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# =============================================================================
# Types
# =============================================================================

@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str
    names: List[str]  # Empty for "import X", populated for "from X import Y, Z"
    alias: Optional[str] = None
    is_relative: bool = False
    level: int = 0  # Number of dots in relative import
    lineno: int = 0
    source_file: Optional[str] = None

    @property
    def full_name(self) -> str:
        """Get full module name including relative prefix."""
        if self.is_relative:
            return "." * self.level + self.module
        return self.module


@dataclass
class DependencyInfo:
    """Information about a module's dependencies."""
    module: str
    source_file: Path
    imports: List[ImportInfo]
    symbols_used: Set[str]
    symbols_defined: Set[str]
    unresolved: List[str] = field(default_factory=list)


@dataclass
class CircularDependency:
    """Represents a circular dependency chain."""
    chain: List[str]
    source_file: str

    def __str__(self) -> str:
        return " -> ".join(self.chain)


# =============================================================================
# Import Dependency Resolver
# =============================================================================

class ImportDependencyResolver:
    """
    Resolve and manage import dependencies.

    PBTSO Phase: PLAN

    Features:
    - Resolve import paths to source files
    - Detect required imports based on symbol usage
    - Find circular dependencies
    - Suggest imports from project-wide analysis

    Usage:
        resolver = ImportDependencyResolver(project_root)
        path = resolver.resolve_import("mymodule.utils", from_file)
        imports = resolver.get_required_imports(symbols_used, available_modules)
    """

    # Standard library modules (partial list for common ones)
    STDLIB_MODULES: FrozenSet[str] = frozenset([
        "abc", "argparse", "ast", "asyncio", "base64", "collections",
        "contextlib", "copy", "csv", "dataclasses", "datetime", "enum",
        "functools", "glob", "hashlib", "http", "io", "itertools", "json",
        "logging", "math", "os", "pathlib", "pickle", "queue", "random",
        "re", "shutil", "socket", "sqlite3", "string", "subprocess", "sys",
        "tempfile", "threading", "time", "typing", "unittest", "urllib",
        "uuid", "warnings", "xml", "zipfile",
    ])

    def __init__(
        self,
        project_root: Path,
        bus: Optional[Any] = None,
    ):
        self.project_root = Path(project_root)
        self.bus = bus

        # Cache for resolved modules
        self.module_map: Dict[str, Path] = {}
        self.symbol_map: Dict[str, Set[str]] = {}  # module -> symbols defined

        # Build initial module map
        self._scan_project()

    def _scan_project(self) -> None:
        """Scan project directory for Python modules."""
        for py_file in self.project_root.rglob("*.py"):
            # Skip hidden directories and venv
            if any(part.startswith(".") for part in py_file.parts):
                continue
            if "venv" in py_file.parts or "node_modules" in py_file.parts:
                continue

            # Calculate module name
            rel_path = py_file.relative_to(self.project_root)
            parts = list(rel_path.parts[:-1])

            if py_file.name == "__init__.py":
                module_name = ".".join(parts) if parts else ""
            else:
                parts.append(py_file.stem)
                module_name = ".".join(parts)

            if module_name:
                self.module_map[module_name] = py_file

    # =========================================================================
    # Resolution
    # =========================================================================

    def resolve_import(
        self,
        import_name: str,
        from_file: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Resolve an import to its source file.

        Args:
            import_name: Module name (e.g., "mypackage.utils")
            from_file: File containing the import (for relative imports)

        Returns:
            Path to source file or None if not found
        """
        # Check if it's a stdlib module
        top_level = import_name.split(".")[0]
        if top_level in self.STDLIB_MODULES:
            return None  # Standard library, no source file

        # Check module map first
        if import_name in self.module_map:
            return self.module_map[import_name]

        # Try relative import resolution
        if from_file:
            result = self._resolve_relative(import_name, from_file)
            if result:
                return result

        # Try various path combinations
        parts = import_name.split(".")

        # Check as direct module
        for i in range(len(parts), 0, -1):
            partial = ".".join(parts[:i])
            if partial in self.module_map:
                return self.module_map[partial]

        # Check as package
        package_init = self.project_root / "/".join(parts) / "__init__.py"
        if package_init.exists():
            return package_init

        # Check as module file
        module_file = self.project_root / "/".join(parts[:-1]) / f"{parts[-1]}.py"
        if module_file.exists():
            return module_file

        return None

    def _resolve_relative(self, import_name: str, from_file: Path) -> Optional[Path]:
        """Resolve relative import from a specific file."""
        current = from_file.parent

        for part in import_name.split("."):
            if part == "":
                continue

            # Check as module file
            candidate = current / f"{part}.py"
            if candidate.exists():
                return candidate

            # Check as package
            candidate = current / part / "__init__.py"
            if candidate.exists():
                return candidate

            current = current / part

        return None

    # =========================================================================
    # Required Imports
    # =========================================================================

    def get_required_imports(
        self,
        symbols_used: Set[str],
        available_modules: Optional[Dict[str, Set[str]]] = None,
    ) -> List[str]:
        """
        Determine required imports based on symbols used.

        Args:
            symbols_used: Set of symbol names used in code
            available_modules: Dict of module -> symbols (uses cached if not provided)

        Returns:
            List of import statements to add
        """
        available = available_modules or self.symbol_map
        imports = []

        for module, symbols in available.items():
            needed = symbols_used & symbols
            if needed:
                if len(needed) == 1:
                    imports.append(f"from {module} import {list(needed)[0]}")
                else:
                    imports.append(f"from {module} import {', '.join(sorted(needed))}")

        return imports

    def analyze_file(self, file_path: Path) -> DependencyInfo:
        """
        Analyze a file for its dependencies.

        Returns:
            DependencyInfo with imports, symbols used, and symbols defined
        """
        source = file_path.read_text()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return DependencyInfo(
                module=file_path.stem,
                source_file=file_path,
                imports=[],
                symbols_used=set(),
                symbols_defined=set(),
            )

        imports = self._extract_imports(tree, file_path)
        symbols_used = self._extract_used_symbols(tree)
        symbols_defined = self._extract_defined_symbols(tree)

        # Find unresolved symbols
        imported_names = set()
        for imp in imports:
            if imp.names:
                imported_names.update(imp.names)
            else:
                imported_names.add(imp.module.split(".")[-1])
                if imp.alias:
                    imported_names.add(imp.alias)

        unresolved = list(symbols_used - symbols_defined - imported_names - self._get_builtins())

        return DependencyInfo(
            module=file_path.stem,
            source_file=file_path,
            imports=imports,
            symbols_used=symbols_used,
            symbols_defined=symbols_defined,
            unresolved=unresolved,
        )

    def _extract_imports(self, tree: ast.Module, source_file: Path) -> List[ImportInfo]:
        """Extract import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[],
                        alias=alias.asname,
                        lineno=node.lineno,
                        source_file=str(source_file),
                    ))

            elif isinstance(node, ast.ImportFrom):
                imports.append(ImportInfo(
                    module=node.module or "",
                    names=[a.name for a in node.names],
                    is_relative=node.level > 0,
                    level=node.level,
                    lineno=node.lineno,
                    source_file=str(source_file),
                ))

        return imports

    def _extract_used_symbols(self, tree: ast.Module) -> Set[str]:
        """Extract all symbol names used in code."""
        used = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Get root name of attribute chain
                current = node
                while isinstance(current, ast.Attribute):
                    current = current.value
                if isinstance(current, ast.Name):
                    used.add(current.id)

        return used

    def _extract_defined_symbols(self, tree: ast.Module) -> Set[str]:
        """Extract all symbols defined in file."""
        defined = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(node.name)
            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)

        return defined

    def _get_builtins(self) -> Set[str]:
        """Get set of Python builtin names."""
        return set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))

    # =========================================================================
    # Circular Dependency Detection
    # =========================================================================

    def find_circular_dependencies(self) -> List[CircularDependency]:
        """Find all circular dependencies in the project."""
        circulars = []
        visited: Set[str] = set()
        rec_stack: List[str] = []

        def dfs(module: str) -> bool:
            visited.add(module)
            rec_stack.append(module)

            # Get imports for this module
            source_file = self.module_map.get(module)
            if not source_file:
                rec_stack.pop()
                return False

            try:
                dep_info = self.analyze_file(source_file)
            except Exception:
                rec_stack.pop()
                return False

            for imp in dep_info.imports:
                dep_module = imp.module
                if not dep_module:
                    continue

                if dep_module in rec_stack:
                    # Found circular dependency
                    cycle_start = rec_stack.index(dep_module)
                    circulars.append(CircularDependency(
                        chain=rec_stack[cycle_start:] + [dep_module],
                        source_file=str(source_file),
                    ))
                    continue

                if dep_module not in visited and dep_module in self.module_map:
                    dfs(dep_module)

            rec_stack.pop()
            return len(circulars) > 0

        for module in self.module_map:
            if module not in visited:
                dfs(module)

        return circulars

    # =========================================================================
    # Suggestions
    # =========================================================================

    def suggest_imports(
        self,
        symbol: str,
        context: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Suggest import statements for a symbol.

        Returns list of (import_statement, confidence) tuples.
        """
        suggestions = []

        # Check project modules
        for module, path in self.module_map.items():
            try:
                tree = ast.parse(path.read_text())
                defined = self._extract_defined_symbols(tree)

                if symbol in defined:
                    suggestions.append((f"from {module} import {symbol}", 0.9))
            except Exception:
                continue

        # Check common patterns
        common_imports = {
            "Path": ("from pathlib import Path", 1.0),
            "Optional": ("from typing import Optional", 1.0),
            "List": ("from typing import List", 1.0),
            "Dict": ("from typing import Dict", 1.0),
            "Any": ("from typing import Any", 1.0),
            "dataclass": ("from dataclasses import dataclass", 1.0),
            "field": ("from dataclasses import field", 1.0),
            "Enum": ("from enum import Enum", 1.0),
            "json": ("import json", 1.0),
            "os": ("import os", 1.0),
            "re": ("import re", 1.0),
            "time": ("import time", 1.0),
            "uuid": ("import uuid", 1.0),
            "asyncio": ("import asyncio", 1.0),
            "logging": ("import logging", 1.0),
        }

        if symbol in common_imports:
            imp, conf = common_imports[symbol]
            suggestions.insert(0, (imp, conf))

        # Query Research Agent if available
        if self.bus and not suggestions:
            self.bus.emit({
                "topic": "a2a.research.query",
                "kind": "query",
                "actor": "code-agent",
                "data": {
                    "query": f"import for symbol: {symbol}",
                    "context": context,
                    "type": "import_suggestion",
                },
            })

        return suggestions

    # =========================================================================
    # Module Map Management
    # =========================================================================

    def refresh(self) -> int:
        """Rescan project and refresh module map. Returns module count."""
        self.module_map.clear()
        self.symbol_map.clear()
        self._scan_project()
        return len(self.module_map)

    def get_module_map(self) -> Dict[str, Path]:
        """Get current module map."""
        return dict(self.module_map)

    def add_module(self, module_name: str, path: Path) -> None:
        """Manually add a module to the map."""
        self.module_map[module_name] = path


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Import Resolver."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Import Dependency Resolver (Step 58)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # resolve command
    resolve_parser = subparsers.add_parser("resolve", help="Resolve import path")
    resolve_parser.add_argument("import_name", help="Import name to resolve")
    resolve_parser.add_argument("--project", default=".", help="Project root")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze file dependencies")
    analyze_parser.add_argument("file", help="File to analyze")
    analyze_parser.add_argument("--project", default=".", help="Project root")

    # circular command
    circular_parser = subparsers.add_parser("circular", help="Find circular dependencies")
    circular_parser.add_argument("--project", default=".", help="Project root")

    # suggest command
    suggest_parser = subparsers.add_parser("suggest", help="Suggest imports for symbol")
    suggest_parser.add_argument("symbol", help="Symbol name")
    suggest_parser.add_argument("--project", default=".", help="Project root")

    args = parser.parse_args()

    resolver = ImportDependencyResolver(Path(args.project))

    if args.command == "resolve":
        path = resolver.resolve_import(args.import_name)
        if path:
            print(f"Resolved: {path}")
        else:
            print("Not found (may be stdlib or external)")
        return 0

    elif args.command == "analyze":
        info = resolver.analyze_file(Path(args.file))
        print(f"Module: {info.module}")
        print(f"Imports: {len(info.imports)}")
        for imp in info.imports:
            print(f"  - {imp.full_name}")
        print(f"Symbols used: {len(info.symbols_used)}")
        print(f"Symbols defined: {len(info.symbols_defined)}")
        print(f"Unresolved: {info.unresolved}")
        return 0

    elif args.command == "circular":
        circulars = resolver.find_circular_dependencies()
        if circulars:
            print(f"Found {len(circulars)} circular dependencies:")
            for c in circulars:
                print(f"  {c}")
        else:
            print("No circular dependencies found")
        return 0

    elif args.command == "suggest":
        suggestions = resolver.suggest_imports(args.symbol)
        if suggestions:
            print(f"Suggestions for '{args.symbol}':")
            for imp, conf in suggestions:
                print(f"  [{conf:.1f}] {imp}")
        else:
            print(f"No suggestions for '{args.symbol}'")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
