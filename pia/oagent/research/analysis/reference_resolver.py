#!/usr/bin/env python3
"""
reference_resolver.py - Reference Resolver (Step 13)

Cross-file reference tracking for symbols, imports, and dependencies.
Resolves where symbols are defined and where they are used.

PBTSO Phase: RESEARCH

Bus Topics:
- a2a.research.reference.resolve
- research.reference.found
- research.reference.missing

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
from typing import Any, Dict, List, Optional, Set, Tuple

from ..bootstrap import AgentBus
from ..parsers.base import ParserRegistry, ParseResult, ImportInfo
from ..index.symbol_store import SymbolIndexStore, Symbol


# ============================================================================
# Data Models
# ============================================================================


class ReferenceType(Enum):
    """Type of reference."""
    IMPORT = "import"           # Module/package import
    CALL = "call"               # Function/method call
    INSTANTIATION = "instantiation"  # Class instantiation
    ATTRIBUTE = "attribute"     # Attribute access
    INHERITANCE = "inheritance" # Class inheritance
    TYPE_HINT = "type_hint"     # Type annotation
    ASSIGNMENT = "assignment"   # Variable assignment
    ARGUMENT = "argument"       # Function argument


class ReferenceDirection(Enum):
    """Direction of reference."""
    OUTGOING = "outgoing"  # This file references another
    INCOMING = "incoming"  # Another file references this


@dataclass
class Reference:
    """Represents a reference to a symbol."""

    symbol_name: str
    reference_type: ReferenceType
    source_path: str
    source_line: int
    source_column: int = 0
    context: Optional[str] = None  # Surrounding code context

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol_name": self.symbol_name,
            "reference_type": self.reference_type.value,
            "source_path": self.source_path,
            "source_line": self.source_line,
            "source_column": self.source_column,
            "context": self.context,
        }


@dataclass
class ResolvedReference:
    """A reference resolved to its definition."""

    reference: Reference
    target_path: Optional[str] = None
    target_line: Optional[int] = None
    target_kind: Optional[str] = None  # class, function, variable
    is_external: bool = False
    module_name: Optional[str] = None
    confidence: float = 1.0  # How confident we are in the resolution

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reference": self.reference.to_dict(),
            "target_path": self.target_path,
            "target_line": self.target_line,
            "target_kind": self.target_kind,
            "is_external": self.is_external,
            "module_name": self.module_name,
            "confidence": self.confidence,
        }


@dataclass
class ReferenceGraph:
    """Graph of references between files."""

    outgoing: Dict[str, List[ResolvedReference]] = field(default_factory=dict)
    incoming: Dict[str, List[ResolvedReference]] = field(default_factory=dict)
    unresolved: List[Reference] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "outgoing": {
                path: [r.to_dict() for r in refs]
                for path, refs in self.outgoing.items()
            },
            "incoming": {
                path: [r.to_dict() for r in refs]
                for path, refs in self.incoming.items()
            },
            "unresolved_count": len(self.unresolved),
        }


# ============================================================================
# Reference Resolver
# ============================================================================


class ReferenceResolver:
    """
    Cross-file reference tracker and resolver.

    Analyzes code to find:
    - Import references and their targets
    - Function/method calls and their definitions
    - Class usage and inheritance relationships
    - Type annotations and their definitions

    PBTSO Phase: RESEARCH

    Example:
        resolver = ReferenceResolver(root="/project")
        refs = resolver.find_references("src/main.py")
        usages = resolver.find_usages("MyClass")
    """

    # Standard library modules (subset)
    PYTHON_STDLIB = {
        "abc", "argparse", "ast", "asyncio", "base64", "collections",
        "concurrent", "contextlib", "copy", "csv", "dataclasses", "datetime",
        "decimal", "enum", "functools", "glob", "hashlib", "heapq", "html",
        "http", "importlib", "inspect", "io", "itertools", "json", "logging",
        "math", "multiprocessing", "operator", "os", "pathlib", "pickle",
        "queue", "random", "re", "shutil", "socket", "sqlite3", "ssl",
        "string", "subprocess", "sys", "tempfile", "threading", "time",
        "typing", "unittest", "urllib", "uuid", "warnings", "xml", "zipfile",
    }

    def __init__(
        self,
        root: Optional[Path] = None,
        symbol_store: Optional[SymbolIndexStore] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the reference resolver.

        Args:
            root: Project root directory
            symbol_store: Symbol index for lookups
            bus: AgentBus for event emission
        """
        self.root = Path(root) if root else Path.cwd()
        self.symbol_store = symbol_store or SymbolIndexStore()
        self.bus = bus or AgentBus()

        # Caches
        self._import_map: Dict[str, Dict[str, str]] = {}  # path -> {alias -> target}
        self._resolution_cache: Dict[str, Optional[Symbol]] = {}

    def find_references(
        self,
        file_path: str,
        include_types: Optional[List[ReferenceType]] = None,
    ) -> List[Reference]:
        """
        Find all outgoing references from a file.

        Args:
            file_path: Path to the source file
            include_types: Filter by reference types (None = all)

        Returns:
            List of references found
        """
        try:
            content = Path(file_path).read_text(errors="ignore")
            tree = ast.parse(content)
        except Exception as e:
            self.bus.emit({
                "topic": "research.reference.error",
                "kind": "error",
                "data": {"path": file_path, "error": str(e)}
            })
            return []

        references = []

        # Build import map for this file
        import_map = self._build_import_map(tree)
        self._import_map[file_path] = import_map

        # Find different types of references
        if include_types is None or ReferenceType.IMPORT in include_types:
            references.extend(self._find_import_refs(tree, file_path))

        if include_types is None or ReferenceType.CALL in include_types:
            references.extend(self._find_call_refs(tree, file_path, content))

        if include_types is None or ReferenceType.INSTANTIATION in include_types:
            references.extend(self._find_instantiation_refs(tree, file_path, content))

        if include_types is None or ReferenceType.INHERITANCE in include_types:
            references.extend(self._find_inheritance_refs(tree, file_path, content))

        if include_types is None or ReferenceType.TYPE_HINT in include_types:
            references.extend(self._find_type_hint_refs(tree, file_path, content))

        return references

    def resolve_references(
        self,
        file_path: str,
        references: Optional[List[Reference]] = None,
    ) -> ReferenceGraph:
        """
        Resolve references to their definitions.

        Args:
            file_path: Source file path
            references: References to resolve (None = find them first)

        Returns:
            ReferenceGraph with resolved references
        """
        if references is None:
            references = self.find_references(file_path)

        graph = ReferenceGraph()
        graph.outgoing[file_path] = []

        for ref in references:
            resolved = self._resolve_reference(ref, file_path)

            if resolved.target_path:
                graph.outgoing[file_path].append(resolved)

                # Add to incoming
                if resolved.target_path not in graph.incoming:
                    graph.incoming[resolved.target_path] = []
                graph.incoming[resolved.target_path].append(resolved)

                self.bus.emit({
                    "topic": "research.reference.found",
                    "kind": "reference",
                    "data": resolved.to_dict()
                })
            else:
                graph.unresolved.append(ref)

        return graph

    def find_usages(
        self,
        symbol_name: str,
        symbol_path: Optional[str] = None,
        search_paths: Optional[List[str]] = None,
    ) -> List[ResolvedReference]:
        """
        Find all usages of a symbol across the codebase.

        Args:
            symbol_name: Name of the symbol to find
            symbol_path: Path where symbol is defined (for disambiguation)
            search_paths: Paths to search in (None = scan all)

        Returns:
            List of references to the symbol
        """
        usages = []

        # Get paths to search
        if search_paths:
            paths = [Path(p) for p in search_paths]
        else:
            paths = list(self.root.rglob("*.py"))

        for path in paths:
            if not path.is_file():
                continue

            try:
                content = path.read_text(errors="ignore")

                # Quick check if symbol is mentioned
                if symbol_name not in content:
                    continue

                # Parse and find references
                refs = self.find_references(str(path))

                for ref in refs:
                    if ref.symbol_name == symbol_name:
                        resolved = self._resolve_reference(ref, str(path))

                        # Check if it resolves to the expected path
                        if symbol_path is None or resolved.target_path == symbol_path:
                            usages.append(resolved)

            except Exception:
                continue

        return usages

    def find_definition(
        self,
        symbol_name: str,
        from_file: str,
        context_line: Optional[int] = None,
    ) -> Optional[ResolvedReference]:
        """
        Find the definition of a symbol.

        Args:
            symbol_name: Symbol to find
            from_file: File containing the reference
            context_line: Line number for context

        Returns:
            Resolved reference to definition, or None
        """
        ref = Reference(
            symbol_name=symbol_name,
            reference_type=ReferenceType.CALL,  # Generic
            source_path=from_file,
            source_line=context_line or 0,
        )

        resolved = self._resolve_reference(ref, from_file)

        if resolved.target_path:
            return resolved
        return None

    def get_reference_chain(
        self,
        start_path: str,
        max_depth: int = 5,
    ) -> Dict[str, List[str]]:
        """
        Get the chain of references starting from a file.

        Args:
            start_path: Starting file
            max_depth: Maximum depth to traverse

        Returns:
            Dict mapping paths to their dependencies
        """
        chain: Dict[str, List[str]] = {}
        visited: Set[str] = set()

        def traverse(path: str, depth: int):
            if depth > max_depth or path in visited:
                return

            visited.add(path)
            graph = self.resolve_references(path)

            deps = set()
            for ref in graph.outgoing.get(path, []):
                if ref.target_path and not ref.is_external:
                    deps.add(ref.target_path)

            chain[path] = list(deps)

            for dep in deps:
                traverse(dep, depth + 1)

        traverse(start_path, 0)
        return chain

    def _build_import_map(self, tree: ast.Module) -> Dict[str, str]:
        """Build mapping of import aliases to module names."""
        import_map = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    import_map[name] = alias.name

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    name = alias.asname or alias.name
                    full_name = f"{module}.{alias.name}" if module else alias.name
                    import_map[name] = full_name

        return import_map

    def _find_import_refs(self, tree: ast.Module, file_path: str) -> List[Reference]:
        """Find import references."""
        refs = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    refs.append(Reference(
                        symbol_name=alias.name,
                        reference_type=ReferenceType.IMPORT,
                        source_path=file_path,
                        source_line=node.lineno,
                        source_column=node.col_offset,
                    ))

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    refs.append(Reference(
                        symbol_name=f"{module}.{alias.name}" if module else alias.name,
                        reference_type=ReferenceType.IMPORT,
                        source_path=file_path,
                        source_line=node.lineno,
                        source_column=node.col_offset,
                    ))

        return refs

    def _find_call_refs(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[Reference]:
        """Find function/method call references."""
        refs = []
        lines = content.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = self._get_call_name(node.func)
                if name:
                    context = lines[node.lineno - 1] if node.lineno <= len(lines) else None
                    refs.append(Reference(
                        symbol_name=name,
                        reference_type=ReferenceType.CALL,
                        source_path=file_path,
                        source_line=node.lineno,
                        source_column=node.col_offset,
                        context=context,
                    ))

        return refs

    def _find_instantiation_refs(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[Reference]:
        """Find class instantiation references."""
        refs = []
        lines = content.split("\n")

        # First pass: collect class names from imports
        import_map = self._import_map.get(file_path, {})
        class_names = set()

        # Query symbol store for known classes
        known_classes = self.symbol_store.query(kind="class", limit=10000)
        class_names.update(c.name for c in known_classes)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = self._get_call_name(node.func)
                if name and (name in class_names or name[0].isupper()):
                    # Heuristic: CamelCase names are likely classes
                    context = lines[node.lineno - 1] if node.lineno <= len(lines) else None
                    refs.append(Reference(
                        symbol_name=name,
                        reference_type=ReferenceType.INSTANTIATION,
                        source_path=file_path,
                        source_line=node.lineno,
                        source_column=node.col_offset,
                        context=context,
                    ))

        return refs

    def _find_inheritance_refs(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[Reference]:
        """Find class inheritance references."""
        refs = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    name = self._get_name(base)
                    if name:
                        refs.append(Reference(
                            symbol_name=name,
                            reference_type=ReferenceType.INHERITANCE,
                            source_path=file_path,
                            source_line=node.lineno,
                            source_column=node.col_offset,
                        ))

        return refs

    def _find_type_hint_refs(
        self,
        tree: ast.Module,
        file_path: str,
        content: str,
    ) -> List[Reference]:
        """Find type hint references."""
        refs = []

        for node in ast.walk(tree):
            # Function argument annotations
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in node.args.args:
                    if arg.annotation:
                        names = self._extract_type_names(arg.annotation)
                        for name in names:
                            refs.append(Reference(
                                symbol_name=name,
                                reference_type=ReferenceType.TYPE_HINT,
                                source_path=file_path,
                                source_line=arg.lineno if hasattr(arg, 'lineno') else node.lineno,
                                source_column=arg.col_offset if hasattr(arg, 'col_offset') else 0,
                            ))

                # Return type annotation
                if node.returns:
                    names = self._extract_type_names(node.returns)
                    for name in names:
                        refs.append(Reference(
                            symbol_name=name,
                            reference_type=ReferenceType.TYPE_HINT,
                            source_path=file_path,
                            source_line=node.lineno,
                        ))

            # Variable annotations
            elif isinstance(node, ast.AnnAssign):
                if node.annotation:
                    names = self._extract_type_names(node.annotation)
                    for name in names:
                        refs.append(Reference(
                            symbol_name=name,
                            reference_type=ReferenceType.TYPE_HINT,
                            source_path=file_path,
                            source_line=node.lineno,
                            source_column=node.col_offset,
                        ))

        return refs

    def _get_call_name(self, node: ast.expr) -> Optional[str]:
        """Extract name from a call expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _get_name(self, node: ast.expr) -> Optional[str]:
        """Extract full name from an expression."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._get_name(node.value)
            if base:
                return f"{base}.{node.attr}"
            return node.attr
        return None

    def _extract_type_names(self, node: ast.expr) -> List[str]:
        """Extract type names from a type annotation."""
        names = []

        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            name = self._get_name(node)
            if name:
                names.append(name)
        elif isinstance(node, ast.Subscript):
            # Generic types like List[str]
            if isinstance(node.value, ast.Name):
                names.append(node.value.id)
            # Also extract inner types
            if isinstance(node.slice, ast.Tuple):
                for elt in node.slice.elts:
                    names.extend(self._extract_type_names(elt))
            else:
                names.extend(self._extract_type_names(node.slice))

        return names

    def _resolve_reference(self, ref: Reference, from_file: str) -> ResolvedReference:
        """Resolve a reference to its definition."""
        # Check cache
        cache_key = f"{from_file}:{ref.symbol_name}:{ref.reference_type.value}"
        if cache_key in self._resolution_cache:
            cached = self._resolution_cache[cache_key]
            if cached:
                return ResolvedReference(
                    reference=ref,
                    target_path=cached.path,
                    target_line=cached.line,
                    target_kind=cached.kind,
                )

        # Check if it's an import
        if ref.reference_type == ReferenceType.IMPORT:
            return self._resolve_import(ref, from_file)

        # Try to resolve via import map
        import_map = self._import_map.get(from_file, {})
        base_name = ref.symbol_name.split(".")[0]

        if base_name in import_map:
            module_name = import_map[base_name]
            target_path = self._module_to_path(module_name)

            if target_path:
                # Search for symbol in target
                symbols = self.symbol_store.query(
                    name=ref.symbol_name.split(".")[-1],
                    path=target_path,
                    limit=1,
                )

                if symbols:
                    symbol = symbols[0]
                    self._resolution_cache[cache_key] = symbol
                    return ResolvedReference(
                        reference=ref,
                        target_path=symbol.path,
                        target_line=symbol.line,
                        target_kind=symbol.kind,
                        module_name=module_name,
                    )

        # Try direct symbol lookup
        symbols = self.symbol_store.query(
            name=ref.symbol_name.split(".")[-1],
            limit=10,
        )

        if symbols:
            # Prefer symbols in same directory
            from_dir = str(Path(from_file).parent)
            for symbol in symbols:
                if symbol.path.startswith(from_dir):
                    self._resolution_cache[cache_key] = symbol
                    return ResolvedReference(
                        reference=ref,
                        target_path=symbol.path,
                        target_line=symbol.line,
                        target_kind=symbol.kind,
                    )

            # Fall back to first match
            symbol = symbols[0]
            self._resolution_cache[cache_key] = symbol
            return ResolvedReference(
                reference=ref,
                target_path=symbol.path,
                target_line=symbol.line,
                target_kind=symbol.kind,
                confidence=0.7,  # Lower confidence for non-local matches
            )

        # Check if it's external
        if self._is_external(ref.symbol_name):
            return ResolvedReference(
                reference=ref,
                is_external=True,
                module_name=ref.symbol_name.split(".")[0],
            )

        # Unresolved
        self._resolution_cache[cache_key] = None
        return ResolvedReference(reference=ref)

    def _resolve_import(self, ref: Reference, from_file: str) -> ResolvedReference:
        """Resolve an import reference."""
        module_name = ref.symbol_name

        # Check if external
        if self._is_external(module_name):
            return ResolvedReference(
                reference=ref,
                is_external=True,
                module_name=module_name,
            )

        # Try to find local module
        target_path = self._module_to_path(module_name, from_file)

        if target_path:
            return ResolvedReference(
                reference=ref,
                target_path=target_path,
                target_line=1,
                target_kind="module",
                module_name=module_name,
            )

        return ResolvedReference(
            reference=ref,
            is_external=True,
            module_name=module_name,
        )

    def _module_to_path(
        self,
        module_name: str,
        from_file: Optional[str] = None,
    ) -> Optional[str]:
        """Convert module name to file path."""
        parts = module_name.split(".")

        # Try relative to project root
        candidate = self.root / "/".join(parts)

        for ext in [".py", ".pyi", "/__init__.py"]:
            full_path = str(candidate) + ext
            if Path(full_path).exists():
                return full_path

        # Try relative to from_file
        if from_file:
            from_dir = Path(from_file).parent
            candidate = from_dir / "/".join(parts)

            for ext in [".py", ".pyi", "/__init__.py"]:
                full_path = str(candidate) + ext
                if Path(full_path).exists():
                    return full_path

        return None

    def _is_external(self, module_name: str) -> bool:
        """Check if module is external."""
        base = module_name.split(".")[0]
        return base in self.PYTHON_STDLIB


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Reference Resolver."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Reference Resolver (Step 13)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Find command
    find_parser = subparsers.add_parser("find", help="Find references in a file")
    find_parser.add_argument("file", help="File to analyze")
    find_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Resolve command
    resolve_parser = subparsers.add_parser("resolve", help="Resolve references")
    resolve_parser.add_argument("file", help="File to analyze")
    resolve_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Usage command
    usage_parser = subparsers.add_parser("usage", help="Find symbol usages")
    usage_parser.add_argument("symbol", help="Symbol name to find")
    usage_parser.add_argument("--root", default=".", help="Project root")
    usage_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    resolver = ReferenceResolver(root=Path(getattr(args, "root", ".")))

    if args.command == "find":
        refs = resolver.find_references(args.file)
        if args.json:
            print(json.dumps([r.to_dict() for r in refs], indent=2))
        else:
            print(f"Found {len(refs)} references in {args.file}:")
            for ref in refs:
                print(f"  {ref.reference_type.value:15} {ref.symbol_name:30} line {ref.source_line}")

    elif args.command == "resolve":
        graph = resolver.resolve_references(args.file)
        if args.json:
            print(json.dumps(graph.to_dict(), indent=2))
        else:
            print(f"Resolved references from {args.file}:")
            for ref in graph.outgoing.get(args.file, []):
                target = ref.target_path or "(external)"
                print(f"  {ref.reference.symbol_name:30} -> {target}")
            print(f"\nUnresolved: {len(graph.unresolved)}")

    elif args.command == "usage":
        usages = resolver.find_usages(args.symbol)
        if args.json:
            print(json.dumps([u.to_dict() for u in usages], indent=2))
        else:
            print(f"Found {len(usages)} usages of '{args.symbol}':")
            for usage in usages:
                print(f"  {usage.reference.source_path}:{usage.reference.source_line}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
