#!/usr/bin/env python3
"""
transformer.py - AST Transformer Base (Step 53)

PBTSO Phase: ITERATE

Provides:
- Abstract base class for AST transformations
- Common transformation utilities
- Source code parsing and unparsing
- Import manipulation helpers

Bus Topics:
- code.ast.transformed
- code.transform.applied

Protocol: DKIN v30
"""

from __future__ import annotations

import ast
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# Types
# =============================================================================

@dataclass
class TransformResult:
    """Result of an AST transformation."""
    success: bool
    source: str
    transformed: str
    changes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "source_len": len(self.source),
            "transformed_len": len(self.transformed),
            "changes": self.changes,
            "warnings": self.warnings,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class ASTLocation:
    """Location in AST for targeted transformations."""
    lineno: int
    col_offset: int = 0
    end_lineno: Optional[int] = None
    end_col_offset: Optional[int] = None

    @classmethod
    def from_node(cls, node: ast.AST) -> "ASTLocation":
        """Create location from AST node."""
        return cls(
            lineno=getattr(node, "lineno", 0),
            col_offset=getattr(node, "col_offset", 0),
            end_lineno=getattr(node, "end_lineno", None),
            end_col_offset=getattr(node, "end_col_offset", None),
        )


# =============================================================================
# Base Transformer
# =============================================================================

class ASTTransformer(ABC):
    """
    Abstract base class for AST transformations.

    PBTSO Phase: ITERATE

    Subclasses implement specific transformations (add imports,
    rename functions, etc.) by overriding the transform() method.

    Usage:
        transformer = MyTransformer()
        result = transformer.apply(source_code, context)
    """

    def __init__(self, bus: Optional[Any] = None):
        self.bus = bus
        self._changes: List[str] = []
        self._warnings: List[str] = []

    @abstractmethod
    def transform(self, tree: ast.Module, context: Dict[str, Any]) -> ast.Module:
        """
        Transform an AST tree.

        Args:
            tree: Parsed AST module
            context: Transformation context (varies by transformer)

        Returns:
            Transformed AST module
        """
        pass

    def apply(self, source: str, context: Optional[Dict[str, Any]] = None) -> TransformResult:
        """
        Apply transformation to source code.

        Args:
            source: Python source code
            context: Transformation context

        Returns:
            TransformResult with transformed code
        """
        context = context or {}
        self._changes = []
        self._warnings = []
        start_time = time.time()

        try:
            # Parse source
            tree = ast.parse(source)

            # Apply transformation
            transformed_tree = self.transform(tree, context)

            # Fix missing locations
            ast.fix_missing_locations(transformed_tree)

            # Unparse back to source
            transformed_source = ast.unparse(transformed_tree)

            elapsed_ms = (time.time() - start_time) * 1000

            # Emit success event
            if self.bus:
                self.bus.emit({
                    "topic": "code.ast.transformed",
                    "kind": "transform",
                    "actor": "code-agent",
                    "data": {
                        "transformer": self.__class__.__name__,
                        "changes": len(self._changes),
                        "elapsed_ms": elapsed_ms,
                    },
                })

            return TransformResult(
                success=True,
                source=source,
                transformed=transformed_source,
                changes=self._changes,
                warnings=self._warnings,
                elapsed_ms=elapsed_ms,
            )

        except SyntaxError as e:
            return TransformResult(
                success=False,
                source=source,
                transformed=source,
                changes=[],
                warnings=[f"Syntax error: {e}"],
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return TransformResult(
                success=False,
                source=source,
                transformed=source,
                changes=[],
                warnings=[f"Transform error: {e}"],
                elapsed_ms=(time.time() - start_time) * 1000,
            )

    def record_change(self, description: str) -> None:
        """Record a change made during transformation."""
        self._changes.append(description)

    def record_warning(self, warning: str) -> None:
        """Record a warning during transformation."""
        self._warnings.append(warning)


# =============================================================================
# Common Transformers
# =============================================================================

class AddImportTransformer(ASTTransformer):
    """
    Add import statements to source code.

    Context:
        imports: List[str] - module names to import
        from_imports: Dict[str, List[str]] - from X import Y mappings
    """

    def transform(self, tree: ast.Module, context: Dict[str, Any]) -> ast.Module:
        imports = context.get("imports", [])
        from_imports = context.get("from_imports", {})

        new_imports: List[ast.stmt] = []

        # Create regular imports
        for module_name in imports:
            if not self._has_import(tree, module_name):
                new_imports.append(
                    ast.Import(names=[ast.alias(name=module_name, asname=None)])
                )
                self.record_change(f"Added import: {module_name}")

        # Create from imports
        for module_name, names in from_imports.items():
            existing = self._get_from_import(tree, module_name)
            if existing:
                # Add to existing from import
                existing_names = {alias.name for alias in existing.names}
                for name in names:
                    if name not in existing_names:
                        existing.names.append(ast.alias(name=name, asname=None))
                        self.record_change(f"Added {name} to from {module_name} import")
            else:
                # Create new from import
                new_imports.append(
                    ast.ImportFrom(
                        module=module_name,
                        names=[ast.alias(name=n, asname=None) for n in names],
                        level=0,
                    )
                )
                self.record_change(f"Added from {module_name} import {', '.join(names)}")

        # Insert new imports at the beginning (after docstring/future imports)
        insert_pos = self._find_import_position(tree)
        tree.body = tree.body[:insert_pos] + new_imports + tree.body[insert_pos:]

        return tree

    def _has_import(self, tree: ast.Module, module_name: str) -> bool:
        """Check if module is already imported."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == module_name for alias in node.names):
                    return True
        return False

    def _get_from_import(self, tree: ast.Module, module_name: str) -> Optional[ast.ImportFrom]:
        """Get existing from import for module."""
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == module_name:
                return node
        return None

    def _find_import_position(self, tree: ast.Module) -> int:
        """Find the position to insert new imports."""
        position = 0

        for i, node in enumerate(tree.body):
            # Skip docstring
            if i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    position = 1
                    continue

            # Skip future imports
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                position = i + 1
                continue

            # Stop at first non-import
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                break

            position = i + 1

        return position


class RemoveImportTransformer(ASTTransformer):
    """
    Remove unused imports from source code.

    Context:
        remove: List[str] - import names to remove
        remove_unused: bool - auto-detect and remove unused imports
    """

    def transform(self, tree: ast.Module, context: Dict[str, Any]) -> ast.Module:
        to_remove = set(context.get("remove", []))
        remove_unused = context.get("remove_unused", False)

        if remove_unused:
            used_names = self._get_used_names(tree)
            to_remove.update(self._find_unused_imports(tree, used_names))

        # Filter out removed imports
        new_body = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                remaining = [a for a in node.names if a.name not in to_remove]
                if remaining:
                    node.names = remaining
                    new_body.append(node)
                else:
                    for a in node.names:
                        self.record_change(f"Removed import: {a.name}")
            elif isinstance(node, ast.ImportFrom):
                remaining = [a for a in node.names if a.name not in to_remove]
                if remaining:
                    node.names = remaining
                    new_body.append(node)
                else:
                    for a in node.names:
                        self.record_change(f"Removed from {node.module} import {a.name}")
            else:
                new_body.append(node)

        tree.body = new_body
        return tree

    def _get_used_names(self, tree: ast.Module) -> Set[str]:
        """Get all names used in the code."""
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Get the root name of attribute chains
                current = node
                while isinstance(current, ast.Attribute):
                    current = current.value
                if isinstance(current, ast.Name):
                    used.add(current.id)
        return used

    def _find_unused_imports(self, tree: ast.Module, used_names: Set[str]) -> Set[str]:
        """Find imports that are not used."""
        unused = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name not in used_names:
                        unused.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name not in used_names:
                        unused.add(alias.name)
        return unused


class RenameTransformer(ASTTransformer):
    """
    Rename symbols (functions, classes, variables) in source code.

    Context:
        renames: Dict[str, str] - old_name -> new_name mappings
    """

    def transform(self, tree: ast.Module, context: Dict[str, Any]) -> ast.Module:
        renames = context.get("renames", {})

        class RenameVisitor(ast.NodeTransformer):
            def __init__(self, renames: Dict[str, str], parent: RenameTransformer):
                self.renames = renames
                self.parent = parent

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                if node.name in self.renames:
                    old_name = node.name
                    node.name = self.renames[old_name]
                    self.parent.record_change(f"Renamed function: {old_name} -> {node.name}")
                self.generic_visit(node)
                return node

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
                if node.name in self.renames:
                    old_name = node.name
                    node.name = self.renames[old_name]
                    self.parent.record_change(f"Renamed async function: {old_name} -> {node.name}")
                self.generic_visit(node)
                return node

            def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
                if node.name in self.renames:
                    old_name = node.name
                    node.name = self.renames[old_name]
                    self.parent.record_change(f"Renamed class: {old_name} -> {node.name}")
                self.generic_visit(node)
                return node

            def visit_Name(self, node: ast.Name) -> ast.Name:
                if node.id in self.renames:
                    node.id = self.renames[node.id]
                return node

        visitor = RenameVisitor(renames, self)
        return visitor.visit(tree)


class DocstringTransformer(ASTTransformer):
    """
    Add or update docstrings for functions and classes.

    Context:
        docstrings: Dict[str, str] - name -> docstring mappings
        add_missing: bool - add placeholder docstrings for undocumented items
    """

    def transform(self, tree: ast.Module, context: Dict[str, Any]) -> ast.Module:
        docstrings = context.get("docstrings", {})
        add_missing = context.get("add_missing", False)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name

                # Check if we have a docstring to set
                if name in docstrings:
                    self._set_docstring(node, docstrings[name])
                    self.record_change(f"Set docstring for: {name}")

                elif add_missing and not self._has_docstring(node):
                    placeholder = f'"""TODO: Document {name}."""'
                    self._set_docstring(node, placeholder)
                    self.record_change(f"Added placeholder docstring for: {name}")

        return tree

    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if node has a docstring."""
        if not hasattr(node, "body") or not node.body:
            return False
        first = node.body[0]
        return (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        )

    def _set_docstring(self, node: ast.AST, docstring: str) -> None:
        """Set or replace docstring for node."""
        # Remove quotes if present
        docstring = docstring.strip()
        if docstring.startswith('"""') and docstring.endswith('"""'):
            docstring = docstring[3:-3]
        elif docstring.startswith("'''") and docstring.endswith("'''"):
            docstring = docstring[3:-3]

        docstring_node = ast.Expr(value=ast.Constant(value=docstring))

        if self._has_docstring(node):
            node.body[0] = docstring_node
        else:
            node.body.insert(0, docstring_node)


# =============================================================================
# Utility Functions
# =============================================================================

def parse_source(source: str) -> Tuple[ast.Module, List[str]]:
    """
    Parse source code to AST with error handling.

    Returns:
        Tuple of (AST module, list of warnings)
    """
    warnings = []
    try:
        tree = ast.parse(source)
        return tree, warnings
    except SyntaxError as e:
        warnings.append(f"Syntax error at line {e.lineno}: {e.msg}")
        raise


def unparse_tree(tree: ast.Module) -> str:
    """
    Convert AST back to source code.

    Uses ast.unparse (Python 3.9+) with fallback formatting.
    """
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def get_defined_names(tree: ast.Module) -> Dict[str, List[ASTLocation]]:
    """
    Get all defined names (functions, classes, variables) with locations.

    Returns:
        Dict mapping name to list of definition locations
    """
    names: Dict[str, List[ASTLocation]] = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if name not in names:
                names[name] = []
            names[name].append(ASTLocation.from_node(node))

        elif isinstance(node, ast.ClassDef):
            name = node.name
            if name not in names:
                names[name] = []
            names[name].append(ASTLocation.from_node(node))

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    if name not in names:
                        names[name] = []
                    names[name].append(ASTLocation.from_node(node))

    return names
