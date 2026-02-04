#!/usr/bin/env python3
"""
python_transformer.py - Python Code Transformer (Step 54)

PBTSO Phase: ITERATE

Provides:
- Python-specific AST transformations
- Method/function manipulation
- Class modification
- Decorator handling

Bus Topics:
- code.transform.python
- code.transform.complete

Protocol: DKIN v30
"""

from __future__ import annotations

import ast
import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .transformer import ASTTransformer, TransformResult


# =============================================================================
# Types
# =============================================================================

@dataclass
class MethodInfo:
    """Information about a method in a class."""
    name: str
    lineno: int
    args: List[str]
    decorators: List[str]
    is_async: bool
    is_property: bool
    is_classmethod: bool
    is_staticmethod: bool
    docstring: Optional[str]


@dataclass
class ClassInfo:
    """Information about a class definition."""
    name: str
    lineno: int
    bases: List[str]
    methods: List[MethodInfo]
    attributes: List[str]
    decorators: List[str]
    docstring: Optional[str]


# =============================================================================
# Python Code Transformer
# =============================================================================

class PythonCodeTransformer(ASTTransformer):
    """
    Python-specific AST transformations.

    PBTSO Phase: ITERATE

    Provides methods for:
    - Adding methods to classes
    - Renaming functions/methods
    - Adding/removing decorators
    - Modifying function signatures
    - Inserting code at specific locations
    """

    def transform(self, tree: ast.Module, context: Dict[str, Any]) -> ast.Module:
        """
        Apply Python-specific transformations.

        Context:
            operation: str - type of transformation
            ... operation-specific parameters
        """
        operation = context.get("operation", "noop")

        if operation == "add_method":
            return self._add_method_to_class(tree, context)
        elif operation == "rename_function":
            return self._rename_function(tree, context)
        elif operation == "add_decorator":
            return self._add_decorator(tree, context)
        elif operation == "remove_decorator":
            return self._remove_decorator(tree, context)
        elif operation == "modify_signature":
            return self._modify_signature(tree, context)
        elif operation == "insert_code":
            return self._insert_code(tree, context)
        else:
            self.record_warning(f"Unknown operation: {operation}")
            return tree

    # =========================================================================
    # High-Level Operations
    # =========================================================================

    def add_method_to_class(
        self,
        source: str,
        class_name: str,
        method_source: str,
        position: str = "end",  # "end", "start", or method name to insert after
    ) -> TransformResult:
        """
        Add a method to a class.

        Args:
            source: Source code containing the class
            class_name: Name of the class to modify
            method_source: Source code of the method to add
            position: Where to insert ("end", "start", or method name)

        Returns:
            TransformResult with modified source
        """
        context = {
            "operation": "add_method",
            "class_name": class_name,
            "method_source": method_source,
            "position": position,
        }
        return self.apply(source, context)

    def rename_function(
        self,
        source: str,
        old_name: str,
        new_name: str,
        rename_calls: bool = True,
    ) -> TransformResult:
        """
        Rename a function and optionally its call sites.

        Args:
            source: Source code
            old_name: Current function name
            new_name: New function name
            rename_calls: Also rename call sites

        Returns:
            TransformResult with modified source
        """
        context = {
            "operation": "rename_function",
            "old_name": old_name,
            "new_name": new_name,
            "rename_calls": rename_calls,
        }
        return self.apply(source, context)

    def add_decorator(
        self,
        source: str,
        target_name: str,
        decorator: str,
        position: int = 0,  # 0 = first (outermost)
    ) -> TransformResult:
        """
        Add a decorator to a function or class.

        Args:
            source: Source code
            target_name: Name of function/class to decorate
            decorator: Decorator to add (without @)
            position: Position in decorator list (0 = outermost)

        Returns:
            TransformResult with modified source
        """
        context = {
            "operation": "add_decorator",
            "target_name": target_name,
            "decorator": decorator,
            "position": position,
        }
        return self.apply(source, context)

    def extract_function(
        self,
        source: str,
        start_line: int,
        end_line: int,
        function_name: str,
        parameters: Optional[List[str]] = None,
    ) -> TransformResult:
        """
        Extract lines into a new function.

        Args:
            source: Source code
            start_line: First line to extract
            end_line: Last line to extract
            function_name: Name for new function
            parameters: Parameters for new function

        Returns:
            TransformResult with extracted function
        """
        context = {
            "operation": "extract_function",
            "start_line": start_line,
            "end_line": end_line,
            "function_name": function_name,
            "parameters": parameters or [],
        }
        return self.apply(source, context)

    # =========================================================================
    # Internal Transformations
    # =========================================================================

    def _add_method_to_class(self, tree: ast.Module, context: Dict[str, Any]) -> ast.Module:
        """Add a method to a class definition."""
        class_name = context["class_name"]
        method_source = context["method_source"]
        position = context.get("position", "end")

        # Parse the method
        try:
            method_tree = ast.parse(method_source)
            if not method_tree.body:
                self.record_warning("Empty method source")
                return tree
            method_node = method_tree.body[0]
        except SyntaxError as e:
            self.record_warning(f"Invalid method source: {e}")
            return tree

        # Find and modify the class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                if position == "start":
                    # Insert after docstring if present
                    insert_pos = 0
                    if node.body and isinstance(node.body[0], ast.Expr):
                        if isinstance(node.body[0].value, ast.Constant):
                            insert_pos = 1
                    node.body.insert(insert_pos, method_node)
                elif position == "end":
                    node.body.append(method_node)
                else:
                    # Insert after specific method
                    for i, child in enumerate(node.body):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if child.name == position:
                                node.body.insert(i + 1, method_node)
                                break
                    else:
                        # Method not found, append at end
                        node.body.append(method_node)

                method_name = getattr(method_node, "name", "unknown")
                self.record_change(f"Added method {method_name} to class {class_name}")
                break
        else:
            self.record_warning(f"Class not found: {class_name}")

        return tree

    def _rename_function(self, tree: ast.Module, context: Dict[str, Any]) -> ast.Module:
        """Rename a function and its call sites."""
        old_name = context["old_name"]
        new_name = context["new_name"]
        rename_calls = context.get("rename_calls", True)

        class RenameVisitor(ast.NodeTransformer):
            def __init__(self, transformer: PythonCodeTransformer):
                self.transformer = transformer
                self.definition_renamed = False
                self.calls_renamed = 0

            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                if node.name == old_name:
                    node.name = new_name
                    self.definition_renamed = True
                    self.transformer.record_change(
                        f"Renamed function definition: {old_name} -> {new_name}"
                    )
                self.generic_visit(node)
                return node

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
                if node.name == old_name:
                    node.name = new_name
                    self.definition_renamed = True
                    self.transformer.record_change(
                        f"Renamed async function definition: {old_name} -> {new_name}"
                    )
                self.generic_visit(node)
                return node

            def visit_Call(self, node: ast.Call) -> ast.Call:
                if rename_calls and isinstance(node.func, ast.Name):
                    if node.func.id == old_name:
                        node.func.id = new_name
                        self.calls_renamed += 1
                self.generic_visit(node)
                return node

        visitor = RenameVisitor(self)
        tree = visitor.visit(tree)

        if visitor.calls_renamed > 0:
            self.record_change(f"Renamed {visitor.calls_renamed} call site(s)")

        if not visitor.definition_renamed:
            self.record_warning(f"Function definition not found: {old_name}")

        return tree

    def _add_decorator(self, tree: ast.Module, context: Dict[str, Any]) -> ast.Module:
        """Add a decorator to a function or class."""
        target_name = context["target_name"]
        decorator_str = context["decorator"]
        position = context.get("position", 0)

        # Parse decorator
        try:
            decorator_node = ast.parse(f"@{decorator_str}\ndef _(): pass").body[0].decorator_list[0]
        except SyntaxError as e:
            self.record_warning(f"Invalid decorator: {e}")
            return tree

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == target_name:
                    # Check if decorator already exists
                    for existing in node.decorator_list:
                        if ast.unparse(existing) == ast.unparse(decorator_node):
                            self.record_warning(f"Decorator already exists on {target_name}")
                            return tree

                    # Insert at position
                    position = min(position, len(node.decorator_list))
                    node.decorator_list.insert(position, decorator_node)
                    self.record_change(f"Added @{decorator_str} to {target_name}")
                    break
        else:
            self.record_warning(f"Target not found: {target_name}")

        return tree

    def _remove_decorator(self, tree: ast.Module, context: Dict[str, Any]) -> ast.Module:
        """Remove a decorator from a function or class."""
        target_name = context["target_name"]
        decorator_str = context["decorator"]

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == target_name:
                    original_len = len(node.decorator_list)
                    node.decorator_list = [
                        d for d in node.decorator_list
                        if not ast.unparse(d).startswith(decorator_str)
                    ]
                    removed = original_len - len(node.decorator_list)
                    if removed > 0:
                        self.record_change(f"Removed {removed} decorator(s) from {target_name}")
                    else:
                        self.record_warning(f"Decorator not found on {target_name}")
                    break
        else:
            self.record_warning(f"Target not found: {target_name}")

        return tree

    def _modify_signature(self, tree: ast.Module, context: Dict[str, Any]) -> ast.Module:
        """Modify a function signature."""
        target_name = context["target_name"]
        add_params = context.get("add_params", [])
        remove_params = context.get("remove_params", [])
        rename_params = context.get("rename_params", {})

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == target_name:
                    # Remove parameters
                    if remove_params:
                        node.args.args = [
                            a for a in node.args.args
                            if a.arg not in remove_params
                        ]
                        self.record_change(f"Removed params: {remove_params}")

                    # Rename parameters
                    if rename_params:
                        for arg in node.args.args:
                            if arg.arg in rename_params:
                                old = arg.arg
                                arg.arg = rename_params[old]
                                self.record_change(f"Renamed param: {old} -> {arg.arg}")

                    # Add parameters
                    for param in add_params:
                        if isinstance(param, str):
                            new_arg = ast.arg(arg=param, annotation=None)
                        elif isinstance(param, dict):
                            new_arg = ast.arg(
                                arg=param["name"],
                                annotation=ast.parse(param.get("type", "")).body[0].value
                                if param.get("type") else None
                            )
                        else:
                            continue
                        node.args.args.append(new_arg)
                        self.record_change(f"Added param: {new_arg.arg}")
                    break
        else:
            self.record_warning(f"Function not found: {target_name}")

        return tree

    def _insert_code(self, tree: ast.Module, context: Dict[str, Any]) -> ast.Module:
        """Insert code at a specific location."""
        code = context["code"]
        location = context.get("location", "end")  # "start", "end", line number, or function name

        try:
            code_nodes = ast.parse(code).body
        except SyntaxError as e:
            self.record_warning(f"Invalid code: {e}")
            return tree

        if location == "start":
            # Insert after imports
            insert_pos = 0
            for i, node in enumerate(tree.body):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    insert_pos = i + 1
                elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
                    break
            tree.body = tree.body[:insert_pos] + code_nodes + tree.body[insert_pos:]

        elif location == "end":
            tree.body.extend(code_nodes)

        elif isinstance(location, int):
            # Insert at specific line
            insert_pos = 0
            for i, node in enumerate(tree.body):
                if hasattr(node, "lineno") and node.lineno >= location:
                    insert_pos = i
                    break
            else:
                insert_pos = len(tree.body)
            tree.body = tree.body[:insert_pos] + code_nodes + tree.body[insert_pos:]

        elif isinstance(location, str):
            # Insert after named function/class
            for i, node in enumerate(tree.body):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == location:
                        tree.body = tree.body[:i+1] + code_nodes + tree.body[i+1:]
                        break

        self.record_change(f"Inserted {len(code_nodes)} statement(s) at {location}")
        return tree

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def analyze_class(self, source: str, class_name: str) -> Optional[ClassInfo]:
        """
        Analyze a class definition.

        Args:
            source: Source code
            class_name: Name of class to analyze

        Returns:
            ClassInfo or None if not found
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                methods = []
                attributes = []

                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(self._get_method_info(child))
                    elif isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                attributes.append(target.id)

                # Get docstring
                docstring = None
                if node.body and isinstance(node.body[0], ast.Expr):
                    if isinstance(node.body[0].value, ast.Constant):
                        docstring = node.body[0].value.value

                return ClassInfo(
                    name=node.name,
                    lineno=node.lineno,
                    bases=[ast.unparse(b) for b in node.bases],
                    methods=methods,
                    attributes=attributes,
                    decorators=[ast.unparse(d) for d in node.decorator_list],
                    docstring=docstring,
                )

        return None

    def _get_method_info(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> MethodInfo:
        """Extract method information from AST node."""
        decorators = [ast.unparse(d) for d in node.decorator_list]

        # Get docstring
        docstring = None
        if node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant):
                docstring = node.body[0].value.value

        return MethodInfo(
            name=node.name,
            lineno=node.lineno,
            args=[a.arg for a in node.args.args],
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_property="property" in decorators,
            is_classmethod="classmethod" in decorators,
            is_staticmethod="staticmethod" in decorators,
            docstring=docstring,
        )

    def get_function_names(self, source: str) -> List[str]:
        """Get all function names in source."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        names = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)
        return names

    def get_class_names(self, source: str) -> List[str]:
        """Get all class names in source."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                names.append(node.name)
        return names
