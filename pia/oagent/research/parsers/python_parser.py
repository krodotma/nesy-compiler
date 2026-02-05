#!/usr/bin/env python3
"""
python_parser.py - Python AST Parser (Step 4)

Parses Python source code and extracts symbols, imports, and structure.

PBTSO Phase: RESEARCH

Bus Topics:
- research.parse.python
- research.symbols.extracted

Protocol: DKIN v30, PAIP v16
"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

from .base import ASTParser, ParseResult, SymbolInfo, ImportInfo, ParserRegistry


class PythonASTParser(ASTParser):
    """
    Python AST parser using the built-in ast module.

    Extracts:
    - Classes and their methods
    - Functions (top-level and nested)
    - Imports (regular and from imports)
    - Module-level variables
    - Docstrings
    - Type annotations
    """

    extensions = [".py", ".pyi"]
    language = "python"

    def parse(self, content: str, path: str) -> ParseResult:
        """
        Parse Python source code and return structured AST data.

        Args:
            content: Python source code
            path: File path (for error reporting)

        Returns:
            ParseResult with extracted symbols
        """
        result = ParseResult(
            language=self.language,
            path=path,
            line_count=len(content.splitlines()),
        )

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            result.success = False
            result.error = f"Syntax error at line {e.lineno}: {e.msg}"
            return result

        # Extract module docstring
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            result.has_docstring = True
            result.module_docstring = tree.body[0].value.value

        # Extract all symbols
        result.classes = self._extract_classes(tree)
        result.functions = self._extract_functions(tree)
        result.methods = self._extract_methods(tree)
        result.imports = self._extract_imports(tree)
        result.variables = self._extract_variables(tree)

        # Emit event
        self._emit_event("research.parse.python", {
            "path": path,
            "classes": len(result.classes),
            "functions": len(result.functions),
            "imports": len(result.imports),
        })

        return result

    def _extract_classes(self, tree: ast.Module) -> List[SymbolInfo]:
        """Extract class definitions from AST."""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Get docstring
                docstring = ast.get_docstring(node)

                # Get decorators
                decorators = self._get_decorators(node)

                # Get base classes
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(ast.unparse(base))

                classes.append(SymbolInfo(
                    name=node.name,
                    kind="class",
                    line=node.lineno,
                    end_line=node.end_lineno,
                    column=node.col_offset,
                    docstring=docstring,
                    decorators=decorators,
                    parameters=[{"base": b} for b in bases],
                ))

        return classes

    def _extract_functions(self, tree: ast.Module) -> List[SymbolInfo]:
        """Extract top-level function definitions from AST."""
        functions = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(self._function_to_symbol(node))

        return functions

    def _extract_methods(self, tree: ast.Module) -> List[SymbolInfo]:
        """Extract class methods from AST."""
        methods = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbol = self._function_to_symbol(item)
                        symbol.kind = "method"
                        symbol.parent = class_name

                        # Determine visibility
                        if symbol.name.startswith("__") and not symbol.name.endswith("__"):
                            symbol.visibility = "private"
                        elif symbol.name.startswith("_"):
                            symbol.visibility = "protected"

                        methods.append(symbol)

        return methods

    def _function_to_symbol(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> SymbolInfo:
        """Convert a function node to SymbolInfo."""
        # Get docstring
        docstring = ast.get_docstring(node)

        # Get decorators
        decorators = self._get_decorators(node)

        # Get return type annotation
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        # Get parameters
        parameters = self._extract_parameters(node.args)

        # Build signature
        sig_parts = []
        for param in parameters:
            part = param["name"]
            if param.get("annotation"):
                part += f": {param['annotation']}"
            if param.get("default"):
                part += f" = {param['default']}"
            sig_parts.append(part)

        signature = f"({', '.join(sig_parts)})"
        if return_type:
            signature += f" -> {return_type}"

        return SymbolInfo(
            name=node.name,
            kind="function",
            line=node.lineno,
            end_line=node.end_lineno,
            column=node.col_offset,
            signature=signature,
            docstring=docstring,
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            return_type=return_type,
            parameters=parameters,
        )

    def _extract_parameters(self, args: ast.arguments) -> List[Dict[str, Any]]:
        """Extract function parameters."""
        parameters = []

        # Positional args
        defaults_offset = len(args.args) - len(args.defaults)
        for i, arg in enumerate(args.args):
            param = {
                "name": arg.arg,
                "kind": "positional",
            }
            if arg.annotation:
                param["annotation"] = ast.unparse(arg.annotation)

            default_idx = i - defaults_offset
            if default_idx >= 0:
                param["default"] = ast.unparse(args.defaults[default_idx])

            parameters.append(param)

        # *args
        if args.vararg:
            param = {"name": f"*{args.vararg.arg}", "kind": "vararg"}
            if args.vararg.annotation:
                param["annotation"] = ast.unparse(args.vararg.annotation)
            parameters.append(param)

        # Keyword-only args
        kw_defaults_offset = 0
        for i, arg in enumerate(args.kwonlyargs):
            param = {
                "name": arg.arg,
                "kind": "keyword_only",
            }
            if arg.annotation:
                param["annotation"] = ast.unparse(arg.annotation)
            if args.kw_defaults[i] is not None:
                param["default"] = ast.unparse(args.kw_defaults[i])
            parameters.append(param)

        # **kwargs
        if args.kwarg:
            param = {"name": f"**{args.kwarg.arg}", "kind": "kwarg"}
            if args.kwarg.annotation:
                param["annotation"] = ast.unparse(args.kwarg.annotation)
            parameters.append(param)

        return parameters

    def _extract_imports(self, tree: ast.Module) -> List[ImportInfo]:
        """Extract import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[],
                        alias=alias.asname,
                        line=node.lineno,
                        is_relative=False,
                    ))

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                imports.append(ImportInfo(
                    module=module,
                    names=names,
                    line=node.lineno,
                    is_relative=node.level > 0,
                    level=node.level,
                ))

        return imports

    def _extract_variables(self, tree: ast.Module) -> List[SymbolInfo]:
        """Extract module-level variable assignments."""
        variables = []

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Check if it's a constant (ALL_CAPS)
                        is_constant = target.id.isupper()

                        variables.append(SymbolInfo(
                            name=target.id,
                            kind="constant" if is_constant else "variable",
                            line=node.lineno,
                            column=node.col_offset,
                        ))

            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    annotation = ast.unparse(node.annotation) if node.annotation else None

                    variables.append(SymbolInfo(
                        name=node.target.id,
                        kind="variable",
                        line=node.lineno,
                        column=node.col_offset,
                        return_type=annotation,  # Using return_type for annotation
                    ))

        return variables

    def _get_decorators(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> List[str]:
        """Extract decorator names from a node."""
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(ast.unparse(decorator))
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
                elif isinstance(decorator.func, ast.Attribute):
                    decorators.append(ast.unparse(decorator.func))
        return decorators


# Register the parser
ParserRegistry.register(PythonASTParser)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Python AST Parser."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Python AST Parser (Step 4)"
    )
    parser.add_argument(
        "file",
        help="Python file to parse"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    py_parser = PythonASTParser()
    result = py_parser.parse_file(args.file)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Parse Result for: {result.path}")
        print(f"  Success: {result.success}")
        if result.error:
            print(f"  Error: {result.error}")
        print(f"  Lines: {result.line_count}")
        print(f"  Classes: {len(result.classes)}")
        for cls in result.classes:
            print(f"    - {cls.name} (line {cls.line})")
        print(f"  Functions: {len(result.functions)}")
        for func in result.functions:
            print(f"    - {func.name}{func.signature or '()'} (line {func.line})")
        print(f"  Imports: {len(result.imports)}")
        for imp in result.imports:
            if imp.names:
                print(f"    - from {imp.module} import {', '.join(imp.names)}")
            else:
                print(f"    - import {imp.module}")

    return 0 if result.success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
