#!/usr/bin/env python3
"""
typescript_parser.py - TypeScript AST Parser (Step 5)

Parses TypeScript/JavaScript source code and extracts symbols using regex-based parsing.
Falls back to subprocess for full AST parsing if Node.js is available.

PBTSO Phase: RESEARCH

Bus Topics:
- research.parse.typescript
- research.symbols.extracted

Protocol: DKIN v30, PAIP v16
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ASTParser, ParseResult, SymbolInfo, ImportInfo, ParserRegistry


class TypeScriptASTParser(ASTParser):
    """
    TypeScript/JavaScript AST parser.

    Uses regex-based extraction for fast parsing without Node.js dependency.
    Can optionally use ts-morph via subprocess for full AST parsing.

    Extracts:
    - Classes and interfaces
    - Functions and methods
    - Imports and exports
    - Type aliases
    - Constants
    """

    extensions = [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"]
    language = "typescript"

    # Regex patterns for TypeScript/JavaScript parsing
    PATTERNS = {
        "class": re.compile(
            r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?\s*\{",
            re.MULTILINE
        ),
        "interface": re.compile(
            r"(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+([\w,\s]+))?\s*\{",
            re.MULTILINE
        ),
        "function": re.compile(
            r"(?:export\s+)?(?:async\s+)?function\s*\*?\s+(\w+)\s*(?:<[^>]+>)?\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?\s*\{",
            re.MULTILINE
        ),
        "arrow_function": re.compile(
            r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*[^=]+)?\s*=>",
            re.MULTILINE
        ),
        "method": re.compile(
            r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:async\s+)?(\w+)\s*(?:<[^>]+>)?\s*\(([^)]*)\)(?:\s*:\s*([^{]+))?\s*\{",
            re.MULTILINE
        ),
        "import": re.compile(
            r"import\s+(?:(?:(\w+)|(?:\{([^}]+)\})|(?:\*\s+as\s+(\w+)))\s+from\s+)?['\"]([^'\"]+)['\"]",
            re.MULTILINE
        ),
        "export": re.compile(
            r"export\s+(?:default\s+)?(?:class|interface|function|const|let|var|type|enum)\s+(\w+)",
            re.MULTILINE
        ),
        "type_alias": re.compile(
            r"(?:export\s+)?type\s+(\w+)(?:<[^>]+>)?\s*=",
            re.MULTILINE
        ),
        "const": re.compile(
            r"(?:export\s+)?const\s+(\w+)\s*(?::\s*([^=]+))?\s*=",
            re.MULTILINE
        ),
        "enum": re.compile(
            r"(?:export\s+)?(?:const\s+)?enum\s+(\w+)\s*\{",
            re.MULTILINE
        ),
    }

    def __init__(self, bus=None, use_subprocess: bool = False):
        """
        Initialize the TypeScript parser.

        Args:
            bus: AgentBus for event emission
            use_subprocess: Whether to use Node.js for full AST parsing
        """
        super().__init__(bus)
        self.use_subprocess = use_subprocess
        self._node_available: Optional[bool] = None

    def parse(self, content: str, path: str) -> ParseResult:
        """
        Parse TypeScript/JavaScript source code.

        Args:
            content: Source code content
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
            if self.use_subprocess and self._check_node_available():
                return self._parse_with_node(content, path)

            # Use regex-based parsing
            result.classes = self._extract_classes(content)
            result.functions = self._extract_functions(content)
            result.imports = self._extract_imports(content)
            result.exports = self._extract_exports(content)
            result.variables = self._extract_constants(content)

            # Extract interfaces as a special kind of class
            interfaces = self._extract_interfaces(content)
            result.classes.extend(interfaces)

            # Extract type aliases as variables
            type_aliases = self._extract_type_aliases(content)
            result.variables.extend(type_aliases)

            # Extract enums as classes
            enums = self._extract_enums(content)
            result.classes.extend(enums)

            # Emit event
            self._emit_event("research.parse.typescript", {
                "path": path,
                "classes": len(result.classes),
                "functions": len(result.functions),
                "imports": len(result.imports),
            })

        except Exception as e:
            result.success = False
            result.error = str(e)

        return result

    def _extract_classes(self, content: str) -> List[SymbolInfo]:
        """Extract class definitions."""
        classes = []
        lines = content.splitlines()

        for match in self.PATTERNS["class"].finditer(content):
            name = match.group(1)
            extends = match.group(2)
            implements = match.group(3)

            line = content[:match.start()].count("\n") + 1

            classes.append(SymbolInfo(
                name=name,
                kind="class",
                line=line,
                parameters=[
                    {"extends": extends} if extends else {},
                    {"implements": implements.split(",")} if implements else {},
                ],
            ))

        return classes

    def _extract_interfaces(self, content: str) -> List[SymbolInfo]:
        """Extract interface definitions."""
        interfaces = []

        for match in self.PATTERNS["interface"].finditer(content):
            name = match.group(1)
            extends = match.group(2)

            line = content[:match.start()].count("\n") + 1

            interfaces.append(SymbolInfo(
                name=name,
                kind="interface",
                line=line,
                parameters=[{"extends": extends.split(",")} if extends else {}],
            ))

        return interfaces

    def _extract_functions(self, content: str) -> List[SymbolInfo]:
        """Extract function definitions."""
        functions = []

        # Regular functions
        for match in self.PATTERNS["function"].finditer(content):
            name = match.group(1)
            params = match.group(2)
            return_type = match.group(3)

            line = content[:match.start()].count("\n") + 1
            is_async = "async" in content[max(0, match.start()-10):match.start()]

            functions.append(SymbolInfo(
                name=name,
                kind="function",
                line=line,
                signature=f"({params.strip()})" if params else "()",
                return_type=return_type.strip() if return_type else None,
                is_async=is_async,
            ))

        # Arrow functions assigned to const/let/var
        for match in self.PATTERNS["arrow_function"].finditer(content):
            name = match.group(1)
            line = content[:match.start()].count("\n") + 1
            is_async = "async" in content[max(0, match.start()-10):match.start()+50]

            functions.append(SymbolInfo(
                name=name,
                kind="function",
                line=line,
                is_async=is_async,
            ))

        return functions

    def _extract_imports(self, content: str) -> List[ImportInfo]:
        """Extract import statements."""
        imports = []

        for match in self.PATTERNS["import"].finditer(content):
            default_import = match.group(1)
            named_imports = match.group(2)
            namespace_import = match.group(3)
            module = match.group(4)

            line = content[:match.start()].count("\n") + 1

            names = []
            alias = None

            if default_import:
                names = [default_import]
            elif named_imports:
                # Parse named imports: { a, b as c, d }
                names = [
                    n.strip().split(" as ")[0].strip()
                    for n in named_imports.split(",")
                ]
            elif namespace_import:
                alias = namespace_import

            imports.append(ImportInfo(
                module=module,
                names=names,
                alias=alias,
                line=line,
                is_relative=module.startswith("."),
            ))

        return imports

    def _extract_exports(self, content: str) -> List[str]:
        """Extract exported names."""
        exports = []

        for match in self.PATTERNS["export"].finditer(content):
            name = match.group(1)
            exports.append(name)

        return exports

    def _extract_constants(self, content: str) -> List[SymbolInfo]:
        """Extract const declarations."""
        constants = []

        for match in self.PATTERNS["const"].finditer(content):
            name = match.group(1)
            type_annotation = match.group(2)

            line = content[:match.start()].count("\n") + 1

            # Skip if it's an arrow function (already captured)
            if "=>" in content[match.end():match.end()+50]:
                continue

            constants.append(SymbolInfo(
                name=name,
                kind="constant",
                line=line,
                return_type=type_annotation.strip() if type_annotation else None,
            ))

        return constants

    def _extract_type_aliases(self, content: str) -> List[SymbolInfo]:
        """Extract type alias definitions."""
        type_aliases = []

        for match in self.PATTERNS["type_alias"].finditer(content):
            name = match.group(1)
            line = content[:match.start()].count("\n") + 1

            type_aliases.append(SymbolInfo(
                name=name,
                kind="type",
                line=line,
            ))

        return type_aliases

    def _extract_enums(self, content: str) -> List[SymbolInfo]:
        """Extract enum definitions."""
        enums = []

        for match in self.PATTERNS["enum"].finditer(content):
            name = match.group(1)
            line = content[:match.start()].count("\n") + 1

            enums.append(SymbolInfo(
                name=name,
                kind="enum",
                line=line,
            ))

        return enums

    def _check_node_available(self) -> bool:
        """Check if Node.js is available."""
        if self._node_available is not None:
            return self._node_available

        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self._node_available = result.returncode == 0
        except Exception:
            self._node_available = False

        return self._node_available

    def _parse_with_node(self, content: str, path: str) -> ParseResult:
        """
        Parse using Node.js and ts-morph for full AST.

        This is more accurate but requires Node.js to be installed.
        """
        result = ParseResult(
            language=self.language,
            path=path,
            line_count=len(content.splitlines()),
        )

        ts_parser_script = """
const ts = require('typescript');

const content = process.argv[2];
const sourceFile = ts.createSourceFile(
    'temp.ts',
    content,
    ts.ScriptTarget.Latest,
    true
);

const result = {
    classes: [],
    functions: [],
    imports: [],
    exports: []
};

function visit(node) {
    if (ts.isClassDeclaration(node) && node.name) {
        result.classes.push({
            name: node.name.text,
            line: sourceFile.getLineAndCharacterOfPosition(node.pos).line + 1
        });
    }
    if (ts.isFunctionDeclaration(node) && node.name) {
        result.functions.push({
            name: node.name.text,
            line: sourceFile.getLineAndCharacterOfPosition(node.pos).line + 1
        });
    }
    if (ts.isImportDeclaration(node)) {
        const module = node.moduleSpecifier.text;
        result.imports.push({ module, line: sourceFile.getLineAndCharacterOfPosition(node.pos).line + 1 });
    }
    ts.forEachChild(node, visit);
}

visit(sourceFile);
console.log(JSON.stringify(result));
"""

        try:
            # Write content to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
                f.write(content)
                temp_path = f.name

            proc_result = subprocess.run(
                ["npx", "-y", "ts-node", "-e", ts_parser_script, "--", content],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if proc_result.returncode == 0:
                parsed = json.loads(proc_result.stdout)
                result.classes = [
                    SymbolInfo(name=c["name"], kind="class", line=c["line"])
                    for c in parsed.get("classes", [])
                ]
                result.functions = [
                    SymbolInfo(name=f["name"], kind="function", line=f["line"])
                    for f in parsed.get("functions", [])
                ]
                result.imports = [
                    ImportInfo(module=i["module"], names=[], line=i["line"])
                    for i in parsed.get("imports", [])
                ]
            else:
                # Fall back to regex parsing
                return self.parse(content, path)

        except Exception as e:
            result.success = False
            result.error = str(e)

        finally:
            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except Exception:
                pass

        return result


# Register the parser
ParserRegistry.register(TypeScriptASTParser)


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for TypeScript AST Parser."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TypeScript AST Parser (Step 5)"
    )
    parser.add_argument(
        "file",
        help="TypeScript/JavaScript file to parse"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--node",
        action="store_true",
        help="Use Node.js for full AST parsing"
    )

    args = parser.parse_args()

    ts_parser = TypeScriptASTParser(use_subprocess=args.node)
    result = ts_parser.parse_file(args.file)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Parse Result for: {result.path}")
        print(f"  Success: {result.success}")
        if result.error:
            print(f"  Error: {result.error}")
        print(f"  Lines: {result.line_count}")
        print(f"  Classes/Interfaces: {len(result.classes)}")
        for cls in result.classes:
            print(f"    - {cls.name} ({cls.kind}, line {cls.line})")
        print(f"  Functions: {len(result.functions)}")
        for func in result.functions:
            print(f"    - {func.name} (line {func.line})")
        print(f"  Imports: {len(result.imports)}")
        for imp in result.imports:
            print(f"    - {imp.module}")
        print(f"  Exports: {result.exports}")

    return 0 if result.success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
