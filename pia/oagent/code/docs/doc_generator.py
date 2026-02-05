#!/usr/bin/env python3
"""
doc_generator.py - Documentation Generator (Step 78)

PBTSO Phase: LOG

Provides:
- Auto-generate docstrings
- API documentation extraction
- Markdown documentation generation
- Code examples generation
- Documentation validation

Bus Topics:
- code.docs.generate
- code.docs.validate
- code.docs.extract

Protocol: DKIN v30, CITIZEN v2
"""

from __future__ import annotations

import ast
import json
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class DocFormat(Enum):
    """Documentation output formats."""
    MARKDOWN = "markdown"
    RST = "rst"
    HTML = "html"
    JSON = "json"


class DocstringStyle(Enum):
    """Docstring styles."""
    GOOGLE = "google"
    NUMPY = "numpy"
    SPHINX = "sphinx"


@dataclass
class DocConfig:
    """Configuration for the documentation generator."""
    docstring_style: DocstringStyle = DocstringStyle.GOOGLE
    output_format: DocFormat = DocFormat.MARKDOWN
    include_private: bool = False
    include_dunder: bool = False
    include_source: bool = False
    include_examples: bool = True
    max_line_length: int = 80
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "docstring_style": self.docstring_style.value,
            "output_format": self.output_format.value,
            "include_private": self.include_private,
            "include_dunder": self.include_dunder,
            "include_source": self.include_source,
            "include_examples": self.include_examples,
        }


# =============================================================================
# Types
# =============================================================================

@dataclass
class FunctionDoc:
    """Documentation for a function."""
    name: str
    signature: str
    docstring: Optional[str]
    parameters: List[Dict[str, Any]]
    returns: Optional[Dict[str, Any]]
    raises: List[Dict[str, Any]]
    examples: List[str]
    is_async: bool = False
    is_method: bool = False
    decorators: List[str] = field(default_factory=list)
    line_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "signature": self.signature,
            "docstring": self.docstring,
            "parameters": self.parameters,
            "returns": self.returns,
            "raises": self.raises,
            "examples": self.examples,
            "is_async": self.is_async,
            "is_method": self.is_method,
            "decorators": self.decorators,
            "line_number": self.line_number,
        }


@dataclass
class ClassDoc:
    """Documentation for a class."""
    name: str
    docstring: Optional[str]
    bases: List[str]
    attributes: List[Dict[str, Any]]
    methods: List[FunctionDoc]
    class_methods: List[FunctionDoc]
    static_methods: List[FunctionDoc]
    properties: List[Dict[str, Any]]
    decorators: List[str] = field(default_factory=list)
    line_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "docstring": self.docstring,
            "bases": self.bases,
            "attributes": self.attributes,
            "methods": [m.to_dict() for m in self.methods],
            "class_methods": [m.to_dict() for m in self.class_methods],
            "static_methods": [m.to_dict() for m in self.static_methods],
            "properties": self.properties,
            "decorators": self.decorators,
            "line_number": self.line_number,
        }


@dataclass
class ModuleDoc:
    """Documentation for a module."""
    name: str
    path: str
    docstring: Optional[str]
    functions: List[FunctionDoc]
    classes: List[ClassDoc]
    constants: List[Dict[str, Any]]
    imports: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "docstring": self.docstring,
            "functions": [f.to_dict() for f in self.functions],
            "classes": [c.to_dict() for c in self.classes],
            "constants": self.constants,
            "imports": self.imports,
        }


@dataclass
class DocResult:
    """Result of documentation generation."""
    success: bool
    content: str
    format: DocFormat
    files_processed: int
    elapsed_ms: float
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "content_length": len(self.content),
            "format": self.format.value,
            "files_processed": self.files_processed,
            "elapsed_ms": self.elapsed_ms,
            "warnings": self.warnings,
            "error": self.error,
        }


# =============================================================================
# Agent Bus with File Locking
# =============================================================================

class LockedAgentBus:
    """Agent bus with file locking for safe concurrent writes."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        line = json.dumps(full_event, ensure_ascii=False, separators=(",", ":")) + "\n"

        fd = os.open(str(self.bus_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        return event_id


# =============================================================================
# Documentation Generator
# =============================================================================

class DocumentationGenerator:
    """
    Documentation generator for Python code.

    PBTSO Phase: LOG

    Responsibilities:
    - Extract documentation from code
    - Generate formatted documentation
    - Auto-generate missing docstrings
    - Validate documentation coverage

    Usage:
        generator = DocumentationGenerator(config)
        result = generator.generate(file_path)
    """

    BUS_TOPICS = {
        "generate": "code.docs.generate",
        "validate": "code.docs.validate",
        "extract": "code.docs.extract",
        "heartbeat": "code.docs.heartbeat",
    }

    def __init__(
        self,
        config: Optional[DocConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or DocConfig()
        self.bus = bus or LockedAgentBus()

    def extract(self, file_path: Path) -> ModuleDoc:
        """
        Extract documentation from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            ModuleDoc with extracted documentation
        """
        content = file_path.read_text()
        tree = ast.parse(content)

        module_docstring = ast.get_docstring(tree)

        functions: List[FunctionDoc] = []
        classes: List[ClassDoc] = []
        constants: List[Dict[str, Any]] = []
        imports: List[str] = []

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if self._should_document(node.name):
                    functions.append(self._extract_function(node))

            elif isinstance(node, ast.AsyncFunctionDef):
                if self._should_document(node.name):
                    functions.append(self._extract_function(node, is_async=True))

            elif isinstance(node, ast.ClassDef):
                if self._should_document(node.name):
                    classes.append(self._extract_class(node))

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append({
                            "name": target.id,
                            "line": node.lineno,
                        })

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        # Emit extract event
        self.bus.emit({
            "topic": self.BUS_TOPICS["extract"],
            "kind": "docs",
            "actor": "doc-generator",
            "data": {
                "file": str(file_path),
                "functions": len(functions),
                "classes": len(classes),
            },
        })

        return ModuleDoc(
            name=file_path.stem,
            path=str(file_path),
            docstring=module_docstring,
            functions=functions,
            classes=classes,
            constants=constants,
            imports=imports,
        )

    def _should_document(self, name: str) -> bool:
        """Check if a name should be documented."""
        if name.startswith('__') and name.endswith('__'):
            return self.config.include_dunder
        if name.startswith('_'):
            return self.config.include_private
        return True

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_async: bool = False,
        is_method: bool = False,
    ) -> FunctionDoc:
        """Extract documentation from a function."""
        docstring = ast.get_docstring(node)

        # Build signature
        params = []
        for arg in node.args.args:
            param = {"name": arg.arg}
            if arg.annotation:
                param["type"] = ast.unparse(arg.annotation)
            params.append(param)

        signature = f"{node.name}({', '.join(p['name'] for p in params)})"

        # Get return type
        returns = None
        if node.returns:
            returns = {"type": ast.unparse(node.returns)}

        # Parse docstring for additional info
        raises = []
        examples = []

        if docstring:
            parsed = self._parse_docstring(docstring)
            raises = parsed.get("raises", [])
            examples = parsed.get("examples", [])

            # Merge parameter info
            for param in params:
                if param["name"] in parsed.get("params", {}):
                    param.update(parsed["params"][param["name"]])

            if parsed.get("returns"):
                returns = returns or {}
                returns.update(parsed["returns"])

        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(ast.unparse(dec))
            elif isinstance(dec, ast.Call):
                decorators.append(ast.unparse(dec.func))

        return FunctionDoc(
            name=node.name,
            signature=signature,
            docstring=docstring,
            parameters=params,
            returns=returns,
            raises=raises,
            examples=examples,
            is_async=is_async or isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
            decorators=decorators,
            line_number=node.lineno,
        )

    def _extract_class(self, node: ast.ClassDef) -> ClassDoc:
        """Extract documentation from a class."""
        docstring = ast.get_docstring(node)

        # Get base classes
        bases = []
        for base in node.bases:
            bases.append(ast.unparse(base))

        # Extract methods and attributes
        methods: List[FunctionDoc] = []
        class_methods: List[FunctionDoc] = []
        static_methods: List[FunctionDoc] = []
        properties: List[Dict[str, Any]] = []
        attributes: List[Dict[str, Any]] = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not self._should_document(item.name):
                    continue

                func_doc = self._extract_function(
                    item,
                    is_async=isinstance(item, ast.AsyncFunctionDef),
                    is_method=True,
                )

                # Check decorators
                is_classmethod = "classmethod" in func_doc.decorators
                is_staticmethod = "staticmethod" in func_doc.decorators
                is_property = "property" in func_doc.decorators

                if is_property:
                    properties.append({
                        "name": item.name,
                        "docstring": func_doc.docstring,
                        "type": func_doc.returns.get("type") if func_doc.returns else None,
                    })
                elif is_classmethod:
                    class_methods.append(func_doc)
                elif is_staticmethod:
                    static_methods.append(func_doc)
                else:
                    methods.append(func_doc)

            elif isinstance(item, ast.AnnAssign):
                if isinstance(item.target, ast.Name):
                    attributes.append({
                        "name": item.target.id,
                        "type": ast.unparse(item.annotation) if item.annotation else None,
                        "line": item.lineno,
                    })

        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Call):
                decorators.append(ast.unparse(dec.func))

        return ClassDoc(
            name=node.name,
            docstring=docstring,
            bases=bases,
            attributes=attributes,
            methods=methods,
            class_methods=class_methods,
            static_methods=static_methods,
            properties=properties,
            decorators=decorators,
            line_number=node.lineno,
        )

    def _parse_docstring(self, docstring: str) -> Dict[str, Any]:
        """Parse docstring to extract structured info."""
        result: Dict[str, Any] = {
            "params": {},
            "returns": None,
            "raises": [],
            "examples": [],
        }

        if self.config.docstring_style == DocstringStyle.GOOGLE:
            result = self._parse_google_docstring(docstring)
        elif self.config.docstring_style == DocstringStyle.NUMPY:
            result = self._parse_numpy_docstring(docstring)

        return result

    def _parse_google_docstring(self, docstring: str) -> Dict[str, Any]:
        """Parse Google-style docstring."""
        result: Dict[str, Any] = {
            "params": {},
            "returns": None,
            "raises": [],
            "examples": [],
        }

        # Parse Args section
        args_match = re.search(
            r'Args:\s*\n((?:\s+\w+.*\n)+)',
            docstring,
            re.MULTILINE
        )

        if args_match:
            args_text = args_match.group(1)
            for match in re.finditer(r'(\w+)(?:\s*\(([^)]+)\))?\s*:\s*(.+)', args_text):
                result["params"][match.group(1)] = {
                    "type": match.group(2),
                    "description": match.group(3).strip(),
                }

        # Parse Returns section
        returns_match = re.search(
            r'Returns:\s*\n\s+(.+)',
            docstring,
            re.MULTILINE
        )

        if returns_match:
            result["returns"] = {"description": returns_match.group(1).strip()}

        # Parse Raises section
        raises_match = re.search(
            r'Raises:\s*\n((?:\s+\w+.*\n)+)',
            docstring,
            re.MULTILINE
        )

        if raises_match:
            raises_text = raises_match.group(1)
            for match in re.finditer(r'(\w+)\s*:\s*(.+)', raises_text):
                result["raises"].append({
                    "type": match.group(1),
                    "description": match.group(2).strip(),
                })

        # Parse Examples section
        examples_match = re.search(
            r'Examples?:\s*\n((?:\s+.*\n)+)',
            docstring,
            re.MULTILINE
        )

        if examples_match:
            result["examples"] = [examples_match.group(1).strip()]

        return result

    def _parse_numpy_docstring(self, docstring: str) -> Dict[str, Any]:
        """Parse NumPy-style docstring."""
        result: Dict[str, Any] = {
            "params": {},
            "returns": None,
            "raises": [],
            "examples": [],
        }

        # Parse Parameters section
        params_match = re.search(
            r'Parameters\s*\n-+\s*\n((?:.*\n)*?)(?:\n\w|\Z)',
            docstring,
            re.MULTILINE
        )

        if params_match:
            params_text = params_match.group(1)
            for match in re.finditer(r'(\w+)\s*:\s*(\w+)\s*\n\s+(.+)', params_text):
                result["params"][match.group(1)] = {
                    "type": match.group(2),
                    "description": match.group(3).strip(),
                }

        return result

    def generate(
        self,
        file_path: Path,
        output_format: Optional[DocFormat] = None,
    ) -> DocResult:
        """
        Generate documentation for a Python file.

        Args:
            file_path: Path to Python file
            output_format: Output format (defaults to config)

        Returns:
            DocResult with generated documentation
        """
        start_time = time.time()
        format_to_use = output_format or self.config.output_format

        if not file_path.exists():
            return DocResult(
                success=False,
                content="",
                format=format_to_use,
                files_processed=0,
                elapsed_ms=(time.time() - start_time) * 1000,
                error=f"File not found: {file_path}",
            )

        try:
            module_doc = self.extract(file_path)

            if format_to_use == DocFormat.MARKDOWN:
                content = self._generate_markdown(module_doc)
            elif format_to_use == DocFormat.JSON:
                content = json.dumps(module_doc.to_dict(), indent=2)
            elif format_to_use == DocFormat.RST:
                content = self._generate_rst(module_doc)
            else:
                content = self._generate_markdown(module_doc)

            # Emit generate event
            self.bus.emit({
                "topic": self.BUS_TOPICS["generate"],
                "kind": "docs",
                "actor": "doc-generator",
                "data": {
                    "file": str(file_path),
                    "format": format_to_use.value,
                    "content_length": len(content),
                },
            })

            return DocResult(
                success=True,
                content=content,
                format=format_to_use,
                files_processed=1,
                elapsed_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return DocResult(
                success=False,
                content="",
                format=format_to_use,
                files_processed=1,
                elapsed_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    def _generate_markdown(self, module: ModuleDoc) -> str:
        """Generate Markdown documentation."""
        lines = []

        # Module header
        lines.append(f"# {module.name}")
        lines.append("")

        if module.docstring:
            lines.append(module.docstring)
            lines.append("")

        # Functions
        if module.functions:
            lines.append("## Functions")
            lines.append("")

            for func in module.functions:
                lines.append(f"### `{func.signature}`")
                lines.append("")

                if func.is_async:
                    lines.append("*async*")
                    lines.append("")

                if func.docstring:
                    lines.append(func.docstring.split('\n')[0])
                    lines.append("")

                if func.parameters:
                    lines.append("**Parameters:**")
                    lines.append("")
                    for param in func.parameters:
                        type_str = f": `{param.get('type')}`" if param.get('type') else ""
                        desc = param.get('description', '')
                        lines.append(f"- `{param['name']}`{type_str} - {desc}")
                    lines.append("")

                if func.returns:
                    lines.append("**Returns:**")
                    lines.append("")
                    type_str = f"`{func.returns.get('type')}`" if func.returns.get('type') else ""
                    desc = func.returns.get('description', '')
                    lines.append(f"- {type_str} {desc}")
                    lines.append("")

        # Classes
        if module.classes:
            lines.append("## Classes")
            lines.append("")

            for cls in module.classes:
                bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
                lines.append(f"### `{cls.name}{bases_str}`")
                lines.append("")

                if cls.docstring:
                    lines.append(cls.docstring.split('\n')[0])
                    lines.append("")

                if cls.attributes:
                    lines.append("**Attributes:**")
                    lines.append("")
                    for attr in cls.attributes:
                        type_str = f": `{attr.get('type')}`" if attr.get('type') else ""
                        lines.append(f"- `{attr['name']}`{type_str}")
                    lines.append("")

                if cls.methods:
                    lines.append("**Methods:**")
                    lines.append("")
                    for method in cls.methods:
                        async_str = "async " if method.is_async else ""
                        lines.append(f"- `{async_str}{method.signature}`")
                    lines.append("")

        return "\n".join(lines)

    def _generate_rst(self, module: ModuleDoc) -> str:
        """Generate reStructuredText documentation."""
        lines = []

        # Module header
        lines.append(module.name)
        lines.append("=" * len(module.name))
        lines.append("")

        if module.docstring:
            lines.append(module.docstring)
            lines.append("")

        # Functions
        if module.functions:
            lines.append("Functions")
            lines.append("---------")
            lines.append("")

            for func in module.functions:
                lines.append(f".. function:: {func.signature}")
                lines.append("")
                if func.docstring:
                    for line in func.docstring.split('\n'):
                        lines.append(f"   {line}")
                    lines.append("")

        return "\n".join(lines)

    def generate_docstring(
        self,
        code: str,
        style: Optional[DocstringStyle] = None,
    ) -> str:
        """
        Generate a docstring for the given code.

        Args:
            code: Function or class definition
            style: Docstring style to use

        Returns:
            Generated docstring
        """
        style = style or self.config.docstring_style

        try:
            tree = ast.parse(code)
            node = tree.body[0]

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return self._generate_function_docstring(node, style)
            elif isinstance(node, ast.ClassDef):
                return self._generate_class_docstring(node, style)

        except (SyntaxError, IndexError):
            pass

        return '"""TODO: Add documentation."""'

    def _generate_function_docstring(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        style: DocstringStyle,
    ) -> str:
        """Generate docstring for a function."""
        lines = ['"""']

        # Summary line
        lines.append(f"TODO: Describe {node.name}.")
        lines.append("")

        if style == DocstringStyle.GOOGLE:
            # Args section
            if node.args.args:
                lines.append("Args:")
                for arg in node.args.args:
                    if arg.arg == "self":
                        continue
                    type_hint = ""
                    if arg.annotation:
                        type_hint = f" ({ast.unparse(arg.annotation)})"
                    lines.append(f"    {arg.arg}{type_hint}: TODO.")
                lines.append("")

            # Returns section
            if node.returns:
                lines.append("Returns:")
                lines.append(f"    {ast.unparse(node.returns)}: TODO.")
                lines.append("")

        elif style == DocstringStyle.NUMPY:
            # Parameters section
            if node.args.args:
                lines.append("Parameters")
                lines.append("----------")
                for arg in node.args.args:
                    if arg.arg == "self":
                        continue
                    type_hint = "object"
                    if arg.annotation:
                        type_hint = ast.unparse(arg.annotation)
                    lines.append(f"{arg.arg} : {type_hint}")
                    lines.append("    TODO.")
                lines.append("")

        lines.append('"""')
        return "\n".join(lines)

    def _generate_class_docstring(
        self,
        node: ast.ClassDef,
        style: DocstringStyle,
    ) -> str:
        """Generate docstring for a class."""
        lines = ['"""']

        # Summary line
        lines.append(f"TODO: Describe {node.name}.")
        lines.append("")

        if style == DocstringStyle.GOOGLE:
            # Attributes section
            attrs = []
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    attrs.append(item)

            if attrs:
                lines.append("Attributes:")
                for attr in attrs:
                    type_hint = ""
                    if attr.annotation:
                        type_hint = f" ({ast.unparse(attr.annotation)})"
                    lines.append(f"    {attr.target.id}{type_hint}: TODO.")
                lines.append("")

        lines.append('"""')
        return "\n".join(lines)

    def validate(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate documentation coverage.

        Args:
            file_path: Path to Python file

        Returns:
            Validation results
        """
        module = self.extract(file_path)

        missing_docstrings = []
        total_items = 0
        documented_items = 0

        # Check module docstring
        total_items += 1
        if module.docstring:
            documented_items += 1
        else:
            missing_docstrings.append(f"Module: {module.name}")

        # Check functions
        for func in module.functions:
            total_items += 1
            if func.docstring:
                documented_items += 1
            else:
                missing_docstrings.append(f"Function: {func.name}")

        # Check classes
        for cls in module.classes:
            total_items += 1
            if cls.docstring:
                documented_items += 1
            else:
                missing_docstrings.append(f"Class: {cls.name}")

            # Check methods
            for method in cls.methods + cls.class_methods + cls.static_methods:
                total_items += 1
                if method.docstring:
                    documented_items += 1
                else:
                    missing_docstrings.append(f"Method: {cls.name}.{method.name}")

        coverage = documented_items / total_items * 100 if total_items > 0 else 100

        # Emit validate event
        self.bus.emit({
            "topic": self.BUS_TOPICS["validate"],
            "kind": "docs",
            "actor": "doc-generator",
            "data": {
                "file": str(file_path),
                "coverage": coverage,
                "missing": len(missing_docstrings),
            },
        })

        return {
            "file": str(file_path),
            "coverage_percent": round(coverage, 2),
            "total_items": total_items,
            "documented_items": documented_items,
            "missing_docstrings": missing_docstrings,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Documentation Generator."""
    import argparse

    parser = argparse.ArgumentParser(description="Documentation Generator (Step 78)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate command
    gen_parser = subparsers.add_parser("generate", help="Generate documentation")
    gen_parser.add_argument("file", help="Python file")
    gen_parser.add_argument("--format", "-f", choices=["markdown", "json", "rst"],
                           default="markdown", help="Output format")
    gen_parser.add_argument("--output", "-o", help="Output file")

    # extract command
    extract_parser = subparsers.add_parser("extract", help="Extract documentation")
    extract_parser.add_argument("file", help="Python file")
    extract_parser.add_argument("--json", action="store_true", help="JSON output")

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate documentation coverage")
    validate_parser.add_argument("file", help="Python file")
    validate_parser.add_argument("--json", action="store_true", help="JSON output")

    # docstring command
    docstring_parser = subparsers.add_parser("docstring", help="Generate docstring for code")
    docstring_parser.add_argument("--style", choices=["google", "numpy", "sphinx"],
                                 default="google", help="Docstring style")

    # stats command
    subparsers.add_parser("stats", help="Show generator stats")

    args = parser.parse_args()

    generator = DocumentationGenerator()

    if args.command == "generate":
        path = Path(args.file)
        format_map = {
            "markdown": DocFormat.MARKDOWN,
            "json": DocFormat.JSON,
            "rst": DocFormat.RST,
        }

        result = generator.generate(path, format_map[args.format])

        if result.success:
            if args.output:
                Path(args.output).write_text(result.content)
                print(f"Documentation written to: {args.output}")
            else:
                print(result.content)
        else:
            print(f"Error: {result.error}")
            return 1
        return 0

    elif args.command == "extract":
        path = Path(args.file)
        module = generator.extract(path)

        if args.json:
            print(json.dumps(module.to_dict(), indent=2))
        else:
            print(f"Module: {module.name}")
            print(f"Functions: {len(module.functions)}")
            print(f"Classes: {len(module.classes)}")
        return 0

    elif args.command == "validate":
        path = Path(args.file)
        result = generator.validate(path)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Coverage: {result['coverage_percent']}%")
            print(f"Documented: {result['documented_items']}/{result['total_items']}")
            if result['missing_docstrings']:
                print("\nMissing docstrings:")
                for item in result['missing_docstrings']:
                    print(f"  - {item}")
        return 0

    elif args.command == "docstring":
        import sys
        code = sys.stdin.read()
        style_map = {
            "google": DocstringStyle.GOOGLE,
            "numpy": DocstringStyle.NUMPY,
            "sphinx": DocstringStyle.SPHINX,
        }
        docstring = generator.generate_docstring(code, style_map[args.style])
        print(docstring)
        return 0

    elif args.command == "stats":
        stats = generator.get_stats()
        print(json.dumps(stats, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
